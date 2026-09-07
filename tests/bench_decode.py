#!/usr/bin/env python3
"""Streaming decode bench + coherence probes for GLM-5.3-Flash EXL3 on :8888.

Decode tok/s = (completion_tokens - 1) / (end - first_token_time).
Does not reimplement the model; drives the live OpenAI API.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8888"
MODEL = "GLM-5.3-Flash-EXL3"
BENCH_PROMPT = (
    "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough."
)
# Same regime as tonyd2wild's 60.6 tok/s re-bench (temp 0, thinking off, warmed).
STRUCTURED_PROMPT = (
    "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
)
NAN_RE = re.compile(r"\bnan\b|locklock", re.I)
SPEC_RE = re.compile(
    r"^(vllm:spec_decode_[a-zA-Z0-9_]+)\{([^}]*)\}\s+(\S+)$"
)


def _post(path: str, body: dict, timeout: float = 600.0, stream: bool = False):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def health() -> tuple[int, str]:
    req = urllib.request.Request(BASE + "/health")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def chat_nonstream(prompt: str, max_tokens: int = 64) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with _post("/v1/chat/completions", body) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        data["_http"] = resp.status
        data["_raw"] = raw
        return data


def spec_snapshot() -> dict[str, float]:
    """vLLM spec-decode counters. per-pos keys are pos:{n}."""
    req = urllib.request.Request(BASE + "/metrics")
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8", "replace")
    out: dict[str, float] = {}
    for line in raw.splitlines():
        m = SPEC_RE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2), float(m.group(3))
        if name.endswith("_created"):
            continue
        if "per_pos" in name:
            pos = re.search(r'position="(\d+)"', labels)
            if pos:
                out[f"pos:{pos.group(1)}"] = val
        else:
            out[name] = out.get(name, 0.0) + val
    return out


def spec_delta(before: dict[str, float], after: dict[str, float]) -> dict:
    drafts = after.get("vllm:spec_decode_num_drafts_total", 0) - before.get(
        "vllm:spec_decode_num_drafts_total", 0
    )
    draft_tok = after.get("vllm:spec_decode_num_draft_tokens_total", 0) - before.get(
        "vllm:spec_decode_num_draft_tokens_total", 0
    )
    acc = after.get("vllm:spec_decode_num_accepted_tokens_total", 0) - before.get(
        "vllm:spec_decode_num_accepted_tokens_total", 0
    )
    pos = []
    for i in range(7):
        k = f"pos:{i}"
        d = after.get(k, 0) - before.get(k, 0)
        pos.append(round(d / drafts, 4) if drafts else 0.0)
    return {
        "drafts": int(drafts),
        "draft_tokens": int(draft_tok),
        "accepted": int(acc),
        "accept_ratio": round(acc / draft_tok, 4) if draft_tok else None,
        "accepted_per_step": round(acc / drafts, 3) if drafts else None,
        "pos": pos,
    }


def stream_bench(max_tokens: int = 200, prompt: str | None = None) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt or BENCH_PROMPT}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    first = None
    chunks: list[str] = []
    usage = None
    finish = None
    with _post("/v1/chat/completions", body, timeout=900) as resp:
        http = resp.status
        buf = b""
        while True:
            # read(n) can wait across HTTP chunks and bias first-token timing.
            piece = resp.read1(256)
            if not piece:
                break
            buf += piece
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                    or ""
                )
                if content:
                    if first is None:
                        first = time.perf_counter()
                    chunks.append(content)
                fr = choices[0].get("finish_reason")
                if fr:
                    finish = fr
    t1 = time.perf_counter()
    text = "".join(chunks)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    ttft = None if first is None else (first - t0)
    decode_toks = max(completion_tokens - 1, 0)
    decode_s = None if first is None else (t1 - first)
    tps = None
    if decode_s and decode_s > 0 and decode_toks > 0:
        tps = decode_toks / decode_s
    nan = bool(NAN_RE.search(text)) or ("nan" in text.lower())
    return {
        "http": http,
        "ttft_s": ttft,
        "wall_s": t1 - t0,
        "decode_s": decode_s,
        "tok_s": tps,
        "delivered_tok_s": completion_tokens / (t1 - t0) if completion_tokens else None,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish,
        "nan": nan,
        "text_head": text[:400],
        "text": text,
        "text_len": len(text),
        "usage": usage,
    }


def median(xs: list[float | None]) -> float | None:
    vals = sorted(x for x in xs if x is not None and math.isfinite(x))
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def coherence() -> dict:
    paris = chat_nonstream("What is the capital of France? Answer with one word.", 64)
    cmp_ = chat_nonstream("Is 9.9 greater than 9.11? Answer yes or no, then one sentence.", 48)
    sky = chat_nonstream("Write one sentence about a sky-blue color.", 48)
    def text_of(d: dict) -> str:
        try:
            msg = d["choices"][0]["message"]
            return (msg.get("content") or "") + "\n" + (msg.get("reasoning") or msg.get("reasoning_content") or "")
        except Exception:
            return d.get("_raw") or ""
    def content_of(d: dict) -> str:
        try:
            return d["choices"][0]["message"].get("content") or ""
        except Exception:
            return ""
    ptxt, ctxt, stxt = text_of(paris), text_of(cmp_), text_of(sky)
    pcontent, ccontent, scontent = content_of(paris), content_of(cmp_), content_of(sky)
    cmp_ok = ("9.9" in (ccontent + ctxt) and ">" in (ccontent + ctxt)) or "yes" in (ccontent + ctxt).lower()
    return {
        "paris": {"ok": "paris" in (pcontent + ptxt).lower(), "text": pcontent[:400] or ptxt[:400], "http": paris.get("_http")},
        "cmp": {"ok": cmp_ok and bool((ccontent or ctxt).strip()), "text": ccontent[:400] or ctxt[:400], "http": cmp_.get("_http")},
        "sky": {"ok": bool(scontent.strip()) and "nan" not in scontent.lower(), "text": scontent[:400], "http": sky.get("_http")},
        "nan": any(NAN_RE.search(t) for t in (ptxt, ctxt, stxt, pcontent, ccontent, scontent)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--skip-coherence", action="store_true")
    ap.add_argument(
        "--structured",
        action="store_true",
        help="Warmed count-1-to-200 decode (temp 0, thinking off). Reports median tok/s + DFlash2 accept.",
    )
    args = ap.parse_args()
    h_code, h_body = health()
    rec: dict = {
        "phase": args.phase,
        "health_code": h_code,
        "health_body": h_body,
        "runs": [],
        "ts": time.time(),
    }
    if h_code != 200:
        Path(args.out).write_text(json.dumps(rec, indent=2))
        print(json.dumps(rec, indent=2))
        return 2
    prompt = STRUCTURED_PROMPT if args.structured else BENCH_PROMPT
    rec["prompt"] = "structured-count-1-200" if args.structured else "hashmap-prose"
    rec["thinking"] = False
    rec["temperature"] = 0
    if args.structured:
        print("[bench] warmup (32 tokens)", flush=True)
        warm = stream_bench(32, prompt=prompt)
        rec["warmup"] = {k: warm[k] for k in ("tok_s", "ttft_s", "completion_tokens")}
        print(json.dumps(rec["warmup"]), flush=True)
        args.skip_coherence = True
    for i in range(args.runs):
        print(f"[bench] run {i+1}/{args.runs} phase={args.phase}", flush=True)
        before = spec_snapshot()
        r = stream_bench(args.max_tokens, prompt=prompt)
        r["spec"] = spec_delta(before, spec_snapshot())
        rec["runs"].append(r)
        print(
            json.dumps(
                {
                    "tok_s": r["tok_s"],
                    "ttft_s": r["ttft_s"],
                    "completion_tokens": r["completion_tokens"],
                    "finish_reason": r["finish_reason"],
                    "nan": r["nan"],
                    **{k: r["spec"][k] for k in ("accept_ratio", "accepted_per_step", "pos")},
                }
            ),
            flush=True,
        )
    rec["tok_s_median"] = median([r["tok_s"] for r in rec["runs"]])
    rec["tok_s_min"] = min((r["tok_s"] for r in rec["runs"] if r["tok_s"] is not None), default=None)
    rec["tok_s_max"] = max((r["tok_s"] for r in rec["runs"] if r["tok_s"] is not None), default=None)
    rec["ttft_median_s"] = median([r["ttft_s"] for r in rec["runs"]])
    rec["completion_tokens_median"] = median([float(r["completion_tokens"]) for r in rec["runs"]])
    rec["accept_ratio_median"] = median(
        [r.get("spec", {}).get("accept_ratio") for r in rec["runs"]]
    )
    rec["accepted_per_step_median"] = median(
        [r.get("spec", {}).get("accepted_per_step") for r in rec["runs"]]
    )
    rec["any_nan"] = any(r["nan"] for r in rec["runs"])
    if not args.skip_coherence:
        print("[bench] coherence", flush=True)
        rec["coherence"] = coherence()
        rec["coherent"] = bool(
            rec["coherence"]["paris"]["ok"]
            and rec["coherence"]["sky"]["ok"]
            and not rec["coherence"]["nan"]
            and rec["coherence"]["cmp"]["ok"]
        )
    # cheap extra health + one-token completion (plan: 4th consistent success)
    h2, _ = health()
    rec["health_code_after"] = h2
    rec["short_after"] = stream_bench(max_tokens=8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in rec if k not in {"runs", "coherence", "short_after"}}, indent=2))
    print("wrote", args.out)
    return 0 if rec.get("tok_s_median") else 1


if __name__ == "__main__":
    sys.exit(main())
