#!/usr/bin/env python3
"""Live cold-prefill ladder for GLM-5.3-Flash-EXL3 on :8888.

Protocol matches the published kit receipts: temp 0, thinking off via
top-level chat_template_kwargs, stream + include_usage, max_tokens=8,
unique salt per cold request, one-at-a-time, TTFT = first content token.
"""
from __future__ import annotations

import json
import secrets
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8888"
SERVED = "GLM-5.3-Flash-EXL3"
FILLER = "the "
TASK = "Reply with OK."
OUT_JSON = Path("/home/mia/NewModels/glm-5.3-flash-sm120/docs/_cold_prefill_raw.json")

# target prompt_tokens, http timeout seconds
LADDER = [
    ("~8k", 8_000, 180),
    ("~12k", 12_000, 180),
    ("~16k", 16_000, 180),
    ("~100k", 100_000, 600),
    ("~256k", 256_000, 1_200),
    ("~300k", 300_000, 1_200),
]


def http_get(path: str, timeout: float = 15.0):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def http_post(path: str, body: dict, timeout: float, stream: bool = False):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize_messages(messages: list[dict], timeout: float = 180.0) -> int:
    body = {"model": SERVED, "messages": messages}
    with http_post("/tokenize", body, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode())
    return int(obj["count"])


def parse_metrics(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        # drop labels for the counters we care about
        if line.startswith("vllm:prefix_cache_hits_total{"):
            out["prefix_cache_hits_total"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:prefix_cache_queries_total{"):
            out["prefix_cache_queries_total"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:prompt_tokens_total{"):
            out["prompt_tokens_total"] = float(line.rsplit(" ", 1)[1])
        elif 'source="local_compute"' in line and line.startswith(
            "vllm:prompt_tokens_by_source_total{"
        ):
            out["local_compute"] = float(line.rsplit(" ", 1)[1])
        elif 'source="local_cache_hit"' in line and line.startswith(
            "vllm:prompt_tokens_by_source_total{"
        ):
            out["local_cache_hit"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:gpu_cache_usage_perc{"):
            out["gpu_cache_usage_perc"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_running{"):
            out["num_requests_running"] = float(line.rsplit(" ", 1)[1])
    return out


def metrics_snapshot() -> dict[str, float]:
    _, raw = http_get("/metrics")
    return parse_metrics(raw)


def unique_salt() -> str:
    return f"COLD-PREFILL salt={uuid.uuid4()} pad={secrets.token_hex(24)}"


def build_user_text(n_filler: int, salt: str) -> str:
    return f"{salt}\n{FILLER * n_filler}\n{TASK}"


def chat_messages(user_text: str, extra: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "user", "content": user_text}]
    if extra:
        msgs.extend(extra)
    return msgs


def calibrate(target: int, salt: str, timeout: float = 180.0) -> tuple[int, int, str]:
    """Pick filler count so tokenize(messages) is within ~2% of target."""
    # overhead: salt + task + chat template, filler ≈ 1 token per "the "
    overhead_text = build_user_text(0, salt)
    overhead = tokenize_messages(chat_messages(overhead_text), timeout=timeout)
    n = max(target - overhead, 1)
    text = build_user_text(n, salt)
    got = tokenize_messages(chat_messages(text), timeout=timeout)
    # one correction step using measured ratio
    if got > 0 and abs(got - target) / target > 0.005:
        per = (got - overhead) / n if n else 1.0
        if per <= 0:
            per = 1.0
        n = max(int(round((target - overhead) / per)), 1)
        text = build_user_text(n, salt)
        got = tokenize_messages(chat_messages(text), timeout=timeout)
    # fine adjust by token delta (each filler ~1 token)
    if abs(got - target) / target > 0.005:
        n = max(n + (target - got), 1)
        text = build_user_text(n, salt)
        got = tokenize_messages(chat_messages(text), timeout=timeout)
    rel = abs(got - target) / target
    print(f"  calibrated tokenize={got} target={target} filler={n} rel={rel:.3%}", flush=True)
    if rel > 0.02:
        print("  WARNING: tokenize still >2% off target", flush=True)
    return n, got, text


def stream_chat(messages: list[dict], timeout: float) -> dict:
    body = {
        "model": SERVED,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 8,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    first_content = None
    chunks: list[str] = []
    reasoning: list[str] = []
    usage = None
    finish = None
    http = None
    err = None
    try:
        with http_post("/v1/chat/completions", body, timeout=timeout) as resp:
            http = resp.status
            buf = b""
            while True:
                piece = resp.read(4096)
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
                    content = delta.get("content") or ""
                    reason = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if reason:
                        reasoning.append(reason)
                    if content:
                        if first_content is None:
                            first_content = time.perf_counter()
                        chunks.append(content)
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish = fr
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    t1 = time.perf_counter()
    text = "".join(chunks)
    reason_text = "".join(reasoning)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    ttft = None if first_content is None else (first_content - t0)
    prefill = None
    if ttft and ttft > 0 and prompt_tokens > 0:
        prefill = prompt_tokens / ttft
    gen_ok = bool(text) and ("ok" in text.lower()) and ("nan" not in text.lower())
    cached = None
    details = (usage or {}).get("prompt_tokens_details") or {}
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        cached = details.get("cached_tokens")
    return {
        "http": http,
        "error": err,
        "finish_reason": finish,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached,
        "ttft_s": ttft,
        "wall_s": t1 - t0,
        "prefill_tok_s": prefill,
        "gen": text,
        "reasoning": reason_text,
        "gen_ok": gen_ok,
        "usage": usage,
    }


def delta_metrics(before: dict, after: dict) -> dict:
    keys = (
        "prefix_cache_hits_total",
        "prefix_cache_queries_total",
        "prompt_tokens_total",
        "local_compute",
        "local_cache_hit",
    )
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in keys}


def main() -> int:
    st, body = http_get("/health")
    print(f"GET /health -> {st} {body!r}", flush=True)
    if st != 200:
        print("STOP: health failed", flush=True)
        return 1
    global SERVED
    st, raw = http_get("/v1/models")
    models = json.loads(raw)
    served = models["data"][0]["id"]
    max_len = models["data"][0].get("max_model_len")
    print(f"GET /v1/models -> {st} id={served} max_model_len={max_len}", flush=True)
    if served != SERVED:
        print(f"NOTE: using served id {served} (not {SERVED})", flush=True)
    SERVED = served

    m0 = metrics_snapshot()
    print(f"metrics at start: {json.dumps(m0)}", flush=True)

    results: list[dict] = []
    eightk_user = None
    eightk_assistant = None

    for i, (name, target, timeout) in enumerate(LADDER):
        salt = unique_salt()
        print(f"\n=== {name} cold target={target} timeout={timeout}s ===", flush=True)
        n, tok_est, user_text = calibrate(target, salt, timeout=min(timeout, 180))
        msgs = chat_messages(user_text)
        before = metrics_snapshot()
        rec = stream_chat(msgs, timeout=timeout)
        after = metrics_snapshot()
        rec["name"] = name
        rec["kind"] = "cold"
        rec["target"] = target
        rec["filler_count"] = n
        rec["tokenize_est"] = tok_est
        rec["salt"] = salt
        rec["metrics_delta"] = delta_metrics(before, after)
        rec["gpu_cache_usage_perc"] = after.get("gpu_cache_usage_perc")
        results.append(rec)
        print(json.dumps({k: rec[k] for k in rec if k not in ("usage", "salt")}, default=str), flush=True)

        if i == 0:
            eightk_user = user_text
            eightk_assistant = rec.get("gen") or ""
            print("\n=== ~8k APC follow-up (same user text + assistant + short user turn) ===", flush=True)
            follow_msgs = chat_messages(
                eightk_user,
                extra=[
                    {"role": "assistant", "content": eightk_assistant},
                    {"role": "user", "content": "Confirm with OK."},
                ],
            )
            before = metrics_snapshot()
            follow = stream_chat(follow_msgs, timeout=timeout)
            after = metrics_snapshot()
            follow["name"] = "~8k follow-up"
            follow["kind"] = "apc"
            follow["target"] = target
            follow["metrics_delta"] = delta_metrics(before, after)
            follow["gpu_cache_usage_perc"] = after.get("gpu_cache_usage_perc")
            follow["assistant_echo"] = eightk_assistant
            results.append(follow)
            print(json.dumps({k: follow[k] for k in follow if k not in ("usage",)}, default=str), flush=True)

    payload = {
        "served_model": SERVED,
        "max_model_len": max_len,
        "started_metrics": m0,
        "ended_metrics": metrics_snapshot(),
        "results": results,
        "wall_clock": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
