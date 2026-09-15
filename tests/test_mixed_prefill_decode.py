#!/usr/bin/env python3
"""C1 vs mixed C2 decode tok/s (issue #6). Thinking off, temp 0, cold prefixes.

C1: one long prompt, measure decode after first token.
C2: start A on a *fresh* prefix (not C1's), wait for first token, start B
with a distinct cold prefix so A decodes while B prefills. Report A's
whole-window tok/s *and* tok/s during the overlap after B is submitted.

Does not assert that B prefills before A finishes (skip starves B). For
GLM53_MIXED_PREFILL_CHUNK=fair, inspect overlap_tok_s, b_ttft_s, and
whether B completed while A was still generating. Whole-window ratio still
fails above 5× (comment used to say 3×).

Decode tok/s = (completion_tokens - 1) / (last - first_token).
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from typing import Any

BASE = "http://127.0.0.1:8888"
MODEL = "GLM-5.3-Flash-EXL3"
TASK = " Count from 1 to 80. Output only the numbers, separated by spaces. No other text."


def _post_stream(body: dict, timeout: float = 1800.0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def make_prompt(tag: str, filler_words: int) -> str:
    return f"SESSION {tag} UNIQUE {tag[::-1]}. " + ("the " * filler_words) + TASK


def stream_one(prompt: str, max_tokens: int, out: dict[str, Any]) -> None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    first = None
    last = None
    usage = None
    finish = None
    events: list[float] = []
    try:
        with _post_stream(body) as resp:
            out["http"] = resp.status
            buf = b""
            while True:
                piece = resp.read(256)
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
                        now = time.perf_counter()
                        events.append(now)
                        if first is None:
                            first = now
                            out["first_event"].set()
                        last = now
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish = fr
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["first_event"].set()
        return
    t1 = time.perf_counter()
    completion = int((usage or {}).get("completion_tokens") or 0)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    decode_s = None if first is None or last is None or last <= first else (last - first)
    toks = max(completion - 1, 0)
    tps = toks / decode_s if decode_s and toks else None
    out.update(
        {
            "ttft_s": None if first is None else (first - t0),
            "wall_s": t1 - t0,
            "decode_s": decode_s,
            "tok_s": tps,
            "completion_tokens": completion,
            "prompt_tokens": prompt_tokens,
            "finish_reason": finish,
            "usage": usage,
            "t0": t0,
            "first": first,
            "last": last,
            "events": events,
        }
    )


def run_request(prompt: str, max_tokens: int) -> dict[str, Any]:
    out: dict[str, Any] = {"first_event": threading.Event()}
    stream_one(prompt, max_tokens, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filler-words", type=int, default=48000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument(
        "--b-max-tokens",
        type=int,
        default=None,
        help="B's max_tokens in C2 (default: same as --max-tokens). Use 8 for a prefill-heavy newcomer.",
    )
    ap.add_argument("--out", default="/tmp/mixed-prefill-decode.json")
    args = ap.parse_args()
    b_max = args.max_tokens if args.b_max_tokens is None else args.b_max_tokens

    p1 = make_prompt("ALPHA", args.filler_words)
    p2 = make_prompt("BRAVO", args.filler_words + 17)
    p3 = make_prompt("CHARLIE", args.filler_words + 31)

    print("[c1] solo long prefill + decode", flush=True)
    c1 = run_request(p1, args.max_tokens)
    keys = ("tok_s", "ttft_s", "prompt_tokens", "completion_tokens", "finish_reason", "error")
    print(json.dumps({k: c1.get(k) for k in keys}), flush=True)

    print("[c2] A decode while B cold-prefills", flush=True)
    a: dict[str, Any] = {"first_event": threading.Event()}
    b: dict[str, Any] = {"first_event": threading.Event()}
    ta = threading.Thread(target=stream_one, args=(p3, args.max_tokens, a), daemon=True)
    tb = threading.Thread(target=stream_one, args=(p2, b_max, b), daemon=True)
    ta.start()
    if not a["first_event"].wait(timeout=900):
        print("A never emitted a first token", flush=True)
        rec = {"c1": c1, "c2_a": a, "c2_b": b, "error": "A TTFT timeout"}
        PathWrite = __import__("pathlib").Path
        PathWrite(args.out).write_text(json.dumps(rec, indent=2, default=str))
        return 2
    b_submit = time.perf_counter()
    print(f"[c2] A first token ttft={a.get('ttft_s')} — launching B", flush=True)
    tb.start()
    ta.join()
    tb.join()
    for label, d in (("A", a), ("B", b)):
        print(
            json.dumps(
                {
                    "lane": label,
                    **{
                        k: d.get(k)
                        for k in (
                            "tok_s",
                            "ttft_s",
                            "prompt_tokens",
                            "completion_tokens",
                            "finish_reason",
                            "error",
                        )
                    },
                }
            ),
            flush=True,
        )

    overlap_tok_s = None
    overlap_events = 0
    a_events = list(a.get("events") or [])
    if a_events:
        after = [t for t in a_events if t >= b_submit]
        overlap_events = len(after)
        if len(after) >= 2:
            overlap_tok_s = (len(after) - 1) / (after[-1] - after[0])
    b_done_before_a = bool(
        a.get("last") and b.get("last") and b["last"] < a["last"]
    )
    rec = {
        "filler_words": args.filler_words,
        "a_max_tokens": args.max_tokens,
        "b_max_tokens": b_max,
        "c1_tok_s": c1.get("tok_s"),
        "c1_ttft_s": c1.get("ttft_s"),
        "c1_prompt_tokens": c1.get("prompt_tokens"),
        "c2_a_tok_s": a.get("tok_s"),
        "c2_a_ttft_s": a.get("ttft_s"),
        "c2_b_tok_s": b.get("tok_s"),
        "c2_b_ttft_s": b.get("ttft_s"),
        "c2_a_prompt_tokens": a.get("prompt_tokens"),
        "c2_b_prompt_tokens": b.get("prompt_tokens"),
        "c2_a_overlap_tok_s": overlap_tok_s,
        "c2_a_overlap_events": overlap_events,
        "c2_b_finished_during_a": b_done_before_a,
        "ratio_c1_over_c2a": None
        if not c1.get("tok_s") or not a.get("tok_s")
        else round(c1["tok_s"] / a["tok_s"], 2),
        "ratio_c1_over_overlap": None
        if not c1.get("tok_s") or not overlap_tok_s
        else round(c1["tok_s"] / overlap_tok_s, 2),
    }
    __import__("pathlib").Path(args.out).write_text(json.dumps(rec, indent=2, default=str))
    print(json.dumps(rec, indent=2), flush=True)
    print("wrote", args.out, flush=True)
    # Fail if mixed decode is still ~11× slower (issue #6). Allow ~3×.
    if rec["c1_tok_s"] and rec["c2_a_tok_s"] and rec["c1_tok_s"] / rec["c2_a_tok_s"] > 5:
        return 3
    return 0 if rec["c1_tok_s"] and rec["c2_a_tok_s"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
