#!/usr/bin/env python3
"""Decode-throughput ladder c1..c8: N distinct streaming chats in parallel.

Aggregate / per-stream tok/s from the server `usage` (not chunk counts), TTFT
and inter-token p50/p95, plus the spec-decode acceptance and draft counters
delta from /metrics for the cell. Coding-agent shaped prompts (~1.5k tokens,
distinct per lane), thinking off, temperature 0, fixed max_tokens.

  python3 tests/bench_c8.py --levels 1,2,4,8 --max-tokens 512 --tag baseline
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import statistics
import time
import urllib.request

URL = os.environ.get("GLM53_URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("SERVED_MODEL_NAME", "GLM-5.3-Flash-EXL3")
KEY = os.environ.get("VLLM_API_KEY")

TASKS = [
    "Implement an LRU cache in Python with O(1) get/put and unit tests.",
    "Write a Rust function that parses RFC3339 timestamps without external crates, with tests.",
    "Explain and implement Dijkstra with a binary heap in Go; include a benchmark.",
    "Write a bash script that rotates logs older than 7 days, compresses them, and is idempotent.",
    "Implement a rate limiter (token bucket) in TypeScript with an async API and tests.",
    "Write a SQL migration and the matching SQLAlchemy model for a multi-tenant audit log.",
    "Implement binary search tree deletion in C with all three cases and a fuzz test harness.",
    "Write a Python asyncio TCP echo server with graceful shutdown and a load-test client.",
]


def prompt_for(lane: int, seed: int, pad_tokens: int) -> str:
    rng = random.Random(seed * 100 + lane)
    # distinct filler so lanes never share a cached prefix
    filler = " ".join(f"ctx{rng.randrange(10**6)}" for _ in range(pad_tokens // 2))
    return (
        f"Project notes (lane {lane}): {filler}\n\n"
        f"Task: {TASKS[lane % len(TASKS)]} Be thorough and show complete code."
    )


def metrics() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(URL + "/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#"):
                    continue
                for key in ("vllm:spec_decode_num_accepted_tokens_total",
                            "vllm:spec_decode_num_draft_tokens_total",
                            "vllm:spec_decode_num_drafts_total",
                            "vllm:num_preemptions_total"):
                    if line.startswith(key):
                        out[key] = float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return out


def one(lane: int, seed: int, pad: int, max_tokens: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_for(lane, seed, pad)}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    hdr = {"Content-Type": "application/json"}
    if KEY:
        hdr["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers=hdr)
    t0 = time.perf_counter()
    ttft = None
    stamps: list[float] = []
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                break
            obj = json.loads(p)
            ch = obj.get("choices")
            if ch and ch[0].get("delta", {}).get("content"):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                stamps.append(now)
            if obj.get("usage"):
                usage = obj["usage"]
    t1 = time.perf_counter()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    comp = (usage or {}).get("completion_tokens", 0)
    return {
        "lane": lane, "ttft": ttft, "wall": t1 - t0, "completion_tokens": comp,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "decode_tps": comp / (t1 - (t0 + (ttft or 0))) if comp and ttft is not None else None,
        "itl_p50": statistics.median(gaps) if gaps else None,
        "itl_p95": sorted(gaps)[int(0.95 * (len(gaps) - 1))] if gaps else None,
    }


def cell(c: int, seed: int, pad: int, max_tokens: int) -> dict:
    m0 = metrics()
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        lanes = list(ex.map(lambda i: one(i, seed, pad, max_tokens), range(c)))
    wall = time.perf_counter() - t0
    m1 = metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    tot = sum(l["completion_tokens"] for l in lanes)
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    draft_toks = d.get("vllm:spec_decode_num_draft_tokens_total", 0)
    return {
        "c": c, "wall_s": round(wall, 2), "completion_tokens": tot,
        "agg_tps": round(tot / wall, 2),
        "per_stream_tps": round(statistics.mean(l["decode_tps"] for l in lanes if l["decode_tps"]), 2),
        "ttft_p50": round(statistics.median(l["ttft"] for l in lanes), 3),
        "ttft_max": round(max(l["ttft"] for l in lanes), 3),
        "itl_p50_ms": round(1000 * statistics.median(l["itl_p50"] for l in lanes if l["itl_p50"]), 1),
        "itl_p95_ms": round(1000 * max(l["itl_p95"] for l in lanes if l["itl_p95"]), 1),
        "prompt_tokens": lanes[0]["prompt_tokens"],
        "spec_accept_rate": round(acc / draft_toks, 3) if draft_toks else None,
        "spec_mean_accept_len": round(acc / drafts, 2) if drafts else None,
        "preemptions": d.get("vllm:num_preemptions_total", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--pad-tokens", type=int, default=1200)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/c8-ladder.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # warm shapes once
    one(0, 999, 64, 32)
    for c in (int(x) for x in a.levels.split(",")):
        for rep in range(a.reps):
            r = cell(c, a.seed + rep, a.pad_tokens, a.max_tokens)
            r.update({"tag": a.tag, "rep": rep, "ts": time.time()})
            print(json.dumps(r), flush=True)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            time.sleep(2)


if __name__ == "__main__":
    main()
