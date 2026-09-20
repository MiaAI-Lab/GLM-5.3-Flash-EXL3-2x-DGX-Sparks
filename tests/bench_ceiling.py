#!/usr/bin/env python3
"""Is decode at the memory-bandwidth ceiling? Measure step time per batch size
and compare with the bytes a step must move.

Runs c streams in lock-step (same prompt length, ignore_eos, fixed tokens) and
takes the per-step time from the *server* counters: delta of
vllm:generation_tokens_total / delta of engine steps (iteration count from
vllm:spec_decode_num_drafts_total when spec is on, or completion tokens / c
when spec is off). It also samples `nvidia-smi dmon -s um` on this node during
the cell for DRAM-utilization %.

Bytes/step model (TP=2, per rank = half of each):
  non-expert weights (attn/KDA/dense/shared/embed/head) = 18.0 GiB total
  routed experts: 288/layer x 42 layers x 12.32 MiB, top-8; distinct experts
  touched per layer at batch B ~ 288*(1-(1-8/288)^B)
so per-rank bytes/step(B) = 0.5 * (18.0 GiB + 42*distinct(B)*12.32 MiB).
Ceiling: 236 GB/s measured pattern read (scripts/hardware/mem-bw-sweep), 273 spec.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import time
import urllib.request

URL = os.environ.get("GLM53_URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("SERVED_MODEL_NAME", "GLM-5.3-Flash-EXL3")
GIB = 1 << 30
NONEXPERT = 18.01 * GIB
EXPERT = 12.32 * (1 << 20)
LAYERS, EXPERTS, TOPK = 42, 288, 8


def bytes_per_step(b: int, spec_q: float = 1.0) -> float:
    # spec_q = mean query rows per stream per step (1 with spec off; k+1 with spec on)
    rows = b * spec_q
    distinct = EXPERTS * (1 - (1 - TOPK / EXPERTS) ** rows)
    return 0.5 * (NONEXPERT + LAYERS * distinct * EXPERT)


def metric(name: str) -> float:
    with urllib.request.urlopen(URL + "/metrics", timeout=10) as r:
        for line in r.read().decode().splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                return float(line.rsplit(" ", 1)[1])
    return 0.0


def one(i: int, tokens: int) -> int:
    body = {"model": MODEL, "messages": [{"role": "user", "content": f"Lane {i}. Write a long essay about the history of computing, in detail."}],
            "max_tokens": tokens, "temperature": 0, "ignore_eos": True, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["usage"]["completion_tokens"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--ceiling", type=float, default=236e9)
    ap.add_argument("--out", default="results/ceiling.jsonl")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    one(0, 16)  # warm
    for c in (int(x) for x in a.levels.split(",")):
        g0 = metric("vllm:generation_tokens_total"); d0 = metric("vllm:spec_decode_num_drafts_total"); acc0 = metric("vllm:spec_decode_num_accepted_tokens_total")
        dmon = subprocess.Popen(["nvidia-smi", "dmon", "-s", "um", "-d", "1"], stdout=subprocess.PIPE, text=True)
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(c) as ex:
            toks = sum(ex.map(lambda i: one(i, a.tokens), range(c)))
        wall = time.perf_counter() - t0
        dmon.terminate(); out = dmon.communicate()[0]
        g1 = metric("vllm:generation_tokens_total"); d1 = metric("vllm:spec_decode_num_drafts_total"); acc1 = metric("vllm:spec_decode_num_accepted_tokens_total")
        gen = g1 - g0; drafts = d1 - d0; acc = acc1 - acc0
        # steps: with spec on, one draft per running stream per step -> steps = drafts / c
        steps = drafts / c if drafts else gen / c
        step_ms = 1000 * wall / steps
        spec_q = 1 + (acc / drafts if drafts else 0)  # rows verified per stream per step ~ 1 + accepted
        b = bytes_per_step(c, spec_q)
        achieved = b / (step_ms / 1000)
        mem = [l.split() for l in out.splitlines() if l and not l.startswith("#")]
        mem_util = [int(r[2]) for r in mem if len(r) > 2 and r[2].isdigit()]
        rec = {"c": c, "tokens": toks, "wall_s": round(wall, 2), "steps": round(steps), "step_ms": round(step_ms, 1),
               "agg_tps": round(toks / wall, 1), "accept_per_step": round(spec_q - 1, 2),
               "model_bytes_per_step_GiB": round(b / GIB, 2), "achieved_GBps": round(achieved / 1e9, 1),
               "ceiling_GBps": a.ceiling / 1e9, "pct_of_ceiling": round(100 * achieved / a.ceiling, 1),
               "ideal_step_ms": round(1000 * b / a.ceiling, 1),
               "dmon_mem_util_pct_median": sorted(mem_util)[len(mem_util) // 2] if mem_util else None,
               "dmon_mem_util_pct_max": max(mem_util) if mem_util else None}
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
