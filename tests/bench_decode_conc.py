#!/usr/bin/env python3
"""Concurrent decode bench (C3/C6) + mixed prefill/decode for GLM-5.3-Flash EXL3.

Drives the live OpenAI API with N barrier-aligned streaming decode sessions
and reports aggregate + per-stream tok/s (same decode-tok/s definition as
tests/bench_decode.py), DFlash acceptance over the window, and completion
spread. Mixed mode adds one fresh cold prefill overlapping the decoders.

    GLM53_BENCH_BASE=http://127.0.0.1:8000 python3 tests/bench_decode_conc.py \
        --streams 3 --max-tokens 256 --prompt prose --runs 2 --out rec.json

    GLM53_BENCH_BASE=... python3 tests/bench_decode_conc.py \
        --streams 3 --max-tokens 256 --prompt prose --runs 1 \
        --mixed-prefill-tokens 16000 --out mixed.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load("bench_decode")
cp = _load("_run_cold_prefill")


def metrics_raw() -> str:
    req = urllib.request.Request(bench.BASE + "/metrics")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", "replace")


def scrape_metrics() -> dict:
    out: dict = {}
    for line in metrics_raw().splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            lhs, val = line.rsplit(" ", 1)
            v = float(val)
        except ValueError:
            continue
        if lhs.startswith("vllm:num_requests_running"):
            out["running"] = v
        elif lhs.startswith("vllm:num_requests_waiting"):
            out["waiting"] = v
        elif lhs.startswith("vllm:prefix_cache_hits_total"):
            out["prefix_hits"] = out.get("prefix_hits", 0.0) + v
        elif lhs.startswith("vllm:prefix_cache_queries_total"):
            out["prefix_queries"] = out.get("prefix_queries", 0.0) + v
    return out


def one_round(streams: int, max_tokens: int, prompt: str,
              mixed_prefill_tokens: int = 0) -> dict:
    barrier = threading.Barrier(streams + (1 if mixed_prefill_tokens else 0))
    results: list = [None] * streams
    errs: list = [None] * streams

    def worker(i: int):
        try:
            barrier.wait(timeout=120)
            results[i] = bench.stream_bench(max_tokens, prompt=prompt)
        except Exception as exc:  # noqa: BLE001
            errs[i] = f"{type(exc).__name__}: {exc}"

    prefill_rec: dict = {}

    def prefill_worker():
        try:
            barrier.wait(timeout=120)
            salt = cp.unique_salt()
            # ~1 token per "the " filler; usage reports the exact count.
            text = cp.build_user_text(mixed_prefill_tokens, salt)
            before = scrape_metrics()
            rec = cp.stream_chat(cp.chat_messages(text), timeout=600)
            after = scrape_metrics()
            rec["prefix_hit_delta"] = (
                after.get("prefix_hits", 0.0) - before.get("prefix_hits", 0.0)
            )
            prefill_rec.update(rec)
        except Exception as exc:  # noqa: BLE001
            prefill_rec["error"] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(streams)]
    if mixed_prefill_tokens:
        threads.append(threading.Thread(target=prefill_worker))
    spec_before = bench.spec_snapshot()
    met_before = scrape_metrics()
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=900)
    wall = time.perf_counter() - t0
    spec = bench.spec_delta(spec_before, bench.spec_snapshot())
    met_after = scrape_metrics()
    ok = [r for r in results if r is not None]
    tps = [r["tok_s"] for r in ok if r.get("tok_s")]
    ends = wall  # all joins done; per-stream gap from ttft+decode_s below
    spans = [(r.get("ttft_s") or 0) + (r.get("decode_s") or 0) for r in ok]
    return {
        "streams": streams,
        "max_tokens": max_tokens,
        "wall_s": wall,
        "completed": len(ok),
        "errors": [e for e in errs if e],
        "aggregate_tok_s": sum(tps) if tps else None,
        "per_stream_tok_s": tps,
        "per_stream_ttft_s": [r.get("ttft_s") for r in ok],
        "per_stream_span_s": spans,
        "max_stream_gap_s": (max(spans) - min(spans)) if spans else None,
        "finish_reasons": [r.get("finish_reason") for r in ok],
        "completion_tokens": [r.get("completion_tokens") for r in ok],
        "prompt_tokens": [r.get("prompt_tokens") for r in ok],
        "any_nan": any(r.get("nan") for r in ok),
        "spec": spec,
        "metrics_before": met_before,
        "metrics_after": met_after,
        "prefill": prefill_rec or None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt", choices=("prose", "structured"), default="prose")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--mixed-prefill-tokens", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    code, _ = bench.health()
    rec: dict = {
        "streams": args.streams,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "mixed_prefill_tokens": args.mixed_prefill_tokens,
        "health_code": code,
        "ts": time.time(),
        "rounds": [],
    }
    if code != 200:
        Path(args.out).write_text(json.dumps(rec, indent=2))
        return 2
    prompt = bench.STRUCTURED_PROMPT if args.prompt == "structured" else bench.BENCH_PROMPT
    print("[conc] warmup (1 stream, 32 tokens)", flush=True)
    w = bench.stream_bench(32, prompt=prompt)
    rec["warmup_tok_s"] = w.get("tok_s")
    for i in range(args.runs):
        print(f"[conc] round {i+1}/{args.runs} streams={args.streams}", flush=True)
        r = one_round(args.streams, args.max_tokens, prompt,
                      args.mixed_prefill_tokens)
        rec["rounds"].append(r)
        print(json.dumps({
            "aggregate_tok_s": r["aggregate_tok_s"],
            "per_stream_tok_s": [round(v, 2) if v else None for v in r["per_stream_tok_s"]],
            "max_gap_s": r["max_stream_gap_s"],
            "accept_ratio": r["spec"].get("accept_ratio"),
            "accepted_per_step": r["spec"].get("accepted_per_step"),
            "prefill_tok_s": (r["prefill"] or {}).get("prefill_tok_s"),
            "errors": r["errors"],
        }), flush=True)
    aggs = [r["aggregate_tok_s"] for r in rec["rounds"] if r["aggregate_tok_s"]]
    s = sorted(aggs)
    # Proper median (median-of-2 is the mean, not the max). Receipts from
    # the 2026-09-13 campaign predate this fix and their aggregate_median
    # field equals max-of-2; compare their rounds instead.
    n = len(s)
    rec["aggregate_median"] = (sum(s) / n) if n else None
    if n:
        rec["aggregate_median"] = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    print("wrote", args.out)
    return 0 if rec["aggregate_median"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
