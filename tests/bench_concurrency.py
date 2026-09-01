#!/usr/bin/env python3
"""Measure whole-group throughput for concurrent structured requests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import threading
import time
import uuid

import bench_decode as bench


def run_trial(args: argparse.Namespace, run_tag: str) -> dict:
    barrier = threading.Barrier(args.clients + 1)
    prompts = [
        f"Benchmark run {run_tag}, lane {lane}. Count from 1 to 200. "
        "Output only the numbers, separated by spaces. No other text."
        for lane in range(1, args.clients + 1)
    ]

    def run_one(prompt: str) -> dict:
        barrier.wait()
        return bench.stream_bench(args.max_tokens, prompt)

    before = bench.spec_snapshot()
    with ThreadPoolExecutor(max_workers=args.clients) as pool:
        futures = [pool.submit(run_one, prompt) for prompt in prompts]
        started = time.perf_counter()
        barrier.wait()
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - started
    after = bench.spec_snapshot()

    completion_tokens = [result["completion_tokens"] for result in results]
    return {
        "run_tag": run_tag,
        "wall_s": wall_s,
        "wall_throughput_tok_s": sum(completion_tokens) / wall_s,
        "sum_per_request_decode_tok_s": sum(
            result["tok_s"]
            for result in results
            if result["tok_s"] is not None
        ),
        "per_request_tok_s": [result["tok_s"] for result in results],
        "per_request_ttft_s": [result["ttft_s"] for result in results],
        "completion_tokens": completion_tokens,
        "http_status": [result["http"] for result in results],
        "finish_reason": [result["finish_reason"] for result in results],
        "any_nan": any(result["nan"] for result in results),
        "speculative": bench.spec_delta(before, after),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=bench.BASE)
    parser.add_argument("--model", default=bench.MODEL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-tag")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    if args.clients < 1:
        parser.error("--clients must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    bench.BASE = args.base_url.rstrip("/")
    bench.MODEL = args.model
    base_tag = args.run_tag or uuid.uuid4().hex
    runs = [
        run_trial(args, f"{base_tag}-r{run_number}")
        for run_number in range(1, args.runs + 1)
    ]
    record = {
        "phase": "structured-concurrency",
        "clients": args.clients,
        "unique_prompts": True,
        "base_run_tag": base_tag,
        "runs_requested": args.runs,
        "max_tokens": args.max_tokens,
        "wall_throughput_median_tok_s": statistics.median(
            run["wall_throughput_tok_s"] for run in runs
        ),
        "runs": runs,
    }

    output = Path(args.out)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    failed = any(
        run["any_nan"]
        or any(status != 200 for status in run["http_status"])
        or any(tokens != args.max_tokens for tokens in run["completion_tokens"])
        or any(reason != "length" for reason in run["finish_reason"])
        for run in runs
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
