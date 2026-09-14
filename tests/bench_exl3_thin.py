#!/usr/bin/env python3
"""Standalone thin/small-M EXL3 decode probe (baseline + A/B).

Builds a synthetic fused-state MoE layer at TP-local decode geometry and
times the exact decode dispatch (`apply_exl3_fused_moe` with tokens <= cap,
i.e. one `exl3_moe` launch) for representative C1/C3/C6 row counts under two
routing distributions:

* correlated: 8-token speculative-block-like groups sharing a hot expert pool
* uniform:   uncorrelated control

Run with the server STOPPED (needs the whole GPU for stable timings)::

    EXL3_TEMP_ROWS_FUSED=128 GLM53_EXL3_MOE_FAST=0 python3 tests/bench_exl3_thin.py
    EXL3_TEMP_ROWS_FUSED=128 GLM53_EXL3_MOE_FAST=1 python3 tests/bench_exl3_thin.py

The two runs differ only in the native dispatch (opt-in fast K4/N256 path vs
stock kernel), so their receipts are directly comparable. Prints one JSON
document per shape to stdout; use --receipt to also append JSONL rows.

Environment:
    GLM53_EXL3_MOE_FAST  0 (default) or 1. Read once per process by the
                         native dispatcher; not hot-reloadable.
    EXL3_TEMP_ROWS_FUSED  fused temp rows / kernel cap (default 128).
    EXL3_THIN_ROWS        comma list overriding the default row sweep.
    EXL3_THIN_ITERS       timed iterations per case (default 100).
    EXL3_THIN_WARMUP      warmup iterations per case (default 20).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time


DEFAULT_ROWS = (1, 8, 24, 48, 64, 128)
HIDDEN = 4096
INTER = 1024  # TP-local intermediate; hidden % 256 == inter % 256 == 0 (N256)
N_EXP = 32
TOPK = 8  # routed experts per token, independent of DFlash verification length


def build_layer(device, shared_suh: bool = True):
    import types

    import torch
    from vllm.model_executor.layers.quantization.exl3 import (
        Exl3Config,
        Exl3MoEMethod,
        MCG_MARKER_SIGNED_INT32,
    )

    moe = types.SimpleNamespace(swiglu_limit=10.0)
    method = Exl3MoEMethod(moe, Exl3Config())
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        num_experts=N_EXP,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTER,
        params_dtype=torch.float16,
    )
    g = torch.Generator(device="cpu")
    g.manual_seed(20260913)
    with torch.no_grad():
        layer.w13_trellis.copy_(
            torch.randint(
                -30000, 30000, tuple(layer.w13_trellis.shape),
                dtype=torch.int16, generator=g,
            )
        )
        layer.w2_trellis.copy_(
            torch.randint(
                -30000, 30000, tuple(layer.w2_trellis.shape),
                dtype=torch.int16, generator=g,
            )
        )
        layer.w13_suh.copy_(torch.randn(tuple(layer.w13_suh.shape), generator=g).half())
        layer.w13_svh.copy_(torch.randn(tuple(layer.w13_svh.shape), generator=g).half())
        layer.w2_suh.copy_(torch.randn(tuple(layer.w2_suh.shape), generator=g).half())
        layer.w2_svh.copy_(torch.randn(tuple(layer.w2_svh.shape), generator=g).half())
        if shared_suh:
            layer.w13_suh[:, 1].copy_(layer.w13_suh[:, 0])
        layer.w13_mcg.fill_(MCG_MARKER_SIGNED_INT32)
        layer.w2_mcg.fill_(MCG_MARKER_SIGNED_INT32)
    layer = layer.to(device)
    method.process_weights_after_loading(layer)
    return method, layer


def make_routing(rows: int, mode: str, seed: int, device):
    import torch

    g = torch.Generator(device="cpu")
    g.manual_seed(1000 + seed)
    if mode == "uniform":
        ids = torch.stack([
            torch.randperm(N_EXP, generator=g)[:TOPK] for _ in range(rows)
        ])
    elif mode == "correlated":
        # Verification-like eight-token blocks share a distinct routed set.
        # Sampling with replacement is not top-k: duplicate experts can make
        # an expert's route count exceed rows and the thin-kernel row cap.
        ids = torch.empty(rows, TOPK, dtype=torch.long)
        for b in range(0, rows, 8):
            pool = torch.randperm(N_EXP, generator=g)[:TOPK]
            nb = min(8, rows - b)
            order = torch.stack([
                torch.randperm(TOPK, generator=g) for _ in range(nb)
            ])
            ids[b:b + nb] = pool[order]
    elif mode == "skewed":
        # Every token selects the same hot experts, each exactly once.
        ids = torch.arange(TOPK).expand(rows, TOPK).clone()
    else:
        raise ValueError(mode)
    gw = torch.Generator(device="cpu")
    gw.manual_seed(7777 + seed)
    weights = torch.rand(rows, TOPK, generator=gw).half()
    weights = weights / weights.sum(dim=1, keepdim=True)
    return ids.to(device), weights.to(device)


def time_case(apply, x, ids, w, layer, warmup: int, iters: int):
    import torch

    from vllm.model_executor.layers.quantization.exl3 import apply_exl3_fused_moe

    inners = layer._exl3_inners
    for _ in range(warmup):
        apply_exl3_fused_moe(x, ids, w, layer, inners, None, 10.0)
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    ms = []
    for _ in range(iters):
        start_ev.record()
        apply_exl3_fused_moe(x, ids, w, layer, inners, None, 10.0)
        end_ev.record()
        end_ev.synchronize()
        ms.append(start_ev.elapsed_time(end_ev))
    torch.cuda.synchronize()
    return ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", default="",
                        help="append JSONL rows to this file")
    parser.add_argument("--iters", type=int,
                        default=int(os.environ.get("EXL3_THIN_ITERS", "100")))
    parser.add_argument("--warmup", type=int,
                        default=int(os.environ.get("EXL3_THIN_WARMUP", "20")))
    parser.add_argument("--rows", default=os.environ.get("EXL3_THIN_ROWS", ""),
                        help="comma list, e.g. 1,8,24,48,64,128")
    parser.add_argument("--independent", action="store_true",
                        help="use independent gate/up SUH (no transform reuse)")
    args = parser.parse_args()

    import torch
    from vllm.model_executor.layers.quantization.exl3 import temp_rows_fused

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    rows = (tuple(int(v) for v in args.rows.split(",") if v.strip())
            or DEFAULT_ROWS)
    fast = os.environ.get("GLM53_EXL3_MOE_FAST", "0")
    cap = temp_rows_fused()
    assert max(rows) <= cap, f"rows {max(rows)} exceed fused cap {cap}"

    import exllamav3_ext

    native_fast = (hasattr(exllamav3_ext, "glm53_fast_moe_version")
                   and exllamav3_ext.glm53_fast_moe_version() == 1)
    _method, layer = build_layer(device, shared_suh=not args.independent)
    gate_alias = (layer._exl3_ptrs["up_suh"] is layer._exl3_ptrs["gate_suh"])

    def _randn(rows, cols, gen):
        return torch.randn(rows, cols, generator=gen,
                           dtype=torch.float16, device="cpu").to(device)

    xg = torch.Generator(device="cpu")
    xg.manual_seed(4242)

    out_rows = []
    for n in rows:
        for mode in ("correlated", "uniform"):
            ids, w = make_routing(n, mode, seed=n, device=device)
            with torch.no_grad():
                x = _randn(n, HIDDEN, xg)
            ms = time_case(None, x, ids, w, layer, args.warmup, args.iters)
            row = {
                "rows": n,
                "routing": mode,
                "fast_env": fast,
                "native_fast_present": native_fast,
                "gate_suh_aliased": gate_alias,
                "cap": cap,
                "hidden": HIDDEN,
                "intermediate_local": INTER,
                "n_exp": N_EXP,
                "topk": TOPK,
                "iters": args.iters,
                "warmup": args.warmup,
                "median_ms": statistics.median(ms),
                "mean_ms": statistics.fmean(ms),
                "min_ms": min(ms),
                "max_ms": max(ms),
                "capability": list(torch.cuda.get_device_capability()),
                "device": torch.cuda.get_device_name(0),
            }
            out_rows.append(row)
            print(json.dumps(row), flush=True)
    if args.receipt:
        with open(args.receipt, "a") as fh:
            for row in out_rows:
                fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
