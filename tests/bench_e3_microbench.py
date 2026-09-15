#!/usr/bin/env python3
"""Isolated MoE-layer timing: E2 (shipped) vs E3 grouped at TP2 per-rank shapes.

Reproducible replacement for the prototype harness (random K4 trellises,
random-normal scales, shared gate/up SUH). Timing uses CUDA events around the
full apply (thin fused kernel + fat path + routing glue, i.e. one MoE layer's
expert work), explicit warmup, and saved repetitions. Routing: a Zipf skew
ladder plus a uniform case; the actual served routing distribution is NOT
reproduced here (no live fixture). Parity at production geometry is asserted
against the LinearEXL3 loop with the frozen E2-derived tolerances.

Synthetic numbers here are NOT end-to-end prefill measurements.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types

os.environ.setdefault("EXL3_FAT_EXPERT_LOG", "0")

import torch  # noqa: E402

HID, INTER, NEXP, TOPK = 4096, 1024, 288, 8


def make_layer(n_exp=NEXP, hidden=HID, inter=INTER, seed=0):
    from vllm.model_executor.layers.quantization.exl3 import (
        MCG_MARKER_SIGNED_INT32, Exl3Config, Exl3MoEMethod,
    )

    moe = types.SimpleNamespace(swiglu_limit=10.0)
    method = Exl3MoEMethod(moe, Exl3Config())
    layer = torch.nn.Module()
    method.create_weights(layer, num_experts=n_exp, hidden_size=hidden,
                          intermediate_size_per_partition=inter, params_dtype=torch.float16)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    with torch.no_grad():
        layer.w13_trellis.copy_(torch.randint(-30000, 30000, tuple(layer.w13_trellis.shape), dtype=torch.int16, generator=g))
        layer.w2_trellis.copy_(torch.randint(-30000, 30000, tuple(layer.w2_trellis.shape), dtype=torch.int16, generator=g))
        for p in (layer.w13_suh, layer.w13_svh, layer.w2_suh, layer.w2_svh):
            p.copy_((torch.randn(tuple(p.shape), generator=g) * 0.5).half())
        layer.w13_suh[:, 1].copy_(layer.w13_suh[:, 0])
        layer.w13_mcg.fill_(MCG_MARKER_SIGNED_INT32)
        layer.w2_mcg.fill_(MCG_MARKER_SIGNED_INT32)
    layer = layer.to("cuda:0")
    method.process_weights_after_loading(layer)
    return layer


def routing(tokens, n_exp=NEXP, topk=TOPK, skew=1.0, seed=0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    p = 1.0 / torch.arange(1, n_exp + 1).float() ** skew
    p = p[torch.randperm(n_exp, generator=g)]
    ids = torch.multinomial(p.expand(tokens, -1), topk, replacement=False, generator=g)
    w = torch.rand(tokens, topk, generator=g).softmax(-1).half()
    return ids.cuda(), w.cuda()


def time_fn(fn, iters=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def set_cap(layer, cap):
    from vllm.model_executor.layers.quantization.exl3 import _FUSED_TEMP_CACHE, build_exl3_fused_state

    os.environ["EXL3_TEMP_ROWS_FUSED"] = str(cap)
    _FUSED_TEMP_CACHE.clear()
    build_exl3_fused_state(layer, layer._exl3_inners)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=7168)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--skews", default="0.5,1.0,1.5,0.0")
    ap.add_argument("--e3-caps", default="32,64,128")
    ap.add_argument("--e2-caps", default="256,32")
    ap.add_argument("--parity-tokens", type=int, default=1024)
    args = ap.parse_args()
    from vllm.model_executor.layers.quantization.exl3 import (
        _record_exl3_fat_resolution, apply_exl3_experts, apply_exl3_python_loop, exl3_fat_diag,
        exl3_fat_moe_symbols,
    )
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_exl3_overlay import _assert_e3_within, _err_stats  # frozen tolerances

    assert exl3_fat_moe_symbols(), "E3 kernels are not loaded"
    os.environ.update({"EXL3_FAT_KERNEL": "1", "EXL3_FAT_SORTED": "0", "EXL3_FAT_BATCHED": "0", "EXL3_MOE_ROW_TILE": "0"})
    rec: dict = {"tokens": args.tokens, "iters": args.iters, "hidden": HID, "intermediate": INTER, "n_exp": NEXP, "topk": TOPK,
                 "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability()),
                 "scales": "random-normal*0.5 (shared gate/up suh), random K4 trellis", "cases": [], "parity": {}}

    # --- parity at production geometry (before any timing; tolerances frozen from E2) ---
    os.environ["EXL3_FAT_GROUPED"] = "1"
    layer = make_layer()
    _record_exl3_fat_resolution(layer)
    assert layer._exl3_fat_effective_tier == "grouped", (layer._exl3_fat_effective_tier, layer._exl3_fat_tier_reason)
    pt = args.parity_tokens
    x = (torch.randn(pt, HID, generator=torch.Generator().manual_seed(3))).half().cuda()
    ids, w = routing(pt, skew=1.0, seed=3)
    y_loop = apply_exl3_python_loop(x, ids.long(), w, layer._exl3_inners, None, 10.0)
    set_cap(layer, 32)
    os.environ["EXL3_FAT_GROUPED"] = "0"
    _record_exl3_fat_resolution(layer)
    y_e2 = apply_exl3_experts(x, ids, w, layer)
    assert layer._exl3_last_fat_fallback == "kernel", layer._exl3_last_fat_fallback
    y_e2b = apply_exl3_experts(x, ids, w, layer)
    e2 = _err_stats(y_loop, y_e2)
    os.environ["EXL3_FAT_GROUPED"] = "1"
    _record_exl3_fat_resolution(layer)
    y_e3 = apply_exl3_experts(x, ids, w, layer)
    assert layer._exl3_last_fat_fallback == "grouped", layer._exl3_last_fat_fallback
    y_e3b = apply_exl3_experts(x, ids, w, layer)
    e3 = _err_stats(y_loop, y_e3)
    counts = torch.bincount(ids.reshape(-1), minlength=NEXP)
    rec["parity"] = {"tokens": pt, "cap": 32, "fat_experts": int((counts > 32).sum()), "max_rows": int(counts.max()),
                     "e2_vs_loop": e2, "e2_repeat": _err_stats(y_e2, y_e2b), "e3_vs_loop": e3,
                     "e3_repeat": _err_stats(y_e3, y_e3b), "e3_vs_e2": _err_stats(y_e2, y_e3)}
    try:
        _assert_e3_within("prod_geometry", e2, e3)
        rec["parity"]["ok"] = True
    except AssertionError as exc:
        rec["parity"]["ok"] = False
        rec["parity"]["error"] = str(exc)
    print("PARITY", "OK" if rec["parity"]["ok"] else "FAIL", json.dumps({k: rec["parity"][k] for k in ("fat_experts", "max_rows")}),
          "e2 maxabs=%.4f nrmse=%.6f | e3 maxabs=%.4f nrmse=%.6f" % (e2["maxabs"], e2["nrmse"], e3["maxabs"], e3["nrmse"]), flush=True)
    del x, y_loop, y_e2, y_e2b, y_e3, y_e3b
    torch.cuda.empty_cache()

    # --- timing ladder ---
    x = torch.randn(args.tokens, HID, dtype=torch.float16, device="cuda")
    gflop = args.tokens * TOPK * 3 * 2 * HID * INTER / 1e9
    for skew in [float(s) for s in args.skews.split(",")]:
        ids, w = routing(args.tokens, skew=skew, seed=1)
        counts = torch.bincount(ids.reshape(-1), minlength=NEXP)
        dist = {"skew": skew, "max_rows": int(counts.max()), "mean_rows": float(counts.float().mean()),
                "n_gt_256": int((counts > 256).sum()), "n_gt_128": int((counts > 128).sum()),
                "n_gt_64": int((counts > 64).sum()), "n_gt_32": int((counts > 32).sum())}
        print(f"\n== routing {dist}", flush=True)
        for tier, caps in (("E2", args.e2_caps), ("E3", args.e3_caps)):
            for cap in [int(c) for c in caps.split(",")]:
                os.environ["EXL3_FAT_GROUPED"] = "1" if tier == "E3" else "0"
                _record_exl3_fat_resolution(layer)
                set_cap(layer, cap)
                torch.cuda.synchronize()
                mem0 = torch.cuda.memory_allocated()
                times = time_fn(lambda: apply_exl3_experts(x, ids, w, layer), iters=args.iters)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                fb = layer._exl3_last_fat_fallback
                med = statistics.median(times)
                case = {"tier": tier, "cap": cap, "routing": dist, "times_ms": times, "median_ms": med,
                        "tflops": gflop / med, "last_fat_fallback": fb, "mem_allocated_before": mem0,
                        "peak_allocated": peak, "diag_grouped_scratch_bytes": exl3_fat_diag()["grouped_scratch_bytes"],
                        "diag_fat_scratch_bytes": exl3_fat_diag()["fat_scratch_bytes"]}
                rec["cases"].append(case)
                print(f"{tier} cap={cap:4d} median={med:8.2f} ms  ({gflop/med:6.1f} TFLOPS) times={['%.1f' % t for t in times]} path={fb}", flush=True)
                with open(args.out, "w") as f:
                    json.dump(rec, f, indent=2)
    # summary: best E2 vs best E3 per skew
    rec["summary"] = []
    for skew in [float(s) for s in args.skews.split(",")]:
        cs = [c for c in rec["cases"] if c["routing"]["skew"] == skew]
        e2 = min((c for c in cs if c["tier"] == "E2"), key=lambda c: c["median_ms"])
        e3 = min((c for c in cs if c["tier"] == "E3"), key=lambda c: c["median_ms"])
        rec["summary"].append({"skew": skew, "best_e2": {"cap": e2["cap"], "ms": e2["median_ms"]},
                               "best_e3": {"cap": e3["cap"], "ms": e3["median_ms"]}, "speedup": e2["median_ms"] / e3["median_ms"]})
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print("SUMMARY", json.dumps(rec["summary"]), flush=True)
    return 0 if rec["parity"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
