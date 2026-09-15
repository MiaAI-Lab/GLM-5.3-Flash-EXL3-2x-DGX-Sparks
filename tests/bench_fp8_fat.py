#!/usr/bin/env python3
"""Large-M FP8 candidate evaluation for TP-local KDA projection shapes.

Compares the shipped FP8-Marlin control against alternative FP8 GEMMs that
could reuse the SAME logical FP8 weights (plus a load-time-retained raw FP8
copy + per-channel scales) for large-M prefill:

  marlin      shipped Glm53DenseFp8Method.apply (control)
  scaled_mm   per-token activation quant + torch.ops.aten._scaled_mm
  cutlass     per-token activation quant + vllm cutlass_scaled_mm
  inductor    torch.compile(max-autotune) mixed bf16-act/fp8-weight linear

Only tensor layout/scale variants that pass the numeric gate (vs BF16
reference within FP8 tolerance) are benchmarked. Activation-quant cost is
timed separately AND included (production would pay it per call).

    python3 tests/bench_fp8_fat.py --out /tmp/fat.json

Needs the image python env (torch + vllm + overlay exl3 module) and an idle
GPU (server stopped).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

N_IN_PROJ, K_IN_PROJ = 12576, 4096
MS_FULL = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512,
           768, 1024, 220, 1536, 3136, 3584, 3683, 7168)


def qwen_quant_act(x: torch.Tensor) -> tuple:
    """Per-token FP8 e4m3 quant of BF16 activations (torch fallback; a fused
    kernel would only be faster — production cost is bounded above by this)."""
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True)
    scale = (amax / 448.0).clamp_min(1e-12)
    xq = (xf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return xq, scale


def time_fn(fn, warmup=20, iters=100):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ms = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        e.synchronize()
        ms.append(s.elapsed_time(e))
    ms.sort()
    n = len(ms)
    return {
        "median": ms[n // 2],
        "p10": ms[n // 10],
        "p90": ms[(9 * n) // 10],
        "min": ms[0],
    }


def err_stats(y: torch.Tensor, ref: torch.Tensor) -> dict:
    yf, rf = y.float(), ref.float()
    d = (yf - rf).abs()
    sc = float(rf.abs().max().clamp_min(1.0))
    cos = float(
        (yf.reshape(-1) @ rf.reshape(-1))
        / yf.norm(p=2)
        / rf.norm(p=2).clamp_min(1e-12)
    )
    mse = float(((yf - rf) ** 2).mean())
    denom = float((rf**2).mean())
    return {
        "maxabs": float(d.max()),
        "relmax": float(d.max()) / sc,
        "rel_rmse": (mse / denom) ** 0.5 if denom > 0 else 0.0,
        "cosim": cos,
    }


def _run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--n", type=int, default=N_IN_PROJ)
    ap.add_argument("--k", type=int, default=K_IN_PROJ)
    ap.add_argument("--ms", default=",".join(str(m) for m in MS_FULL))
    args = ap.parse_args()
    from vllm.model_executor.layers.quantization.exl3 import Glm53DenseFp8Method
    import vllm._custom_ops as vops

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    torch.manual_seed(11)
    n, k = args.n, args.k
    ms = [int(v) for v in args.ms.split(",") if v.strip()]

    # Logical weights: BF16 master + retained raw FP8 [N,K] + fp32 scales [N].
    w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cpu")
    wf = w_bf16.float()
    scales = (wf.abs().amax(dim=1).clamp_min(1e-12) / 448.0).to(torch.float32)
    w_fp8 = ((wf / scales[:, None]).clamp(-448.0, 448.0)).to(torch.float8_e4m3fn)
    del wf
    w_bf16_d, w_fp8_d = w_bf16.to(device), w_fp8.to(device)
    scales_d = scales.to(device)
    retain_bytes = w_fp8_d.numel() + scales_d.numel() * 4

    # Marlin control (consumes its own prepared copy, like serving).
    meth = Glm53DenseFp8Method("kda", "model.layers.0.self_attn.in_proj_qkvbfg_a")
    mlayer = torch.nn.Module()
    mlayer.weight = torch.nn.Parameter(w_bf16.clone(), requires_grad=False)
    mlayer.output_size_per_partition = n
    mlayer.input_size_per_partition = k
    mlayer = mlayer.to(device)
    meth.process_weights_after_loading(mlayer)

    rec = {"device": torch.cuda.get_device_name(0),
           "capability": list(torch.cuda.get_device_capability()),
           "n": n, "k": k, "retain_bytes_per_rank": retain_bytes,
           "cases": []}

    for m in ms:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        y_ref = torch.nn.functional.linear(x, w_bf16_d)
        y_marlin = meth.apply(mlayer, x, None)
        torch.cuda.synchronize()
        row = {"m": m,
               "marlin_ms": time_fn(lambda: meth.apply(mlayer, x, None),
                                    iters=args.iters),
               "marlin_err": err_stats(y_marlin, y_ref)}
        xq, sa = qwen_quant_act(x)
        row["actquant_ms"] = time_fn(lambda: qwen_quant_act(x), iters=args.iters)
        sb_row = scales_d.view(1, n)

        # torch native _scaled_mm over layout/scale variants. _scaled_mm wants
        # mat2 [K,N] with stride(0)==1, i.e. a col-major view: the retained
        # [N,K] row-major FP8 satisfies this via .t() with ZERO copy.
        sb_1d = scales_d
        for tag, b, sb in (("cm", w_fp8_d.t(), sb_row),
                           ("cm_s1d", w_fp8_d.t(), sb_1d),
                           ("t", w_fp8_d.t().contiguous(), sb_row)):
            try:
                y = torch.ops.aten._scaled_mm(
                    xq, b, sa, sb, None, None, torch.bfloat16, False)
                torch.cuda.synchronize()
                es = err_stats(y, y_ref)
                ok = es["rel_rmse"] < 0.05
            except Exception as exc:  # noqa: BLE001
                es, ok = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}, False
            key = f"scaled_mm_{tag}"
            row[key + "_err"] = es
            if ok:
                def _timed_scaled_mm(b=b, sb=sb):
                    xqq, saa = qwen_quant_act(x)
                    return torch.ops.aten._scaled_mm(
                        xqq, b, saa, sb, None, None, torch.bfloat16, False)

                row[key + "_ms"] = time_fn(_timed_scaled_mm, iters=args.iters)
                row[key + "_ms_noquant"] = time_fn(
                    lambda b=b: torch.ops.aten._scaled_mm(
                        xq, b, sa, sb, None, None, torch.bfloat16, False),
                    iters=args.iters)

        # vLLM cutlass_scaled_mm over the same variants.
        for tag, b, sb in (("nt", w_fp8_d, sb_row),
                           ("t", w_fp8_d.t().contiguous(), sb_row)):
            try:
                y = vops.cutlass_scaled_mm(xq, b, sa, sb, torch.bfloat16, None)
                torch.cuda.synchronize()
                es = err_stats(y, y_ref)
                ok = es["rel_rmse"] < 0.05
            except Exception as exc:  # noqa: BLE001
                es, ok = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}, False
            key = f"cutlass_{tag}"
            row[key + "_err"] = es
            if ok:
                row[key + "_ms"] = time_fn(
                    lambda b=b: vops.cutlass_scaled_mm(
                        *qwen_quant_act(x), b, sb, torch.bfloat16, None),
                    iters=args.iters)

        # Inductor mixed-precision (dequant fused-or-not, empirically).
        try:
            @torch.compile(mode="max-autotune", dynamic=True)
            def ind_fn(x_, w_, s_):
                return torch.nn.functional.linear(x_, (w_.float() * s_).to(torch.bfloat16))
            y = ind_fn(x, w_fp8_d, scales_d.view(n, 1))
            torch.cuda.synchronize()
            es = err_stats(y, y_ref)
            ok = es["rel_rmse"] < 0.05
        except Exception as exc:  # noqa: BLE001
            es, ok = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}, False
        row["inductor_err"] = es
        if ok:
            row["inductor_ms"] = time_fn(lambda: ind_fn(x, w_fp8_d, scales_d.view(n, 1)),
                                         iters=args.iters)

        # Return control: marlin again at the end.
        row["marlin_ms_ret"] = time_fn(lambda: meth.apply(mlayer, x, None),
                                       iters=max(10, args.iters // 10))
        rec["cases"].append(row)
        slim = {"m": m}
        for kk, vv in row.items():
            if kk.endswith("_ms") and isinstance(vv, dict):
                slim[kk] = round(vv["median"], 4)
            elif kk.endswith("_err") and isinstance(vv, dict) and "rel_rmse" in vv:
                slim[kk] = round(vv["rel_rmse"], 6)
        print(json.dumps(slim), flush=True)

    json.dump(rec, open(args.out, "w"))
    print("wrote", args.out)
    return 0


def main() -> int:
    """Entry point: run the bench inside real single-rank (TP=1) vLLM
    model-parallel state (required: production
    ``process_weights_after_loading`` calls
    ``get_tensor_model_parallel_world_size()``)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _vllm_tp1 import single_rank_model_parallel

    with single_rank_model_parallel():
        return _run()


if __name__ == "__main__":
    raise SystemExit(main())
