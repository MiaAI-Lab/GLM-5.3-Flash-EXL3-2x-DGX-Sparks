#!/usr/bin/env python3
"""Large-M KDA in_proj: BF16 path vs stock Marlin, one harness.

Two questions:

1. Numerics vs M: the shipped BF16 copy against stock Marlin (rel RMSE,
   maxabs, relmax, cosine, per-row mean/p99), with the M=63/64/65/66 boundary
   diagnostics and the large-M band, plus the `ideal` oracle (fp32 GEMM on the
   exact logical weight) that makes Marlin's own floor measurable.

2. Crossover timing: Marlin vs BF16 with warmup, median, p10/p90 and an A/B/A
   drift check, so the dispatch threshold stays justified by data.

The `logical` oracle is the retained BF16 copy cast to fp32 (fp32 GEMM on the
weight as retained), NOT the fp32-exact fp8-times-fp32-scale product: the
shipped method deliberately retains only the stored-scale copy, so the true
fp32-exact product is not available without re-reading the checkpoint. The
`marlin_vs_logical == bf16_vs_marlin` reading is still useful: it shows the
GEMM backend contributes negligibly next to the copy's own rounding.

Usage (in the image, on the head node, server STOPPED):
  python3 tests/bench_kda_bf16_large_m.py --out /out/large_m.json \
      --checkpoint <snapshot> --weights real --layer 0
  python3 tests/bench_kda_bf16_large_m.py --out /out/large_m_syn.json \
      --weights synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unittest.mock
from pathlib import Path

import torch

N_IN_PROJ, K_IN_PROJ = 12576, 4096
# Boundary diagnostics, the measured serving bands, and the crossover sweep.
DEFAULT_MS = (1, 8, 32, 63, 64, 65, 66, 96, 128, 192, 220, 256, 384, 512, 768,
              1024, 1536, 3136, 3584, 3683, 7168)


def time_fn(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    n = len(times)
    return {
        "median": times[n // 2], "p10": times[n // 10], "p90": times[(9 * n) // 10],
        "min": times[0], "max": times[-1], "n": n,
    }


def err_stats(y: torch.Tensor, ref: torch.Tensor) -> dict:
    yf, rf = y.float(), ref.float()
    diff = (yf - rf).abs()
    scale = float(rf.abs().max().clamp_min(1e-6))
    mse = float(((yf - rf) ** 2).mean())
    denom = float((rf ** 2).mean())
    cos = float(torch.nn.functional.cosine_similarity(
        yf.reshape(-1), rf.reshape(-1), dim=0))
    # Per-row relative error, the shape a downstream layer actually sees.
    row_err = (yf - rf).pow(2).mean(dim=1).sqrt()
    row_ref = rf.pow(2).mean(dim=1).sqrt().clamp_min(1e-12)
    row_rel = row_err / row_ref
    return {
        "rel_rmse": (mse / denom) ** 0.5 if denom > 0 else 0.0,
        "maxabs": float(diff.max()),
        "relmax": float(diff.max()) / scale,
        "cosine": cos,
        "row_rel_mean": float(row_rel.mean()),
        "row_rel_p99": (float(row_rel.quantile(0.99)) if row_rel.numel() > 1
                        else float(row_rel.max())),
    }


def build_real_in_proj(checkpoint: str, layer: int, tp_size: int, tp_rank: int) -> torch.Tensor:
    """Reconstruct the TP-local fused in_proj_qkvbfg_a weight from the shipped
    checkpoint (non-routed tensors are stored at native BF16 dtype).

    Mirrors vllm/models/glm5next/nvidia/kda.py: sizes
    [proj, proj, proj, heads, head_dim, head_dim] with shards 4,5 replicated.
    """
    from safetensors import safe_open

    ckpt = Path(checkpoint)
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.self_attn"
    names = {
        "q": f"{prefix}.q_proj.weight",
        "k": f"{prefix}.k_proj.weight",
        "v": f"{prefix}.v_proj.weight",
        "b": f"{prefix}.b_proj.weight",
        "f_a": f"{prefix}.f_a_proj.weight",
        "g_a": f"{prefix}.g_a_proj.weight",
    }
    parts = []
    cache: dict[str, object] = {}
    for key, name in names.items():
        if name not in index:
            raise KeyError(f"{name} not in checkpoint index")
        shard = index[name]
        if shard not in cache:
            # NOTE: the handle must stay open for get_tensor(); do not use it as
            # a context manager or the file closes before the read.
            cache[shard] = safe_open(str(ckpt / shard), framework="pt", device="cpu")
        tensor = cache[shard].get_tensor(name)
        if key in ("f_a", "g_a"):
            parts.append(tensor)
            continue
        rows = tensor.shape[0]
        if rows % tp_size:
            raise RuntimeError(f"{name}: rows {rows} not divisible by tp {tp_size}")
        per = rows // tp_size
        parts.append(tensor[tp_rank * per : (tp_rank + 1) * per])
    return torch.cat(parts, dim=0)


def build_method_layer(exl3mod, master: torch.Tensor, enabled: bool, n: int, k: int,
                       device):
    """Real Glm53DenseFp8Method on a fake layer, through the shipped code path."""
    import vllm.distributed as dist

    prev = os.environ.get("GLM53_KDA_BF16_LARGE_M")
    try:
        os.environ["GLM53_KDA_BF16_LARGE_M"] = "1" if enabled else "0"
        method = exl3mod.Glm53DenseFp8Method(
            "kda", "model.layers.0.self_attn.in_proj_qkvbfg_a")
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(master.clone(), requires_grad=False)
        layer.output_size_per_partition = n
        layer.input_size_per_partition = k
        layer = layer.to(device)
        started = time.perf_counter()
        # Report TP=2: the retention path is qualified for TP=2 only, and
        # Marlin prep itself never reads the world size.
        with unittest.mock.patch.object(
                dist, "get_tensor_model_parallel_world_size", lambda: 2):
            method.process_weights_after_loading(layer)
        load_ms = (time.perf_counter() - started) * 1e3
    finally:
        if prev is None:
            os.environ.pop("GLM53_KDA_BF16_LARGE_M", None)
        else:
            os.environ["GLM53_KDA_BF16_LARGE_M"] = prev
    return method, layer, load_ms


def _run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", choices=("synthetic", "real"), default="synthetic")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--ms", default=",".join(str(m) for m in DEFAULT_MS))
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--act-rms", type=float, default=0.2,
                    help="activation scale; 0.2 matches the measured in_proj rms")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    n, k = N_IN_PROJ, K_IN_PROJ
    ms = [int(v) for v in args.ms.split(",") if v.strip()]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    import vllm.model_executor.layers.quantization.exl3 as exl3mod
    from _vllm_tp1 import single_rank_model_parallel

    with single_rank_model_parallel():
        if args.weights == "real":
            if not args.checkpoint:
                print("--checkpoint is required with --weights real", file=sys.stderr)
                return 2
            master = build_real_in_proj(args.checkpoint, args.layer,
                                        args.tp_size, args.tp_rank)
        else:
            master = torch.randn(n, k, generator=gen, dtype=torch.bfloat16)
        master_d = master.to(device)

        meth_off, layer_off, load_off_ms = build_method_layer(
            exl3mod, master_d, False, n, k, device)
        meth_on, layer_on, load_on_ms = build_method_layer(
            exl3mod, master_d, True, n, k, device)

        if not hasattr(layer_on, "glm53_bf16_lm_w"):
            print("BF16 retention did not happen; is this SM121?", file=sys.stderr)
            return 3
        w_bf16 = layer_on.glm53_bf16_lm_w                 # the shipped copy
        # fp32 GEMM on the weight AS RETAINED (see module docstring: this is
        # not the fp32-exact fp8-times-fp32-scale product).
        logical = w_bf16.float()

        rec = {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "n": n, "k": k,
            "weights": args.weights, "layer": args.layer,
            "tp_size": args.tp_size, "tp_rank": args.tp_rank,
            "act_rms": args.act_rms,
            "bytes": {
                "retained_bytes": int(w_bf16.numel() * w_bf16.element_size()),
                "bytes_per_layer_rank_mib": w_bf16.numel() * w_bf16.element_size() / 2**20,
                "elements": int(w_bf16.numel()),
                "dtype": str(w_bf16.dtype),
                "threshold": int(layer_on.glm53_bf16_lm_min_m),
                "retained_attributes": sorted(
                    attr for attr in dir(layer_on) if attr.startswith("glm53_bf16_lm")),
            },
            "load_ms": {"off": load_off_ms, "on": load_on_ms},
            "cases": [],
        }
        print(f"BF16 retained bytes/layer-rank: {rec['bytes']['retained_bytes']} "
              f"({rec['bytes']['bytes_per_layer_rank_mib']:.2f} MiB) "
              f"dtype={rec['bytes']['dtype']} threshold={rec['bytes']['threshold']}")
        print(f"load ms off/on: {load_off_ms:.1f}/{load_on_ms:.1f}")

        for m in ms:
            x = torch.randn(m, k, generator=gen, dtype=torch.bfloat16).to(device)
            x = x * (args.act_rms / float(x.float().pow(2).mean().sqrt()))

            y_marlin = meth_off.apply(layer_off, x, None)
            y_bf16 = meth_on.apply(layer_on, x, None)
            xs = x.reshape(-1, k)
            y_logical = torch.nn.functional.linear(xs.float(), logical, None).to(torch.bfloat16)
            torch.cuda.synchronize()

            row = {
                "m": m,
                "marlin_ms": time_fn(lambda: meth_off.apply(layer_off, x, None), iters=args.iters),
                "bf16_ms": time_fn(lambda: meth_on.apply(layer_on, x, None), iters=args.iters),
                "marlin_drift_ms": time_fn(lambda: meth_off.apply(layer_off, x, None),
                                           warmup=2, iters=max(5, args.iters // 3)),
                "marlin_vs_logical": err_stats(y_marlin, y_logical),
                "bf16_vs_marlin": err_stats(y_bf16, y_marlin),
                "bf16_vs_logical": err_stats(y_bf16, y_logical),
            }
            for key in ("marlin_ms", "bf16_ms"):
                row[key] = {kk: row[key][kk] for kk in ("median", "p10", "p90", "min", "max")}
            marlin_med = row["marlin_ms"]["median"]
            row["bf16_over_marlin"] = row["bf16_ms"]["median"] / marlin_med
            row["marlin_drift_ratio"] = row["marlin_drift_ms"]["median"] / marlin_med
            rec["cases"].append(row)
            print(f"M={m:>5}: marlin {marlin_med:8.4f} bf16 {row['bf16_ms']['median']:8.4f} ms "
                  f"| bf16/marlin {row['bf16_over_marlin']:.2f}x "
                  f"| bf16-vs-marlin {row['bf16_vs_marlin']['rel_rmse']:.5f} "
                  f"marlin-vs-logical {row['marlin_vs_logical']['rel_rmse']:.5f}")

    with open(args.out, "w") as handle:
        json.dump(rec, handle, indent=1)
    print("wrote", args.out)
    return 0


def main() -> int:
    return _run()


if __name__ == "__main__":
    raise SystemExit(main())
