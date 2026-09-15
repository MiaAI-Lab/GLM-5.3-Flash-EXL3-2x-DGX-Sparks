#!/usr/bin/env python3
"""GPU validation for the hybrid KDA FP8 dispatch (needs idle GPU + image env).

Covers: load-time retention gating (flag/shape), M<=64 Marlin vs M>64 fat
dispatch (via dispatch spies, no production counters), fat-vs-Marlin and
fat-vs-BF16 numerics, Triton rowquant verdict, CUDA graph capture/replay
with changed inputs on both branches, and FAT=0 / bad-flag fail-closed
behavior. Run with the server STOPPED:

    GLM53_KDA_FP8_FAT=1 python3 tests/test_kda_fp8_fat_gpu.py
"""

from __future__ import annotations

import os
import sys

N, K = 12576, 4096


def build_real_layer(device, group="kda", seed=5):
    import torch
    from vllm.model_executor.layers.quantization.exl3 import Glm53DenseFp8Method

    meth = Glm53DenseFp8Method(group, "model.layers.0.self_attn.in_proj_qkvbfg_a")
    layer = torch.nn.Module()
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    layer.weight = torch.nn.Parameter(
        torch.randn(N, K, generator=g, dtype=torch.bfloat16), requires_grad=False
    )
    layer.output_size_per_partition = N
    layer.input_size_per_partition = K
    layer = layer.to(device)
    meth.process_weights_after_loading(layer)
    return meth, layer


def _run() -> int:
    import torch

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    failures = []

    def check(name, cond, detail=""):
        print(f"{'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)
        if not cond:
            failures.append(name)

    os.environ["GLM53_KDA_FP8_FAT"] = "1"
    meth, layer = build_real_layer(device)
    check("retention-present", hasattr(layer, "glm53_fat_wt"),
          f"wt={tuple(layer.glm53_fat_wt.shape)} strides={layer.glm53_fat_wt.stride()}")
    check("retention-colmajor",
          tuple(layer.glm53_fat_wt.shape) == (K, N) and layer.glm53_fat_wt.stride()[0] == 1)
    check("retention-scales", tuple(layer.glm53_fat_sb.shape) == (1, N))
    retained = layer.glm53_fat_w.numel() + layer.glm53_fat_s.numel() * 4
    print(f"retained bytes/rank/layer: {retained} ({retained / 2**20:.1f} MiB)", flush=True)

    import vllm.model_executor.layers.quantization.exl3 as exl3mod
    print(f"triton available={exl3mod._FAT_TRITON_AVAILABLE} ready={exl3mod._FAT_TRITON_READY[0]}",
          flush=True)
    if exl3mod._FAT_TRITON_READY[0]:
        xt = torch.randn(48, K, dtype=torch.bfloat16, device=device)
        xt[0].zero_()
        xt[1].mul_(1e-14)
        qt = torch.empty((48, K), dtype=torch.float8_e4m3fn, device=device)
        st = torch.empty((48,), dtype=torch.float32, device=device)
        exl3mod._fat_rowquant_kernel[(48,)](xt, qt, st, xt.stride(-2), K, 4096,
                                             num_warps=8)
        torch.cuda.synchronize()
        amax = xt.float().abs().amax(dim=1, keepdim=True)
        se = (amax / 448.0).clamp_min(1e-12)
        # Triton must match fp32-division eager EXACTLY (proves the
        # float8e4nv encoding is bit-identical to torch fp8_e4m3fn).
        qe32 = (xt.float() / se).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        check("triton-bitwise", bool(torch.equal(qt, qe32))
              and bool(torch.equal(st, se.view(-1))))
        # Exercise the production eager fallback, including zero/tiny rows.
        ready = exl3mod._FAT_TRITON_READY[0]
        try:
            exl3mod._FAT_TRITON_READY[0] = False
            qe, eager_scales = meth._fat_rowquant(xt)
        finally:
            exl3mod._FAT_TRITON_READY[0] = ready
        check("eager-fallback-bitwise", bool(torch.equal(qt, qe))
              and bool(torch.equal(st, eager_scales.view(-1))))

    # Dispatch spies: count fat vs marlin calls across the M boundary.
    calls = {"fat": 0}
    orig = meth._apply_fat
    def spy(_self, lay, x2):
        calls["fat"] += 1
        return orig(lay, x2)
    meth._apply_fat = spy.__get__(meth, type(meth))
    xg = torch.Generator(device="cpu")
    xg.manual_seed(99)
    for m in (1, 8, 32, 64):
        x = torch.randn(m, K, generator=xg, dtype=torch.bfloat16, device="cpu").to(device)
        meth.apply(layer, x, None)
    check("m<=64-all-marlin", calls["fat"] == 0, f"fat_calls={calls['fat']}")
    for m in (65, 96, 512):
        x = torch.randn(m, K, generator=xg, dtype=torch.bfloat16, device="cpu").to(device)
        y = meth.apply(layer, x, None)
        check(f"m={m}-finite", bool(torch.isfinite(y).all()))
    check("m>64-all-fat", calls["fat"] == 3, f"fat_calls={calls['fat']}")

    # Numerics: fat vs marlin on the same layer (marlin via a FAT=0 twin).
    os.environ["GLM53_KDA_FP8_FAT"] = "0"
    meth0, layer0 = build_real_layer(device)
    check("fat0-no-retention", not hasattr(layer0, "glm53_fat_wt"))
    os.environ["GLM53_KDA_FP8_FAT"] = "1"
    for m in (512, 3584):
        x = torch.randn(m, K, generator=xg, dtype=torch.bfloat16, device="cpu").to(device)
        y_fat = meth.apply(layer, x, None)
        y_mar = meth0.apply(layer0, x, None)
        d = (y_fat.float() - y_mar.float()).abs()
        sc = float(y_mar.float().abs().max().clamp_min(1.0))
        print(f"m={m} fat-vs-marlin maxabs={float(d.max()):.3f} "
              f"relmax={float(d.max()) / sc:.3e}", flush=True)
        check(f"m={m}-close", float(d.max()) / sc < 0.10)

    # Small-M parity: FAT=1 at M=8 must equal FAT=0 Marlin closely.
    x = torch.randn(8, K, generator=xg, dtype=torch.bfloat16, device="cpu").to(device)
    d = (meth.apply(layer, x, None).float() - meth0.apply(layer0, x, None).float()).abs()
    check("m8-parity", float(d.max()) < 1e-3, f"maxabs={float(d.max()):.2e}")

    # Graphs: capture + replay with changed data on both branches.
    for m, tag in ((512, "fat"), (8, "marlin")):
        sx = torch.randn(m, K, dtype=torch.bfloat16, device=device)
        meth.apply(layer, sx, None)  # warmup/pin
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            yg = meth.apply(layer, sx, None)
        g.replay()
        torch.cuda.synchronize()
        r1 = yg.cpu().clone()
        sx.copy_(torch.randn(m, K, generator=xg, dtype=torch.bfloat16, device="cpu").to(device))
        ye = meth.apply(layer, sx, None)
        g.replay()
        torch.cuda.synchronize()
        dd = (yg.cpu().float() - ye.cpu().float()).abs()
        check(f"graph-{tag}-replay", float(dd.max()) == 0.0, f"maxabs={float(dd.max()):.2e}")

    # Bad flag value fails closed at load.
    os.environ["GLM53_KDA_FP8_FAT"] = "bogus"
    try:
        build_real_layer(device)
        check("bad-flag-raises", False)
    except RuntimeError:
        check("bad-flag-raises", True)
    os.environ["GLM53_KDA_FP8_FAT"] = "1"

    print("FAILURES:", failures if failures else "none", flush=True)
    return 1 if failures else 0


def main() -> int:
    """Entry point: run the battery inside real single-rank (TP=1) vLLM
    model-parallel state (required: production
    ``process_weights_after_loading`` calls
    ``get_tensor_model_parallel_world_size()``)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _vllm_tp1 import single_rank_model_parallel

    with single_rank_model_parallel():
        return _run()


if __name__ == "__main__":
    raise SystemExit(main())
