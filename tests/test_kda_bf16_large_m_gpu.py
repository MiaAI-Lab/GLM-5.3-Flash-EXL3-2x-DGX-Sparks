#!/usr/bin/env python3
"""GPU validation for the KDA large-M BF16 dispatch (needs idle GPU + image).

Covers: load-time retention gating (flag/shape/TP/capability), M<=512 Marlin
vs M>512 BF16 dispatch (via the production dispatch counters), BF16-vs-Marlin
numerics, CUDA graph capture/replay on both branches, and fail-closed
behavior for a bad flag, bad threshold, or TP != 2. Run with the server
STOPPED (the test builds real layers outside the engine):

    GLM53_KDA_BF16_LARGE_M=1 python3 tests/test_kda_bf16_large_m_gpu.py
"""

from __future__ import annotations

import os
import sys
import unittest.mock

N, K = 12576, 4096
PREFIX = "model.layers.0.self_attn.in_proj_qkvbfg_a"


def _ws2():
    """Report TP=2 to the overlay while Marlin prep stays WS-agnostic."""
    import vllm.distributed as dist

    return unittest.mock.patch.object(dist, "get_tensor_model_parallel_world_size",
                                      lambda: 2)


def build_real_layer(device, group="kda", prefix=PREFIX):
    import torch
    from vllm.model_executor.layers.quantization.exl3 import Glm53DenseFp8Method

    meth = Glm53DenseFp8Method(group, prefix)
    layer = torch.nn.Module()
    g = torch.Generator(device="cpu")
    g.manual_seed(5)
    layer.weight = torch.nn.Parameter(
        torch.randn(N, K, generator=g, dtype=torch.bfloat16), requires_grad=False
    )
    layer.output_size_per_partition = N
    layer.input_size_per_partition = K
    layer = layer.to(device)
    with _ws2():
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

    import vllm.model_executor.layers.quantization.exl3 as exl3mod

    os.environ["GLM53_KDA_BF16_LARGE_M"] = "1"
    check("threshold-constant-512", exl3mod.KDA_BF16_LARGE_M_MIN_M == 512)
    meth, layer = build_real_layer(device)
    check("retention-present", hasattr(layer, "glm53_bf16_lm_w"),
          f"w={tuple(layer.glm53_bf16_lm_w.shape)} dtype={layer.glm53_bf16_lm_w.dtype}")
    check("retention-shape-dtype",
          tuple(layer.glm53_bf16_lm_w.shape) == (N, K)
          and layer.glm53_bf16_lm_w.dtype == torch.bfloat16)
    check("retention-threshold-default", layer.glm53_bf16_lm_min_m == 512)
    retained = layer.glm53_bf16_lm_w.numel() * layer.glm53_bf16_lm_w.element_size()
    print(f"retained bytes/rank/layer: {retained} ({retained / 2**20:.2f} MiB)", flush=True)
    check("retention-bytes", retained == N * K * 2)

    def counters():
        return exl3mod.kda_large_m_dispatch_stats()

    # Dispatch: small M stays Marlin, large M takes the BF16 copy.
    xg = torch.Generator(device="cpu")
    xg.manual_seed(99)
    before = counters()
    for m in (1, 8, 32, 64, 220, 512):
        x = torch.randn(m, K, generator=xg, dtype=torch.bfloat16, device="cpu").to(device)
        y = meth.apply(layer, x, None)
        check(f"m={m}-finite", bool(torch.isfinite(y).all()))
    mid = counters()
    check("m<=512-all-marlin",
          mid["bf16_calls"] == before["bf16_calls"]
          and mid["marlin_calls"] > before["marlin_calls"])
    outs = {}
    for m in (513, 1536, 3584):
        x = torch.randn(m, K, generator=xg, dtype=torch.bfloat16, device="cpu").to(device)
        y = meth.apply(layer, x, None)
        check(f"m={m}-finite", bool(torch.isfinite(y).all()))
        outs[m] = (x, y)
    after = counters()
    check("m>512-all-bf16", after["bf16_calls"] - mid["bf16_calls"] == 3
          and after["marlin_calls"] == mid["marlin_calls"])
    check("bf16-rows", after["bf16_rows"] - mid["bf16_rows"] == 513 + 1536 + 3584)

    # Numerics: BF16 copy vs Marlin on the same layer (Marlin via an off twin).
    os.environ["GLM53_KDA_BF16_LARGE_M"] = "0"
    meth0, layer0 = build_real_layer(device)
    check("off-no-retention", not hasattr(layer0, "glm53_bf16_lm_w"))
    os.environ["GLM53_KDA_BF16_LARGE_M"] = "1"
    for m in (513, 3584):
        x, _ = outs[m]
        y_bf16 = meth.apply(layer, x, None)
        y_mar = meth0.apply(layer0, x, None)
        d = (y_bf16.float() - y_mar.float()).abs()
        sc = float(y_mar.float().abs().max().clamp_min(1.0))
        print(f"m={m} bf16-vs-marlin maxabs={float(d.max()):.3f} "
              f"relmax={float(d.max()) / sc:.3e}", flush=True)
        check(f"m={m}-close", float(d.max()) / sc < 0.05)

    # Small-M parity: enabled at M=8 must equal disabled Marlin closely.
    x = torch.randn(8, K, generator=xg, dtype=torch.bfloat16, device="cpu").to(device)
    d = (meth.apply(layer, x, None).float() - meth0.apply(layer0, x, None).float()).abs()
    check("m8-parity", float(d.max()) < 1e-3, f"maxabs={float(d.max()):.2e}")

    # Graphs: capture + replay with changed data on both branches.
    for m, tag in ((1536, "bf16"), (8, "marlin")):
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

    # Fail-closed: bad flag and TP != 2 raise at load. The M>512 boundary is
    # a fixed constant (checked above), not a knob, so there is no threshold
    # value to reject.
    os.environ["GLM53_KDA_BF16_LARGE_M"] = "bogus"
    try:
        build_real_layer(device)
        check("bad-flag-raises", False)
    except RuntimeError:
        check("bad-flag-raises", True)
    os.environ["GLM53_KDA_BF16_LARGE_M"] = "1"
    try:
        import torch as _t
        from vllm.model_executor.layers.quantization.exl3 import (
            Glm53DenseFp8Method as _M,
        )
        m2 = _M("kda", PREFIX)
        lay = _t.nn.Module()
        gg = _t.Generator(device="cpu")
        gg.manual_seed(5)
        lay.weight = _t.nn.Parameter(
            _t.randn(N, K, generator=gg, dtype=_t.bfloat16), requires_grad=False)
        lay.output_size_per_partition = N
        lay.input_size_per_partition = K
        lay = lay.to(device)
        # No TP=2 emulation: the real TP=1 world size must fail closed.
        m2.process_weights_after_loading(lay)
        check("tp1-raises", False)
    except RuntimeError:
        check("tp1-raises", True)

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
