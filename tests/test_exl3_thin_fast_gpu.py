#!/usr/bin/env python3
"""GPU battery for the opt-in SM121 thin-decode path (one dispatch per run).

Run TWICE (server stopped), once per native dispatch::

    GLM53_EXL3_MOE_FAST=0 python3 tests/test_exl3_thin_fast_gpu.py --out /tmp/thin_fast0.pt
    GLM53_EXL3_MOE_FAST=1 python3 tests/test_exl3_thin_fast_gpu.py --out /tmp/thin_fast1.pt
    # stock repeat yardstick (ordinary atomic-ordering variation):
    GLM53_EXL3_MOE_FAST=0 python3 tests/test_exl3_thin_fast_gpu.py --out /tmp/thin_fast0b.pt
    python3 tests/compare_thin_fast.py /tmp/thin_fast0.pt /tmp/thin_fast0b.pt /tmp/thin_fast1.pt

Battery per SUH geometry (shared = alias path, independent = separate path):
  * rows {1,2,7,8,24,48,64,127,128} x routing {correlated, uniform, skewed}
  * repeat determinism (same case twice in-process)
  * non-default-stream execution
  * large-amplitude inputs (x32, finite)
  * CUDA graph capture + replay with changed inputs (routing AND data)
  * N_off=0 fallback geometry (hidden % 256 != inter % 256): must stay on the
    stock path and match across modes

Saves every case output; the comparator (CPU-only) does the cross-mode math.
`--smoke` runs a 2-case subset for bring-up.
"""

from __future__ import annotations

import argparse
import os
import sys

from bench_exl3_thin import N_EXP, TOPK, make_routing


HIDDEN = 4096
INTER = 1024
SUH_MODES = ("shared", "independent")
CASE_ROWS = (1, 2, 7, 8, 24, 48, 64, 127, 128)
ROUTINGS = ("correlated", "uniform", "skewed")


def build_layer(device, hidden: int, inter: int, shared_suh: bool, seed: int):
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
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        params_dtype=torch.float16,
    )
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    with torch.no_grad():
        layer.w13_trellis.copy_(
            torch.randint(-30000, 30000, tuple(layer.w13_trellis.shape),
                          dtype=torch.int16, generator=g)
        )
        layer.w2_trellis.copy_(
            torch.randint(-30000, 30000, tuple(layer.w2_trellis.shape),
                          dtype=torch.int16, generator=g)
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


def run_case(layer, x, ids, w):
    import torch
    from vllm.model_executor.layers.quantization.exl3 import apply_exl3_fused_moe

    out = apply_exl3_fused_moe(
        x, ids, w, layer, layer._exl3_inners, None, 10.0)
    torch.cuda.synchronize()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    import exllamav3_ext
    from vllm.model_executor.layers.quantization.exl3 import temp_rows_fused

    receipt = {
        "meta": {
            "fast_env": os.environ.get("GLM53_EXL3_MOE_FAST", "0"),
            "native_fast": bool(getattr(exllamav3_ext, "glm53_fast_moe_version", lambda: 0)()),
            "cap": temp_rows_fused(),
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
        },
        "cases": {},
    }

    rows = (8, 64) if args.smoke else CASE_ROWS
    routings = ("correlated", "uniform") if args.smoke else ROUTINGS
    def _randn(rows, cols, gen):
        return torch.randn(rows, cols, generator=gen,
                           dtype=torch.float16, device="cpu").to(device)

    xg = torch.Generator(device="cpu")
    xg.manual_seed(31337)

    layers = {}
    for suh in SUH_MODES:
        _m, layer = build_layer(device, HIDDEN, INTER, suh == "shared", seed=11)
        layers[suh] = layer
        receipt["meta"][f"aliased_{suh}"] = bool(
            layer._exl3_ptrs["up_suh"] is layer._exl3_ptrs["gate_suh"])
    _m, fb_layer = build_layer(device, HIDDEN, 1072, True, seed=12)
    layers["fallback"] = fb_layer

    for lname, layer in layers.items():
        hidden = HIDDEN
        lroutings = ("correlated",) if lname == "fallback" else routings
        lrows = (8, 64) if lname == "fallback" else rows
        for n in lrows:
            for mode in lroutings:
                key = f"{lname}/rows{n}/{mode}"
                ids, w = make_routing(n, mode, seed=n, device=device)
                with torch.no_grad():
                    x = _randn(n, hidden, xg)
                y1 = run_case(layer, x, ids, w)
                y2 = run_case(layer, x, ids, w)  # in-process repeat
                receipt["cases"][key] = y1.cpu()
                receipt["cases"][key + "/repeat"] = y2.cpu()
                assert torch.isfinite(y1).all(), key

    # Non-default stream (shared-SUH layer, 64 rows correlated).
    layer = layers["shared"]
    ids, w = make_routing(64, "correlated", seed=64, device=device)
    with torch.no_grad():
        x = _randn(64, HIDDEN, xg)
    y_default = run_case(layer, x, ids, w)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y_stream = run_case(layer, x, ids, w)
    torch.cuda.current_stream().wait_stream(s)
    receipt["cases"]["shared/stream64"] = y_stream.cpu()
    receipt["cases"]["shared/default64"] = y_default.cpu()

    # Large amplitude inputs (x32, still finite in fp16).
    with torch.no_grad():
        x_big = _randn(24, HIDDEN, xg) * 32.0
    ids, w = make_routing(24, "uniform", seed=24, device=device)
    y_big = run_case(layer, x_big, ids, w)
    assert torch.isfinite(y_big).all()
    receipt["cases"]["shared/bigamp24"] = y_big.cpu()

    # CUDA graph capture + replay with changed inputs (data AND routing).
    n = 48
    static_x = torch.randn(n, HIDDEN, dtype=torch.float16, device=device)
    static_ids, static_w = make_routing(n, "correlated", seed=48, device=device)
    static_ids = static_ids.clone()
    static_w = static_w.clone()
    layer.expert_map = torch.arange(N_EXP, dtype=torch.long, device="cpu")
    from vllm.model_executor.layers.quantization.exl3 import apply_exl3_experts

    # One eager apply through the shipped entry pins expert_map to CUDA;
    # capture forbids the CPU->GPU copy.
    y_eager = apply_exl3_experts(static_x, static_ids, static_w, layer, fused=True)
    assert layer.expert_map.device.type == "cuda"
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            apply_exl3_experts(static_x, static_ids, static_w, layer, fused=True)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_graph = apply_exl3_experts(
            static_x, static_ids, static_w, layer, fused=True)
    g.replay()
    torch.cuda.synchronize()
    receipt["cases"]["shared/graph48/eager"] = y_eager.cpu()
    receipt["cases"]["shared/graph48/replay1"] = y_graph.cpu().clone()
    # Changed inputs between replays.
    with torch.no_grad():
        static_x.copy_(_randn(n, HIDDEN, xg))
    ids2, w2 = make_routing(n, "uniform", seed=481, device=device)
    static_ids.copy_(ids2)
    static_w.copy_(w2)
    y_eager2 = apply_exl3_experts(static_x, static_ids, static_w, layer, fused=True)
    g.replay()
    torch.cuda.synchronize()
    receipt["cases"]["shared/graph48/eager2"] = y_eager2.cpu()
    receipt["cases"]["shared/graph48/replay2"] = y_graph.cpu().clone()

    torch.save(receipt, args.out)
    print(f"saved {len(receipt['cases'])} case tensors to {args.out}", flush=True)
    print(f"meta={receipt['meta']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
