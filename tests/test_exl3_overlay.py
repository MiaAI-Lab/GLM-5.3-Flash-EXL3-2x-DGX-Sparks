#!/usr/bin/env python3
"""Drive the *shipped* EXL3 path. Fail if the overlay only registered a name.

This file is copied into the image at /opt/glm53/test_exl3_overlay.py and is
the image-build / post-build self-check. It imports vLLM's registered method
and LinearEXL3 — it does not reimplement the GEMM.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _check_quant_registry() -> None:
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config

    assert "exl3" in QUANTIZATION_METHODS, QUANTIZATION_METHODS
    cfg_cls = get_quantization_config("exl3")
    assert cfg_cls is Exl3Config, cfg_cls
    cfg = cfg_cls.from_config(
        {
            "quant_method": "exl3",
            "bits": 4,
            "codebook": "mcg",
            "scope": "glm53_routed_experts_only",
        }
    )
    assert cfg.get_name() == "exl3"
    assert cfg.override_quantization_method({"quant_method": "exl3"}, None) == "exl3"
    print("exl3 registry OK", flush=True)


def _check_tp_shard() -> None:
    import torch
    from vllm.model_executor.layers.quantization.exl3 import (
        shard_exl3_col,
        shard_exl3_row,
    )

    trellis = torch.arange(2 * 4 * 16 * 64, dtype=torch.int16).reshape(16, 8, 64)
    col0 = shard_exl3_col(trellis, "trellis", tp_rank=0, tp_size=2)
    col1 = shard_exl3_col(trellis, "trellis", tp_rank=1, tp_size=2)
    assert col0.shape == (16, 4, 64)
    assert col1.shape == (16, 4, 64)
    assert not torch.equal(col0, col1)
    assert torch.equal(torch.cat([col0, col1], dim=1), trellis)

    suh = torch.arange(32, dtype=torch.float16)
    row0 = shard_exl3_row(suh, "suh", tp_rank=0, tp_size=2)
    row1 = shard_exl3_row(suh, "suh", tp_rank=1, tp_size=2)
    assert row0.tolist() == list(range(16))
    assert row1.tolist() == list(range(16, 32))
    svh = torch.arange(8, dtype=torch.float16)
    assert torch.equal(shard_exl3_col(svh, "suh", 0, 2), svh)
    print("exl3 TP shard rules OK (gate/up col, down row)", flush=True)


def _check_ext_arch() -> None:
    import exllamav3_ext

    assert hasattr(exllamav3_ext, "exl3_moe"), dir(exllamav3_ext)
    assert hasattr(exllamav3_ext, "exl3_moe_max_concurrency")
    print("exllamav3_ext.exl3_moe present", flush=True)
    so = exllamav3_ext.__file__
    dump = subprocess.check_output(["cuobjdump", "-lelf", so], text=True, stderr=subprocess.STDOUT)
    arches = {
        line.strip().split()[-1]
        for line in dump.splitlines()
        if "sm_" in line or "gencode" in line.lower()
    }
    joined = dump.lower()
    has_121 = "sm_121" in joined or "compute_121" in joined
    has_120_only = ("sm_120" in joined or "compute_120" in joined) and not has_121
    if not has_121:
        raise AssertionError(
            f"exllamav3_ext is not an SM121 cubin ({so}):\n{dump[-2000:]}"
        )
    if has_120_only:
        raise AssertionError("exllamav3_ext is SM120-only; SM121 native kernels required")
    print(f"exllamav3_ext arch OK {so} arches={sorted(arches) or 'see cuobjdump'}", flush=True)


def _check_e2_diag_static() -> None:
    """E2 tier resolution and the diag schema are machine-checkable."""
    from vllm.model_executor.layers.quantization.exl3 import (
        EXL3_FAT_DIAG_KEYS,
        EXL3_FAT_DIAG_SCHEMA,
        configured_fat_tier,
        exl3_fat_diag,
        exl3_fat_symbols,
        resolve_exl3_fat_tier,
    )

    diag = exl3_fat_diag()
    assert diag["schema"] == EXL3_FAT_DIAG_SCHEMA == 1, diag["schema"]
    assert tuple(sorted(diag)) == tuple(sorted(EXL3_FAT_DIAG_KEYS)), (
        set(diag) ^ set(EXL3_FAT_DIAG_KEYS)
    )

    keys = ("EXL3_FAT_SORTED", "EXL3_FAT_BATCHED", "EXL3_FAT_KERNEL")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        assert configured_fat_tier() == "legacy"
        assert resolve_exl3_fat_tier(True) == ("legacy", "none_requested")

        os.environ["EXL3_FAT_SORTED"] = "1"
        assert configured_fat_tier() == "sorted"
        assert resolve_exl3_fat_tier(False) == ("sorted", "sorted_ok")

        os.environ["EXL3_FAT_SORTED"] = "0"
        os.environ["EXL3_FAT_BATCHED"] = "1"
        assert configured_fat_tier() == "batched"
        # No shared SUH → the stacked gate+up GEMM would be wrong; sorted is
        # the legitimate cap, and the reason must say so.
        assert resolve_exl3_fat_tier(False) == ("sorted", "shared_suh_absent")
        assert resolve_exl3_fat_tier(True, (True, True, True)) == (
            "batched",
            "batched_ok",
        )

        os.environ["EXL3_FAT_BATCHED"] = "0"
        os.environ["EXL3_FAT_KERNEL"] = "1"
        assert configured_fat_tier() == "kernel"
        assert resolve_exl3_fat_tier(False) == ("sorted", "shared_suh_absent")
        # The checkpoint cap precedes the image check: without shared SUH the
        # kernel would not run, so missing fat symbols must not raise.
        assert resolve_exl3_fat_tier(False, (True, False, False)) == (
            "sorted",
            "shared_suh_absent",
        )
        # An explicit kernel request fails closed when the symbols are absent
        # instead of silently running (and reporting) a lower tier.
        for symbols in ((True, False, True), (True, True, False)):
            try:
                resolve_exl3_fat_tier(True, symbols)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"kernel tier must fail closed: {symbols}")
        symbols = exl3_fat_symbols()
        if symbols[1] and symbols[2]:
            assert resolve_exl3_fat_tier(True) == ("kernel", "kernel_ok")
        else:
            try:
                resolve_exl3_fat_tier(True)
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    "this image lacks the fat kernel; the live resolve must fail closed"
                )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("exl3 E2 diag schema + tier resolution OK", flush=True)



def _check_gpu_gemm() -> None:
    import torch
    from vllm.model_executor.layers.quantization.exl3 import (
        MCG_MARKER_SIGNED_INT32,
        execute_exl3_linear,
        load_linear_exl3_cls,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for EXL3 GEMM self-check")
    device = torch.device("cuda:0")
    # One 16×16-tile K4 MCG matrix (256×256). Not a mock of LinearEXL3:
    # execute_exl3_linear is the shipped expert GEMM entry.
    in_f, out_f, bits = 256, 256, 4
    trellis = torch.zeros((in_f // 16, out_f // 16, bits * 16), dtype=torch.int16, device=device)
    suh = torch.ones(in_f, dtype=torch.float16, device=device)
    svh = torch.ones(out_f, dtype=torch.float16, device=device)
    mcg = torch.tensor([MCG_MARKER_SIGNED_INT32], dtype=torch.int32, device=device)
    x = torch.randn(4, in_f, dtype=torch.float16, device=device)
    cls = load_linear_exl3_cls()
    assert cls.__name__ == "LinearEXL3", cls
    y = execute_exl3_linear(x, trellis, suh, svh, mcg, out_dtype=torch.float32)
    assert y.shape == (4, out_f), y.shape
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all(), y
    # Persistent BF16 reconstruct of a *layer* of 288 experts would be tens of
    # GiB; this path only materializes the working GEMM tile inside LinearEXL3.
    print(
        f"exl3 GPU GEMM OK LinearEXL3 y={tuple(y.shape)} "
        f"finite mean={float(y.mean()):.4f}",
        flush=True,
    )
    _check_fat_kernel(device)
    _check_fused_vs_loop(device)
    _check_fused_fat_and_row_tile(device)
    _check_mixed_thin_fat(device)
    _check_e2_diag(device)
    _check_apply_expert_map(device)
    _check_fused_cudagraph(device)



def _check_fat_kernel(device) -> None:
    """Compare E2 direct and scatter epilogues with LinearEXL3 reconstruction."""
    import exllamav3_ext
    import torch
    from vllm.model_executor.layers.quantization.exl3 import (
        MCG_MARKER_SIGNED_INT32,
        execute_exl3_linear,
    )

    if not hasattr(exllamav3_ext, "exl3_fat_gemm"):
        print("exl3 E2 fat kernel absent (E1 image)", flush=True)
        return
    assert hasattr(exllamav3_ext, "exl3_fat_gemm_scatter")

    rows = 145
    in_f = out_f = 256
    generator = torch.Generator(device="cpu")
    generator.manual_seed(41)
    trellis = torch.randint(
        -30000,
        30000,
        (in_f // 16, out_f // 16, 4 * 16),
        dtype=torch.int16,
        generator=generator,
    ).to(device)
    suh = torch.where(
        torch.rand(in_f, generator=generator) > 0.5,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ).half().to(device)
    svh = torch.where(
        torch.rand(out_f, generator=generator) > 0.5,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ).half().to(device)
    mcg = torch.tensor(
        [MCG_MARKER_SIGNED_INT32], dtype=torch.int32, device=device
    )
    x = torch.randn(rows, in_f, dtype=torch.float16, device=device)
    reference = execute_exl3_linear(
        x, trellis, suh, svh, mcg, out_dtype=torch.float32
    )
    xh = torch.empty_like(x)
    exllamav3_ext.had_r_128(x, xh, suh, None, 1.0)
    direct = torch.empty(rows, out_f, dtype=torch.float32, device=device)
    exllamav3_ext.exl3_fat_gemm(
        xh, trellis, direct, svh, 4, True, False
    )

    bound = max(
        0.15, 0.08 * float(reference.float().abs().max().clamp_min(1.0))
    )
    direct_err = float((reference - direct).abs().max())
    assert torch.isfinite(direct).all()
    assert direct_err < bound, (
        f"E2 direct vs reconstruct maxabs={direct_err} bound={bound}"
    )

    token_idx = torch.randperm(rows + 17, device=device)[:rows].contiguous()
    route_weight = torch.rand(rows, dtype=torch.float16, device=device)
    expected = torch.zeros(
        rows + 17, out_f, dtype=torch.float32, device=device
    )
    expected.index_add_(
        0, token_idx, reference * route_weight.float().unsqueeze(-1)
    )
    scattered = torch.zeros_like(expected)
    exllamav3_ext.exl3_fat_gemm_scatter(
        xh,
        trellis,
        scattered,
        svh,
        token_idx,
        route_weight,
        4,
        True,
        False,
    )
    scatter_err = float((expected - scattered).abs().max())
    assert torch.isfinite(scattered).all()
    assert scatter_err < bound, (
        f"E2 scatter vs reconstruct maxabs={scatter_err} bound={bound}"
    )
    print(
        f"exl3 E2 direct/scatter parity OK rows={rows} "
        f"direct={direct_err:.5f} scatter={scatter_err:.5f} bound={bound:.5f}",
        flush=True,
    )


def _tiny_layer(device, n_exp: int = 3, hidden: int = 256, inter: int = 256):
    import types

    import torch
    from vllm.model_executor.layers.quantization.exl3 import (
        MCG_MARKER_SIGNED_INT32,
        Exl3Config,
        Exl3MoEMethod,
    )

    moe = types.SimpleNamespace(swiglu_limit=10.0)
    method = Exl3MoEMethod(moe, Exl3Config())
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        num_experts=n_exp,
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        params_dtype=torch.float16,
    )
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    with torch.no_grad():
        layer.w13_trellis.copy_(
            torch.randint(-30000, 30000, tuple(layer.w13_trellis.shape), dtype=torch.int16, generator=g)
        )
        layer.w2_trellis.copy_(
            torch.randint(-30000, 30000, tuple(layer.w2_trellis.shape), dtype=torch.int16, generator=g)
        )
        layer.w13_suh.copy_(torch.randn(tuple(layer.w13_suh.shape), generator=g).half())
        layer.w13_svh.copy_(torch.randn(tuple(layer.w13_svh.shape), generator=g).half())
        layer.w2_suh.copy_(torch.randn(tuple(layer.w2_suh.shape), generator=g).half())
        layer.w2_svh.copy_(torch.randn(tuple(layer.w2_svh.shape), generator=g).half())
        layer.w13_suh[:, 1].copy_(layer.w13_suh[:, 0])
        layer.w13_mcg.fill_(MCG_MARKER_SIGNED_INT32)
        layer.w2_mcg.fill_(MCG_MARKER_SIGNED_INT32)
    layer = layer.to(device)
    method.process_weights_after_loading(layer)
    return method, layer


def _check_fused_vs_loop(device) -> None:
    import torch
    from vllm.model_executor.layers.quantization.exl3 import apply_exl3_experts

    method, layer = _tiny_layer(device)
    del method
    x = torch.randn(2, 256, dtype=torch.float16, device=device)
    ids = torch.tensor([[0, 2], [0, 1]], dtype=torch.long, device=device)
    w = torch.tensor([[0.6, 0.4], [0.5, 0.5]], dtype=torch.float16, device=device)
    y_loop = apply_exl3_experts(x, ids, w, layer, fused=False)
    y_fused = apply_exl3_experts(x, ids, w, layer, fused=True)
    assert layer._exl3_last_apply == "fused", layer._exl3_last_apply
    assert y_loop.shape == y_fused.shape == (2, 256)
    assert torch.isfinite(y_loop).all() and torch.isfinite(y_fused).all()
    err = (y_loop.float() - y_fused.float()).abs()
    scale = float(y_loop.float().abs().mean().clamp_min(1e-3))
    max_err = float(err.max())
    # fp16 trellis GEMM noise, not bit-identical
    bound = max(0.15, 0.08 * float(y_loop.float().abs().max().clamp_min(1.0)))
    assert max_err < bound, f"fused vs loop maxabs={max_err} bound={bound} mean_scale={scale}"
    print(
        f"exl3 fused vs LinearEXL3 loop OK maxabs={max_err:.5f} bound={bound:.5f}",
        flush=True,
    )


def _check_fused_fat_and_row_tile(device) -> None:
    """T > temp rows: isolated fat tiers and row tiles match the full loop."""
    import os

    import torch
    from vllm.model_executor.layers.quantization.exl3 import (
        apply_exl3_experts,
        reset_exl3_fat_expert_stats,
    )

    prev_tile = os.environ.get("EXL3_MOE_ROW_TILE")
    prev_sorted = os.environ.get("EXL3_FAT_SORTED")
    prev_batched = os.environ.get("EXL3_FAT_BATCHED")
    prev_kernel = os.environ.get("EXL3_FAT_KERNEL")
    prev_log = os.environ.get("EXL3_FAT_EXPERT_LOG")
    prev_rows = os.environ.get("EXL3_TEMP_ROWS_FUSED")
    os.environ["EXL3_FAT_EXPERT_LOG"] = "1"
    os.environ["EXL3_TEMP_ROWS_FUSED"] = "128"
    try:
        method, layer = _tiny_layer(device)
        del method
        tokens = 128 + 32
        x = torch.randn(tokens, 256, dtype=torch.float16, device=device)
        # Both routed experts exceed the 128-row fused cap.
        ids = torch.zeros(tokens, 2, dtype=torch.long, device=device)
        ids[:, 1] = 1
        w = torch.full((tokens, 2), 0.5, dtype=torch.float16, device=device)
        os.environ["EXL3_MOE_ROW_TILE"] = "0"
        reset_exl3_fat_expert_stats()
        y_loop = apply_exl3_experts(x, ids, w, layer, fused=False)

        os.environ["EXL3_FAT_SORTED"] = "0"
        os.environ["EXL3_FAT_BATCHED"] = "0"
        os.environ["EXL3_FAT_KERNEL"] = "0"
        y_legacy = apply_exl3_experts(x, ids, w, layer, fused=True)
        assert layer._exl3_last_fat_fallback == "legacy"

        os.environ["EXL3_FAT_SORTED"] = "1"
        y_sorted = apply_exl3_experts(x, ids, w, layer, fused=True)
        assert layer._exl3_last_fat_fallback == "sorted"

        os.environ["EXL3_FAT_SORTED"] = "0"
        os.environ["EXL3_FAT_BATCHED"] = "1"
        y_batched = apply_exl3_experts(x, ids, w, layer, fused=True)
        assert layer._exl3_last_fat_fallback == "batched"

        assert (
            torch.isfinite(y_loop).all()
            and torch.isfinite(y_legacy).all()
            and torch.isfinite(y_sorted).all()
            and torch.isfinite(y_batched).all()
        )
        bound = max(0.15, 0.08 * float(y_loop.float().abs().max().clamp_min(1.0)))
        err_legacy = float((y_loop.float() - y_legacy.float()).abs().max())
        err_sorted = float((y_loop.float() - y_sorted.float()).abs().max())
        err_batched = float((y_loop.float() - y_batched.float()).abs().max())
        assert err_legacy < bound, (
            f"legacy fat fallback vs loop maxabs={err_legacy} bound={bound}"
        )
        assert err_sorted < bound, (
            f"sorted fat fallback vs loop maxabs={err_sorted} bound={bound}"
        )
        assert err_batched < bound, (
            f"batched fat fallback vs loop maxabs={err_batched} bound={bound}"
        )
        import exllamav3_ext

        err_kernel = None
        if hasattr(exllamav3_ext, "exl3_fat_gemm"):
            os.environ["EXL3_FAT_BATCHED"] = "0"
            os.environ["EXL3_FAT_KERNEL"] = "1"
            y_kernel = apply_exl3_experts(x, ids, w, layer, fused=True)
            assert layer._exl3_last_fat_fallback == "kernel"
            assert torch.isfinite(y_kernel).all()
            err_kernel = float(
                (y_loop.float() - y_kernel.float()).abs().max()
            )
            assert err_kernel < bound, (
                f"kernel fat fallback vs loop maxabs={err_kernel} bound={bound}"
            )
            os.environ["EXL3_FAT_KERNEL"] = "0"

        os.environ["EXL3_MOE_ROW_TILE"] = "1"
        y_tile = apply_exl3_experts(x, ids, w, layer, fused=True)
        assert torch.isfinite(y_tile).all()
        err_tile = float((y_loop.float() - y_tile.float()).abs().max())
        assert err_tile < bound, f"row-tile vs loop maxabs={err_tile} bound={bound}"
    finally:
        if prev_tile is None:
            os.environ.pop("EXL3_MOE_ROW_TILE", None)
        else:
            os.environ["EXL3_MOE_ROW_TILE"] = prev_tile
        if prev_sorted is None:
            os.environ.pop("EXL3_FAT_SORTED", None)
        else:
            os.environ["EXL3_FAT_SORTED"] = prev_sorted
        if prev_batched is None:
            os.environ.pop("EXL3_FAT_BATCHED", None)
        else:
            os.environ["EXL3_FAT_BATCHED"] = prev_batched
        if prev_kernel is None:
            os.environ.pop("EXL3_FAT_KERNEL", None)
        else:
            os.environ["EXL3_FAT_KERNEL"] = prev_kernel
        if prev_log is None:
            os.environ.pop("EXL3_FAT_EXPERT_LOG", None)
        else:
            os.environ["EXL3_FAT_EXPERT_LOG"] = prev_log
        if prev_rows is None:
            os.environ.pop("EXL3_TEMP_ROWS_FUSED", None)
        else:
            os.environ["EXL3_TEMP_ROWS_FUSED"] = prev_rows
    print(
        "exl3 legacy/sorted/batched/kernel fat fallback + row-tile vs loop OK "
        f"T={tokens} legacy={err_legacy:.5f} sorted={err_sorted:.5f} "
        f"batched={err_batched:.5f} kernel={err_kernel} "
        f"tile={err_tile:.5f} bound={bound:.5f}",
        flush=True,
    )


def _check_mixed_thin_fat(device) -> None:
    """One oversized expert plus fused thin experts must compose exactly once."""
    import torch
    from vllm.model_executor.layers.quantization.exl3 import apply_exl3_experts

    keys = (
        "EXL3_MOE_ROW_TILE",
        "EXL3_FAT_SORTED",
        "EXL3_FAT_BATCHED",
        "EXL3_FAT_KERNEL",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        _method, layer = _tiny_layer(device)
        tokens = 200
        x = torch.randn(tokens, 256, dtype=torch.float16, device=device)
        ids = torch.empty(tokens, 2, dtype=torch.long, device=device)
        ids[:, 0] = 0
        ids[:100, 1] = 1
        ids[100:, 1] = 2
        weights = torch.full(
            (tokens, 2), 0.5, dtype=torch.float16, device=device
        )

        y_loop = apply_exl3_experts(x, ids, weights, layer, fused=False)
        os.environ["EXL3_MOE_ROW_TILE"] = "0"
        os.environ["EXL3_FAT_SORTED"] = "1"
        os.environ["EXL3_FAT_BATCHED"] = "1"
        os.environ["EXL3_FAT_KERNEL"] = "0"
        y_batched = apply_exl3_experts(x, ids, weights, layer, fused=True)
        assert layer._exl3_last_fat_fallback == "batched"
        assert torch.isfinite(y_loop).all() and torch.isfinite(y_batched).all()
        bound = max(
            0.15, 0.08 * float(y_loop.float().abs().max().clamp_min(1.0))
        )
        err = float((y_loop.float() - y_batched.float()).abs().max())
        assert err < bound, (
            f"mixed thin+fat batched vs loop maxabs={err} bound={bound}"
        )
        import exllamav3_ext

        if hasattr(exllamav3_ext, "exl3_fat_gemm"):
            os.environ["EXL3_FAT_KERNEL"] = "1"
            y_kernel = apply_exl3_experts(x, ids, weights, layer, fused=True)
            assert layer._exl3_last_fat_fallback == "kernel"
            kernel_err = float(
                (y_loop.float() - y_kernel.float()).abs().max()
            )
            assert torch.isfinite(y_kernel).all() and kernel_err < bound, (
                f"mixed thin+fat kernel vs loop maxabs={kernel_err} bound={bound}"
            )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        f"exl3 mixed thin+fat composition OK maxabs={err:.5f} bound={bound:.5f}",
        flush=True,
    )


def _check_e2_diag(device) -> None:
    """Exact E2 counters for one fat prefill; degradation never poses as kernel."""
    import torch
    from vllm.model_executor.layers.quantization.exl3 import (
        _FAT_SCRATCH_BYTES,
        _FAT_SCRATCH_CACHE,
        apply_exl3_experts,
        exl3_fat_diag,
        reset_exl3_fat_diag_counters,
    )

    import exllamav3_ext

    if not hasattr(exllamav3_ext, "exl3_fat_gemm"):
        print(
            "exl3 E2 diag counters skipped (E1 image; fail-closed covered statically)",
            flush=True,
        )
        return

    keys = (
        "EXL3_MOE_ROW_TILE",
        "EXL3_FAT_SORTED",
        "EXL3_FAT_BATCHED",
        "EXL3_FAT_KERNEL",
        "EXL3_TEMP_ROWS_FUSED",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXL3_MOE_ROW_TILE"] = "0"
        os.environ["EXL3_FAT_SORTED"] = "0"
        os.environ["EXL3_FAT_BATCHED"] = "0"
        os.environ["EXL3_FAT_KERNEL"] = "1"
        os.environ["EXL3_TEMP_ROWS_FUSED"] = "128"
        _method, layer = _tiny_layer(device, n_exp=3)
        assert layer._exl3_fat_effective_tier == "kernel", (
            layer._exl3_fat_effective_tier,
            layer._exl3_fat_tier_reason,
        )
        diag = exl3_fat_diag()
        assert diag["configured_tier"] == "kernel", diag["configured_tier"]
        assert diag["effective_tier"] == "kernel", diag["effective_tier"]
        assert diag["tier_reason"] == "kernel_ok", diag["tier_reason"]
        assert diag["shared_suh"] is True
        assert diag["shared_suh_layers"] >= 1
        assert diag["sym_fat_gemm"] and diag["sym_fat_gemm_scatter"]
        assert diag["sym_exl3_moe"]
        assert diag["cap_ok"], (diag["cap_major"], diag["cap_minor"])
        assert diag["tp_size"] >= 1
        assert diag["fused_temps_bytes"] > 0

        tokens = 160  # > the 128-row cap, so both routed experts are fat
        x = torch.randn(tokens, 256, dtype=torch.float16, device=device)
        ids = torch.zeros(tokens, 2, dtype=torch.long, device=device)
        ids[:, 1] = 1
        w = torch.full((tokens, 2), 0.5, dtype=torch.float16, device=device)

        # One accepted E2 call, measured from a clean counter window with an
        # empty scratch cache: every number below is exact, not a delta.
        _FAT_SCRATCH_CACHE.clear()
        _FAT_SCRATCH_BYTES.clear()
        reset_exl3_fat_diag_counters()
        y_kernel = apply_exl3_experts(x, ids, w, layer, fused=True)
        assert torch.isfinite(y_kernel).all()
        assert layer._exl3_last_fat_fallback == "kernel", layer._exl3_last_fat_fallback
        assert layer._exl3_last_fat_reason == "kernel_ok"
        diag = exl3_fat_diag()
        assert diag["prefill_layer_calls"] == 1, diag["prefill_layer_calls"]
        assert diag["thin_calls"] == 0 and diag["row_tile_calls"] == 0
        assert diag["fallback_calls"] == {
            "kernel": 1,
            "batched": 0,
            "sorted": 0,
            "legacy": 0,
        }, diag["fallback_calls"]
        assert diag["fallback_reasons"] == {"kernel_ok": 1}, diag["fallback_reasons"]
        assert diag["direct_calls"] == 2, diag["direct_calls"]
        assert diag["scatter_calls"] == 2, diag["scatter_calls"]
        assert diag["fat_expert_runs"] == 2, diag["fat_expert_runs"]
        assert diag["fat_scratch_allocs"] == 1, diag["fat_scratch_allocs"]
        assert diag["fat_scratch_bytes"] == diag["fat_scratch_peak_bytes"] > 0, (
            diag["fat_scratch_bytes"],
            diag["fat_scratch_peak_bytes"],
        )

        # A decode-sized call must not inherit the "kernel" label or counters.
        xd = torch.randn(2, 256, dtype=torch.float16, device=device)
        idsd = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device)
        wd = torch.full((2, 2), 0.5, dtype=torch.float16, device=device)
        apply_exl3_experts(xd, idsd, wd, layer, fused=True)
        assert layer._exl3_last_fat_fallback == "none", layer._exl3_last_fat_fallback
        assert layer._exl3_last_fat_reason == "no_fat_experts"
        assert exl3_fat_diag()["direct_calls"] == 2

        # Checkpoint without shared SUH: the kernel request visibly degrades.
        layer._exl3_shared_w13_suh = False
        before = exl3_fat_diag()
        y_sorted = apply_exl3_experts(x, ids, w, layer, fused=True)
        after = exl3_fat_diag()
        assert torch.isfinite(y_sorted).all()
        assert layer._exl3_last_fat_fallback == "sorted", layer._exl3_last_fat_fallback
        assert layer._exl3_last_fat_reason == "degraded_shared_suh"
        assert after["fallback_calls"]["sorted"] - before["fallback_calls"]["sorted"] == 1
        assert after["direct_calls"] == before["direct_calls"]
        assert after["scatter_calls"] == before["scatter_calls"]
        assert (
            after["fallback_reasons"].get("degraded_shared_suh", 0)
            - before["fallback_reasons"].get("degraded_shared_suh", 0)
            == 1
        )

        # Row tiles preempt every fat tier even with the kernel requested.
        layer._exl3_shared_w13_suh = True
        os.environ["EXL3_MOE_ROW_TILE"] = "1"
        before = exl3_fat_diag()
        y_tile = apply_exl3_experts(x, ids, w, layer, fused=True)
        after = exl3_fat_diag()
        assert torch.isfinite(y_tile).all()
        assert layer._exl3_last_fat_fallback == "row_tile"
        assert after["row_tile_calls"] - before["row_tile_calls"] == 1
        assert after["fallback_calls"]["kernel"] == before["fallback_calls"]["kernel"]
        assert after["direct_calls"] == before["direct_calls"]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(
        "exl3 E2 diag counters OK: kernel call = prefill_layer_calls 1, "
        "fallback kernel=1, direct=2, scatter=2, fat_expert_runs=2, "
        "scratch_allocs=1; shared-SUH loss -> sorted+degraded_shared_suh; "
        "row tiles preempt",
        flush=True,
    )


def _check_apply_expert_map(device) -> None:
    import os

    import torch
    from vllm.model_executor.layers.quantization.exl3 import apply_exl3_experts

    _method, layer = _tiny_layer(device, n_exp=3)
    # global 1 is not on this rank
    layer.expert_map = torch.tensor([0, -1, 2], dtype=torch.long, device=device)
    x = torch.randn(2, 256, dtype=torch.float16, device=device)
    ids = torch.tensor([[0, 1], [2, 1]], dtype=torch.long, device=device)
    w = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float16, device=device)
    y_fused = apply_exl3_experts(x, ids, w, layer, fused=True)
    y_loop = apply_exl3_experts(x, ids, w, layer, fused=False)
    assert torch.isfinite(y_fused).all() and torch.isfinite(y_loop).all()
    err = float((y_fused.float() - y_loop.float()).abs().max())
    bound = max(0.15, 0.08 * float(y_loop.float().abs().max().clamp_min(1.0)))
    assert err < bound, f"expert_map -1 fused vs loop maxabs={err} bound={bound}"
    prev = os.environ.get("EXL3_FUSED_MOE")
    os.environ["EXL3_FUSED_MOE"] = "0"
    try:
        y_env = apply_exl3_experts(x, ids, w, layer)
        assert layer._exl3_last_apply == "loop", layer._exl3_last_apply
        assert torch.isfinite(y_env).all()
    finally:
        if prev is None:
            os.environ.pop("EXL3_FUSED_MOE", None)
        else:
            os.environ["EXL3_FUSED_MOE"] = prev
    print("exl3 apply expert_map -1 + EXL3_FUSED_MOE=0 loop OK", flush=True)


def _check_fused_cudagraph(device) -> None:
    import torch
    from vllm.model_executor.layers.quantization.exl3 import apply_exl3_experts

    _method, layer = _tiny_layer(device, n_exp=3)
    # CPU map: first eager apply must pin it so capture does not CPU→CUDA copy.
    layer.expert_map = torch.tensor([0, -1, 2], dtype=torch.long, device="cpu")
    x = torch.randn(2, 256, dtype=torch.float16, device=device)
    ids = torch.tensor([[0, 1], [2, 1]], dtype=torch.long, device=device)
    w = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float16, device=device)
    static_x = x.clone()
    static_ids = ids.clone()
    static_w = w.clone()
    y_eager = apply_exl3_experts(static_x, static_ids, static_w, layer, fused=True)
    assert layer.expert_map.device.type == "cuda", layer.expert_map.device
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            apply_exl3_experts(static_x, static_ids, static_w, layer, fused=True)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_graph = apply_exl3_experts(static_x, static_ids, static_w, layer, fused=True)
    g.replay()
    torch.cuda.synchronize()
    err = float((y_graph.float() - y_eager.float()).abs().max())
    bound = max(0.15, 0.08 * float(y_eager.float().abs().max().clamp_min(1.0)))
    assert torch.isfinite(y_graph).all(), y_graph
    assert err < bound, f"cudagraph vs eager maxabs={err} bound={bound}"
    print(f"exl3 fused CUDA graph capture OK maxabs={err:.5f}", flush=True)


def _check_dflash2() -> None:
    from pathlib import Path

    from vllm.model_executor.models.qwen3_dflash import (
        DFlashQwen3DecoderLayer,
        DFlashQwen3ForCausalLM,
        DFlashQwen3Model,
    )
    from vllm.model_executor.models.qwen3_dflash2 import (
        DFlash2Qwen3DecoderLayer,
        DFlash2Qwen3ForCausalLM,
        DFlash2Qwen3Model,
    )
    from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS

    assert DFlashQwen3Model.decoder_layer_cls is DFlashQwen3DecoderLayer
    assert DFlashQwen3ForCausalLM.model_cls is DFlashQwen3Model
    assert DFlash2Qwen3Model.decoder_layer_cls is DFlash2Qwen3DecoderLayer
    assert DFlash2Qwen3ForCausalLM.model_cls is DFlash2Qwen3Model
    assert _SPECULATIVE_DECODING_MODELS["DFlash2DraftModel"] == (
        "qwen3_dflash2",
        "DFlash2Qwen3ForCausalLM",
    )
    qwen = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py"
    ).read_text()
    assert "self.decoder_layer_cls(" in qwen
    assert "DFlashQwen3DecoderLayer(" not in qwen.split("self.layers")[1].split("def embed_input_ids")[0]
    spec_init = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/__init__.py"
    ).read_text()
    assert "DFlash2Speculator" in spec_init
    assert "DFlash2DraftModel" in spec_init
    dflash_utils = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash/utils.py"
    ).read_text()
    assert 'draft_kv = "auto"' in dflash_utils
    assert '"fp8_ds_mla"' in dflash_utils
    glm = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py"
    ).read_text()
    assert "class Glm5NextModel(nn.Module, EagleModelMixin):" in glm
    assert "SupportsEagle3" in glm
    assert "aux_hidden_state_layers" in glm
    assert "layer.hc_post(hidden_states, residual, post, comb)" in glm
    assert "hc_contract(" in glm
    assert "return hidden_states, aux_hidden_states" in glm
    kv = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"
    ).read_text()
    assert "DFLASH2-DRAFTER-GROUP" in kv
    assert "type(v) is SlidingWindowSpec" in kv
    # Standalone DFlash2 must not inherit the 1152-token MLA manager block
    # (that doubled per-block bytes and pinned concurrency at ~1× max_len).
    assert "compact_block = 64" in kv
    assert "page_size_padded=mla_page" in kv
    assert "padded slot-share block=%d" in kv
    assert "s.block_size != 64 or s.page_size_padded != mla_page" in kv
    standalone = kv.split("PADDED SLOT-SHARE:")[1].split("draft_uniform")[0]
    assert "compact_block" in standalone
    assert "page_size_padded=mla_page" in standalone
    assert "new_draft_specs = dict(draft_specs)" not in standalone
    src = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py").read_text()
    # Top-level is_causal must win so GLM-5.3-Flash-DFlash2 (is_causal=false,
    # all sliding_attention) does not silently draft as causal DFlash1.
    assert 'getattr(config, "is_causal", None)' in src
    print("dflash2 overlay OK", flush=True)


def main() -> int:
    require_gpu = os.environ.get("EXL3_SELFCHECK_GPU", "1") != "0"
    _check_quant_registry()
    _check_tp_shard()
    _check_ext_arch()
    _check_e2_diag_static()
    _check_dflash2()
    if require_gpu:
        _check_gpu_gemm()
    else:
        print("EXL3_SELFCHECK_GPU=0 — skipped GPU GEMM", flush=True)
    print("glm53 EXL3 overlay verify OK", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"EXL3 overlay verify FAILED: {exc}", file=sys.stderr, flush=True)
        raise
