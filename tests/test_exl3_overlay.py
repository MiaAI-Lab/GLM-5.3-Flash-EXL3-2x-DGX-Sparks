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
        exl3_fat_moe_symbols,
        exl3_fat_symbols,
        resolve_exl3_fat_tier,
    )

    diag = exl3_fat_diag()
    assert diag["schema"] == EXL3_FAT_DIAG_SCHEMA == 2, diag["schema"]
    assert tuple(sorted(diag)) == tuple(sorted(EXL3_FAT_DIAG_KEYS)), (
        set(diag) ^ set(EXL3_FAT_DIAG_KEYS)
    )
    for key in ("sym_fat_moe", "grouped_calls", "grouped_scratch_bytes", "grouped_eligible"):
        assert key in diag, key
    assert "grouped" in diag["fallback_calls"], diag["fallback_calls"]

    keys = ("EXL3_FAT_SORTED", "EXL3_FAT_BATCHED", "EXL3_FAT_KERNEL", "EXL3_FAT_GROUPED")
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

        # E3 grouped tier (experimental, opt-in): default OFF leaves E2 alone.
        os.environ["EXL3_FAT_KERNEL"] = "1"
        assert configured_fat_tier() == "kernel"
        os.environ["EXL3_FAT_GROUPED"] = "0"
        assert configured_fat_tier() == "kernel"
        os.environ["EXL3_FAT_GROUPED"] = "1"
        assert configured_fat_tier() == "grouped"
        # Checkpoint cap precedes every image check, as for E2.
        assert resolve_exl3_fat_tier(False, (True, True, True), True) == (
            "sorted",
            "shared_suh_absent",
        )
        # Missing E3 symbols with grouped requested: fail closed at load.
        try:
            resolve_exl3_fat_tier(True, (True, True, True), False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("grouped tier must fail closed without E3 symbols")
        assert resolve_exl3_fat_tier(True, (True, True, True), True) == (
            "grouped",
            "grouped_ok",
        )
        assert resolve_exl3_fat_tier(
            True, (True, True, True), True, (True, "eligible")
        ) == ("grouped", "grouped_ok")
        # Ineligible checkpoint/device: deliberate, visible fallback to E2,
        # which is itself subject to the E2 symbol check.
        assert resolve_exl3_fat_tier(
            True, (True, True, True), True, (False, "bits_3")
        ) == ("kernel", "grouped_ineligible_bits_3:kernel_ok")
        try:
            resolve_exl3_fat_tier(True, (True, False, True), True, (False, "bits_3"))
        except RuntimeError:
            pass
        else:
            raise AssertionError("ineligible grouped -> kernel must still need E2 symbols")
        # The cap is not changed implicitly by the grouped flag.
        from vllm.model_executor.layers.quantization.exl3 import temp_rows_fused

        prev_rows = os.environ.pop("EXL3_TEMP_ROWS_FUSED", None)
        try:
            assert temp_rows_fused() == 128, temp_rows_fused()
            os.environ["EXL3_TEMP_ROWS_FUSED"] = "32"
            assert temp_rows_fused() == 32
        finally:
            if prev_rows is None:
                os.environ.pop("EXL3_TEMP_ROWS_FUSED", None)
            else:
                os.environ["EXL3_TEMP_ROWS_FUSED"] = prev_rows
        live_grouped = exl3_fat_moe_symbols()
        if live_grouped:
            assert resolve_exl3_fat_tier(True, symbols=(True, True, True)) == ("grouped", "grouped_ok")
        else:
            try:
                resolve_exl3_fat_tier(True, symbols=(True, True, True))
            except RuntimeError:
                pass
            else:
                raise AssertionError("this image lacks the E3 kernels; grouped must fail closed")
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
    _check_grouped_fat(device)



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
    prev_grouped = os.environ.get("EXL3_FAT_GROUPED")
    os.environ["EXL3_FAT_EXPERT_LOG"] = "1"
    os.environ["EXL3_TEMP_ROWS_FUSED"] = "128"
    os.environ["EXL3_FAT_GROUPED"] = "0"
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
        if prev_grouped is None:
            os.environ.pop("EXL3_FAT_GROUPED", None)
        else:
            os.environ["EXL3_FAT_GROUPED"] = prev_grouped
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
        "EXL3_FAT_GROUPED",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXL3_FAT_GROUPED"] = "0"
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
        "EXL3_FAT_GROUPED",
        "EXL3_TEMP_ROWS_FUSED",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXL3_MOE_ROW_TILE"] = "0"
        os.environ["EXL3_FAT_SORTED"] = "0"
        os.environ["EXL3_FAT_BATCHED"] = "0"
        os.environ["EXL3_FAT_KERNEL"] = "1"
        os.environ["EXL3_FAT_GROUPED"] = "0"
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
            "grouped": 0,
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



def _grouped_reference_tables(counts: list[int], cap: int, tile: int):
    """Straightforward host reference for build_grouped_fat_tables."""
    sorted_off = [0]
    for c in counts:
        sorted_off.append(sorted_off[-1] + c)
    segs: list[tuple[int, int, int]] = []
    row_src: list[int] = []
    row_expert: list[int] = []
    row0 = 0
    for e, c in enumerate(counts):
        if c <= cap:
            continue
        for t in range(0, c, tile):
            segs.append((e, row0 + t, min(tile, c - t)))
        for i in range(c):
            row_src.append(sorted_off[e] + i)
            row_expert.append(e)
        row0 += c
    return segs, row_src, row_expert, row0


def _check_grouped_tables(device) -> None:
    import torch
    from vllm.model_executor.layers.quantization.exl3 import build_grouped_fat_tables

    cases = [
        ("no_fat", [5, 3, 0, 7], 8, [64]),
        ("all_fat", [9, 20, 33], 8, [64]),
        ("mixed_skewed", [1, 200, 8, 9, 0, 64, 1000, 2], 32, [64]),
        ("ragged_63_64_65", [63, 64, 65, 127, 128, 129, 1], 32, [64, 16]),
        ("cap_boundary", [31, 32, 33], 32, [64]),
        ("non_pow2", [100, 37, 3], 16, [64]),
        ("trailing_zero", [50, 0, 0], 8, [64]),
        ("uniform", [40] * 6, 32, [64]),
        ("one_expert_huge", [5000, 1], 32, [64]),
    ]
    for name, counts, cap, tiles in cases:
        n_exp = len(counts)
        # Sorted layout with a trailing invalid/nonlocal sentinel bucket.
        local = torch.cat(
            [torch.full((c,), e, dtype=torch.long) for e, c in enumerate(counts)]
            + [torch.full((3,), n_exp, dtype=torch.long)]
        )
        perm = torch.randperm(local.numel())
        local = local[perm].to(device)
        flat_token = torch.arange(local.numel(), device=device)  # unique per route
        flat_weight = torch.rand(local.numel(), device=device).half()
        order = local.argsort()
        token_sorted = flat_token[order]
        weight_sorted = flat_weight[order]
        expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=device)
        expert_count.scatter_add_(0, local, torch.ones_like(local))
        cnt = expert_count[:n_exp]
        rows_cap = int(local.numel())
        for tile in tiles:
            tb = build_grouped_fat_tables(cnt, cap, token_sorted, weight_sorted, rows_cap, tile)
            segs, row_src, row_expert, num_rows = _grouped_reference_tables(counts, cap, tile)
            ns = int(tb["num_segs"].item())
            nr = int(tb["num_rows"].item())
            assert ns == len(segs), (name, tile, ns, len(segs))
            assert nr == num_rows, (name, tile, nr, num_rows)
            got_segs = list(zip(tb["seg_expert"][:ns].tolist(), tb["seg_row0"][:ns].tolist(), tb["seg_rows"][:ns].tolist()))
            assert got_segs == segs, (name, tile, got_segs[:5], segs[:5])
            assert tb["row_expert"][:nr].tolist() == row_expert, (name, tile)
            exp_tok = token_sorted[torch.tensor(row_src, dtype=torch.long, device=device)] if row_src else token_sorted[:0]
            assert torch.equal(tb["row_token"][:nr], exp_tok), (name, tile)
            exp_w = weight_sorted[torch.tensor(row_src, dtype=torch.long, device=device)] if row_src else weight_sorted[:0]
            assert torch.equal(tb["row_weight"][:nr], exp_w), (name, tile)
            # No fat row may point at a sentinel route (they sort last).
            assert all(src < sum(counts) for src in row_src)
            assert int(tb["seg_expert"].numel()) >= ns and int(tb["row_token"].numel()) == rows_cap
    print(f"exl3 E3 row/segment tables vs reference OK ({len(cases)} cases)", flush=True)


def _err_stats(ref, got):
    import torch

    ref = ref.float()
    got = got.float()
    err = (ref - got).abs()
    per_tok = err.max(dim=1).values
    rms_ref = float(ref.pow(2).mean().sqrt().clamp_min(1e-6))
    return {
        "maxabs": float(err.max()),
        "meanabs": float(err.mean()),
        "nrmse": float((ref - got).pow(2).mean().sqrt()) / rms_ref,
        "per_token_max": float(per_tok.max()),
        "per_token_p99": float(per_tok.kthvalue(max(1, int(0.99 * per_tok.numel()))).values),
        "ref_max": float(ref.abs().max()),
        "ref_mean": float(ref.abs().mean()),
        "finite": bool(torch.isfinite(got).all()),
    }


# Frozen BEFORE any E3 measurement (plan: tolerances derive from E2-vs-loop
# and E2-repeat observations, never widened after a failure): E3 may differ
# from the LinearEXL3 loop by at most 1.5x what E2 differs, plus an absolute
# floor of 1e-3 x max|ref| for the per-token/max metrics and 1e-4 for nRMSE.
E3_TOL_FACTOR = 1.5
E3_TOL_ABS_REL = 1e-3
E3_TOL_NRMSE_ABS = 1e-4


def _assert_e3_within(name: str, e2: dict, e3: dict) -> None:
    floor = E3_TOL_ABS_REL * e2["ref_max"]
    assert e3["finite"], f"{name}: E3 produced non-finite values"
    for key in ("maxabs", "per_token_max", "per_token_p99"):
        bound = E3_TOL_FACTOR * e2[key] + floor
        assert e3[key] <= bound, f"{name}: E3 {key}={e3[key]:.5f} > {bound:.5f} (E2 {e2[key]:.5f})"
    bound = E3_TOL_FACTOR * e2["nrmse"] + E3_TOL_NRMSE_ABS
    assert e3["nrmse"] <= bound, f"{name}: E3 nrmse={e3['nrmse']:.6f} > {bound:.6f} (E2 {e2['nrmse']:.6f})"
    coarse = max(0.15, 0.08 * max(1.0, e2["ref_max"]))
    assert e3["maxabs"] < coarse, f"{name}: E3 maxabs {e3['maxabs']} exceeds coarse bound {coarse}"


def _grouped_env(cap: int):
    return {
        "EXL3_MOE_ROW_TILE": "0",
        "EXL3_FAT_SORTED": "0",
        "EXL3_FAT_BATCHED": "0",
        "EXL3_FAT_KERNEL": "1",
        "EXL3_FAT_GROUPED": "1",
        "EXL3_FAT_EXPERT_LOG": "0",
        "EXL3_TEMP_ROWS_FUSED": str(cap),
    }


def _real_expert_layer(device, n_exp: int = 3, cap: int = 32):
    """Layer built from real checkpoint expert tensors (layer 3, experts 0..n).

    Uses the head node's HF cache when mounted; returns None otherwise.
    """
    import glob
    import json

    import torch
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod

    snaps = glob.glob("/root/.cache/huggingface/hub/models--brandonmusic--GLM-5.3-Flash-tr3-4bpw/snapshots/*/model.safetensors.index.json")
    if not snaps:
        return None
    try:
        from safetensors import safe_open
    except Exception:
        return None
    index_path = snaps[0]
    root = index_path.rsplit("/", 1)[0]
    wmap = json.load(open(index_path))["weight_map"]
    prefix = "model.language_model.layers.3.mlp.experts"
    tensors = {}
    files = {}
    for e in range(n_exp):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suf in ("trellis", "suh", "svh", "mcg"):
                name = f"{prefix}.{e}.{proj}.{suf}"
                files.setdefault(wmap[name], []).append(name)
    for fname, names in files.items():
        with safe_open(f"{root}/{fname}", framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name)
    g0 = tensors[f"{prefix}.0.gate_proj.trellis"]
    hidden = int(g0.shape[0] * 16)
    inter = int(g0.shape[1] * 16)
    import types

    moe = types.SimpleNamespace(swiglu_limit=10.0)
    method = Exl3MoEMethod(moe, Exl3Config())
    layer = torch.nn.Module()
    method.create_weights(layer, num_experts=n_exp, hidden_size=hidden,
                          intermediate_size_per_partition=inter, params_dtype=torch.float16)
    with torch.no_grad():
        for e in range(n_exp):
            for si, proj in ((0, "gate_proj"), (1, "up_proj")):
                layer.w13_trellis[e, si].copy_(tensors[f"{prefix}.{e}.{proj}.trellis"])
                layer.w13_suh[e, si].copy_(tensors[f"{prefix}.{e}.{proj}.suh"])
                layer.w13_svh[e, si].copy_(tensors[f"{prefix}.{e}.{proj}.svh"])
                layer.w13_mcg[e, si].copy_(tensors[f"{prefix}.{e}.{proj}.mcg"].reshape(-1)[:1])
            layer.w2_trellis[e].copy_(tensors[f"{prefix}.{e}.down_proj.trellis"])
            layer.w2_suh[e].copy_(tensors[f"{prefix}.{e}.down_proj.suh"])
            layer.w2_svh[e].copy_(tensors[f"{prefix}.{e}.down_proj.svh"])
            layer.w2_mcg[e].copy_(tensors[f"{prefix}.{e}.down_proj.mcg"].reshape(-1)[:1])
    layer = layer.to(device)
    method.process_weights_after_loading(layer)
    return layer, hidden, inter


def _check_grouped_fat(device) -> None:
    """E3 grouped tier: tables, parity vs loop/E2 within frozen tolerances,
    real checkpoint scales, graph replay with changing data, scratch growth,
    diag counters, ineligible fallback, missing-symbol fail-closed."""
    import json

    import torch
    from vllm.model_executor.layers.quantization import exl3 as exl3mod
    from vllm.model_executor.layers.quantization.exl3 import (
        _FAT_GROUPED_CACHE,
        _record_exl3_fat_resolution,
        apply_exl3_experts,
        exl3_fat_diag,
        exl3_fat_moe_symbols,
        reset_exl3_fat_diag_counters,
        resolve_exl3_fat_tier,
    )

    require = os.environ.get("EXL3_SELFCHECK_REQUIRE_GROUPED", "0") == "1"
    if not exl3_fat_moe_symbols():
        if require:
            raise AssertionError("EXL3_SELFCHECK_REQUIRE_GROUPED=1 but the E3 kernels are absent")
        print("exl3 E3 grouped kernels absent (E2 image) — grouped checks skipped", flush=True)
        return
    _check_grouped_tables(device)

    keys = tuple(_grouped_env(32).keys())
    previous = {key: os.environ.get(key) for key in keys}
    results: dict = {}
    try:
        cap = 32
        os.environ.update(_grouped_env(cap))
        # --- routing with fat + thin + overlapping contributions ---
        _method, layer = _tiny_layer(device, n_exp=4)
        assert layer._exl3_fat_effective_tier == "grouped", (layer._exl3_fat_effective_tier, layer._exl3_fat_tier_reason)
        tokens = 300
        g = torch.Generator(device="cpu")
        g.manual_seed(7)
        x = torch.randn(tokens, 256, generator=g).half().to(device)
        ids = torch.zeros(tokens, 2, dtype=torch.long)
        ids[:, 0] = 0                       # fat: 300 rows
        ids[:200, 1] = 1                    # fat: 200 rows
        ids[200:280, 1] = 2                 # fat: 80 rows
        ids[280:, 1] = 3                    # thin: 20 rows (fused kernel)
        ids = ids.to(device)
        w = torch.rand(tokens, 2, generator=g).softmax(-1).half().to(device)

        y_loop = apply_exl3_experts(x, ids, w, layer, fused=False)
        os.environ["EXL3_FAT_GROUPED"] = "0"
        y_e2 = apply_exl3_experts(x, ids, w, layer, fused=True)
        assert layer._exl3_last_fat_fallback == "kernel", layer._exl3_last_fat_fallback
        y_e2b = apply_exl3_experts(x, ids, w, layer, fused=True)
        os.environ["EXL3_FAT_GROUPED"] = "1"
        e2 = _err_stats(y_loop, y_e2)
        e2_rep = _err_stats(y_e2, y_e2b)
        # E3 measured only after E2 statistics (tolerances) are fixed.
        _FAT_GROUPED_CACHE.clear()
        reset_exl3_fat_diag_counters()
        y_e3 = apply_exl3_experts(x, ids, w, layer, fused=True)
        assert layer._exl3_last_fat_fallback == "grouped", layer._exl3_last_fat_fallback
        assert layer._exl3_last_fat_reason == "grouped_ok"
        y_e3b = apply_exl3_experts(x, ids, w, layer, fused=True)
        e3 = _err_stats(y_loop, y_e3)
        e3_rep = _err_stats(y_e3, y_e3b)
        e3_vs_e2 = _err_stats(y_e2, y_e3)
        results["mixed"] = {"e2_vs_loop": e2, "e2_repeat": e2_rep, "e3_vs_loop": e3, "e3_repeat": e3_rep, "e3_vs_e2": e3_vs_e2}
        _assert_e3_within("mixed", e2, e3)
        diag = exl3_fat_diag()
        assert diag["grouped_calls"] == 2, diag["grouped_calls"]
        assert diag["fallback_calls"]["grouped"] == 2, diag["fallback_calls"]
        assert diag["direct_calls"] == 0 and diag["scatter_calls"] == 0, (diag["direct_calls"], diag["scatter_calls"])
        assert diag["grouped_scratch_bytes"] > 0 and diag["grouped_eligible"] and diag["sym_fat_moe"]
        assert diag["effective_tier"] == "grouped" and diag["configured_tier"] == "grouped"

        # --- value regimes: ordinary small, saturation/clamp, near-zero ---
        for label, scale in (("small", 0.05), ("saturate", 40.0), ("near_zero", 1e-3)):
            xs = (x.float() * scale).half()
            yl = apply_exl3_experts(xs, ids, w, layer, fused=False)
            os.environ["EXL3_FAT_GROUPED"] = "0"
            ye2 = apply_exl3_experts(xs, ids, w, layer, fused=True)
            os.environ["EXL3_FAT_GROUPED"] = "1"
            ye3 = apply_exl3_experts(xs, ids, w, layer, fused=True)
            s2 = _err_stats(yl, ye2)
            s3 = _err_stats(yl, ye3)
            results[label] = {"e2_vs_loop": s2, "e3_vs_loop": s3}
            _assert_e3_within(label, s2, s3)

        # --- all-invalid routes above the cap: no crash, zero output ---
        layer.expert_map = torch.tensor([0, -1, 2, 3], dtype=torch.long, device=device)
        ids_bad = torch.full((tokens, 2), 1, dtype=torch.long, device=device)
        y_bad = apply_exl3_experts(x, ids_bad, w, layer, fused=True)
        assert torch.isfinite(y_bad).all() and float(y_bad.abs().max()) == 0.0
        layer.expert_map = None

        # --- scratch growth outside capture, warm reuse ---
        base_ptr = _FAT_GROUPED_CACHE[next(iter(_FAT_GROUPED_CACHE))]["h13"].data_ptr()
        xb = torch.randn(1200, 256, generator=g).half().to(device)
        idsb = torch.zeros(1200, 2, dtype=torch.long, device=device)
        idsb[:, 1] = 1
        wb = torch.full((1200, 2), 0.5, dtype=torch.float16, device=device)
        ylb = apply_exl3_experts(xb, idsb, wb, layer, fused=False)
        os.environ["EXL3_FAT_GROUPED"] = "0"
        ye2b = apply_exl3_experts(xb, idsb, wb, layer, fused=True)
        os.environ["EXL3_FAT_GROUPED"] = "1"
        ye3b = apply_exl3_experts(xb, idsb, wb, layer, fused=True)
        grown = _FAT_GROUPED_CACHE[next(iter(_FAT_GROUPED_CACHE))]["h13"]
        assert int(grown.shape[0]) >= 2400, grown.shape
        sb2 = _err_stats(ylb, ye2b)
        sb3 = _err_stats(ylb, ye3b)
        results["grown_1200"] = {"e2_vs_loop": sb2, "e3_vs_loop": sb3}
        _assert_e3_within("grown_1200", sb2, sb3)
        apply_exl3_experts(x, ids, w, layer, fused=True)
        assert _FAT_GROUPED_CACHE[next(iter(_FAT_GROUPED_CACHE))]["h13"].data_ptr() == grown.data_ptr()
        del base_ptr

        # --- CUDA graph: capture a batch > cap, replay with changed data ---
        static_x = x.clone()
        static_ids = ids.clone()
        static_w = w.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                apply_exl3_experts(static_x, static_ids, static_w, layer, fused=True)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        before_calls = exl3_fat_diag()["grouped_calls"]
        with torch.cuda.graph(graph):
            y_graph = apply_exl3_experts(static_x, static_ids, static_w, layer, fused=True)
        graph.replay()
        torch.cuda.synchronize()
        y_eager = apply_exl3_experts(static_x, static_ids, static_w, layer, fused=True)
        r0 = _err_stats(y_eager, y_graph)
        assert r0["finite"] and r0["maxabs"] <= E3_TOL_FACTOR * e3_rep["maxabs"] + E3_TOL_ABS_REL * r0["ref_max"], r0
        # New activations, ids, weights and a different fat/thin split in place.
        static_x.copy_(torch.randn(tokens, 256, generator=g).half().to(device))
        new_ids = torch.zeros(tokens, 2, dtype=torch.long)
        new_ids[:, 0] = 2                   # fat
        new_ids[:40, 1] = 0                 # fat (40 > 32)
        new_ids[40:60, 1] = 1               # thin
        new_ids[60:, 1] = 3                 # fat
        static_ids.copy_(new_ids.to(device))
        static_w.copy_(torch.rand(tokens, 2, generator=g).softmax(-1).half().to(device))
        graph.replay()
        torch.cuda.synchronize()
        y_graph2 = y_graph.clone()
        y_eager2 = apply_exl3_experts(static_x, static_ids, static_w, layer, fused=True)
        y_loop2 = apply_exl3_experts(static_x, static_ids, static_w, layer, fused=False)
        r1 = _err_stats(y_eager2, y_graph2)
        assert r1["finite"] and r1["maxabs"] <= E3_TOL_FACTOR * e3_rep["maxabs"] + E3_TOL_ABS_REL * r1["ref_max"], r1
        r_loop = _err_stats(y_loop2, y_graph2)
        assert r_loop["maxabs"] < max(0.15, 0.08 * max(1.0, r_loop["ref_max"])), r_loop
        # The replay must NOT be the stale first result.
        stale = _err_stats(y_eager, y_graph2)
        assert stale["maxabs"] > 10 * r1["maxabs"] + 1e-3, "graph replay ignored live data"
        results["graph"] = {"replay_vs_eager_first": r0, "replay_vs_eager_changed": r1, "replay_vs_loop_changed": r_loop}
        # Python counters only count capture, not replays (documented).
        assert exl3_fat_diag()["grouped_calls"] >= before_calls + 1

        # --- real checkpoint experts at production geometry ---
        real = _real_expert_layer(device, n_exp=3, cap=cap)
        if real is None:
            print("exl3 E3 real-checkpoint parity SKIPPED (HF cache not mounted)", flush=True)
            results["real"] = None
        else:
            rlayer, hidden, inter = real
            assert rlayer._exl3_fat_effective_tier == "grouped", (rlayer._exl3_fat_effective_tier, rlayer._exl3_fat_tier_reason)
            rt = 200
            rids = torch.zeros(rt, 2, dtype=torch.long)
            rids[:, 0] = 0
            rids[:150, 1] = 1
            rids[150:, 1] = 2
            rids = rids.to(device)
            rw = torch.rand(rt, 2, generator=g).softmax(-1).half().to(device)
            results["real"] = {"hidden": hidden, "intermediate": inter}
            for label, scale in (("unit", 1.0), ("small", 0.01), ("saturate", 50.0)):
                rx = (torch.randn(rt, hidden, generator=g) * scale).half().to(device)
                yl = apply_exl3_experts(rx, rids, rw, rlayer, fused=False)
                os.environ["EXL3_FAT_GROUPED"] = "0"
                ye2 = apply_exl3_experts(rx, rids, rw, rlayer, fused=True)
                assert rlayer._exl3_last_fat_fallback == "kernel"
                ye2b = apply_exl3_experts(rx, rids, rw, rlayer, fused=True)
                os.environ["EXL3_FAT_GROUPED"] = "1"
                ye3 = apply_exl3_experts(rx, rids, rw, rlayer, fused=True)
                assert rlayer._exl3_last_fat_fallback == "grouped"
                s2 = _err_stats(yl, ye2)
                s3 = _err_stats(yl, ye3)
                results["real"][label] = {"e2_vs_loop": s2, "e2_repeat": _err_stats(ye2, ye2b), "e3_vs_loop": s3, "e3_vs_e2": _err_stats(ye2, ye3)}
                _assert_e3_within(f"real_{label}", s2, s3)
            del rlayer
            torch.cuda.empty_cache()

        # --- ineligible checkpoint: deliberate fallback to the E2 kernel ---
        saved_bits = layer._exl3_bits
        layer._exl3_bits = 3
        _record_exl3_fat_resolution(layer)
        assert layer._exl3_fat_effective_tier == "kernel", layer._exl3_fat_effective_tier
        assert layer._exl3_fat_tier_reason.startswith("grouped_ineligible_bits_3"), layer._exl3_fat_tier_reason
        y_fb = apply_exl3_experts(x, ids, w, layer, fused=True)
        assert layer._exl3_last_fat_fallback == "kernel", layer._exl3_last_fat_fallback
        assert torch.isfinite(y_fb).all()
        layer._exl3_bits = saved_bits
        _record_exl3_fat_resolution(layer)
        assert layer._exl3_fat_effective_tier == "grouped"

        # --- missing symbols with grouped requested: fail closed ---
        saved_cache = list(exl3mod._FAT_MOE_EXT_CACHE)
        exl3mod._FAT_MOE_EXT_CACHE[:] = [None]
        try:
            resolve_exl3_fat_tier(True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("grouped without symbols must fail closed")
        finally:
            exl3mod._FAT_MOE_EXT_CACHE[:] = saved_cache
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    out = os.environ.get("EXL3_SELFCHECK_JSON")
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
    print(
        "exl3 E3 grouped OK: mixed e2_vs_loop maxabs={:.4f} nrmse={:.6f} | e3_vs_loop maxabs={:.4f} nrmse={:.6f} | "
        "e3_vs_e2 maxabs={:.4f} | e3_repeat maxabs={:.5f} | real={}".format(
            results["mixed"]["e2_vs_loop"]["maxabs"], results["mixed"]["e2_vs_loop"]["nrmse"],
            results["mixed"]["e3_vs_loop"]["maxabs"], results["mixed"]["e3_vs_loop"]["nrmse"],
            results["mixed"]["e3_vs_e2"]["maxabs"], results["mixed"]["e3_repeat"]["maxabs"],
            "skipped" if results.get("real") is None else "ok",
        ),
        flush=True,
    )


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
