"""Loader for the SENS8G_V1 graph-safe router (kernel_lab, not serving).

Provenance: adapted from the donor loader (same repo/branch/SHA as the
kernel port; donor: kernels/spec_router/sens8_loader.py @ 00548bd).
Same contract: SM12x gate, load_inline build, ExtensionUnavailable ->
caller must run the stock router. Never crash the process for this.

Errors other than ExtensionUnavailable are NOT swallowed: a builder bug
must be loud, a missing GPU must be quiet.
"""
from __future__ import annotations

import os
from pathlib import Path

N_EXPERTS = 288
TOPK = 8
MAX_BLOCK_ROWS = 16
ARCH_FLAG = "-arch=sm_121a"


class ExtensionUnavailable(RuntimeError):
    """Fused kernel cannot be used (no GPU, wrong arch, build failure)."""


_MOD = None
_REASON: str | None = None


def _cuda_source() -> str:
    return (Path(__file__).parent / "sens8g_router.cu").read_text()


def _check_arch(*, need_gpu: bool = True) -> None:
    try:
        import torch
    except Exception as e:
        raise ExtensionUnavailable(f"no torch: {e}") from e
    # Lab builds may run in a GPU-less container (nvcc only); the SM gate
    # still applies at USE time on a real GPU box. Opt-in via env.
    if not torch.cuda.is_available():
        if os.environ.get("SENS8_ALLOW_CPU_BUILD") == "1" and not need_gpu:
            return
        raise ExtensionUnavailable("cuda unavailable")
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception as e:
        raise ExtensionUnavailable(f"no capability: {e}") from e
    if major != 12:
        raise ExtensionUnavailable(f"SM{major}x unsupported (need SM12x)")


def get_extension():
    global _MOD, _REASON
    if _MOD is not None:
        return _MOD
    if _REASON is not None:
        raise ExtensionUnavailable(_REASON)
    try:
        import torch as _t
        _check_arch(need_gpu=_t.cuda.is_available())
        from torch.utils.cpp_extension import load_inline

        _MOD = load_inline(
            name="glm53exl3_sens8g_router",
            cpp_sources=["int glm53exl3_sens8g_stub(void) { return 0; }"],
            cuda_sources=[_cuda_source()],
            extra_cuda_cflags=[ARCH_FLAG, "-O3"],
            build_directory=os.environ.get("SENS8_BUILD_DIR") or None,
            verbose=False,
        )
    except ExtensionUnavailable:
        raise
    except Exception as e:
        _REASON = f"build failed: {type(e).__name__}: {e}"
        raise ExtensionUnavailable(_REASON) from e
    return _MOD


def is_available() -> bool:
    try:
        get_extension()
        return True
    except ExtensionUnavailable:
        return False


MODE_STOCK, MODE_SENS8 = 0, 1


def make_meta(mode, blocks):
    """Fixed [6] int32 metadata: {mode, nb, len0..len3}. Full overwrite
    per step by the caller (never partial updates)."""
    import torch
    assert mode in (MODE_STOCK, MODE_SENS8)
    if mode == MODE_SENS8:
        assert 1 <= len(blocks) <= 4 and all(1 <= n <= 16 for n in blocks)
        lens = list(blocks) + [0] * (4 - len(blocks))
        return torch.tensor([mode, len(blocks)] + lens, dtype=torch.int32)
    return torch.zeros(6, dtype=torch.int32)


def fused_forward(logits, bias, meta, ids, w, dbg):
    """Single fixed grid launch. Shape/dtype gates here (defense in
    depth; the kernel TORCH_CHECKs too). dbg: CUDA int32 [2] launch
    counters (device-side; read offline, never in the hot path)."""
    import torch

    if logits.dim() != 2 or logits.size(1) != N_EXPERTS:
        raise ExtensionUnavailable(f"logits shape {tuple(logits.shape)}")
    if logits.dtype not in (torch.float32, torch.bfloat16):
        raise ExtensionUnavailable(f"logits dtype {logits.dtype}")
    if bias.numel() != N_EXPERTS or bias.dtype != torch.float32:
        raise ExtensionUnavailable("bias must be fp32 [288]")
    if meta.numel() != 6 or meta.dtype != torch.int32:
        raise ExtensionUnavailable("meta must be int32 [6]")
    if dbg.numel() != 2 or dbg.dtype != torch.int32:
        raise ExtensionUnavailable("dbg must be int32 [2]")
    get_extension().forward(logits, bias, meta, ids, w, dbg)
