# SPDX-License-Identifier: AGPL-3.0-only
"""Display-reserve KV backing for GB10 (installed as vllm/v1/worker/glm53_display_kv.py).

The UEFI display reservation (~2 GiB per Spark) is invisible to CUDA. With
``nvidia_drm modeset=1 fbdev=0`` and no desktop, a DRM dumb buffer of up to
~2032 MiB can be created there, mmapped and registered with CUDA as
device-mapped I/O memory (``libglm53_display_kv.so``). This module:

* ``credit()``            -- opens the pool once per worker (before the KV budget
                             is computed) and returns its byte size so
                             ``determine_available_memory`` can add it to the
                             KV budget ("additive": ordinary KV is untouched);
* ``alloc_int8(n, dev)``  -- replaces ``torch.zeros(n, int8, device)`` in the KV
                             tensor allocators: carves from the pool while a
                             tensor fits, otherwise falls back to ``torch.zeros``.

Contract: every byte credited is either handed out or left unused; the pool
never grows the ordinary budget, and a failed pool open with the knob forced
on aborts boot instead of silently serving with a smaller cache.

Technique: coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark (AGPL-3.0).

Knobs (worker environment):
  GLM53_DISPLAY_KV       1 (default) | 0 | auto. ``auto`` disables quietly when
                         the DRM node is missing or lacks dumb buffers; ``1``
                         raises; ``0`` never touches DRM.
  GLM53_DISPLAY_KV_MIB   pool size, multiple of 16, default 1792 (max seen 2032).
  GLM53_DRM_CARD         DRM node inside the container, default /dev/dri/card0.
  GLM53_DISPLAY_KV_LIB   path of libglm53_display_kv.so
                         (default /usr/local/lib/libglm53_display_kv.so).
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

MARK = "[glm53-display-kv]"
ALIGN = 4096
_owner = None
_plan: dict[int, bool] = {}
_state = {"mode": None, "credited": 0, "carved": 0, "tensors": 0, "fallback": 0}


def _log(msg: str) -> None:
    print(f"{MARK} {msg}", file=sys.stderr, flush=True)


def _mode() -> str:
    raw = os.environ.get("GLM53_DISPLAY_KV", "1").strip().lower()
    if raw in ("1", "on", "true", "yes"):
        return "on"
    if raw in ("0", "off", "false", "no", ""):
        return "off"
    if raw == "auto":
        return "auto"
    raise ValueError(f"GLM53_DISPLAY_KV={raw!r}: expected 1, 0 or auto")


def _pool_bytes() -> int:
    mib = int(os.environ.get("GLM53_DISPLAY_KV_MIB", "1792"))
    if mib <= 0 or mib % 16:
        raise ValueError(f"GLM53_DISPLAY_KV_MIB={mib}: must be a positive multiple of 16")
    return mib << 20


class Owner:
    """Process-lifetime owner of one display pool (never destroyed while views live)."""

    def __init__(self, library: str, drm_card: str, nbytes: int):
        import torch

        if not torch.cuda.is_initialized():
            raise RuntimeError("CUDA must be initialised before opening the display pool")
        self.lib = ctypes.CDLL(library)
        self.lib.glm53_display_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.glm53_display_create.restype = ctypes.c_void_p
        self.lib.glm53_display_pointer.argtypes = [ctypes.c_void_p]
        self.lib.glm53_display_pointer.restype = ctypes.c_uint64
        self.lib.glm53_display_destroy.argtypes = [ctypes.c_void_p]
        self.lib.glm53_display_destroy.restype = None
        self.lib.glm53_display_error.restype = ctypes.c_char_p
        torch.cuda.current_stream().synchronize()  # ensure a current context
        self.handle = self.lib.glm53_display_create(drm_card.encode(), nbytes)
        if not self.handle:
            raise RuntimeError(self.lib.glm53_display_error().decode())
        self.pointer = int(self.lib.glm53_display_pointer(self.handle))
        self.size = nbytes
        self.used = 0
        self.views: list = []

    def carve(self, nbytes: int):
        """Return an int8 CUDA tensor of exactly ``nbytes`` from the pool or None."""
        import torch

        off = (self.used + ALIGN - 1) // ALIGN * ALIGN
        if off + nbytes > self.size:
            return None
        view = _View(self.pointer + off, nbytes)
        t = torch.as_tensor(view, device="cuda:0")
        if t.data_ptr() != view.ptr or t.dtype != torch.int8 or t.numel() != nbytes:
            raise RuntimeError("CUDA array interface copied or misinterpreted external storage")
        self.used = off + nbytes
        self.views.append(view)  # keep alive for the process lifetime
        return t


class _View:
    def __init__(self, ptr: int, nbytes: int):
        self.ptr = ptr
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "strides": None,
            "typestr": "|i1",
            "data": (ptr, False),
            "version": 3,
        }


def credit() -> int:
    """Open the pool (once) and return the bytes to add to the KV budget."""
    global _owner
    mode = _mode()
    if _state["mode"] is not None:
        return _state["credited"]
    _state["mode"] = mode
    if mode == "off":
        _log("disabled (GLM53_DISPLAY_KV=0)")
        return 0
    lib = os.environ.get("GLM53_DISPLAY_KV_LIB", "/usr/local/lib/libglm53_display_kv.so")
    card = os.environ.get("GLM53_DRM_CARD", "/dev/dri/card0")
    nbytes = _pool_bytes()
    try:
        if not Path(lib).exists():
            raise RuntimeError(f"{lib} missing (overlay not applied?)")
        if not Path(card).exists():
            raise RuntimeError(f"{card} missing: run the container with --device {card}")
        _owner = Owner(lib, card, nbytes)
    except Exception as exc:  # noqa: BLE001 - policy decision below
        if mode == "on":
            raise RuntimeError(
                f"{MARK} display pool unavailable: {exc}. Need nvidia_drm modeset=1 "
                "fbdev=0 on the host, an idle headless GPU and --device /dev/dri/card0; "
                "set GLM53_DISPLAY_KV=0 to serve without it."
            ) from exc
        _log(f"auto: display pool unavailable ({exc}); serving with ordinary KV only")
        return 0
    _state["credited"] = nbytes
    _log(json.dumps(dict(stage="pool_open", bytes=nbytes, mib=nbytes >> 20, ptr=hex(_owner.pointer), card=card)))
    return nbytes


def plan(sizes: list[int]) -> None:
    """Choose which of the upcoming allocations come from the pool.

    First-fit decreasing over the allocation sizes (index -> in pool). The KV
    tensors are allocated in layer order with mixed sizes, so plain greedy
    could strand a few hundred MiB of pool and spill that onto ordinary
    memory beyond the profiled budget; FFD keeps the stranded amount to at
    most the smallest tensor. Without a plan ``alloc_int8`` is greedy.
    """
    global _plan
    _plan = {}
    if _owner is None:
        return
    left = _owner.size - _owner.used
    for i in sorted(range(len(sizes)), key=lambda k: -sizes[k]):
        need = (sizes[i] + ALIGN - 1) // ALIGN * ALIGN
        if need <= left:
            _plan[i] = True
            left -= need
    _log(json.dumps(dict(stage="plan", tensors=len(sizes), from_pool=len(_plan),
                         pool_mib=_owner.size >> 20, planned_mib=sum(sizes[i] for i in _plan) >> 20,
                         stranded_mib=left >> 20)))


def alloc_int8(nbytes: int, device, index: int | None = None):
    """Drop-in for torch.zeros(nbytes, dtype=torch.int8, device=device).

    ``index`` is the allocation's position in the list given to ``plan``;
    when a plan exists only planned indices are carved.
    """
    import torch

    if _owner is not None and (not _plan or index is None or _plan.get(index)):
        t = _owner.carve(nbytes)
        if t is not None:
            t.zero_()
            _state["carved"] += nbytes
            _state["tensors"] += 1
            return t
    if _owner is not None:
        _state["fallback"] += 1
    return torch.zeros(nbytes, dtype=torch.int8, device=device)


def report() -> None:
    if _owner is None:
        return
    _log(json.dumps(dict(stage="kv_allocated", pool_mib=_owner.size >> 20, carved_mib=_state["carved"] >> 20,
                         tensors_from_pool=_state["tensors"], tensors_ordinary=_state["fallback"],
                         unused_pool_mib=(_owner.size - _owner.used) >> 20)))
