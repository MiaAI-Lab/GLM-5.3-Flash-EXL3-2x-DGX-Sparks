# SPDX-License-Identifier: Apache-2.0
"""EXL3/MCG trellis quantization for GLM-5.3-Flash routed experts.

Checkpoint ABI (brandonmusic/GLM-5.3-Flash-tr3-4bpw):
  quant_method=exl3, codebook=mcg, scope=glm53_routed_experts_only
  per expert matrix: trellis (int16) + suh/svh (fp16) + mcg (int32 marker)

Non-routed tensors stay native (UnquantizedLinearMethod) unless the pack
config carries a ``non_routed_exl3`` block, in which case the declared
dense linears run Exl3LinearMethod ([dense-exl3], Alexbob0/MIT port).
Experts never expand to a persistent BF16 weight; LinearEXL3 /
exllamav3_ext runs the trellis GEMM. TP=2 shards gate/up column-wise and down row-wise; the MoE
runner all-reduces the combined output.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger(__name__)

EXLLAMAV3_COMMIT = "c5d9c657966ffeeaa9353f0cc899f18629da4a13"
EXLLAMAV3_VERSION = "0.0.43"
MCG_MULTIPLIER = 0xCBAC1FED
MCG_MARKER_SIGNED_INT32 = -877912083
# [dense-exl3] mul1 codebook marker (0x83DCD12D as signed int32) — the
# turboderp dense quants ship mul1 tensors; the routed experts stay mcg.
MUL1_MARKER_SIGNED_INT32 = -2082680531
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg")
SWIGLU_LIMIT_DEFAULT = 10.0
# Default fused-kernel temp rows/expert. 1024 covers MNBT=1024 in one launch
# but measured slower than 128+fallback (P2b). Override with EXL3_TEMP_ROWS_FUSED.
TEMP_ROWS_FUSED = 128
MOE_ACT_SILU = 0
# Shared fused scratch: decode is sequential across layers.
_FUSED_TEMP_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_FAT_SCRATCH_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_FAT_COUNT_CACHE: dict[tuple, tuple[torch.Tensor, torch.cuda.Stream]] = {}
# Grouped (E3) fat-expert scratch: fat-row activation buffers, grown once.
_FAT_GROUPED_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_FAT_GROUPED_BYTES: dict[tuple, int] = {}
_FAT_BUCKET_EDGES = (16, 32, 64, 128, 256, 512, 1024, 2048)
_FAT_STATS: dict[str, Any] = {
    "layers": 0,
    "fat_layers": 0,
    "fat_experts": 0,
    "max_rows": 0,
    "sum_max_rows": 0,
    "hist": [0] * (len(_FAT_BUCKET_EDGES) + 1),
}
_FAT_TIERS = ("grouped", "kernel", "batched", "sorted", "legacy")
# Machine-checkable E2/E3 (fat-expert) diagnostics. Load-time fields mirror the
# most recent weight load; counters are monotonic per process and are the
# ground truth for what actually ran. The key set is contract — extend it only
# together with a bump of EXL3_FAT_DIAG_SCHEMA.
# Schema 2 adds the E3 grouped tier: sym_fat_moe, grouped_calls,
# grouped_scratch_bytes, grouped_eligible (and "grouped" in fallback_calls).
EXL3_FAT_DIAG_SCHEMA = 2
EXL3_FAT_DIAG_KEYS = (
    "schema",
    "configured_tier",
    "effective_tier",
    "tier_reason",
    "shared_suh",
    "shared_suh_layers",
    "moe_layers_loaded",
    "sym_exl3_moe",
    "sym_fat_gemm",
    "sym_fat_gemm_scatter",
    "sym_fat_moe",
    "grouped_calls",
    "grouped_scratch_bytes",
    "grouped_eligible",
    "cap_major",
    "cap_minor",
    "cap_ok",
    "tp_rank",
    "tp_size",
    "fused_temps_allocs",
    "fused_temps_bytes",
    "fat_scratch_allocs",
    "fat_scratch_bytes",
    "fat_scratch_peak_bytes",
    "prefill_layer_calls",
    "thin_calls",
    "row_tile_calls",
    "fallback_calls",
    "fallback_reasons",
    "fat_expert_runs",
    "direct_calls",
    "scatter_calls",
    "fat_stat_layers",
    "fat_layers",
    "fat_expert_slots",
    "max_rows",
)
_EXL3_FAT_DIAG: dict[str, Any] = {
    "schema": EXL3_FAT_DIAG_SCHEMA,
    "configured_tier": "legacy",
    "effective_tier": "legacy",
    "tier_reason": "unresolved",
    "shared_suh": False,
    "shared_suh_layers": 0,
    "moe_layers_loaded": 0,
    "sym_exl3_moe": False,
    "sym_fat_gemm": False,
    "sym_fat_gemm_scatter": False,
    "sym_fat_moe": False,
    "grouped_calls": 0,
    "grouped_scratch_bytes": 0,
    "grouped_eligible": False,
    "cap_major": -1,
    "cap_minor": -1,
    "cap_ok": False,
    "tp_rank": -1,
    "tp_size": 1,
    "fused_temps_allocs": 0,
    "fused_temps_bytes": 0,
    "fat_scratch_allocs": 0,
    "fat_scratch_bytes": 0,
    "fat_scratch_peak_bytes": 0,
    "prefill_layer_calls": 0,
    "thin_calls": 0,
    "row_tile_calls": 0,
    "fallback_calls": {tier: 0 for tier in _FAT_TIERS},
    "fallback_reasons": {},
    "fat_expert_runs": 0,
    "direct_calls": 0,
    "scatter_calls": 0,
}
_FAT_SCRATCH_BYTES: dict[tuple, int] = {}
_exl3_fat_tier_logged = False


def fused_moe_row_tile_enabled() -> bool:
    """GPU row tiles instead of LinearEXL3 fallback. Prefill-only; decode stays one launch.

    Measured slower than the 128-row fallback at MNBT=1024 (8 full-grid launches).
    Default off; keep for MNBT > temp rows if a later bump still overflows.
    """
    return os.environ.get("EXL3_MOE_ROW_TILE", "0") != "0"


def temp_rows_fused() -> int:
    raw = os.environ.get("EXL3_TEMP_ROWS_FUSED", "").strip()
    if not raw:
        return int(TEMP_ROWS_FUSED)
    return max(1, int(raw))

def sorted_fat_fallback_enabled() -> bool:
    """Use the existing expert-sorted buffers for oversized prefill experts."""
    return os.environ.get("EXL3_FAT_SORTED", "0") != "0"

def batched_fat_fallback_enabled() -> bool:
    """Enable E1 batched fat experts; implies expert-sorted routing."""
    return os.environ.get("EXL3_FAT_BATCHED", "0") != "0"

def fat_kernel_enabled() -> bool:
    """Enable the E2 fat kernel; implies E1 batching and sorted routing."""
    return os.environ.get("EXL3_FAT_KERNEL", "0") != "0"


def grouped_fat_enabled() -> bool:
    """Enable the E3 grouped fat-expert kernels (experimental, default OFF).

    One gather + one gate/up + one down launch cover every fat expert of a
    layer from device-side segment tables: no per-expert launches and no
    host synchronization on the routing counts. Needs the exl3_fat_moe
    kernels (exllamav3_ext built with exl3_fat_moe.cu, or the additive
    exl3_fat_moe_ext module). EXL3_FAT_GROUPED=0 (default) leaves the E2
    path and its cap untouched.
    """
    return os.environ.get("EXL3_FAT_GROUPED", "0") != "0"


def fat_expert_log_enabled() -> bool:
    """Per-step routing histogram. It costs one host sync per MoE layer under
    the grouped tier (the grouped path has none of its own), so it defaults
    off there."""
    default = "0" if grouped_fat_enabled() else "1"
    return os.environ.get("EXL3_FAT_EXPERT_LOG", default) != "0"

def configured_fat_tier() -> str:
    """Highest fat tier the env requests: grouped > kernel > batched > sorted > legacy."""
    if grouped_fat_enabled():
        return "grouped"
    if fat_kernel_enabled():
        return "kernel"
    if batched_fat_fallback_enabled():
        return "batched"
    if sorted_fat_fallback_enabled():
        return "sorted"
    return "legacy"


def exl3_fat_symbols() -> tuple[bool, bool, bool]:
    """(exl3_moe, exl3_fat_gemm, exl3_fat_gemm_scatter) availability."""
    try:
        ext = load_exllamav3_ext()
    except Exception:
        return False, False, False
    return (
        hasattr(ext, "exl3_moe"),
        hasattr(ext, "exl3_fat_gemm"),
        hasattr(ext, "exl3_fat_gemm_scatter"),
    )


EXL3_FAT_MOE_SYMBOLS = (
    "exl3_fat_moe_gather",
    "exl3_fat_moe_gateup",
    "exl3_fat_moe_down",
    "exl3_fat_moe_tile_rows_gateup",
    "exl3_fat_moe_tile_rows_down",
)
# Grouped kernels: 16 B vector atomics (sm_90+); the image builds sm_121a.
EXL3_FAT_MOE_MIN_CAPABILITY = (9, 0)
_FAT_MOE_EXT_CACHE: list = []


def load_fat_moe_ext():
    """Module carrying the E3 kernels, or None.

    Two supported sources: exllamav3_ext itself (bindings patched at the
    full image build by patch_exl3_fat_kernel.py) or the additive
    `exl3_fat_moe_ext` module (layered candidate image). Resolved once.
    """
    if _FAT_MOE_EXT_CACHE:
        return _FAT_MOE_EXT_CACHE[0]
    found = None
    try:
        ext = load_exllamav3_ext()
        if all(hasattr(ext, name) for name in EXL3_FAT_MOE_SYMBOLS):
            found = ext
    except Exception:
        found = None
    if found is None:
        try:
            import exl3_fat_moe_ext  # noqa: F401

            if all(hasattr(exl3_fat_moe_ext, name) for name in EXL3_FAT_MOE_SYMBOLS):
                found = exl3_fat_moe_ext
        except Exception:
            found = None
    _FAT_MOE_EXT_CACHE.append(found)
    return found


def exl3_fat_moe_symbols() -> bool:
    """True when every E3 grouped kernel entry point is importable."""
    return load_fat_moe_ext() is not None


def grouped_fat_eligibility(layer: torch.nn.Module) -> tuple[bool, str]:
    """Load-time checkpoint/device eligibility for the K4/MCG grouped kernels.

    The grouped pointer-table interface does not carry K/mcg/mul1 like the E2
    GEMM interface, so those invariants are checked here, once per layer,
    before the tier is resolved: 4-bit trellis (64 int16 words per tile),
    MCG codebook without mul1, shared gate/up SUH, tile-friendly dimensions
    (hidden % 256 for the 2x128 down tile and the gather's 128-wide blocks,
    intermediate % 128 for the gate/up tile), a device capability with 16 B
    vector atomics, and every packed tensor on one CUDA device.
    """
    bits = int(getattr(layer, "_exl3_bits", 0) or 0)
    if bits != 4:
        return False, f"bits_{bits}"
    k_words = int(getattr(layer, "_exl3_k_words", 0) or 0)
    if k_words != 64:
        return False, f"k_words_{k_words}"
    inners = getattr(layer, "_exl3_inners", None) or []
    if not inners:
        return False, "no_inners"
    for pack in inners:
        for which in ("gate", "up", "down"):
            inner = pack[which]
            if int(getattr(inner, "K", 0)) != 4:
                return False, f"K_{getattr(inner, 'K', '?')}"
            if not bool(getattr(inner, "mcg", False)):
                return False, "not_mcg"
            if bool(getattr(inner, "mul1", False)):
                return False, "mul1"
    if not bool(getattr(layer, "_exl3_shared_w13_suh", False)):
        return False, "shared_suh_absent"
    hidden = int(getattr(layer, "_exl3_hidden_size", 0) or 0)
    inter = int(getattr(layer, "_exl3_intermediate_local", 0) or 0)
    if hidden <= 0 or hidden % 256:
        return False, f"hidden_{hidden}"
    if inter <= 0 or inter % 128:
        return False, f"intermediate_{inter}"
    device = layer.w13_trellis.device
    if device.type != "cuda":
        return False, f"device_{device.type}"
    for name in ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh"):
        if getattr(layer, name).device != device:
            return False, f"device_mismatch_{name}"
    cap = exl3_device_capability()
    if cap < EXL3_FAT_MOE_MIN_CAPABILITY:
        return False, f"capability_{cap[0]}.{cap[1]}"
    return True, "eligible"


def exl3_device_capability() -> tuple[int, int]:
    """CUDA capability of the current device; (-1, -1) without a GPU."""
    if not torch.cuda.is_available():
        return -1, -1
    try:
        return tuple(int(v) for v in torch.cuda.get_device_capability())
    except Exception:
        return -1, -1


def _exl3_tp_rank_size() -> tuple[int, int]:
    """TP identity so each rank's diag line is attributable; -1 outside vLLM."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return (
            int(get_tensor_model_parallel_rank()),
            int(get_tensor_model_parallel_world_size()),
        )
    except Exception:
        return -1, 1


def resolve_exl3_fat_tier(
    shared_suh: bool,
    symbols: tuple[bool, bool, bool] | None = None,
    fat_moe_symbols: bool | None = None,
    grouped_eligible: tuple[bool, str] | None = None,
) -> tuple[str, str]:
    """Map the configured fat tier onto what this image + checkpoint can run.

    Order matters: the checkpoint cap comes first. E1 batched, the E2 kernel
    and the E3 grouped kernels run gate+up behind a single input Hadamard
    (gate.suh), so a checkpoint without shared SUH caps the tier at sorted —
    a legitimate lower tier that needs no fat symbols, whatever the image.
    Only when the kernel would actually run does a missing symbol become an
    image/flag mismatch, failing closed here at load instead of mid-prefill.

    Grouped: a request without the E3 symbols fails closed. A checkpoint or
    device the grouped kernels cannot run (K4/MCG, dimensions, capability)
    deliberately falls back to the E2 kernel tier with the reason recorded,
    and that fallback is itself subject to the E2 symbol check.
    """
    configured = configured_fat_tier()
    if configured == "legacy":
        return configured, "none_requested"
    if not shared_suh and configured in ("grouped", "kernel", "batched"):
        return "sorted", "shared_suh_absent"
    reason_prefix = ""
    if configured == "grouped":
        if fat_moe_symbols is None:
            fat_moe_symbols = exl3_fat_moe_symbols()
        if not fat_moe_symbols:
            raise RuntimeError(
                "EXL3_FAT_GROUPED=1 requires "
                + "/".join(EXL3_FAT_MOE_SYMBOLS)
                + " (exllamav3_ext or exl3_fat_moe_ext); this image was built "
                "without exl3_fat_moe.cu — set EXL3_FAT_GROUPED=0 (E2 kernel) "
                "or build the E3 candidate image"
            )
        if grouped_eligible is None:
            grouped_eligible = (True, "eligible")
        if grouped_eligible[0]:
            return configured, "grouped_ok"
        # Ineligible for the grouped kernels: fall back to E2 deliberately.
        configured = "kernel"
        reason_prefix = f"grouped_ineligible_{grouped_eligible[1]}:"
    if symbols is None:
        symbols = exl3_fat_symbols()
    if configured == "kernel":
        missing = [
            name
            for name, present in zip(
                ("exl3_fat_gemm", "exl3_fat_gemm_scatter"), symbols[1:]
            )
            if not present
        ]
        if missing:
            raise RuntimeError(
                "EXL3_FAT_KERNEL=1 requires exllamav3_ext."
                + "/".join(missing)
                + "; this image was built without the fat kernel — unset "
                "EXL3_FAT_KERNEL or serve an E2 image"
            )
    return configured, f"{reason_prefix}{configured}_ok"


def exl3_fat_diag() -> dict[str, Any]:
    """Snapshot of the E2 diagnostics; the key set is EXL3_FAT_DIAG_KEYS."""
    diag = dict(_EXL3_FAT_DIAG)
    diag["fallback_calls"] = dict(_EXL3_FAT_DIAG["fallback_calls"])
    diag["fallback_reasons"] = dict(_EXL3_FAT_DIAG["fallback_reasons"])
    diag.update(
        fat_stat_layers=_FAT_STATS["layers"],
        fat_layers=_FAT_STATS["fat_layers"],
        fat_expert_slots=_FAT_STATS["fat_experts"],
        max_rows=_FAT_STATS["max_rows"],
    )
    return diag


def _exl3_fat_diag_line() -> str:
    d = exl3_fat_diag()
    parts = [
        f"schema={d['schema']}",
        f"configured_tier={d['configured_tier']}",
        f"effective_tier={d['effective_tier']}",
        f"tier_reason={d['tier_reason']}",
        f"shared_suh={int(d['shared_suh'])}",
        f"shared_suh_layers={d['shared_suh_layers']}/{d['moe_layers_loaded']}",
        f"sym_exl3_moe={int(d['sym_exl3_moe'])}",
        f"sym_fat_gemm={int(d['sym_fat_gemm'])}",
        f"sym_fat_gemm_scatter={int(d['sym_fat_gemm_scatter'])}",
        f"sym_fat_moe={int(d['sym_fat_moe'])}",
        f"grouped_eligible={int(d['grouped_eligible'])}",
        f"grouped_calls={d['grouped_calls']}",
        f"grouped_scratch_bytes={d['grouped_scratch_bytes']}",
        f"cap={d['cap_major']}.{d['cap_minor']}",
        f"cap_ok={int(d['cap_ok'])}",
        f"tp_rank={d['tp_rank']} tp_size={d['tp_size']}",
        f"prefill_layer_calls={d['prefill_layer_calls']}",
        f"thin_calls={d['thin_calls']}",
        f"row_tile_calls={d['row_tile_calls']}",
        "fallback_calls="
        + ",".join(f"{t}={d['fallback_calls'][t]}" for t in _FAT_TIERS),
        "fallback_reasons="
        + (
            ",".join(f"{r}={n}" for r, n in sorted(d["fallback_reasons"].items()))
            or "none"
        ),
        f"fat_expert_runs={d['fat_expert_runs']}",
        f"direct_calls={d['direct_calls']}",
        f"scatter_calls={d['scatter_calls']}",
        f"fat_layers={d['fat_layers']}",
        f"fat_expert_slots={d['fat_expert_slots']}",
        f"max_rows={d['max_rows']}",
        f"fused_temps_allocs={d['fused_temps_allocs']}",
        f"fused_temps_bytes={d['fused_temps_bytes']}",
        f"fat_scratch_allocs={d['fat_scratch_allocs']}",
        f"fat_scratch_bytes={d['fat_scratch_bytes']}",
        f"fat_scratch_peak_bytes={d['fat_scratch_peak_bytes']}",
    ]
    return " ".join(parts)


def _record_exl3_fat_reason(reason: str) -> None:
    reasons = _EXL3_FAT_DIAG["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def _record_exl3_fat_tier(layer: torch.nn.Module, tier: str, reason: str) -> None:
    _EXL3_FAT_DIAG["fallback_calls"][tier] += 1
    _record_exl3_fat_reason(reason)
    layer._exl3_last_fat_fallback = tier
    layer._exl3_last_fat_reason = reason


def _record_exl3_fat_resolution(layer: torch.nn.Module) -> None:
    """Resolve the E2 tier once per MoE layer at weight load and log once.

    Per-layer truth lands on layer._exl3_fat_effective_tier; the module state
    mirrors the most recent load. A resolution that changes between layers of
    one model is a checkpoint property worth a loud line, not a silent one.
    """
    global _exl3_fat_tier_logged
    shared_suh = bool(getattr(layer, "_exl3_shared_w13_suh", False))
    grouped_eligible = grouped_fat_eligibility(layer)
    layer._exl3_grouped_eligible = grouped_eligible
    effective_tier, tier_reason = resolve_exl3_fat_tier(
        shared_suh, grouped_eligible=grouped_eligible
    )
    layer._exl3_fat_effective_tier = effective_tier
    layer._exl3_fat_tier_reason = tier_reason

    diag = _EXL3_FAT_DIAG
    sym_moe, sym_gemm, sym_scatter = exl3_fat_symbols()
    cap_major, cap_minor = exl3_device_capability()
    diag["moe_layers_loaded"] += 1
    if shared_suh:
        diag["shared_suh_layers"] += 1
    diag["shared_suh"] = diag["shared_suh_layers"] == diag["moe_layers_loaded"]
    diag["configured_tier"] = configured_fat_tier()
    diag["sym_exl3_moe"] = sym_moe
    diag["sym_fat_gemm"] = sym_gemm
    diag["sym_fat_gemm_scatter"] = sym_scatter
    diag["sym_fat_moe"] = exl3_fat_moe_symbols()
    diag["grouped_eligible"] = bool(grouped_eligible[0])
    diag["cap_major"] = cap_major
    diag["cap_minor"] = cap_minor
    # LinearEXL3 (and the fat GEMM built on it) needs >= Ampere; GB10 is SM121.
    diag["cap_ok"] = (cap_major, cap_minor) >= (8, 0)
    diag["tp_rank"], diag["tp_size"] = _exl3_tp_rank_size()

    if diag["tier_reason"] == "unresolved":
        diag["effective_tier"] = effective_tier
        diag["tier_reason"] = tier_reason
    elif diag["effective_tier"] != effective_tier:
        logger.warning(
            "exl3 e2 diag tier changed %s -> %s (%s): %s",
            diag["effective_tier"],
            effective_tier,
            tier_reason,
            _exl3_fat_diag_line(),
        )
        diag["effective_tier"] = effective_tier
        diag["tier_reason"] = tier_reason
    if not _exl3_fat_tier_logged:
        _exl3_fat_tier_logged = True
        if (
            diag["effective_tier"] != diag["configured_tier"]
            and diag["configured_tier"] != "legacy"
        ):
            logger.warning("exl3 e2 diag degraded %s", _exl3_fat_diag_line())
        else:
            logger.info("exl3 e2 diag %s", _exl3_fat_diag_line())


def reset_exl3_fat_diag_counters() -> None:
    """Zero the E2 runtime counters; load-time fields and live bytes stay.

    Scratch current/peak restart from the resident cache so a windowed read
    (e.g. exactly one cold request) still reports honest byte counts.
    """
    diag = _EXL3_FAT_DIAG
    for key in (
        "prefill_layer_calls",
        "thin_calls",
        "row_tile_calls",
        "fat_expert_runs",
        "direct_calls",
        "scatter_calls",
        "fat_scratch_allocs",
        "grouped_calls",
    ):
        diag[key] = 0
    diag["fallback_calls"] = {tier: 0 for tier in _FAT_TIERS}
    diag["fallback_reasons"] = {}
    diag["fat_scratch_bytes"] = sum(_FAT_SCRATCH_BYTES.values())
    diag["fat_scratch_peak_bytes"] = diag["fat_scratch_bytes"]
    diag["grouped_scratch_bytes"] = sum(_FAT_GROUPED_BYTES.values())


def reset_exl3_fat_expert_stats() -> None:
    _FAT_STATS["layers"] = 0
    _FAT_STATS["fat_layers"] = 0
    _FAT_STATS["fat_experts"] = 0
    _FAT_STATS["max_rows"] = 0
    _FAT_STATS["sum_max_rows"] = 0
    _FAT_STATS["hist"] = [0] * (len(_FAT_BUCKET_EDGES) + 1)


def _fat_bucket(n: int) -> int:
    for i, edge in enumerate(_FAT_BUCKET_EDGES):
        if n <= edge:
            return i
    return len(_FAT_BUCKET_EDGES)


def record_exl3_fat_expert_stats(
    counts: torch.Tensor,
    *,
    max_rows: int | None = None,
    counts_host: list[int] | None = None,
) -> dict[str, Any]:
    """Prefill-only routing stats. Reuse an existing host copy when available."""
    if counts_host is None:
        if max_rows is None:
            max_rows = int(counts.max().item())
        n_fat = int((counts > temp_rows_fused()).sum().item())
    else:
        if max_rows is None:
            max_rows = max(counts_host, default=0)
        cap = temp_rows_fused()
        n_fat = sum(n > cap for n in counts_host)
    st = _FAT_STATS
    st["layers"] += 1
    st["sum_max_rows"] += max_rows
    st["hist"][_fat_bucket(max_rows)] += 1
    if max_rows > st["max_rows"]:
        st["max_rows"] = max_rows
    if n_fat:
        st["fat_layers"] += 1
        st["fat_experts"] += n_fat
    # 42 routed-MoE layers per engine step (MoE from layer 3 of 45).
    if st["layers"] % 42 == 0:
        avg = st["sum_max_rows"] / st["layers"]
        le128 = sum(st["hist"][:4])
        gt128 = sum(st["hist"][4:])
        logger.info(
            "exl3 fat-expert P0: layers=%d fat_layers=%d (%.1f%%) fat_expert_slots=%d "
            "max_rows=%d avg_max_rows=%.1f hist_le128=%d hist_gt128=%d hist=%s",
            st["layers"],
            st["fat_layers"],
            100.0 * st["fat_layers"] / st["layers"],
            st["fat_experts"],
            st["max_rows"],
            avg,
            le128,
            gt128,
            st["hist"],
        )
        logger.info("exl3 e2 diag %s", _exl3_fat_diag_line())
    return {
        "max_rows": max_rows,
        "n_fat": n_fat,
        "layers": st["layers"],
        "fat_layers": st["fat_layers"],
    }


def _narrow_tp(tensor: torch.Tensor, dim: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    if tp_size <= 1:
        return tensor
    size = int(tensor.shape[dim])
    if size % tp_size:
        raise ValueError(
            f"EXL3 TP shard: dim {dim} size {size} is not divisible by tp={tp_size}"
        )
    chunk = size // tp_size
    return tensor.narrow(dim, chunk * tp_rank, chunk).contiguous()


def shard_exl3_col(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Gate/up: trellis dim 1 and svh dim 0 are column-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 1, tp_rank, tp_size)
    if suffix == "svh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def shard_exl3_row(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Down: trellis dim 0 and suh dim 0 are row-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    if suffix == "suh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def _install_exllamav3_namespace() -> None:
    """Load LinearEXL3 without running exllamav3/__init__.py (FlashAttention)."""
    if "exllamav3.modules.quant.exl3" in sys.modules:
        return
    import exllamav3_ext  # noqa: F401  — compiled extension must exist

    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("exllamav3 package is not installed in this image")
    package_root = Path(list(spec.submodule_search_locations)[0])

    # Stub only packages whose __init__.py pulls FlashAttention / serving extras.
    # Leave .ext, .util, and .modules.quant as real modules so LinearEXL3 loads.
    for name, path in (
        ("exllamav3", package_root),
        ("exllamav3.modules", package_root / "modules"),
        ("exllamav3.model", package_root / "model"),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__file__ = str(path / "__init__.py")
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module

    if "exllamav3.model.config" not in sys.modules:
        config = types.ModuleType("exllamav3.model.config")
        config.__file__ = str(package_root / "model/config.py")
        config.__package__ = "exllamav3.model"
        config.Config = type("Config", (), {})
        sys.modules[config.__name__] = config


def load_linear_exl3_cls():
    _install_exllamav3_namespace()
    return importlib.import_module("exllamav3.modules.quant.exl3").LinearEXL3

def make_linear_exl3(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: torch.Tensor | None = None,
    mul1: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype = torch.float16,
):
    """Build a LinearEXL3 over already-sharded packed tensors. No BF16 expand.

    [dense-exl3] mcg became optional and mul1 was added: the dense overlay
    tensors carry the mul1 codebook while the routed experts stay mcg. Exactly
    one of the two must be non-None (LinearEXL3 asserts internally).
    """
    cls = load_linear_exl3_cls()
    return cls(
        config=None,
        in_features=int(suh.numel()),
        out_features=int(svh.numel()),
        trellis=trellis.contiguous(),
        suh=suh.contiguous(),
        svh=svh.contiguous(),
        mcg=mcg.contiguous() if mcg is not None else None,
        mul1=mul1.contiguous() if mul1 is not None else None,
        out_dtype=out_dtype,
        transformers_fix=True,
    )


def execute_exl3_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Real EXL3 expert GEMM entry (LinearEXL3 / exllamav3_ext)."""
    inner = make_linear_exl3(trellis, suh, svh, mcg, out_dtype=torch.float16)
    return inner.forward(x.contiguous().half(), {}, out_dtype=out_dtype)


def fused_moe_enabled() -> bool:
    return os.environ.get("EXL3_FUSED_MOE", "1") != "0"


def load_exllamav3_ext():
    import exllamav3_ext

    return exllamav3_ext


def _exl3_moe_accepts_num_active(fn) -> bool:
    try:
        import inspect

        if "num_active" in inspect.signature(fn).parameters:
            return True
    except (TypeError, ValueError):
        pass
    doc = getattr(fn, "__doc__", None) or ""
    return "num_active" in doc or "arg29" in doc or doc.count("arg") >= 30


def pin_exl3_expert_map(
    layer: torch.nn.Module, device: torch.device
) -> torch.Tensor | None:
    """Move expert_map onto `device` once. CUDA graph capture forbids a CPU→GPU copy."""
    emap = getattr(layer, "expert_map", None)
    if emap is None:
        return None
    if emap.device != device or emap.dtype != torch.long:
        layer.expert_map = emap.to(device=device, dtype=torch.long)
    return layer.expert_map


def map_topk_to_local(
    ids: torch.Tensor,
    n_local: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """ids (T, K) global expert ids → local ids, invalid/non-local → n_local sentinel.

    `expert_map` must already live on `ids.device` (see pin_exl3_expert_map).
    """
    flat = ids.reshape(-1)
    if expert_map is None:
        invalid = (flat < 0) | (flat >= n_local)
        return torch.where(invalid, flat.new_full(flat.shape, n_local), flat)
    if expert_map.device != flat.device or expert_map.dtype != torch.long:
        raise RuntimeError(
            "EXL3 expert_map is not pinned to the hidden-state device; "
            "call pin_exl3_expert_map before fused apply (CUDA graphs forbid the copy)"
        )
    n_global = int(expert_map.numel())
    safe = flat.clamp(min=0, max=max(n_global - 1, 0))
    mapped = expert_map[safe] if n_global else flat.new_full(flat.shape, n_local)
    invalid = (flat < 0) | (flat >= n_global) | (mapped < 0) | (mapped >= n_local)
    return torch.where(invalid, flat.new_full(flat.shape, n_local), mapped)


def apply_exl3_python_loop(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
    *,
    only_experts: set[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unique-expert LinearEXL3 loop. `only_experts` is local ids (fat-expert fallback)."""
    tokens, hidden = x2d.shape
    if out is None:
        out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    unique = torch.unique(ids)
    for raw in unique.tolist():
        e_raw = int(raw)
        if e_raw < 0:
            continue
        e = e_raw
        if expert_map is not None:
            mapped = int(expert_map[e].item()) if expert_map.numel() > e else e
            if mapped < 0:
                continue
            e = mapped
        if e >= len(inners):
            continue
        if only_experts is not None and e not in only_experts:
            continue
        token_idx, k_pos = (ids == int(raw)).nonzero(as_tuple=True)
        h = x2d.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        up = pack["up"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        down = pack["down"].forward(act.contiguous().half(), {}, out_dtype=torch.float32)
        scale = weights[token_idx, k_pos].unsqueeze(-1).to(dtype=torch.float32)
        out.index_add_(0, token_idx, down * scale)
    return out

def apply_exl3_sorted_fat(
    xh: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    counts_host: list[int],
    inners: list[dict[str, Any]],
    limit: float,
    cap: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run oversized experts from contiguous slices of the existing sort.

    One `counts.tolist()` in the caller replaces the legacy per-expert
    `unique.tolist()`, expert-map `.item()`, and `(ids == expert).nonzero()`
    synchronizations. Slices are views; only the LinearEXL3 inputs and outputs
    allocate, as they do in the legacy fallback.
    """
    offset = 0
    for e, n_rows in enumerate(counts_host):
        start = offset
        offset += n_rows
        if n_rows <= cap:
            continue
        _EXL3_FAT_DIAG["fat_expert_runs"] += 1
        token_idx = token_sorted[start:offset]
        h = xh.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h, {}, out_dtype=torch.float32)
        up = pack["up"].forward(h, {}, out_dtype=torch.float32)
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        down = pack["down"].forward(
            act.contiguous().half(), {}, out_dtype=torch.float32
        )
        scale = weight_sorted[start:offset].unsqueeze(-1).to(dtype=torch.float32)
        out.index_add_(0, token_idx, down * scale)
    return out


def _fat_scratch(
    device: torch.device,
    rows: int,
    gate: Any,
) -> dict[str, torch.Tensor]:
    """Return shared prefill scratch, growing once if the configured chunk grows."""
    hidden = int(gate.in_features)
    intermediate = int(gate.out_features)
    configured = int(
        os.environ.get(
            "EXL3_FAT_SCRATCH_ROWS",
            os.environ.get("MAX_NUM_BATCHED_TOKENS", "0"),
        )
        or 0
    )
    needed = max(256, rows, configured)
    capacity = 1 << (needed - 1).bit_length()
    key = (
        str(device),
        hidden,
        intermediate,
        int(gate.K),
        int(gate.trellis.shape[-1]),
    )
    scratch = _FAT_SCRATCH_CACHE.get(key)
    if scratch is not None and int(scratch["h"].shape[0]) >= rows:
        return scratch

    in_tiles, out_tiles, k_words = map(int, gate.trellis.shape)
    scratch = {
        "packed13": torch.empty(
            (in_tiles, 2 * out_tiles, k_words),
            dtype=torch.int16,
            device=device,
        ),
        "svh13": torch.empty(
            2 * intermediate, dtype=torch.float16, device=device
        ),
        "w13": torch.empty(
            (hidden, 2 * intermediate), dtype=torch.float16, device=device
        ),
        "w2": torch.empty(
            (intermediate, hidden), dtype=torch.float16, device=device
        ),
        "h": torch.empty(
            (capacity, hidden), dtype=torch.float16, device=device
        ),
        "h13": torch.empty(
            (capacity, hidden), dtype=torch.float16, device=device
        ),
        "gate_up": torch.empty(
            (capacity, 2 * intermediate), dtype=torch.float32, device=device
        ),
        "act": torch.empty(
            (capacity, intermediate), dtype=torch.float32, device=device
        ),
        "act_h": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
        "h2": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
        "down": torch.empty(
            (capacity, hidden), dtype=torch.float32, device=device
        ),
    }
    _FAT_SCRATCH_CACHE[key] = scratch
    _FAT_SCRATCH_BYTES[key] = sum(
        t.numel() * t.element_size() for t in scratch.values()
    )
    diag = _EXL3_FAT_DIAG
    diag["fat_scratch_allocs"] += 1
    diag["fat_scratch_bytes"] = sum(_FAT_SCRATCH_BYTES.values())
    if diag["fat_scratch_bytes"] > diag["fat_scratch_peak_bytes"]:
        diag["fat_scratch_peak_bytes"] = diag["fat_scratch_bytes"]
    return scratch


def _stage_counts_to_host(
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.cuda.Stream]:
    """Copy routing counts on a side stream before launching thin experts."""
    key = (str(counts.device), int(counts.numel()))
    cached = _FAT_COUNT_CACHE.get(key)
    if cached is None:
        host = torch.empty(
            int(counts.numel()), dtype=counts.dtype, device="cpu", pin_memory=True
        )
        stream = torch.cuda.Stream(device=counts.device)
        cached = (host, stream)
        _FAT_COUNT_CACHE[key] = cached
    host, stream = cached
    current = torch.cuda.current_stream(counts.device)
    with torch.cuda.stream(stream):
        stream.wait_stream(current)
        host.copy_(counts, non_blocking=True)
    return host, stream


def apply_exl3_batched_fat(
    xh: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    counts_host: list[int],
    inners: list[dict[str, Any]],
    limit: float,
    cap: int,
    out: torch.Tensor,
    use_kernel: bool = False,
) -> torch.Tensor:
    """Run fat experts with persistent buffers and optional direct trellis GEMM."""
    ext = load_exllamav3_ext()
    offset = 0
    for e, n_rows in enumerate(counts_host):
        start = offset
        offset += n_rows
        if n_rows <= cap:
            continue
        _EXL3_FAT_DIAG["fat_expert_runs"] += 1

        token_idx = token_sorted[start:offset]
        gate = inners[e]["gate"]
        up = inners[e]["up"]
        down = inners[e]["down"]
        scratch = _fat_scratch(xh.device, n_rows, gate)
        intermediate = int(gate.out_features)

        h = scratch["h"][:n_rows]
        h13 = scratch["h13"][:n_rows]
        torch.index_select(xh, 0, token_idx, out=h)
        ext.had_r_128(h, h13, gate.suh, None, 1.0)

        packed13 = scratch["packed13"]
        out_tiles = int(gate.trellis.shape[1])
        packed13[:, :out_tiles].copy_(gate.trellis)
        packed13[:, out_tiles:].copy_(up.trellis)
        gate_up = scratch["gate_up"][:n_rows]
        svh13 = scratch["svh13"]
        svh13[:intermediate].copy_(gate.svh)
        svh13[intermediate:].copy_(up.svh)
        if use_kernel:
            if not hasattr(ext, "exl3_fat_gemm"):
                raise RuntimeError(
                    "EXL3_FAT_KERNEL=1 requires exllamav3_ext.exl3_fat_gemm"
                )
            ext.exl3_fat_gemm(
                h13, packed13, gate_up, svh13, gate.K, gate.mcg, gate.mul1
            )
            _EXL3_FAT_DIAG["direct_calls"] += 1
        else:
            w13 = scratch["w13"]
            ext.reconstruct(w13, packed13, gate.K, gate.mcg, gate.mul1)
            ext.hgemm(h13, w13, gate_up)
            ext.had_r_128(gate_up, gate_up, None, svh13, 1.0)

        gate_out = gate_up[:, :intermediate]
        up_out = gate_up[:, intermediate:]
        gate_out.clamp_(max=limit)
        up_out.clamp_(min=-limit, max=limit)
        act = scratch["act"][:n_rows]
        torch.sigmoid(gate_out, out=act)
        act.mul_(gate_out).mul_(up_out)
        act_h = scratch["act_h"][:n_rows]
        act_h.copy_(act)

        h2 = scratch["h2"][:n_rows]
        ext.had_r_128(act_h, h2, down.suh, None, 1.0)
        if use_kernel:
            if not hasattr(ext, "exl3_fat_gemm_scatter"):
                raise RuntimeError(
                    "EXL3_FAT_KERNEL=1 requires "
                    "exllamav3_ext.exl3_fat_gemm_scatter"
                )
            ext.exl3_fat_gemm_scatter(
                h2,
                down.trellis,
                out,
                down.svh,
                token_idx,
                weight_sorted[start:offset],
                down.K,
                down.mcg,
                down.mul1,
            )
            _EXL3_FAT_DIAG["scatter_calls"] += 1
        else:
            w2 = scratch["w2"]
            ext.reconstruct(w2, down.trellis, down.K, down.mcg, down.mul1)
            down_out = scratch["down"][:n_rows]
            ext.hgemm(h2, w2, down_out)
            ext.had_r_128(down_out, down_out, None, down.svh, 1.0)
            down_out.mul_(weight_sorted[start:offset].unsqueeze(-1))
            out.index_add_(0, token_idx, down_out)
    return out


def _grouped_scratch(
    device: torch.device, rows: int, hidden: int, intermediate: int
) -> dict[str, torch.Tensor]:
    """Fat-row activation buffers for the grouped tier, grown once.

    Capacity covers every routed slot of the configured prefill chunk
    (MAX_NUM_BATCHED_TOKENS x top-k, EXL3_FAT_GROUPED_TOPK, default 8) so
    steady-state prefill never reallocates; a larger request grows it once
    more (outside CUDA graph capture, where growth would be illegal).
    Persistent: h13 [rows, hidden] fp16 + h2 [rows, intermediate] fp16.
    """
    configured = int(
        os.environ.get(
            "EXL3_FAT_SCRATCH_ROWS",
            os.environ.get("MAX_NUM_BATCHED_TOKENS", "0"),
        )
        or 0
    )
    topk = int(os.environ.get("EXL3_FAT_GROUPED_TOPK", "8") or 8)
    needed = max(256, rows)
    key = (str(device), hidden, intermediate)
    scratch = _FAT_GROUPED_CACHE.get(key)
    if scratch is not None and int(scratch["h13"].shape[0]) >= needed:
        return scratch
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "EXL3 grouped scratch growth during CUDA graph capture; warm the "
            f"largest shape first (need {needed} rows)"
        )
    capacity = max(needed, configured * topk)
    scratch = {
        "h13": torch.empty((capacity, hidden), dtype=torch.float16, device=device),
        "h2": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
    }
    _FAT_GROUPED_CACHE[key] = scratch
    _FAT_GROUPED_BYTES[key] = sum(
        t.numel() * t.element_size() for t in scratch.values()
    )
    _EXL3_FAT_DIAG["grouped_scratch_bytes"] = sum(_FAT_GROUPED_BYTES.values())
    return scratch


def _excl_cumsum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inclusive = torch.cumsum(x, 0)
    return inclusive - x, inclusive


def build_grouped_fat_tables(
    counts: torch.Tensor,
    cap: int,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    rows_cap: int,
    tile_rows: int,
) -> dict[str, torch.Tensor]:
    """Device-side row/segment tables for the grouped fat kernels.

    Fat experts (count > cap) are laid out back to back in expert order in a
    fat-row buffer; each kernel CTA owns one `tile_rows` slice of one expert.
    Everything is computed with device ops on capacity-sized tensors and the
    kernels read the live `num_rows` / `num_segs`, so no host sync happens
    and the layer stays CUDA-graph capturable. `counts` excludes the
    invalid/nonlocal sentinel bucket, which the sort places after every
    real expert, so sentinel routes never enter a fat segment.
    """
    n_exp = int(counts.numel())
    device = counts.device
    fat_rows = torch.where(counts > cap, counts, torch.zeros_like(counts))
    row_off, row_cum = _excl_cumsum(fat_rows)
    sorted_off, _ = _excl_cumsum(counts)
    tiles = (fat_rows + (tile_rows - 1)) // tile_rows
    tile_off, tile_cum = _excl_cumsum(tiles)
    num_segs = tile_cum[-1:].to(torch.int32)
    num_rows = row_cum[-1:].to(torch.int32)
    max_segs = (rows_cap + tile_rows - 1) // tile_rows + n_exp
    seg = torch.arange(max_segs, device=device)
    e = torch.searchsorted(tile_cum, seg, right=True).clamp_(max=n_exp - 1)
    local_tile = seg - tile_off[e]
    seg_row0 = row_off[e] + local_tile * tile_rows
    seg_rows = torch.clamp(fat_rows[e] - local_tile * tile_rows, min=0, max=tile_rows)
    r = torch.arange(rows_cap, device=device)
    re = torch.searchsorted(row_cum, r, right=True).clamp_(max=n_exp - 1)
    src = (sorted_off[re] + (r - row_off[re])).clamp_(max=rows_cap - 1)
    return {
        "seg_expert": e.to(torch.int32),
        "seg_row0": seg_row0.to(torch.int32),
        "seg_rows": seg_rows.to(torch.int32),
        "num_segs": num_segs,
        "num_rows": num_rows,
        "row_expert": re.to(torch.int32),
        "row_token": token_sorted.index_select(0, src),
        "row_weight": weight_sorted.index_select(0, src),
    }


def apply_exl3_grouped_fat(
    xh: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    layer: torch.nn.Module,
    cap: int,
    limit: float,
) -> None:
    """E3: every fat expert of the layer in three launches, no host sync."""
    ext = load_fat_moe_ext()
    if ext is None:
        raise RuntimeError("EXL3 grouped tier selected but the E3 kernels are not loaded")
    ptrs = layer._exl3_ptrs
    device = ptrs["gate_trellis"].device
    if xh.device != device or out.device != device or counts.device != device:
        raise RuntimeError(
            f"EXL3 grouped tier: activations on {xh.device}, experts on {device}"
        )
    if not (xh.is_contiguous() and out.is_contiguous() and out.dtype == torch.float32):
        raise RuntimeError("EXL3 grouped tier needs contiguous fp16 input / fp32 output")
    hidden = int(xh.shape[1])
    intermediate = int(layer._exl3_intermediate_local)
    rows_cap = int(token_sorted.numel())
    scratch = _grouped_scratch(device, rows_cap, hidden, intermediate)
    h13 = scratch["h13"][:rows_cap]
    h2 = scratch["h2"][:rows_cap]
    tile_gu = int(ext.exl3_fat_moe_tile_rows_gateup())
    tile_dn = int(ext.exl3_fat_moe_tile_rows_down())
    token_sorted = token_sorted.contiguous()
    weight_sorted = weight_sorted.contiguous()
    tg = build_grouped_fat_tables(
        counts, cap, token_sorted, weight_sorted, rows_cap, tile_gu
    )
    td = (
        tg
        if tile_dn == tile_gu
        else build_grouped_fat_tables(
            counts, cap, token_sorted, weight_sorted, rows_cap, tile_dn
        )
    )
    ext.exl3_fat_moe_gather(
        xh, tg["row_token"], tg["row_expert"], ptrs["gate_suh"], h13, tg["num_rows"]
    )
    ext.exl3_fat_moe_gateup(
        h13,
        ptrs["gate_trellis"],
        ptrs["up_trellis"],
        ptrs["gate_svh"],
        ptrs["up_svh"],
        ptrs["down_suh"],
        h2,
        tg["seg_expert"],
        tg["seg_row0"],
        tg["seg_rows"],
        tg["num_segs"],
        float(limit),
    )
    ext.exl3_fat_moe_down(
        h2,
        ptrs["down_trellis"],
        ptrs["down_svh"],
        out,
        td["row_token"],
        td["row_weight"],
        td["seg_expert"],
        td["seg_row0"],
        td["seg_rows"],
        td["num_segs"],
    )
    _EXL3_FAT_DIAG["grouped_calls"] += 1


def exl3_moe_fast_requested() -> bool:
    """Opt-in SM121 K4/N256 thin-decode dispatch (default off).

    Mirrors the native dispatcher's validation: anything other than 0/1
    raises at load instead of surfacing as a native TORCH_CHECK on the
    first decode call.
    """
    raw = os.environ.get("GLM53_EXL3_MOE_FAST", "0")
    if raw not in ("0", "1"):
        raise RuntimeError("GLM53_EXL3_MOE_FAST must be 0 or 1")
    return raw == "1"


def build_exl3_fused_state(layer: torch.nn.Module, inners: list[dict[str, Any]]) -> None:
    """Pointer tables + fused temps, once after load. No per-token alloc."""
    import exllamav3_ext

    # Fail closed: an explicitly requested fast thin-decode path must never
    # silently run the stock kernel on an image built without it.
    fast = exl3_moe_fast_requested()
    if fast:
        if not hasattr(exllamav3_ext, "glm53_fast_moe_version"):
            raise RuntimeError(
                "GLM53_EXL3_MOE_FAST=1 requires the native decode-pipeline "
                "image (exllamav3_ext.glm53_fast_moe_version); this image "
                "was built without overlay/patch_exl3_decode_pipeline.py"
            )
        if exllamav3_ext.glm53_fast_moe_version() != 1:
            raise RuntimeError("Unsupported native EXL3 decode-pipeline version")

    device = layer.w13_trellis.device
    n_exp = len(inners)
    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)

    def _ptrs(which: str, attr: str) -> torch.Tensor:
        return torch.tensor(
            [int(getattr(pack[which], attr).data_ptr()) for pack in inners],
            dtype=torch.int64,
            device=device,
        )

    layer._exl3_ptrs = {
        "gate_trellis": _ptrs("gate", "trellis"),
        "gate_suh": _ptrs("gate", "suh"),
        "gate_svh": _ptrs("gate", "svh"),
        "up_trellis": _ptrs("up", "trellis"),
        "up_suh": _ptrs("up", "suh"),
        "up_svh": _ptrs("up", "svh"),
        "down_trellis": _ptrs("down", "trellis"),
        "down_suh": _ptrs("down", "suh"),
        "down_svh": _ptrs("down", "svh"),
    }
    # Gate/up SUH equality was verified across every expert at load time
    # (layer._exl3_shared_w13_suh, torch.equal on the packed tensors),
    # before weights were released. Aliasing the pointer tables is what lets
    # the native fast path prove the reuse predicate by pointer identity
    # (gate_ptrs_suh.data_ptr() == up_ptrs_suh.data_ptr()) and skip the
    # redundant up-input Hadamard. Only the fast path needs it, so FAST=0
    # leaves the tables exactly as the stock path builds them; FAST=1 with an
    # unequal checkpoint keeps both tables and takes the independent-transform
    # fast kernel.
    if fast and bool(getattr(layer, "_exl3_shared_w13_suh", False)):
        layer._exl3_ptrs["up_suh"] = layer._exl3_ptrs["gate_suh"]
    idx = int(device.index) if device.index is not None else 0
    concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(idx))
    if concurrency < 1:
        concurrency = 1
    rows = temp_rows_fused()
    key = (str(device), hidden, intermediate, concurrency, rows)
    temps = _FUSED_TEMP_CACHE.get(key)
    if temps is None:
        temps = (
            torch.empty((concurrency, rows, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, intermediate), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, intermediate), dtype=torch.float16, device=device),
        )
        _FUSED_TEMP_CACHE[key] = temps
        _EXL3_FAT_DIAG["fused_temps_allocs"] += 1
    # Layers share one cache entry, so assign (never accumulate) the bytes.
    _EXL3_FAT_DIAG["fused_temps_bytes"] = sum(
        t.numel() * t.element_size() for t in temps
    )
    layer._exl3_fused_temps = temps
    layer._exl3_fused_concurrency = concurrency
    layer._exl3_k = int(layer._exl3_bits)


def _exl3_moe_launch(
    fn: Any,
    xh: torch.Tensor,
    out: torch.Tensor,
    expert_count: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    temps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ptrs: dict[str, torch.Tensor],
    k: int,
    limit: float,
    n_active_host: int | None,
) -> None:
    args = (
        xh,
        out,
        expert_count,
        token_sorted,
        weight_sorted,
        temps[0],
        temps[1],
        temps[2],
        temps[3],
        MOE_ACT_SILU,
        k,
        k,
        k,
        ptrs["gate_trellis"],
        ptrs["gate_suh"],
        ptrs["gate_svh"],
        ptrs["up_trellis"],
        ptrs["up_suh"],
        ptrs["up_svh"],
        ptrs["down_trellis"],
        ptrs["down_suh"],
        ptrs["down_svh"],
        True,
        False,
        True,
        False,
        True,
        False,
        float(limit),
    )
    if n_active_host is not None:
        fn(*args, n_active_host)
    else:
        fn(*args)


def _exl3_moe_row_tiles(
    fn: Any,
    xh: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    temps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ptrs: dict[str, torch.Tensor],
    k: int,
    limit: float,
    n_active_host: int | None,
    max_rows: int,
) -> None:
    """Launch exl3_moe once per 128-row slice of the sorted expert-token buffer.

    Prefill-only (host syncs). Decode never reaches here: tokens <= temp rows.
    Kernel skips experts with count > temp rows; tiles keep every expert inside temps.
    """
    n_exp = int(counts.shape[0])
    device = counts.device
    tile = int(temps[0].shape[1])
    n_tiles = (max_rows + tile - 1) // tile
    prefix = torch.empty(n_exp, dtype=torch.long, device=device)
    prefix[0] = 0
    if n_exp > 1:
        prefix[1:] = counts[:-1].cumsum(0)
    for t in range(n_tiles):
        row0 = t * tile
        tile_counts = (counts - row0).clamp(min=0, max=tile)
        n_tile = int(tile_counts.sum().item())
        if n_tile == 0:
            continue
        cum = tile_counts.cumsum(0)
        idx = torch.arange(n_tile, device=device)
        expert_id = torch.searchsorted(cum, idx, right=True)
        local_row = idx - (cum[expert_id] - tile_counts[expert_id])
        src = prefix[expert_id] + row0 + local_row
        tile_ec = torch.zeros(n_exp + 1, dtype=torch.long, device=device)
        tile_ec[:n_exp] = tile_counts
        _exl3_moe_launch(
            fn,
            xh,
            out,
            tile_ec,
            token_sorted.index_select(0, src),
            weight_sorted.index_select(0, src),
            temps,
            ptrs,
            k,
            limit,
            n_active_host,
        )


def apply_exl3_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
) -> torch.Tensor:
    """One exl3_moe launch per layer when tokens or hottest expert fit temp rows.

    Cap is temps dim1 (`EXL3_TEMP_ROWS_FUSED`, default 128). Overflow uses GPU
    row tiles if `EXL3_MOE_ROW_TILE=1`; otherwise fat experts use the highest
    enabled tier: kernel implies batched, batched implies sorted, then legacy.
    Decode (tokens ≤ cap) stays a single graph-safe launch.
    """
    import exllamav3_ext

    tokens, hidden = x2d.shape
    n_exp = len(inners)
    ptrs = getattr(layer, "_exl3_ptrs", None)
    temps = getattr(layer, "_exl3_fused_temps", None)
    if not ptrs or temps is None:
        raise RuntimeError("EXL3 fused pointer tables were not built after weight load")

    local = map_topk_to_local(ids, n_exp, expert_map)
    topk = int(ids.shape[-1])
    flat_token = torch.arange(tokens, device=x2d.device, dtype=torch.long).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(dtype=torch.float16)
    order = local.argsort()
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    # scatter_add stays on GPU. torch.bincount can host-stage and break CUDA graphs.
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=local.device)
    expert_count.scatter_add_(
        0, local.long(), torch.ones(local.shape, dtype=torch.long, device=local.device)
    )
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    xh = x2d.contiguous().half()

    counts = expert_count[:n_exp]
    fn = exllamav3_ext.exl3_moe
    # -1 = unknown active count: max-concurrency grid, no .item() host sync.
    n_active_host = -1 if _exl3_moe_accepts_num_active(fn) else None
    k = int(getattr(layer, "_exl3_k", 4))
    # Actual kernel cap is the allocated temp dim1 (env-selected at load).
    cap = int(temps[0].shape[1])

    # Reset per call so a "kernel" label from an earlier prefill cannot
    # masquerade through later decode/thin/row-tile calls.
    layer._exl3_last_fat_fallback = "none"
    layer._exl3_last_fat_reason = "no_fat_experts"

    if tokens <= cap:
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
        return out

    # Prefill larger than temps. E1 copies routing counts on a side stream and
    # launches thin experts immediately, overlapping the D2H synchronization.
    # Decode never reaches here (capture sizes << cap).
    _EXL3_FAT_DIAG["prefill_layer_calls"] += 1
    use_row_tiles = fused_moe_row_tile_enabled()
    if (
        grouped_fat_enabled()
        and not use_row_tiles
        and getattr(layer, "_exl3_fat_effective_tier", None) == "grouped"
    ):
        # E3: thin experts in the fused kernel (it skips count > cap), every
        # fat expert in three grouped launches driven by device-side tables.
        # No host sync, so this branch is CUDA-graph capturable.
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
        apply_exl3_grouped_fat(
            xh, out, counts, token_sorted, weight_sorted, layer, cap, limit
        )
        _record_exl3_fat_tier(layer, "grouped", "grouped_ok")
        if fat_expert_log_enabled():
            record_exl3_fat_expert_stats(counts)
        return out
    want_fat_kernel = fat_kernel_enabled() or grouped_fat_enabled()
    want_batched_fat = batched_fat_fallback_enabled() or want_fat_kernel
    use_sorted_fat = sorted_fat_fallback_enabled() or want_batched_fat
    use_batched_fat = (
        want_batched_fat
        and bool(getattr(layer, "_exl3_shared_w13_suh", False))
    )
    use_fat_kernel = use_batched_fat and want_fat_kernel
    launched = False
    counts_host = None
    if use_batched_fat and not use_row_tiles:
        counts_cpu, count_stream = _stage_counts_to_host(counts)
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
        launched = True
        count_stream.synchronize()
        counts_host = counts_cpu.tolist()
    elif use_sorted_fat:
        counts_host = counts.tolist()

    max_rows = (
        max(counts_host, default=0)
        if counts_host is not None
        else int(counts.max().item())
    )
    if fat_expert_log_enabled():
        record_exl3_fat_expert_stats(
            counts, max_rows=max_rows, counts_host=counts_host
        )
    if max_rows <= cap:
        _EXL3_FAT_DIAG["thin_calls"] += 1
        _record_exl3_fat_reason("thin_only")
        if not launched:
            _exl3_moe_launch(
                fn, xh, out, expert_count, token_sorted, weight_sorted,
                temps, ptrs, k, limit, n_active_host,
            )
        return out

    if use_row_tiles:
        _EXL3_FAT_DIAG["row_tile_calls"] += 1
        layer._exl3_last_fat_fallback = "row_tile"
        layer._exl3_last_fat_reason = "row_tile_preempts_fat"
        _record_exl3_fat_reason("row_tile_preempts_fat")
        _exl3_moe_row_tiles(
            fn, xh, out, counts, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host, max_rows,
        )
        return out

    if not launched:
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, k, limit, n_active_host,
        )
    if use_batched_fat:
        if use_fat_kernel:
            _record_exl3_fat_tier(layer, "kernel", "kernel_ok")
        else:
            _record_exl3_fat_tier(layer, "batched", "batched_ok")
        assert counts_host is not None
        apply_exl3_batched_fat(
            xh,
            token_sorted,
            weight_sorted,
            counts_host,
            inners,
            limit,
            cap,
            out,
            use_kernel=use_fat_kernel,
        )
    elif use_sorted_fat:
        _record_exl3_fat_tier(
            layer,
            "sorted",
            "degraded_shared_suh" if want_batched_fat else "sorted_ok",
        )
        assert counts_host is not None
        apply_exl3_sorted_fat(
            xh,
            token_sorted,
            weight_sorted,
            counts_host,
            inners,
            limit,
            cap,
            out,
        )
    else:
        _record_exl3_fat_tier(layer, "legacy", "legacy_default")
        fat = (counts > cap).nonzero(as_tuple=False).view(-1)
        if fat.numel():
            apply_exl3_python_loop(
                x2d,
                ids,
                weights,
                inners,
                expert_map,
                limit,
                only_experts=set(int(i) for i in fat.tolist()),
                out=out,
            )
            _EXL3_FAT_DIAG["fat_expert_runs"] += int(fat.numel())
    return out


def apply_exl3_experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer: torch.nn.Module,
    *,
    limit: float = SWIGLU_LIMIT_DEFAULT,
    fused: bool | None = None,
) -> torch.Tensor:
    """Shipped routed-expert apply. `fused=None` honors EXL3_FUSED_MOE."""
    inners = getattr(layer, "_exl3_inners", None)
    if not inners:
        raise RuntimeError("EXL3 experts were not built after weight load")
    tokens, hidden = x.shape[-2], x.shape[-1]
    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = pin_exl3_expert_map(layer, x2d.device)
    have_ptrs = bool(getattr(layer, "_exl3_ptrs", None))
    if fused is True and not have_ptrs:
        raise RuntimeError("EXL3 fused apply requested but pointer tables are missing")
    use_fused = (fused_moe_enabled() if fused is None else bool(fused)) and have_ptrs
    if use_fused:
        try:
            import exllamav3_ext

            use_fused = hasattr(exllamav3_ext, "exl3_moe")
        except Exception:
            use_fused = False
    if use_fused:
        out = apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit)
        layer._exl3_last_apply = "fused"
    else:
        out = apply_exl3_python_loop(x2d, ids, weights, inners, expert_map, limit)
        layer._exl3_last_apply = "loop"
    return out.to(dtype=x.dtype)


def _suffix_from_mapped_name(weight_name: str) -> str:
    tail = weight_name.rsplit(".", 1)[-1]
    for suffix in EXL3_SUFFIXES:
        if tail == suffix or tail.endswith("_" + suffix):
            return suffix
    raise ValueError(f"not an EXL3 packed name: {weight_name}")


def _prefix_has_suffix(prefix: str, suffix: str) -> bool:
    """Module-path suffix match: "self_attn.o_proj" matches
    "model.layers.3.self_attn.o_proj" but not "...cross_attn.o_proj_x"."""
    return prefix == suffix or prefix.endswith("." + suffix)


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """Routed-experts-only EXL3/MCG. Dense / shared / attention stay native
    unless the pack config carries a ``non_routed_exl3`` block ([dense-exl3])."""

    def __init__(
        self,
        bits: int = 4,
        codebook: str = "mcg",
        scope: str = "glm53_routed_experts_only",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.codebook = str(codebook)
        self.scope = str(scope)
        self.raw_config = dict(kwargs)
        if self.codebook != "mcg":
            raise ValueError(
                f"this overlay only implements codebook=mcg; got {self.codebook!r}"
            )
        if self.bits not in (3, 4, 5, 6):
            raise ValueError(f"unsupported EXL3 bits={self.bits}")
        # [dense-exl3] Non-routed dense linear config, written by
        # vllm-exl3's tools/dense_overlay.py: {"layers": {module_prefix:
        # {"bits": K[, "bf16_shards": [...]]}, ...}}. kv_b_proj is refused:
        # MLA weight absorption reads
        # that weight directly, so an EXL3 swap would silently break.
        raw_nr = kwargs.get("non_routed_exl3") or {}
        self.non_routed_exl3: dict[str, Any] = dict(raw_nr) if raw_nr else {}
        for prefix, layer_cfg in (self.non_routed_exl3.get("layers") or {}).items():
            if _prefix_has_suffix(prefix, "self_attn.kv_b_proj"):
                raise ValueError(
                    "non_routed_exl3 cannot cover self_attn.kv_b_proj: MLA "
                    "weight absorption reads that weight directly"
                )
            k = (layer_cfg or {}).get("bits")
            if k is None or int(k) not in (2, 3, 4, 5, 6):
                raise ValueError(
                    f"unsupported non_routed_exl3 bits={k} for {prefix}"
                )
            raw_bs = (layer_cfg or {}).get("bf16_shards") or []
            if any(type(i) is not int or i < 0 for i in raw_bs):
                raise ValueError(f"non_routed_exl3 bf16_shards must be nonnegative integers: {raw_bs}")
            bs = sorted(raw_bs)
            if bs != raw_bs or (bs and bs != list(range(bs[0], bs[0] + len(bs)))):
                raise ValueError(
                    f"non_routed_exl3 bf16_shards for {prefix} must be a "
                    f"sorted contiguous run (got {raw_bs})"
                )

    def _assert_non_routed_built(self) -> None:
        """[dense-exl3] Fail loud when declared modules never got an
        Exl3LinearMethod: the pack index dropped their BF16 tensors, so they
        would serve uninitialized weights (vLLM's strict-load check is off
        for quantized models). Runs after construction; construction of every
        module precedes any process_weights_after_loading call."""
        declared = set(self.non_routed_exl3.get("layers") or {})
        if not declared:
            return
        missing = sorted(declared - {p for p, _ in _DENSE_EXL3_MODULES})
        if missing:
            raise RuntimeError(
                f"[dense-exl3] {len(missing)}/{len(declared)} declared "
                f"modules were never constructed (e.g. {missing[:3]}); "
                "their BF16 tensors are absent from the pack index — pack "
                "prefixes do not match this model"
            )

    def offset_draft_layer_prefixes(self, offset: int) -> None:
        """[dense-exl3] Shift declared ``layers.N`` prefixes by a DFlash
        draft's ``start_layer_id``.

        The image builds draft layers under runtime prefixes offset by the
        target's layer count (it keeps draft KV-cache layer names distinct
        from the target's), while an EXL3 draft pack declares
        checkpoint-relative indices (``model.layers.0`` …). Call once with
        the offset before the draft's layers are constructed; a second call
        on the same instance is a no-op. Non-layer keys (``model.fc``) are
        untouched. BF16 drafts never call this (their quant config is None).
        """
        layers = self.non_routed_exl3.get("layers") or {}
        if not offset or not layers or getattr(self, "_draft_prefix_offset", 0):
            return
        self.non_routed_exl3["layers"] = {
            re.sub(
                r"layers\.(\d+)\.",
                lambda m: f"layers.{int(m.group(1)) + offset}.",
                prefix,
            ): layer_cfg
            for prefix, layer_cfg in layers.items()
        }
        self._draft_prefix_offset = offset

    # [dense-exl3] --- non-routed lookups -------------------------------
    def _matches_non_routed_exl3(self, prefix: str) -> bool:
        return prefix in (self.non_routed_exl3.get("layers") or {})

    def _bits_for_non_routed(self, prefix: str) -> int:
        return int(self.non_routed_exl3["layers"][prefix]["bits"])

    def _bf16_shards_for(self, prefix: str) -> list[int]:
        return list(self.non_routed_exl3.get("layers", {}).get(prefix, {}).get("bf16_shards", []))

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # LinearEXL3 uses CUDA >= Ampere; GB10 is SM121.
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        skip = {
            "bits",
            "codebook",
            "scope",
            "quant_method",
            # tr3 ships a 37 MiB per-tensor ledger; keep it off the config object.
            "tensor_storage",
        }
        return cls(
            bits=int(config.get("bits", 4)),
            codebook=str(config.get("codebook", "mcg")),
            scope=str(config.get("scope", "glm53_routed_experts_only")),
            **{k: v for k, v in config.items() if k not in skip},
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        method = str((hf_quant_cfg or {}).get("quant_method", "")).lower()
        if method == "exl3":
            return "exl3"
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )

        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(layer.moe_config, self)
        if isinstance(layer, ParallelLMHead):
            # [dense-exl3] pack-declared EXL3 lm_head (the 4.05bpw overlay can
            # carry lm_head K6 mul1). Matched by the exact runtime prefix
            # (language_model.lm_head in the vision-wrapped model). A
            # non-matching head — and always embed_tokens, a plain
            # VocabParallelEmbedding — falls back to the unquantized method.
            if self._matches_non_routed_exl3(prefix):
                return Exl3LinearMethod(
                    self, prefix, bits=self._bits_for_non_routed(prefix)
                )
            return None
        if isinstance(layer, LinearBase):
            # [dense-exl3] pack-declared dense EXL3 linears. Mutually
            # exclusive with the FP8-overlay groups on the same module:
            # fail loudly instead of picking a winner.
            if self._matches_non_routed_exl3(prefix):
                group = _glm53_dense_fp8_group(prefix)
                if group is not None:
                    raise RuntimeError(
                        f"[dense-exl3] {prefix} is EXL3-dense (pack "
                        f"non_routed_exl3) and also targets GLM53_DENSE_FP8 "
                        f"group '{group}' — unset one; they are mutually "
                        "exclusive per module"
                    )
                return Exl3LinearMethod(
                    self, prefix, bits=self._bits_for_non_routed(prefix)
                )
            group = _glm53_dense_fp8_group(prefix)  # [glm53-dense-fp8]
            if group is not None:
                return Glm53DenseFp8Method(group, prefix)
            return UnquantizedLinearMethod()
        return None


# ----------------------------------------------------------------------------
# [glm53-dense-fp8] Optional FP8 weight-only (Marlin) path for the BF16 dense
# projections. GLM53_DENSE_FP8=off (default) | comma list of groups:
#   shared  mlp.shared_experts.{gate_up_proj,down_proj}
#   dense   mlp.{gate_up_proj,down_proj} of the dense-MLP layers
#   kda     self_attn.{in_proj_qkvbfg_a,f_b_proj,g_b_proj,o_proj} of KDA layers
#   mla     self_attn.{fused_qkv_a_proj,q_b_proj,o_proj} of MLA layers (kv_b_proj
#           stays BF16: MLA reads its weight directly for the absorbed matmuls)
# Every group is target-layer only: a prefix is claimed when its layer index is
# inside the target's hf_text_config layer_types (and, for kda/mla, the layer's
# type matches). A speculative draft — DFlash2 numbers its layers after the
# target's — is never claimed, whatever pack declares it.
# Weights load as BF16 exactly as today (all custom loaders untouched, ABLIT edits
# o_proj at the end of load_weights), then process_weights_after_loading quantizes
# per output channel to FP8 e4m3 and repacks for the Marlin kernel. PROVISIONAL:
# changes target numerics; needs a KLD panel before it can become a default.
# ----------------------------------------------------------------------------
_GLM53_DENSE_FP8_SUFFIXES = {
    "shared": (".mlp.shared_experts.gate_up_proj", ".mlp.shared_experts.down_proj"),
    "dense": (".mlp.gate_up_proj", ".mlp.down_proj"),
    "kda": (".self_attn.in_proj_qkvbfg_a", ".self_attn.f_b_proj", ".self_attn.g_b_proj", ".self_attn.o_proj"),
    "mla": (".self_attn.fused_qkv_a_proj", ".self_attn.q_b_proj", ".self_attn.o_proj"),
}


def _glm53_dense_fp8_groups() -> set[str]:
    raw = os.environ.get("GLM53_DENSE_FP8", "off").strip().lower()
    if raw in ("", "off", "0", "no", "none"):
        return set()
    if raw in ("all", "on", "1"):
        return {"shared", "dense", "kda", "mla"}
    groups = {g.strip() for g in raw.split(",") if g.strip()}
    unknown = groups - set(_GLM53_DENSE_FP8_SUFFIXES)
    if unknown:
        raise ValueError(f"GLM53_DENSE_FP8: unknown group(s) {sorted(unknown)}")
    return groups


def _glm53_layer_types() -> list[str] | None:
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config().model_config.hf_text_config
        lt = getattr(cfg, "layer_types", None)
        return list(lt) if lt else None
    except Exception:  # noqa: BLE001
        return None


def _glm53_dense_fp8_group(prefix: str, groups: set[str] | None = None, layer_types: list[str] | None = None) -> str | None:
    """Group name if `prefix` (vLLM module path) is an allow-listed dense projection
    of a *target* layer."""
    groups = _glm53_dense_fp8_groups() if groups is None else groups
    if not groups:
        return None
    if ".mtp" in prefix or "visual" in prefix or "draft" in prefix:
        return None
    m = re.search(r"\.layers\.(\d+)\.", prefix)
    layer_idx = int(m.group(1)) if m else None
    lt = _glm53_layer_types() if layer_types is None else layer_types
    # Target-only: every group covers the target model's own layers
    # 0..len(layer_types)-1, and GLM53_DENSE_FP8 must never claim a module the
    # pack serves from another checkpoint. A DFlash2 draft is built with
    # start_layer_id = the target's layer count, so its layers are numbered
    # after the target's (45..49 on a 45-layer target) and an EXL3 draft pack
    # declares dense modules under that same "layers.N" shape; an unbounded
    # classifier claims them and trips the mutual-exclusion refusal below on a
    # pack that never overlapped a target layer. Layer index past the target's
    # layer_types is not a target module, for any group. With no layer_types
    # there is no bound to consult: dense/shared keep their historical match
    # and kda/mla stay skipped in the loop.
    if layer_idx is not None and lt is not None and layer_idx >= len(lt):
        return None
    for group in ("shared", "dense", "kda", "mla"):
        if group not in groups:
            continue
        if not any(prefix.endswith(s) for s in _GLM53_DENSE_FP8_SUFFIXES[group]):
            continue
        if group == "dense" and ".shared_experts." in prefix:
            continue
        if group in ("kda", "mla"):
            if lt is None or layer_idx is None or layer_idx >= len(lt):
                continue
            is_kda = lt[layer_idx] == "linear_attention"
            if (group == "kda") != is_kda:
                continue
        return group
    return None


_GLM53_TP3_UNALIGNED_KDA_SUFFIXES = (
    ".self_attn.f_b_proj",
    ".self_attn.g_b_proj",
)


def _glm53_use_marlin(group: str, prefix: str, tp_size: int) -> bool:
    """Whether this projection's activation layout satisfies Marlin."""
    # TP=3 f_a/g_a are 128-wide views of an 8,726-wide merged KDA projection.
    # Their row pitch is not divisible by 8, and f_a's byte offset is not
    # 16-aligned at capture size 1. Keeping just these two small projections in
    # BF16 avoids invalid Marlin inputs and allocations inside CUDA graphs.
    return not (
        tp_size == 3
        and group == "kda"
        and prefix.endswith(_GLM53_TP3_UNALIGNED_KDA_SUFFIXES)
    )


# --- KDA large-M BF16 path ------------------------------------------------
# Stock runs FP8-Marlin W8A16 for every M. With GLM53_KDA_BF16_LARGE_M=1 the
# KDA in_proj keeps Marlin for small M and serves large-M prefill from a
# load-time BF16 copy of the SAME logical FP8 weight: the e4m3 tensor times
# the stored per-output-channel scale that Marlin itself multiplies by.
# Activations stay BF16, so there is no activation-quantization term.
# Fixed dispatch boundary from SM121 microbenchmarks on in_proj [12576x4096]:
# decode never exceeds M=220, prefill chunks land at M>=768, and the band
# 221-511 is empty in every measured step, so 512 sits inside the gap and
# keeps ordinary decode entirely on stock Marlin. Intentionally not
# user-configurable: 512 is the boundary actually measured and qualified.
KDA_BF16_LARGE_M_MIN_M = 512
# TP-local in_proj_qkvbfg_a shapes: local_heads * 385 + 256.
# TP2/TP4 shard 64 heads (local 32/16); TP3 pads 64→66 (local 22).
# q/k/v/b are sharded; f_a/g_a (128 each) remain replicated.
# Everything else stays Marlin by construction.
KDA_BF16_LARGE_M_SHAPES_BY_TP = {
    2: (12576, 4096),
    3: (8726, 4096),
    4: (6416, 4096),
}
KDA_BF16_LARGE_M_SHAPES = frozenset(KDA_BF16_LARGE_M_SHAPES_BY_TP.values())
# Rows per dequant chunk: bounds the peak fp32 intermediate to ~8 MiB at
# K=4096 instead of one [12576,4096] fp32 allocation (~196 MiB).
KDA_BF16_LARGE_M_CHUNK_ROWS = 512
# BF16 is the shipped dtype: same 2 bytes/weight as an fp16 copy, no
# activation conversion, and it matches the activation dtype exactly.
KDA_BF16_LARGE_M_DTYPE = torch.bfloat16


def kda_bf16_large_m_enabled() -> bool:
    """Opt-in large-M BF16 dispatch for the KDA in_proj (default off)."""
    raw = os.environ.get("GLM53_KDA_BF16_LARGE_M", "0")
    if raw not in ("0", "1"):
        raise RuntimeError("GLM53_KDA_BF16_LARGE_M must be 0 or 1")
    return raw == "1"


def kda_bf16_large_m_logical_weight(
    fp8: torch.Tensor,
    scales: torch.Tensor,
    out_dtype: torch.dtype = KDA_BF16_LARGE_M_DTYPE,
    chunk_rows: int = KDA_BF16_LARGE_M_CHUNK_ROWS,
) -> torch.Tensor:
    """Materialize the logical FP8 weight (fp8 x per-output-channel scale).

    The product of an e4m3 value (<=4 significant bits) and a bf16 scale
    (<=8 significant bits) needs at most 12 significant bits, so the fp32
    intermediate is exact and the single cast to `out_dtype` rounds once. Row
    chunking only bounds the temporary; it does not change the result.
    """
    n = int(fp8.shape[0])
    out = torch.empty(fp8.shape, dtype=out_dtype, device=fp8.device)
    step = max(1, int(chunk_rows))
    for start in range(0, n, step):
        stop = min(start + step, n)
        block = fp8[start:stop].to(torch.float32)
        block = block * scales[start:stop].to(torch.float32).unsqueeze(1)
        out[start:stop] = block
    return out


# Observability only: `apply` is not called when a CUDA graph replays, so these
# counters describe eager calls plus graph-capture decisions. The authoritative
# per-step M histogram comes from the model runner.
_KDA_LARGE_M_DISPATCH_STATS: dict[str, int] = {
    "bf16_calls": 0, "bf16_rows": 0,
    "marlin_calls": 0, "marlin_rows": 0,
}
_KDA_LARGE_M_STATS_DUMP_EVERY = 512


def kda_large_m_dispatch_stats() -> dict[str, int]:
    return dict(_KDA_LARGE_M_DISPATCH_STATS)


def _kda_large_m_note(which: str, rows: int) -> None:
    stats = _KDA_LARGE_M_DISPATCH_STATS
    stats[which + "_calls"] += 1
    stats[which + "_rows"] += rows
    if stats[which + "_calls"] % _KDA_LARGE_M_STATS_DUMP_EVERY:
        return
    path = os.environ.get("KDA_LARGE_M_STATS_PATH", "")
    if not path:
        return
    try:
        if torch.cuda.is_current_stream_capturing():
            return
        tmp = path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(stats, handle, sort_keys=True)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001  (observability must never break serving)
        pass


class Glm53DenseFp8Method(UnquantizedLinearMethod):
    """BF16 weight at load time; per-output-channel FP8 e4m3 + Marlin at apply."""

    def __init__(self, group: str, prefix: str) -> None:
        super().__init__()
        self.group = group
        self.prefix = prefix
        self.ready = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
        if not _glm53_use_marlin(self.group, self.prefix, tp_size):
            logger.warning_once(
                "[glm53-dense-fp8] TP=3 keeps KDA f_b_proj/g_b_proj in BF16 "
                "because their merged-projection views violate Marlin input "
                "pitch/alignment requirements"
            )
            return
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_fp8_layer_for_marlin,
        )

        w = layer.weight.data
        if w.dtype != torch.bfloat16 and w.dtype != torch.float16:
            raise RuntimeError(f"[glm53-dense-fp8] expected a BF16/FP16 weight, got {w.dtype} for {self.group}")
        n, k = w.shape
        assert n == layer.output_size_per_partition and k == layer.input_size_per_partition, (w.shape, layer)
        wf = w.float()
        scales = wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0  # [N] per output channel
        fp8 = (wf / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        del wf
        layer.orig_dtype = w.dtype
        layer.weight = torch.nn.Parameter(fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scales.to(layer.orig_dtype), requires_grad=False)
        layer.weight_block_size = None
        # The STORED (dtype-rounded) per-channel scale is what Marlin actually
        # multiplies by, so it is also the scale the BF16 copy must use.
        # Held as a local: prepare_fp8_layer_for_marlin replaces weight_scale.
        scales_stored = layer.weight_scale.detach()
        prepare_fp8_layer_for_marlin(layer, size_k_first=False)
        layer.glm53_fp8_n, layer.glm53_fp8_k = n, k
        self._retain_bf16_large_m_weights(
            layer, fp8, scales_stored, n, k, tp_size)
        self.ready = True

    def _retain_bf16_large_m_weights(
        self,
        layer: torch.nn.Module,
        fp8: torch.Tensor,
        scales_stored: torch.Tensor,
        n: int,
        k: int,
        tp_size: int,
    ) -> None:
        """Load-time retention for the large-M BF16 path.

        Materializes the logical weight the stock path already implements --
        the FP8 e4m3 tensor times the STORED per-output-channel scale that
        Marlin consumes -- as one BF16 [N,K] copy, and retains nothing else.
        Cost when enabled: TP2 12576 x 4096 x 2 bytes = 98.25 MiB per
        layer-rank (~3.26 GiB/rank, 34 KDA layers); TP3 8726 x 4096 x 2
        bytes = 68.17 MiB per layer-rank (~2.26 GiB/rank); TP4 6416 x
        4096 x 2 bytes = 50.1 MiB per layer-rank (~1.66 GiB/rank).

        Fail-closed: a KDA in_proj layer with the feature enabled must satisfy
        every predicate (TP=2, TP=3 or TP=4, SM121 capability, exact TP-local
        shape, e4m3 weight with a BF16 stored scale); anything else raises at
        load instead of silently running Marlin under a large-M label.
        Non-candidate layers retain nothing and stay on Marlin.
        """
        if not kda_bf16_large_m_enabled():
            return
        if not (
            self.group == "kda"
            and self.prefix.endswith("in_proj_qkvbfg_a")
        ):
            return
        expected = KDA_BF16_LARGE_M_SHAPES_BY_TP.get(tp_size)
        if expected is None:
            raise RuntimeError(
                "GLM53_KDA_BF16_LARGE_M=1 requires TP=2, TP=3 or TP=4 "
                f"(got tp_size={tp_size})"
            )
        try:
            cap = torch.cuda.get_device_capability(fp8.device)
        except Exception as exc:
            raise RuntimeError(
                "GLM53_KDA_BF16_LARGE_M=1 requires a CUDA device "
                f"with SM121 capability ({exc!r})"
            )
        if tuple(int(v) for v in cap) != (12, 1):
            raise RuntimeError(
                "GLM53_KDA_BF16_LARGE_M=1 is qualified for SM121/GB10 "
                f"only (got capability {tuple(int(v) for v in cap)})"
            )
        if (n, k) != expected:
            raise RuntimeError(
                "GLM53_KDA_BF16_LARGE_M=1 is qualified for the "
                f"TP{tp_size}-local KDA in_proj shape {expected[0]}x{expected[1]} "
                f"(got [{n}x{k}])"
            )
        if fp8.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "GLM53_KDA_BF16_LARGE_M=1 expects an e4m3 logical weight "
                f"(got {fp8.dtype})"
            )
        # process_weights_after_loading stores scales in layer.orig_dtype.
        # FP16 scales therefore identify an unqualified FP16 activation path.
        if scales_stored.dtype != torch.bfloat16:
            raise RuntimeError(
                "GLM53_KDA_BF16_LARGE_M=1 requires BF16 weights/activations "
                f"(stored scale dtype is {scales_stored.dtype}); FP16 is not qualified"
            )
        # Fixed qualified boundary: stored on the layer so capture and replay
        # cannot disagree about the branch.
        threshold = KDA_BF16_LARGE_M_MIN_M
        try:
            w_bf16 = kda_bf16_large_m_logical_weight(
                fp8, scales_stored, KDA_BF16_LARGE_M_DTYPE)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"kda bf16-large-m: dequant failed ({exc!r})"
            )
        # Run the real GEMM once at load, at the full [N,K] width, so a broken
        # cuBLAS path fails during load instead of on the first prefill.
        # Deliberately tiny M: this is a launch smoke test (and a one-time
        # cuBLAS workspace init), not a benchmark; load time stays flat.
        probe_m = 2
        try:
            torch.nn.functional.linear(
                torch.zeros((probe_m, k), dtype=KDA_BF16_LARGE_M_DTYPE,
                              device=fp8.device),
                w_bf16,
            )
            torch.cuda.synchronize(fp8.device)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"kda bf16-large-m: cuBLAS probe failed ({exc!r})"
            )
        layer.glm53_bf16_lm_w = w_bf16
        layer.glm53_bf16_lm_n = n
        layer.glm53_bf16_lm_k = k
        # Resolved once: capture and replay cannot disagree about the branch.
        layer.glm53_bf16_lm_min_m = threshold
        logger.info(
            "kda bf16-large-m retained for %s [%dx%d] +%.1f MiB/rank (M>%d), dtype=%s",
            self.group, n, k, w_bf16.numel() * w_bf16.element_size() / 2**20,
            threshold, str(KDA_BF16_LARGE_M_DTYPE).replace("torch.", ""),
        )

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if not self.ready:
            return super().apply(layer, x, bias)
        # Hybrid dispatch: M is tensor metadata (no host sync). The retained
        # BF16 copy exists only when every load-time predicate passed; small M
        # stays Marlin (measured win), anything else falls through to Marlin.
        # Per-capture-size CUDA graphs bake the branch taken at capture.
        n = int(layer.glm53_fp8_n)
        k = int(layer.glm53_fp8_k)
        in_proj_shape = (n, k) in KDA_BF16_LARGE_M_SHAPES
        if bias is None and x.dim() >= 2 and int(x.shape[-1]) == k:
            m = x.numel() // k
            wb = getattr(layer, "glm53_bf16_lm_w", None)
            if wb is not None and m > int(layer.glm53_bf16_lm_min_m):
                out = F.linear(x.reshape(-1, k), wb)
                _kda_large_m_note("bf16", m)
                return out.reshape(x.shape[:-1] + (n,))
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
        )

        if in_proj_shape and x.dim() >= 2 and int(x.shape[-1]) == k:
            _kda_large_m_note("marlin", x.numel() // k)
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=n,
            size_k=k,
            bias=bias,
        )


# [dense-exl3] ---------------------------------------------------------------
# Dense-EXL3 layer registry + torch custom op: the LinearEXL3 call is opaque
# to dynamo/inductor (no graph break per module) and cudagraph-capturable,
# same approach as vLLM's own ops. EXL3 shards produce fp16 (the fast
# bitcoder path is fp16-only); the bf16 tail shards run one GEMM in the
# activation dtype and never round-trip through fp16.
_DENSE_EXL3_LAYERS: list = []
_DENSE_EXL3_MODULES: list[tuple[str, int]] = []  # (prefix, bits) per method
_dense_exl3_build_checked = False

# [dense-exl3] prefill BF16 retention (GLM53_DENSE_EXL3_PREFILL_BF16) --------
# Measured (per-shape microbench over the pack's real trellises): the dense
# E-vs-F prefill gap is the activation passes around the EXL3 reconstruct
# path (per-shard had_r_128 in/out, casts, cat), not the weight reconstruct.
# A selected module therefore keeps a load-time BF16 copy — every EXL3
# shard's get_weight_tensor() (hadamards pre-applied, i.e. exactly the
# weight the reconstruct path multiplies by) concatenated with any bf16
# tail rows in output_sizes order — and rows > 144 run ONE F.linear in
# x.dtype: the same logical weights, byte-for-byte F's LARGE_M arithmetic.
# Rows <= 144 stay on the EXL3 custom op (bitcoder), so decode is
# untouched. Cost: 2 B/weight per rank, logged per module and as one boot
# summary line.
# Dispatch boundary: LinearEXL3's own AUTO_RECONSTRUCT_THRESHOLD (144;
# above it forward switches to the reconstruct path). Deliberately a
# separate constant from the FP8 path's KDA_BF16_LARGE_M_MIN_M (512): that
# one is the measured Marlin crossover, this one is the EXL3 library's.
DENSE_EXL3_PREFILL_BF16_MIN_M = 144
# Module-type vocabulary (suffix match on the vLLM module path; kda/mla
# o_proj disambiguated by layer_types like the FP8 groups).
_DENSE_EXL3_PREFILL_BF16_SUFFIXES = {
    "shared_gate_up": (".mlp.shared_experts.gate_up_proj",),
    "shared_down": (".mlp.shared_experts.down_proj",),
    "dense_gate_up": (".mlp.gate_up_proj",),
    "dense_down": (".mlp.down_proj",),
    "kda_in": (".self_attn.in_proj_qkvbfg_a",),
    "kda_o": (".self_attn.o_proj",),
    "mla_qkv_a": (".self_attn.fused_qkv_a_proj",),
    "mla_q_b": (".self_attn.q_b_proj",),
    "mla_o": (".self_attn.o_proj",),
}
# (prefix, module_type, bytes) per retained module, for the summary line.
_DENSE_EXL3_PREFILL_BF16_RETAINED: list[tuple[str, str, int]] = []


def _dense_exl3_prefill_bf16_types() -> set[str]:
    """Module types selected by GLM53_DENSE_EXL3_PREFILL_BF16.

    off | all | comma list of _DENSE_EXL3_PREFILL_BF16_SUFFIXES keys.
    start.sh resolves and forwards the pack-selected profile or the manual
    default (off). Backward compatible:
    GLM53_KDA_BF16_LARGE_M=1 (the phase-1 in_proj knob) selects kda_in.
    """
    raw = os.environ.get("GLM53_DENSE_EXL3_PREFILL_BF16", "off").strip().lower()
    if raw in ("", "off", "0", "no", "none"):
        types: set[str] = set()
    elif raw in ("all", "on", "1"):
        types = set(_DENSE_EXL3_PREFILL_BF16_SUFFIXES)
    else:
        types = {t.strip() for t in raw.split(",") if t.strip()}
        unknown = types - set(_DENSE_EXL3_PREFILL_BF16_SUFFIXES)
        if unknown:
            raise ValueError(
                "GLM53_DENSE_EXL3_PREFILL_BF16: unknown module "
                f"type(s) {sorted(unknown)}"
            )
    if kda_bf16_large_m_enabled():
        types.add("kda_in")
    return types


def _dense_exl3_prefill_bf16_type(
    prefix: str,
    types: set[str],
    layer_types: list[str] | None = None,
) -> str | None:
    """Module type of `prefix` (vLLM module path) if selected for retention."""
    if not types:
        return None
    if ".mtp" in prefix or "visual" in prefix or "draft" in prefix:
        return None
    m = re.search(r"\.layers\.(\d+)\.", prefix)
    layer_idx = int(m.group(1)) if m else None
    for mtype in sorted(types):
        if not any(prefix.endswith(s) for s in _DENSE_EXL3_PREFILL_BF16_SUFFIXES[mtype]):
            continue
        if mtype in ("dense_gate_up", "dense_down") and ".shared_experts." in prefix:
            continue
        if mtype in ("kda_in", "kda_o", "mla_o"):
            # o_proj exists on KDA and MLA layers; in_proj only on KDA.
            lt = _glm53_layer_types() if layer_types is None else layer_types
            if lt is None or layer_idx is None or layer_idx >= len(lt):
                continue
            is_kda = lt[layer_idx] == "linear_attention"
            if (mtype != "mla_o") != is_kda:
                continue
        return mtype
    return None



def _dense_exl3_forward_impl(x: torch.Tensor, handle: int) -> torch.Tensor:
    entry = _DENSE_EXL3_LAYERS[handle]
    linears = [lin for lin in entry["linears"] if lin is not None]
    bf16_weight = entry["bf16_weight"]
    x_fp16 = x.to(torch.float16).contiguous()
    if bf16_weight is None:
        if len(linears) == 1:
            return linears[0].forward(x_fp16, {}, out_dtype=torch.float16).to(x.dtype)
        return torch.cat(
            [lin.forward(x_fp16, {}, out_dtype=torch.float16) for lin in linears],
            dim=-1,
        ).to(x.dtype)
    # bf16 shards are a validated contiguous tail: one GEMM covers them and
    # the result stays in x.dtype exactly.
    outputs = [
        lin.forward(x_fp16, {}, out_dtype=torch.float16).to(x.dtype)
        for lin in linears
    ]
    outputs.append(F.linear(x, bf16_weight))
    return torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]


def _dense_exl3_forward_fake(x: torch.Tensor, handle: int) -> torch.Tensor:
    entry = _DENSE_EXL3_LAYERS[handle]
    out = sum(entry["output_sizes"])
    return x.new_empty((*x.shape[:-1], out), dtype=x.dtype)

_dense_exl3_op_registered = False


def _register_dense_exl3_op() -> None:
    global _dense_exl3_op_registered
    if _dense_exl3_op_registered:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="dense_exl3_forward",
        op_func=_dense_exl3_forward_impl,
        mutates_args=[],
        fake_impl=_dense_exl3_forward_fake,
    )
    _dense_exl3_op_registered = True


def _dense_exl3_warmup_autotune(capture_sizes: list[int] | None = None) -> int:
    """Eagerly tune every dense-EXL3 bitcoder GEMM before CUDA-graph capture.

    exllamav3's coop autotuner (coop_autotune.cu) tunes on first sight of a
    launch hash and synchronizes the stream; inside graph capture that
    deadlocks (py-spy 20260924T151053: capture -> shared_experts -> custom
    op -> BC run_alloc -> exl3_gemm -> tune -> cudaStreamSynchronize). The
    hash keys on MIN(roundup_pow2(rows), 16) plus per-module dims
    (exl3_gemm.cu:gemm_autotune_hash), so one warmup forward per unique
    module shape x row bucket covers every later capture. Buckets come
    from vLLM's cudagraph capture sizes <= LinearEXL3's
    AUTO_RECONSTRUCT_THRESHOLD (144; above it forward switches to the
    reconstruct path, which does not use the bitcoder). No capture sizes
    readable -> warm all five buckets (seconds either way). Returns the
    number of warmed GEMMs."""
    rows_buckets: set[int] = set()
    if capture_sizes:
        for s in capture_sizes:
            if 0 < s <= 144:
                rows_buckets.add(min(1 << (int(s) - 1).bit_length(), 16))
    if not rows_buckets:
        rows_buckets = {1, 2, 4, 8, 16}
    if not _DENSE_EXL3_LAYERS or not torch.cuda.is_available():
        return 0
    device = next(
        lin.trellis.device
        for entry in _DENSE_EXL3_LAYERS
        for lin in entry["linears"]
        if lin is not None
    )
    warmed: set[tuple] = set()
    count = 0
    for entry in _DENSE_EXL3_LAYERS:
        for lin in entry["linears"]:
            if lin is None:
                continue
            # LinearEXL3 stores mcg as a bool (mcg_tensor is not None)
            cb = "mcg" if lin.mcg else "mul1"
            key = (str(device), int(lin.in_features), int(lin.out_features),
                   int(getattr(lin, "K", 0) or 0), cb)
            for rows in sorted(rows_buckets):
                if (key, rows) in warmed:
                    continue
                x = torch.zeros(rows, lin.in_features, dtype=torch.float16, device=device)
                lin.forward(x, {}, out_dtype=torch.float16)
                warmed.add((key, rows))
                count += 1
    torch.cuda.synchronize(device)
    logger.info(
        "[dense-exl3] coop-autotune warmup: %d GEMMs (%d row buckets %s) "
        "tuned before graph capture",
        count, len(rows_buckets), sorted(rows_buckets),
    )
    return count


# vLLM's QKVParallelLinear stacked mapping passes string shard ids
# ("q"/"k"/"v"); MergedColumnParallelLinear passes ints. The DFlash2 draft's
# qkv_proj is the first QKVParallelLinear on this path (the target's
# MLA/KDA linears never exercise string ids).
_QKV_SHARD_IDS = {"q": 0, "k": 1, "v": 2}


class Exl3LinearMethod(LinearMethodBase):
    """Non-routed (dense) EXL3 linear method for attention/MLP dense projections.

    Ported from Alexbob0/glm53-flash-dense-exl3-tp2 (MIT; itself from
    vcruz305/vllm-exl3, validated TP=1 there) with: TP from
    ``layer.tp_rank/tp_size`` (so ``disable_tp`` replicated layers such as
    DeepSeekV2FusedQkvAProjLinear load unsharded), both mcg and mul1 codebook
    markers accepted (turboderp dense tensors are mul1), and mixed BF16
    shards (KDA in_proj b/f_a/g_a) staged next to the EXL3 shards.
    """

    def __init__(self, quant_config: "Exl3Config", prefix: str, bits: int) -> None:
        self.quant_config = quant_config
        self.prefix = prefix
        self.bits = int(bits)
        self._loaded_parts: set[tuple[int, str]] = set()
        _DENSE_EXL3_MODULES.append((prefix, self.bits))

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from vllm.model_executor.layers.linear import RowParallelLinear

        self.n_shards = len(output_partition_sizes)
        self.output_sizes = list(output_partition_sizes)
        self.bf16_shards = self.quant_config._bf16_shards_for(self.prefix)
        self.is_row_parallel = isinstance(layer, RowParallelLinear)
        if self.bf16_shards and self.bf16_shards[-1] != self.n_shards - 1:
            # the forward op appends the bf16 rows as one contiguous tail GEMM
            raise ValueError(
                f"non_routed_exl3 bf16_shards {self.bf16_shards} for "
                f"{self.prefix} must be the shard tail "
                f"(n_shards={self.n_shards})"
            )
        # Per-layer TP geometry: replicated (disable_tp) layers report
        # tp_size == 1 whatever the world size, which is exactly the slicing
        # the checkpoint expects for them. Replicated shard ids (KDA
        # in_proj_qkvbfg_a f_a/g_a) load the per-rank width as-is.
        self.tp_rank = int(getattr(layer, "tp_rank", 0) or 0)
        self.tp_size = int(getattr(layer, "tp_size", 1) or 1)
        if self.tp_size == 3:
            raise ValueError("dense EXL3 does not support TP=3 trellis/head padding; use TP=2")
        self.replicated_shards = frozenset(getattr(layer, "replicated_shard_ids", ()) or ())
        # vLLM's LinearBase stamps the GLOBAL tp_rank/tp_size even on
        # ReplicatedLinear (disable_tp "take[s] no effect for replicated
        # linear layers"): the weight is duplicated per rank, nothing is
        # sliced, and output_partition_sizes already reports the FULL
        # output — sum(output_sizes) == output_size. Loading such a layer
        # with the world tp_size would column-shard the full checkpoint
        # tensor against a full-size destination (the DFlash2 draft's fc
        # and conv kernel_projection hit exactly this at draft TP=2:
        # dest (4096,) vs loaded (2048,)). Reconcile to the replicated
        # geometry, like vLLM's own replicated weight_loader. RowParallel
        # also reports the full output but genuinely shards the input, so
        # it is excluded; genuinely column-sharded layers partition the
        # output (sum < output_size) and keep the world geometry.
        if (
            not self.is_row_parallel
            and self.tp_size > 1
            and sum(self.output_sizes) == output_size
        ):
            self.tp_rank, self.tp_size = 0, 1
        self.in_per_partition = input_size_per_partition
        # [dense-exl3] The ParallelLMHead is the only non-LinearBase layer
        # this method serves: vLLM shards it as one contiguous block of the
        # PADDED vocab per rank — exactly the column-shard narrow the loader
        # already runs, as long as padded == org. GLM-5.3-Flash's vocab
        # (154880) is a 64 multiple, so vLLM pads nothing (77440 rows/rank
        # at TP2); refuse an actually-padded vocab rather than guess at
        # pad-row encodings.
        self.is_vocab_head = not isinstance(layer, LinearBase)
        if self.is_vocab_head and output_size != int(
            getattr(layer, "org_vocab_size", -1)
        ):
            raise ValueError(
                f"[dense-exl3] {self.prefix}: padded vocab {output_size} != "
                f"org vocab {getattr(layer, 'org_vocab_size', None)}; the "
                "EXL3 lm_head requires a 64-aligned vocab"
            )

        k_words = self.bits * 16
        for i, out_size in enumerate(self.output_sizes):
            if i in self.bf16_shards:
                continue
            if self.in_per_partition % 16 or out_size % 16:
                raise ValueError(
                    "EXL3 trellis tiles are 16-wide; "
                    f"shard {i}: in={self.in_per_partition} out={out_size}"
                )

        in_tiles = self.in_per_partition // 16
        total_out_tiles = sum(
            s // 16 for i, s in enumerate(self.output_sizes) if i not in self.bf16_shards
        )

        bf16_rows = sum(self.output_sizes[i] for i in self.bf16_shards)
        params = {
            "trellis": Parameter(
                torch.empty(in_tiles, total_out_tiles, k_words, dtype=torch.int16),
                requires_grad=False,
            ),
            "suh": Parameter(
                torch.empty(self.n_shards, self.in_per_partition, dtype=torch.float16),
                requires_grad=False,
            ),
            "svh": Parameter(
                torch.empty(sum(self.output_sizes), dtype=torch.float16),
                requires_grad=False,
            ),
            "mcg": Parameter(
                torch.zeros(self.n_shards, 1, dtype=torch.int32), requires_grad=False
            ),
            "mul1": Parameter(
                torch.zeros(self.n_shards, 1, dtype=torch.int32), requires_grad=False
            ),
            "weight": Parameter(
                torch.empty(bf16_rows, self.in_per_partition, dtype=params_dtype),
                requires_grad=False,
            ),
        }
        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        for suffix, param in params.items():
            layer.register_parameter(suffix, param)
            set_weight_attrs(param, extra)
            param.weight_loader = functools.partial(self._load_exl3, suffix)

        # ABLIT reads this marker at load time (overlay/ablit_runtime.py).
        layer._exl3_linear_n_shards = self.n_shards

    def _load_exl3(
        self,
        suffix: str,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id=None,
    ) -> None:
        if loaded_shard_id is None:
            shard_idx = 0
        elif isinstance(loaded_shard_id, str):
            try:
                shard_idx = _QKV_SHARD_IDS[loaded_shard_id]
            except KeyError:
                raise ValueError(
                    f"unknown EXL3 shard id {loaded_shard_id!r} for "
                    f"{self.prefix}"
                ) from None
        else:
            shard_idx = int(loaded_shard_id)
        if not 0 <= shard_idx < self.n_shards:
            raise ValueError(
                f"shard_idx={shard_idx} out of range for n_shards={self.n_shards}"
            )
        # Effective TP per shard: replicated shards load as-is.
        eff_rank = 0 if shard_idx in self.replicated_shards else self.tp_rank
        eff_tp = 1 if shard_idx in self.replicated_shards else self.tp_size

        if suffix == "weight":
            # BF16 staging: keep declared bf16 shards, discard the stale
            # BF16 copy of EXL3-replaced shards.
            if shard_idx not in self.bf16_shards:
                return
            expected_out = self.output_sizes[shard_idx]
            loaded = loaded_weight.detach()
            if loaded.shape[0] != expected_out:
                # TP column-sharded shard: the checkpoint holds the full
                # width; slice this rank's rows out of it.
                if (not self.is_row_parallel and eff_tp > 1
                        and loaded.shape[0] == expected_out * eff_tp):
                    loaded = loaded[
                        eff_rank * expected_out : (eff_rank + 1) * expected_out
                    ]
                else:
                    raise RuntimeError(
                        f"EXL3 bf16 shard {shard_idx} shape mismatch: "
                        f"expected out={expected_out} (tp {eff_rank}/{eff_tp}) "
                        f"got {tuple(loaded_weight.shape)}"
                    )
            bf16_idx = self.bf16_shards.index(shard_idx)
            row_start = sum(self.output_sizes[i] for i in self.bf16_shards[:bf16_idx])
            param.data[row_start : row_start + expected_out].copy_(loaded)
            self._loaded_parts.add((shard_idx, suffix))
            return
        if suffix in ("mcg", "mul1"):
            dest = param.data[shard_idx]
            loaded_val = (
                loaded_weight.detach().reshape(-1)[0].item()
                if loaded_weight.numel() > 0
                else 0
            )
            dest[0] = int(loaded_val)
            self._loaded_parts.add((shard_idx, suffix))
            return
        if shard_idx in self.bf16_shards:
            # EXL3 packed part of a shard declared BF16: the overlay
            # never ships these; refuse instead of guessing.
            raise RuntimeError(
                f"EXL3 linear shard {shard_idx} is declared bf16 but a "
                f"{suffix} tensor arrived for it"
            )

        loaded = loaded_weight.detach().contiguous()
        if self.is_row_parallel:
            sharded = shard_exl3_row(loaded, suffix, eff_rank, eff_tp)
        else:
            sharded = shard_exl3_col(loaded, suffix, eff_rank, eff_tp)

        if suffix == "trellis":
            out_tiles_start = sum(
                s // 16
                for i, s in enumerate(self.output_sizes[:shard_idx])
                if i not in self.bf16_shards
            )
            out_tiles_end = out_tiles_start + self.output_sizes[shard_idx] // 16
            dest = param.data[:, out_tiles_start:out_tiles_end, :]
        elif suffix == "suh":
            dest = param.data[shard_idx]
        elif suffix == "svh":
            out_start = sum(self.output_sizes[:shard_idx])
            dest = param.data[out_start : out_start + self.output_sizes[shard_idx]]
        else:
            raise ValueError(f"unknown EXL3 suffix={suffix}")

        if tuple(dest.shape) != tuple(sharded.shape):
            raise RuntimeError(
                f"EXL3 linear load shape mismatch shard={shard_idx} "
                f"suffix={suffix}: dest {tuple(dest.shape)} != "
                f"loaded {tuple(sharded.shape)} (tp {self.tp_rank}/{self.tp_size})"
            )
        dest.copy_(sharded)
        self._loaded_parts.add((shard_idx, suffix))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "trellis"):
            return
        if self.is_vocab_head:
            # [dense-exl3] LogitsProcessor routes a head_dtype that differs
            # from the activation dtype through an unquantized-only branch
            # that raises for quant methods — refuse at load, not at the
            # first request (the recipe never sets head_dtype; an
            # --hf-overrides fp32 head is the only way to hit this).
            from vllm.config import get_current_vllm_config

            mc = get_current_vllm_config().model_config
            head_dtype = getattr(mc, "head_dtype", None)
            if head_dtype is not None and head_dtype != getattr(mc, "dtype", None):
                raise RuntimeError(
                    f"[dense-exl3] {self.prefix}: head_dtype={head_dtype} "
                    "needs the unquantized lm_head; unset it or drop the "
                    "lm_head key from the pack"
                )
        output_sizes = self.output_sizes
        bf16_shards = self.bf16_shards

        mcg_vals = layer.mcg.reshape(-1).tolist()
        mul1_vals = layer.mul1.reshape(-1).tolist()
        for i in range(self.n_shards):
            if i in bf16_shards:
                continue
            mcg_set = mcg_vals[i] != 0
            mul1_set = mul1_vals[i] != 0
            if mcg_set == mul1_set:
                raise RuntimeError(
                    f"EXL3 linear shard {i}: exactly one codebook marker must "
                    f"be set (mcg={mcg_vals[i]}, mul1={mul1_vals[i]})"
                )
            if mcg_set and mcg_vals[i] != MCG_MARKER_SIGNED_INT32:
                raise RuntimeError(
                    f"EXL3 linear shard {i}: bad mcg marker {mcg_vals[i]}"
                )
            if mul1_set and mul1_vals[i] != MUL1_MARKER_SIGNED_INT32:
                raise RuntimeError(
                    f"EXL3 linear shard {i}: bad mul1 marker {mul1_vals[i]}"
                )
        required = {
            (i, part) for i in range(self.n_shards)
            for part in (("weight",) if i in bf16_shards else ("trellis", "suh", "svh"))
        }
        missing = required - self._loaded_parts
        if missing:
            raise RuntimeError(f"EXL3 {self.prefix}: missing weight parts {sorted(missing)}")

        linears: list = []
        tile_start = 0
        out_start = 0
        for i in range(self.n_shards):
            if i in bf16_shards:
                linears.append(None)
                out_start += output_sizes[i]  # svh is indexed by absolute output offset
                continue
            tiles = output_sizes[i] // 16
            trellis_shard = layer.trellis[:, tile_start : tile_start + tiles, :].contiguous()
            tile_start += tiles
            suh_shard = layer.suh[i].contiguous()
            svh_shard = layer.svh[
                out_start : out_start + output_sizes[i]
            ].contiguous()
            out_start += output_sizes[i]
            mcg_shard = layer.mcg[i].contiguous() if mcg_vals[i] else None
            mul1_shard = layer.mul1[i].contiguous() if mul1_vals[i] else None
            linears.append(
                make_linear_exl3(
                    trellis_shard,
                    suh_shard,
                    svh_shard,
                    mcg_shard,
                    mul1_shard,
                    out_dtype=torch.float16,
                )
            )

        bf16_weight = layer.weight.data if bf16_shards else None
        self._retain_bf16_large_m(layer, linears, bf16_weight)
        for name in ("trellis", "suh", "svh", "mcg", "mul1", "weight"):
            delattr(layer, name)
        _register_dense_exl3_op()
        _DENSE_EXL3_LAYERS.append(
            {
                "linears": linears,
                "bf16_shards": list(bf16_shards),
                "output_sizes": list(output_sizes),
                "bf16_weight": bf16_weight,
            }
        )
        layer._exl3_dense_handle = len(_DENSE_EXL3_LAYERS) - 1
        if self.is_vocab_head:
            logger.info(
                "[dense-exl3] %s active (K=%d, vocab shard %d x %d, custom op)",
                self.prefix,
                self.bits,
                self.output_sizes[0],
                self.in_per_partition,
            )
        else:
            logger.info(
                "[dense-exl3] %s active (K=%d, %d shards, bf16_shards=%s, custom op)",
                self.prefix,
                self.bits,
                self.n_shards,
                bf16_shards or "-",
            )
        if len(_DENSE_EXL3_LAYERS) >= len(_DENSE_EXL3_MODULES):
            # Last constructed module has loaded: one summary line for the
            # boot log. declared == built is already enforced (Exl3MoEMethod
            # calls _assert_non_routed_built); this asserts built == loaded.
            assert len(_DENSE_EXL3_LAYERS) == len(_DENSE_EXL3_MODULES), (
                len(_DENSE_EXL3_LAYERS),
                len(_DENSE_EXL3_MODULES),
            )
            k_hist: dict[int, int] = {}
            for _, k in _DENSE_EXL3_MODULES:
                k_hist[k] = k_hist.get(k, 0) + 1
            logger.info(
                "[dense-exl3] %d EXL3-dense modules loaded %s",
                len(_DENSE_EXL3_LAYERS),
                {f"K{k}": n for k, n in sorted(k_hist.items())},
            )
            if _DENSE_EXL3_PREFILL_BF16_RETAINED:
                by_type: dict[str, int] = {}
                total_bytes = 0
                for _, mtype, nbytes in _DENSE_EXL3_PREFILL_BF16_RETAINED:
                    by_type[mtype] = by_type.get(mtype, 0) + 1
                    total_bytes += nbytes
                logger.info(
                    "[dense-exl3] prefill-bf16 retention: %d modules %s, "
                    "%.2f GiB/rank (M>%d)",
                    len(_DENSE_EXL3_PREFILL_BF16_RETAINED),
                    {t: n for t, n in sorted(by_type.items())},
                    total_bytes / 2**30,
                    DENSE_EXL3_PREFILL_BF16_MIN_M,
                )
            from vllm.config import get_current_vllm_config

            comp = getattr(get_current_vllm_config(), "compilation_config", None)
            _dense_exl3_warmup_autotune(
                list(getattr(comp, "cudagraph_capture_sizes", None) or [])
            )

    def _retain_bf16_large_m(
        self, layer: torch.nn.Module, linears: list, bf16_weight: torch.Tensor | None
    ) -> None:
        """Load-time BF16 copy for the dense-EXL3 prefill path
        (GLM53_DENSE_EXL3_PREFILL_BF16).

        One [sum(output_sizes) x K] BF16 tensor per selected module: every
        EXL3 shard's get_weight_tensor() (hadamards pre-applied, so exactly
        the weight the reconstruct path multiplies by) concatenated with
        the bf16 tail rows in output_sizes order. Rows > 144 then run a
        single F.linear in x.dtype — the same logical weights as the custom
        op, differing only in the fp16->bf16 rounding of the reconstructed
        rows and in accumulation order. Rows <= 144 stay on the EXL3 custom
        op. Cost: 2 B/weight per rank per selected module (TP2 kda_in
        12576x4096x2 B = 98.25 MiB/layer-rank, ~3.26 GiB/rank over 34
        layers)."""
        # Retention is target-model dense projections only: never the EXL3
        # lm_head (a vocab-parallel head, not a dense projection) and never
        # DFlash2 draft modules — their runtime prefixes carry no "draft"
        # marker and an offset draft mlp suffix-matches dense_gate_up/down,
        # so the draft quant config's prefix-offset marker is the guard.
        if getattr(self, "is_vocab_head", False):
            return
        if self.quant_config.scope == "dflash2_draft" or getattr(self.quant_config, "_draft_prefix_offset", 0):
            return
        mtype = _dense_exl3_prefill_bf16_type(
            self.prefix, _dense_exl3_prefill_bf16_types())
        if mtype is None:
            return
        # Concatenate in output_sizes order; the bf16 shards are a
        # validated contiguous tail (create_weights), so their rows of the
        # staged weight follow the EXL3 rows.
        # Allocate once; release each reconstructed shard before the next.
        w = None
        tail_row = out_row = 0
        for i, lin in enumerate(linears):
            rows = self.output_sizes[i]
            if lin is not None:
                part = lin.get_weight_tensor().t()
            else:
                if bf16_weight is None:
                    raise RuntimeError(
                        f"[dense-exl3] {self.prefix}: shard {i} is bf16 but "
                        "no staged bf16 weight exists"
                    )
                part = bf16_weight[tail_row:tail_row + rows]
                tail_row += rows
            if w is None:
                w = part.new_empty((sum(self.output_sizes), self.in_per_partition),
                                   dtype=torch.bfloat16)
            w[out_row:out_row + rows].copy_(part)
            out_row += rows
            del part
        n, k = w.shape
        if k != self.in_per_partition:
            raise RuntimeError(
                f"[dense-exl3] prefill-bf16 {self.prefix}: reconstructed "
                f"input dim {k} != per-partition {self.in_per_partition}"
            )
        layer.glm53_bf16_lm_w = w
        layer.glm53_bf16_lm_n = n
        layer.glm53_bf16_lm_k = k
        # Resolved once: capture and replay cannot disagree about the branch.
        layer.glm53_bf16_lm_min_m = DENSE_EXL3_PREFILL_BF16_MIN_M
        nbytes = w.numel() * w.element_size()
        _DENSE_EXL3_PREFILL_BF16_RETAINED.append((self.prefix, mtype, nbytes))
        logger.info(
            "[dense-exl3] prefill-bf16 retained %s: type=%s [%dx%d] "
            "+%.1f MiB/rank (M>%d), dtype=bf16, source=exl3-reconstruct",
            self.prefix, mtype, n, k, nbytes / 2**20,
            DENSE_EXL3_PREFILL_BF16_MIN_M,
        )

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        handle = getattr(layer, "_exl3_dense_handle", None)
        if handle is None:
            raise RuntimeError("EXL3 linear layers were not built after weight load")
        # Hybrid dispatch: M is tensor metadata (no host sync); the retained
        # copy exists only on selected modules. Per-capture-size CUDA graphs
        # bake the branch taken at capture. Rows > 144: one BF16 GEMM in
        # x.dtype — exactly F's LARGE_M arithmetic; rows <= 144: the custom
        # op (bitcoder), unchanged.
        w = getattr(layer, "glm53_bf16_lm_w", None)
        if w is not None and x.dtype == w.dtype and bias is None and x.dim() >= 2:
            k = int(layer.glm53_bf16_lm_k)
            if int(x.shape[-1]) == k:
                m = x.numel() // k
                if m > int(layer.glm53_bf16_lm_min_m):
                    _kda_large_m_note("bf16", m)
                    y = F.linear(x.reshape(-1, k), w)
                    return y.reshape(x.shape[:-1] + (int(layer.glm53_bf16_lm_n),))
        y = torch.ops.vllm.dense_exl3_forward(x, handle)
        if bias is not None:
            y = y + bias
        return y


class Exl3MoEMethod(FusedMoEMethodBase):
    """Packed MCG trellis experts: create/load packed tensors, LinearEXL3 apply."""

    def __init__(self, moe, quant_config: Exl3Config) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.bits = quant_config.bits
        self._logged = False

    def get_fused_moe_quant_config(self, layer: "RoutedExperts") -> FusedMoEQuantConfig | None:
        return None

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError(
                "EXL3 trellis tiles are 16-wide; "
                f"hidden={hidden_size} intermediate_local={intermediate_size_per_partition}"
            )
        k_words = self.bits * 16
        in_tiles = hidden_size // 16
        out_tiles = intermediate_size_per_partition // 16

        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}

        # w13_* : stacked [expert, {gate=0, up=1}, ...] so the stock
        # expert_params_mapping (experts.w13_ + suffix) hits these names.
        w13_trellis = Parameter(
            torch.empty(
                num_experts, 2, in_tiles, out_tiles, k_words, dtype=torch.int16
            ),
            requires_grad=False,
        )
        w13_suh = Parameter(
            torch.empty(num_experts, 2, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w13_svh = Parameter(
            torch.empty(
                num_experts, 2, intermediate_size_per_partition, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w13_mcg = Parameter(
            torch.empty(num_experts, 2, 1, dtype=torch.int32),
            requires_grad=False,
        )
        w2_trellis = Parameter(
            torch.empty(
                num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16
            ),
            requires_grad=False,
        )
        w2_suh = Parameter(
            torch.empty(
                num_experts, intermediate_size_per_partition, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w2_svh = Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w2_mcg = Parameter(
            torch.empty(num_experts, 1, dtype=torch.int32),
            requires_grad=False,
        )

        packed = {
            "w13_trellis": w13_trellis,
            "w13_suh": w13_suh,
            "w13_svh": w13_svh,
            "w13_mcg": w13_mcg,
            "w2_trellis": w2_trellis,
            "w2_suh": w2_suh,
            "w2_svh": w2_svh,
            "w2_mcg": w2_mcg,
        }
        for name, param in packed.items():
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra)
            param.weight_loader = self._load_exl3
            param._exl3_owner = layer
        if hasattr(layer, "w13_weight") or hasattr(layer, "w2_weight"):
            raise RuntimeError("EXL3 create_weights must not allocate dense expert weights")

        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_local = intermediate_size_per_partition
        layer._exl3_k_words = k_words
        layer._exl3_bits = self.bits

    def _load_exl3(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str = "w1",
        expert_id: int = 0,
        return_success: bool = False,
    ) -> bool | None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        layer = param
        # param is the Parameter; expert_id is already physical. Map to local
        # via the owning module if present on the weight_loader closure... we
        # look up from param's __dict__ after register. RoutedExperts.weight_loader
        # maps global→local; glm5next calls *our* loader, so map here.
        owner = getattr(param, "_exl3_owner", None)
        if owner is not None:
            local_id = owner._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
            expert_id = local_id

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        suffix = _suffix_from_mapped_name(weight_name)
        loaded = loaded_weight.detach().contiguous()

        if shard_id in ("w1", "w3"):
            shard_idx = 0 if shard_id == "w1" else 1
            sharded = shard_exl3_col(loaded, suffix, tp_rank, tp_size)
            dest = param.data[expert_id, shard_idx]
        elif shard_id == "w2":
            sharded = shard_exl3_row(loaded, suffix, tp_rank, tp_size)
            dest = param.data[expert_id]
        else:
            raise ValueError(f"unknown EXL3 shard_id={shard_id}")

        if tuple(dest.shape) != tuple(sharded.shape):
            raise RuntimeError(
                f"EXL3 load shape mismatch {weight_name} shard={shard_id} "
                f"expert={expert_id}: dest {tuple(dest.shape)} != "
                f"loaded {tuple(sharded.shape)}"
            )
        dest.copy_(sharded)
        return True if return_success else None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "w13_trellis"):
            return
        # [dense-exl3] First process call: every module was constructed long
        # before weight load, so a declared-but-unbuilt dense module is a
        # pack/model prefix mismatch serving empty BF16 weights — refuse.
        global _dense_exl3_build_checked
        if not _dense_exl3_build_checked:
            _dense_exl3_build_checked = True
            self.quant_config._assert_non_routed_built()
        # Bind owner for any late loads; stitch LinearEXL3 handles.
        for name in (
            "w13_trellis",
            "w13_suh",
            "w13_svh",
            "w13_mcg",
            "w2_trellis",
            "w2_suh",
            "w2_svh",
            "w2_mcg",
        ):
            getattr(layer, name)._exl3_owner = layer

        mcg13 = layer.w13_mcg.reshape(-1)
        mcg2 = layer.w2_mcg.reshape(-1)
        if not torch.all(mcg13 == MCG_MARKER_SIGNED_INT32) or not torch.all(
            mcg2 == MCG_MARKER_SIGNED_INT32
        ):
            raise RuntimeError(
                "EXL3 mcg marker is not the MCG int32 0xCBAC1FED / "
                f"{MCG_MARKER_SIGNED_INT32}; packed ABI mismatch"
            )

        n_exp = int(layer.w13_trellis.shape[0])
        layer._exl3_shared_w13_suh = bool(
            torch.equal(layer.w13_suh[:, 0], layer.w13_suh[:, 1])
        )
        inners: list[dict[str, Any]] = []
        for e in range(n_exp):
            gate = make_linear_exl3(
                layer.w13_trellis[e, 0],
                layer.w13_suh[e, 0],
                layer.w13_svh[e, 0],
                layer.w13_mcg[e, 0],
            )
            up = make_linear_exl3(
                layer.w13_trellis[e, 1],
                layer.w13_suh[e, 1],
                layer.w13_svh[e, 1],
                layer.w13_mcg[e, 1],
            )
            down = make_linear_exl3(
                layer.w2_trellis[e],
                layer.w2_suh[e],
                layer.w2_svh[e],
                layer.w2_mcg[e],
            )
            inners.append({"gate": gate, "up": up, "down": down})
        layer._exl3_inners = inners
        # Tier resolution needs the LinearEXL3 handles (K/mcg/mul1) for the
        # grouped eligibility check, so it runs once the inners exist.
        _record_exl3_fat_resolution(layer)
        fused_ok = False
        fused_err = None
        if fused_moe_enabled():
            try:
                import exllamav3_ext

                if hasattr(exllamav3_ext, "exl3_moe"):
                    build_exl3_fused_state(layer, inners)
                    fused_ok = True
                else:
                    fused_err = "exllamav3_ext.exl3_moe missing"
            except Exception as exc:
                fused_err = repr(exc)
                layer._exl3_ptrs = None
        if exl3_moe_fast_requested() and not fused_ok:
            # Fail closed: an explicitly requested fast thin-decode path must
            # never silently run the stock kernel or the Python loop. The
            # version gate inside build_exl3_fused_state raises through the
            # same path; this also covers fused disabled / exl3_moe missing.
            raise RuntimeError(
                "GLM53_EXL3_MOE_FAST=1 requires the fused exl3_moe path on an "
                "image built with overlay/patch_exl3_decode_pipeline.py; "
                f"load-time setup failed: {fused_err or 'EXL3_FUSED_MOE=0'}"
            )
        if not self._logged:
            if fused_ok:
                logger.info(
                    "EXL3 MCG trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=exl3_moe concurrency=%s "
                    "(no BF16 expert reconstruct at load)",
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    getattr(layer, "_exl3_fused_concurrency", "?"),
                )
            else:
                logger.info(
                    "EXL3 MCG trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=python_loop (%s) "
                    "(no BF16 expert reconstruct at load)",
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    fused_err or "EXL3_FUSED_MOE=0",
                )
            self._logged = True

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        limit = getattr(self.moe, "swiglu_limit", None) or SWIGLU_LIMIT_DEFAULT
        return apply_exl3_experts(
            x, topk_ids, topk_weights, layer, limit=float(limit)
        )
