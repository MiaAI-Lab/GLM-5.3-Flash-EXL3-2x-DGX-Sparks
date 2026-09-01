# SPDX-License-Identifier: Apache-2.0
"""EXL3/MCG trellis quantization for GLM-5.3-Flash routed experts.

Checkpoint ABI (brandonmusic/GLM-5.3-Flash-tr3-4bpw):
  quant_method=exl3, codebook=mcg, scope=glm53_routed_experts_only
  per expert matrix: trellis (int16) + suh/svh (fp16) + mcg (int32 marker)

Non-routed tensors stay native (UnquantizedLinearMethod). Experts never
expand to a persistent BF16 weight; LinearEXL3 / exllamav3_ext runs the
trellis GEMM. TP=2 shards gate/up column-wise and down row-wise; the MoE
runner all-reduces the combined output.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
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
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
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
_FAT_BUCKET_EDGES = (16, 32, 64, 128, 256, 512, 1024, 2048)
_FAT_STATS: dict[str, Any] = {
    "layers": 0,
    "fat_layers": 0,
    "fat_experts": 0,
    "max_rows": 0,
    "sum_max_rows": 0,
    "hist": [0] * (len(_FAT_BUCKET_EDGES) + 1),
}
_FAT_TIERS = ("kernel", "batched", "sorted", "legacy")
# Machine-checkable E2 (fat-expert) diagnostics. Load-time fields mirror the
# most recent weight load; counters are monotonic per process and are the
# ground truth for what actually ran. The key set is contract — extend it only
# together with a bump of EXL3_FAT_DIAG_SCHEMA.
EXL3_FAT_DIAG_SCHEMA = 1
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


def fat_expert_log_enabled() -> bool:
    return os.environ.get("EXL3_FAT_EXPERT_LOG", "1") != "0"

def configured_fat_tier() -> str:
    """Highest fat tier the env requests: kernel > batched > sorted > legacy."""
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
) -> tuple[str, str]:
    """Map the configured fat tier onto what this image + checkpoint can run.

    Order matters: the checkpoint cap comes first. E1 batched and the E2
    kernel run gate+up as one stacked GEMM behind a single input Hadamard
    (gate.suh), so a checkpoint without shared SUH caps the tier at sorted —
    a legitimate lower tier that needs no fat symbols, whatever the image.
    Only when the kernel would actually run does a missing symbol become an
    image/flag mismatch, failing closed here at load instead of mid-prefill.
    """
    configured = configured_fat_tier()
    if configured == "legacy":
        return configured, "none_requested"
    if not shared_suh and configured in ("kernel", "batched"):
        return "sorted", "shared_suh_absent"
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
    return configured, f"{configured}_ok"


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
    effective_tier, tier_reason = resolve_exl3_fat_tier(shared_suh)
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
    ):
        diag[key] = 0
    diag["fallback_calls"] = {tier: 0 for tier in _FAT_TIERS}
    diag["fallback_reasons"] = {}
    diag["fat_scratch_bytes"] = sum(_FAT_SCRATCH_BYTES.values())
    diag["fat_scratch_peak_bytes"] = diag["fat_scratch_bytes"]


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
    mcg: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float16,
):
    """Build a LinearEXL3 over already-sharded packed tensors. No BF16 expand."""
    cls = load_linear_exl3_cls()
    return cls(
        config=None,
        in_features=int(suh.numel()),
        out_features=int(svh.numel()),
        trellis=trellis.contiguous(),
        suh=suh.contiguous(),
        svh=svh.contiguous(),
        mcg=mcg.contiguous(),
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


def build_exl3_fused_state(layer: torch.nn.Module, inners: list[dict[str, Any]]) -> None:
    """Pointer tables + fused temps, once after load. No per-token alloc."""
    import exllamav3_ext

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
    want_fat_kernel = fat_kernel_enabled()
    want_batched_fat = batched_fat_fallback_enabled() or want_fat_kernel
    use_sorted_fat = sorted_fat_fallback_enabled() or want_batched_fat
    use_batched_fat = (
        want_batched_fat
        and bool(getattr(layer, "_exl3_shared_w13_suh", False))
    )
    use_fat_kernel = use_batched_fat and want_fat_kernel
    use_row_tiles = fused_moe_row_tile_enabled()
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


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """Routed-experts-only EXL3/MCG. Dense / shared / attention stay native."""

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

        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(layer.moe_config, self)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None


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
        _record_exl3_fat_resolution(layer)
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
