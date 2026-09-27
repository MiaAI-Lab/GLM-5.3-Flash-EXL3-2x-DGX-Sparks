#!/usr/bin/env python3
"""Cold-load fixes for unified-memory GB10 hosts with 64 KiB kernel pages.

Two independent problems, both in ``model_executor/model_loader/weight_utils.py``:

1. InstantTensor budget on UMA (``instanttensor_weights_iterator``).
   InstantTensor sizes its pinned ring buffer against
   ``torch.cuda.mem_get_info()[0] * max_free_mem_usage`` and raises when the
   buffer does not fit. On DGX Spark ``mem_get_info`` free is host ``MemFree``:
   page cache counts as *used*. After a 164 GiB rsync/read the head reports
   ~2 GiB free and the load dies before the first byte is read (receipt in
   logs/head.log 2026-09-18 19:36: ``buffer_size (1268776960 B) exceeds device
   memory budget (1255964672 B)``), or survives with ``io_depth`` shrunk from
   512 to double digits and loads at a fraction of the NVMe ceiling.

   Fix: before opening, measure ``MemAvailable`` (clean page cache is
   reclaimable on demand, so it — not ``MemFree`` — is what the load can
   actually use) and pin an explicit ``max_free_mem_usage`` / ``buffer_size``
   against that window. The fraction may exceed 1 relative to cuda free:
   InstantTensor treats it as pure arithmetic against ``mem_get_info`` and
   the pinned allocation reclaims clean cache, so ``io_depth`` stays at the
   backend default even with a full page cache. Containers cannot drop
   caches (``/proc/sys`` is read-only without ``CAP_SYS_ADMIN``); the
   in-container attempt is kept as a harmless no-op and the budget no longer
   depends on it. The launcher forwards ``INSTANTTENSOR_*`` to both ranks
   when set, and set values are used, with one exception (#273): a bare
   ``INSTANTTENSOR_MAX_FREE_MEM_USAGE`` (no ``INSTANTTENSOR_BUFFER_SIZE``)
   that cannot cover the pinned buffer while ``MemAvailable`` holds the load
   is raised to the MemAvailable-sized fraction, with a warning. It is never
   lowered. An explicit pair is used as given, which does not guarantee the
   load fits.

2. File-backed 64 KiB mmap sources (``safetensors_weights_iterator``).
   ``cuMemcpyHtoDAsync`` wedges on this GB10 driver when the source is a
   file-backed mapping on a 64 KiB-page kernel (rocket fork, weight_utils.py
   ``# Stage off the file mmap into anonymous memory``). Every non-InstantTensor
   safetensors path (``LOAD_FORMAT=`` auto, the draft model when the loader
   falls back, secondary weights) goes through this iterator. Fix: when the
   page size is not 4 KiB, ``clone()`` each tensor off the mmap into anonymous
   memory before yielding it. On 4 KiB kernels the iterator is byte-identical
   to stock.

Idempotent; fails closed on anchor drift. Kill switch: ``GLM53_COLD_LOAD_UMA=0``
leaves the file untouched (logged).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_NAME = "GLM53_COLD_LOAD_UMA"
TARGET = Path(
    os.environ.get(
        "GLM53_WEIGHT_UTILS_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "model_loader/weight_utils.py",
    )
)
MARK = "# [glm53-cold-load-uma:v1]"

# --- 1. InstantTensor budget -------------------------------------------------
ANCHOR_IT_OPEN = (
    "    # copy=True yields tensors that own their memory, staying valid after the\n"
    "    # context exits or InstantTensor reuses its buffer.\n"
    "    with instanttensor.safe_open(\n"
    "        hf_weights_files,\n"
    "        framework=\"pt\",\n"
    "        device=device,\n"
    "        process_group=process_group,\n"
    "        copy=True,\n"
    "    ) as f:\n"
)
NEW_IT_OPEN = (
    "    " + MARK + "\n"
    "    _glm53_uma_prepare_instanttensor_budget(hf_weights_files)\n"
    "    # copy=True yields tensors that own their memory, staying valid after the\n"
    "    # context exits or InstantTensor reuses its buffer.\n"
    "    with instanttensor.safe_open(\n"
    "        hf_weights_files,\n"
    "        framework=\"pt\",\n"
    "        device=device,\n"
    "        process_group=process_group,\n"
    "        copy=True,\n"
    "        max_free_mem_usage=_GLM53_UMA_STATE.get(\"max_free_mem_usage\"),\n"
    "        buffer_size=_GLM53_UMA_STATE.get(\"buffer_size\"),\n"
    "    ) as f:\n"
)

ANCHOR_IT_DEF = "def instanttensor_weights_iterator(\n"
HELPER = '''
# [glm53-cold-load-uma:v1] helpers -------------------------------------------
_GLM53_UMA_STATE: dict[str, object] = {}


def _glm53_meminfo_kib(field: str) -> int | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _glm53_uma_page_size() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return 4096


def _glm53_env_number(name: str, kind, lo, hi):
    """Parse an optional numeric env var; invalid or out-of-range values are
    ignored with a warning instead of aborting the model load."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = kind(raw)
    except (TypeError, ValueError):
        logger.warning("[glm53-cold-load-uma] ignoring %s=%r (not %s)", name, raw, kind.__name__)
        return None
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        logger.warning("[glm53-cold-load-uma] ignoring %s=%r (outside [%s, %s])", name, raw, lo, hi)
        return None
    return value


def _glm53_uma_drop_caches() -> bool:
    """Drop clean page cache. Only possible with CAP_SYS_ADMIN, which the
    stock container does not have; returns False there. The budget no longer
    depends on this succeeding."""
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as fh:
            fh.write("1\\n")
        return True
    except OSError:
        return False


def _glm53_uma_prepare_instanttensor_budget(hf_weights_files: list[str]) -> None:
    """Keep InstantTensor's device-memory budget honest on unified memory.

    Sets ``_GLM53_UMA_STATE[\\"max_free_mem_usage\\"]`` / ``[\\"buffer_size\\"]``
    (None = InstantTensor/env default). Reads ``torch.cuda.mem_get_info()``
    (the same query InstantTensor makes) and ``/proc/meminfo``; allocates
    nothing on the device. On UMA with a full page cache the budget is sized
    against ``MemAvailable`` — clean cache is reclaimable on demand — so the
    fraction may exceed 1 relative to cuda free and dropping caches is not
    required.
    """
    _GLM53_UMA_STATE.clear()
    if os.environ.get("GLM53_COLD_LOAD_UMA", "1") != "1":
        return  # runtime kill switch; the image is already patched at build

    import torch

    if not (torch.cuda.is_available() and current_platform.is_cuda()):
        return
    try:
        import instanttensor  # noqa: F401
    except ImportError:
        return

    env_budget = _glm53_env_number("INSTANTTENSOR_MAX_FREE_MEM_USAGE", float, 0.0, None)
    env_buffer = _glm53_env_number("INSTANTTENSOR_BUFFER_SIZE", int, 1, None)
    # Default buffer target: 4 GiB keeps io_depth at the AIO/uring default
    # (512 // world_size x 8 MiB chunks) — measured 5.08 GB/s on this kit,
    # the single-reader O_DIRECT ceiling of the 1 TB NVMe.
    buffer_target = env_buffer if env_buffer else 4 * (1 << 30)
    # Need the buffer plus per-tensor copies in flight (copy=True clones the
    # largest tensor once) and a 2 GiB margin for the allocator.
    largest = 0
    try:
        largest = max(os.path.getsize(p) for p in hf_weights_files) if hf_weights_files else 0
    except OSError:
        pass
    need_bytes = buffer_target + min(largest, 4 * (1 << 30)) + 2 * (1 << 30)

    mem_free = _glm53_meminfo_kib("MemFree")
    mem_avail = _glm53_meminfo_kib("MemAvailable")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    uma = mem_free is not None and abs(free_bytes - mem_free * 1024) < (8 << 30)
    if not uma:
        # Discrete GPU: device memory is not the host page cache; stock path.
        _GLM53_UMA_STATE["max_free_mem_usage"] = env_budget
        _GLM53_UMA_STATE["buffer_size"] = env_buffer
        return

    avail_bytes = mem_avail * 1024 if mem_avail is not None else None
    window_holds = avail_bytes is not None and avail_bytes >= need_bytes
    dropped = False
    if free_bytes < need_bytes and window_holds:
        dropped = _glm53_uma_drop_caches()
        free_bytes, _ = torch.cuda.mem_get_info()
    # Budget = fraction of *current* cuda free. When the MemAvailable window
    # holds the whole load, ask for exactly what the load needs: clean page
    # cache is reclaimable on demand, so the fraction may exceed 1 (bounded
    # at 90% of MemAvailable) — InstantTensor does not cap it and the pinned
    # allocation reclaims the cache. Otherwise keep InstantTensor's 0.5
    # default when free memory is plentiful, else the stock-capped fraction.
    if free_bytes >= 2 * need_bytes:
        auto_frac = 0.5
    elif window_holds:
        auto_frac = min(need_bytes, int(0.9 * avail_bytes)) / max(free_bytes, 1)
    else:
        auto_frac = min(0.95, need_bytes / max(free_bytes, 1))
    frac = env_budget
    if frac is None:
        frac = auto_frac
    elif (
        env_buffer is None
        and window_holds
        and int(free_bytes * frac) < buffer_target
        and auto_frac > frac
    ):
        # A bare INSTANTTENSOR_MAX_FREE_MEM_USAGE (the pre-UMA workaround from
        # #204/#273) scales cuda free, which is MemFree here: with the page
        # cache full it cannot even hold the buffer pinned below, and
        # InstantTensor aborts. Only in that case raise it to the
        # MemAvailable-sized fraction; a fraction that covers the buffer is
        # the caller's choice and is kept, and it is never lowered.
        # An explicit INSTANTTENSOR_BUFFER_SIZE keeps the caller's pair as-is.
        logger.warning(
            "[glm53-cold-load-uma] INSTANTTENSOR_MAX_FREE_MEM_USAGE=%.2f gives a "
            "%.1f GiB budget (cuda free %.1f GiB counts page cache as used) for "
            "a %.1f GiB load; using the MemAvailable-sized %.2f instead. Unset "
            "it, or also set INSTANTTENSOR_BUFFER_SIZE to keep an explicit pair.",
            frac,
            free_bytes * frac / (1 << 30),
            free_bytes / (1 << 30),
            need_bytes / (1 << 30),
            auto_frac,
        )
        frac = auto_frac
    budget = int(free_bytes * frac)
    if env_buffer:
        buffer_size = env_buffer
    elif window_holds:
        # The window holds the load: pin the io_depth-preserving size.
        buffer_size = buffer_target
    else:
        # Genuinely short: best effort under the budget, but never None —
        # None falls back to InstantTensor's default buffer, which aborts
        # against a small budget, while an explicit floor is enlarged to the
        # largest tensor and loads whenever the budget covers it.
        buffer_size = max(min(buffer_target, budget - (1 << 30)), 1)
    _GLM53_UMA_STATE["max_free_mem_usage"] = frac
    _GLM53_UMA_STATE["buffer_size"] = buffer_size
    logger.info(
        "[glm53-cold-load-uma] page=%d KiB MemFree=%.1f GiB MemAvailable=%.1f GiB "
        "cuda_free=%.1f GiB dropped_caches=%s -> max_free_mem_usage=%.2f "
        "buffer_size=%s",
        _glm53_uma_page_size() // 1024,
        (mem_free or 0) / (1 << 20),
        (mem_avail or 0) / (1 << 20),
        free_bytes / (1 << 30),
        dropped,
        frac,
        f"{buffer_size / (1 << 30):.2f} GiB" if buffer_size else "auto",
    )
    if free_bytes < need_bytes and not window_holds:
        logger.warning(
            "[glm53-cold-load-uma] only %.1f GiB free / %.1f GiB available for "
            "a %.1f GiB load window; InstantTensor will shrink io_depth or "
            "raise. Free host memory before launch.",
            free_bytes / (1 << 30),
            (avail_bytes or 0) / (1 << 30),
            need_bytes / (1 << 30),
        )


'''

# The v1 helper (PR #230) differed only in the budget-fraction block below.
# Images baked with it are upgraded in place by prepare() (#273).
_V1_BUDGET_BLOCK = '''\
    frac = env_budget
    if frac is None:
        if free_bytes >= 2 * need_bytes:
            frac = 0.5
        elif window_holds:
            frac = min(need_bytes, int(0.9 * avail_bytes)) / max(free_bytes, 1)
        else:
            frac = min(0.95, need_bytes / max(free_bytes, 1))
'''
_BUDGET_BLOCK_START = "    if free_bytes >= 2 * need_bytes:\n        auto_frac = 0.5\n"
_BUDGET_BLOCK_END = "        frac = auto_frac\n"
_b0 = HELPER.index(_BUDGET_BLOCK_START)
_b1 = HELPER.index(_BUDGET_BLOCK_END, HELPER.index("auto_frac,\n        )\n", _b0)) + len(_BUDGET_BLOCK_END)
HELPER_V1 = HELPER[:_b0] + _V1_BUDGET_BLOCK + HELPER[_b1:]

# --- 2. file-backed 64 KiB mmap -----------------------------------------------
ANCHOR_ST_YIELD = (
    "            with safe_open(st_file, framework=\"pt\") as f:\n"
    "                for name in f.keys():  # noqa: SIM118\n"
    "                    if should_skip_weight(name, local_expert_ids):\n"
    "                        continue\n"
    "                    param = f.get_tensor(name)\n"
    "                    yield name, param\n"
)
NEW_ST_YIELD = (
    "            with safe_open(st_file, framework=\"pt\") as f:\n"
    "                for name in f.keys():  # noqa: SIM118\n"
    "                    if should_skip_weight(name, local_expert_ids):\n"
    "                        continue\n"
    "                    param = f.get_tensor(name)\n"
    "                    " + MARK + " cuMemcpyHtoDAsync wedges on GB10 when the\n"
    "                    # source is a file-backed 64 KiB-page mapping: stage the\n"
    "                    # tensor off the mmap into anonymous memory first.\n"
    "                    if _GLM53_UMA_STAGE_MMAP:\n"
    "                        param = param.clone()\n"
    "                    yield name, param\n"
)
STAGE_FLAG = (
    "\n" + MARK + " clone file-backed tensors when the kernel page is not 4 KiB.\n"
    "_GLM53_UMA_STAGE_MMAP = _glm53_uma_page_size() != 4096 and os.environ.get(\n"
    "    \"GLM53_COLD_LOAD_STAGE_MMAP\", \"1\"\n"
    ") == \"1\"\n"
)


def verified_state(src: str) -> str:
    if src.count(MARK) >= 4:
        helper_state = "patched" if HELPER in src else "patched-v1" if HELPER_V1 in src else None
        for needle in (NEW_IT_OPEN, NEW_ST_YIELD, STAGE_FLAG):
            if needle not in src:
                helper_state = None
        if helper_state is None:
            raise SystemExit(f"{TARGET}: partially patched — source drift")
        return helper_state
    if src.count(MARK):
        raise SystemExit(f"{TARGET}: partial marks ({src.count(MARK)}) — source drift")
    for name, needle in (
        ("instanttensor safe_open", ANCHOR_IT_OPEN),
        ("instanttensor def", ANCHOR_IT_DEF),
        ("safetensors yield", ANCHOR_ST_YIELD),
    ):
        if src.count(needle) != 1:
            raise SystemExit(f"{TARGET}: expected exactly one {name} anchor, got {src.count(needle)}")
    if "from vllm.platforms import current_platform" not in src or "logger = init_logger(__name__)" not in src:
        raise SystemExit(f"{TARGET}: missing current_platform/logger — source drift")
    if "\nimport os\n" not in src:
        raise SystemExit(f"{TARGET}: 'import os' missing — source drift")
    return "stock"


def prepare(src: str) -> str:
    if MARK in src and HELPER_V1 in src and HELPER not in src:
        # Baked with the v1 helper: swap in the current one (#273).
        return src.replace(HELPER_V1, HELPER, 1)
    if MARK in src:
        # Already carries the overlay (verified_state guards partial marks);
        # a second apply must be a no-op.
        return src
    src = src.replace(ANCHOR_IT_OPEN, NEW_IT_OPEN, 1)
    src = src.replace(ANCHOR_IT_DEF, HELPER + ANCHOR_IT_DEF, 1)
    # The mmap flag must be defined before safetensors_weights_iterator runs;
    # module level, right after the helpers (which are above the def).
    src = src.replace(HELPER + ANCHOR_IT_DEF, HELPER + STAGE_FLAG + "\n\n" + ANCHOR_IT_DEF, 1)
    src = src.replace(ANCHOR_ST_YIELD, NEW_ST_YIELD, 1)
    return src


def main() -> int:
    if os.environ.get(ENV_NAME, "1") != "1":
        print(f"[glm53-cold-load-uma] {ENV_NAME}={os.environ.get(ENV_NAME)!r} — not applied")
        return 0
    src = TARGET.read_text()
    state = verified_state(src)
    if state == "patched":
        print(f"[glm53-cold-load-uma] {TARGET}: already patched")
        return 0
    out = prepare(src)
    if state == "patched-v1":
        if verified_state(out) != "patched":
            raise SystemExit("v1 helper upgrade self-check failed")
        compile(out, str(TARGET), "exec")
        TARGET.write_text(out)
        print(f"[glm53-cold-load-uma] upgraded the v1 budget helper in {TARGET}")
        return 0
    if verified_state(out) != "patched":
        raise SystemExit("patch self-check failed")
    compile(out, str(TARGET), "exec")
    TARGET.write_text(out)
    print(f"[glm53-cold-load-uma] patched {TARGET} (page={os.sysconf('SC_PAGE_SIZE')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
