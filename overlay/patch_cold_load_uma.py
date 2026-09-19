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

   Fix: before opening, measure ``MemAvailable`` (page cache is reclaimable on
   this box) and, when ``MemFree`` is short of the requested budget, drop clean
   page cache from inside the container (``/proc/sys/vm/drop_caches`` is
   writable under ``--privileged`` or with ``CAP_SYS_ADMIN``; otherwise the
   launcher already did it on the host and this is a no-op) and pin an
   explicit ``max_free_mem_usage`` / ``buffer_size`` that keeps
   ``io_depth`` at the backend default. The launcher forwards
   ``INSTANTTENSOR_*`` unchanged; this patch only supplies defaults.

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


def _glm53_uma_drop_caches() -> bool:
    """Drop clean page cache. Only possible with CAP_SYS_ADMIN; returns False
    when the container cannot (the launcher dropped on the host instead)."""
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
    (None = InstantTensor/env default) and drops page cache when MemFree is
    short. Pure host-side bookkeeping; never touches CUDA state.
    """
    import torch

    _GLM53_UMA_STATE.clear()
    if not (torch.cuda.is_available() and current_platform.is_cuda()):
        return
    try:
        import instanttensor  # noqa: F401
    except ImportError:
        return

    env_budget = os.environ.get("INSTANTTENSOR_MAX_FREE_MEM_USAGE")
    env_buffer = os.environ.get("INSTANTTENSOR_BUFFER_SIZE")
    # Default buffer target: 4 GiB keeps io_depth at the AIO/uring default
    # (512 // world_size x 8 MiB chunks) — measured 5.08 GB/s on this kit,
    # the single-reader O_DIRECT ceiling of the 1 TB NVMe.
    buffer_target = int(env_buffer) if env_buffer else 4 * (1 << 30)
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
        _GLM53_UMA_STATE["max_free_mem_usage"] = float(env_budget) if env_budget else None
        _GLM53_UMA_STATE["buffer_size"] = int(env_buffer) if env_buffer else None
        return

    dropped = False
    if free_bytes < need_bytes and mem_avail is not None and mem_avail * 1024 >= need_bytes:
        dropped = _glm53_uma_drop_caches()
        free_bytes, _ = torch.cuda.mem_get_info()
    # Budget = fraction of *current* free. Ask for exactly what the buffer
    # needs (plus margin) so a later CUDA allocation is never starved, but
    # never below InstantTensor's 0.5 default when free memory is plentiful.
    frac = float(env_budget) if env_budget else None
    if frac is None:
        frac = 0.5 if free_bytes >= 2 * need_bytes else min(0.95, need_bytes / max(free_bytes, 1))
    budget = int(free_bytes * frac)
    buffer_size = int(env_buffer) if env_buffer else min(buffer_target, max(budget - (1 << 30), 0))
    if buffer_size <= 0:
        buffer_size = None
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
    if free_bytes < need_bytes:
        logger.warning(
            "[glm53-cold-load-uma] only %.1f GiB free for a %.1f GiB load "
            "window; InstantTensor will shrink io_depth. Drop page cache on the "
            "host before launch (start.sh does this with sudo -n).",
            free_bytes / (1 << 30),
            need_bytes / (1 << 30),
        )


'''

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
        for needle in (NEW_IT_OPEN, HELPER, NEW_ST_YIELD, STAGE_FLAG):
            if needle not in src:
                raise SystemExit(f"{TARGET}: partially patched — source drift")
        return "patched"
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
    if verified_state(out) != "patched":
        raise SystemExit("patch self-check failed")
    compile(out, str(TARGET), "exec")
    TARGET.write_text(out)
    print(f"[glm53-cold-load-uma] patched {TARGET} (page={os.sysconf('SC_PAGE_SIZE')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
