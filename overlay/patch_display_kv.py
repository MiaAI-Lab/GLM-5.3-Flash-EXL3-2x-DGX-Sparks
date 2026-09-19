#!/usr/bin/env python3
"""Additive display-reserve KV backing on GB10 (overlay/display_kv/).

Wires ``glm53_display_kv`` into vLLM in three places:

1. ``gpu_worker.determine_available_memory``: after the profiled ordinary KV
   budget is computed, open the DRM display pool and ADD its size to
   ``available_kv_cache_memory_bytes``. Ordinary budget is unchanged; the extra
   blocks are backed by the pool.
2. ``gpu_model_runner._allocate_kv_cache_tensors``: the two
   ``torch.zeros(size, int8, device)`` become ``alloc_int8(size, device)``,
   which carves from the pool while a tensor fits and falls back to
   ``torch.zeros`` otherwise; ``report()`` logs the split once.
3. ``kv_connector_model_runner_mixin`` cross-layer allocation: same swap (not
   the path this model takes, patched for consistency).
4. ``gpu/attn_utils._allocate_kv_cache`` (the v2 model runner, which is what
   this image actually runs for GLM-5.3): same plan + swap as (2).

Also installs ``glm53_display_kv.py`` next to ``gpu_worker.py`` and
``libglm53_display_kv.so`` into /usr/local/lib (both staged by the launcher
under /opt/glm53/display_kv/). Idempotent; fails closed on anchor drift.

Marker: [glm53-display-kv]
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

VLLM = Path(os.environ.get("GLM53_VLLM_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"))
STAGE = Path(os.environ.get("GLM53_DISPLAY_KV_STAGE", "/opt/glm53/display_kv"))
LIB_DST = Path(os.environ.get("GLM53_DISPLAY_KV_LIB", "/usr/local/lib/libglm53_display_kv.so"))
MARK = "# [glm53-display-kv]"

WORKER = VLLM / "v1/worker/gpu_worker.py"
WORKER_OLD = (
    "        self.available_kv_cache_memory_bytes = (\n"
    "            self.requested_memory\n"
    "            - profile_result.non_kv_cache_memory\n"
    "            - cudagraph_memory_estimate_applied\n"
    "        )\n"
)
WORKER_NEW = WORKER_OLD + (
    f"        {MARK} additive DRM display-reserve pool (overlay/display_kv)\n"
    "        from vllm.v1.worker import glm53_display_kv as _glm53_dkv\n"
    "        self.available_kv_cache_memory_bytes += _glm53_dkv.credit()\n"
)

RUNNER = VLLM / "v1/worker/gpu_model_runner.py"
RUNNER_OLD_A = (
    "                if packed_backing is None:\n"
    "                    packed_backing = torch.zeros(\n"
    "                        kv_cache_tensor.size,\n"
    "                        dtype=torch.int8,\n"
    "                        device=self.device,\n"
    "                    )\n"
)
RUNNER_NEW_A = (
    "                if packed_backing is None:\n"
    f"                    {MARK}\n"
    "                    packed_backing = _glm53_dkv.alloc_int8(\n"
    "                        kv_cache_tensor.size, self.device, _glm53_i\n"
    "                    )\n"
)
RUNNER_OLD_B = (
    "            else:\n"
    "                tensor = torch.zeros(\n"
    "                    kv_cache_tensor.size, dtype=torch.int8, device=self.device\n"
    "                )\n"
    "            for layer_name in kv_cache_tensor.shared_by:\n"
    "                kv_cache_raw_tensors[layer_name] = tensor\n"
)
RUNNER_NEW_B = (
    "            else:\n"
    f"                {MARK}\n"
    "                tensor = _glm53_dkv.alloc_int8(\n"
    "                    kv_cache_tensor.size, self.device, _glm53_i\n"
    "                )\n"
    "            for layer_name in kv_cache_tensor.shared_by:\n"
    "                kv_cache_raw_tensors[layer_name] = tensor\n"
    "        _glm53_dkv.report()\n"
)
RUNNER_HEAD_OLD = (
    "        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}\n"
    "        packed_backing: torch.Tensor | None = None\n"
    "        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:\n"
)
RUNNER_HEAD_NEW = (
    f"        {MARK} pool-vs-ordinary plan over the tensor list (first-fit\n"
    "        # decreasing); the packed layout has one backing so its plan is trivial.\n"
    "        from vllm.v1.worker import glm53_display_kv as _glm53_dkv\n"
    "        _glm53_dkv.plan([t.size for t in kv_cache_config.kv_cache_tensors])\n"
    "        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}\n"
    "        packed_backing: torch.Tensor | None = None\n"
    "        for _glm53_i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):\n"
)

MIXIN = VLLM / "v1/worker/kv_connector_model_runner_mixin.py"
MIXIN_OLD = (
    "        cross_layers_kv_cache = (\n"
    "            torch.zeros(total_size, dtype=torch.int8, device=device)\n"
)
MIXIN_NEW = (
    f"        {MARK}\n"
    "        from vllm.v1.worker import glm53_display_kv as _glm53_dkv\n"
    "        cross_layers_kv_cache = (\n"
    "            _glm53_dkv.alloc_int8(total_size, device)\n"
)

# v2 runner (vllm/v1/worker/gpu/): same two allocations, module-level function.
ATTN = VLLM / "v1/worker/gpu/attn_utils.py"
ATTN_OLD_HEAD = (
    "    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}\n"
    "    packed_backing: torch.Tensor | None = None\n"
    "    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:\n"
    "        if kv_cache_tensor.block_stride > 0:\n"
    "            # Allocate once; all packed tensors alias the same backing.\n"
    "            if packed_backing is None:\n"
    "                packed_backing = torch.zeros(\n"
    "                    kv_cache_tensor.size, dtype=torch.int8, device=device\n"
    "                )\n"
    "            tensor = packed_backing\n"
    "        else:\n"
    "            tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)\n"
    "        for layer_name in kv_cache_tensor.shared_by:\n"
    "            kv_cache_raw_tensors[layer_name] = tensor\n"
)
ATTN_NEW_HEAD = (
    f"    {MARK} pool-vs-ordinary plan (first-fit decreasing) over the tensor list\n"
    "    from vllm.v1.worker import glm53_display_kv as _glm53_dkv\n"
    "    _glm53_dkv.plan([t.size for t in kv_cache_config.kv_cache_tensors])\n"
    "    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}\n"
    "    packed_backing: torch.Tensor | None = None\n"
    "    for _glm53_i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):\n"
    "        if kv_cache_tensor.block_stride > 0:\n"
    "            # Allocate once; all packed tensors alias the same backing.\n"
    "            if packed_backing is None:\n"
    "                packed_backing = _glm53_dkv.alloc_int8(\n"
    "                    kv_cache_tensor.size, device, _glm53_i\n"
    "                )\n"
    "            tensor = packed_backing\n"
    "        else:\n"
    "            tensor = _glm53_dkv.alloc_int8(kv_cache_tensor.size, device, _glm53_i)\n"
    "        for layer_name in kv_cache_tensor.shared_by:\n"
    "            kv_cache_raw_tensors[layer_name] = tensor\n"
    "    _glm53_dkv.report()\n"
)

EDITS = [
    (WORKER, [(WORKER_OLD, WORKER_NEW)]),
    (RUNNER, [(RUNNER_HEAD_OLD, RUNNER_HEAD_NEW), (RUNNER_OLD_A, RUNNER_NEW_A), (RUNNER_OLD_B, RUNNER_NEW_B)]),
    (MIXIN, [(MIXIN_OLD, MIXIN_NEW)]),
    (ATTN, [(ATTN_OLD_HEAD, ATTN_NEW_HEAD)]),
]


def apply(path: Path, pairs) -> str:
    src = path.read_text()
    if all(new in src for _, new in pairs):
        return "already patched"
    if MARK in src:
        raise SystemExit(f"{path}: partial marker — source drift")
    for old, _ in pairs:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"{path}: expected exactly one anchor, got {n}:\n{old}")
    for old, new in pairs:
        src = src.replace(old, new, 1)
    compile(src, str(path), "exec")
    path.write_text(src)
    return "patched"


def install_files() -> None:
    mod_src = STAGE / "glm53_display_kv.py"
    lib_src = STAGE / "libglm53_display_kv.so"
    if not mod_src.exists():
        raise SystemExit(f"{mod_src} missing: launcher did not stage overlay/display_kv")
    mod_dst = VLLM / "v1/worker/glm53_display_kv.py"
    if not mod_dst.exists() or mod_dst.read_bytes() != mod_src.read_bytes():
        shutil.copyfile(mod_src, mod_dst)
    if not lib_src.exists():
        # The launcher only stages the .so when the pool is in use; with
        # GLM53_DISPLAY_KV=0 credit() never touches it.
        if os.environ.get("GLM53_DISPLAY_KV", "1").strip().lower() in ("0", "off", "false", "no", ""):
            print(f"[glm53-display-kv] installed {mod_dst} (pool disabled; no library)")
            return
        raise SystemExit(f"{lib_src} missing: launcher did not build libglm53_display_kv.so")
    LIB_DST.parent.mkdir(parents=True, exist_ok=True)
    if not LIB_DST.exists() or LIB_DST.read_bytes() != lib_src.read_bytes():
        shutil.copyfile(lib_src, LIB_DST)
        LIB_DST.chmod(0o755)
    print(f"[glm53-display-kv] installed {mod_dst} and {LIB_DST}")


def main() -> int:
    install_files()
    for path, pairs in EDITS:
        print(f"[glm53-display-kv] {path}: {apply(path, pairs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
