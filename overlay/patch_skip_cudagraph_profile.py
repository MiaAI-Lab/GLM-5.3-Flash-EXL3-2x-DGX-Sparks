#!/usr/bin/env python3
"""Skip the CUDA-graph memory dry-capture when its result is discarded.

``gpu_worker.py`` always runs ``profile_cudagraph_memory()`` (a throwaway KV
cache + dry capture of the two largest graphs per mode) and only *applies* the
estimate when ``VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`` is set. With the
launcher's ``CG_ESTIMATE=0`` the number is computed, logged and dropped:
11 s of the boot on this kit (``13:04:29 breakable_cudagraph`` ->
``13:04:40 Estimated CUDA graph memory``) for nothing.

This makes the profile run only when the flag is on. The only observable
difference with the flag off is the missing "Estimated CUDA graph memory" INFO
line and ``self.cudagraph_memory_estimate == 0`` (which was already unused on
that path: ``peak_activation_memory`` adds the *applied* value).

Idempotent; fails closed on anchor drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TARGET = Path(
    os.environ.get(
        "GLM53_GPU_WORKER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py",
    )
)
MARK = "# [glm53-skip-cudagraph-profile]"
OLD = (
    "        cudagraph_memory_estimate = 0\n"
    "        if (\n"
    "            current_platform.is_cuda_alike()\n"
    "            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE\n"
    "        ):\n"
    "            cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()\n"
)
NEW = (
    "        cudagraph_memory_estimate = 0\n"
    "        if (\n"
    "            current_platform.is_cuda_alike()\n"
    "            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE\n"
    "            " + MARK + " the estimate is only applied when the\n"
    "            # flag is on; do not spend the dry capture otherwise.\n"
    "            and envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS\n"
    "        ):\n"
    "            cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()\n"
)


def verified_state(src: str) -> str:
    if NEW in src:
        return "patched"
    if MARK in src:
        raise SystemExit(f"{TARGET}: partial mark — source drift")
    if src.count(OLD) != 1:
        raise SystemExit(f"{TARGET}: expected exactly one profile_cudagraph_memory gate, got {src.count(OLD)}")
    if "envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" not in src:
        raise SystemExit(f"{TARGET}: flag not referenced — source drift")
    return "stock"


def prepare(src: str) -> str:
    return src if NEW in src else src.replace(OLD, NEW, 1)


def main() -> int:
    src = TARGET.read_text()
    if verified_state(src) == "patched":
        print(f"[glm53-skip-cudagraph-profile] {TARGET}: already patched")
        return 0
    out = prepare(src)
    assert verified_state(out) == "patched"
    compile(out, str(TARGET), "exec")
    TARGET.write_text(out)
    print(f"[glm53-skip-cudagraph-profile] patched {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
