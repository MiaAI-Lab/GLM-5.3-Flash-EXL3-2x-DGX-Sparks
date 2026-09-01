# SPDX-License-Identifier: Apache-2.0
"""Interpreter bootstrap: mounts the KLD logit-dump hook in every engine process.

The vLLM worker tree is spawned (``VLLM_WORKER_MULTIPROC_METHOD=spawn``), so a
``sitecustomize`` on ``PYTHONPATH`` is the only place that runs before the model
worker imports ``vllm``. No-op unless ``GLM_KLD_DUMP_DIR`` is set, so it is safe
to mount into a container that also serves production traffic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if os.environ.get("GLM_KLD_DUMP_DIR"):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import logit_dump_hook  # noqa: F401  (import performs the install)
    except Exception as error:  # pragma: no cover - never break an unrelated run
        print(f"[glm53-kld] logit dump hook failed to install: {error!r}", file=sys.stderr, flush=True)
