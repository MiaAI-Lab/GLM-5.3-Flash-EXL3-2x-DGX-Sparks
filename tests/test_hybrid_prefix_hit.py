#!/usr/bin/env python3
"""Apply overlay/patch_hybrid_prefix_hit.py to a copy of kv_cache_coordinator.py."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(
    p
    for p in (
        HERE / "patch_hybrid_prefix_hit.py",
        HERE.parent / "overlay" / "patch_hybrid_prefix_hit.py",
    )
    if p.is_file()
)
SRC = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py"
)


def main() -> int:
    if not PATCH.is_file():
        raise SystemExit(f"missing {PATCH}")
    src = Path(os.environ.get("GLM53_KV_COORDINATOR_PY_SRC", SRC))
    if not src.is_file():
        raise SystemExit(f"missing kv_cache_coordinator.py at {src}")
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "kv_cache_coordinator.py"
        shutil.copyfile(src, dst)
        env = os.environ.copy()
        env["GLM53_KV_COORDINATOR_PY"] = str(dst)
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        text = dst.read_text()
        assert "[glm53-hybrid-apc]" in text
        assert text.count("[glm53-hybrid-apc]") >= 3
        assert "# [glm53-dflash-eagle-verify-v3]" in text
        assert "def _glm53_is_draft_swa_spec(" in text
        assert "swa_ids or set(" in text
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        assert dst.read_text() == text
    print("hybrid prefix-hit patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
