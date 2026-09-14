#!/usr/bin/env python3
"""Apply overlay/patch_scheduler_decode_floor.py to a copy of scheduler.py."""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PATCH = next(
    p
    for p in (
        HERE / "patch_scheduler_decode_floor.py",
        HERE.parent / "overlay" / "patch_scheduler_decode_floor.py",
    )
    if p.is_file()
)
SRC = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)


def main() -> int:
    if not PATCH.is_file():
        raise SystemExit(f"missing {PATCH}")
    src = Path(os.environ.get("GLM53_SCHEDULER_PY_SRC", SRC))
    if not src.is_file():
        # Host unit test: copy from a live container if present.
        raise SystemExit(f"missing scheduler.py at {src}")
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "scheduler.py"
        shutil.copyfile(src, dst)
        env = os.environ.copy()
        env["GLM53_SCHEDULER_PY"] = str(dst)
        env["GLM53_MIXED_PREFILL_CHUNK"] = "skip"
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        text = dst.read_text()
        module = ast.parse(text)
        helper = next(node for node in module.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_glm53_mixed_prefill_policy")
        namespace = {"os": os}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(dst), "exec"), namespace)
        policy = namespace[helper.name]
        peer = SimpleNamespace(request_id="peer", num_prompt_tokens=100, num_computed_tokens=200)
        newcomer = SimpleNamespace(request_id="new", num_prompt_tokens=1000, num_computed_tokens=0)
        with patch.dict(os.environ, {"GLM53_MIXED_PREFILL_CHUNK": "skip"}):
            assert policy([peer], newcomer) == 0
            assert policy([], newcomer) is None
            os.environ["GLM53_MIXED_PREFILL_CHUNK"] = "0"
            assert policy([peer], newcomer) is None
            os.environ["GLM53_MIXED_PREFILL_CHUNK"] = "128"
            assert policy([peer], newcomer) == 128
        # idempotent
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        assert dst.read_text() == text
    print("scheduler decode-floor patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
