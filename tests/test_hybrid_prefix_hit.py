#!/usr/bin/env python3
"""Apply overlay/patch_hybrid_prefix_hit.py to a copy of kv_cache_coordinator.py."""
from __future__ import annotations

import ast
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


def helper_namespace(text: str) -> dict:
    tree = ast.parse(text)
    names = {
        "_glm53_inner_kv_spec",
        "_glm53_is_draft_swa_spec",
        "_glm53_dflash_swa_replay_tokens",
        "_glm53_dflash_replay_safe_hit",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    ns: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "helpers", "exec"), ns)
    return ns


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
        assert "def _glm53_is_draft_swa_spec(" in text
        assert "def _glm53_dflash_swa_replay_tokens(" in text
        assert "def _glm53_dflash_replay_safe_hit(" in text
        assert "self.dflash_swa_replay_tokens" in text
        assert "replay clamp hit=%d->%d" in text
        assert "[glm53-dflash-swa-replay-v1]" in text
        assert "[glm53-dflash-swa-replay-v2]" in text
        assert "draft_replay_ready" in text
        assert "reusing reconciled " in text
        assert "DFlash boundary hit=%d" in text
        assert "swa_ids or set(" in text
        helpers = helper_namespace(text)
        safe_hit = helpers["_glm53_dflash_replay_safe_hit"]
        replay_tokens = helpers["_glm53_dflash_swa_replay_tokens"]
        kpool_spec = type("KpoolTailSpec", (), {"sliding_window": 4})()
        dflash_spec = type("SlidingWindowSpec", (), {"sliding_window": 2048})()
        group_type = type("Group", (), {})
        kpool_group = group_type()
        kpool_group.kv_cache_spec = kpool_spec
        dflash_group = group_type()
        dflash_group.kv_cache_spec = dflash_spec
        assert replay_tokens([kpool_group, dflash_group]) == 2048
        assert replay_tokens([kpool_group]) == 0
        assert safe_hit(21504, 21504, 2048, 3584) == 17920
        assert safe_hit(21504, 21568, 2048, 3584) == 17920
        assert safe_hit(21504, 23550, 2048, 3584) == 17920
        assert safe_hit(21504, 23551, 2048, 3584) == 21504
        assert safe_hit(21504, 23552, 2048, 3584) == 21504
        assert safe_hit(0, 0, 2048, 3584) == 0
        assert safe_hit(21504, 21504, 0, 3584) == 21504
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        assert dst.read_text() == text
    print("hybrid prefix-hit patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
