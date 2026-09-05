#!/usr/bin/env python3
"""Execute APC overlays against exact legacy image sources before model boot."""
from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from test_apc_per_group_retention import test_composed_runtime_paths  # noqa: E402


def required_env(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    path = Path(value) if value else Path()
    if not value or not path.is_file():
        raise SystemExit(f"{name} must name a source file extracted from the exact image")
    return path


def apply(script: Path, coordinator: Path, block_pool: Path, manager: Path) -> None:
    env = os.environ.copy()
    env.update(
        GLM53_KV_COORDINATOR_PY=str(coordinator),
        GLM53_BLOCK_POOL_PY=str(block_pool),
        GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY=str(manager),
    )
    subprocess.check_call([sys.executable, str(script)], env=env)


def main() -> int:
    coordinator_src = required_env("GLM53_EXACT_IMAGE_KV_COORDINATOR_PY")
    block_pool_src = required_env("GLM53_EXACT_IMAGE_BLOCK_POOL_PY")
    manager_src = required_env("GLM53_EXACT_IMAGE_SINGLE_TYPE_MANAGER_PY")
    hybrid = ROOT / "overlay/patch_hybrid_prefix_hit.py"
    pergroup = ROOT / "overlay/patch_apc_per_group_retention.py"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        coordinator = tmp / "kv_cache_coordinator.py"
        block_pool = tmp / "block_pool.py"
        manager = tmp / "single_type_kv_cache_manager.py"
        for src, dst in (
            (coordinator_src, coordinator),
            (block_pool_src, block_pool),
            (manager_src, manager),
        ):
            shutil.copyfile(src, dst)

        apply(hybrid, coordinator, block_pool, manager)
        apply(pergroup, coordinator, block_pool, manager)
        first = tuple(p.read_bytes() for p in (coordinator, block_pool, manager))
        for path in (coordinator, block_pool, manager):
            py_compile.compile(str(path), doraise=True)

        test_composed_runtime_paths(
            coordinator.read_text(), block_pool.read_text(), manager.read_text()
        )

        # The exact-image migration itself must be idempotent, not just a
        # pristine composition used by the broader host test.
        apply(hybrid, coordinator, block_pool, manager)
        apply(pergroup, coordinator, block_pool, manager)
        second = tuple(p.read_bytes() for p in (coordinator, block_pool, manager))
        if first != second:
            raise AssertionError("exact-image overlay migration is not byte-idempotent")

    print("exact-image APC migration init/cache/hit/free regression OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
