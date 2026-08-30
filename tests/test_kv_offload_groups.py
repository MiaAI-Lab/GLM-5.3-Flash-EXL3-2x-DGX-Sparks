#!/usr/bin/env python3
"""Regression tests for the KV-offload group/staging patch."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_kv_offload_groups.py",
        ROOT / "overlay" / "patch_kv_offload_groups.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_kv_offload_groups import MARK, apply, targets  # noqa: E402

INSTALLED = Path("/usr/local/lib/python3.12/dist-packages/vllm")

# Exact fragments from vLLM 487ecf187 / glm53-flash image. Each is the anchor a
# corresponding edit pins; if upstream reflows any of them the patch must fail
# closed rather than silently serve an unpatched engine.
CONFIG_FIXTURE = '''from typing import TYPE_CHECKING

from vllm.v1.kv_offload.config import (
    OffloadingConfig,
)


def build_offloading_config(vllm_config, kv_cache_config):
    _, tokens_per_hash = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    for group in groups:
        assert group.tokens_per_block % tokens_per_hash == 0, (
            f"tokens_per_block={group.tokens_per_block} not divisible by "
            f"tokens_per_hash={tokens_per_hash}. "
            f"Hybrid models (e.g. Mamba+Attention) need "
            f"--enable-prefix-caching to align block sizes."
        )
'''


def _write(tmp_path: Path, rel: str, text: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_targets_cover_the_five_patched_files(tmp_path: Path) -> None:
    names = [p.name for p, _ in targets(tmp_path)]
    assert names == [
        "config.py",
        "scheduler.py",
        "worker.py",
        "shared_offload_region.py",
        "gpu_worker.py",
    ]
    # every target is resolved under the supplied root, never a live install
    assert all(str(p).startswith(str(tmp_path)) for p, _ in targets(tmp_path))


def test_apply_is_idempotent_and_marks_the_file(tmp_path: Path) -> None:
    f = _write(tmp_path, "config.py", CONFIG_FIXTURE)
    edits = dict((p.name, e) for p, e in targets(tmp_path))["config.py"]

    assert apply(f, edits) is True
    once = f.read_text()
    assert MARK in once
    # the hash-alignment assert is what aborts the boot on KpoolTailSpec
    assert "assert group.tokens_per_block % tokens_per_hash == 0" not in once

    assert apply(f, edits) is True
    assert f.read_text() == once, "second apply must be a no-op"


def test_apply_fails_closed_on_a_drifted_anchor(tmp_path: Path) -> None:
    f = _write(tmp_path, "config.py", CONFIG_FIXTURE.replace("tokens_per_hash", "tph"))
    edits = dict((p.name, e) for p, e in targets(tmp_path))["config.py"]
    before = f.read_text()

    assert apply(f, edits) is False, "drifted anchor must not be patched"
    assert f.read_text() == before, "a failed apply must leave the file untouched"


def test_env_kill_switch_skips_everything() -> None:
    env = dict(os.environ, GLM53_SKIP_KV_OFFLOAD_GROUPS_PATCH="1")
    out = subprocess.run(
        [sys.executable, str(PATCH)], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0
    assert "skipped via env" in out.stdout


def test_patch_applies_to_the_installed_vllm() -> None:
    """Runs only inside the image: every anchor must still be present."""
    if not INSTALLED.is_dir():
        return  # not in the container; the fixture tests above still ran
    for path, edits in targets(INSTALLED):
        src = path.read_text()
        if MARK in src:
            continue  # already patched by start.sh
        for old, _new in edits:
            assert src.count(old) == 1, f"anchor drifted in {path.name}: {old[:60]!r}"
