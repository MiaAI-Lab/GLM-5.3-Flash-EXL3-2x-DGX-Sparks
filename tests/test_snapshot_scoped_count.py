#!/usr/bin/env python3
"""Regression guard: weight completeness is scoped to the active snapshot.

A repo-wide count can accept 119 stale shards in snapshots/old plus one shard
in the active snapshot as "120 present", then resolve the incomplete active
snapshot and fail deep in vLLM load. Related: issue #52 (stale refs/main).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def function(name: str) -> str:
    text = START.read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}\n", text)
    assert match, f"missing function {name}"
    return match.group(0)


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )


def test_target_completeness_is_scoped_to_active_snapshot() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        active = repo / "snapshots" / "active"
        old = repo / "snapshots" / "old"
        active.mkdir(parents=True)
        old.mkdir(parents=True)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("active", encoding="utf-8")
        (active / "model-00120-of-00120.safetensors").touch()
        for index in range(1, 120):
            (old / f"model-{index:05d}-of-00120.safetensors").touch()

        result = run_bash(
            "set -euo pipefail\n"
            + function("count_shards")
            + f"count_shards {str(repo)!r}\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1", result.stdout


def test_empty_repo_counts_zero() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        result = run_bash(
            "set -euo pipefail\n"
            + function("count_shards")
            + f"count_shards {str(repo)!r}\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "0", result.stdout


if __name__ == "__main__":
    test_target_completeness_is_scoped_to_active_snapshot()
    test_empty_repo_counts_zero()
    print("snapshot-scoped count guard OK")
