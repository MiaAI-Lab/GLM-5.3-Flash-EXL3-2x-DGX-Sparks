#!/usr/bin/env python3
"""Anchor checks for the bring-up robustness patches in start.sh.

Same static-analysis style as test_warm_restart_stdout.py: the launcher is a
generated-heredoc-heavy script, so these tests pin the markers that keep the
robustness behaviours wired (worker death detection, revision-keyed sync
marker, HF CLI fallback, worker cache writability preflight).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source() -> str:
    return (ROOT / "start.sh").read_text()


def test_worker_death_detection_wired() -> None:
    src = _source()
    assert "worker_fail=0" in src, "worker death detection missing from wait_for_health"
    assert '[ "$worker_fail" -ge 3 ]' in src, "3-strike tolerance missing"
    assert "not running on ${WORKER_SSH}" in src, "worker death message missing"
    assert 'dead_side="worker"' in src and 'dead_side="head"' in src


def test_sync_revision_marker_wired() -> None:
    src = _source()
    assert ".glm53-exl3-synced" in src, "revision marker file missing"
    assert "FORCE_SYNC" in src, "FORCE_SYNC escape hatch missing"
    assert 'refs/main' in src, "marker must key on the snapshot commit (refs/main)"
    # both weights and DFlash2 go through the marker-checked helper
    assert src.count("sync_repo_to_worker ") >= 2


def test_hf_cli_fallback_wired() -> None:
    src = _source()
    assert "resolve_hf_bin()" in src, "resolve_hf_bin helper missing"
    assert src.count("resolve_hf_bin || die") == 3, "expected 3 call sites (weights, dflash, download-only)"
    assert "huggingface_hub.commands.huggingface_cli" in src, "python fallback missing"
    assert '"${HF_BIN_CMD[@]}" download' in src, "hf_download_repo must use the resolved array"


def test_worker_cache_writability_preflight_wired() -> None:
    src = _source()
    assert "worker cannot write $WORKER_CACHE_DIR/hub" in src
    assert "test -w '$WORKER_CACHE_DIR/hub'" in src


def test_running_container_checks_are_pipefail_safe() -> None:
    src = _source()
    assert "docker inspect -f '{{.State.Running}}' \"$CONTAINER_HEAD\"" in src
    assert "docker inspect -f '{{.State.Running}}' \"$CONTAINER_HEAD\" 2>/dev/null | grep -q true" not in src
    assert "worker_ssh \"docker inspect -f '{{.State.Running}}' '$CONTAINER_WORKER' 2>/dev/null\" | grep -q true" not in src
    assert "head_fail=0" in src
    assert "head container missing during startup" in src
    assert "with_cluster_lock" in src
    assert "steal_cluster_lock_for_stop" in src


def test_check_port_free_detects_representative_listeners() -> None:
    """Inside double quotes, \\$ must expand to an end anchor, not a literal $."""
    src = _source()
    assert 'grep -qE "[:.]${port}\\$"' in src
    assert 'grep -qE "[:.]${port}\\\\$"' not in src

    script = r"""
set -euo pipefail
port=8000
printf '%s\n' '0.0.0.0:8000' '[::]:8000' | awk '{print $1}' | grep -qE "[:.]${port}\$"
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    test_worker_death_detection_wired()
    test_sync_revision_marker_wired()
    test_hf_cli_fallback_wired()
    test_worker_cache_writability_preflight_wired()
    test_running_container_checks_are_pipefail_safe()
    test_check_port_free_detects_representative_listeners()
    print("start.sh bring-up robustness anchors OK")
