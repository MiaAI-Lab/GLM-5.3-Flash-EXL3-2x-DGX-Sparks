#!/usr/bin/env python3
"""Host regressions for adaptive-k capture selection; no serving or GPU calls."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from test_launcher_rank_parity import Harness

START = Path(__file__).resolve().parents[1] / "start.sh"
STOCK = "--cudagraph-capture-sizes 1 2 4 8 16 24 32"
UNION = "--cudagraph-capture-sizes 1 2 3 4 5 6 8 9 10 12 15 16 20 24 32"


def run(mode: str | None = None, extra: str = "", **config: str) -> str:
    # Exercise the shipped configuration block, not a copy of its algorithm.
    source = START.read_text()
    begin = source.index('ENFORCE_EAGER="${ENFORCE_EAGER:-0}"')
    end = source.index("# 1 = fused exl3_moe", begin)
    script = "set -euo pipefail\n" + source[begin:end]
    script += '\nconfigure_capture_sizes\nprintf "%s" "${EXTRA_ARGS:-}"\n'
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "SPEC_METHOD": "dflash",
        "DFLASH_TOKENS": "7",
        "MAX_NUM_SEQS": "4",
        "ENFORCE_EAGER": "0",
        "EXTRA_ARGS": extra,
    }
    if mode is not None:
        env["GLM53_ADAPTIVE_K"] = mode
    env.update(config)
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_modes_match_runtime_normalization() -> None:
    for mode in ("ema", "on", "1", "\t EMA \r\n", "\u2003On\u2003"):
        assert run(mode) == UNION, repr(mode)
    for mode in (None, "", "off", "unknown", "e ma"):
        # Disabled modes must not start parsing an unused adaptive k-set.
        assert run(mode, GLM53_ADAPTIVE_K_SET="unused") == STOCK, repr(mode)


def test_configured_k_set_and_batch_geometry() -> None:
    # Do not silently impose the default k=2/4 choices on a custom set.
    assert run("ema", GLM53_ADAPTIVE_K_SET="7, 1,5,5, ,") == (
        "--cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32"
    )
    # Filter k beyond the full draft length and retain full-length warmup shapes
    # even if the configured set does not explicitly include that length.
    assert run(
        "ema", GLM53_ADAPTIVE_K_SET="2,7", DFLASH_TOKENS="4", MAX_NUM_SEQS="3"
    ) == "--cudagraph-capture-sizes 1 2 3 4 5 6 8 9 10 15 16 24 32"
    # Runtime accepts zero and drops negative query lengths rather than rejecting
    # the set; mirror that behavior without changing the runtime's semantics.
    assert run("ema", GLM53_ADAPTIVE_K_SET="-2,-1,0", MAX_NUM_SEQS="3") == (
        "--cudagraph-capture-sizes 1 2 3 4 8 16 24 32"
    )
    assert run("ema", GLM53_ADAPTIVE_K_SET="", MAX_NUM_SEQS="4") == UNION


def test_explicit_capture_override_wins() -> None:
    for extra in (
        "--cudagraph-capture-sizes 1 2 4",
        "--cudagraph-capture-sizes=1 2 4 --kv-cache-memory-bytes 123",
        "--kv-cache-memory-bytes 123\t--cudagraph-capture-sizes\t1 3",
        "\n--cudagraph-capture-sizes 1 5\n",
        "cudagraph-capture-sizes 1 2 4",
    ):
        for mode in ("ema", "off"):
            # Invalid automatic inputs also prove that overrides bypass generation.
            assert run(mode, extra, GLM53_ADAPTIVE_K_SET="unused") == extra


def test_unrelated_extra_args_are_preserved() -> None:
    extra = "--kv-cache-memory-bytes 123"
    assert run("ema", extra) == f"{extra} {UNION}"
    # A similarly named option is not a caller capture-size override.
    extra = "--not-cudagraph-capture-sizes 1"
    assert run("ema", extra) == f"{extra} {UNION}"


def test_eager_and_non_dflash_paths_are_unchanged() -> None:
    for extra in ("", "--kv-cache-memory-bytes 123"):
        assert run("ema", extra, ENFORCE_EAGER="1") == extra
    for method in ("mtp", "none"):
        assert run("ema", SPEC_METHOD=method, GLM53_ADAPTIVE_K_SET="unused") == (
            "--cudagraph-capture-sizes 1 2 3 4 6 8 12"
        )


def test_management_commands_do_not_parse_capture_configuration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        harness = Harness(Path(tmp))
        for command in ("stop", "status", "logs", "--help"):
            result = harness.run(
                command, GLM53_ADAPTIVE_K="ema", GLM53_ADAPTIVE_K_SET="invalid",
                MAX_NUM_SEQS="invalid",
            )
            assert result.returncode == 0, (command, result.stderr)
            assert "ValueError" not in result.stderr
        assert not harness.host_touching_calls(), "help must not touch either host"


def test_invalid_capture_configuration_fails_before_restart_stop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        harness = Harness(Path(tmp))
        result = harness.run(
            "restart", GLM53_ADAPTIVE_K="ema", GLM53_ADAPTIVE_K_SET="invalid",
        )
        assert result.returncode != 0
        assert not harness.host_touching_calls()
        # Numeric validation precedes expansion of the configured capture grid.
        result = harness.run(
            "restart", GLM53_ADAPTIVE_K="ema", GLM53_ADAPTIVE_K_SET="invalid",
            MAX_NUM_SEQS="invalid",
        )
        assert result.returncode == 2
        assert "MAX_NUM_SEQS" in result.stderr
        assert "ValueError" not in result.stderr
        assert not harness.host_touching_calls()


if __name__ == "__main__":
    test_modes_match_runtime_normalization()
    test_configured_k_set_and_batch_geometry()
    test_explicit_capture_override_wins()
    test_unrelated_extra_args_are_preserved()
    test_eager_and_non_dflash_paths_are_unchanged()
    test_management_commands_do_not_parse_capture_configuration()
    test_invalid_capture_configuration_fails_before_restart_stop()
    print("adaptive-k capture-list selection OK")
