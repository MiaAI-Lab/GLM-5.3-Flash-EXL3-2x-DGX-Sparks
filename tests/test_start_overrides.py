#!/usr/bin/env python3
"""Regression test for caller overrides that must win over ``.env``."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _config_preamble() -> str:
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"
    return preamble


def _configuration_fragment() -> str:
    source = (ROOT / "start.sh").read_text()
    marker = "# 1 = fused exl3_moe (decode)."
    fragment, separator, _rest = source.partition(marker)
    assert separator, "start.sh post-graph configuration marker is missing"
    return fragment


def _run_configuration(
    env_overrides: dict[str, str], command: str = "start"
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(
            _configuration_fragment()
            + '\nprintf "EXTRA_ARGS=%s\\n" "${EXTRA_ARGS:-}"\n'
        )
        script.chmod(0o755)
        settings = {
            "DFLASH_TOKENS": "7",
            "DFLASH_VERIFY_TOKENS": "0",
            "MAX_NUM_SEQS": "4",
        }
        settings.update(env_overrides)
        (tmp / ".env").write_text(
            "".join(f"{key}={value}\n" for key, value in settings.items())
        )
        env = os.environ.copy()
        for key in settings:
            env.pop(key, None)
        return subprocess.run(
            ["bash", str(script), command], capture_output=True, text=True, env=env
        )


def test_dflash_inline_overrides_win() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(
            _config_preamble()
            + '\nprintf "DFLASH_TOKENS=%s DFLASH_VERIFY_TOKENS=%s\\n" '
            '"${DFLASH_TOKENS:-unset}" "${DFLASH_VERIFY_TOKENS:-unset}"\n'
        )
        script.chmod(0o755)
        (tmp / ".env").write_text(
            "DFLASH_TOKENS=7\nDFLASH_VERIFY_TOKENS=0\n"
        )

        env = os.environ.copy()
        env["DFLASH_TOKENS"] = "3"
        env["DFLASH_VERIFY_TOKENS"] = "1"
        result = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.stdout.strip() == "DFLASH_TOKENS=3 DFLASH_VERIFY_TOKENS=1"


def test_verify_width_is_advertised_to_cuda_graph_manager() -> None:
    source = (ROOT / "start.sh").read_text()
    assignment = (
        'spec["num_speculative_tokens_per_batch_size"]='
        '[[1,int(os.environ["MAX_NUM_SEQS"]),verify]]'
    )
    # The head and worker must construct byte-equivalent speculative configs.
    assert source.count(assignment) == 2


def test_runtime_width_capture_sizes_and_early_validation() -> None:
    result = _run_configuration(
        {
            "DFLASH_TOKENS": "7",
            "DFLASH_VERIFY_TOKENS": "3",
            "MAX_NUM_SEQS": "5",
        }
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "EXTRA_ARGS=--cudagraph-capture-sizes 1 2 4 8 12 16 20 24 32"
    )

    for variable, value, expected in (
        ("DFLASH_VERIFY_TOKENS", "wat", "must be 0 or a positive integer"),
        ("DFLASH_VERIFY_TOKENS", "2", "must be a graph-aligned prefix"),
        ("MAX_NUM_SEQS", "0", "must be a positive integer"),
        ("MAX_NUM_SEQS", "65", "must be at most 64"),
    ):
        result = _run_configuration({variable: value})
        assert result.returncode != 0
        assert expected in result.stderr

    # Malformed launch tuning must never block recovery/status subcommands.
    result = _run_configuration(
        {"DFLASH_TOKENS": "wat", "DFLASH_VERIFY_TOKENS": "also-wat"},
        command="stop",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "EXTRA_ARGS="

    # An irrelevant DFlash proposal value must not block an MTP rollback.
    result = _run_configuration(
        {"SPEC_METHOD": "mtp", "DFLASH_TOKENS": "wat"}
    )
    assert result.returncode == 0, result.stderr


def test_runtime_width_patches_are_wired_to_both_ranks() -> None:
    source = (ROOT / "start.sh").read_text()
    for patch_name in (
        "patch_dflash2_verify_width.py",
        "patch_gdn_runtime_width.py",
    ):
        assert source.count(f"python3 /opt/glm53/{patch_name}") == 2
        assert source.count(f"/opt/glm53/{patch_name}:ro") == 2


if __name__ == "__main__":
    test_dflash_inline_overrides_win()
    test_verify_width_is_advertised_to_cuda_graph_manager()
    test_runtime_width_capture_sizes_and_early_validation()
    test_runtime_width_patches_are_wired_to_both_ranks()
    print("start.sh caller override regression OK")
