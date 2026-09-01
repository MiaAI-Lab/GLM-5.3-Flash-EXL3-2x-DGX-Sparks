#!/usr/bin/env python3
"""Regression test for caller overrides that must win over ``.env``."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_max_num_seqs_inline_override_wins() -> None:
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(
            preamble
            + '\nprintf "MAX_NUM_SEQS=%s\\n" "${MAX_NUM_SEQS:-unset}"\n'
        )
        script.chmod(0o755)
        (tmp / ".env").write_text("MAX_NUM_SEQS=2\n")

        env = os.environ.copy()
        env["MAX_NUM_SEQS"] = "4"
        result = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.stdout.strip() == "MAX_NUM_SEQS=4"


def _preamble_probe(env: dict[str, str], env_file: str, probe: str) -> str:
    """Run start.sh's pre-configuration preamble with a synthetic .env."""
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(preamble + "\n" + probe + "\n")
        script.chmod(0o755)
        (tmp / ".env").write_text(env_file)
        result = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, **env},
        )
    return result.stdout.strip()


def test_default_reasoning_effort_caller_override_is_setness_aware() -> None:
    """An explicitly EMPTY caller value must beat .env, not be swallowed by it.

    The knob's own default is empty, so ``[ -n "$_cli_x" ]`` cannot tell
    ``GLM53_DEFAULT_REASONING_EFFORT= ./start.sh`` (deliberately back to the
    template default) apart from an unset var. Only ``${VAR+1}`` can.
    """
    probe = 'printf "EFFORT=[%s]\\n" "${GLM53_DEFAULT_REASONING_EFFORT-unset}"'
    env_file = "GLM53_DEFAULT_REASONING_EFFORT=high\n"

    # caller unset -> .env wins
    os.environ.pop("GLM53_DEFAULT_REASONING_EFFORT", None)
    assert _preamble_probe({}, env_file, probe) == "EFFORT=[high]"

    # caller sets a value -> caller wins
    assert _preamble_probe(
        {"GLM53_DEFAULT_REASONING_EFFORT": "low"}, env_file, probe
    ) == "EFFORT=[low]"

    # caller sets it EMPTY -> caller still wins (the setness-aware case)
    assert _preamble_probe(
        {"GLM53_DEFAULT_REASONING_EFFORT": ""}, env_file, probe
    ) == "EFFORT=[]"


if __name__ == "__main__":
    test_max_num_seqs_inline_override_wins()
    test_default_reasoning_effort_caller_override_is_setness_aware()
    print("start.sh caller override regression OK")
