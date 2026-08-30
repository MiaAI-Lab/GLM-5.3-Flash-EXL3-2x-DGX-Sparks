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


def test_other_inline_overrides_win() -> None:
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    values = {
        "DFLASH_TOKENS": ("3", "7"),
        "MAX_MODEL_LEN": ("800000", "900000"),
        "MAX_NUM_BATCHED_TOKENS": ("512", "1024"),
        "CG_ESTIMATE": ("1", "0"),
    }
    output = "".join(f'{name}=${{{name}:-unset}}\\n' for name in values)

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(preamble + f'\nprintf "{output}"\n')
        script.chmod(0o755)
        (tmp / ".env").write_text(
            "".join(f"{name}={from_env}\\n" for name, (from_env, _) in values.items())
        )

        env = os.environ.copy()
        env.update({name: inline for name, (_, inline) in values.items()})
        result = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.stdout.splitlines() == [
        f"{name}={inline}" for name, (_, inline) in values.items()
    ]


if __name__ == "__main__":
    test_max_num_seqs_inline_override_wins()
    test_other_inline_overrides_win()
    print("start.sh caller override regression OK")
