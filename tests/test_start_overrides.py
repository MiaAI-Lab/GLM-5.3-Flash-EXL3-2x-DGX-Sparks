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


def test_mixed_prefill_chunk_inline_override_wins() -> None:
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(
            preamble
            + '\nprintf "GLM53_MIXED_PREFILL_CHUNK=%s\\n" "${GLM53_MIXED_PREFILL_CHUNK:-unset}"\n'
        )
        script.chmod(0o755)
        (tmp / ".env").write_text("GLM53_MIXED_PREFILL_CHUNK=2\n")

        env = os.environ.copy()
        env["GLM53_MIXED_PREFILL_CHUNK"] = "skip"
        result = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.stdout.strip() == "GLM53_MIXED_PREFILL_CHUNK=skip"


if __name__ == "__main__":
    test_max_num_seqs_inline_override_wins()
    test_mixed_prefill_chunk_inline_override_wins()
    print("start.sh caller override regression OK")
