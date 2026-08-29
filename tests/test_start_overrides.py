#!/usr/bin/env python3
"""Regression test for caller overrides that must win over ``.env``."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_max_num_seqs_inline_override_wins() -> None:
    source = (ROOT / "start.sh").read_text(encoding="utf-8")
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    if os.name == "nt":
        # On Windows, verify bash variable precedence via direct parsing / execution
        assert "_cli_max_num_seqs=\"${MAX_NUM_SEQS-}\"" in preamble
        assert '[ -n "${_cli_max_num_seqs}" ] && MAX_NUM_SEQS="$_cli_max_num_seqs"' in preamble
        return

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(
            preamble
            + '\nprintf "MAX_NUM_SEQS=%s\\n" "${MAX_NUM_SEQS:-unset}"\n',
            encoding="utf-8",
            newline="\n"
        )
        script.chmod(0o755)
        (tmp / ".env.example").write_text("MAX_NUM_SEQS=2\n", encoding="utf-8", newline="\n")
        (tmp / ".env").write_text("MAX_NUM_SEQS=2\n", encoding="utf-8", newline="\n")

        env = os.environ.copy()
        env["MAX_NUM_SEQS"] = "4"
        result = subprocess.run(
            ["bash", "start.sh"],
            cwd=str(tmp),
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        assert "MAX_NUM_SEQS=4" in result.stdout.strip()


if __name__ == "__main__":
    test_max_num_seqs_inline_override_wins()
    print("start.sh caller override regression OK")
