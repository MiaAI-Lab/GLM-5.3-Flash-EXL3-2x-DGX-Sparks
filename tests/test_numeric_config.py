#!/usr/bin/env python3
"""CPU-only tests for launcher numeric type/range validation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def guard_source() -> str:
    source = START.read_text()
    begin = source.index("# GLM53 numeric config guard (begin)")
    end_marker = "# GLM53 numeric config guard (end)"
    end = source.index(end_marker, begin) + len(end_marker)
    return source[begin:end]


def validate(util: str, model: str, seqs: str, batch: str) -> subprocess.CompletedProcess[str]:
    script = (
        guard_source()
        + '\nGPU_MEM_UTIL="$1"; MAX_MODEL_LEN="$2"; MAX_NUM_SEQS="$3"; '
        + 'MAX_NUM_BATCHED_TOKENS="$4"\n'
        + 'validate_numeric_config || exit $?\n'
        + 'printf "%s|%s|%s|%s\\n" "$GPU_MEM_UTIL" "$MAX_MODEL_LEN" '
        + '"$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS"\n'
    )
    return subprocess.run(
        ["bash", "-c", script, "test", util, model, seqs, batch],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )


def expect_rc(values: tuple[str, str, str, str], expected: int) -> None:
    result = validate(*values)
    assert result.returncode == expected, (values, result.returncode, result.stdout, result.stderr)


def test_matrix() -> None:
    expect_rc(("0.87", "1000000", "4", "1024"), 0)
    expect_rc((".87", "01000000", "0004", "01024"), 0)
    expect_rc(("1.0", "1000000", "4096", "8388608"), 0)
    expect_rc(("0", "1000000", "4", "1024"), 2)
    expect_rc(("8.7", "1000000", "4", "1024"), 2)
    expect_rc(("nope", "1000000", "4", "1024"), 2)
    expect_rc(("0.87", "0", "4", "1024"), 2)
    expect_rc(("0.87", "1000001", "4", "1024"), 2)
    expect_rc(("0.87", "1000000", "O4", "1024"), 2)
    expect_rc(("0.87", "1000000", "4097", "1024"), 2)
    expect_rc(("0.87", "1000000", "4", "1024\r"), 2)
    expect_rc(("0.87", "1000000", "4", "18446744073709551615"), 2)


def test_decimal_normalization() -> None:
    result = validate(".87", "01000000", "0004", "01024")
    assert result.returncode == 0
    assert result.stdout.strip() == ".87|1000000|4|1024"


def test_restart_validates_before_stop() -> None:
    source = START.read_text()
    main = source.index("main() {")
    validation = source.index("start|restart) validate_numeric_config", main)
    restart = source.index("restart)  stop; start", main)
    assert validation < restart


if __name__ == "__main__":
    test_matrix()
    test_decimal_normalization()
    test_restart_validates_before_stop()
    print("numeric config tests: PASS")
