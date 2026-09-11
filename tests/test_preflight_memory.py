#!/usr/bin/env python3
"""CPU-only tests for the GB10 MemAvailable preflight guard."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
GIB_KIB = 1048576


def guard_source() -> str:
    source = START.read_text()
    begin = source.index("# GLM53 preflight memory guard (begin)")
    end_marker = "# GLM53 preflight memory guard (end)"
    end = source.index(end_marker, begin) + len(end_marker)
    return source[begin:end]


def call_memory(total: int, available: int, util: str) -> subprocess.CompletedProcess[str]:
    script = guard_source() + '\npreflight_memory worker "$1" "$2" "$3"\n'
    return subprocess.run(
        ["bash", "-c", script, "test", str(total), str(available), util],
        text=True,
        capture_output=True,
        check=False,
    )


def expect_rc(total_gib: int, available_gib: int, util: str, expected: int) -> None:
    result = call_memory(total_gib * GIB_KIB, available_gib * GIB_KIB, util)
    assert result.returncode == expected, (result.returncode, result.stdout, result.stderr)


def test_matrix() -> None:
    expect_rc(122, 102, "0.87", 1)
    expect_rc(122, 116, "0.87", 0)
    expect_rc(122, 117, "0.87", 0)
    expect_rc(122, 100, "1.0", 1)
    result = call_memory(122 * GIB_KIB, 1024, "0.87")
    assert result.returncode == 1
    expect_rc(100, 52, "0.5", 0)
    result = call_memory(100 * GIB_KIB, 52 * GIB_KIB - 1, "0.5")
    assert result.returncode == 1
    result = call_memory(122 * GIB_KIB, 116 * GIB_KIB, "bad")
    assert result.returncode == 2
    result = call_memory(122 * GIB_KIB, 116 * GIB_KIB, "0")
    assert result.returncode == 2
    result = call_memory(122 * GIB_KIB, 116 * GIB_KIB, "1.1")
    assert result.returncode == 2


def test_reads_memavailable_not_memfree() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        fixture = Path(raw_tmp) / "meminfo"
        fixture.write_text(
            "MemTotal:       127926272 kB\n"
            "MemFree:             1024 kB\n"
            "MemAvailable:    121634816 kB\n"
        )
        script = guard_source() + '\nread_meminfo_kib "$1"\n'
        result = subprocess.run(
            ["bash", "-c", script, "test", str(fixture)],
            text=True,
            capture_output=True,
            check=True,
        )
    assert result.stdout.strip() == "127926272 121634816"


def test_guard_is_wired_after_port_checks() -> None:
    source = START.read_text()
    port = source.index('check_port_free "$MASTER_PORT" MASTER_PORT')
    head = source.index('preflight_memory head "$head_total"')
    worker = source.index('preflight_memory worker "$worker_total"')
    assert port < head < worker


if __name__ == "__main__":
    test_matrix()
    test_reads_memavailable_not_memfree()
    test_guard_is_wired_after_port_checks()
    print("preflight memory tests: PASS")
