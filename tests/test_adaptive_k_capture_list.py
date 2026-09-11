#!/usr/bin/env python3
"""start.sh adds the adaptive-k CUDA-graph capture list without duplicating EXTRA_ARGS."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

START = Path(__file__).resolve().parents[1] / "start.sh"
UNION = "1 2 3 4 5 6 8 9 10 12 15 16 20 24 32"
STOCK = "1 2 4 8 16 24 32"


def capture_block() -> str:
    src = START.read_text()
    start = src.index('ENFORCE_EAGER="${ENFORCE_EAGER:-0}"')
    end = src.index("\nfi\n", src.index("esac", start)) + len("\nfi\n")
    return src[start:end]


def run(mode: str | None, extra: str | None) -> str:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "SPEC_METHOD": "dflash",
        "ENFORCE_EAGER": "0",
        "EXTRA_ARGS": extra or "",
    }
    if mode is not None:
        env["GLM53_ADAPTIVE_K"] = mode
    script = 'set -u; ' + capture_block().rstrip("\n") + '; printf "%s" "$EXTRA_ARGS"'
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def main() -> int:
    assert run("off", "") == f"--cudagraph-capture-sizes {STOCK}"
    for mode in ("ema", "on", "1", "EMA"):
        assert run(mode, "") == f"--cudagraph-capture-sizes {UNION}", mode
    # Caller-supplied graphs always win; no duplication.
    for mode in ("ema", "off"):
        assert run(mode, "--cudagraph-capture-sizes 1 2 4") == "--cudagraph-capture-sizes 1 2 4", mode
    print("adaptive-k capture-list selection OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
