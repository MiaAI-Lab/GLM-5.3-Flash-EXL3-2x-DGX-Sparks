#!/usr/bin/env python3
"""Overlay recipe hashing produces a usable image stamp."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def test_overlay_recipe_hash_runs() -> None:
    source = START.read_text()
    begin = source.index("overlay_recipe_hash() {")
    end = source.index("\nimage_recipe_stamp()")
    script = f"SCRIPT_DIR={str(ROOT)!r}\n" + source[begin:end] + "overlay_recipe_hash\n"
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = result.stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{64}", digest), digest


if __name__ == "__main__":
    test_overlay_recipe_hash_runs()
    print("image recipe tests: PASS")
