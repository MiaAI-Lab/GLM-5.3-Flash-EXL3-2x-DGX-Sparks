#!/usr/bin/env python3
"""Defaults and overlay recipe-stamp rebuild contract."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
DOCKERFILE = ROOT / "Dockerfile"
ENV_EXAMPLE = ROOT / ".env.example"


def test_documented_defaults() -> None:
    start = START.read_text()
    example = ENV_EXAMPLE.read_text()
    assert 'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-7168}"' in start
    assert 'EXL3_FAT_KERNEL="${EXL3_FAT_KERNEL:-1}"' in start
    assert "MAX_NUM_BATCHED_TOKENS=7168" in example
    assert re.search(r"^EXL3_FAT_KERNEL=1$", example, re.M)


def test_recipe_stamp_wiring() -> None:
    start = START.read_text()
    dockerfile = DOCKERFILE.read_text()
    assert "overlay_recipe_hash() {" in start
    assert "image_recipe_stamp() {" in start
    assert 'SKIP_BUILD:-0' in start
    assert "--build-arg" in start and "GLM53_RECIPE_STAMP" in start
    assert "ARG GLM53_RECIPE_STAMP=unknown" in dockerfile
    assert "LABEL glm53.recipe.stamp=${GLM53_RECIPE_STAMP}" in dockerfile
    assert dockerfile.rstrip().endswith("LABEL glm53.recipe.stamp=${GLM53_RECIPE_STAMP}")


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
    test_documented_defaults()
    test_recipe_stamp_wiring()
    test_overlay_recipe_hash_runs()
    print("image recipe tests: PASS")
