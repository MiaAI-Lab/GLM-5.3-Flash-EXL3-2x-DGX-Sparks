#!/usr/bin/env python3
"""Defaults and overlay recipe-stamp rebuild contract."""

from __future__ import annotations

import re
import subprocess
import tempfile
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


# A pull must never silently replace an image whose recipe stamp already
# matched the repo. ``ensure_image`` decides the rebuild from the image present
# *before* the pull; with a locally built (BUILD=1) image the stamp matched, so
# no rebuild was scheduled — and the pull then swapped in the published image
# (different stamp, no locally compiled artifacts). That image was launched, and
# a config that needs the Dockerfile build (``GLM53_EXL3_MOE_FAST=1`` requires the
# patched extension) failed closed at load, so the pair never became healthy.
# The stamp is therefore re-checked on the image that will actually run.
ENSURE_IMAGE_HARNESS = """
set -u
STATE=__STATE__
IMAGE=ghcr.io/example/kit:exl3-instanttensor
LOGDIR="$STATE/logs"; mkdir -p "$LOGDIR"
docker() { return 0; }
log()  { printf '[log] %s\\n' "$*"; }
warn() { printf '[warn] %s\\n' "$*"; }
die()  { printf '[die] %s\\n' "$*"; exit 9; }
overlay_recipe_hash() { printf 'REPOSTAMP\\n'; }
image_recipe_stamp()  { cat "$STATE/stamp" 2>/dev/null || true; }
build_image() { printf 'REPOSTAMP' > "$STATE/stamp"; printf 'BUILD\\n' >> "$STATE/actions"; }
pull_image()  { printf 'PUBLISHED' > "$STATE/stamp"; printf 'PULL\\n' >> "$STATE/actions"; }
pull_image_on_worker() { printf 'WORKERPULL\\n' >> "$STATE/actions"; return 1; }
ship_image_to_worker() { printf 'SHIP\\n' >> "$STATE/actions"; }
image_from_registry() { return 0; }
local_image_key()  { printf 'headkey\\n'; }
worker_image_key() { printf 'workerkey\\n'; }
images_match() { [ "$1" = "$2" ]; }
worker_ssh() { return 0; }
worker_ok=0
SKIP_OVERLAY_VERIFY=1
__BODY__
"""

RUN_TAIL = """
ensure_image
printf 'FINAL_STAMP=%s\\n' "$(cat "$STATE/stamp")"
printf 'ACTIONS=%s\\n' "$(tr '\\n' ',' < "$STATE/actions" 2>/dev/null)"
"""


def _ensure_image_body() -> str:
    source = START.read_text()
    begin = source.index("ensure_image() {")
    end = source.index("adopt_complete_weights() {", begin)
    body = source[begin:end]
    return body[: body.rindex("}") + 1]


def test_pull_rechecks_recipe_stamp_on_the_image_that_runs() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        state = Path(raw_tmp) / "state"
        state.mkdir()
        # the local image is a BUILD=1 overlay build: its stamp equals the repo's
        (state / "stamp").write_text("REPOSTAMP")

        harness = (
            ENSURE_IMAGE_HARNESS.replace("__STATE__", str(state))
            .replace("__BODY__", _ensure_image_body())
            + RUN_TAIL
        )
        result = subprocess.run(["bash", "-c", harness], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    final = [line for line in result.stdout.splitlines() if line.startswith("FINAL_STAMP=")]
    actions = [line for line in result.stdout.splitlines() if line.startswith("ACTIONS=")]
    assert final == ["FINAL_STAMP=REPOSTAMP"], result.stdout
    assert "BUILD" in actions[0], result.stdout
    assert "not rebuilding" not in result.stdout, result.stdout


def test_pull_recheck_respects_skip_build() -> None:
    """``SKIP_BUILD=1`` still means "keep GHCR" — the re-check must not fight it."""
    with tempfile.TemporaryDirectory() as raw_tmp:
        state = Path(raw_tmp) / "state"
        state.mkdir()
        (state / "stamp").write_text("REPOSTAMP")

        harness = (
            ENSURE_IMAGE_HARNESS.replace("__STATE__", str(state))
            .replace("__BODY__", _ensure_image_body())
            + "\nSKIP_BUILD=1\n"
            + RUN_TAIL
        )
        result = subprocess.run(["bash", "-c", harness], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "FINAL_STAMP=PUBLISHED" in result.stdout, result.stdout


if __name__ == "__main__":
    test_documented_defaults()
    test_recipe_stamp_wiring()
    test_overlay_recipe_hash_runs()
    test_pull_rechecks_recipe_stamp_on_the_image_that_runs()
    test_pull_recheck_respects_skip_build()
    print("image recipe tests: PASS")
