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


def _pull_helper(path: Path) -> str:
    source = path.read_text()
    begin = source.index("pull_image_keeping_repo_stamp() {")
    end = source.index("\nensure_image() {", begin)
    return source[begin:end]


def test_pull_helper_matches_on_every_launcher() -> None:
    body = _pull_helper(START)
    assert _pull_helper(ROOT / "start-tp3.sh") == body
    assert _pull_helper(ROOT / "start-tp4.sh") == body
    for name in ("start.sh", "start-tp3.sh", "start-tp4.sh"):
        text = (ROOT / name).read_text()
        assert 'pull_image_keeping_repo_stamp "$wanted_stamp" "$have_stamp"' in text
        assert '[ "${PULL_KEPT_LOCAL:-0}" != "1" ]' in text


# Fake docker: tags live as files, image ids map to recipe stamps.
PULL_HARNESS = r"""
set -u
STATE=__STATE__
IMAGE=ghcr.io/example/kit:exl3-instanttensor
mkdir -p "$STATE/tag" "$STATE/id"
enc() { printf '%s' "$1" | tr '/:' '__'; }
put() { printf '%s' "$2" > "$STATE/tag/$(enc "$1")"; }
get() { cat "$STATE/tag/$(enc "$1")"; }
put "$IMAGE" local1
printf 'REPO' > "$STATE/id/local1"
printf '__PUB_STAMP__' > "$STATE/id/pub1"
docker() {
    if [ "$1" = tag ]; then put "$3" "$(get "$2")"; return 0; fi
    if [ "$1" = image ]; then get "$5"; return 0; fi
    if [ "$1" = pull ]; then
        if [ "${FAIL_PULL:-0}" = 1 ]; then return 1; fi
        put "$2" pub1
        printf 'PULL\n' >> "$STATE/actions"
        return 0
    fi
    if [ "$1" = rmi ]; then printf 'RMI %s\n' "$2" >> "$STATE/actions"; return 0; fi
    printf 'unexpected docker %s\n' "$*" >&2
    return 1
}
login_ghcr_if_token() { :; }
log() { :; }
warn() { printf '[warn] %s\n' "$*"; }
die() {
    printf '[die] %s\n' "$*"
    printf 'TAG=%s\n' "$(get "$IMAGE" 2>/dev/null || true)"
    printf 'ACTIONS=%s\n' "$(tr '\n' ',' < "$STATE/actions" 2>/dev/null || true)"
    exit 9
}
image_recipe_stamp() { cat "$STATE/id/$(get "$IMAGE")"; }
pull_image() { put "$IMAGE" pub1; printf 'PLAIN\n' >> "$STATE/actions"; }
__BODY__
pull_image_keeping_repo_stamp "$WANTED" "$HAVE"
printf 'KEPT=%s\n' "$PULL_KEPT_LOCAL"
printf 'TAG=%s\n' "$(get "$IMAGE")"
printf 'ACTIONS=%s\n' "$(tr '\n' ',' < "$STATE/actions" 2>/dev/null || true)"
"""


def _run_pull(have: str, wanted: str, pub_stamp: str, *, skip_build: str = "0", fail_pull: str = "0") -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as raw:
        state = Path(raw) / "state"
        state.mkdir()
        harness = (
            PULL_HARNESS.replace("__STATE__", str(state)).replace("__PUB_STAMP__", pub_stamp).replace("__BODY__", _pull_helper(START))
        )
        env = {
            "WANTED": wanted,
            "HAVE": have,
            "SKIP_BUILD": skip_build,
            "FAIL_PULL": fail_pull,
            "PATH": "/usr/bin:/bin",
        }
        return subprocess.run(["bash", "-c", harness], capture_output=True, text=True, env=env)


def test_pull_keeps_local_image_when_published_stamp_differs() -> None:
    result = _run_pull("REPO", "REPO", "PUBLISHED")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "KEPT=1" in result.stdout
    assert "TAG=local1" in result.stdout
    assert "ACTIONS=PULL," in result.stdout
    assert "PLAIN" not in result.stdout
    assert "kept the local image" in result.stdout


def test_pull_adopts_published_image_when_stamp_matches() -> None:
    result = _run_pull("REPO", "REPO", "REPO")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "KEPT=0" in result.stdout
    assert "TAG=pub1" in result.stdout


def test_skip_build_still_replaces_the_local_tag() -> None:
    result = _run_pull("REPO", "REPO", "PUBLISHED", skip_build="1")
    assert result.returncode == 0, result.stderr + result.stdout
    assert "KEPT=0" in result.stdout
    assert "TAG=pub1" in result.stdout
    assert "PLAIN" in result.stdout


def test_failed_pull_restores_the_held_tag() -> None:
    result = _run_pull("REPO", "REPO", "PUBLISHED", fail_pull="1")
    assert result.returncode == 9, result.stdout
    assert "TAG=local1" in result.stdout
    assert "docker pull" in result.stdout


if __name__ == "__main__":
    test_documented_defaults()
    test_recipe_stamp_wiring()
    test_overlay_recipe_hash_runs()
    test_pull_helper_matches_on_every_launcher()
    test_pull_keeps_local_image_when_published_stamp_differs()
    test_pull_adopts_published_image_when_stamp_matches()
    test_skip_build_still_replaces_the_local_tag()
    test_failed_pull_restores_the_held_tag()
    print("image recipe tests: PASS")
