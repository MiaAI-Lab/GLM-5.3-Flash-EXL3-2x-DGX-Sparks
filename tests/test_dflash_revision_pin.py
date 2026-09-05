#!/usr/bin/env python3
"""Functional regression for the launcher's immutable DFlash revision pin."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIN = "dc77ff1c99eeb2df044ee3d4f0094eb033fee410"
STALE = "1111111111111111111111111111111111111111"
MODEL = "incoai/GLM-5.3-Flash-DFlash2"
CACHE_NAME = "models--incoai--GLM-5.3-Flash-DFlash2"


def _launcher(tmp: Path) -> Path:
    script = tmp / "start.fn.sh"
    source = (ROOT / "start.sh").read_text()
    assert source.rstrip().endswith('main "$@"')
    script.write_text(source.rstrip()[: -len('main "$@"')] + '"$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (tmp / ".env").write_text("")
    return script


def _env(tmp: Path, **extra: str) -> dict[str, str]:
    blocked = {
        "DFLASH_MODEL",
        "DFLASH_REVISION",
        "HF_BIN",
        "HF_HOME",
        "REFRESH_WEIGHTS",
        "SKIP_DOWNLOAD",
        "SPEC_METHOD",
    }
    env = {key: value for key, value in os.environ.items() if key not in blocked}
    env.update(
        HOME=str(tmp / "home"),
        HF_HOME=str(tmp / "hf"),
        DFLASH_MODEL=MODEL,
        DFLASH_REVISION=PIN,
        SPEC_METHOD="dflash",
        **extra,
    )
    return env


def _snapshot(cache: Path, revision: str) -> Path:
    path = cache / "snapshots" / revision
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}\n")
    (path / "model.safetensors").write_bytes(b"test")
    return path


def _run(script: Path, env: dict[str, str], function: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), function],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_resolve_rewrites_stale_main_to_complete_pin() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = _launcher(tmp)
        cache = tmp / "hf" / "hub" / CACHE_NAME
        _snapshot(cache, STALE)
        _snapshot(cache, PIN)
        ref = cache / "refs" / "main"
        ref.parent.mkdir(parents=True)
        ref.write_text(STALE)

        result = _run(script, _env(tmp), "resolve_dflash_dir")

        assert result.returncode == 0, result.stderr
        assert ref.read_text() == PIN
        assert result.stdout.rstrip().endswith(f"/{CACHE_NAME}/snapshots/{PIN}")


def test_resolve_rejects_stale_cache_when_pin_is_missing() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = _launcher(tmp)
        cache = tmp / "hf" / "hub" / CACHE_NAME
        _snapshot(cache, STALE)
        ref = cache / "refs" / "main"
        ref.parent.mkdir(parents=True)
        ref.write_text(STALE)

        result = _run(script, _env(tmp), "resolve_dflash_dir")

        assert result.returncode == 1
        assert f"pinned DFlash2 snapshot {PIN} is incomplete" in result.stderr
        assert ref.read_text() == STALE


def test_download_does_not_accept_an_unrelated_cached_snapshot() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = _launcher(tmp)
        cache = tmp / "hf" / "hub" / CACHE_NAME
        _snapshot(cache, STALE)
        calls = tmp / "hf-calls"
        fake_hf = tmp / "hf-stub"
        fake_hf.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >>"$DFLASH_TEST_CALLS"
model=$2
shift 2
revision=
while (($#)); do
    if [[ $1 == --revision ]]; then revision=$2; shift 2; else shift; fi
done
cache_name="models--${model//\\//--}"
snapshot="$HF_HOME/hub/$cache_name/snapshots/$revision"
mkdir -p "$snapshot"
printf '{}\\n' >"$snapshot/config.json"
printf test >"$snapshot/model.safetensors"
"""
        )
        fake_hf.chmod(fake_hf.stat().st_mode | stat.S_IXUSR)
        env = _env(
            tmp,
            HF_BIN=str(fake_hf),
            DFLASH_TEST_CALLS=str(calls),
        )

        result = _run(script, env, "download_dflash")

        assert result.returncode == 0, result.stderr
        assert f"--revision {PIN}" in calls.read_text()
        assert (cache / "refs" / "main").read_text() == PIN
        assert (cache / "snapshots" / PIN / "model.safetensors").is_file()


if __name__ == "__main__":
    test_resolve_rewrites_stale_main_to_complete_pin()
    test_resolve_rejects_stale_cache_when_pin_is_missing()
    test_download_does_not_accept_an_unrelated_cached_snapshot()
    print("DFlash immutable revision regression OK")
