#!/usr/bin/env python3
"""Host regressions for drafter revision selection and download."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"

PIN = "dc77ff1c99eeb2df044ee3d4f0094eb033fee410"


def source() -> str:
    return START.read_text(encoding="utf-8")


def function(name: str) -> str:
    text = source()
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}\n", text)
    assert match, f"missing function {name}"
    return match.group(0)


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )


def test_revision_override_and_empty_value_survive_env() -> None:
    preamble = source().split('DFLASH_TOKENS=')[0]
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "start.sh"
        script.write_text(preamble + '\nprintf "[%s]" "$DFLASH_REVISION"\n')
        env_file = Path(tmp) / ".env"
        env = {"PATH": os.environ["PATH"], "HOME": tmp, "USER": "fixture"}
        for configured, caller, expected in (
            ("", None, PIN),
            (f"DFLASH_REVISION={PIN}\n", "override", "override"),
            (f"DFLASH_REVISION={PIN}\n", "", ""),
            ("DFLASH_REVISION=\n", None, ""),
        ):
            env_file.write_text(configured)
            current = env.copy()
            if caller is not None:
                current["DFLASH_REVISION"] = caller
            result = subprocess.run(
                ["bash", str(script)], capture_output=True, text=True, env=current
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout == f"[{expected}]", result.stdout


def test_resolution_and_sync_marker_ignore_stale_main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("old")
        for revision in ("old", PIN):
            snapshot = repo / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").touch()
            (snapshot / "model.safetensors").touch()
        script = f"""
set -euo pipefail
log() {{ :; }}
die() {{ printf '%s\\n' "$*" >&2; exit 97; }}
{function("ensure_dflash_refs_main")}
{function("resolve_dflash_dir")}
{function("sync_repo_marker_rev")}
DFLASH_PATH={shlex.quote(tmp)}
DFLASH_CACHE_NAME=fixture
DFLASH_REVISION={PIN}
resolve_dflash_dir
printf '\\n'
sync_repo_marker_rev "$DFLASH_PATH" "$DFLASH_REVISION"
"""
        result = run_bash(script)
        assert result.returncode == 0, result.stderr
        path, marker = result.stdout.splitlines()
        assert path.endswith(f"/snapshots/{PIN}"), path
        assert marker == PIN, marker
        (repo / "snapshots" / PIN / "model.safetensors").unlink()
        result = run_bash(script)
        assert result.returncode == 97, result.stdout
        (repo / "snapshots" / PIN / "config.json").unlink()
        (repo / "snapshots" / PIN).rmdir()
        result = run_bash(script)
        assert result.returncode == 97, result.stdout


def test_download_and_resolution_use_the_pin() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args_file = Path(tmp) / "hf-args"
        script = f"""
set -euo pipefail
log() {{ :; }}
die() {{ printf 'DIE:%s\\n' "$*" >&2; exit 97; }}
{function("ensure_dflash_refs_main")}
{function("resolve_dflash_dir")}
{function("download_dflash")}
resolve_hf_bin() {{ HF_BIN_CMD=(mock_hf); }}
mock_hf() {{
    printf '%s\\n' "$@" > "$ARGS_FILE"
    mkdir -p "$DFLASH_PATH/snapshots/$DFLASH_REVISION"
    touch "$DFLASH_PATH/snapshots/$DFLASH_REVISION/config.json"
    touch "$DFLASH_PATH/snapshots/$DFLASH_REVISION/model.safetensors"
}}
ARGS_FILE={shlex.quote(str(args_file))}
DFLASH_PATH={shlex.quote(str(Path(tmp) / "dflash"))}
DFLASH_CACHE_NAME=models--incoai--DFlash
DFLASH_MODEL=incoai/DFlash
DFLASH_REVISION={PIN}
HF_CACHE_DIR={shlex.quote(tmp)}
SPEC_METHOD=dflash
SKIP_DOWNLOAD=0
REFRESH_WEIGHTS=0
download_dflash
printf 'resolved=%s\\n' "$(resolve_dflash_dir)"
"""
        result = run_bash(script)
        assert result.returncode == 0, result.stderr
        assert args_file.read_text(encoding="utf-8").splitlines() == [
            "download",
            "incoai/DFlash",
            "--revision",
            PIN,
        ]
        assert result.stdout.strip().endswith(f"/snapshots/{PIN}")


if __name__ == "__main__":
    test_resolution_and_sync_marker_ignore_stale_main()
    test_revision_override_and_empty_value_survive_env()
    test_download_and_resolution_use_the_pin()
    print("dflash revision-pin guard OK")
