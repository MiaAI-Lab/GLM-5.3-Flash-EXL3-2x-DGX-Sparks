#!/usr/bin/env python3
"""Worker sync: marker trust, snapshot verification and drafter repair.

Runs start.sh's real sync helpers in bash against two fixtures — a head-side HF
cache and a "worker" tree. `worker_ssh` executes the probe command locally
against the worker fixture, and `rsync` is the real rsync with only the
`host:` prefix of the destination rewritten, so the probe expression, the rsync
flags (-a vs -aL) and the marker write are all exercised as shipped; the
test-owned `rsync-fail` file makes a transfer die the way an interrupted one
does. No SSH, network, GPU, container or service is involved.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"

REV = "dc77ff1c99eeb2df044ee3d4f0094eb033fee410"
TARGET_CACHE = "models--test--target"
DRAFT_CACHE = "models--test--draft"
SHARDS = (
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
)
EXPECTED_SHARDS = str(len(SHARDS))
DRAFT_WEIGHT = "DFlash2 draft weight bytes\n"


def _payload(name: str) -> str:
    """The head's bytes for an entry — what a complete worker copy holds."""
    return f"payload {name}"


def _function(name: str) -> str:
    """Slice a top-level ``name() { ... }`` definition out of start.sh."""
    src = START.read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}\n", src)
    assert match, f"missing function {name}"
    return match.group(0)


# The probe runs against the fixture tree; only the "host:" prefix is rewritten.
HARNESS = """\
set -euo pipefail
WORKER_CACHE_DIR="@@WORKER_CACHE@@"
WORKER_SSH=worker.invalid
FORCE_SYNC=@@FORCE@@
RSYNC_LOG="@@TMP@@/rsync.log"
: >"$RSYNC_LOG"

log() { printf '[log] %s\\n' "$*"; }
warn() { printf '[warn] %s\\n' "$*" >&2; }
die() { printf 'ERROR: %s\\n' "$*" >&2; exit 1; }

worker_ssh() { bash -c "$1"; }

rsync() {
    local -a translated=()
    local arg
    for arg in "$@"; do
        case "$arg" in
            "$WORKER_SSH":*) translated+=("${arg#"$WORKER_SSH":}") ;;
            *) translated+=("$arg") ;;
        esac
    done
    printf 'rsync %s\\n' "$*" >>"$RSYNC_LOG"
    # An interrupted transfer: the destination keeps whatever it already had.
    [ ! -e "@@TMP@@/rsync-fail" ] || return 23
    command rsync "${translated[@]}"
}

@@FUNCTIONS@@
@@TAIL@@
"""

SYNC_FUNCTIONS = (
    "sync_repo_marker_rev",
    "snapshot_required_sizes",
    "worker_snapshot_complete",
    "sync_repo_to_worker",
)

VERIFY_FUNCTIONS = (
    "snapshot_required_sizes",
    "worker_snapshot_complete",
    "verify_worker_model_snapshot",
)


# ------------------------------- fixtures ----------------------------------


def _blob(repo: Path, name: str, payload: str) -> Path:
    path = repo / "blobs" / name
    path.write_text(payload, encoding="utf-8")
    return path


def _link(snap: Path, name: str, target: Path) -> None:
    """Snapshot entry as the HF cache writes it: a relative link into blobs."""
    (snap / name).symlink_to(os.path.relpath(target, snap))


def _repo(
    repo: Path,
    *,
    shards: Sequence[str] = SHARDS,
    extra: Sequence[str] = (),
    index: bool = True,
    valid_index: bool = True,
    config: bool = True,
) -> Path:
    """Create an HF-cache-shaped repo; returns its pinned snapshot dir.

    The index always lists every shard in SHARDS: callers leave shards out to
    build truncated trees.
    """
    snap = repo / "snapshots" / REV
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(REV, encoding="utf-8")
    (repo / "blobs").mkdir()
    for shard in shards:
        _link(snap, shard, _blob(repo, shard, _payload(shard)))
    for name in extra:
        (snap / name).write_text("unrelated payload\n", encoding="utf-8")
    if config:
        (snap / "config.json").write_text("{}\n", encoding="utf-8")
    if index:
        text = (
            json.dumps({"weight_map": {f"layer.{i}": s for i, s in enumerate(SHARDS)}})
            if valid_index
            else "{ truncated"
        )
        (snap / "model.safetensors.index.json").write_text(text, encoding="utf-8")
    return snap


def _marker(repo: Path) -> Path:
    return repo / ".glm53-exl3-synced"


def _draft_source(tmp: Path, *, config_outside: bool = False) -> Path:
    """Head drafter whose model weight links outside the synced repo tree.

    With ``config_outside`` the drafter's config.json does too — the layout
    where the bounded weight repair cannot make the snapshot complete.
    """
    repo = tmp / "head" / "hub" / DRAFT_CACHE
    snap = repo / "snapshots" / REV
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(REV, encoding="utf-8")
    (repo / "blobs").mkdir()
    outside = tmp / "head" / "hub" / "elsewhere"
    outside.mkdir()
    (outside / "config.json").write_text("{}\n", encoding="utf-8")
    (outside / "model.safetensors").write_text(DRAFT_WEIGHT, encoding="utf-8")
    _link(snap, "model.safetensors", outside / "model.safetensors")
    if config_outside:
        _link(snap, "config.json", outside / "config.json")
    else:
        (snap / "config.json").write_text("{}\n", encoding="utf-8")
    return repo


def _draft_worker(worker_cache: Path, *, config: bool = True) -> Path:
    """Drafter already on the worker: readable config, dangling weight link."""
    repo = worker_cache / "hub" / DRAFT_CACHE
    snap = repo / "snapshots" / REV
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(REV, encoding="utf-8")
    if config:
        (snap / "config.json").write_text("{}\n", encoding="utf-8")
    (snap / "model.safetensors").symlink_to("../../../elsewhere/model.safetensors")
    return repo


# ------------------------------- harness -----------------------------------


def _run(tmp: Path, script: str) -> subprocess.CompletedProcess[str]:
    path = tmp / "case.sh"
    path.write_text(script, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "USER": "fixture"}
    return subprocess.run(
        ["bash", str(path)], capture_output=True, text=True, env=env, cwd=tmp
    )


def _run_sync(
    tmp: Path,
    src: Path,
    worker_cache: Path,
    cache_name: str,
    label: str,
    *,
    force: bool = False,
    functions: Sequence[str] = SYNC_FUNCTIONS,
) -> subprocess.CompletedProcess[str]:
    tail = (
        f'sync_repo_to_worker "{src}" "{cache_name}" "{label}" "{REV}"\n'
        "printf 'sync done\\n'\n"
    )
    return _script_run(tmp, worker_cache, force, functions, tail)


def _script_run(
    tmp: Path,
    worker_cache: Path,
    force: bool,
    functions: Sequence[str],
    tail: str,
) -> subprocess.CompletedProcess[str]:
    script = (
        HARNESS.replace("@@WORKER_CACHE@@", str(worker_cache))
        .replace("@@TMP@@", str(tmp))
        .replace("@@FORCE@@", "1" if force else "0")
        .replace("@@FUNCTIONS@@", "\n".join(_function(name) for name in functions))
        .replace("@@TAIL@@", tail)
    )
    return _run(tmp, script)


def _rsync_calls(tmp: Path) -> list[str]:
    log = tmp / "rsync.log"
    return [line for line in log.read_text(encoding="utf-8").splitlines() if line]


def _count_probe(snap: Path) -> int:
    """A count-only probe, to show what the index + size check adds over it."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"find -L '{snap}' -maxdepth 1 -type f -name '*.safetensors' "
            "2>/dev/null | wc -l | tr -d '[:space:]'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return int(result.stdout)


# ------------------------------- regressions -------------------------------


def test_complete_snapshot_with_matching_marker_skips_rsync() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "head" / "hub" / TARGET_CACHE
        _repo(src)
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / TARGET_CACHE
        _repo(worker_repo, extra=("unrelated-extra.safetensors",))
        _marker(worker_repo).write_text(REV, encoding="utf-8")

        result = _run_sync(tmp_path, src, worker_cache, TARGET_CACHE, "weights")

        assert result.returncode == 0, result.stderr
        assert "rsync skipped" in result.stdout, result.stdout
        assert _rsync_calls(tmp_path) == []


def test_missing_required_shard_is_not_certified_by_an_unrelated_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "head" / "hub" / TARGET_CACHE
        _repo(src)
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / TARGET_CACHE
        snap = _repo(worker_repo, shards=SHARDS[:-1], extra=("stray.safetensors",))
        _marker(worker_repo).write_text(REV, encoding="utf-8")
        # The count contract alone would accept this tree; the index does not.
        assert _count_probe(snap) == int(EXPECTED_SHARDS)

        result = _run_sync(tmp_path, src, worker_cache, TARGET_CACHE, "weights")

        assert result.returncode == 0, result.stderr
        assert "snapshot is incomplete — re-syncing" in result.stderr, result.stderr
        assert len(_rsync_calls(tmp_path)) == 1, _rsync_calls(tmp_path)
        assert (snap / SHARDS[-1]).exists(), "re-sync did not restore the shard"
        assert _marker(worker_repo).read_text(encoding="utf-8") == REV


def test_nonempty_truncated_shard_is_not_certified() -> None:
    """A nonempty --partial leftover must not pass for the head's shard."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "head" / "hub" / TARGET_CACHE
        _repo(src)
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / TARGET_CACHE
        snap = _repo(worker_repo)
        _marker(worker_repo).write_text(REV, encoding="utf-8")
        truncated = SHARDS[0]
        (snap / truncated).write_text("7 bytes", encoding="utf-8")
        assert (snap / truncated).read_text(encoding="utf-8") == "7 bytes"

        result = _run_sync(tmp_path, src, worker_cache, TARGET_CACHE, "weights")

        assert result.returncode == 0, result.stderr
        assert "snapshot is incomplete — re-syncing" in result.stderr, result.stderr
        assert len(_rsync_calls(tmp_path)) == 1, _rsync_calls(tmp_path)
        assert (snap / truncated).read_text(encoding="utf-8") == _payload(truncated)
        assert _marker(worker_repo).read_text(encoding="utf-8") == REV


def test_directory_in_place_of_a_shard_fails_closed() -> None:
    """A nonempty directory passed the old test -s probe; nothing may certify it."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "head" / "hub" / TARGET_CACHE
        _repo(src)
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / TARGET_CACHE
        snap = _repo(worker_repo)
        _marker(worker_repo).write_text(REV, encoding="utf-8")
        (snap / SHARDS[0]).unlink()
        (snap / SHARDS[0]).mkdir()
        (snap / SHARDS[0] / "leftover").write_text("x", encoding="utf-8")

        result = _run_sync(tmp_path, src, worker_cache, TARGET_CACHE, "weights")

        assert result.returncode != 0
        assert "rsync skipped" not in result.stdout, result.stdout
        assert "snapshot is incomplete — re-syncing" in result.stderr, result.stderr
        assert not _marker(worker_repo).exists(), "a failed sync must not keep the old marker"


def test_symlink_to_a_directory_is_not_certified() -> None:
    """A shard entry resolving to a directory must not stand in for the file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "head" / "hub" / TARGET_CACHE
        _repo(src)
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / TARGET_CACHE
        snap = _repo(worker_repo)
        _marker(worker_repo).write_text(REV, encoding="utf-8")
        decoy = worker_repo / "blobs" / "decoy"
        decoy.mkdir()
        (decoy / "weight").write_text("z" * 64, encoding="utf-8")
        (snap / SHARDS[0]).unlink()
        (snap / SHARDS[0]).symlink_to(os.path.relpath(decoy, snap))
        assert (snap / SHARDS[0]).is_dir()

        result = _run_sync(tmp_path, src, worker_cache, TARGET_CACHE, "weights")

        assert result.returncode == 0, result.stderr
        assert "rsync skipped" not in result.stdout, result.stdout
        assert "snapshot is incomplete — re-syncing" in result.stderr, result.stderr
        assert len(_rsync_calls(tmp_path)) == 1, _rsync_calls(tmp_path)
        assert (snap / SHARDS[0]).read_text(encoding="utf-8") == _payload(SHARDS[0])
        assert _marker(worker_repo).read_text(encoding="utf-8") == REV


def test_missing_head_shard_fails_closed_without_a_transfer() -> None:
    """A head the loader cannot satisfy must not start a 164 GiB transfer."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "head" / "hub" / TARGET_CACHE
        _repo(src, shards=SHARDS[:-1])
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / TARGET_CACHE
        _repo(worker_repo)
        _marker(worker_repo).write_text("stale", encoding="utf-8")

        result = _run_sync(tmp_path, src, worker_cache, TARGET_CACHE, "weights")

        assert result.returncode != 0
        assert "cannot be verified on the head" in result.stderr, result.stderr
        assert _rsync_calls(tmp_path) == []
        assert _marker(worker_repo).read_text(encoding="utf-8") == "stale"


def test_head_sidecar_that_is_not_a_regular_file_fails_closed() -> None:
    """A head entry the loader cannot read as a file is not a sync candidate."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "head" / "hub" / TARGET_CACHE
        snap = _repo(src)
        (snap / "config.json").unlink()
        (snap / "config.json").mkdir()
        (snap / "config.json" / "leftover").write_text("x", encoding="utf-8")
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / TARGET_CACHE
        _repo(worker_repo, extra=("unrelated-extra.safetensors",))
        _marker(worker_repo).write_text("stale", encoding="utf-8")

        result = _run_sync(tmp_path, src, worker_cache, TARGET_CACHE, "weights")

        assert result.returncode != 0
        assert "cannot be verified on the head" in result.stderr, result.stderr
        assert _rsync_calls(tmp_path) == []
        assert _marker(worker_repo).read_text(encoding="utf-8") == "stale"


def test_dangling_draft_weight_is_repaired_by_value() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = _draft_source(tmp_path)
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = _draft_worker(worker_cache)
        _marker(worker_repo).write_text(REV, encoding="utf-8")
        weight = worker_repo / "snapshots" / REV / "model.safetensors"
        assert weight.is_symlink() and not weight.exists()

        result = _run_sync(
            tmp_path, src, worker_cache, DRAFT_CACHE, "DFlash2 draft"
        )

        assert result.returncode == 0, result.stderr
        assert "snapshot is incomplete — re-syncing" in result.stderr, result.stderr
        calls = _rsync_calls(tmp_path)
        assert len(calls) == 2, calls
        assert "-aL" not in calls[0] and "-a " in calls[0], calls
        assert "-aL" in calls[1], calls
        assert not weight.is_symlink(), "repair left the dangling link in place"
        assert weight.read_text(encoding="utf-8") == DRAFT_WEIGHT
        assert _marker(worker_repo).read_text(encoding="utf-8") == REV


def test_failed_repair_does_not_stamp_marker() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # The drafter's config.json also links outside the synced tree, so it
        # stays dangling on the worker: only the weight is repairable AT ALL,
        # and the snapshot must still fail closed.
        src = _draft_source(tmp_path, config_outside=True)
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / DRAFT_CACHE
        (worker_repo / "snapshots" / REV).mkdir(parents=True)
        weight = worker_repo / "snapshots" / REV / "model.safetensors"

        result = _run_sync(
            tmp_path, src, worker_cache, DRAFT_CACHE, "DFlash2 draft"
        )

        assert result.returncode != 0
        assert "snapshot is incomplete on worker" in result.stderr, result.stderr
        assert not _marker(worker_repo).exists()
        assert not weight.is_symlink(), "the bounded weight repair did not run"
        assert weight.read_text(encoding="utf-8") == DRAFT_WEIGHT
        config = worker_repo / "snapshots" / REV / "config.json"
        assert config.is_symlink() and not config.exists()


def test_forced_sync_ignores_a_matching_marker() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "head" / "hub" / TARGET_CACHE
        _repo(src)
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / TARGET_CACHE
        _repo(worker_repo)
        _marker(worker_repo).write_text(REV, encoding="utf-8")

        result = _run_sync(
            tmp_path, src, worker_cache, TARGET_CACHE, "weights", force=True
        )

        assert result.returncode == 0, result.stderr
        assert "rsync skipped" not in result.stdout, result.stdout
        calls = _rsync_calls(tmp_path)
        assert len(calls) == 1, calls
        assert _marker(worker_repo).read_text(encoding="utf-8") == REV


def test_unusable_head_index_fails_closed_without_a_transfer() -> None:
    """A truncated index cannot name the loader's files: refuse, do not count."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "head" / "hub" / TARGET_CACHE
        _repo(src, valid_index=False)
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / TARGET_CACHE
        _repo(worker_repo)
        _marker(worker_repo).write_text("stale", encoding="utf-8")

        result = _run_sync(tmp_path, src, worker_cache, TARGET_CACHE, "weights")

        assert result.returncode != 0
        assert "cannot be verified on the head" in result.stderr, result.stderr
        assert _rsync_calls(tmp_path) == []
        assert _marker(worker_repo).read_text(encoding="utf-8") == "stale"


def test_interrupted_sync_clears_the_marker_and_the_next_run_repairs() -> None:
    """A transfer that dies mid-flight must not leave a complete-sync claim."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "head" / "hub" / TARGET_CACHE
        _repo(src)
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / TARGET_CACHE
        snap = _repo(worker_repo)
        _marker(worker_repo).write_text(REV, encoding="utf-8")
        partial = SHARDS[-1]
        (snap / partial).write_text("partial", encoding="utf-8")
        (tmp_path / "rsync-fail").write_text("", encoding="utf-8")

        interrupted = _run_sync(tmp_path, src, worker_cache, TARGET_CACHE, "weights")

        assert interrupted.returncode != 0
        assert not _marker(worker_repo).exists(), "the old marker survived a failed sync"
        assert (snap / partial).read_text(encoding="utf-8") == "partial"

        (tmp_path / "rsync-fail").unlink()
        repaired = _run_sync(tmp_path, src, worker_cache, TARGET_CACHE, "weights")

        assert repaired.returncode == 0, repaired.stderr
        assert (snap / partial).read_text(encoding="utf-8") == _payload(partial)
        assert _marker(worker_repo).read_text(encoding="utf-8") == REV


def _run_verify(tmp: Path, src: Path, worker_cache: Path) -> subprocess.CompletedProcess[str]:
    """Run verify_worker_model_snapshot — the SKIP_SYNC path — against fixtures."""
    tail = (
        f'MODEL_SNAPSHOT="{REV}"\nMODEL_PATH="{src}"\n'
        f'MODEL_CACHE_NAME="{TARGET_CACHE}"\nNFS_SHARE=0\n'
        "verify_worker_model_snapshot\nprintf 'verify ok\\n'\n"
    )
    return _script_run(tmp, worker_cache, False, VERIFY_FUNCTIONS, tail)


def test_pinned_skip_sync_verification_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "head" / "hub" / TARGET_CACHE
        _repo(src)
        worker_cache = tmp_path / "worker" / ".cache" / "huggingface"
        worker_repo = worker_cache / "hub" / TARGET_CACHE
        snap = _repo(worker_repo)

        ok = _run_verify(tmp_path, src, worker_cache)
        assert ok.returncode == 0, ok.stderr
        assert "verify ok" in ok.stdout, ok.stdout

        (snap / SHARDS[0]).unlink()
        (snap / "stray.safetensors").write_text("unrelated payload\n", encoding="utf-8")
        missing = _run_verify(tmp_path, src, worker_cache)
        assert missing.returncode != 0
        assert "pinned model snapshot is incomplete on worker" in missing.stderr, missing.stderr

        shutil.rmtree(worker_repo)
        snap = _repo(worker_repo)
        (snap / SHARDS[1]).write_text("7 bytes", encoding="utf-8")
        truncated = _run_verify(tmp_path, src, worker_cache)
        assert truncated.returncode != 0
        assert "pinned model snapshot is incomplete on worker" in truncated.stderr, truncated.stderr


if __name__ == "__main__":
    test_complete_snapshot_with_matching_marker_skips_rsync()
    test_missing_required_shard_is_not_certified_by_an_unrelated_file()
    test_nonempty_truncated_shard_is_not_certified()
    test_directory_in_place_of_a_shard_fails_closed()
    test_symlink_to_a_directory_is_not_certified()
    test_missing_head_shard_fails_closed_without_a_transfer()
    test_head_sidecar_that_is_not_a_regular_file_fails_closed()
    test_dangling_draft_weight_is_repaired_by_value()
    test_failed_repair_does_not_stamp_marker()
    test_forced_sync_ignores_a_matching_marker()
    test_unusable_head_index_fails_closed_without_a_transfer()
    test_interrupted_sync_clears_the_marker_and_the_next_run_repairs()
    test_pinned_skip_sync_verification_fails_closed()
    print("worker snapshot sync regression OK")
