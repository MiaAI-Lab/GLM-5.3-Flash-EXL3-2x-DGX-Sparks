#!/usr/bin/env python3
"""CPU-only tests for the prebuilt abliterated model preset."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = "bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit"
REVISION = "14858211ed81d7fa773f8a0db02f38f36d230252"
CACHE = "models--bullerwins--GLM-5.3-Flash-exl3-4bpw-ablit"
CONTROLLED_ENV = {
    "GLM53_MODEL_PRESET",
    "SKIP_DOWNLOAD",
    "SKIP_SYNC",
    "SPEC_METHOD",
    "HF_HOME",
    "WORKER_HOME",
}


def _prefix(marker: str) -> str:
    source = (ROOT / "start.sh").read_text()
    prefix, found, _rest = source.partition(marker)
    assert found, f"start.sh marker missing: {marker}"
    return prefix


def _run(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    process_env = {k: v for k, v in os.environ.items() if k not in CONTROLLED_ENV}
    process_env.update(env)
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=process_env
    )


def test_wrapper_selects_preset_and_forwards_arguments() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        wrapper = tmp / "start-abliterated.sh"
        wrapper.write_text((ROOT / "start-abliterated.sh").read_text())
        wrapper.chmod(0o755)
        delegate = tmp / "start.sh"
        delegate.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'PRESET=%s\\n' \"$GLM53_MODEL_PRESET\"\n"
            "printf 'ARG=%s\\n' \"$@\"\n"
        )
        delegate.chmod(0o755)
        result = subprocess.run(
            [str(wrapper), "restart", "sentinel"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "GLM53_MODEL_PRESET": "wrong"},
        )
    assert result.stdout.splitlines() == [
        "PRESET=abliterated",
        "ARG=restart",
        "ARG=sentinel",
    ]


def test_preset_is_pinned_and_preserves_regular_serve_settings() -> None:
    probe = r'''
printf '%s\n' "$MODEL" "$MODEL_FALLBACK" "$MODEL_REVISION" "$MODEL_SNAPSHOT"
printf '%s\n' "$MODEL_CACHE_NAME" "$MODEL_FALLBACK_CACHE_NAME" "$ABLIT" "$PORT"
printf '%s\n' "$EXPECTED_SHARDS"
'''
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(
            _prefix("# ------------------------------- helpers -----------------------------------")
            + probe
        )
        script.chmod(0o755)
        (tmp / ".env").write_text(
            "MODEL=wrong/model\nMODEL_FALLBACK=wrong/fallback\n"
            "MODEL_REVISION=wrong\nMODEL_CACHE_NAME=wrong-cache\n"
            "MODEL_FALLBACK_CACHE_NAME=wrong-fallback-cache\nABLIT=1\nPORT=9123\n"
        )
        # A caller EXPECTED_SHARDS must not lower the pinned preset inventory.
        selected = _run(
            script, {"GLM53_MODEL_PRESET": "abliterated", "EXPECTED_SHARDS": "1"}
        )
        regular = _run(script, {"GLM53_MODEL_PRESET": "", "EXPECTED_SHARDS": "1"})

    assert selected.returncode == 0, selected.stderr
    assert selected.stdout.splitlines() == [
        MODEL,
        MODEL,
        REVISION,
        REVISION,
        CACHE,
        CACHE,
        "0",
        "9123",
        "120",
    ]
    assert regular.returncode == 0, regular.stderr
    assert regular.stdout.splitlines()[:3] == [
        "wrong/model",
        "wrong/fallback",
        "wrong",
    ]
    assert regular.stdout.splitlines()[-3:] == ["0", "9123", "1"]


def _make_snapshot(repo: Path, revision: str, shards: int = 120) -> None:
    """Snapshot with hub-style shard links into repo/blobs.

    Every link resolves to a real blob file, so a complete snapshot here also
    exercises the launcher's link-following shard count.
    """
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n")
    (snapshot / "model.safetensors.index.json").write_text("{}\n")
    blobs = repo / "blobs"
    blobs.mkdir(exist_ok=True)
    for index in range(1, shards + 1):
        blob = blobs / f"{revision}-{index:05d}"
        blob.touch()
        (snapshot / f"model-{index:05d}-of-00120.safetensors").symlink_to(
            os.path.relpath(blob, snapshot)
        )


def test_exact_snapshot_wins_over_refs_and_is_required_on_both_nodes() -> None:
    probe = r'''
worker_ssh() { bash -c "$*"; }
sync_weights
resolved="$(resolve_model_dir)"
marker_rev="$(sync_repo_marker_rev "$MODEL_PATH" "$MODEL_SNAPSHOT")"
printf '%s\n' "$resolved" "$marker_rev"
'''
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        hf_home = tmp / "hf"
        head_repo = hf_home / "hub" / CACHE
        worker_home = tmp / "worker"
        worker_repo = worker_home / ".cache" / "huggingface" / "hub" / CACHE
        other = "f" * 40
        _make_snapshot(head_repo, REVISION)
        _make_snapshot(head_repo, other)
        _make_snapshot(worker_repo, REVISION)
        (head_repo / "refs").mkdir()
        (head_repo / "refs" / "main").write_text(other)
        (worker_repo / ".glm53-exl3-synced").write_text(REVISION)

        script = tmp / "start.sh"
        script.write_text(
            _prefix(
                "# ------------------------ inner container scripts --------------------------"
            )
            + probe
        )
        script.chmod(0o755)
        (tmp / ".env").write_text(
            f"HF_HOME={hf_home}\nWORKER_HOME={worker_home}\n"
        )
        env = {
            "GLM53_MODEL_PRESET": "abliterated",
            "SKIP_DOWNLOAD": "1",
            "SPEC_METHOD": "none",
        }
        passed = _run(script, env)
        assert passed.returncode == 0, passed.stderr
        assert passed.stdout.splitlines()[-2:] == [
            f"/root/.cache/huggingface/hub/{CACHE}/snapshots/{REVISION}",
            REVISION,
        ]
        skipped = _run(script, {**env, "SKIP_SYNC": "1"})
        assert skipped.returncode == 0, skipped.stderr

        # A shard link whose blob is gone must not count on the worker either:
        # the pinned gate follows links to real files on both nodes.
        worker_blob = worker_repo / "blobs" / f"{REVISION}-00120"
        worker_blob.unlink()
        failed_worker = _run(script, env)
        assert failed_worker.returncode != 0
        worker_blob.touch()

        head_shard = head_repo / "snapshots" / REVISION / "model-00120-of-00120.safetensors"
        head_shard.unlink()
        failed_head = _run(script, env)
        assert failed_head.returncode != 0


def test_failed_download_cannot_adopt_another_cached_revision() -> None:
    probe = r'''
resolve_hf_bin() { HF_BIN_CMD=(/usr/bin/python3 "$FAKE_HF_BIN"); }
download_weights
'''
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        hf_home = tmp / "hf"
        repo = hf_home / "hub" / CACHE
        _make_snapshot(repo, REVISION, shards=119)
        _make_snapshot(repo, "f" * 40)
        script = tmp / "start.sh"
        script.write_text(
            _prefix(
                "# ------------------------ inner container scripts --------------------------"
            )
            + probe
        )
        script.chmod(0o755)
        (tmp / ".env").write_text(f"HF_HOME={hf_home}\n")
        fake_hf = tmp / "hf.py"
        fake_hf.write_text("""import os
import sys
from pathlib import Path
if os.environ.get("FAIL_DOWNLOAD") == "1":
    raise SystemExit(1)
args = sys.argv[1:]
revision = args[args.index("--revision") + 1] if "--revision" in args else "f" * 40
repo = Path(os.environ["FAKE_HF_REPO"])
blob = repo / "blobs" / f"{revision}-00120"
blob.touch()
shard = repo / "snapshots" / revision / "model-00120-of-00120.safetensors"
if not shard.exists():
    shard.symlink_to(f"../../blobs/{blob.name}")
""")
        env = {
            "GLM53_MODEL_PRESET": "abliterated",
            "SKIP_DOWNLOAD": "0",
            "SPEC_METHOD": "none",
            # A caller value must not lower the pinned 120-shard gate.
            "EXPECTED_SHARDS": "1",
            "FAKE_HF_BIN": str(fake_hf),
            "FAKE_HF_REPO": str(repo),
        }
        failed = _run(script, {**env, "FAIL_DOWNLOAD": "1"})
        assert failed.returncode != 0
        pinned_shard = repo / "snapshots" / REVISION / "model-00120-of-00120.safetensors"
        assert not pinned_shard.exists()
        # The real download wrapper must select the pinned revision. A request
        # for refs/main merely refreshes the other snapshot and cannot pass.
        completed = _run(script, env)
        assert completed.returncode == 0, completed.stderr
        assert pinned_shard.is_file()


if __name__ == "__main__":
    test_wrapper_selects_preset_and_forwards_arguments()
    test_preset_is_pinned_and_preserves_regular_serve_settings()
    test_exact_snapshot_wins_over_refs_and_is_required_on_both_nodes()
    test_failed_download_cannot_adopt_another_cached_revision()
    print("abliterated preset tests: PASS")
