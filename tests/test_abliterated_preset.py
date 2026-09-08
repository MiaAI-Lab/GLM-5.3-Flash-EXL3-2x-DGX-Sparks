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
        selected = _run(script, {"GLM53_MODEL_PRESET": "abliterated"})
        regular = _run(script, {"GLM53_MODEL_PRESET": ""})

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
    ]
    assert regular.returncode == 0, regular.stderr
    assert regular.stdout.splitlines()[:3] == [
        "wrong/model",
        "wrong/fallback",
        "wrong",
    ]
    assert regular.stdout.splitlines()[-2:] == ["1", "9123"]


def _make_snapshot(repo: Path, revision: str, shards: int = 120) -> None:
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n")
    (snapshot / "model.safetensors.index.json").write_text("{}\n")
    for index in range(1, shards + 1):
        (snapshot / f"model-{index:05d}-of-00120.safetensors").touch()


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

        worker_shard = (
            worker_repo / "snapshots" / REVISION / "model-00120-of-00120.safetensors"
        )
        worker_shard.unlink()
        failed_worker = _run(script, env)
        assert failed_worker.returncode != 0
        assert "incomplete on worker" in failed_worker.stderr
        worker_shard.touch()

        head_shard = head_repo / "snapshots" / REVISION / "model-00120-of-00120.safetensors"
        head_shard.unlink()
        failed_head = _run(script, env)
        assert failed_head.returncode != 0
        assert "pinned model snapshot is incomplete" in failed_head.stderr


def test_failed_download_cannot_adopt_another_cached_revision() -> None:
    probe = r'''
resolve_hf_bin() { return 0; }
hf_download_repo() { return 1; }
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
        result = _run(
            script,
            {
                "GLM53_MODEL_PRESET": "abliterated",
                "SKIP_DOWNLOAD": "0",
                "SPEC_METHOD": "none",
            },
        )
    assert result.returncode != 0
    assert "119 / 120 shards in the selected snapshot" in result.stderr


if __name__ == "__main__":
    test_wrapper_selects_preset_and_forwards_arguments()
    test_preset_is_pinned_and_preserves_regular_serve_settings()
    test_exact_snapshot_wins_over_refs_and_is_required_on_both_nodes()
    test_failed_download_cannot_adopt_another_cached_revision()
    print("abliterated preset tests: PASS")
