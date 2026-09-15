#!/usr/bin/env python3
"""Run the real preflight with fake sysfs and local-only worker commands."""

from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HCAS = {
    "head": ("rocep1s0f1", "roceP2p1s0f1"),
    "worker": ("rocep1s0f0", "roceP2p1s0f0"),
}
GIDS = {"head": "4", "worker": "3"}
POPULATED = "0000:0000:0000:0000:0000:ffff:c000:0201"
ZERO = "0000:0000:0000:0000:0000:0000:0000:0000"

# Only preflight is loaded: no .env, entrypoint, model, or deployment access.
# Its body is executed unchanged, following the function-extraction pattern in
# test_long_prefill_threshold.py. All host-facing dependencies are replaced.
STUBS = r"""
set -euo pipefail
warn() { printf '%s\n' "$*" >&2; }
die() { warn "$*"; exit 1; }
log() { printf '%s\n' "$*"; }
docker() { return 0; }
curl() { return 0; }
rsync() { return 0; }
ip() { printf 'inet %s/24\n' "$HEAD_IP"; }
nvidia-smi() { printf 'GPU 0: GB10\n'; }
hostname() { printf 'fake-head\n'; }
check_port_free() { return 0; }
df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted\nfake 999999999 0 999999999 0%% /\n'; }
cat() {
    # Never fall through to real sysfs (or any other host file).
    case "$1" in
        /sys/class/infiniband/*)
            command cat "$FAKE_SYSFS/$FAKE_NODE/${1#/sys/class/infiniband/}" ;;
        *) printf 'unexpected cat: %s\n' "$*" >&2; return 1 ;;
    esac
}
worker_ssh() {
    # A fresh local shell gives the worker its own fake sysfs tree. The real
    # ssh executable is never invoked, even for the initial connectivity check.
    FAKE_NODE=worker bash -c "$1"
}
export -f cat docker nvidia-smi df
"""


def run_preflight(tmp_path, head_count=2, worker_count=2, broken=None):
    source = (ROOT / "start.sh").read_text()
    begin = source.index("preflight() {")
    end = source.index("\n}\n", begin) + 3
    for node in HCAS:
        for hca in HCAS[node]:
            port = tmp_path / "sysfs" / node / hca / "ports/1"
            (port / "gids").mkdir(parents=True)
            (port / "gid_attrs/types").mkdir(parents=True)
            for index in range(8):
                # Only this rank's index and an alternative RoCEv2 index are
                # usable. Using the other rank's configured index must fail.
                value = POPULATED if str(index) in (GIDS[node], "1") else ZERO
                (port / "gids" / str(index)).write_text(value + "\n")
                (port / "gid_attrs/types" / str(index)).write_text("RoCE v2\n")
    if broken:
        node, hca_index, kind = broken
        entry = (tmp_path / "sysfs" / node / HCAS[node][hca_index]
                 / "ports/1/gids" / GIDS[node])
        if kind == "missing":
            entry.unlink()
        else:
            entry.write_text(ZERO + "\n")

    # The unrelated overlay existence and cache writability checks stay inside
    # the exercised function; their inputs are temporary files, never checkout
    # or deployment artifacts.
    for name in ("overlay/patch_ablit.py", "overlay/ablit_runtime.py", "ablit/LAYER_MAP.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    placeholder = tmp_path / "overlay-placeholder"
    placeholder.touch()
    env = {
        "PATH": "/usr/bin:/bin", "LC_ALL": "C", "HOME": str(tmp_path),
        "USER": "fake-user", "FAKE_SYSFS": str(tmp_path / "sysfs"),
        "FAKE_NODE": "head", "HEAD_IP": "192.0.2.1", "WORKER_SSH": "fake-worker",
        "HEAD_CX7_IB": ",".join(HCAS["head"][:head_count]),
        "WORKER_CX7_IB": ",".join(HCAS["worker"][:worker_count]),
        "HEAD_GID": GIDS["head"], "WORKER_GID": GIDS["worker"],
        "TP": "2", "NNODES": "2", "CONTAINER_WORKER": "fake-worker",
        "PORT": "8888", "MASTER_PORT": "29521", "SCRIPT_DIR": str(tmp_path),
        "ABLIT": "0", "HF_CACHE_DIR": str(tmp_path / "head-cache"),
        "WORKER_HOME": str(tmp_path), "WORKER_CACHE_DIR": str(tmp_path / "worker-cache"),
    }
    for key in ("STOP_PATCH_HOST", "SCHED_PATCH_HOST", "DRAFTER_PATCH_HOST",
                "APC_PATCH_HOST", "PERGROUP_PATCH_HOST", "XGRAMMAR_PATCH_HOST",
                "KPOOL_TAIL_PATCH_HOST", "SPINWAIT_PATCH_HOST", "ADAPTIVE_K_PATCH_HOST",
                "DENSE_FP8_PATCH_HOST", "EXL3_OVERLAY_HOST"):
        env[key] = str(placeholder)
    return subprocess.run(
        ["bash", "-c", STUBS + source[begin:end] + "\npreflight\n"],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )


@pytest.mark.parametrize("head_count,worker_count", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_literal_nic_lists_on_both_nodes(tmp_path, head_count, worker_count):
    result = run_preflight(tmp_path, head_count, worker_count)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("node", ["head", "worker"])
@pytest.mark.parametrize("hca_index", [0, 1], ids=["primary", "secondary"])
@pytest.mark.parametrize("kind", ["missing", "zero"])
def test_every_selected_nic_must_have_a_gid(tmp_path, node, hca_index, kind):
    result = run_preflight(tmp_path, broken=(node, hca_index, kind))
    assert result.returncode != 0, "preflight accepted an unusable selected NIC"
    # Operators must be able to identify the failing node, device, and index.
    assert any(node in line and HCAS[node][hca_index] in line and GIDS[node] in line
               for line in result.stderr.splitlines()), result.stderr


def test_failure_displays_alternative_gids_and_types_for_every_device(tmp_path):
    result = run_preflight(tmp_path, broken=("worker", 1, "zero"))
    assert result.returncode != 0
    rows = [line.split() for line in result.stderr.splitlines()]
    for node, devices in HCAS.items():
        for device in devices:
            assert [node, device, "gid1:", POPULATED, "RoCE", "v2"] in rows, result.stderr
