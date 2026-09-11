#!/usr/bin/env python3
"""Exercise the launcher's GID preflight without Docker, SSH, or real sysfs."""
import os
from pathlib import Path
import subprocess
import tempfile

SOURCE = (Path(__file__).resolve().parents[1] / "start.sh").read_text()
BLOCK = SOURCE[SOURCE.index("    # Each rank's GID"):SOURCE.index('    [ "$TP" = "2" ]')]

def check(selector, broken=None):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for node in ("head", "worker"):
            for hca in ("rocep1s0f0", "roceP2p1s0f0"):
                path = root / node / hca / "ports/1/gids/3"
                path.parent.mkdir(parents=True)
                path.write_text("0000:0000:0000:0000:0000:ffff:0a64:5001\n")
        if broken:
            node, hca, kind = broken
            path = root / node / hca / "ports/1/gids/3"
            if kind == "missing":
                path.unlink()
            else:
                path.write_text("0000:0000:0000:0000:0000:0000:0000:0000\n")
        block = BLOCK.replace("/sys/class/infiniband", str(root / "head"))
        script = """
set -euo pipefail
warn() { echo "$*" >&2; }
die() { echo "$*" >&2; exit 1; }
worker_ssh() {
    local cmd="$1"
    bash -c "${cmd//"HEADROOT"/"WORKERROOT"}"
}
preflight_gids() {
BLOCK
}
preflight_gids
""".replace("HEADROOT", str(root / "head")).replace("WORKERROOT", str(root / "worker")).replace("BLOCK", block)
        env = dict(os.environ, HEAD_CX7_IB=selector, WORKER_CX7_IB=selector,
                   HEAD_GID="3", WORKER_GID="3")
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)

def test_gid_preflight():
    dual = "rocep1s0f0,roceP2p1s0f0"
    for selector in ("rocep1s0f0", dual):
        result = check(selector)
        assert result.returncode == 0, result.stderr
    for node, kind in (("head", "missing"), ("worker", "zero")):
        result = check(dual, (node, "roceP2p1s0f0", kind))
        assert result.returncode != 0, "bad second HCA passed preflight"
        assert node in result.stderr and "roceP2p1s0f0" in result.stderr, result.stderr

if __name__ == "__main__":
    test_gid_preflight()
    print("single/dual HCA preflight and missing/zero secondary GID checks passed")
