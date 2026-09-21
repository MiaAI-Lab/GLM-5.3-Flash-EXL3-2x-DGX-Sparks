"""CPU-only Bash contract tests; no SSH, Docker or GPU required."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("BASH_EXE") or shutil.which("bash")
ENV = dict(os.environ)
if BASH:
    ENV["PATH"] = str(Path(BASH).parent) + os.pathsep + ENV.get("PATH", "")
HELPER = (ROOT / "files/switchless-ring.sh").read_text(encoding="utf-8")
CONFIG = """
NCCL_SWITCHLESS_RING_ONLY=1 TP=4 NNODES=4 USE_HOST_NCCL=1
HEAD_IP=10.1.1.1 WORKER_IP=10.1.1.2 WORKER2_IP=10.1.1.3 WORKER3_IP=10.1.1.4
"""
GID = """
RING_HCAS='=hca0:1,hca1:1' RING_GID=3
cat() {
  case "$1" in
    */state) echo '4: ACTIVE';;
    */types/3) echo 'RoCE v2';;
    */gids/3) echo '0000:0000:0000:0000:0000:ffff:0a00:0001';;
    *) return 1;;
  esac
}
"""
CLUSTER = """
NCCL_HOST_DIR=/trusted NCCL_SO_NAME=libnccl.so.2.30.7 IMAGE=test-image
HEAD_CX7_IB='hca0,hca1' HEAD_GID=3
_tp4_rank_nccl_dir() { echo /trusted; }
_tp4_rank_cx7_ib() { echo hca0,hca1; }
_tp4_rank_gid() { echo 3; }
worker_ssh_n() { bash -s; }
ring_probe_node() { printf 'image-id\\thash\\t/pip/libnccl.so.2\\n'; }
"""
PROBE = """
RING_DIR=/dev NCCL_SO_NAME=null IMAGE=test RING_REQUIRE_IDLE=1
ring_check_gid() { return 0; }
grep() { return 0; }
sha256sum() { echo 'testhash  /dev/null'; }
docker() { if [[ $1 == image ]]; then echo sha256:image; else echo /pip/libnccl.so.2; fi; }
nvidia-smi() { return 0; }
"""

@unittest.skipUnless(BASH, "Bash required")
class RingTests(unittest.TestCase):
    def run_script(self, body, success=True):
        result = subprocess.run([BASH, "-s"], input="set -euo pipefail\n" + HELPER + CONFIG + body,
                                text=True, encoding="utf-8", capture_output=True, env=ENV)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result

    def test_valid_config(self):
        self.run_script("ring_validate")

    def test_bad_configs(self):
        for change in ("TP=2", "NNODES=3", "USE_HOST_NCCL=0", "NCCL_NCHANNELS=0",
                       "WORKER_IP=$HEAD_IP", "HEAD_IP=CHANGE_ME_HEAD",
                       "NCCL_SWITCHLESS_RING_ONLY=typo", "NCCL_IB_SUBNET_PREFIX_LEN=33"):
            with self.subTest(change=change):
                self.run_script(change + "\nring_validate", success=False)

    def test_two_roce_ports(self):
        self.run_script(GID + "ring_check_gid /mock")

    def test_port_rejections(self):
        for hcas in ("hca0", "hca0,hca0", "hca0,hca0:1", "hca0,^hca1", ""):
            with self.subTest(hcas=hcas):
                self.run_script(GID + f"RING_HCAS='{hcas}'\nring_check_gid /mock", False)

    def test_bad_second_port(self):
        self.run_script(GID.replace("case \"$1\" in", "case \"$1\" in\n    */hca1/*) return 1;;") +
                        "ring_check_gid /mock", False)

    def test_reject_v1_zero_gid_and_inactive(self):
        for original, replacement in (("RoCE v2", "RoCE v1"), ("4: ACTIVE", "2: INIT"),
                                      ("ffff:0a00:0001", "ffff:0000:0000")):
            self.run_script(GID.replace(original, replacement) + "ring_check_gid /mock", False)

    def test_all_ranks_and_mount_paths(self):
        r = self.run_script(CLUSTER + 'ring_preflight_all 0\n[[ ${#RING_MOUNT_PATHS[@]} == 4 ]]')
        self.assertEqual(r.stderr.count("identity OK"), 4)

    def test_remote_failure(self):
        self.run_script(CLUSTER + 'worker_ssh_n() { return 42; }\nring_preflight_all 0', False)

    def test_remote_mismatch(self):
        self.run_script(CLUSTER + 'worker_ssh_n() { echo different-image; }\nring_preflight_all 0', False)

    def test_probe_success(self):
        self.run_script(PROBE + 'ring_probe_node')

    def test_probe_failures(self):
        for change in ('grep() { return 1; }', 'docker() { return 1; }',
                       'nvidia-smi() { echo 1234; }', 'nvidia-smi() { return 1; }',
                       'NCCL_SO_NAME=missing-switchless-test-file'):
            with self.subTest(change=change):
                self.run_script(PROBE + change + '\nring_probe_node', False)

    def test_env_and_stock_noop(self):
        r = self.run_script('args=(); ring_env_args args; printf "%s\\n" "${args[@]}"')
        self.assertIn("NCCL_ALGO=Ring", r.stdout)
        self.assertIn("NCCL_MIN_NCHANNELS=4", r.stdout)
        self.assertNotIn("LD_PRELOAD", r.stdout)
        self.run_script('NCCL_SWITCHLESS_RING_ONLY=0; args=(); ring_env_args args; [[ ${#args[@]} == 0 ]]')

    def test_launcher_syntax(self):
        for path in (ROOT / "start-tp4.sh", ROOT / "files/switchless-ring.sh", ROOT / ".env.tp4.ring.example"):
            r = subprocess.run([BASH, "-n"], input=path.read_text(encoding="utf-8"), text=True,
                               encoding="utf-8", capture_output=True, env=ENV)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_preflight_before_destructive_operation(self):
        src = (ROOT / "start-tp4.sh").read_text(encoding="utf-8")
        launch = src.split("launch_cluster() {", 1)[1]
        self.assertLess(launch.index("ring_preflight_all 1"), launch.index("docker rm -f"))

if __name__ == "__main__":
    unittest.main()
