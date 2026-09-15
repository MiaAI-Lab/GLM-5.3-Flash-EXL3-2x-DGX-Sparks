#!/usr/bin/env python3
"""Host regression: bearer auth reaches only the head, outside process argv."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def test_api_key_is_head_only_and_not_in_docker_argv() -> None:
    source = START.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^launch_cluster\(\) \{\n.*?^\}\n", source)
    assert match, "missing launch_cluster"
    launch = match.group(0)
    for key in ("fixture bearer 'with spaces' $literal", ""):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture"
            fixture.touch()
            docker = root / "docker"
            docker.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['CAPTURE'], 'a') as output:\n"
                "    output.write(json.dumps([sys.argv[1:], os.getenv('VLLM_API_KEY')]) + '\\n')\n"
            )
            docker.chmod(0o755)
            values = {
                name: "fixture"
                for name in re.findall(r"\$(?:\{)?([A-Z][A-Z0-9_]*)", launch)
            }
            for name in re.findall(r'\[ -f "\$([A-Z0-9_]+)" \]', launch):
                values[name] = str(fixture)
            for name in ("CACHE_ROOT", "TRITON_HOST_CACHE", "TILELANG_HOST_CACHE"):
                values[name] = str(root / "cache")
            values.update(USE_HOST_NCCL="0", VLLM_API_KEY=key)
            script = (
                "set -eo pipefail\n"
                "log() { :; }\n"
                "die() { printf '%s\\n' \"$*\" >&2; exit 1; }\n"
                "scp() { :; }\n"
                "worker_ssh() { printf '%s\\0' \"$@\" >> \"$WORKER_CAPTURE\"; }\n"
                + "\n".join(f"{name}={shlex.quote(value)}" for name, value in values.items())
                + "\n" + launch + "\nlaunch_cluster\n"
            )
            env = {name: value for name, value in os.environ.items() if name != "VLLM_API_KEY"}
            env.update(
                PATH=f"{tmp}:{os.environ['PATH']}",
                CAPTURE=str(root / "head.jsonl"),
                WORKER_CAPTURE=str(root / "worker.args"),
            )
            result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            calls = [json.loads(line) for line in (root / "head.jsonl").read_text().splitlines()]
            runs = [(args, bearer) for args, bearer in calls if args[0] == "run"]
            assert len(runs) == 1, runs
            args, bearer = runs[0]
            assert bearer == key
            assert any(args[index:index + 2] == ["-e", "VLLM_API_KEY"] for index in range(len(args)))
            worker = (root / "worker.args").read_text()
            assert "VLLM_API_KEY" not in worker, worker
            if key:
                assert all(key not in arg for argv, _ in calls for arg in argv)
                assert key not in worker
                assert key not in result.stdout + result.stderr


if __name__ == "__main__":
    test_api_key_is_head_only_and_not_in_docker_argv()
    print("api-key argv guard OK (stubbed launch, configured and empty auth)")
