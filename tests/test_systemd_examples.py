#!/usr/bin/env python3
"""Keep the systemd example units as a worker-first docker-wait wrapper."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "systemd"
SCRIPTS = (
    "run-glm53-systemd.sh",
    "launch-glm53.sh",
    "coordinate-glm53-worker.sh",
    "wait-glm53-health.sh",
    "env.sh",
)


def test_scripts_parse() -> None:
    for name in SCRIPTS:
        path = EXAMPLES / name
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, f"{name}: {completed.stderr}"


def test_head_unit_owns_lifecycle() -> None:
    text = (EXAMPLES / "glm53-head.service").read_text(encoding="utf-8")
    assert "Wants=network-online.target" in text
    assert "After=network-online.target docker.service" in text
    assert "coordinate-glm53-worker.sh" in text
    assert "wait-glm53-health.sh" in text
    assert "TimeoutStartSec=35min" in text
    assert "Restart=on-failure" in text
    assert "docker rm -f" in text
    assert "--restart" not in text


def test_worker_unit_does_not_self_heal() -> None:
    text = (EXAMPLES / "glm53-worker.service").read_text(encoding="utf-8")
    assert "Restart=no" in text
    assert "run-glm53-systemd.sh 1" in text
    assert "TimeoutStartSec=35min" in text
    assert "./start.sh" not in text


def test_coordinator_requires_host_keys() -> None:
    text = (EXAMPLES / "coordinate-glm53-worker.sh").read_text(encoding="utf-8")
    assert "StrictHostKeyChecking=yes" in text
    assert "BatchMode=yes" in text
    assert "systemctl restart" in text
    assert "StrictHostKeyChecking=no" not in text


def test_readiness_is_http() -> None:
    text = (EXAMPLES / "wait-glm53-health.sh").read_text(encoding="utf-8")
    assert "/health" in text
    assert "/v1/models" in text
    assert "docker logs" not in text


def test_launcher_splits_ranks() -> None:
    text = (EXAMPLES / "launch-glm53.sh").read_text(encoding="utf-8")
    assert "./start.sh" in text
    assert "env -u TAIL" in text
    assert "docker container inspect" in text


def test_docs_keep_gpu_mem_default() -> None:
    text = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert "GPU_MEM_UTIL=0.87" in text
    assert "do not** raise the shipped default" in text or "Do not** raise" in text
    assert "#205" in text
    assert "#230" in text
    assert "SKIP_PULL=1" in text
    assert "SKIP_BUILD=1" in text
    assert "@sha256:" in text
    assert "LaunchAgent" not in text
    assert "com.joelclaw" not in text


if __name__ == "__main__":
    test_scripts_parse()
    test_head_unit_owns_lifecycle()
    test_worker_unit_does_not_self_heal()
    test_coordinator_requires_host_keys()
    test_readiness_is_http()
    test_launcher_splits_ranks()
    test_docs_keep_gpu_mem_default()
    print("systemd examples regression OK")
