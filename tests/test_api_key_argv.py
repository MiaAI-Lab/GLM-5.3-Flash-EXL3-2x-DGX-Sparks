#!/usr/bin/env python3
"""Regression guard: VLLM_API_KEY stays out of argv and off the worker.

Merged PR #30 promised the bearer token never lands in argv and is passed to
the head container only (the worker runs --headless and serves no API). This
locks in both properties.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def function(name: str) -> str:
    text = START.read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}\n", text)
    assert match, f"missing function {name}"
    return match.group(0)


def test_api_key_is_head_only_and_not_in_docker_argv() -> None:
    launch = function("launch_cluster")
    # worker: no credential in the remote docker command at all
    assert "serve_env+=\" -e VLLM_API_KEY=" not in launch
    # head: value must come from the docker client's environment, not argv
    assert '-e VLLM_API_KEY="$VLLM_API_KEY"' not in launch
    assert 'VLLM_API_KEY="$VLLM_API_KEY" docker run' in launch
    assert "-e VLLM_API_KEY \\" in launch


if __name__ == "__main__":
    test_api_key_is_head_only_and_not_in_docker_argv()
    print("api-key argv guard OK")
