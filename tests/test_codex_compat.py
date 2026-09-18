#!/usr/bin/env python3
"""Verify patch_codex_compat.py markers exist in the served vLLM tree."""
import ast
import os
import sys

VLLM = os.environ.get(
    "VLLM", "/usr/local/lib/python3.12/dist-packages/vllm"
)

CASES = [
    ("api_server.py middleware", VLLM + "/entrypoints/openai/api_server.py", "_CodexCompatMiddleware"),
    ("resp_utils fallback marker", VLLM + "/entrypoints/openai/responses/utils.py", "__codex_compat_patched"),
    ("resp_protocol widened input", VLLM + "/entrypoints/openai/responses/protocol.py", "__codex_compat_patched"),
]


def main() -> int:
    failures = 0
    for label, path, marker in CASES:
        if not os.path.exists(path):
            print(f"FAIL {label}: {path} missing")
            failures += 1
            continue
        src = open(path, encoding="utf-8").read()
        ast.parse(src)  # syntax gate
        if marker in src:
            print(f"OK   {label}: marker present, ast clean")
        else:
            print(f"FAIL {label}: marker '{marker}' missing in {path}")
            failures += 1
    if failures:
        print(f"TEST_CODEX_COMPAT_FAIL ({failures})")
        return 1
    print("TEST_CODEX_COMPAT_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
