#!/usr/bin/env python3
"""Regression tests for bearer auth in the streaming decode benchmark."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("bench_decode.py")
SPEC = importlib.util.spec_from_file_location("bench_decode", MODULE_PATH)
assert SPEC and SPEC.loader
bench_decode = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench_decode)


def _headers(**env: str) -> dict[str, str]:
    original = {name: os.environ.get(name) for name in ("API_KEY", "VLLM_API_KEY")}
    try:
        for name in original:
            os.environ.pop(name, None)
        os.environ.update(env)
        return bench_decode._auth_headers()
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_keyless_serve_has_no_auth_header() -> None:
    assert _headers() == {}


def test_vllm_api_key_is_sent_as_bearer_token() -> None:
    assert _headers(VLLM_API_KEY="secret") == {"Authorization": "Bearer secret"}


def test_post_applies_bearer_header() -> None:
    captured = []
    original_urlopen = bench_decode.urllib.request.urlopen
    original_api_key = os.environ.get("API_KEY")
    original_vllm_api_key = os.environ.get("VLLM_API_KEY")

    def fake_urlopen(request, timeout):
        captured.append(request)
        return object()

    try:
        os.environ.pop("API_KEY", None)
        os.environ["VLLM_API_KEY"] = "secret"
        bench_decode.urllib.request.urlopen = fake_urlopen
        bench_decode._post("/v1/chat/completions", {})
    finally:
        bench_decode.urllib.request.urlopen = original_urlopen
        if original_api_key is None:
            os.environ.pop("API_KEY", None)
        else:
            os.environ["API_KEY"] = original_api_key
        if original_vllm_api_key is None:
            os.environ.pop("VLLM_API_KEY", None)
        else:
            os.environ["VLLM_API_KEY"] = original_vllm_api_key

    assert len(captured) == 1
    assert captured[0].get_header("Authorization") == "Bearer secret"


def test_api_key_compatibility_alias_is_supported() -> None:
    assert _headers(API_KEY="secret") == {"Authorization": "Bearer secret"}


if __name__ == "__main__":
    test_keyless_serve_has_no_auth_header()
    test_vllm_api_key_is_sent_as_bearer_token()
    test_post_applies_bearer_header()
    test_api_key_compatibility_alias_is_supported()
    print("bench_decode bearer auth regression OK")
