#!/usr/bin/env python3
"""Regression anchors for the DEFAULT_MAX_NEW_TOKENS decode-hygiene wiring
(issue #43): omitted max_tokens otherwise allows unbounded decode (budget
~max_model_len - prompt) that can grow KV until it preempts other sessions."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_default_is_bounded() -> None:
    src = (ROOT / "start.sh").read_text()
    assert 'DEFAULT_MAX_NEW_TOKENS="${DEFAULT_MAX_NEW_TOKENS:-65536}"' in src


def test_flag_passed_on_both_ranks() -> None:
    src = (ROOT / "start.sh").read_text()
    hits = src.count(r'ARGS+=(--override-generation-config "{\"max_new_tokens\": ${DEFAULT_MAX_NEW_TOKENS}}")')
    assert hits == 2, f"expected override flag in head AND worker inner scripts, found {hits}"


def test_head_env_block_passes_var() -> None:
    src = (ROOT / "start.sh").read_text()
    assert '-e DEFAULT_MAX_NEW_TOKENS="$DEFAULT_MAX_NEW_TOKENS"' in src, (
        "head docker run block must pass DEFAULT_MAX_NEW_TOKENS explicitly"
    )


def test_env_passthrough_and_docs() -> None:
    assert "DEFAULT_MAX_NEW_TOKENS MODEL_DIR EXTRA_ARGS; do" in (ROOT / "start.sh").read_text()
    assert "DEFAULT_MAX_NEW_TOKENS=65536" in (ROOT / ".env.example").read_text()
    assert "`DEFAULT_MAX_NEW_TOKENS`" in (ROOT / "README.md").read_text()


if __name__ == "__main__":
    test_default_is_bounded()
    test_flag_passed_on_both_ranks()
    test_env_passthrough_and_docs()
    test_head_env_block_passes_var()
    print("default-max-new-tokens wiring OK")
