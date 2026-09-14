#!/usr/bin/env python3
"""Behavioral regressions for DEFAULT_MAX_NEW_TOKENS (issue #43 decode hygiene).

Contract under test: the value is a server-side default for requests that
OMIT max_tokens/max_completion_tokens — never a cap on explicit client
limits — with empty = opt-out and malformed values rejected by
validate_numeric_config before restart stops the healthy pair.

The serving-side checks exec the verbatim ``get_max_tokens`` from the
pinned runtime (vLLM install layer
sha256:2c55b4653d4b2c7d4497169b14edc16f44b3fc3058a9ab9cd302e365783e7cbb,
file vllm/entrypoints/serve/utils/api_utils.py) after applying
overlay/patch_default_max_new_tokens.py — the same source the container
patches at boot. The platform cap is stubbed; the completion-call harness
uses real Pydantic v2 normalization and field tracking, without GPU imports.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"

sys.path.insert(0, str(ROOT / "overlay"))
from patch_default_max_new_tokens import (
    OLD, LIMITS_PATH, COMPLETION_PATH, PROTOCOL_PATH,
    apply_text, apply_completion_text, apply_protocol_text, main as patch_main,
)

# The pinned vLLM fixture is licensed under Apache-2.0.
# Copyright contributors to the vLLM project.

# Verbatim copy of get_max_tokens from the pinned image layer (see module
# docstring for provenance). The patch's OLD anchor must be a substring.
PINNED_GET_MAX_TOKENS = '''def get_max_tokens(
    max_model_len: int,
    max_tokens: int | None,
    input_length: int,
    default_sampling_params: dict,
    override_max_tokens: int | None = None,
    truncate_prompt_tokens: int | None = None,
) -> int:
    if truncate_prompt_tokens is not None:
        limit = truncate_prompt_tokens
        input_length = min(
            input_length,
            max_model_len if limit == -1 else limit,
        )
    if max_model_len < input_length:
        raise ValueError(
            f"Input length ({input_length}) exceeds model's maximum "
            f"context length ({max_model_len})."
        )
    model_max_tokens = max_model_len - input_length
    platform_max_tokens = current_platform.get_max_output_tokens(input_length)
    fallback_max_tokens = (
        max_tokens
        if max_tokens is not None
        else default_sampling_params.get("max_tokens")
    )

    return min(
        val
        for val in (
            model_max_tokens,
            fallback_max_tokens,
            override_max_tokens,
            platform_max_tokens,
        )
        if val is not None
    )
'''


class _NoPlatformCap:
    """current_platform stub: CUDA returns None (no platform output cap)."""

    @staticmethod
    def get_max_output_tokens(input_length: int) -> None:
        return None


def _load_patched_get_max_tokens():
    """Apply the overlay to the pinned source and exec the result."""
    assert OLD in PINNED_GET_MAX_TOKENS, "patch anchor drifted from pinned source"
    patched, status = apply_text(PINNED_GET_MAX_TOKENS)
    assert status == "applied", status
    ns = {"os": os, "current_platform": _NoPlatformCap}
    exec(patched, ns)
    return ns["get_max_tokens"]


def _run_bash(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _clean_env(*drop: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["LC_ALL"] = "C"
    return env


# ---------------------------------------------------------------------------
# Serving semantics: omitted vs explicit, against the patched pinned limiter
# ---------------------------------------------------------------------------

def test_omitted_request_gets_default() -> None:
    get_max_tokens = _load_patched_get_max_tokens()
    with patch.dict(os.environ, {"DEFAULT_MAX_NEW_TOKENS": "65536"}):
        assert get_max_tokens(1_000_000, None, 1000, {"max_tokens": 32}) == 65536


def test_explicit_request_beats_default() -> None:
    get_max_tokens = _load_patched_get_max_tokens()
    with patch.dict(os.environ, {"DEFAULT_MAX_NEW_TOKENS": "65536"}):
        assert get_max_tokens(1_000_000, 200_000, 1000, {"max_tokens": 32}) == 200_000
        assert get_max_tokens(1_000_000, 128, 1000, {"max_tokens": 32}) == 128


def test_remaining_context_bound_still_applies() -> None:
    get_max_tokens = _load_patched_get_max_tokens()
    with patch.dict(os.environ, {"DEFAULT_MAX_NEW_TOKENS": "65536"}):
        assert get_max_tokens(100_000, None, 90_000, {}) == 10_000
        assert get_max_tokens(100_000, 500_000, 90_000, {}) == 10_000
        assert get_max_tokens(32768, None, 1000, {}) == 31768


def test_independent_server_caps_are_preserved() -> None:
    get_max_tokens = _load_patched_get_max_tokens()
    with patch.dict(os.environ, {"DEFAULT_MAX_NEW_TOKENS": "65536"}):
        assert get_max_tokens(1_000_000, 200_000, 1000, {}, 4096) == 4096
        assert get_max_tokens(1_000_000, None, 1000, {}, 4096) == 4096


def test_env_unset_keeps_stock_hard_cap() -> None:
    get_max_tokens = _load_patched_get_max_tokens()
    with patch.dict(os.environ):
        os.environ.pop("DEFAULT_MAX_NEW_TOKENS", None)
        assert get_max_tokens(1_000_000, 200_000, 1000, {}, 65536) == 65536
        assert get_max_tokens(1_000_000, None, 1000, {"max_tokens": 32}) == 32


# Pinned completion call and before-validator, without GPU-serving imports.
# Pydantic must observe omitted input before the validator normalizes null.
COMPLETION_FIXTURE = '''import io
class Serving:
    def limit(self, request, max_model_len, engine_inputs):
        for engine_input in engine_inputs:
            max_tokens = get_max_tokens(
                max_model_len,
                request.max_tokens,
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )
        return max_tokens
'''

PROTOCOL_FIXTURE = '''from pydantic import BaseModel, model_validator
class CompletionRequest(BaseModel):
    max_tokens: int | None = 16
    truncate_prompt_tokens: int | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_null_max_tokens(cls, data):
        if isinstance(data, dict) and data.get("max_tokens") is None:
            data = data.copy()
            data["max_tokens"] = cls.model_fields["max_tokens"].default
        return data
'''


def test_completion_distinguishes_omission_from_explicit_sixteen() -> None:
    source, status = apply_completion_text(COMPLETION_FIXTURE)
    assert status == "applied"
    protocol, status = apply_protocol_text(PROTOCOL_FIXTURE)
    assert status == "applied"
    namespace = {"get_max_tokens": _load_patched_get_max_tokens()}
    exec(protocol, namespace)
    exec(source, namespace)
    request_type = namespace["CompletionRequest"]
    serving = namespace["Serving"]()
    serving.default_sampling_params = {}
    serving.override_max_tokens = None
    serving._extract_prompt_len = lambda length: length
    with patch.dict(os.environ, {"DEFAULT_MAX_NEW_TOKENS": "65536"}):
        for data, expected in (
            ({}, 65536),
            ({"max_tokens": 16}, 16),
            ({"max_tokens": 200000}, 200000),
            ({"max_tokens": None}, 16),
        ):
            request = request_type.model_validate(data)
            assert serving.limit(request, 1000000, [1000]) == expected
        os.environ["DEFAULT_MAX_NEW_TOKENS"] = ""
        assert serving.limit(request_type.model_validate({}), 1000000, [1000]) == 16


def test_patch_apply_skip_drift() -> None:
    out, status = apply_text(PINNED_GET_MAX_TOKENS)
    assert status == "applied"
    out2, status2 = apply_text(out)
    assert status2 == "skipped" and out2 == out
    _, status3 = apply_text("def get_max_tokens():\n    return 0\n")
    assert status3.startswith("missing:"), status3
    drifted = out.replace("fallback_max_tokens = int(configured_default)", "fallback_max_tokens = 1")
    unchanged, rejected = apply_text(drifted)
    assert rejected.startswith("drifted:") and unchanged == drifted


def test_all_targets_are_checked_before_writing() -> None:
    with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"DEFAULT_MAX_NEW_TOKENS": "65536"}):
        root = Path(td)
        limits, completion, protocol = (
            root / LIMITS_PATH, root / COMPLETION_PATH, root / PROTOCOL_PATH,
        )
        limits.parent.mkdir(parents=True)
        completion.parent.mkdir(parents=True)
        limits.write_text("import os\n" + PINNED_GET_MAX_TOKENS)
        completion.write_text(COMPLETION_FIXTURE)
        protocol.write_text(PROTOCOL_FIXTURE.replace('data.get("max_tokens")', 'data.get("max_tokens", 16)'))
        targets = (limits, completion, protocol)
        original = tuple(target.read_bytes() for target in targets)
        assert patch_main(["patch", td]) == 1
        assert tuple(target.read_bytes() for target in targets) == original
        protocol.write_text(PROTOCOL_FIXTURE)
        assert patch_main(["patch", td]) == 0
        first = tuple(target.read_bytes() for target in targets)
        assert patch_main(["patch", td]) == 0
        assert tuple(target.read_bytes() for target in targets) == first


# ---------------------------------------------------------------------------
# Launcher semantics: unset-only default, empty opt-out, caller over .env
# ---------------------------------------------------------------------------

def _env_default_script() -> str:
    """Real .env loader + the real DEFAULT_MAX_NEW_TOKENS assignment line."""
    src = START.read_text()
    begin = src.index("_caller_overrides=()")
    end = src.index("# ----------------------------- configuration")
    loader = src[begin:end]
    assign = next(
        line for line in src.splitlines()
        if line.startswith('DEFAULT_MAX_NEW_TOKENS="${DEFAULT_MAX_NEW_TOKENS')
    )
    return (
        loader
        + assign
        + '\nprintf "%s\\n" "${DEFAULT_MAX_NEW_TOKENS-empty}"\n'
    )


def _run_env_default(caller: str | None, dotenv: str | None) -> str:
    with tempfile.TemporaryDirectory() as td:
        env_lines = "" if dotenv is None else f"DEFAULT_MAX_NEW_TOKENS={dotenv}\n"
        (Path(td) / ".env").write_text(env_lines)
        env = _clean_env("DEFAULT_MAX_NEW_TOKENS")
        env["SCRIPT_DIR"] = td
        if caller is not None:
            env["DEFAULT_MAX_NEW_TOKENS"] = caller
        result = _run_bash(_env_default_script(), env)
        assert result.returncode == 0, (result.returncode, result.stderr)
        return result.stdout.strip()


def test_unset_only_default_expansion() -> None:
    # Unset + no .env entry -> built-in default.
    assert _run_env_default(None, None) == "65536"
    # Unset + .env entry -> .env wins.
    assert _run_env_default(None, "12345") == "12345"


def test_explicit_empty_opts_out() -> None:
    # Caller empty beats .env and the built-in default: opt-out preserved.
    assert _run_env_default("", "12345") == ""
    assert _run_env_default("", None) == ""
    # .env empty (no caller export) also opts out.
    assert _run_env_default(None, "") == ""


def test_caller_value_beats_dotenv() -> None:
    assert _run_env_default("777", "12345") == "777"


def test_rank_arguments_do_not_create_a_hard_cap() -> None:
    from test_launcher_rank_parity import Harness, rank_runs

    with tempfile.TemporaryDirectory() as td:
        harness = Harness(Path(td))
        ranks = rank_runs(harness, DEFAULT_MAX_NEW_TOKENS="65536")
        assert ranks is not None
        result = harness.run("write_inner_scripts", entry="start.fn.sh")
        assert result.returncode == 0, result.stderr
        for rank, label in zip(ranks[:2], ("head", "worker")):
            script = (harness.repo / f".glm53-exl3-{label}.inner.sh").read_text()
            begin = script.index("ARGS=(")
            end = script.index('\n[ -f "${MODEL_DIR}/config.json"', begin)
            env = _clean_env()
            env.update(rank.env)
            env.update(SPEC_METHOD="none", EXTRA_ARGS="--generation-config independent-config")
            result = _run_bash(
                "say() { :; }\n" + script[begin:end] + '\nprintf "%s\\0" "${ARGS[@]}"\n',
                env,
            )
            assert result.returncode == 0, result.stderr
            args = result.stdout.split("\0")
            assert "--override-generation-config" not in args
            assert args[args.index("--generation-config") + 1] == "independent-config"
            with patch.dict(os.environ, {"DEFAULT_MAX_NEW_TOKENS": rank.env["DEFAULT_MAX_NEW_TOKENS"]}):
                assert _load_patched_get_max_tokens()(1000000, None, 1000, {}) == 65536


def test_validation_runs_before_restart_stop() -> None:
    from test_launcher_rank_parity import Harness

    with tempfile.TemporaryDirectory() as td:
        harness = Harness(Path(td))
        result = harness.run("restart", DEFAULT_MAX_NEW_TOKENS="bad")
        assert result.returncode == 2, result.stderr
        assert harness.calls() == []


# ---------------------------------------------------------------------------
# Validation: malformed values rejected before lifecycle side effects
# ---------------------------------------------------------------------------

def _guard_source() -> str:
    source = START.read_text()
    begin = source.index("# GLM53 numeric config guard (begin)")
    end_marker = "# GLM53 numeric config guard (end)"
    end = source.index(end_marker, begin) + len(end_marker)
    return source[begin:end]


def _validate(value: str | None, max_model_len: int = 1000000) -> subprocess.CompletedProcess[str]:
    script = (
        _guard_source()
        + '\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN="$TEST_MODEL_LEN"; MAX_NUM_SEQS=4; '
        + 'MAX_NUM_BATCHED_TOKENS=1024; GLM53_INDEXER_WORKSPACE=stock; '
        + 'GLM53_SPINWAIT_MS=stock\n'
        + 'validate_numeric_config || exit $?\n'
        + 'printf "%s\\n" "${DEFAULT_MAX_NEW_TOKENS-empty}"\n'
    )
    env = _clean_env("DEFAULT_MAX_NEW_TOKENS")
    env["TEST_MODEL_LEN"] = str(max_model_len)
    if value is not None:
        env["DEFAULT_MAX_NEW_TOKENS"] = value
    return _run_bash(script, env)


def test_validation_accepts_unset_empty_and_positive() -> None:
    for value, expected in ((None, "empty"), ("", ""), ("65536", "65536"), ("001", "1")):
        result = _validate(value)
        assert result.returncode == 0, (value, result.stderr)
        assert result.stdout.strip() == expected, (value, result.stdout)
    assert _validate("65536", max_model_len=32768).returncode == 0


def test_validation_rejects_malformed() -> None:
    for bad in ("nope", "-1", "0", "1.5", " 64", "64 ", "1000001", "64\n"):
        result = _validate(bad)
        assert result.returncode == 2, (bad, result.returncode, result.stdout)


if __name__ == "__main__":
    test_omitted_request_gets_default()
    test_explicit_request_beats_default()
    test_remaining_context_bound_still_applies()
    test_env_unset_keeps_stock_hard_cap()
    test_patch_apply_skip_drift()
    test_unset_only_default_expansion()
    test_explicit_empty_opts_out()
    test_caller_value_beats_dotenv()
    test_independent_server_caps_are_preserved()
    test_completion_distinguishes_omission_from_explicit_sixteen()
    test_all_targets_are_checked_before_writing()
    test_rank_arguments_do_not_create_a_hard_cap()
    test_validation_accepts_unset_empty_and_positive()
    test_validation_rejects_malformed()
    test_validation_runs_before_restart_stop()
    print("default-max-new-tokens behavioral regressions: PASS")
