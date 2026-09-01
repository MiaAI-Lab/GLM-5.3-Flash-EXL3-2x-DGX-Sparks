#!/usr/bin/env python3
"""Regression test for caller overrides that must win over ``.env``."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_max_num_seqs_inline_override_wins() -> None:
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(
            preamble
            + '\nprintf "MAX_NUM_SEQS=%s\\n" "${MAX_NUM_SEQS:-unset}"\n'
        )
        script.chmod(0o755)
        (tmp / ".env").write_text("MAX_NUM_SEQS=2\n")

        env = os.environ.copy()
        env["MAX_NUM_SEQS"] = "4"
        result = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.stdout.strip() == "MAX_NUM_SEQS=4"


def _run_preamble(env_file: str, caller: dict[str, str], probe: str) -> str:
    source = (ROOT / "start.sh").read_text()
    marker = "# ----------------------------- configuration -------------------------------"
    preamble, separator, _rest = source.partition(marker)
    assert separator, "start.sh configuration marker is missing"

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(preamble + probe)
        script.chmod(0o755)
        (tmp / ".env").write_text(env_file)

        env = {k: v for k, v in os.environ.items()
               if k not in ("GLM53_INDEXER_WORKSPACE", "GLM53_SPINWAIT_MS")}
        env.update(caller)
        result = subprocess.run(
            ["bash", str(script)], check=True, capture_output=True, text=True, env=env
        )
    return result.stdout.strip()


def test_indexer_workspace_caller_capture_is_setness_aware() -> None:
    """An explicitly EMPTY caller value must not be swallowed by ``.env``.

    ``GLM53_INDEXER_WORKSPACE=`` is an operator error; the enum guard has to see
    it. A ``[ -n "$_cli_..." ]`` restore would silently hand back the ``.env``
    value instead, so the capture uses the ``${VAR+1}`` setness probe.
    """
    probe = '\nprintf "V=[%s]\\n" "${GLM53_INDEXER_WORKSPACE-UNSET}"\n'
    env_file = "GLM53_INDEXER_WORKSPACE=rightsize\n"

    # Caller silent: .env wins.
    assert _run_preamble(env_file, {}, probe) == "V=[rightsize]"
    # Caller sets a real value: caller wins (the pre-existing contract).
    assert _run_preamble(
        env_file, {"GLM53_INDEXER_WORKSPACE": "stock"}, probe
    ) == "V=[stock]"
    # Caller sets it EMPTY: the empty value survives to the guard.
    assert _run_preamble(
        env_file, {"GLM53_INDEXER_WORKSPACE": ""}, probe
    ) == "V=[]"
    # ... and with no .env value either.
    assert _run_preamble("", {"GLM53_INDEXER_WORKSPACE": ""}, probe) == "V=[]"
    # Unset on both sides stays unset until the configuration default.
    assert _run_preamble("", {}, probe) == "V=[UNSET]"


def test_spinwait_caller_capture_is_setness_aware() -> None:
    probe = '\nprintf "V=[%s]\\n" "${GLM53_SPINWAIT_MS-UNSET}"\n'
    env_file = "GLM53_SPINWAIT_MS=16\n"

    assert _run_preamble(env_file, {}, probe) == "V=[16]"
    assert _run_preamble(
        env_file, {"GLM53_SPINWAIT_MS": "stock"}, probe
    ) == "V=[stock]"
    assert _run_preamble(
        env_file, {"GLM53_SPINWAIT_MS": ""}, probe
    ) == "V=[]"
    assert _run_preamble("", {"GLM53_SPINWAIT_MS": ""}, probe) == "V=[]"
    assert _run_preamble("", {}, probe) == "V=[UNSET]"


if __name__ == "__main__":
    test_max_num_seqs_inline_override_wins()
    test_indexer_workspace_caller_capture_is_setness_aware()
    test_spinwait_caller_capture_is_setness_aware()
    print("start.sh caller override regression OK")
