#!/usr/bin/env python3
"""Regression tests for caller overrides that must win over ``.env``."""

from __future__ import annotations

import re
import shlex
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

        # isolated: no BASH_ENV / PATH surprises from the developer shell
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "USER": "glm53", "MAX_NUM_SEQS": "4"}
        result = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    assert result.stdout.strip() == "MAX_NUM_SEQS=4"


def _run_preamble(env_file: str, caller: dict[str, str], probe: str) -> str:
    """Run start.sh's pre-configuration preamble with a synthetic .env."""
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

        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "USER": "glm53"}
        env.update(caller)
        result = subprocess.run(
            ["bash", str(script)], check=True, capture_output=True, text=True, env=env
        )
    return result.stdout.strip()


def test_default_reasoning_effort_caller_override_is_setness_aware() -> None:
    """An explicitly EMPTY caller value must beat .env, not be swallowed by it.

    The knob's own default is empty, so ``[ -n "$_cli_x" ]`` cannot tell
    ``GLM53_DEFAULT_REASONING_EFFORT= ./start.sh`` (deliberately back to the
    template default) apart from an unset var. Only ``${VAR+1}`` can.
    """
    probe = '\nprintf "EFFORT=[%s]\\n" "${GLM53_DEFAULT_REASONING_EFFORT-unset}"\n'
    env_file = "GLM53_DEFAULT_REASONING_EFFORT=high\n"

    # caller unset -> .env wins
    assert _run_preamble(env_file, {}, probe) == "EFFORT=[high]"

    # caller sets a value -> caller wins
    assert _run_preamble(
        env_file, {"GLM53_DEFAULT_REASONING_EFFORT": "low"}, probe
    ) == "EFFORT=[low]"

    # caller sets it EMPTY -> caller still wins (the setness-aware case)
    assert _run_preamble(
        env_file, {"GLM53_DEFAULT_REASONING_EFFORT": ""}, probe
    ) == "EFFORT=[]"


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


def test_every_env_example_key_preserves_caller_setness() -> None:
    keys = re.findall(
        r"^(?:# )?([A-Za-z_][A-Za-z0-9_]*)=", (ROOT / ".env.example").read_text(), re.M
    )
    keys.append("FUTURE_LAUNCHER_KNOB")
    dotenv = "".join(f"{key}=dotenv\n" for key in keys)
    child_probe = 'printf "[%s]\\n" ' + " ".join(
        f'"${{{key}-UNSET}}"' for key in keys
    )
    probe = '\n"$BASH" -c ' + shlex.quote(child_probe) + "\n"
    for value in ("caller", ""):
        caller = dict.fromkeys(keys, value)
        assert _run_preamble(dotenv, caller, probe) == "\n".join(
            f"[{value}]" for key in keys
        )
    assert _run_preamble(dotenv, {}, probe) == "\n".join(
        "[dotenv]" for key in keys
    )


def test_shell_assignments_preserve_caller_values() -> None:
    dotenv = (
        "export FUTURE_A=dotenv; FUTURE_B=dotenv\n"
        "declare -x FUTURE_C=dotenv\n"
        "unset FUTURE_D\n"
        "FUTURE_E+=suffix\n"
        "SCRIPT_DIR=/dotenv-internal\n"
    )
    caller = {
        "FUTURE_A": 'quoted \"value\" = with spaces',
        "FUTURE_B": "first line\nsecond line",
        "FUTURE_C": "",
        "FUTURE_D": "caller",
        "FUTURE_E": "prefix",
        "SHELLOPTS": "braceexpand:hashall",
    }
    probe = (
        '\nprintf "[%s]\\n" "$FUTURE_A" "$FUTURE_B" "$FUTURE_C" '
        '"$FUTURE_D" "$FUTURE_E" "$SCRIPT_DIR"\n'
    )
    assert _run_preamble(dotenv, caller, probe) == (
        '[quoted "value" = with spaces]\n[first line\nsecond line]\n'
        "[]\n[caller]\n[prefix]\n[/dotenv-internal]"
    )


if __name__ == "__main__":
    test_every_env_example_key_preserves_caller_setness()
    test_shell_assignments_preserve_caller_values()
    test_max_num_seqs_inline_override_wins()
    test_default_reasoning_effort_caller_override_is_setness_aware()
    test_indexer_workspace_caller_capture_is_setness_aware()
    test_spinwait_caller_capture_is_setness_aware()
    print("start.sh caller override regression OK")
