#!/usr/bin/env python3
"""Budget-aware prefill defaults, explicit opt-out, and both-rank argv."""

import os
import subprocess
import tempfile
from pathlib import Path

try:
    import pytest
except ImportError:  # host smoke runs without pytest
    class pytest:  # type: ignore[no-redef]
        class mark:
            @staticmethod
            def parametrize(*_args, **_kwargs):
                return lambda fn: fn

from test_numeric_config import guard_source
from test_start_overrides import _run_preamble

ROOT = Path(__file__).resolve().parents[1]
KEY = "LONG_PREFILL_TOKEN_THRESHOLD"
FLAG = "--long-prefill-token-threshold"


def _run_through_default(
    env_file: str, caller: dict[str, str], launcher: str = "start.sh"
) -> str:
    path = ROOT / launcher
    source = path.read_text()
    # Execute configuration and the real guard without any service lifecycle.
    probe = source[:source.index("CHAT_TEMPLATE_HOST=")]
    probe += "\n" + guard_source(path)
    probe += '\nGLM53_SPINWAIT_MS=stock\nvalidate_numeric_config || exit $?\n'
    probe += 'printf "V=[%s]\\n" "${LONG_PREFILL_TOKEN_THRESHOLD-UNSET}"\n'
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        script = tmp / "start.sh"
        script.write_text(probe)
        script.chmod(0o755)
        (tmp / ".env").write_text(env_file)
        if launcher == "start-tp4.sh":
            (tmp / ".env.tp4").write_text("")
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "USER": "glm53"}
        env.update(caller)
        result = subprocess.run(
            ["bash", str(script)], check=True, capture_output=True, text=True, env=env
        )
    return result.stdout.strip()


def test_unset_defaults_to_3584() -> None:
    assert _run_through_default("", {}) == "V=[3584]"
    assert _run_through_default("MAX_NUM_SEQS=4\n", {}) == "V=[3584]"


def test_explicit_empty_keeps_stock_scheduler() -> None:
    assert _run_through_default(f"{KEY}=3584\n", {KEY: ""}) == "V=[]"
    assert _run_through_default(f"{KEY}=\n", {}) == "V=[]"
    assert _run_through_default(f"{KEY}=1024\n", {KEY: "3584"}) == "V=[3584]"

@pytest.mark.parametrize("launcher", ["start.sh", "start-tp4.sh"])
def test_budget_aware_default_and_overrides(launcher: str) -> None:
    for budget, threshold in (("7168", "3584"), ("02048", "1024"),
                              ("6144", "3072"), ("8192", "3584"), ("1", "1")):
        assert _run_through_default(
            "", {"MAX_NUM_BATCHED_TOKENS": budget}, launcher
        ) == f"V=[{threshold}]"
    assert _run_through_default(
        "MAX_NUM_BATCHED_TOKENS=7168\n",
        {"MAX_NUM_BATCHED_TOKENS": "2048"}, launcher
    ) == "V=[1024]"
    assert _run_through_default(
        f"{KEY}=3584\n",
        {"MAX_NUM_BATCHED_TOKENS": "2048", KEY: ""}, launcher
    ) == "V=[]"
    assert _run_through_default(
        "MAX_NUM_BATCHED_TOKENS=2048\n",
        {KEY: "001536"}, launcher
    ) == "V=[1536]"
    for caller in (
        {"MAX_NUM_BATCHED_TOKENS": "2048", KEY: "3584"},
        {"MAX_NUM_BATCHED_TOKENS": "0"},
        {"MAX_NUM_BATCHED_TOKENS": "18446744073709551615"},
        {"MAX_NUM_BATCHED_TOKENS": "invalid"},
    ):
        try:
            _run_through_default("", caller, launcher)
        except subprocess.CalledProcessError as error:
            assert error.returncode == 2, error.stderr
        else:
            raise AssertionError(f"invalid configuration accepted: {caller}")


@pytest.mark.parametrize("value", [None, "", "3584"])
def test_both_rank_argv(value: str | None) -> None:
    source = (ROOT / "start.sh").read_text()
    begin = source.index("write_inner_scripts() {")
    end = source.index('\n}\n', begin) + 3
    env = {"PATH": os.environ["PATH"], "SERVED_MODEL_NAME": "test",
           "PORT": "8888", "TP": "2", "NNODES": "2", "HEAD_IP": "127.0.0.1",
           "MASTER_PORT": "29500", "SPEC_METHOD": "none"}
    if value is not None:
        env[KEY] = value
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        env.update(HEAD_SCRIPT=str(tmp / "head.sh"), WORKER_SCRIPT=str(tmp / "worker.sh"))
        subprocess.run(["bash", "-c", source[begin:end] + '\nwrite_inner_scripts'],
                       env=env, check=True, capture_output=True)
        for rank, name in enumerate(("head.sh", "worker.sh")):
            script = (tmp / name).read_text().split('[ -f "${MODEL_DIR}/config.json" ]')[0]
            script += '\nprintf "%s\\0" "${ARGS[@]}"\n'
            result = subprocess.run(["bash", "-c", script], env=env,
                                    check=True, capture_output=True)
            # Head status messages precede the NUL-delimited argv.
            args = result.stdout.decode().splitlines()[-1].split("\0")[:-1]
            assert args[args.index("--node-rank") + 1] == str(rank)
            assert args.count(FLAG) == (1 if value else 0)
            if value:
                assert args[args.index(FLAG) + 1] == value


@pytest.mark.parametrize("value,expected", [("", 0), ("3584", 0), ("003584", 0),
                                             ("0", 2), ("-1", 2), ("1.5", 2),
                                             (" 3584", 2), ("7169", 2)])
def test_validation(value: str, expected: int) -> None:
    script = (guard_source() + '\nGPU_MEM_UTIL=0.85; MAX_MODEL_LEN=850000; '
              'MAX_NUM_SEQS=4; MAX_NUM_BATCHED_TOKENS=7168; GLM53_SPINWAIT_MS=stock\n'
              'validate_numeric_config || exit $?\n'
              'printf "%s" "$LONG_PREFILL_TOKEN_THRESHOLD"\n')
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                            env={"PATH": os.environ["PATH"], KEY: value})
    assert result.returncode == expected, result.stderr
    if expected == 0:
        assert result.stdout == ("3584" if value else "")


def test_caller_can_restore_stock_over_env() -> None:
    probe = '\nprintf "V=[%s]\\n" "${LONG_PREFILL_TOKEN_THRESHOLD-UNSET}"\n'
    assert _run_preamble(f"{KEY}=3584\n", {KEY: ""}, probe) == "V=[]"
    assert _run_preamble(f"{KEY}=1024\n", {KEY: "3584"}, probe) == "V=[3584]"


if __name__ == "__main__":
    test_unset_defaults_to_3584()
    test_explicit_empty_keeps_stock_scheduler()
    for launcher in ("start.sh", "start-tp4.sh"):
        test_budget_aware_default_and_overrides(launcher)
    for value in (None, "", "3584"):
        test_both_rank_argv(value)
    for value, expected in (("", 0), ("3584", 0), ("003584", 0),
                            ("0", 2), ("-1", 2), ("1.5", 2),
                            (" 3584", 2), ("7169", 2)):
        test_validation(value, expected)
    test_caller_can_restore_stock_over_env()
    print("long prefill threshold tests: PASS")
