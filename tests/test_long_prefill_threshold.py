#!/usr/bin/env python3
"""Host-only scheduler opt-in and both-rank argv regression."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from test_numeric_config import guard_source
from test_start_overrides import _run_preamble

ROOT = Path(__file__).resolve().parents[1]
KEY = "LONG_PREFILL_TOKEN_THRESHOLD"
FLAG = "--long-prefill-token-threshold"


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
