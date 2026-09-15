#!/usr/bin/env python3
"""CPU-only tests for launcher numeric type/range validation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
START_TP4 = ROOT / "start-tp4.sh"


def guard_source(path: Path = START) -> str:
    source = path.read_text()
    begin = source.index("# GLM53 numeric config guard (begin)")
    end_marker = "# GLM53 numeric config guard (end)"
    end = source.index(end_marker, begin) + len(end_marker)
    return source[begin:end]


def validate(util: str, model: str, seqs: str, batch: str) -> subprocess.CompletedProcess[str]:
    script = (
        guard_source()
        + '\nGPU_MEM_UTIL="$1"; MAX_MODEL_LEN="$2"; MAX_NUM_SEQS="$3"; '
        + 'MAX_NUM_BATCHED_TOKENS="$4"; GLM53_SPINWAIT_MS=stock\n'
        + 'validate_numeric_config || exit $?\n'
        + 'printf "%s|%s|%s|%s\\n" "$GPU_MEM_UTIL" "$MAX_MODEL_LEN" '
        + '"$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS"\n'
    )
    return subprocess.run(
        ["bash", "-c", script, "test", util, model, seqs, batch],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )


def expect_rc(values: tuple[str, str, str, str], expected: int) -> None:
    result = validate(*values)
    assert result.returncode == expected, (values, result.returncode, result.stdout, result.stderr)


def test_matrix() -> None:
    expect_rc(("0.87", "1000000", "4", "1024"), 0)
    expect_rc((".87", "01000000", "0004", "01024"), 0)
    expect_rc(("1.0", "1000000", "4096", "8388608"), 0)
    expect_rc(("0", "1000000", "4", "1024"), 2)
    expect_rc(("8.7", "1000000", "4", "1024"), 2)
    expect_rc(("nope", "1000000", "4", "1024"), 2)
    expect_rc(("0.87", "0", "4", "1024"), 2)
    expect_rc(("0.87", "1000001", "4", "1024"), 2)
    expect_rc(("0.87", "1000000", "O4", "1024"), 2)
    expect_rc(("0.87", "1000000", "4097", "1024"), 2)
    expect_rc(("0.87", "1000000", "4", "1024\r"), 2)
    expect_rc(("0.87", "1000000", "4", "18446744073709551615"), 2)


def test_decimal_normalization() -> None:
    result = validate(".87", "01000000", "0004", "01024")
    assert result.returncode == 0
    assert result.stdout.strip() == ".87|1000000|4|1024"


def validate_enum(value: str | None) -> subprocess.CompletedProcess[str]:
    """Run validate_numeric_config with only GLM53_INDEXER_WORKSPACE varying."""
    script = (
        guard_source()
        + '\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4; '
        + 'MAX_NUM_BATCHED_TOKENS=1024; GLM53_SPINWAIT_MS=stock\n'
        + 'validate_numeric_config || exit $?\n'
        + 'printf "%s\\n" "${GLM53_INDEXER_WORKSPACE-unset}"\n'
    )
    env = {k: v for k, v in os.environ.items() if k != "GLM53_INDEXER_WORKSPACE"}
    env["LC_ALL"] = "C"
    if value is not None:
        env["GLM53_INDEXER_WORKSPACE"] = value
    return subprocess.run(
        ["bash", "-c", script], text=True, capture_output=True, check=False, env=env
    )


def test_indexer_workspace_enum() -> None:
    """Strict enum: default on UNSET only, then a literal match.

    ``overlay/patch_indexer_workspace.py``'s ``_glm53_workspace_mode`` applies
    the same rule inside the container, so an empty or case-variant value must
    fail here rather than change meaning across the boundary.
    """
    for good in (None, "stock", "rightsize"):
        result = validate_enum(good)
        assert result.returncode == 0, (good, result.stderr)
    for bad in ("", " ", "Stock", "RIGHTSIZE", " rightsize ", "1", "on", "true",
                "rightsize\n"):
        result = validate_enum(bad)
        assert result.returncode == 2, (bad, result.returncode, result.stdout)
        assert "GLM53_INDEXER_WORKSPACE must be one of" in result.stderr, bad


def validate_spinwait(value: str | None) -> subprocess.CompletedProcess[str]:
    script = (
        guard_source()
        + '\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4; '
        + 'MAX_NUM_BATCHED_TOKENS=1024; GLM53_INDEXER_WORKSPACE=stock\n'
        + 'validate_numeric_config || exit $?\n'
        + 'printf "%s\\n" "${GLM53_SPINWAIT_MS-unset}"\n'
    )
    env = {k: v for k, v in os.environ.items() if k != "GLM53_SPINWAIT_MS"}
    env["LC_ALL"] = "C"
    if value is not None:
        env["GLM53_SPINWAIT_MS"] = value
    else:
        env["GLM53_SPINWAIT_MS"] = "stock"
    return subprocess.run(
        ["bash", "-c", script], text=True, capture_output=True, check=False, env=env
    )


def test_spinwait_numeric_contract() -> None:
    for raw, canonical in (("stock", "stock"), ("1", "1"), ("016", "16"), ("1000", "1000")):
        result = validate_spinwait(raw)
        assert result.returncode == 0, (raw, result.stderr)
        assert result.stdout.strip() == canonical, (raw, result.stdout)
    for bad in ("", "0", "1001", "-1", "1.5", "nan", " 16", "16 ", "STOCK"):
        result = validate_spinwait(bad)
        assert result.returncode == 2, (bad, result.returncode, result.stdout)
        assert "GLM53_SPINWAIT_MS must" in result.stderr, bad


def test_mixed_prefill_contract() -> None:
    script = (
        guard_source()
        + '\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4; '
        + 'MAX_NUM_BATCHED_TOKENS=1024; GLM53_SPINWAIT_MS=stock; '
        + 'GLM53_INDEXER_WORKSPACE=stock\n'
        + 'validate_numeric_config || exit $?\n'
        + 'printf "%s|%s|%s\\n" "${GLM53_MIXED_PREFILL_CHUNK-}" '
        + '"${GLM53_FAIR_PREFILL_CHUNK-}" "${GLM53_FAIR_PREFILL_SHARE-}"\n'
    )

    def run(extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("GLM53_MIXED") and not k.startswith("GLM53_FAIR")}
        env["LC_ALL"] = "C"
        env.update(extra)
        return subprocess.run(
            ["bash", "-c", script], text=True, capture_output=True, check=False, env=env
        )

    for good in ("skip", "-1", "0", "off", "no", "fair", "128", "1024"):
        result = run({"GLM53_MIXED_PREFILL_CHUNK": good})
        assert result.returncode == 0, (good, result.stderr)
    for bad in ("", "Skip", "true", "-2", "1025", "1.5", "fair "):
        result = run({"GLM53_MIXED_PREFILL_CHUNK": bad})
        assert result.returncode == 2, (bad, result.returncode, result.stdout, result.stderr)
    result = run({
        "GLM53_MIXED_PREFILL_CHUNK": "fair",
        "GLM53_FAIR_PREFILL_CHUNK": "256",
        "GLM53_FAIR_PREFILL_SHARE": "0.20",
        "GLM53_FAIR_PREFILL_MAX_INTERVAL_MS": "2000",
        "GLM53_FAIR_PREFILL_MAX_CHUNKS": "1",
    })
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fair|256|0.20"
    result = run({
        "GLM53_MIXED_PREFILL_CHUNK": "fair",
        "GLM53_FAIR_PREFILL_SHARE": "1.1",
    })
    assert result.returncode == 2
    result = run({
        "GLM53_MIXED_PREFILL_CHUNK": "fair",
        "GLM53_FAIR_PREFILL_CHUNK": "0",
    })
    assert result.returncode == 2

    for value, expected in (("1000", 0), ("1", 0), ("0", 2), ("-1", 2), ("1.5", 2), ("600001", 2)):
        result = run({"GLM53_FAIR_PREFILL_MAX_STEP_MS": value})
        assert result.returncode == expected, (value, result.stderr)
    for launcher in (START, ROOT / "start-tp3.sh", START_TP4):
        source = launcher.read_text()
        assert 'GLM53_FAIR_PREFILL_MAX_STEP_MS="${GLM53_FAIR_PREFILL_MAX_STEP_MS:-1000}"' in source
        assert '-e "GLM53_FAIR_PREFILL_MAX_STEP_MS=$GLM53_FAIR_PREFILL_MAX_STEP_MS"' in source
        if launcher.name == "start-tp3.sh":
            assert 'GLM53_MIXED_PREFILL_CHUNK="${GLM53_MIXED_PREFILL_CHUNK:-0}"' in source
        elif launcher.name == "start-tp4.sh":
            assert 'GLM53_MIXED_PREFILL_CHUNK="${GLM53_MIXED_PREFILL_CHUNK:-skip}"' in source
        else:
            assert 'GLM53_MIXED_PREFILL_CHUNK="${GLM53_MIXED_PREFILL_CHUNK:-fair}"' in source
        guard = guard_source(launcher)
        for value, expected in (("1000", 0), ("0", 1)):
            script = guard + '\nGLM53_FAIR_PREFILL_MAX_STEP_MS="$1"\n' + '_glm53_canonical_positive_int GLM53_FAIR_PREFILL_MAX_STEP_MS "$GLM53_FAIR_PREFILL_MAX_STEP_MS" 600000\n'
            checked = subprocess.run(["bash", "-c", script, "test", value], capture_output=True, text=True)
            assert bool(checked.returncode) == bool(expected), (launcher, value, checked.stderr)
    for env_example, chunk in (
        (ROOT / ".env.example", "fair"),
        (ROOT / ".env.tp3.example", "0"),
        (ROOT / ".env.tp4.example", "skip"),
    ):
        text = env_example.read_text()
        assert f"GLM53_MIXED_PREFILL_CHUNK={chunk}" in text, env_example


def test_restart_validates_before_stop() -> None:
    source = START.read_text()
    main = source.index("main() {")
    validation = source.index("start|restart) validate_numeric_config", main)
    restart = source.index("restart)  stop; start", main)
    assert validation < restart


def test_tp4_rejects_retention_override() -> None:
    script = (
        guard_source(START_TP4)
        + '\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4; '
        + 'MAX_NUM_BATCHED_TOKENS=1024; GLM53_INDEXER_WORKSPACE=stock; '
        + 'GLM53_SPINWAIT_MS=stock; export "$1=$2"\n'
        + 'validate_numeric_config\n'
    )
    for knob in ("GLM53_APC_RETENTION_INTERVAL", "GLM53_APC_RETENTION_INTERVAL_SWA"):
        for value, expected in (("", 0), ("0", 2), ("14336", 2)):
            result = subprocess.run(
                ["bash", "-c", script, "test", knob, value],
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "LC_ALL": "C"},
            )
            assert result.returncode == expected, (value, result.stderr)

    source = START_TP4.read_text()
    assert '_cli_apc_swa_set="${GLM53_APC_RETENTION_INTERVAL_SWA+1}"' in source
    assert '[ -n "${_cli_apc_swa_set}" ] && GLM53_APC_RETENTION_INTERVAL_SWA="$_cli_apc_swa"' in source


if __name__ == "__main__":
    test_matrix()
    test_decimal_normalization()
    test_indexer_workspace_enum()
    test_spinwait_numeric_contract()
    test_mixed_prefill_contract()
    test_restart_validates_before_stop()
    test_tp4_rejects_retention_override()
    print("numeric config tests: PASS")
