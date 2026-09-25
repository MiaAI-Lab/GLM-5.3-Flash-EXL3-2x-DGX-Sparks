#!/usr/bin/env python3
"""CPU-only tests for launcher numeric type/range validation."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


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
        assert "GLM53_INDEXER_WORKSPACE" in result.stderr, bad


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
        assert "GLM53_SPINWAIT_MS" in result.stderr, bad


def test_kv_capacity_log_flag() -> None:
    script = (
        guard_source()
        + '\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4; '
        + 'MAX_NUM_BATCHED_TOKENS=1024; GLM53_INDEXER_WORKSPACE=stock; '
        + 'GLM53_SPINWAIT_MS=stock; export GLM53_KV_CAPACITY_LOG="$1"\n'
        + 'validate_numeric_config\n'
    )
    for value, expected in (("0", 0), ("1", 0), ("", 2), ("2", 2)):
        result = subprocess.run(
            ["bash", "-c", script, "test", value],
            text=True, capture_output=True, timeout=10,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        assert result.returncode == expected, (value, result.stdout, result.stderr)


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
        guard = guard_source(launcher)
        for value, expected in (("1000", 0), ("0", 1)):
            script = guard + '\nGLM53_FAIR_PREFILL_MAX_STEP_MS="$1"\n' + '_glm53_canonical_positive_int GLM53_FAIR_PREFILL_MAX_STEP_MS "$GLM53_FAIR_PREFILL_MAX_STEP_MS" 600000\n'
            checked = subprocess.run(["bash", "-c", script, "test", value], capture_output=True, text=True)
            assert bool(checked.returncode) == bool(expected), (launcher, value, checked.stderr)


def test_thin_decode_flag_rejects_bad_values_before_host_actions() -> None:
    """GLM53_EXL3_MOE_FAST is exactly 0/1 and validated before any stop.

    The native dispatcher and ``overlay/exl3.py`` refuse anything else at
    load, so a typo must not cost a running pair.
    """
    from test_launcher_rank_parity import Harness

    with tempfile.TemporaryDirectory() as directory:
        harness = Harness(Path(directory))
        for value in ("0", "1"):
            result = harness.run(
                "validate_numeric_config", entry="start.fn.sh",
                GLM53_EXL3_MOE_FAST=value)
            assert result.returncode == 0, (value, result.stderr)
            assert not harness.host_touching_calls()
        for value in ("", "yes", " 1", "1 ", "2"):
            result = harness.run("restart", GLM53_EXL3_MOE_FAST=value)
            assert result.returncode == 2, (value, result.stderr)
            assert "GLM53_EXL3_MOE_FAST" in result.stderr, value
            assert not harness.host_touching_calls(), (value, harness.calls())


def test_kda_bf16_large_m_flag_rejects_bad_values_before_host_actions() -> None:
    """GLM53_KDA_BF16_LARGE_M is exactly 0/1 and validated before any stop.

    ``overlay/exl3.py`` refuses anything else at model load, so a typo must
    not cost a running pair.
    """
    from test_launcher_rank_parity import Harness

    with tempfile.TemporaryDirectory() as directory:
        harness = Harness(Path(directory))
        for value in ("0", "1"):
            result = harness.run(
                "validate_numeric_config", entry="start.fn.sh",
                GLM53_KDA_BF16_LARGE_M=value)
            assert result.returncode == 0, (value, result.stderr)
            assert not harness.host_touching_calls()
        for value in ("", "yes", " 1", "1 ", "2", "true"):
            result = harness.run("restart", GLM53_KDA_BF16_LARGE_M=value)
            assert result.returncode == 2, (value, result.stderr)
            assert "GLM53_KDA_BF16_LARGE_M" in result.stderr, value
            assert not harness.host_touching_calls(), (value, harness.calls())


def test_tp3_kda_bf16_large_m_flag_is_0_or_1() -> None:
    """start-tp3.sh validates GLM53_KDA_BF16_LARGE_M before any stop."""
    guard = guard_source(ROOT / "start-tp3.sh")
    for value, expected in (("0", 0), ("1", 0), ("", 2), ("yes", 2), ("2", 2)):
        script = (
            guard
            + '\nGLM53_KDA_BF16_LARGE_M="$1"\n'
            + '_glm53_validate_enum GLM53_KDA_BF16_LARGE_M '
            + '"$GLM53_KDA_BF16_LARGE_M" 0 1\n'
        )
        result = subprocess.run(
            ["bash", "-c", script, "test", value],
            capture_output=True, text=True)
        assert result.returncode == expected, (value, result.stderr)
        if expected:
            assert "GLM53_KDA_BF16_LARGE_M" in result.stderr


# #207 makes the prefix-cache retention intervals configurable on every launcher
# (start.sh / start-tp3.sh / start-tp4.sh): "" (unset) and 0 pass, and a positive
# value must sit on the 3584-token scheduler-block grid, at most 1e6.
RETENTION_LAUNCHERS = (START, ROOT / "start-tp3.sh", START_TP4)
RETENTION_OK = (None, "", "0", "14336", "03584", "999936")
RETENTION_BAD = ("1", "-3584", "3584.0", "1003520")


def coop_overlay_selected_source(launcher: Path) -> str:
    """The real coop-overlay selector, for the launchers whose guard calls it.

    start-tp3.sh's guard asks whether the selected overlay wants the
    cooperative MoE runtime. ``EXL3_OVERLAY_HOST`` is unset here, so the real
    function reports "not selected" and the guard skips that branch instead of
    failing on a missing command.
    """
    source = launcher.read_text()
    marker = "_glm53_coop_overlay_selected() {"
    if marker not in source:
        return ""
    begin = source.index(marker)
    return source[begin:source.index("\n}\n", begin) + 3] + "\n"


def validate_retention(
    launcher: Path, knob: str, value: str | None, spec_method: str = "none"
) -> subprocess.CompletedProcess[str]:
    """Run one launcher's own numeric guard with only ``knob`` set.

    ``None`` leaves the knob unset, ``""`` exports it empty — the two are
    distinct contracts ("inherit" vs "explicitly empty").
    """
    export = "" if value is None else 'export "$2=$3"\n'
    script = (
        guard_source(launcher)
        + "\n"
        + coop_overlay_selected_source(launcher)
        + '\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4; '
        + 'MAX_NUM_BATCHED_TOKENS=1024; GLM53_INDEXER_WORKSPACE=stock; '
        + 'GLM53_SPINWAIT_MS=stock; HAREM_KDA_FLASHKDA=0; SPEC_METHOD="$1"\n'
        + export
        + 'validate_numeric_config || exit $?\n'
        + f'printf "%s\\n" "${{{knob}-unset}}"\n'
    )
    env = {
        key: val
        for key, val in os.environ.items()
        if not key.startswith("GLM53_") and key != "SPEC_METHOD"
    }
    env["LC_ALL"] = "C"
    return subprocess.run(
        ["bash", "-c", script, "test", spec_method, knob, value or ""],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def canonical_retention(value: str | None) -> str:
    """What the guard must hand to the ranks: unset stays unset, zeros stripped."""
    if value is None:
        return "unset"
    if value == "":
        return ""
    return value.lstrip("0") or "0"


@pytest.mark.parametrize("launcher", RETENTION_LAUNCHERS, ids=lambda path: path.name)
def test_global_retention_interval_contract(launcher: Path) -> None:
    """The global interval is a token count on the scheduler-block grid.

    Invalid values return launcher error 2; accepted values are normalized by
    the real guard before launch.
    """
    for value in RETENTION_OK:
        result = validate_retention(launcher, "GLM53_APC_RETENTION_INTERVAL", value)
        assert result.returncode == 0, (launcher, value, result.stderr)
        assert result.stdout.strip() == canonical_retention(value), (launcher, value)
    for value in RETENTION_BAD:
        result = validate_retention(launcher, "GLM53_APC_RETENTION_INTERVAL", value)
        assert result.returncode == 2, (launcher, value, result.stdout, result.stderr)
        assert "GLM53_APC_RETENTION_INTERVAL" in result.stderr, (launcher, value)


@pytest.mark.parametrize("launcher", RETENTION_LAUNCHERS, ids=lambda path: path.name)
def test_swa_retention_interval_needs_the_dflash_drafter(launcher: Path) -> None:
    """The SWA interval is the DFlash2 drafter's; anything else is refused.

    Empty and unset stay valid for every speculator (they inherit the global
    policy), while a set value requires ``SPEC_METHOD=dflash``.
    """
    for value in RETENTION_OK:
        result = validate_retention(
            launcher, "GLM53_APC_RETENTION_INTERVAL_SWA", value, spec_method="dflash"
        )
        assert result.returncode == 0, (launcher, value, result.stderr)
        assert result.stdout.strip() == canonical_retention(value), (launcher, value)
    for value in RETENTION_BAD:
        result = validate_retention(
            launcher, "GLM53_APC_RETENTION_INTERVAL_SWA", value, spec_method="dflash"
        )
        assert result.returncode == 2, (launcher, value, result.stdout, result.stderr)
    for value in (None, ""):
        result = validate_retention(
            launcher, "GLM53_APC_RETENTION_INTERVAL_SWA", value, spec_method="mtp"
        )
        assert result.returncode == 0, (launcher, value, result.stderr)
    for value in ("0", "3584"):
        for spec_method in ("mtp", "none"):
            result = validate_retention(
                launcher,
                "GLM53_APC_RETENTION_INTERVAL_SWA",
                value,
                spec_method=spec_method,
            )
            assert result.returncode == 2, (launcher, value, spec_method, result.stdout)
            assert "SPEC_METHOD=dflash" in result.stderr, (launcher, value, spec_method)



@pytest.mark.parametrize("launcher", ("start.sh", "start-tp3.sh", "start-tp4.sh"))
def test_loader_controls_precedence_and_early_rejection(launcher: str) -> None:
    from test_launcher_rank_parity import Harness

    with tempfile.TemporaryDirectory() as directory:
        harness = Harness(Path(directory), launcher=launcher)
        fn = harness.repo / "start.fn.sh"
        source = fn.read_text()
        fn.write_text(source.rsplit('"$@"', 1)[0] + '''
inspect_loader() {
    validate_numeric_config || return
    printf '%s|%s|%s\\n' "$GLM53_LOAD_CLONE" "$GLM53_LOAD_PREFETCH" "$LOAD_FORMAT"
}
"$@"
''')
        dotenv = harness.repo / ".env"
        shared = dotenv.read_text()
        topology = None if launcher == "start.sh" else harness.repo / (".env." + launcher[6:-3])
        for clone, depth, expected in (("0", "00016", "0|16|"), ("1", "0", "1|0|")):
            # Caller overrides both shared and topology files, including LOAD_FORMAT=.
            dotenv.write_text(shared + "\nGLM53_LOAD_CLONE=1\nGLM53_LOAD_PREFETCH=8\n")
            result = harness.run("inspect_loader", entry="start.fn.sh",
                                 GLM53_LOAD_CLONE=clone, GLM53_LOAD_PREFETCH=depth, LOAD_FORMAT="")
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == expected
            assert not harness.host_touching_calls()
        dotenv.write_text(shared + "\nGLM53_LOAD_CLONE=0\nGLM53_LOAD_PREFETCH=6\n")
        if topology:
            topology.write_text(topology.read_text() + "\nGLM53_LOAD_CLONE=1\nGLM53_LOAD_PREFETCH=3\n")
        result = harness.run("inspect_loader", entry="start.fn.sh")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().split("|")[:2] == (["1", "3"] if topology else ["0", "6"])
        # Every error must reach the pre-stop gate, even if .env supplies a valid value.
        for key, values in (("GLM53_LOAD_CLONE", ("", "01", "yes", "2")),
                            ("GLM53_LOAD_PREFETCH", ("", "-1", "17", "1.5", " 2", "99999999999999999999"))):
            for value in values:
                result = harness.run("restart", **{key: value})
                assert result.returncode == 2, (launcher, key, value, result.stderr)
                assert key in result.stderr
                assert not harness.host_touching_calls(), harness.calls()


if __name__ == "__main__":
    test_matrix()
    test_decimal_normalization()
    test_indexer_workspace_enum()
    test_spinwait_numeric_contract()
    test_kv_capacity_log_flag()
    test_mixed_prefill_contract()
    test_thin_decode_flag_rejects_bad_values_before_host_actions()
    test_kda_bf16_large_m_flag_rejects_bad_values_before_host_actions()
    test_tp3_kda_bf16_large_m_flag_is_0_or_1()
    for _launcher in RETENTION_LAUNCHERS:
        test_global_retention_interval_contract(_launcher)
        test_swa_retention_interval_needs_the_dflash_drafter(_launcher)
    for _launcher in ("start.sh", "start-tp3.sh", "start-tp4.sh"):
        test_loader_controls_precedence_and_early_rejection(_launcher)
    print("numeric config tests: PASS")
