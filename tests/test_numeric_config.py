#!/usr/bin/env python3
"""CPU-only tests for launcher numeric type/range validation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def guard_source() -> str:
    source = START.read_text()
    begin = source.index("# GLM53 numeric config guard (begin)")
    end_marker = "# GLM53 numeric config guard (end)"
    end = source.index(end_marker, begin) + len(end_marker)
    return source[begin:end]


def validate(
    util: str, model: str, seqs: str, batch: str, **extra: str
) -> subprocess.CompletedProcess[str]:
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
        env={**os.environ, "LC_ALL": "C", **extra},
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


MTP_LIST = "1 2 3 4 6 8 12"
DFLASH_LIST = "1 2 4 8 16 24 32"


def test_mtp_capture_envelope() -> None:
    """Issue #88: with the launcher-generated MTP capture list (max 12) vLLM's
    profiling pass on this image rejects MAX_NUM_SEQS > 12 in EngineCore init.
    The guard refuses that combination (rc 2) and names the real error; the
    DFlash list, eager mode and a caller-supplied list are not gated."""
    ok = [
        ("mtp", "12", MTP_LIST),
        ("mtp", "0012", MTP_LIST),
        ("mtp", "4", MTP_LIST),
        ("none", "12", MTP_LIST),
        ("dflash", "16", DFLASH_LIST),
        ("dflash", "4096", DFLASH_LIST),
        ("mtp", "16", ""),          # ENFORCE_EAGER=1 or caller list: nothing generated
        ("mtp", "4096", ""),
    ]
    for spec, seqs, sizes in ok:
        r = validate("0.87", "1000000", seqs, "2048", SPEC_METHOD=spec, GLM53_CG_CAPTURE_SIZES=sizes)
        assert r.returncode == 0, (spec, seqs, sizes, r.returncode, r.stderr)
    # unset SPEC_METHOD in the spliced harness == launcher default dflash: never gated
    r = validate("0.87", "1000000", "16", "2048", GLM53_CG_CAPTURE_SIZES=MTP_LIST)
    assert r.returncode == 0, r.stderr
    bad = [
        ("mtp", "13", MTP_LIST),
        ("mtp", "16", MTP_LIST),
        ("mtp", "0016", MTP_LIST),
        ("mtp", "4096", MTP_LIST),
        ("none", "16", MTP_LIST),
        ("eagle", "16", MTP_LIST),  # any non-dflash value takes the launcher's mtp branch
    ]
    for spec, seqs, sizes in bad:
        r = validate("0.87", "1000000", seqs, "2048", SPEC_METHOD=spec, GLM53_CG_CAPTURE_SIZES=sizes)
        assert r.returncode == 2, (spec, seqs, sizes, r.returncode, r.stderr)
        assert "Mamba cache blocks (12)" in r.stderr and "issue #88" in r.stderr, r.stderr
        assert f"MAX_NUM_SEQS={int(seqs)} exceeds" in r.stderr, r.stderr
        assert "at most 12" in r.stderr and "ENFORCE_EAGER=1" in r.stderr, r.stderr
    # the cap follows the list, not a second hard-coded 12
    r = validate("0.87", "1000000", "16", "2048", SPEC_METHOD="mtp", GLM53_CG_CAPTURE_SIZES="1 2 3 4 6 8 12 16")
    assert r.returncode == 0, r.stderr
    r = validate("0.87", "1000000", "17", "2048", SPEC_METHOD="mtp", GLM53_CG_CAPTURE_SIZES="1 2 3 4 6 8 12 16")
    assert r.returncode == 2 and "Mamba cache blocks (16)" in r.stderr, r.stderr
    # the range check still runs first: a bad MAX_NUM_SEQS is reported as such
    r = validate("0.87", "1000000", "O16", "2048", SPEC_METHOD="mtp", GLM53_CG_CAPTURE_SIZES=MTP_LIST)
    assert r.returncode == 2 and "positive base-10 integer" in r.stderr, r.stderr


def prologue_capture_sizes(**env: str) -> str:
    """Run the REAL start.sh prologue + configuration (trailing `main "$@"`
    swapped for `"$@"`, then `eval` a printf) from an allow-listed environment
    and return the GLM53_CG_CAPTURE_SIZES it generated. The prologue makes no
    host calls; PATH is the system dirs only."""
    import tempfile

    text = START.read_text()
    assert text.rstrip().endswith('\nmain "$@"')
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "start.fn.sh").write_text(text.rstrip()[: -len('main "$@"')] + '"$@"\n')
        (tmp / ".env").write_text((ROOT / ".env.example").read_text())
        base = {"PATH": "/usr/bin:/bin", "HOME": raw, "USER": "glm53", "LC_ALL": "C"}
        r = subprocess.run(
            ["bash", "./start.fn.sh", "eval", 'printf "%s" "${GLM53_CG_CAPTURE_SIZES-<unset>}"'],
            cwd=tmp, text=True, capture_output=True, check=False, env={**base, **env},
        )
    assert r.returncode == 0, r.stderr[-400:]
    return r.stdout


def test_prologue_generated_list_feeds_the_guard() -> None:
    """The value the guard reads is the one the real prologue generates: set
    for the launcher-generated lists, empty under ENFORCE_EAGER=1 or when the
    caller supplies its own --cudagraph-capture-sizes."""
    cases = {
        "mtp generated": ({"SPEC_METHOD": "mtp"}, MTP_LIST, 2),
        "none generated": ({"SPEC_METHOD": "none"}, MTP_LIST, 2),
        "dflash generated": ({"SPEC_METHOD": "dflash"}, DFLASH_LIST, 0),
        "mtp eager": ({"SPEC_METHOD": "mtp", "ENFORCE_EAGER": "1"}, "", 0),
        "mtp caller list": ({"SPEC_METHOD": "mtp", "EXTRA_ARGS": "--cudagraph-capture-sizes 1 2 4 8 16"}, "", 0),
    }
    for label, (env, want_sizes, want_rc) in cases.items():
        sizes = prologue_capture_sizes(**env)
        assert sizes == want_sizes, (label, sizes)
        r = validate("0.87", "1000000", "16", "2048", SPEC_METHOD=env["SPEC_METHOD"], GLM53_CG_CAPTURE_SIZES=sizes)
        assert r.returncode == want_rc, (label, r.returncode, r.stderr)


def test_launcher_generates_the_lists_it_validates() -> None:
    """The capture list the launcher appends to EXTRA_ARGS and the one the
    guard reads are the same variable (byte-identical lists as before)."""
    src = START.read_text()
    assert f'GLM53_CG_CAPTURE_SIZES="{MTP_LIST}"' in src
    assert f'GLM53_CG_CAPTURE_SIZES="{DFLASH_LIST}"' in src
    assert '--cudagraph-capture-sizes $GLM53_CG_CAPTURE_SIZES"' in src
    assert "_glm53_validate_cudagraph_capture_envelope || return" in guard_source()
    assert "_glm53_validate_cudagraph_capture_envelope() {" in guard_source()
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


def test_restart_validates_before_stop() -> None:
    source = START.read_text()
    main = source.index("main() {")
    validation = source.index("start|restart) validate_numeric_config", main)
    restart = source.index("restart)  stop; start", main)
    assert validation < restart


if __name__ == "__main__":
    test_matrix()
    test_decimal_normalization()
    test_mtp_capture_envelope()
    test_prologue_generated_list_feeds_the_guard()
    test_launcher_generates_the_lists_it_validates()
    test_indexer_workspace_enum()
    test_spinwait_numeric_contract()
    test_restart_validates_before_stop()
    print("numeric config tests: PASS")
