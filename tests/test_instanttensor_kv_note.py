#!/usr/bin/env python3
"""Regression test for the InstantTensor KV-fit preflight note (#204).

The note is diagnostic only: it must fire for exactly the combination measured not to boot
(loader on, MAX_MODEL_LEN >= 850000, GPU_MEM_UTIL <= 0.85, no --kv-cache-memory-bytes), stay
silent otherwise, never return non-zero, so it can never abort a boot under set -e, and decide
the same way under a comma-decimal LC_NUMERIC (#242). The TP3/TP4 EXTRA_ARGS precedence path is
driven through the real launcher preamble, including the dropped-cap diagnostic.
"""
from __future__ import annotations

import functools
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def note_source() -> str:
    source = START.read_text()
    begin = source.index("# GLM53 InstantTensor KV-fit note (begin)")
    end_marker = "# GLM53 InstantTensor KV-fit note (end)"
    end = source.index(end_marker, begin) + len(end_marker)
    return source[begin:end]


def _run(
    load_format: str,
    max_model_len: str,
    util: str,
    extra_args: str,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    script = (
        "set -euo pipefail\n"
        "warn() { printf 'NOTE:%s\\n' \"$*\" >&2; }\n"
        + note_source()
        + '\npreflight_instanttensor_kv_note "$1" "$2" "$3" "$4"\n'
        + 'printf "REACHED\\n"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script, "_", load_format, max_model_len, util, extra_args],
        capture_output=True, text=True, timeout=30, env=env,
    )
    assert "REACHED" in result.stdout, f"function aborted the script: {result.stderr}"
    return result.returncode, result.stderr


def test_note_fires_for_the_measured_failing_combination() -> None:
    rc, err = _run("instanttensor", "850000", "0.85", "")
    assert rc == 0
    assert "NOTE:" in err and "#204" in err
    assert "--kv-cache-memory-bytes 15032385536" in err, "must name the flag"
    assert "keeping any flags" in err, "must say to append, not replace (#242 review)"
    # larger context and lower share are the same failure or worse
    assert "NOTE:" in _run("instanttensor", "900000", "0.85", "")[1]
    assert "NOTE:" in _run("instanttensor", "850000", "0.80", "")[1]
    # an unrelated EXTRA_ARGS entry does not count as a KV size
    assert "NOTE:" in _run("instanttensor", "850000", "0.85", "--no-async-scheduling")[1]
    # a flag that merely starts with the KV flag's name is not the KV flag
    assert "NOTE:" in _run("instanttensor", "850000", "0.85", "--kv-cache-memory-bytes-invalid 1")[1]


def test_note_is_silent_when_the_combination_is_not_the_failing_one() -> None:
    for case in (
        ("", "850000", "0.85", ""),                       # loader off
        ("instanttensor", "700000", "0.85", ""),          # below the 850k floor
        ("instanttensor", "850000", "0.88", ""),          # higher share (marginal, but not the stock case)
        ("instanttensor", "850000", "0.85", "--kv-cache-memory-bytes 15032385536"),
        ("instanttensor", "850000", "0.85", "--kv-cache-memory-bytes=15032385536"),
        ("instanttensor", "850000", "0.85", "--foo --kv-cache-memory-bytes 1 --bar"),
        ("instanttensor", "850000", "0.85", "--foo\t--kv-cache-memory-bytes 1"),     # tab-separated
        ("instanttensor", "850000", "0.85", "--foo\n--kv-cache-memory-bytes=1"),    # newline-separated
        ("instanttensor", "not-a-number", "0.85", ""),   # malformed input must not fire or fail
        ("instanttensor", "850000", "", ""),             # unset share (a sliced harness) must not fire or fail
        ("instanttensor", "850000", "abc", ""),          # malformed share likewise
    ):
        rc, err = _run(*case)
        assert rc == 0, case
        assert "NOTE:" not in err, case


COMMA_DECIMAL_LOCALES = (
    "de_DE.UTF-8", "de_DE.utf8",
    "fr_FR.UTF-8", "fr_FR.utf8",
    "nl_NL.UTF-8", "nl_NL.utf8",
    "es_ES.UTF-8", "es_ES.utf8",
    "it_IT.UTF-8", "it_IT.utf8",
    "pt_BR.UTF-8", "pt_BR.utf8",
)


@functools.lru_cache(maxsize=1)
def comma_decimal_locale() -> str:
    """A comma-decimal locale that is really installed and in effect.

    Probed through `locale -k LC_NUMERIC`: an unavailable name falls back to C silently, so
    without the probe the boundary test below would pass vacuously. No locale is required to
    run this suite — when none is installed the test skips with that reason. The probe inherits
    os.environ, so a locale exposed through LOCPATH counts. #242 review.
    """
    for name in COMMA_DECIMAL_LOCALES:
        probe = subprocess.run(
            ["locale", "-k", "LC_NUMERIC"], env={**os.environ, "LC_ALL": name},
            capture_output=True, text=True, timeout=30,
        )
        if 'decimal_point=","' in probe.stdout:
            return name
    pytest.skip(
        "no comma-decimal locale installed, nothing to compare the note's boundary against; "
        f"tried {', '.join(COMMA_DECIMAL_LOCALES)}"
    )


@pytest.mark.parametrize("util,fires", [
    ("0.85", True), (".85", True), ("0.8", True), ("0.851", False), ("1.0", False),
])
def test_note_boundary_decides_the_same_under_a_comma_decimal_locale(util: str, fires: bool) -> None:
    """#242 review: the cut is pinned to a command-local C locale as portability hardening, so an
    ambient comma-decimal LC_NUMERIC cannot move it. Fires at or below 0.85, silent above it,
    and never aborts — in either locale."""
    for locale_name in ("C", comma_decimal_locale()):
        env = {**os.environ, "LC_ALL": locale_name, "LC_NUMERIC": locale_name}
        rc, err = _run("instanttensor", "850000", util, "", env=env)
        assert rc == 0, (locale_name, util, err)
        assert ("NOTE:" in err) is fires, (locale_name, util, err)


def _tp_precedence_path(launcher: str) -> str:
    """start-tp3/tp4 from the first _cli_ capture through the EXTRA_ARGS restore: the whole
    caller -> .env -> shared-cap strip -> topology overlay -> caller-restore path, matched by
    text so it survives edits around it. That region is captures, sources and assignments."""
    source = (ROOT / launcher).read_text()
    begin = source.index('_cli_mtp="${MTP_TOKENS-}"\n')
    end_line = '[ -n "${_cli_extra_args_set}" ] && EXTRA_ARGS="$_cli_extra_args"\n'
    end = source.index(end_line) + len(end_line)
    return source[begin:end]


def _run_tp_precedence(launcher: str, n: str, dotenv: str, tpenv: str, caller: dict[str, str]) -> tuple[str, str]:
    script = 'set -euo pipefail\nSCRIPT_DIR="$PWD"\n' + _tp_precedence_path(launcher) + '\nprintf "%s" "${EXTRA_ARGS-<unset>}"\n'
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        (tmp / ".env").write_text(dotenv)
        (tmp / f".env.tp{n}").write_text(tpenv)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp)}
        env.update(caller)
        r = subprocess.run(["bash", "-c", script], cwd=tmp, capture_output=True, text=True, env=env, timeout=30)
        assert r.returncode == 0, (launcher, r.stderr)
        return r.stdout, r.stderr


CAP = 'EXTRA_ARGS="--kv-cache-memory-bytes 15032385536"\n'
CALLER = "--kv-cache-memory-bytes 21474836480 --max-log-len 7"


@pytest.mark.parametrize("launcher,n", [("start-tp3.sh", "3"), ("start-tp4.sh", "4")])
def test_tp3_tp4_extra_args_precedence_end_to_end(launcher: str, n: str) -> None:
    """#204 / PR #242 review: the shared .env template ships a 14 GiB cap sized for TP=2, though
    an operator can put any cap there. start-tp3/tp4 must drop that token from the FILE-derived
    value (keeping other flags) while preserving caller precedence verbatim — including an
    explicit empty — and topology-file overrides. Runs the real launcher preamble, not the
    extracted strip loop. `want_note` is the dropped-cap diagnostic: it must name the topology
    file, echo no argument values, and never appear when the caller supplied EXTRA_ARGS (their
    value is restored, so nothing was lost)."""
    cases = [
        # (.env, .env.tpN, caller env, expected EXTRA_ARGS, dropped-cap diagnostic expected)
        (CAP, "", {}, "", True),                                                              # inherited cap dropped
        ('EXTRA_ARGS="--kv-cache-memory-bytes 15032385536 --no-async-scheduling"\n', "", {}, "--no-async-scheduling", True),
        ('EXTRA_ARGS="--kv-cache-memory-bytes=15032385536 --x"\n', "", {}, "--x", True),      # '=' spelling in .env
        ('EXTRA_ARGS="--no-async-scheduling"\n', "", {}, "--no-async-scheduling", False),     # no cap to drop
        (CAP, "", {"EXTRA_ARGS": CALLER}, CALLER, False),                                     # reviewer case 1
        ("", "", {"EXTRA_ARGS": CALLER}, CALLER, False),                                      # reviewer case 2
        (CAP, 'EXTRA_ARGS="--foo"\n', {"EXTRA_ARGS": ""}, "", False),                         # explicit empty caller wins
        (CAP, 'EXTRA_ARGS="--foo"\n', {}, "--foo", True),                                     # topology override kept
        (CAP, 'EXTRA_ARGS="--kv-cache-memory-bytes 20000000000"\n', {}, "--kv-cache-memory-bytes 20000000000", True),  # topology pin survives; shared drop still reported
        (CAP, 'EXTRA_ARGS="--foo"\n', {"EXTRA_ARGS": "--bar"}, "--bar", False),               # caller beats topology
        (CAP, "", {"EXTRA_ARGS": "--kv-cache-memory-bytes=1 --x"}, "--kv-cache-memory-bytes=1 --x", False),  # caller's own cap kept
    ]
    for dotenv, tpenv, caller, want, want_note in cases:
        got, err = _run_tp_precedence(launcher, n, dotenv, tpenv, caller)
        assert got == want, (launcher, dotenv, tpenv, caller, got)
        assert ("NOTE:" in err) is want_note, (launcher, dotenv, tpenv, caller, err)
        if want_note:
            assert "--kv-cache-memory-bytes" in err, (launcher, err)
            assert f".env.tp{n}" in err, (launcher, err)
            assert "15032385536" not in err, ("must not echo argument contents", launcher, err)


if __name__ == "__main__":
    test_note_fires_for_the_measured_failing_combination()
    test_note_is_silent_when_the_combination_is_not_the_failing_one()
    util_cases = [("0.85", True), (".85", True), ("0.8", True), ("0.851", False), ("1.0", False)]
    for util, fires in util_cases:
        try:
            test_note_boundary_decides_the_same_under_a_comma_decimal_locale(util, fires)
        except pytest.skip.Exception as exc:
            print(f"locale boundary test skipped: {exc}")
            break
    for launcher, n in (("start-tp3.sh", "3"), ("start-tp4.sh", "4")):
        test_tp3_tp4_extra_args_precedence_end_to_end(launcher, n)
    print("instanttensor kv-fit note OK")
