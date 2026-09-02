#!/usr/bin/env python3
"""Regression tests for caller overrides that must win over ``.env`` (#28, #44, #91).

The README rule is "prefix env wins over .env". start.sh implements it in the
prologue (before the configuration section): remember the caller's non-empty
value of every key .env assigns, source .env, re-apply them. These tests drive
that prologue two ways:

  1. The spliced preamble (everything before the configuration marker) with a
     one-key .env -- the original #28 regression, unchanged.
  2. The REAL start.sh, with its trailing ``main "$@"`` swapped for ``"$@"`` so
     the full prologue + configuration section runs and a printf can then read
     the resolved values, from an allow-listed environment with docker / ssh /
     scp / rsync / curl / ip / nvidia-smi stubbed first on PATH (nothing
     touches a host; the stubs record every call and the tests assert none).
     Every key .env.example defines (45 at the time of writing) is exported by
     the caller and must survive; a child process must inherit the override
     (the ``set -a`` contract); with no caller export .env must still beat the
     script defaults; an explicitly empty export must NOT override (the
     ``[ -n ]`` semantics the per-knob allowlist had); a readonly exported
     shell variable (SHELLOPTS) must not break the prologue, even when a
     heredoc line in .env looks like `SHELLOPTS=...`; a .env naming a
     launcher-internal variable (SCRIPT_DIR) must not replay it; a .env with
     comments, blank lines, indentation, ``export KEY=``, ``KEY+=``, a CRLF
     line (sourced with the carriage return kept, as before) and a quoted
     JSON value must parse. The matrix runs under every bash found on
     the host (``bash`` on PATH, /bin/bash -- 3.2 on macOS -- and Homebrew's
     5.x when present).

Run:  python3 tests/test_start_overrides.py   (or pytest)
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"
ENV_EXAMPLE = ROOT / ".env.example"
CONFIG_MARKER = "# ----------------------------- configuration -------------------------------"

# Same shape as the sed in start.sh: an assignment at line start, optional
# `export`, optional leading whitespace. Comments (`# KEY=`) do not count.
KEY_RE = re.compile(r"^[ \t]*(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)\+?=", re.M)
HOST_TOOLS = ("docker", "ssh", "scp", "rsync", "curl", "ip", "nvidia-smi")
SEP = "\x1f"

STUB = """#!/usr/bin/env bash
# Records every invocation; never touches a host.
{ printf '%s\\x1f' "$(basename "$0")" "$@"; printf '\\n'; } >> "$GLM53_STUB_LOG"
exit 0
"""


def env_keys(text: str) -> list[str]:
    seen: list[str] = []
    for key in KEY_RE.findall(text):
        if key not in seen:
            seen.append(key)
    return seen


def bashes() -> list[str]:
    """Every distinct bash on this host: PATH's, /bin/bash (macOS 3.2), Homebrew 5.x."""
    found: list[str] = []
    seen: set[str] = set()
    for cand in (shutil.which("bash"), "/bin/bash", "/opt/homebrew/bin/bash", "/usr/local/bin/bash"):
        if not cand or not Path(cand).is_file() or not os.access(cand, os.X_OK):
            continue
        real = os.path.realpath(cand)
        if real in seen:
            continue
        seen.add(real)
        found.append(cand)
    assert found, "no bash found"
    return found


def bash_version(bash: str) -> str:
    out = subprocess.run([bash, "-c", 'printf "%s" "$BASH_VERSION"'], capture_output=True, text=True, check=True)
    return out.stdout.strip()


class Harness:
    """Throwaway copy of the launcher plus a stub PATH (no real docker / ssh)."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.repo.mkdir()
        shutil.copy2(START, self.repo / "start.sh")
        shutil.copy2(ENV_EXAMPLE, self.repo / ".env.example")
        text = (self.repo / "start.sh").read_text()
        assert text.rstrip().endswith('\nmain "$@"'), 'start.sh must end with main "$@"'
        (self.repo / "start.fn.sh").write_text(text.rstrip()[: -len('main "$@"')] + '"$@"\n')
        self.home = tmp / "home"
        self.home.mkdir()
        self.bin = tmp / "bin"
        self.bin.mkdir()
        for tool in HOST_TOOLS:
            p = self.bin / tool
            p.write_text(STUB)
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = tmp / "calls.log"

    def env(self, extra: dict[str, str]) -> dict[str, str]:
        # Allow-listed: nothing from the developer shell leaks in (no BASH_ENV,
        # HF_*, GLM53_*, SKIP_* ...); PATH is the stubs plus the system dirs only.
        env = {
            "PATH": f"{self.bin}{os.pathsep}/usr/bin:/bin",
            "HOME": str(self.home),
            "USER": "glm53-overrides",
            "LC_ALL": "C",
            "TERM": "dumb",
            "GLM53_STUB_LOG": str(self.log),
        }
        env.update(extra)
        return env

    def host_calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return [line.split(SEP)[0] for line in self.log.read_text().splitlines() if line]

    def resolve(self, bash: str, dotenv: str, keys: list[str], caller: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        """Run the full prologue + configuration under `bash`; return the
        resolved value of every key in the launcher and in a child process."""
        (self.repo / ".env").write_text(dotenv)
        if self.log.exists():
            self.log.unlink()
        names = " ".join(keys)
        # `eval` runs in the launcher's own context after the whole prologue.
        body = (
            f'for _n in {names}; do printf "P\\x1f%s\\x1f%s\\n" "$_n" "${{!_n-<unset>}}"; done; '
            f'"$BASH" -c \'for _n in {names}; do printf "C\\x1f%s\\x1f%s\\n" "$_n" "${{!_n-<unset>}}"; done\''
        )
        # bytes, not text=True: universal newlines would turn a CRLF value's
        # carriage return into a newline before we can see it.
        r = subprocess.run(
            [bash, "./start.fn.sh", "eval", body],
            cwd=self.repo,
            capture_output=True,
            check=False,
            env=self.env(caller),
        )
        assert r.returncode == 0, (bash, r.returncode, r.stderr[-800:].decode(errors="replace"))
        parent: dict[str, str] = {}
        child: dict[str, str] = {}
        for line in r.stdout.decode(errors="replace").split("\n"):
            if SEP not in line:
                continue
            scope, name, value = line.split(SEP, 2)
            (parent if scope == "P" else child)[name] = value
        assert not self.host_calls(), f"prologue must not touch a host: {self.host_calls()}"
        return parent, child


def caller_value(key: str) -> str:
    if key == "LIMIT_MM":
        return '{"image":9,"video":2}'
    if key == "SERVED_MODEL_NAME":
        return 'caller "quoted" value with spaces = and equals'
    return f"caller-{key}"


def test_no_per_knob_allowlist_remains() -> None:
    src = START.read_text()
    # The generic [ -n ] replay cannot express "an explicitly EMPTY caller
    # export must reach validate_numeric_config", which upstream's
    # GLM53_INDEXER_WORKSPACE / GLM53_SPINWAIT_MS knobs require, so exactly
    # those two keep a documented setness-aware capture. No other _cli_*.
    import re as _re
    cli_tokens = set(_re.findall(r"_cli_[A-Za-z0-9_]*", src))
    allowed = {
        "_cli_indexer_workspace", "_cli_indexer_workspace_set",
        "_cli_spinwait_ms", "_cli_spinwait_ms_set",
    }
    assert cli_tokens <= allowed, (
        f"the per-knob _cli_* allowlist must be gone (generic rule, #91); "
        f"unexpected: {sorted(cli_tokens - allowed)}"
    )
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "export -p" not in code, "never replay every export (readonly declare -rx fails under set -e)"
    prologue = src.partition(CONFIG_MARKER)[0]
    assert 'source "$SCRIPT_DIR/.env"' in prologue
    assert "_caller_overrides" in prologue and "${!_k:+x}" in prologue
    assert 'declare -p "$_k"' in prologue, "only exported names are captured"
    assert "*r*) continue" in prologue, "readonly (declare -rx) names are never replayed"


def test_max_num_seqs_inline_override_wins() -> None:
    """The original #28 regression: spliced preamble, one-key .env."""
    source = START.read_text()
    preamble, separator, _rest = source.partition(CONFIG_MARKER)
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


def test_every_env_example_key_survives_a_caller_export() -> None:
    example = ENV_EXAMPLE.read_text()
    keys = env_keys(example)
    assert len(keys) >= 40, keys
    for key in ("MAX_NUM_SEQS", "GLM53_MIXED_PREFILL_CHUNK", "CG_ESTIMATE", "MAX_MODEL_LEN", "LIMIT_MM", "SPEC_METHOD",
                # new names since the 45-key receipt: PR77 E2 kernel + #86/#96 knobs
                "EXL3_FAT_KERNEL", "GLM53_INDEXER_WORKSPACE", "GLM53_SPINWAIT_MS"):
        assert key in keys, key
    caller = {k: caller_value(k) for k in keys}
    with tempfile.TemporaryDirectory() as raw_tmp:
        h = Harness(Path(raw_tmp))
        for bash in bashes():
            ver = bash_version(bash)
            parent, child = h.resolve(bash, example, keys, caller)
            lost = {k: parent.get(k) for k in keys if parent.get(k) != caller[k]}
            assert not lost, f"{bash} ({ver}): .env clobbered caller exports: {lost}"
            not_inherited = {k: child.get(k) for k in keys if child.get(k) != caller[k]}
            assert not not_inherited, f"{bash} ({ver}): child did not inherit: {not_inherited}"
            print(f"  ok   {len(keys)} .env.example keys survive a caller export under {bash} ({ver})")


def test_env_still_beats_script_defaults_without_caller_exports() -> None:
    """No caller export: the .env value wins over the start.sh default and is
    exported to children (unchanged behaviour)."""
    keys = ["MAX_NUM_SEQS", "MAX_MODEL_LEN", "GLM53_MIXED_PREFILL_CHUNK", "CG_ESTIMATE", "DFLASH_TOKENS", "SPEC_METHOD", "LIMIT_MM"]
    dotenv = (
        "MAX_NUM_SEQS=2\nMAX_MODEL_LEN=200000\nGLM53_MIXED_PREFILL_CHUNK=512\n"
        "CG_ESTIMATE=0\nDFLASH_TOKENS=5\nSPEC_METHOD=mtp\nLIMIT_MM='{\"image\":1,\"video\":0}'\n"
    )
    expect = {
        "MAX_NUM_SEQS": "2", "MAX_MODEL_LEN": "200000", "GLM53_MIXED_PREFILL_CHUNK": "512",
        "CG_ESTIMATE": "0", "DFLASH_TOKENS": "5", "SPEC_METHOD": "mtp", "LIMIT_MM": '{"image":1,"video":0}',
    }
    with tempfile.TemporaryDirectory() as raw_tmp:
        h = Harness(Path(raw_tmp))
        for bash in bashes():
            parent, child = h.resolve(bash, dotenv, keys, {})
            assert parent == expect, (bash, parent)
            assert child == expect, (bash, child)
            print(f"  ok   .env beats script defaults with no caller export under {bash}")


def test_empty_caller_export_does_not_override() -> None:
    """`MAX_NUM_SEQS= ./start.sh` keeps the .env value: same [ -n ] rule as the
    allowlist this replaces (DFLASH_DRAFT_TP treats empty as meaningful, and
    an empty export is the usual accident of `VAR= cmd` typos)."""
    keys = ["MAX_NUM_SEQS", "DFLASH_DRAFT_TP", "CG_ESTIMATE"]
    dotenv = "MAX_NUM_SEQS=2\nDFLASH_DRAFT_TP=2\nCG_ESTIMATE=0\n"
    with tempfile.TemporaryDirectory() as raw_tmp:
        h = Harness(Path(raw_tmp))
        for bash in bashes():
            parent, _child = h.resolve(bash, dotenv, keys, {"MAX_NUM_SEQS": "", "DFLASH_DRAFT_TP": "", "CG_ESTIMATE": "1"})
            assert parent == {"MAX_NUM_SEQS": "2", "DFLASH_DRAFT_TP": "2", "CG_ESTIMATE": "1"}, (bash, parent)
            print(f"  ok   empty caller export does not override .env under {bash}")


def test_readonly_exports_and_odd_env_lines() -> None:
    """SHELLOPTS in the environment is imported readonly+exported by bash
    (`declare -rx`); an `eval "$(export -p)"` replay dies on it under set -e.
    The .env-key rule never replays it. The .env itself carries every line
    shape the parser must accept."""
    dotenv = (
        "# GLM-5.3-Flash EXL3 -- comment lines are not keys\n"
        "\n"
        "   \n"
        "# MAX_NUM_SEQS=99\n"
        "MAX_NUM_SEQS=2\n"
        "export SPEC_METHOD=mtp\n"
        "  CG_ESTIMATE=0\n"
        "DFLASH_TOKENS=5\r\n"
        "GLM53_MIXED_PREFILL_CHUNK=512 # trailing comment\n"
        "LIMIT_MM='{\"image\":4,\"video\":1}'\n"
        "MAX_MODEL_LEN=\n"
        "MODEL_REVISION+=-suffix\n"
        "SCRIPT_DIR=/nowhere\n"
        # assignment-looking text inside a heredoc: the lexical scanner picks
        # SHELLOPTS up, and the readonly filter must then refuse to replay it
        # (export "SHELLOPTS=..." would abort the launcher under set -e).
        "cat >/dev/null <<'NOTES'\n"
        "SHELLOPTS=ignored\n"
        "NOTES\n"
    )
    keys = ["MAX_NUM_SEQS", "SPEC_METHOD", "CG_ESTIMATE", "DFLASH_TOKENS", "GLM53_MIXED_PREFILL_CHUNK", "LIMIT_MM", "MAX_MODEL_LEN", "MODEL_REVISION", "SCRIPT_DIR"]
    caller = {
        "MAX_NUM_SEQS": "4", "SPEC_METHOD": "dflash", "CG_ESTIMATE": "1", "DFLASH_TOKENS": "7",
        "GLM53_MIXED_PREFILL_CHUNK": "skip", "LIMIT_MM": '{"image":9,"video":2}', "MAX_MODEL_LEN": "300000",
        "MODEL_REVISION": "abc123",
        "SHELLOPTS": "braceexpand:hashall:interactive-comments",
    }
    with tempfile.TemporaryDirectory() as raw_tmp:
        h = Harness(Path(raw_tmp))
        for bash in bashes():
            parent, child = h.resolve(bash, dotenv, keys, caller)
            want = {k: caller[k] for k in keys if k != "SCRIPT_DIR"}
            # SCRIPT_DIR is a launcher-internal (non-exported) variable: a .env
            # that assigns it is sourced as before, but the launcher's own
            # pre-source value must never be captured and replayed over it.
            want["SCRIPT_DIR"] = "/nowhere"
            assert parent == want, (bash, parent)
            assert {k: child[k] for k in want if k != "SCRIPT_DIR"} == {k: want[k] for k in want if k != "SCRIPT_DIR"}, (bash, child)
            # and with no caller export the odd lines source as bash sources them
            parent, _ = h.resolve(bash, dotenv, keys, {"SHELLOPTS": "braceexpand:hashall"})
            assert parent["MAX_NUM_SEQS"] == "2" and parent["SPEC_METHOD"] == "mtp" and parent["CG_ESTIMATE"] == "0"
            assert parent["DFLASH_TOKENS"] == "5\r", "CRLF line: carriage return stays in the value (unchanged; no normalisation)"
            assert parent["LIMIT_MM"] == '{"image":4,"video":1}'
            assert parent["MAX_MODEL_LEN"] == "1000000", "empty .env value falls through to the launcher default"
            assert parent["MODEL_REVISION"].endswith("-suffix"), "KEY+= line is sourced as an append"
            assert parent["SCRIPT_DIR"] == "/nowhere"
            print(f"  ok   readonly SHELLOPTS export, internal SCRIPT_DIR not replayed, comment/blank/indented/export/+=/CRLF/JSON .env lines under {bash}")
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
    test_no_per_knob_allowlist_remains()
    test_max_num_seqs_inline_override_wins()
    test_every_env_example_key_survives_a_caller_export()
    test_env_still_beats_script_defaults_without_caller_exports()
    test_empty_caller_export_does_not_override()
    test_readonly_exports_and_odd_env_lines()
    test_indexer_workspace_caller_capture_is_setness_aware()
    test_spinwait_caller_capture_is_setness_aware()
    print("start.sh caller override regression OK")
