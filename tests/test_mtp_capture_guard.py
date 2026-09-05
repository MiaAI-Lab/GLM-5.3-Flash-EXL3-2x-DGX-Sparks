#!/usr/bin/env python3
"""Stubbed dry runs for the MAX_NUM_SEQS / MTP capture-list guard (issue #88).

On this image vLLM's profile_cudagraph_memory() builds a throwaway KV config
with num_blocks = the largest CUDA-graph capture size and does not pass
is_profiling=True, so the Mamba guard raises
`max_num_seqs (16) exceeds available Mamba cache blocks (12)` in EngineCore
init, after weight load. The launcher's non-DFlash list is `1 2 3 4 6 8 12`,
so `SPEC_METHOD=mtp MAX_NUM_SEQS=16 ./start.sh restart` stopped the running
DFlash server and then failed to replace it.

These tests drive the REAL start.sh under bash from an allow-listed
environment with docker / ssh / scp / rsync / curl / ip / nvidia-smi stubbed
first on PATH (each stub records its argv and exits 0; nothing talks to a
host; PATH is the stubs plus /usr/bin:/bin only) and a .env copied from
.env.example, then assert:

  1. `restart` (and `start`) with SPEC_METHOD=mtp MAX_NUM_SEQS=16 exits 2
     with ZERO docker / ssh calls -- before `stop` -- and the message names
     the vLLM error and the fix. Same for 13, for SPEC_METHOD=none and for
     CG_ESTIMATE=0 (not the trigger).
  2. Controls pass the gate and reach `stop` on BOTH ranks (docker rm -f on
     the head, `docker rm -f` over ssh on the worker): the stock .env, mtp/12,
     mtp/0012, dflash/16, mtp/16 with ENFORCE_EAGER=1, and mtp/16 with the
     caller's own --cudagraph-capture-sizes in EXTRA_ARGS (not parsed, not
     gated). The stubbed preflight then fails, which is fine.
  3. The generated argv is unchanged: EXTRA_ARGS still ends with
     `--cudagraph-capture-sizes 1 2 3 4 6 8 12` (mtp) / `1 2 4 8 16 24 32`
     (dflash), and is untouched under ENFORCE_EAGER=1 or a caller list.
  4. Static: the guard lives inside the numeric-config sentinels, is called
     from validate_numeric_config, and main() runs that before
     `restart) stop; start`.

Run:  python3 tests/test_mtp_capture_guard.py   (or pytest)
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
START = ROOT / "start.sh"

FAILURES: list[str] = []
SEP = "\x1f"
HOST_TOOLS = ("docker", "ssh", "scp", "rsync", "curl", "ip", "nvidia-smi")
MTP_LIST = "1 2 3 4 6 8 12"
DFLASH_LIST = "1 2 4 8 16 24 32"

STUB = """#!/usr/bin/env bash
# Records every invocation; never touches a host.
{ printf '%s\\x1f' "$(basename "$0")" "$@"; printf '\\n'; } >> "$GLM53_STUB_LOG"
case "$(basename "$0")" in
    ip) printf 'inet %s/24\\n' "${GLM53_STUB_HEAD_IP:-10.0.0.1}" ;;
esac
exit 0
"""


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


class Harness:
    """A throwaway copy of the launcher checkout plus a stub PATH."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.repo.mkdir()
        shutil.copy2(START, self.repo / "start.sh")
        shutil.copy2(ROOT / ".env.example", self.repo / ".env.example")
        (self.repo / ".env").write_text((ROOT / ".env.example").read_text())
        for sub in ("overlay", "files", "ablit", "scripts"):
            if (ROOT / sub).is_dir():
                shutil.copytree(ROOT / sub, self.repo / sub)
        self.home = tmp / "home"
        self.home.mkdir()
        self.bin = tmp / "bin"
        self.bin.mkdir()
        for tool in HOST_TOOLS:
            p = self.bin / tool
            p.write_text(STUB)
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = tmp / "calls.log"
        text = (self.repo / "start.sh").read_text()
        assert text.rstrip().endswith('\nmain "$@"'), 'start.sh must end with main "$@"'
        (self.repo / "start.fn.sh").write_text(text.rstrip()[: -len('main "$@"')] + '"$@"\n')

    def env(self, **extra: str) -> dict[str, str]:
        env = {
            "PATH": f"{self.bin}{os.pathsep}/usr/bin:/bin",
            "HOME": str(self.home),
            "USER": "glm53-mtp-guard",
            "LC_ALL": "C",
            "TERM": "dumb",
            "GLM53_STUB_LOG": str(self.log),
        }
        env.update(extra)
        return env

    def calls(self) -> list[list[str]]:
        if not self.log.exists():
            return []
        out = []
        for line in self.log.read_text().splitlines():
            argv = line.split(SEP)
            if argv and argv[-1] == "":
                argv.pop()
            out.append(argv)
        return out

    def host_touching_calls(self) -> list[list[str]]:
        return [c for c in self.calls() if c and c[0] in ("docker", "ssh", "scp", "rsync")]

    def run(self, *argv: str, entry: str = "start.sh", **extra: str) -> subprocess.CompletedProcess[str]:
        if self.log.exists():
            self.log.unlink()
        return subprocess.run(
            ["bash", f"./{entry}", *argv],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=False,
            env=self.env(**extra),
        )


def fails_closed(h: Harness, label: str, cmd: str = "restart", **env: str) -> None:
    r = h.run(cmd, **env)
    calls = h.host_touching_calls()
    err = r.stderr.strip()
    named = (
        "Mamba cache blocks (12)" in err
        and "issue #88" in err
        and "at most 12" in err
        and "ENFORCE_EAGER=1" in err
        and f"MAX_NUM_SEQS={int(env['MAX_NUM_SEQS'])} exceeds" in err
    )
    check(
        r.returncode == 2 and not calls and named,
        f"{label}: rc={r.returncode}, host-touching calls={len(calls)}, message names the vLLM error + fix={named} ({err[-120:]!r})",
    )


def reaches_stop(h: Harness, label: str, **env: str) -> None:
    r = h.run("restart", **env)
    calls = h.host_touching_calls()
    head_rm = any(c[:3] == ["docker", "rm", "-f"] for c in calls)
    worker_rm = any(c[0] == "ssh" and "docker rm -f" in c[-1] for c in calls)
    last = (r.stderr.strip().splitlines() or [""])[-1]
    check(
        head_rm and worker_rm and "issue #88" not in r.stderr,
        f"{label} (head rm={head_rm} worker rm={worker_rm}; later rc={r.returncode} is the stubbed preflight: {last[:90]!r})",
    )


def extra_args(h: Harness, **env: str) -> str:
    r = h.run("eval", 'printf "%s" "${EXTRA_ARGS-}"', entry="start.fn.sh", **env)
    assert r.returncode == 0, r.stderr[-400:]
    assert not h.host_touching_calls()
    return r.stdout


def part_static() -> None:
    print("Static: guard placement")
    text = START.read_text()
    begin = text.index("# GLM53 numeric config guard (begin)")
    end = text.index("# GLM53 numeric config guard (end)")
    guard = text[begin:end]
    check(
        "_glm53_validate_cudagraph_capture_envelope() {" in guard
        and "_glm53_validate_cudagraph_capture_envelope || return" in guard,
        "S1 the envelope check is defined inside the numeric-config sentinels and called from validate_numeric_config",
    )
    check(
        guard.index("_glm53_canonical_positive_int MAX_NUM_SEQS") < guard.index("_glm53_validate_cudagraph_capture_envelope || return"),
        "S1 it runs after MAX_NUM_SEQS is canonicalised (leading zeros stripped, range-checked)",
    )
    main_at = text.index("main() {")
    check(
        text.index("start|restart) validate_numeric_config", main_at) < text.index("restart)  stop; start", main_at),
        "S2 main() runs validate_numeric_config before `restart) stop; start`",
    )
    check(
        f'GLM53_CG_CAPTURE_SIZES="{MTP_LIST}"' in text
        and f'GLM53_CG_CAPTURE_SIZES="{DFLASH_LIST}"' in text
        and '--cudagraph-capture-sizes $GLM53_CG_CAPTURE_SIZES"' in text,
        "S3 the list the guard reads is the list appended to EXTRA_ARGS (one variable, no second constant)",
    )


def part_dry_runs(h: Harness) -> None:
    print("Dry runs: restart fails closed before stop / controls reach stop")
    reaches_stop(h, "C0 control: stock .env (dflash, MAX_NUM_SEQS=4) passes the gate and reaches stop on both ranks")
    reaches_stop(h, "C1 control: SPEC_METHOD=mtp MAX_NUM_SEQS=12 (the envelope) reaches stop", SPEC_METHOD="mtp", MAX_NUM_SEQS="12")
    reaches_stop(h, "C2 control: SPEC_METHOD=mtp MAX_NUM_SEQS=0012 (canonical 12) reaches stop", SPEC_METHOD="mtp", MAX_NUM_SEQS="0012")
    reaches_stop(h, "C3 control: SPEC_METHOD=dflash MAX_NUM_SEQS=16 is not gated (32-list, unmeasured)", SPEC_METHOD="dflash", MAX_NUM_SEQS="16")
    reaches_stop(h, "C4 control: SPEC_METHOD=mtp MAX_NUM_SEQS=16 ENFORCE_EAGER=1 (no graphs, no profiling pass)", SPEC_METHOD="mtp", MAX_NUM_SEQS="16", ENFORCE_EAGER="1")
    reaches_stop(
        h, "C5 control: SPEC_METHOD=mtp MAX_NUM_SEQS=16 with the caller's own --cudagraph-capture-sizes in EXTRA_ARGS (not parsed, not gated)",
        SPEC_METHOD="mtp", MAX_NUM_SEQS="16", EXTRA_ARGS="--cudagraph-capture-sizes 1 2 4 8 16",
    )
    fails_closed(h, "F1 restart SPEC_METHOD=mtp MAX_NUM_SEQS=16 (issue #88 repro) exits 2 with nothing stopped", SPEC_METHOD="mtp", MAX_NUM_SEQS="16")
    fails_closed(h, "F2 start   SPEC_METHOD=mtp MAX_NUM_SEQS=16 exits 2 with zero host calls", cmd="start", SPEC_METHOD="mtp", MAX_NUM_SEQS="16")
    fails_closed(h, "F3 restart SPEC_METHOD=mtp MAX_NUM_SEQS=13 (one above the envelope)", SPEC_METHOD="mtp", MAX_NUM_SEQS="13")
    fails_closed(h, "F4 restart SPEC_METHOD=mtp MAX_NUM_SEQS=0016 (leading zeros canonicalised first)", SPEC_METHOD="mtp", MAX_NUM_SEQS="0016")
    fails_closed(h, "F5 restart SPEC_METHOD=none MAX_NUM_SEQS=16 (same launcher list, same vLLM path)", SPEC_METHOD="none", MAX_NUM_SEQS="16")
    fails_closed(h, "F6 restart SPEC_METHOD=mtp MAX_NUM_SEQS=16 CG_ESTIMATE=0 (CG_ESTIMATE is not the trigger)", SPEC_METHOD="mtp", MAX_NUM_SEQS="16", CG_ESTIMATE="0")
    fails_closed(h, "F7 restart SPEC_METHOD=mtp MAX_NUM_SEQS=16 ENFORCE_EAGER=0 (explicit graphs)", SPEC_METHOD="mtp", MAX_NUM_SEQS="16", ENFORCE_EAGER="0")

    print("Generated argv unchanged")
    mtp = extra_args(h, SPEC_METHOD="mtp")
    dfl = extra_args(h, SPEC_METHOD="dflash")
    check(mtp.endswith(f"--cudagraph-capture-sizes {MTP_LIST}"), f"G1 mtp EXTRA_ARGS still ends with the MTP list ({mtp!r})")
    check(dfl.endswith(f"--cudagraph-capture-sizes {DFLASH_LIST}"), f"G2 dflash EXTRA_ARGS still ends with the DFlash list ({dfl!r})")
    eager = extra_args(h, SPEC_METHOD="mtp", ENFORCE_EAGER="1")
    check("cudagraph-capture-sizes" not in eager, f"G3 ENFORCE_EAGER=1 generates no list ({eager!r})")
    own = extra_args(h, SPEC_METHOD="mtp", EXTRA_ARGS="--cudagraph-capture-sizes 1 2 4 8 16 --foo bar")
    check(own == "--cudagraph-capture-sizes 1 2 4 8 16 --foo bar", f"G4 a caller list is passed through untouched ({own!r})")
    r = h.run("eval", 'printf "%s|%s" "$GLM53_CG_CAPTURE_SIZES" "$MAX_NUM_SEQS"', entry="start.fn.sh", SPEC_METHOD="mtp", MAX_NUM_SEQS="16")
    check(r.stdout == f"{MTP_LIST}|16", f"G5 the prologue records the generated list for the guard ({r.stdout!r})")


def main() -> int:
    part_static()
    with tempfile.TemporaryDirectory() as raw_tmp:
        part_dry_runs(Harness(Path(raw_tmp)))
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("\nmtp capture guard: PASS")
    return 0


def test_mtp_capture_guard() -> None:
    assert main() == 0, FAILURES


if __name__ == "__main__":
    raise SystemExit(main())
