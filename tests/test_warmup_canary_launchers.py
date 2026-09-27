#!/usr/bin/env python3
"""post_ready_warmup status mapping in start.sh / start-tp3.sh / start-tp4.sh.

The shipped function text is extracted from each launcher and executed under a
harness with a stub scripts/boot-shape-warmup.sh, so the exit-status mapping is
exercised through real bash — no cluster, docker, ssh, GPU or network.
Properties pinned:

  rc 0      warmup fine: nothing reported, the start continues
  rc 1      ordinary incomplete warmup: one WARN, the start continues (the
            pre-canary behaviour; it must not become fatal)
  rc 3      degenerate engine: collect_failure_logs + the launcher's own
            teardown primitive (stop_containers in start.sh, stop in
            start-tp3.sh / start-tp4.sh) + die, in that order, and the shell
            stops where the function was called — READY is never reached
  rc 3, teardown failing  still fatal, still no READY: a best-effort teardown
            that reports failure is named as failed instead of being presented
            as a stopped cluster, and never becomes success
  rc other  any other nonzero status is a WARN, not a fatal verdict
  skip      GLM53_BOOT_SHAPE_WARMUP=0 never invokes the sweep

Nothing here asserts message wording for its own sake: the observable state is
the exit status, whether the shell reached its next statement, and the ordered
call log of the stub warmup / log collector / teardown primitive.

Run:  python3 tests/test_warmup_canary_launchers.py   (or pytest)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LAUNCHERS = (("start.sh", "TP=2"), ("start-tp3.sh", "TP=3"), ("start-tp4.sh", "TP=4"))

FAILURES: list[str] = []

STUB_WARMUP = '''#!/usr/bin/env bash
# Stub for scripts/boot-shape-warmup.sh: records that it ran, exits STUB_RC.
printf 'rc=%s\\n' "${STUB_RC:-0}" >> "$STUB_LOG"
exit "${STUB_RC:-0}"
'''

HARNESS = '''#!/usr/bin/env bash
# Runs the launcher's own post_ready_warmup() text against stubbed surroundings.
# The sweep, the failure-log collector and the launcher's teardown primitive
# (stop_containers on TP=2, stop on TP=3/4) each append one line to $STUB_LOG,
# so both "did it run" and the order it ran in are observable.
set -euo pipefail
log()  { printf 'log: %s\\n' "$*"; }
warn() { printf 'warn: %s\\n' "$*" >&2; }
die()  { printf 'die: %s\\n' "$*" >&2; exit 1; }
collect_failure_logs() {
    printf 'collect=%s\\n' "$*" >> "$STUB_LOG"
    printf 'collect: %s\\n' "$*"
}
stop_containers() { teardown stop_containers; }
stop()            { teardown stop; }
teardown() {
    printf 'teardown=%s\\n' "$1" >> "$STUB_LOG"
    printf 'teardown: %s\\n' "$1"
    return "${STUB_STOP_RC:-0}"
}

SCRIPT_DIR="$HARNESS_DIR"
MAX_NUM_SEQS=4
GLM53_WARMUP_REQ_TIMEOUT=240
DFLASH_TOKENS=7
TRITON_HOST_CACHE=/tmp/triton
VLLM_API_KEY=""
PORT=8123
SERVED_MODEL_NAME=glm53-test
LOGDIR="$HARNESS_DIR/logs"
export STUB_LOG="$HARNESS_DIR/calls.txt"

__FUNCTION__

post_ready_warmup
printf 'returned\\n'
'''


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


def function_text(launcher: Path) -> str:
    """The launcher's own post_ready_warmup(), verbatim."""
    lines = launcher.read_text().splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith("post_ready_warmup() {"))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip() == "}")
    body = "".join(lines[start:end + 1])
    assert "boot-shape-warmup.sh" in body, f"{launcher}: extraction grabbed the wrong block"
    return body


def harness_for(launcher: Path, tmp: Path) -> Path:
    """Stub tree for one launcher: scripts/boot-shape-warmup.sh + harness.sh."""
    home = tmp / launcher.stem
    (home / "scripts").mkdir(parents=True, exist_ok=True)
    stub = home / "scripts" / "boot-shape-warmup.sh"
    stub.write_text(STUB_WARMUP)
    stub.chmod(0o755)
    harness = home / "harness.sh"
    harness.write_text(
        HARNESS.replace("__FUNCTION__", function_text(ROOT / launcher).rstrip("\n"))
        .replace("$HARNESS_DIR", str(home))
    )
    harness.chmod(0o755)
    return harness


def call_log(home: Path) -> list[tuple[str, str]]:
    """The stub calls of the last run, in the order they happened."""
    log = home / "calls.txt"
    if not log.exists():
        return []
    return [tuple(line.split("=", 1)) for line in log.read_text().splitlines()]


def calls(home: Path) -> dict[str, str]:
    return dict(call_log(home))


def run(harness: Path, stub_rc: str, extra_env: dict[str, str] | None = None) -> tuple[
    subprocess.CompletedProcess[str], dict[str, str]
]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(harness.parent),
        "LC_ALL": "C",
        "TERM": "dumb",
        "STUB_RC": stub_rc,
        **(extra_env or {}),
    }
    (harness.parent / "calls.txt").unlink(missing_ok=True)
    done = subprocess.run(["bash", str(harness)], env=env, text=True,
                          capture_output=True, timeout=60)
    return done, calls(harness.parent)


def part_mapping(launcher: str, tp: str, tmp: Path) -> None:
    print(f"{launcher}: rc -> fatal/warn mapping")
    harness = harness_for(Path(launcher), tmp)
    teardown = "stop_containers" if launcher == "start.sh" else "stop"

    def steps() -> list[str]:
        return [key for key, _ in call_log(harness.parent)]

    done, call = run(harness, "0")
    check(done.returncode == 0 and "returned" in done.stdout,
          f"{launcher} rc0: start continues (rc={done.returncode})")
    check(done.stderr == "", f"{launcher} rc0: nothing reported (stderr={done.stderr.strip()[:120]!r})")
    check(call.get("rc") == "0", f"{launcher} rc0: warmup was invoked (call={call})")
    check(steps() == ["rc"], f"{launcher} rc0: no teardown on a healthy sweep (steps={steps()})")

    done, _ = run(harness, "1")
    check(done.returncode == 0 and "returned" in done.stdout,
          f"{launcher} rc1: still nonfatal (rc={done.returncode})")
    check(f"may JIT mid-serve on {tp}" in done.stderr and "die:" not in done.stderr,
          f"{launcher} rc1: one warning, no fatal (stderr={done.stderr.strip()[:160]!r})")
    check(steps() == ["rc"], f"{launcher} rc1: no failure-log collection or teardown (steps={steps()})")

    done, _ = run(harness, "2")
    check(done.returncode == 0 and "warn:" in done.stderr and "die:" not in done.stderr,
          f"{launcher} rc2: unexpected nonzero stays a warning (stderr={done.stderr.strip()[:160]!r})")
    check(steps() == ["rc"], f"{launcher} rc2: no failure-log collection or teardown (steps={steps()})")

    done, call = run(harness, "3")
    check(done.returncode == 1, f"{launcher} rc3: fatal (rc={done.returncode})")
    check("die: engine failed the post-ready correctness canary" in done.stderr
          and "GLM53_WARMUP_CANARY=0 skips the check" in done.stderr,
          f"{launcher} rc3: die names the canary (stderr={done.stderr.strip()[:200]!r})")
    check(steps() == ["rc", "collect", "teardown"],
          f"{launcher} rc3: sweep -> failure logs -> teardown, in that order (steps={steps()})")
    check(call.get("teardown") == teardown,
          f"{launcher} rc3: uses its own lifecycle primitive ({teardown}), call={call})")
    check("returned" not in done.stdout and "may JIT mid-serve" not in done.stderr,
          f"{launcher} rc3: the shell stops where the warmup was called — no READY path")

    # A teardown that cannot stop the cluster is not success: the start still
    # fails, READY is still never reached, and the failure is not hidden.
    failed, call = run(harness, "3", {"STUB_STOP_RC": "1"})
    check(failed.returncode == 1 and "returned" not in failed.stdout,
          f"{launcher} rc3+teardown failure: still fatal, still no READY (rc={failed.returncode})")
    check(steps() == ["rc", "collect", "teardown"],
          f"{launcher} rc3+teardown failure: logs first, teardown still attempted (steps={steps()})")
    check(call.get("teardown") == teardown and "rc=1" in failed.stderr,
          f"{launcher} rc3+teardown failure: the failed teardown is named, not reported as done "
          f"(stderr={failed.stderr.strip()[:200]!r})")
    check("stopped" not in failed.stderr.lower(),
          f"{launcher} rc3+teardown failure: the message does not claim the cluster is stopped "
          f"(stderr={failed.stderr.strip()[:200]!r})")

    done, call = run(harness, "130")
    check(done.returncode == 0 and steps() == ["rc"],
          f"{launcher} rc130: an interrupted sweep stays a warning, no teardown (steps={steps()})")


def part_skip(launcher: str, tmp: Path) -> None:
    print(f"{launcher}: GLM53_BOOT_SHAPE_WARMUP=0")
    harness = harness_for(Path(launcher), tmp)

    done, call = run(harness, "0", {"GLM53_BOOT_SHAPE_WARMUP": "0"})
    check(done.returncode == 0 and "skipped" in done.stdout and call == {},
          f"{launcher}: GLM53_BOOT_SHAPE_WARMUP=0 never invokes the sweep "
          f"(stdout={done.stdout.strip()[:120]!r}, call={call})")


def main() -> int:
    for launcher, tp in LAUNCHERS:
        if not (ROOT / launcher).is_file():
            raise SystemExit(f"missing {launcher}")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        for launcher, tp in LAUNCHERS:
            part_mapping(launcher, tp, tmp)
            part_skip(launcher, tmp)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("warmup canary launcher mapping OK (stub warmup, no cluster)")
    return 0


def test_warmup_canary_launchers() -> None:
    """pytest entry point (the script form above is what the README documents)."""
    FAILURES.clear()
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
