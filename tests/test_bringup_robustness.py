#!/usr/bin/env python3
"""Behaviour checks for the bring-up lifecycle lock and health window.

Run the launcher's real function bodies against stubbed docker/ssh commands
and real local flock holders, without containers, network access, or GPUs.

The docker/ssh seam records each call's argv and start_unlocked() records its
own entry, so the lifecycle claims are read off the executed arguments and the
exact configured container names -- never off joined command text.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# One exact-target predicate for both suites: the rank-parity harness applies it
# to its stub log, this fixture to its own recorder (same record shape). A
# container is removed only by an rm whose operand IS that name -- never by a
# word-boundary substring, and never by a kill that leaves it behind.
from test_launcher_rank_parity import removes_container, stops_container  # noqa: E402

# The names this fixture configures the launcher copy with, and the ones it
# asserts on: they come from here, not from start.sh's ${VAR:-default}.
HEAD_CONTAINER = "glm53-exl3-head"
WORKER_CONTAINER = "glm53-exl3-worker"

# The recorders write one call per line, argv elements separated by \037, so the
# assertions read the executed arguments rather than joined command text.
SEP = "\x1f"

# Only ever advisory PID-file text: not this process, not the lock holder.
ADVISORY_PID = "424242"


def _source() -> str:
    return (ROOT / "start.sh").read_text(encoding="utf-8")


def _function(name: str) -> str:
    """Slice a top-level ``name() { ... }`` definition out of start.sh."""
    src = _source()
    match = re.search(rf"^{re.escape(name)}\(\)\s*\{{", src, re.M)
    assert match, f"{name}() is missing from start.sh"
    line_end = src.index("\n", match.end())
    if src[match.start():line_end].rstrip().endswith("}"):
        return src[match.start():line_end] + "\n"  # one-liner (log/warn/die)
    end = src.index("\n}\n", match.end())
    return src[match.start():end + 3]


def _run_script(script: Path, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(cwd), "USER": "glm53", **env}
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, cwd=cwd)


# ------------------------------- lifecycle lock -----------------------------

_LIFECYCLE_HEADER = """\
set -euo pipefail
RECORD="{tmp}/record"
: >"$RECORD"
LOGDIR="{tmp}/logs"
mkdir -p "$LOGDIR"
CLUSTER_LOCK="$LOGDIR/cluster.lock"
CLUSTER_LOCK_PID="$LOGDIR/cluster.lock.pid"
# The bound under test. Fast cases pass 1s so a contended stop does not cost
# 30s; the default is the launcher's shipped CLUSTER_LOCK_WAIT.
CLUSTER_LOCK_WAIT={wait}
CONTAINER_HEAD={head}
CONTAINER_WORKER={worker}
WORKER_SSH=fixture.invalid
NFS_SHARE=0
SEQ=$'\\037'
# One write per call: stop_containers() runs the worker ssh in the background
# while the head docker calls run in the foreground, so a record must never be
# assembled from several writes. Each argv element is its own field, so the
# reader sees the executed arguments of the call.
docker() {{ local IFS="$SEQ"; printf 'docker%s%s\\n' "$SEQ" "$*" >>"$RECORD"; }}
# WORKER_SSH_DELAY models a worker teardown that takes time to finish: the record
# lands when the command ends, exactly as the real ssh's does. The delayed
# restart case arms it to prove the start arm is entered only after the
# backgrounded worker teardown completed (a dropped `wait` lands it after).
worker_ssh() {{ local IFS="$SEQ"; [ -z "${{WORKER_SSH_DELAY:-}}" ] || sleep "$WORKER_SSH_DELAY"; printf 'ssh%s%s\\n' "$SEQ" "$*" >>"$RECORD"; }}
start_unlocked() {{
    # Independent flock attempt: proves the lock is held across the start arm.
    if flock -n "$CLUSTER_LOCK" true 2>/dev/null; then
        printf 'start_unlocked lock_free\\n' >>"$RECORD"
    else
        printf 'start_unlocked lock_held\\n' >>"$RECORD"
    fi
}}
banner() {{ :; }}
validate_numeric_config() {{ :; }}
configure_capture_sizes() {{ :; }}
validate_overlay_artifacts() {{ :; }}
"""


def _shipped_lock_wait() -> int:
    """The launcher's CLUSTER_LOCK_WAIT — the default bound these tests run."""
    match = re.search(r"^CLUSTER_LOCK_WAIT=(\d+)$", _source(), re.M)
    assert match, "CLUSTER_LOCK_WAIT is missing from start.sh"
    return int(match.group(1))


def _lifecycle_script(tmp: Path, command: str, wait_seconds: int) -> str:
    """start.sh's lifecycle dispatch with docker/ssh replaced by recorders."""
    functions = (
        "log", "warn", "die",
        "with_cluster_lock", "with_cluster_lock_for_stop",
        "stop_containers", "stop", "start", "main",
    )
    header = _LIFECYCLE_HEADER.format(
        tmp=tmp, wait=wait_seconds, head=HEAD_CONTAINER, worker=WORKER_CONTAINER
    )
    return header + "".join(_function(name) for name in functions) + f"\nmain {command}\n"


def _run_lifecycle(
    tmp: Path, command: str, wait_seconds: int | None = None, worker_delay: str | None = None
) -> subprocess.CompletedProcess:
    """Run one lifecycle command; default is the shipped lock-wait bound."""
    script = tmp / "lifecycle.sh"
    script.write_text(_lifecycle_script(tmp, command, wait_seconds or _shipped_lock_wait()))
    script.chmod(0o755)
    return _run_script(script, {} if worker_delay is None else {"WORKER_SSH_DELAY": worker_delay}, tmp)


def _hold_lock(lock: Path, seconds: int = 300) -> subprocess.Popen:
    """Take the real kernel lock from a separate process."""
    holder = subprocess.Popen(["flock", "--no-fork", "-x", str(lock), "sleep", str(seconds)])
    for _ in range(100):
        probe = subprocess.run(["flock", "-n", str(lock), "true"], capture_output=True)
        if probe.returncode != 0:
            return holder
        time.sleep(0.05)
    holder.kill()
    holder.wait()
    raise AssertionError("background flock holder never acquired the lock")


def _locked_logs(tmp: Path) -> tuple[Path, Path]:
    logs = tmp / "logs"
    logs.mkdir()
    lock = logs / "cluster.lock"
    lock.touch()
    pid_file = logs / "cluster.lock.pid"
    pid_file.write_text(f"{ADVISORY_PID}\n")
    return lock, pid_file


def _call(line: str) -> list[str]:
    """One recorded call as [tool, argv...]: the recorders write the tool, then
    each argv element, \\037-separated."""
    return line.split(SEP)


def _tool(record: list[str], index: int) -> str:
    """The tool that made the recorded call at `index`."""
    return record[index].split(SEP)[0]


def _removed_indices(record: list[str], name: str) -> list[int]:
    """Indices of record lines that remove exactly `name` (see the shared
    removes_container predicate: kill alone is not a removal)."""
    return [i for i, line in enumerate(record) if removes_container(_call(line), name)]


def _stopped_indices(record: list[str], name: str) -> list[int]:
    """Indices of record lines that stop or remove exactly `name` (kill / rm)."""
    return [i for i, line in enumerate(record) if stops_container(_call(line), name)]


def _assert_both_containers_removed(record: list[str]) -> tuple[list[int], list[int]]:
    """Both ranks are removed: the head by a local docker call, the worker by an
    ssh call (it only exists on the other host). A kill without rm leaves the
    container behind, and a suffixed sibling is a different container, so
    neither satisfies this."""
    head = _removed_indices(record, HEAD_CONTAINER)
    worker = _removed_indices(record, WORKER_CONTAINER)
    assert head and all(_tool(record, i) == "docker" for i in head), record
    assert worker and all(_tool(record, i) == "ssh" for i in worker), record
    return head, worker


def test_lifecycle_commands_refuse_while_the_lock_is_held() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        lock, pid_file = _locked_logs(tmp)
        inode = lock.stat().st_ino
        decoy = subprocess.Popen(["sleep", "300"])
        holder = None
        try:
            pid_file.write_text(f"{decoy.pid}\n")
            holder = _hold_lock(lock)
            for command in ("start", "restart", "stop"):
                result = _run_lifecycle(tmp, command, wait_seconds=1)
                assert result.returncode == 1, (command, result.stdout, result.stderr)
                assert holder.poll() is None, "the lock holder was signalled"
                assert decoy.poll() is None, "the advisory PID was signalled"
                assert lock.stat().st_ino == inode, "a refused command replaced the lock file"
                assert (tmp / "record").read_text() == "", f"{command} touched containers without the lock"
                assert pid_file.read_text() == f"{decoy.pid}\n", f"{command} overwrote the holder's pid"
        finally:
            for process in (holder, decoy):
                if process is not None:
                    process.kill()
                    process.wait()


def test_stop_takes_a_free_lock_and_removes_both_containers() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        _locked_logs(tmp)  # lock free, pid file stale/advisory only
        result = _run_lifecycle(tmp, "stop", wait_seconds=1)
        assert result.returncode == 0, result.stderr
        record = (tmp / "record").read_text().splitlines()
        # Teardown must remove both containers -- the head locally, the worker
        # over ssh -- and must not start anything. The exact command shape (kill
        # first, combined vs separate, redirection) stays implementation detail
        # and is deliberately not pinned.
        _assert_both_containers_removed(record)
        assert not any(line.startswith("start_unlocked") for line in record), record


def test_restart_holds_one_lock_across_stop_and_start() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        _locked_logs(tmp)
        # The worker teardown is deliberately slow: its record can only precede
        # the start arm if stop_containers() waited for the backgrounded ssh, so
        # this is a completion proof and not a scheduling coincidence.
        result = _run_lifecycle(tmp, "restart", wait_seconds=1, worker_delay="0.35")
        assert result.returncode == 0, result.stderr
        record = (tmp / "record").read_text().splitlines()
        # The subject is the lock discipline and the phase order, not removal:
        # the stop work on both ranks is recorded (kill or rm -- removal itself
        # is the stop test's claim), the start arm is entered only after it is
        # over, and it runs with the lock still held (start_unlocked re-tries
        # the flock).
        head = _stopped_indices(record, HEAD_CONTAINER)
        worker = _stopped_indices(record, WORKER_CONTAINER)
        started = [i for i, line in enumerate(record) if line.startswith("start_unlocked")]
        assert head and worker, record
        assert all(_tool(record, i) == "docker" for i in head), record
        assert all(_tool(record, i) == "ssh" for i in worker), record
        assert started == [len(record) - 1], record
        assert max(head + worker) < started[0], record
        assert record[-1] == "start_unlocked lock_held", record


# ------------------------------- health wait --------------------------------

# docker/ssh seam: docker inspect answers from a scripted sequence, so the test
# drives the real health loop without containers, ssh or a GPU.
_DOCKER_SEQUENCE_STUB = """\
docker() {
    case "${1:-}" in
        inspect)
            if [ "${2:-}" = "-f" ]; then
                INSPECTS=$(( $(cat "$TMP_DIR/inspects" 2>/dev/null || printf 0) + 1 ))
                printf '%s' "$INSPECTS" >"$TMP_DIR/inspects"
                if [ "$(sed -n "${INSPECTS}p" "$TMP_DIR/sequence")" = "true" ]; then
                    printf 'true\\n'
                else
                    printf 'false\\n'
                fi
                return 0
            fi
            printf 'running\\n'
            return 0
        ;;
        *) return 0 ;;
    esac
}
"""

# A running container whose inspect exits nonzero (grep -q closing the pipe
# early, docker hiccup): the old `docker inspect ... | grep -q true` pipeline
# turned that into "head container exited".
_DOCKER_NONZERO_EXIT_STUB = """\
docker() {
    case "${1:-}" in
        inspect)
            if [ "${2:-}" = "-f" ]; then
                printf 'true\\n'
                return 141
            fi
            return 0
        ;;
        *) return 0 ;;
    esac
}
"""


def _health_script(tmp: Path, docker_stub: str, ready_timeout: int) -> str:
    preamble = f"""\
set -euo pipefail
TMP_DIR="{tmp}"
READY_TIMEOUT={ready_timeout}
PORT=8888
CONTAINER_HEAD={HEAD_CONTAINER}
CONTAINER_WORKER={WORKER_CONTAINER}
WORKER_SSH=glm53@10.0.0.2
curl() {{ return 1; }}
sleep() {{ :; }}
worker_ssh() {{ printf 'true\\n'; }}
{docker_stub}
"""
    body = "".join(_function(name) for name in ("log", "warn", "wait_for_health"))
    return preamble + body + '\nif wait_for_health; then printf "RESULT=healthy\\n"; else printf "RESULT=unhealthy\\n"; fi\n'


def _run_health(tmp: Path, docker_stub: str, ready_timeout: int) -> subprocess.CompletedProcess:
    script = tmp / "health.sh"
    script.write_text(_health_script(tmp, docker_stub, ready_timeout))
    script.chmod(0o755)
    return _run_script(script, {}, tmp)


def test_health_wait_needs_three_consecutive_head_misses() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        (tmp / "sequence").write_text("false\ntrue\nfalse\nfalse\nfalse\n")
        result = _run_health(tmp, _DOCKER_SEQUENCE_STUB, ready_timeout=120)
        assert "RESULT=unhealthy" in result.stdout, result.stdout
        # The true inspect resets the window, so the failure needs three more
        # misses: five inspects, not three.
        assert (tmp / "inspects").read_text() == "5", result.stdout


def test_health_wait_does_not_report_a_running_head_as_dead() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        delayed_health = """
curl() {
    local calls=$(( $(cat "$TMP_DIR/curls" 2>/dev/null || printf 0) + 1 ))
    printf '%s' "$calls" >"$TMP_DIR/curls"
    [ "$calls" -ge 6 ]
}
"""
        result = _run_health(tmp, _DOCKER_NONZERO_EXIT_STUB + delayed_health, ready_timeout=120)
        assert "RESULT=healthy" in result.stdout, result.stdout + result.stderr




if __name__ == "__main__":
    test_lifecycle_commands_refuse_while_the_lock_is_held()
    test_stop_takes_a_free_lock_and_removes_both_containers()
    test_restart_holds_one_lock_across_stop_and_start()
    test_health_wait_needs_three_consecutive_head_misses()
    test_health_wait_does_not_report_a_running_head_as_dead()
    print("start.sh lifecycle and health behavior OK")
