#!/usr/bin/env python3
"""Static + dry-run launcher regression for the kv-offload store tier knobs.

A  Knob guard matrix (numeric-config block lifted by its sentinels): the
   three bool knobs are exactly 0/1; GLM53_KV_OFFLOAD_RESTORE=1 is REFUSED
   (stage 1 is store-only); CPU_GB is a canonical positive int <= 64; the
   store dir must be absolute with no quotes/whitespace; KEEP_BOUNDARIES is
   an integer 0..1024. All validated only when the tier is on, except the
   bool shape and the restore refusal (always).
B  Pre-stop gate: `./start.sh restart` with a bad knob, or with the scope or
   store overlay missing/truncated/mis-pointed, exits 2 with ZERO docker or
   ssh calls (the healthy pair is never stopped); a valid configuration
   reaches the stop path.
C  knob=0 byte-parity: the generated inner scripts gate --kv-transfer-config
   on the env, both docker runs replay GLM53_KV_OFFLOAD=0, and NO store
   mount or store mkdir appears anywhere.
D  knob=1 both-rank parity: every GLM53_KV_OFFLOAD* env is present and
   identical on both ranks (container dir path, CPU GiB, restore, drafter,
   keep-boundaries), the store dir is mounted rw on both ranks from the same
   host path, both overlays ride the scp -> /tmp -> mount chain, and
   GLM53_OVERLAY_ORDER pins scope BEFORE store_local in BOTH inner scripts.
   The parity comparator itself is exercised with a synthetic one-rank
   mismatch. The inner scripts' JSON builder is executed and its shape
   checked (OffloadingConnector / TieringOffloadingSpec / cpu_bytes /
   blocks_per_chunk=1 / NO secondary tiers — Codex OFFLOAD1 finding 3).

Everything drives the shipped start.sh under bash from an allow-listed
environment with docker/ssh/scp/rsync/curl/ip/nvidia-smi stubbed first on
PATH. Nothing talks to a real host.

Run:  python3 tests/test_launcher_kv_offload.py   (or pytest)
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
START = ROOT / "start.sh"

FAILURES: list[str] = []
SEP = "\x1f"

KNOBS = (
    "GLM53_KV_OFFLOAD",
    "GLM53_KV_OFFLOAD_DIR",
    "GLM53_KV_OFFLOAD_CPU_GB",
    "GLM53_KV_OFFLOAD_RESTORE",
    "GLM53_KV_OFFLOAD_DRAFTER",
    "GLM53_KV_OFFLOAD_KEEP_BOUNDARIES",
)

STUB = """#!/usr/bin/env bash
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


def source() -> str:
    return START.read_text()


def guard_source() -> str:
    text = source()
    begin = text.index("# GLM53 numeric config guard (begin)")
    end_marker = "# GLM53 numeric config guard (end)"
    return text[begin : text.index(end_marker, begin) + len(end_marker)]


def base_env(**extra: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/"),
        "USER": "glm53-kvo",
        "LC_ALL": "C",
        "TERM": "dumb",
    }
    env.update(extra)
    return env


def run_guard(extra_env: dict[str, str]) -> tuple[int, str]:
    script = (
        guard_source()
        + "\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4\n"
        + "MAX_NUM_BATCHED_TOKENS=1024\n"
        + 'GLM53_KV_OFFLOAD_DIR="${GLM53_KV_OFFLOAD_DIR:-/tmp/kvo}"\n'
        + 'GLM53_KV_OFFLOAD_CPU_GB="${GLM53_KV_OFFLOAD_CPU_GB:-4}"\n'
        + 'GLM53_KV_OFFLOAD_KEEP_BOUNDARIES="${GLM53_KV_OFFLOAD_KEEP_BOUNDARIES:-2}"\n'
        + "validate_numeric_config || exit $?\n"
        + 'echo OK\n'
    )
    r = subprocess.run(
        ["bash", "-c", script], text=True, capture_output=True, env=base_env(**extra_env)
    )
    return r.returncode, (r.stderr or r.stdout).strip()


def part_a() -> None:
    print("Part A: knob guard matrix")
    ok_cases = [
        {},
        {"GLM53_KV_OFFLOAD": "0"},
        {"GLM53_KV_OFFLOAD": "1"},
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_CPU_GB": "64"},
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_KEEP_BOUNDARIES": "0"},
        {"GLM53_KV_OFFLOAD": "0", "GLM53_KV_OFFLOAD_CPU_GB": "junk"},  # gated off
        # Stage 2: restore is accepted iff the store tier is on.
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_RESTORE": "1"},
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_RESTORE": "0"},
    ]
    for env in ok_cases:
        rc, out = run_guard(env)
        check(rc == 0, f"A1 accepted: {env} (rc={rc} {out[:60]!r})")
    bad_cases = [
        {"GLM53_KV_OFFLOAD": ""},
        {"GLM53_KV_OFFLOAD": "2"},
        {"GLM53_KV_OFFLOAD": "yes"},
        {"GLM53_KV_OFFLOAD_RESTORE": "1"},  # refused even with the tier off
        {"GLM53_KV_OFFLOAD_RESTORE": "x"},
        {"GLM53_KV_OFFLOAD_DRAFTER": "01"},
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_CPU_GB": "0"},
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_CPU_GB": "65"},
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_CPU_GB": "4.5"},
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_KEEP_BOUNDARIES": "-1"},
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_KEEP_BOUNDARIES": "1025"},
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_DIR": "relative/path"},
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_DIR": "/tmp/has space"},
        {"GLM53_KV_OFFLOAD": "1", "GLM53_KV_OFFLOAD_DIR": "/tmp/it's"},
    ]
    for env in bad_cases:
        rc, out = run_guard(env)
        named = any(k.split("_")[0] == "GLM53" and k in out for k in env) or "GLM53" in out
        check(rc == 2 and named, f"A2 rejected rc=2 with named error: {env} (rc={rc} {out[:70]!r})")


class Harness:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.repo = tmp / "repo"
        self.repo.mkdir()
        shutil.copy2(START, self.repo / "start.sh")
        shutil.copy2(ROOT / ".env.example", self.repo / ".env.example")
        (self.repo / ".env").write_text((ROOT / ".env.example").read_text())
        for sub in ("overlay", "files", "ablit"):
            if (ROOT / sub).is_dir():
                shutil.copytree(ROOT / sub, self.repo / sub)
        self.home = tmp / "home"
        self.home.mkdir()
        self.bin = tmp / "bin"
        self.bin.mkdir()
        for tool in ("docker", "ssh", "scp", "rsync", "curl", "ip", "nvidia-smi"):
            p = self.bin / tool
            p.write_text(STUB)
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = tmp / "calls.log"
        text = (self.repo / "start.sh").read_text()
        assert text.rstrip().endswith('\nmain "$@"')
        (self.repo / "start.fn.sh").write_text(
            text.rstrip()[: -len('main "$@"')] + '"$@"\n'
        )

    def env(self, **extra: str) -> dict[str, str]:
        return base_env(
            PATH=f"{self.bin}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
            HOME=str(self.home),
            GLM53_STUB_LOG=str(self.log),
            **extra,
        )

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

    def run(self, args: list[str], entry: str = "start.sh", **extra: str):
        if self.log.exists():
            self.log.unlink()
        return subprocess.run(
            ["bash", f"./{entry}", *args],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=False,
            env=self.env(**extra),
        )

    def host_calls(self) -> list[list[str]]:
        return [c for c in self.calls() if c and c[0] in ("docker", "ssh", "scp", "rsync")]


def part_b(h: Harness) -> None:
    print("Part B: pre-stop gate fails closed")
    r = h.run(["restart"], GLM53_KV_OFFLOAD="maybe")
    check(
        r.returncode == 2 and not h.host_calls(),
        f"B1 bad knob: rc=2, zero docker/ssh calls (rc={r.returncode}, calls={len(h.host_calls())})",
    )
    r = h.run(["restart"], GLM53_KV_OFFLOAD_RESTORE="1")
    check(
        r.returncode == 2
        and "requires GLM53_KV_OFFLOAD=1" in r.stderr
        and not h.host_calls(),
        "B2 RESTORE=1 without the store tier refused pre-stop (stage-2 rule)",
    )
    r = h.run(["restart"], GLM53_KV_OFFLOAD="0", GLM53_KV_OFFLOAD_RESTORE="1")
    check(
        r.returncode == 2
        and "requires GLM53_KV_OFFLOAD=1" in r.stderr
        and not h.host_calls(),
        "B2b explicit OFFLOAD=0 + RESTORE=1 refused pre-stop",
    )
    restore_p = h.repo / "overlay" / "patch_kv_offload_restore_g0.py"
    restore_backup = restore_p.read_text()
    restore_p.write_text(restore_backup[: len(restore_backup) // 2])
    r = h.run(["restart"])
    check(
        r.returncode == 2 and not h.host_calls(),
        "B2c truncated restore overlay: rc=2 before any host call",
    )
    restore_p.write_text(restore_backup)
    scope = h.repo / "overlay" / "patch_kv_offload_scope.py"
    backup = scope.read_text()
    scope.unlink()
    r = h.run(["restart"])
    check(
        r.returncode == 2 and not h.host_calls(),
        "B3 missing scope overlay: rc=2 before any host call",
    )
    scope.write_text(backup[: len(backup) // 2])
    r = h.run(["restart"])
    check(
        r.returncode == 2 and not h.host_calls(),
        "B4 truncated scope overlay: rc=2 before any host call",
    )
    scope.write_text(backup)
    store_p = h.repo / "overlay" / "patch_kv_offload_store_local.py"
    store_backup = store_p.read_text()
    store_p.write_text(backup)  # mis-pointed: scope contents under store name
    r = h.run(["restart"])
    check(
        r.returncode == 2 and not h.host_calls(),
        "B5 mis-pointed store overlay (identity string): rc=2 pre-stop",
    )
    store_p.write_text(store_backup)
    r = h.run(["restart"])
    stopped = any(c[:3] == ["docker", "rm", "-f"] or (c[0] == "docker" and "stop" in c) for c in h.host_calls()) or any(
        c[0] == "ssh" for c in h.host_calls()
    )
    check(
        r.returncode != 2 and stopped,
        f"B6 valid config passes the gate and reaches the stop path (rc={r.returncode})",
    )


def _inner_scripts(h: Harness, **env: str) -> tuple[str, str]:
    r = h.run(["eval", "write_inner_scripts"], entry="start.fn.sh", **env)
    assert r.returncode == 0, r.stderr[-800:]
    head = (h.repo / ".glm53-exl3-head.inner.sh").read_text()
    worker = (h.repo / ".glm53-exl3-worker.inner.sh").read_text()
    return head, worker


def _launch(h: Harness, **env: str):
    r = h.run(
        ["eval", "MODEL_DIR=/models/glm DFLASH_MODEL_DIR=/models/dflash; write_inner_scripts; launch_cluster"],
        entry="start.fn.sh",
        **env,
    )
    assert r.returncode == 0, (r.stderr[-1200:], r.stdout[-400:])
    calls = h.calls()
    head_run = next(
        c for c in calls if c[0] == "docker" and c[1] == "run" and "glm53-exl3-head" in c
    )
    worker_cmds = [c[-1] for c in calls if c[0] == "ssh" and "docker run" in c[-1]]
    assert worker_cmds, "no worker docker run captured"
    return calls, head_run, worker_cmds[0]


def _env_of_head(head_run: list[str]) -> dict[str, str]:
    envs: dict[str, str] = {}
    for i, tok in enumerate(head_run):
        if tok == "-e" and i + 1 < len(head_run) and "=" in head_run[i + 1]:
            k, v = head_run[i + 1].split("=", 1)
            envs[k] = v
    return envs


def _env_of_worker(worker_cmd: str) -> dict[str, str]:
    envs: dict[str, str] = {}
    for tok in shlex.split(worker_cmd):
        if "=" in tok and tok.split("=", 1)[0].replace("_", "").isalnum():
            k, v = tok.split("=", 1)
            if k.isupper():
                envs[k] = v
    return envs


def part_c(h: Harness) -> None:
    print("Part C: knob=0 parity (byte-identical serving)")
    head, worker = _inner_scripts(h)
    gate = 'if [ "${GLM53_KV_OFFLOAD:-0}" = "1" ]; then'
    check(
        gate in head and gate in worker,
        "C1 both inner scripts gate --kv-transfer-config on the env knob",
    )
    calls, head_run, worker_cmd = _launch(h)
    henv, wenv = _env_of_head(head_run), _env_of_worker(worker_cmd)
    check(
        henv.get("GLM53_KV_OFFLOAD") == "0" and wenv.get("GLM53_KV_OFFLOAD") == "0",
        "C2 knob replayed as 0 to BOTH ranks",
    )
    # A mount token is "host:/data/glm53-kv-offload" (no '='); the env replay
    # token "GLM53_KV_OFFLOAD_DIR=/data/..." must NOT count as a mount.
    mounts = [
        t for t in head_run if ":/data/glm53-kv-offload" in t and "=" not in t
    ]
    worker_mounts = [
        t
        for t in shlex.split(worker_cmd)
        if ":/data/glm53-kv-offload" in t and "=" not in t
    ]
    check(
        not mounts and not worker_mounts,
        "C3 knob=0: no store mount on either rank",
    )
    check(
        henv.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"
        and wenv.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True",
        "C3b knob=0 keeps the stock PYTORCH_CUDA_ALLOC_CONF on BOTH ranks "
        "(byte-identical container env)",
    )
    mkdirs = [c for c in calls if c[0] == "ssh" and "glm53-kv-offload" in c[-1] and "mkdir" in c[-1]]
    check(not mkdirs, "C4 knob=0: no store mkdir on the worker")


def part_d(h: Harness) -> None:
    print("Part D: knob=1 both-rank parity")
    store_dir = str(h.tmp / "kv-store")
    env = {
        "GLM53_KV_OFFLOAD": "1",
        "GLM53_KV_OFFLOAD_DIR": store_dir,
        "GLM53_KV_OFFLOAD_CPU_GB": "6",
        "GLM53_KV_OFFLOAD_KEEP_BOUNDARIES": "3",
    }
    calls, head_run, worker_cmd = _launch(h, **env)
    henv, wenv = _env_of_head(head_run), _env_of_worker(worker_cmd)
    expected = {
        "GLM53_KV_OFFLOAD": "1",
        "GLM53_KV_OFFLOAD_DIR": "/data/glm53-kv-offload",
        "GLM53_KV_OFFLOAD_CPU_GB": "6",
        "GLM53_KV_OFFLOAD_RESTORE": "0",
        "GLM53_KV_OFFLOAD_DRAFTER": "0",
        "GLM53_KV_OFFLOAD_KEEP_BOUNDARIES": "3",
    }
    for k, v in expected.items():
        check(
            henv.get(k) == v and wenv.get(k) == v,
            f"D1 {k}={v!r} present and identical on BOTH ranks "
            f"(head={henv.get(k)!r} worker={wenv.get(k)!r})",
        )
    # vLLM refuses OffloadingConnector + expandable_segments:True (VMM remap
    # under pinned KV memory; live receipt-window finding): knob=1 must DROP
    # the env on both ranks; knob=0 keeps the stock value (see part_c).
    check(
        "PYTORCH_CUDA_ALLOC_CONF" not in henv
        and "PYTORCH_CUDA_ALLOC_CONF" not in wenv,
        f"D1b knob=1 drops PYTORCH_CUDA_ALLOC_CONF on BOTH ranks "
        f"(head={henv.get('PYTORCH_CUDA_ALLOC_CONF')!r} "
        f"worker={wenv.get('PYTORCH_CUDA_ALLOC_CONF')!r})",
    )
    head_mount = f"{store_dir}:/data/glm53-kv-offload"
    check(
        any(head_mount == t for t in head_run),
        "D2 head mounts the store dir rw at the container path",
    )
    check(
        f"-v '{head_mount}'" in worker_cmd,
        "D3 worker mounts the SAME host dir at the same container path",
    )
    mkdirs = [c for c in calls if c[0] == "ssh" and store_dir in c[-1] and "mkdir" in c[-1]]
    check(bool(mkdirs), "D4 worker store dir mkdir'ed over ssh")

    # scp -> /tmp -> mount chain for both overlays.
    for name in (
        "patch_kv_offload_scope.py",
        "patch_kv_offload_store_local.py",
        "patch_kv_offload_restore_g0.py",
    ):
        scps = [
            c
            for c in calls
            if c[0] == "scp" and any(name in a for a in c) and any(f"/tmp/{name}" in a for a in c)
        ]
        head_m = any(f"/opt/glm53/{name}:ro" in t for t in head_run)
        worker_m = f"-v '/tmp/{name}:/opt/glm53/{name}:ro'" in worker_cmd
        check(
            bool(scps) and head_m and worker_m,
            f"D5 {name}: scp chain + both-rank mounts",
        )

    # Overlay order: scope before store_local, identically in both scripts.
    head, worker = _inner_scripts(h, **env)
    m = re.findall(r"python3 /opt/glm53/(patch_[a-z0-9_]+\.py)", head)
    check(
        m.index("patch_kv_offload_scope.py")
        < m.index("patch_kv_offload_store_local.py")
        < m.index("patch_kv_offload_restore_g0.py"),
        "D6 overlay order pins scope -> store_local -> restore_g0",
    )
    m2 = re.findall(r"python3 /opt/glm53/(patch_[a-z0-9_]+\.py)", worker)
    check(m == m2, "D7 both ranks apply the identical overlay list")

    # Stage 2: RESTORE=1 launch replays the knob identically to both ranks
    # and changes nothing else about the connector JSON.
    env_r = dict(env)
    env_r["GLM53_KV_OFFLOAD_RESTORE"] = "1"
    _, head_run_r, worker_cmd_r = _launch(h, **env_r)
    henv_r, wenv_r = _env_of_head(head_run_r), _env_of_worker(worker_cmd_r)
    check(
        henv_r.get("GLM53_KV_OFFLOAD_RESTORE") == "1"
        and wenv_r.get("GLM53_KV_OFFLOAD_RESTORE") == "1",
        "D11 RESTORE=1 replayed identically to BOTH ranks",
    )

    # The comparator itself must catch a planted one-rank difference.
    planted = dict(wenv)
    planted["GLM53_KV_OFFLOAD_CPU_GB"] = "5"
    mismatch = [k for k in expected if henv.get(k) != planted.get(k)]
    check(
        mismatch == ["GLM53_KV_OFFLOAD_CPU_GB"],
        "D8 synthetic one-rank mismatch is detected by the parity comparison",
    )

    # JSON shape: execute the inner script's builder verbatim.
    snippet = re.search(
        r"python3 -S -c '([^']*GLM53_KV_OFFLOAD_CPU_GB[^']*)'", head, re.S
    )
    check(snippet is not None, "D9 inner script builds the connector JSON via python3")
    if snippet:
        out = subprocess.run(
            [sys.executable, "-S", "-c", snippet.group(1)],
            text=True,
            capture_output=True,
            env={**os.environ, "GLM53_KV_OFFLOAD_CPU_GB": "6"},
        )
        cfg = json.loads(out.stdout.strip())
        extra = cfg["kv_connector_extra_config"]
        check(
            cfg["kv_connector"] == "OffloadingConnector"
            and cfg["kv_role"] == "kv_both"
            and extra["spec_name"] == "TieringOffloadingSpec"
            and extra["cpu_bytes_to_use"] == 6 * (1 << 30)
            and extra["blocks_per_chunk"] == 1
            and extra["offload_prompt_only"] is False
            and "secondary_tiers" not in extra,
            f"D10 connector JSON shape (got {cfg})",
        )


def main() -> int:
    part_a()
    with tempfile.TemporaryDirectory() as td:
        h = Harness(Path(td))
        part_b(h)
        part_c(h)
        part_d(h)
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
