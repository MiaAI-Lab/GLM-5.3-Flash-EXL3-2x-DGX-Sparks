#!/usr/bin/env python3
"""Static + dry-run regression for the two-rank launcher (start.sh).

Hardening asked for by the production-like tester run on PRs #83/#84:

  A  GLM53_APC_RETENTION_INTERVAL_SWA guard -- "" (auto) or 0 pass, anything
     else must be a positive multiple of 3584 no larger than 1,000,000; the
     canonical value is what the ranks receive. Runs on the checkout whose
     launcher actually forwards that knob to the containers (detected from
     the `-e VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA=` line, not guessed).
  B  Pre-stop gate -- `./start.sh restart` with a bad knob, or with ANY
     mounted overlay missing / empty / unparseable / pointed at a different
     overlay, exits 2 BEFORE the first docker or ssh call, so healthy
     containers are never stopped for a launch that cannot succeed.
  C  Overlay order -- one list (GLM53_OVERLAY_ORDER) pinned
     hybrid -> per-group -> fine-grained (-> no-store, where shipped) is
     emitted verbatim into BOTH rank inner scripts.
  D  Rank parity -- for every /opt/glm53 patch the head bind-mounts host
     file S, the worker's mount is fed from /tmp/X and the scp that produced
     /tmp/X read the same S; both ranks receive identical effective values
     for GLM53_APC_RETENTION_INTERVAL, GLM53_APC_RETENTION_INTERVAL_SWA,
     GLM53_FINEGRAINED_APC and GLM53_APC_NO_STORE (launcher names) and the container-side names they
     map to (VLLM_PREFIX_CACHE_RETENTION_INTERVAL[_SWA], GLM53_FINEGRAINED_APC);
     a knob the launcher wires must be PRESENT on both ranks, not merely
     equal. The comparison itself is exercised with a synthetic one-rank
     mismatch so a silent pass cannot hide behind equality.

Everything drives the shipped start.sh under bash from an allow-listed
environment (PATH with docker / ssh / scp / rsync / curl / ip / nvidia-smi
stubbed first, HOME pointed at a temp dir, no BASH_ENV / PYTHONPATH / HF_*
/ GLM53_* / VLLM_* leakage). Nothing talks to a real host. The test is
branch-agnostic: it discovers which prefix-cache overlays this checkout
ships from the `*_PATCH_HOST="${*_PATCH_HOST:-` assignments and reports it.

Run:  python3 tests/test_launcher_rank_parity.py   (or pytest)
"""

from __future__ import annotations

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

BLOCK = 3584
RETENTION_MAX = 1_000_000
SWA = "GLM53_APC_RETENTION_INTERVAL_SWA"
SWA_FORWARD = '-e "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA=$GLM53_APC_RETENTION_INTERVAL_SWA"'
FG = "GLM53_FINEGRAINED_APC"
FG_FORWARD = '-e "GLM53_FINEGRAINED_APC=$GLM53_FINEGRAINED_APC"'
NS = "GLM53_APC_NO_STORE"
NS_FORWARD = '-e "GLM53_APC_NO_STORE=$GLM53_APC_NO_STORE"'

# Launcher knobs the tester asked to see reach both ranks identically, and the
# container-side names the launcher maps them to.
LAUNCHER_KNOBS = ("GLM53_APC_RETENTION_INTERVAL", SWA, FG, NS)
CONTAINER_NAMES = LAUNCHER_KNOBS + (
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA",
)

PINNED = (
    "patch_hybrid_prefix_hit.py",
    "patch_apc_per_group_retention.py",
    "patch_apc_fine_grained_hits.py",
)
# Ships on its own (sampling_params / request / block_pool, no coordinator
# anchors) and, where listed, must follow the three coordinator overlays.
NO_STORE = "patch_apc_no_store.py"
APC_HOST_VARS = {
    "APC_PATCH_HOST": "patch_hybrid_prefix_hit.py",
    "PERGROUP_PATCH_HOST": "patch_apc_per_group_retention.py",
    "FINEHIT_PATCH_HOST": "patch_apc_fine_grained_hits.py",
    "NOSTORE_PATCH_HOST": NO_STORE,
}

SEP = "\x1f"

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


def source() -> str:
    return START.read_text()


def guard_source() -> str:
    """start.sh's numeric-config guard, lifted out by its sentinels (same
    technique as tests/test_numeric_config.py)."""
    text = source()
    begin = text.index("# GLM53 numeric config guard (begin)")
    end_marker = "# GLM53 numeric config guard (end)"
    end = text.index(end_marker, begin) + len(end_marker)
    return text[begin:end]


def base_env(**extra: str) -> dict[str, str]:
    """Allow-listed environment: nothing from the developer/CI shell leaks in
    (no BASH_ENV, PYTHONPATH, HF_HOME, EXTRA_ARGS, SKIP_*, *_PATCH_HOST ...)."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/"),
        "USER": "glm53-parity",
        "LC_ALL": "C",
        "TERM": "dumb",
    }
    env.update(extra)
    return env


def host_vars() -> dict[str, str]:
    """Every `<NAME>_PATCH_HOST="${<NAME>_PATCH_HOST:-$SCRIPT_DIR/overlay/<file>}"`
    assignment in start.sh, from the real assignment, not from substring hits."""
    out: dict[str, str] = {}
    for m in re.finditer(
        r'^([A-Z_]+_PATCH_HOST)="\$\{\1:-\$SCRIPT_DIR/overlay/([A-Za-z0-9_.]+)\}"$', source(), re.M
    ):
        out[m.group(1)] = m.group(2)
    return out


def shipped_apc_vars() -> dict[str, str]:
    return {v: b for v, b in host_vars().items() if v in APC_HOST_VARS}


def wires_swa() -> bool:
    return SWA_FORWARD in source()


def wires_fg() -> bool:
    return FG_FORWARD in source()


def wires_ns() -> bool:
    return NS_FORWARD in source()


# ------------------------------------------------------------------ part A --


def run_retention_guard(value: str | None) -> tuple[int, str, str]:
    script = (
        guard_source()
        + "\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4\n"
        + "MAX_NUM_BATCHED_TOKENS=1024\n"
        + "GLM53_INDEXER_WORKSPACE=stock; GLM53_SPINWAIT_MS=stock\n"
        + f"{FG}=1\n"
        + "validate_numeric_config || exit $?\n"
        + f'printf "%s\\n" "${{{SWA}-unset}}"\n'
    )
    env = base_env()
    if value is not None:
        env[SWA] = value
    r = subprocess.run(["bash", "-c", script], text=True, capture_output=True, env=env)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def part_a() -> None:
    print("Part A: GLM53_APC_RETENTION_INTERVAL_SWA guard (numeric config block)")
    if not wires_swa():
        check(
            f"_glm53_validate_retention_interval {SWA}" not in guard_source(),
            "A0 this launcher does not forward the SWA knob and does not pretend to validate it",
        )
        print("  skip A1-A3 (knob not forwarded by this checkout; validate-what-you-forward)")
        return
    text = guard_source()
    check(
        f"_glm53_validate_retention_interval {SWA}" in text,
        "A1 the forwarded SWA knob is validated inside the numeric config guard",
    )
    check(
        f"GLM53_APC_BLOCK_TOKENS={BLOCK}" in text and f"GLM53_APC_RETENTION_MAX={RETENTION_MAX}" in text,
        f"A1 grid {BLOCK} and cap {RETENTION_MAX:,} are the documented constants",
    )
    accepted = {
        None: "unset",
        "": "",
        "0": "0",
        "00": "0",
        str(BLOCK): str(BLOCK),
        "0014336": "14336",
        str(279 * BLOCK): str(279 * BLOCK),  # 999,936 = largest legal value
    }
    for value, canonical in accepted.items():
        rc, out, err = run_retention_guard(value)
        check(
            rc == 0 and out == canonical,
            f"A2 {SWA}={value!r} accepted, ranks receive {canonical!r} (rc={rc} out={out!r} {err})",
        )
    rejected = [
        "1", "3000", "3585", "1000000", str(280 * BLOCK), "-3584", "+3584",
        "3584.0", "1e3", " 3584", "3584 ", "3584\r", "nope", "0x", "0 ",
        "99999999999999999999",
    ]
    for value in rejected:
        rc, out, err = run_retention_guard(value)
        check(
            rc == 2 and SWA in err,
            f"A3 {SWA}={value!r} rejected with rc=2 and a named error (rc={rc} err={err[:60]!r})",
        )


# --------------------------------------------------------------- harness --


class Harness:
    """A throwaway copy of the launcher checkout plus a stub PATH."""

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

        # A copy whose trailing `main "$@"` is replaced by `"$@"`, so a single
        # launcher function can be driven with the real configuration preamble.
        text = (self.repo / "start.sh").read_text()
        assert text.rstrip().endswith('\nmain "$@"'), "start.sh must end with main \"$@\""
        (self.repo / "start.fn.sh").write_text(text.rstrip()[: -len('main "$@"')] + '"$@"\n')

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

    def run(self, cmd: str, entry: str = "start.sh", **extra: str) -> subprocess.CompletedProcess[str]:
        if self.log.exists():
            self.log.unlink()
        return subprocess.run(
            ["bash", f"./{entry}", cmd],
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=False,
            env=self.env(**extra),
        )

    def host_touching_calls(self) -> list[list[str]]:
        return [c for c in self.calls() if c and c[0] in ("docker", "ssh", "scp", "rsync")]


# ------------------------------------------------------------------ part B --


def parses(text: str) -> bool:
    import ast
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def longest_parseable_prefix(text: str) -> str | None:
    lines = text.splitlines(keepends=True)
    for cut in range(len(lines) - 1, 0, -1):
        candidate = "".join(lines[:cut])
        if candidate.strip() and parses(candidate):
            return candidate
    return None



def control(h: Harness, label: str) -> None:
    r = h.run("restart")
    calls = h.host_touching_calls()
    head_rm = any(c[:3] == ["docker", "rm", "-f"] for c in calls)
    worker_rm = any(c[0] == "ssh" and "docker rm -f" in c[-1] for c in calls)
    last = (r.stderr.strip().splitlines() or [""])[-1]
    check(
        head_rm and worker_rm,
        f"{label} (head rm={head_rm} worker rm={worker_rm}; later rc={r.returncode} is the stubbed preflight: {last[:100]!r})",
    )


def part_b(h: Harness) -> None:
    print("Part B: restart fails closed before any container is stopped")
    text = source()
    main_at = text.index("main() {")
    v_num = text.index("validate_numeric_config", main_at)
    v_art = text.index("validate_overlay_artifacts", main_at)
    restart = text.index("restart)  stop; start", main_at)
    check(v_num < restart and v_art < restart, "B1 main() runs both validators before `restart) stop; start`")
    check(
        "start|restart) validate_numeric_config; validate_overlay_artifacts ;;" in text,
        "B1 the validators share the start|restart arm",
    )
    guard_begin = text.index("# GLM53 overlay artifact guard (begin)")
    guard_end = text.index("# GLM53 overlay artifact guard (end)")
    guard = text[guard_begin:guard_end]
    check(guard_begin < guard_end, "B1 the artifact guard has its own sentinel block")
    check("|-\"" not in guard and '|-"' not in guard, "B1 every artifact carries an identity string (no untagged entries)")
    check(
        guard.count("|$main_guard\"") >= 6 and '|    return report"' in guard and "| tail -n 1 || true)" in guard,
        "B1 every artifact carries its exact last line as an EOF sentinel (checked against the last non-blank line)",
    )
    check(
        '"$CHAT_TEMPLATE_HOST"' in guard and "ablit/LAYER_MAP.json" in guard,
        "B1 the chat template and the ablit layer map are gated too",
    )

    vars_ = host_vars()
    shipped = shipped_apc_vars()
    check("APC_PATCH_HOST" in shipped, "B2 checkout ships patch_hybrid_prefix_hit.py")
    check(
        "PERGROUP_PATCH_HOST" in shipped or "FINEHIT_PATCH_HOST" in shipped or "NOSTORE_PATCH_HOST" in shipped,
        "B2 checkout ships at least one of per-group / fine-grained / no-store",
    )
    for var in vars_:
        check(f'"${var}|' in guard, f"B2 {var} ({vars_[var]}) is in the artifact guard")
    check("overlay/patch_ablit.py|" in guard and "overlay/ablit_runtime.py|" in guard, "B2 ablit hook + runtime are in the artifact guard")
    check(
        'for entry in "${artifacts[@]}"' in guard and "artifacts[@]}\" -eq 0" in guard,
        "B2 the guard iterates a bash array and refuses an empty list (no process-substitution status gap)",
    )

    # Control FIRST: with a valid configuration the entrypoint gets PAST the
    # validators and reaches stop (the stubs then fail preflight, which is
    # fine). Without this, every negative case below could pass for the
    # wrong reason (a guard that refuses everything).
    control(h, "B3 control: valid restart passes the gate and reaches stop on both ranks")

    def fails_closed(label: str, **env: str) -> None:
        r = h.run("restart", **env)
        calls = h.host_touching_calls()
        last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        check(
            r.returncode == 2 and not calls,
            f"{label}: rc={r.returncode}, host-touching calls={len(calls)} ({last[:90]!r})",
        )

    if wires_swa():
        fails_closed(f"B3 restart with {SWA}=3000 exits 2 with nothing stopped", **{SWA: "3000"})
        fails_closed(f"B3 restart with {SWA}=1003520 exits 2 with nothing stopped", **{SWA: "1003520"})
    if wires_fg():
        fails_closed(f"B3 restart with {FG}=yes exits 2 with nothing stopped", **{FG: "yes"})
    if wires_ns():
        fails_closed(f"B3 restart with {NS}=yes exits 2 with nothing stopped", **{NS: "yes"})
        fails_closed(f"B3 restart with {NS}= (explicitly empty) exits 2 with nothing stopped", **{NS: ""})

    broken = h.tmp / "broken_patch.py"
    broken.write_text("def (:\n    pass\n")
    empty = h.tmp / "empty_patch.py"
    empty.write_text("")
    video = h.repo / "overlay" / "patch_glm_video_placeholders.py"
    hybrid = h.repo / "overlay" / "patch_hybrid_prefix_hit.py"
    for var, base in vars_.items():
        wrong = hybrid if base == video.name else video
        fails_closed(f"B4 {var} missing ({base})", **{var: str(h.tmp / "does-not-exist.py")})
        fails_closed(f"B4 {var} pointed at a different overlay ({wrong.name})", **{var: str(wrong)})
    blank = h.tmp / "whitespace_only.py"
    blank.write_text("\n   \n\t\n")
    for var, base in shipped.items():
        fails_closed(f"B4 {var} empty file ({base})", **{var: str(empty)})
        fails_closed(f"B4 {var} whitespace-only file ({base}; pipefail-safe rc=2 diagnostic)", **{var: str(blank)})
        fails_closed(f"B4 {var} unparseable file ({base})", **{var: str(broken)})
    # Truncation: for EVERY guarded Python artifact, the longest strict prefix
    # (by lines) that still parses. It carries the identity string and is
    # valid Python, so only the EOF-sentinel check can refuse it -- and must.
    guarded = {var: h.repo / "overlay" / base for var, base in vars_.items()}
    guarded["overlay/patch_ablit.py"] = h.repo / "overlay" / "patch_ablit.py"
    guarded["overlay/ablit_runtime.py"] = h.repo / "overlay" / "ablit_runtime.py"
    for var, path in guarded.items():
        prefix_text = longest_parseable_prefix(path.read_text())
        check(prefix_text is not None and parses(prefix_text), f"B4 {path.name}: a strict prefix that still parses exists ({len(prefix_text.splitlines()) if prefix_text else 0} lines)")
        if prefix_text is None:
            continue
        if var.startswith("overlay/"):
            original = path.read_text()
            path.write_text(prefix_text)
            try:
                fails_closed(f"B4 {path.name} truncated to its longest parseable prefix (fixed-path artifact)")
            finally:
                path.write_text(original)
        else:
            prefix = h.tmp / f"truncated_{path.name}"
            prefix.write_text(prefix_text)
            fails_closed(f"B4 {var} truncated to its longest parseable prefix ({path.name})", **{var: str(prefix)})
    fails_closed("B4 CHAT_TEMPLATE_HOST missing", CHAT_TEMPLATE_HOST=str(h.tmp / "no-template.jinja"))
    (h.tmp / "template-dir").mkdir(exist_ok=True)
    (h.tmp / "template-dir" / "x").write_text("x")
    fails_closed("B4 CHAT_TEMPLATE_HOST is a (non-empty) directory", CHAT_TEMPLATE_HOST=str(h.tmp / "template-dir"))
    layer_map = h.repo / "ablit" / "LAYER_MAP.json"
    saved = layer_map.read_text()
    layer_map.write_text("{ not json")
    try:
        fails_closed("B4 ablit/LAYER_MAP.json not JSON")
    finally:
        layer_map.write_text(saved)

    control(h, "B5 control after the negative cases (the harness copy is intact)")


# ------------------------------------------------------------------ part C --


def overlay_order() -> list[str]:
    m = re.search(r"^GLM53_OVERLAY_ORDER=\(\n(.*?)^\)\n", source(), re.S | re.M)
    assert m, "GLM53_OVERLAY_ORDER=( ... ) not found in start.sh"
    return [ln.strip().split()[0] for ln in m.group(1).splitlines() if ln.strip() and not ln.strip().startswith("#")]


def apply_sequence(script: Path) -> list[str]:
    return re.findall(r"^\s*python3 /opt/glm53/(patch_[a-z0-9_]+\.py)$", script.read_text(), re.M)


def part_c(h: Harness) -> None:
    print("Part C: overlay order pinned once, emitted to both ranks")
    text = source()
    order = overlay_order()
    idx = {name: order.index(name) for name in PINNED if name in order}
    check(len(idx) == 3, f"C1 all three prefix-cache overlays are listed: {sorted(idx)}")
    check(
        len(idx) == 3 and idx[PINNED[0]] < idx[PINNED[1]] < idx[PINNED[2]],
        "C1 pinned order hybrid -> per-group -> fine-grained",
    )
    if wires_ns() or NO_STORE in order:
        check(
            NO_STORE in order and len(idx) == 3 and order.index(NO_STORE) > idx[PINNED[2]],
            "C1 no-store is listed after fine-grained (no shared anchors, but one pinned order)",
        )
    check(
        'emit_overlay_block >> "$HEAD_SCRIPT"' in text and 'emit_overlay_block >> "$WORKER_SCRIPT"' in text,
        "C2 both inner scripts take the block from emit_overlay_block",
    )
    check("python3 /opt/glm53/patch_" not in text, "C2 no literal per-rank patch ladder remains in start.sh")

    r = h.run("write_inner_scripts", entry="start.fn.sh")
    head = h.repo / ".glm53-exl3-head.inner.sh"
    worker = h.repo / ".glm53-exl3-worker.inner.sh"
    check(
        r.returncode == 0 and head.is_file() and worker.is_file(),
        f"C3 write_inner_scripts produced both inner scripts (rc={r.returncode} {r.stderr.strip()[:80]!r})",
    )
    if head.is_file() and worker.is_file():
        hs, ws = apply_sequence(head), apply_sequence(worker)
        check(hs == order, f"C3 head applies exactly GLM53_OVERLAY_ORDER ({len(hs)} entries)")
        check(ws == order, f"C3 worker applies exactly GLM53_OVERLAY_ORDER ({len(ws)} entries)")
        check(hs == ws, "C3 head and worker apply sequences are identical")
        for s in (head, worker):
            body = s.read_text()
            check(
                body.index("/opt/glm53/patch_hybrid_prefix_hit.py")
                < body.index("/opt/glm53/patch_apc_per_group_retention.py")
                < body.index("/opt/glm53/patch_apc_fine_grained_hits.py"),
                f"C3 {s.name}: hybrid -> per-group -> fine-grained in the generated script",
            )
            check(
                "python3 /opt/glm53/patch_ablit.py" in body
                and body.index("patch_kpool_tail_slotmap.py") < body.index("python3 /opt/glm53/patch_ablit.py"),
                f"C3 {s.name}: ablit still applies last",
            )
            if NO_STORE in order:
                check(
                    body.index("/opt/glm53/patch_apc_fine_grained_hits.py") < body.index(f"/opt/glm53/{NO_STORE}") < body.index("/opt/glm53/patch_xgrammar_termination.py"),
                    f"C3 {s.name}: fine-grained -> no-store -> xgrammar in the generated script",
                )
            check(
                re.search(r"if \[ -f /opt/glm53/patch_apc_per_group_retention\.py \]; then\n\s+python3", body) is not None,
                f"C3 {s.name}: every slot is `[ -f ]`-guarded (unmounted sibling slot is a no-op)",
            )


# ------------------------------------------------------------------ part D --


class Rank:
    def __init__(self, argv: list[str]) -> None:
        self.env: dict[str, str] = {}
        self.mounts: dict[str, str] = {}  # container path -> source path
        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok == "-e" and i + 1 < len(argv):
                k, _, v = argv[i + 1].partition("=")
                self.env[k] = v
                i += 2
                continue
            if tok == "-v" and i + 1 < len(argv):
                parts = argv[i + 1].split(":")
                if len(parts) >= 2 and parts[1].startswith("/opt/glm53/") and parts[1].endswith(".py"):
                    self.mounts[parts[1]] = parts[0]
                i += 2
                continue
            i += 1


def parity_issues(head: Rank, worker: Rank, scp: dict[str, str], required: dict[str, str]) -> list[str]:
    """Everything that would make the two ranks differ. `scp` maps the
    worker-side /tmp file to the host file it was copied from; `required`
    maps container env names the launcher wires to the value expected."""
    issues = []
    for name in CONTAINER_NAMES:
        if head.env.get(name) != worker.env.get(name):
            issues.append(f"env {name}: head={head.env.get(name)!r} worker={worker.env.get(name)!r}")
    for name, value in required.items():
        if head.env.get(name) != value or worker.env.get(name) != value:
            issues.append(f"env {name} expected {value!r} on both ranks: head={head.env.get(name)!r} worker={worker.env.get(name)!r}")
    if set(head.mounts) != set(worker.mounts):
        issues.append(f"mount set differs: head-only={sorted(set(head.mounts) - set(worker.mounts))} worker-only={sorted(set(worker.mounts) - set(head.mounts))}")
    for dest, head_src in head.mounts.items():
        worker_tmp = worker.mounts.get(dest)
        worker_src = scp.get(worker_tmp or "")
        if worker_src != head_src:
            issues.append(f"{dest}: head mounts {head_src}, worker mounts {worker_tmp} which scp fed from {worker_src}")
    return issues


def rank_runs(h: Harness, **env: str) -> tuple[Rank, Rank, dict[str, str]] | None:
    r = h.run("launch_cluster", entry="start.fn.sh", MODEL_DIR="/root/.cache/huggingface/x", **env)
    if r.returncode != 0:
        print(f"    launch_cluster rc={r.returncode}: {r.stderr.strip()[-300:]}")
        return None
    head = worker = None
    scp: dict[str, str] = {}
    for c in h.calls():
        if c[:2] == ["docker", "run"]:
            head = Rank(c[2:])
        elif c[0] == "ssh" and c[-1].lstrip().startswith("docker run"):
            worker = Rank(shlex.split(c[-1])[2:])
        elif c[0] == "scp" and len(c) >= 3 and ":" in c[-1]:
            scp[c[-1].split(":", 1)[1]] = c[-2]
    if head is None or worker is None:
        print(f"    could not find both docker run lines (head={head is not None} worker={worker is not None})")
        return None
    return head, worker, scp


def part_d(h: Harness) -> None:
    print("Part D: both ranks mount the same host artifacts and get identical effective values")
    shipped = shipped_apc_vars()
    order = overlay_order()

    scenarios: list[tuple[str, dict[str, str]]] = [("defaults", {})]
    if wires_swa():
        scenarios += [("SWA=14336", {SWA: "14336"}), ("SWA=0", {SWA: "0"}), ("SWA unset (auto)", {})]
    if wires_fg():
        scenarios += [("FINEGRAINED=0", {FG: "0"}), ("FINEGRAINED=1", {FG: "1"})]
    if wires_swa() and wires_fg():
        scenarios.append(("SWA=14336 + FINEGRAINED=0", {SWA: "14336", FG: "0"}))
    if wires_ns():
        scenarios += [("NO_STORE=0", {NS: "0"}), ("NO_STORE=1", {NS: "1"}), ("NO_STORE unset (default 1)", {})]

    first = None
    for label, env in scenarios:
        got = rank_runs(h, **env)
        check(got is not None, f"D1 [{label}] launch_cluster dry-run captured both docker run lines")
        if got is None:
            continue
        head, worker, scp = got
        first = first or got
        required: dict[str, str] = {}
        if wires_swa() and env.get(SWA, ""):
            required["VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA"] = env[SWA]
        if wires_fg():
            required[FG] = env.get(FG, "1")
        if wires_ns():
            required[NS] = env.get(NS, "1")
        issues = parity_issues(head, worker, scp, required)
        check(not issues, f"D2 [{label}] rank parity: " + ("; ".join(issues) if issues else "no differences"))
        for name in CONTAINER_NAMES:
            hv, wv = head.env.get(name), worker.env.get(name)
            print(f"         {name}: head={hv!r} worker={wv!r}" + ("  (not wired by this launcher)" if hv is None and wv is None else ""))
        if wires_swa() and not env.get(SWA, ""):
            check(
                "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA" not in head.env and "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA" not in worker.env,
                f"D2 [{label}] empty SWA (auto) is forwarded to neither rank",
            )
        mounted = {Path(p).name for p in head.mounts}
        check(len(head.mounts) >= 9, f"D3 [{label}] {len(head.mounts)} /opt/glm53 patch mounts, identical chain head-mount = scp source = worker-mount")
        # Expected source -> destination map for EVERY *_PATCH_HOST: the head
        # mount, the scp source and the worker mount must all be that file.
        wrong_map = []
        for var, base in host_vars().items():
            dest = f"/opt/glm53/{base}"
            src = head.mounts.get(dest, "")
            if Path(src).resolve() != (h.repo / "overlay" / base).resolve() or dest not in worker.mounts:
                wrong_map.append(f"{var}: {dest} <- {src or 'unmounted'}")
        check(not wrong_map, f"D3 [{label}] every *_PATCH_HOST maps to its own /opt/glm53 destination on both ranks (wrong={wrong_map})")
        for var, base in shipped.items():
            src = head.mounts.get(f"/opt/glm53/{base}", "")
            check(
                base in mounted and Path(src).resolve() == (h.repo / "overlay" / base).resolve(),
                f"D3 [{label}] {base} ({var}) mounted on both ranks from the checkout's overlay/ copy ({src})",
            )
        unlisted = sorted(m for m in mounted if m.startswith("patch_") and m not in order)
        check(not unlisted, f"D3 [{label}] every mounted patch_*.py is in GLM53_OVERLAY_ORDER (unlisted={unlisted})")

    # Negative self-check: the comparison must notice a one-rank difference.
    if first is not None:
        head, worker, scp = first
        tampered = Rank([])
        tampered.env = dict(worker.env)
        tampered.mounts = dict(worker.mounts)
        knob = FG if FG in tampered.env else (NS if NS in tampered.env else "GLM53_MIXED_PREFILL_CHUNK")
        tampered.env[knob] = "0" if tampered.env.get(knob) != "0" else "1"
        dest = next(iter(sorted(tampered.mounts)))
        del tampered.mounts[dest]
        issues = parity_issues(head, tampered, scp, {})
        check(
            any(f"env {knob}" in i for i in issues) and any("mount set differs" in i for i in issues),
            f"D4 synthetic one-rank mismatch is reported ({len(issues)} issues: {issues[:2]})",
        )
        bad_scp = dict(scp)
        wdest = sorted(worker.mounts)[0]
        bad_scp[worker.mounts[wdest]] = "/somewhere/else.py"
        issues = parity_issues(head, worker, bad_scp, {})
        check(any(wdest in i for i in issues), f"D4 a worker scp fed from a different host file is reported ({issues[:1]})")


# ------------------------------------------------------------------- main --


def main() -> int:
    if not START.is_file():
        raise SystemExit(f"missing {START}")
    print(f"launcher: {START}")
    print(f"ships: {', '.join(f'{v}={b}' for v, b in shipped_apc_vars().items())}; forwards SWA={wires_swa()} FINEGRAINED={wires_fg()} NO_STORE={wires_ns()}")
    part_a()
    with tempfile.TemporaryDirectory() as raw:
        h = Harness(Path(raw))
        part_b(h)
        part_c(h)
        part_d(h)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("launcher rank-parity / order / pre-stop gate OK")
    return 0


def test_launcher_rank_parity() -> None:
    """pytest entry point (the script form above is what the README documents)."""
    FAILURES.clear()
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
