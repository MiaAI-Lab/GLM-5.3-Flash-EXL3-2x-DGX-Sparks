#!/usr/bin/env python3
"""Behavioural regression for scripts/boot-shape-warmup.sh (stub curl only).

The shipped script is executed end to end with WARMUP_CURL (its existing test
seam) pointed at a fake curl in a temp dir. Nothing here touches a network, a
GPU or a real 2-node kit; no timing threshold is asserted.

Three consumer-visible outcomes:

  pass          exit 0 and a "24/24 requests ok" summary, with every ladder /
                prefill prompt arriving at the API byte-exact — n copies of
                "hello", single spaces, no trailing space — including the
                65536 rung (5 ladder + 4 prefill + 15 batch arms at the
                explicitly configured GLM53_WARMUP_MAX_CONCURRENCY=4)
  mismatch      a rung whose /tokenize count disagrees is reported failed, the
                rest of the sweep still runs, exit 1 with "23/24"
  small-context a deployment that refuses the 65536 prefill (HTTP 400) still
                warms the other 23 shapes and exits 1 — the launcher WARNs
  wide          GLM53_WARMUP_MAX_CONCURRENCY=10 fires one C=N serve-default
                burst for every N in 5..10 (exactly N chat requests each,
                69 total) and no longer warns that shapes above C=4 are cold
  capped        GLM53_WARMUP_BURST_MAX=6 at concurrency 10 stops at C=6
                (35 total) and warns that widths above C=6 are not pre-warmed

Run:  python3 tests/test_boot_shape_warmup.py   (or pytest)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPT = ROOT / "scripts" / "boot-shape-warmup.sh"
BASE = "http://warmup.invalid"
MODEL = "GLM-5.3-Flash-EXL3"

# LADDER_S + PREFILL_S as the script ships them.
RUNGS = (1, 24, 56, 120, 248, 3584, 7168, 14336, 65536)
# 5 ladder + 4 prefill + 15 batch arms at the configured concurrency of 4.
TOTAL = 24
# Each width N above 4 adds one C=N burst of N chat requests.
WIDE = 10
CAP = 6
# One prefill rung is enough to trip the check: the reported count is compared
# against the requested s, so a single disagreement fails that rung only.
MISMATCH_S = 7168
# small-context mode's deployment limit: the 65536 rung is the only one past it.
CONTEXT_LIMIT = 32768

FAILURES: list[str] = []

FAKE_CURL = '''#!/usr/bin/env python3
"""Stub curl: records what the warmup script sent, answers like the server.

Modes (WARMUP_FAKE_MODE):
  pass          report {"count": <prompt words>} as the tokenizer would
  mismatch      over-report one prefill rung's token count
  small-context refuse the oversized prefill with the deployment's limit
"""
import hashlib
import json
import os
import sys
from urllib.parse import urlsplit

args = sys.argv[1:]
url = next(arg for arg in args if arg.startswith("http://warmup.invalid/"))
path = urlsplit(url).path
payload = {}
for flag in ("--data-binary", "-d"):
    if flag in args:
        raw = args[args.index(flag) + 1]
        if raw.startswith("@"):
            with open(raw[1:]) as stream:
                payload = json.load(stream)
        else:
            payload = json.loads(raw)
        break
prompt = payload.get("prompt", "")
words = len(prompt.split())
# Chat arms tag their prompt "[warmup <nonce> <arm>-<i>] ..."; keep the arm.
content = "".join(m.get("content", "") for m in payload.get("messages", []))
arm = None
if "[warmup " in content:
    arm = content.split("[warmup ", 1)[1].split("]", 1)[0].split(" ", 1)[1].rsplit("-", 1)[0]
record = json.dumps({"path": path, "words": words, "bytes": len(prompt), "arm": arm,
                     "sha256": hashlib.sha256(prompt.encode()).hexdigest()}) + "\\n"
fd = os.open(os.environ["WARMUP_FAKE_LOG"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
os.write(fd, record.encode())
os.close(fd)
mode = os.environ["WARMUP_FAKE_MODE"]
if path == "/tokenize":
    if mode == "mismatch" and words == 7168:
        words += 1
    print(json.dumps({"count": words}))
elif mode == "small-context" and words > 32768:
    print("HTTP 400: configured context limit exceeded", file=sys.stderr)
    sys.exit(22)
else:
    print("{}")
'''


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


def ladder_prompt(n: int) -> str:
    """The bytes mk_ladder_prompt must hand the API for rung s=n."""
    return " ".join(["hello"] * n)


def run(mode: str, tmp: Path, label: str | None = None,
        **overrides: str) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    """Execute the shipped script with the stub curl; return (process, records).

    Allow-listed environment: only the seam, the stub's own variables and any
    GLM53_WARMUP_* ``overrides``. ``label`` keeps each run's log separate.
    """
    stub = tmp / "curl"
    stub.write_text(FAKE_CURL)
    stub.chmod(0o755)
    log = tmp / f"{label or mode}.jsonl"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp),
        "LC_ALL": "C",
        "TERM": "dumb",
        "WARMUP_CURL": str(stub),
        "GLM53_WARMUP_MAX_CONCURRENCY": "4",
        "WARMUP_FAKE_MODE": mode,
        "WARMUP_FAKE_LOG": str(log),
        **overrides,
    }
    done = subprocess.run(["bash", str(SCRIPT), BASE, MODEL], env=env, text=True,
                          capture_output=True, timeout=60)
    records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return done, records


def summary(stdout: str) -> tuple[int, int] | None:
    m = re.search(r"boot-shape-warmup: (\d+)/(\d+) requests ok", stdout)
    return (int(m.group(1)), int(m.group(2))) if m else None


def sent(records: list[dict], path: str, words: int) -> list[dict]:
    return [r for r in records if r["path"] == path and r["words"] == words]


def part_pass(tmp: Path) -> None:
    print("pass: full sweep including the 65536 rung")
    done, records = run("pass", tmp)
    check(done.returncode == 0,
          f"P1 exit 0 (got {done.returncode}; stderr={done.stderr.strip()[:200]!r})")
    check(summary(done.stdout) == (TOTAL, TOTAL),
          f"P2 summary {TOTAL}/{TOTAL} requests ok (got {summary(done.stdout)})")
    wrong = []
    for n in RUNGS:
        want = hashlib.sha256(ladder_prompt(n).encode()).hexdigest()
        for path in ("/tokenize", "/v1/completions"):
            hits = sent(records, path, n)
            if len(hits) != 1 or hits[0]["sha256"] != want:
                wrong.append(f"{path} s={n}: {len(hits)} prompt(s), bytes={[h['bytes'] for h in hits]}")
    check(not wrong,
          f"P3 every ladder/prefill prompt reaches the API byte-exact, incl. s=65536 (wrong={wrong})")
    wider = sorted(a for a in arm_counts(records) if burst_width(a) > 4)
    check(not wider, f"P4 concurrency 4 fires no burst wider than C=4 (got {wider})")


def arm_counts(records: list[dict]) -> dict[str, int]:
    """Chat requests per warmup arm, keyed by the arm tag in the prompt."""
    counts: dict[str, int] = {}
    for r in records:
        if r["path"] == "/v1/chat/completions" and r["arm"]:
            counts[r["arm"]] = counts.get(r["arm"], 0) + 1
    return counts


def burst_width(arm: str) -> int:
    """N for a "short-cN" burst arm, 0 for any other arm."""
    m = re.fullmatch(r"short-c(\d+)", arm)
    return int(m.group(1)) if m else 0


def part_wide(tmp: Path) -> None:
    total = TOTAL + sum(range(5, WIDE + 1))
    print(f"wide: concurrency {WIDE} bursts every width 5..{WIDE}")
    done, records = run("pass", tmp, "wide", GLM53_WARMUP_MAX_CONCURRENCY=str(WIDE))
    check(done.returncode == 0,
          f"W1 exit 0 (got {done.returncode}; stderr={done.stderr.strip()[:200]!r})")
    check(summary(done.stdout) == (total, total),
          f"W2 summary {total}/{total} requests ok (got {summary(done.stdout)})")
    counts = arm_counts(records)
    wrong = {n: counts.get(f"short-c{n}") for n in range(5, WIDE + 1)
             if counts.get(f"short-c{n}") != n}
    check(not wrong, f"W3 each C=N burst sends exactly N chat requests (width: got {wrong})")
    check("not pre-warmed" not in done.stderr,
          f"W4 no cold-shape warning when every width is covered (stderr={done.stderr.strip()[:200]!r})")


def part_capped(tmp: Path) -> None:
    total = TOTAL + sum(range(5, CAP + 1))
    print(f"capped: GLM53_WARMUP_BURST_MAX={CAP} at concurrency {WIDE}")
    done, records = run("pass", tmp, "capped", GLM53_WARMUP_MAX_CONCURRENCY=str(WIDE),
                        GLM53_WARMUP_BURST_MAX=str(CAP))
    check(done.returncode == 0,
          f"C1 exit 0 (got {done.returncode}; stderr={done.stderr.strip()[:200]!r})")
    check(summary(done.stdout) == (total, total),
          f"C2 summary {total}/{total} requests ok (got {summary(done.stdout)})")
    widths = sorted(burst_width(a) for a in arm_counts(records) if burst_width(a) > 4)
    check(widths == list(range(5, CAP + 1)),
          f"C3 bursts run for widths 5..{CAP} only (got {widths})")
    check(f"above C={CAP} are not pre-warmed" in done.stderr,
          f"C4 warns that widths above the cap are cold (stderr={done.stderr.strip()[:200]!r})")
    # A cap above MAX_NUM_SEQS is clamped to it; a non-numeric cap falls back to it.
    for label, cap in (("cap-high", "20"), ("cap-junk", "abc")):
        done, records = run("pass", tmp, label, GLM53_WARMUP_MAX_CONCURRENCY=str(CAP),
                            GLM53_WARMUP_BURST_MAX=cap)
        widths = sorted(burst_width(a) for a in arm_counts(records) if burst_width(a) > 4)
        check(done.returncode == 0 and widths == list(range(5, CAP + 1)),
              f"C5 GLM53_WARMUP_BURST_MAX={cap} at concurrency {CAP} bursts 5..{CAP} "
              f"(exit {done.returncode}, got {widths})")


def part_mismatch(tmp: Path) -> None:
    print(f"mismatch: /tokenize disagrees on rung s={MISMATCH_S}")
    done, records = run("mismatch", tmp)
    check(done.returncode == 1, f"M1 exit 1 (got {done.returncode})")
    check(summary(done.stdout) == (TOTAL - 1, TOTAL),
          f"M2 summary {TOTAL - 1}/{TOTAL} requests ok (got {summary(done.stdout)})")
    check(str(MISMATCH_S) in done.stderr and str(MISMATCH_S + 1) in done.stderr,
          f"M3 stderr identifies the expected and reported counts (stderr={done.stderr.strip()[:200]!r})")
    missing = [n for n in RUNGS
               if n != MISMATCH_S and len(sent(records, "/v1/completions", n)) != 1]
    check(not missing,
          f"M4 the rest of the sweep still runs, 65536 included (missing={missing})")


def part_small_context(tmp: Path) -> None:
    print(f"small-context: the deployment refuses the 65536 prefill (limit {CONTEXT_LIMIT})")
    done, records = run("small-context", tmp)
    check(done.returncode == 1, f"S1 exit 1 (got {done.returncode})")
    check(summary(done.stdout) == (TOTAL - 1, TOTAL),
          f"S2 summary {TOTAL - 1}/{TOTAL} requests ok (got {summary(done.stdout)})")
    missing = [n for n in RUNGS if n < 65536 and len(sent(records, "/v1/completions", n)) != 1]
    check(not missing, f"S4 the shapes the deployment does hold are still warmed (missing={missing})")


def main() -> int:
    if not SCRIPT.is_file():
        raise SystemExit(f"missing {SCRIPT}")
    print(f"warmup script: {SCRIPT}")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        part_pass(tmp)
        part_mismatch(tmp)
        part_small_context(tmp)
        part_wide(tmp)
        part_capped(tmp)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("boot-shape-warmup behaviour OK (stub curl, no live server)")
    return 0


def test_boot_shape_warmup() -> None:
    """pytest entry point (the script form above is what the README documents)."""
    FAILURES.clear()
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
