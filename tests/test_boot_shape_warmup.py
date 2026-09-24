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

Degenerate-engine canary (GLM53_WARMUP_CANARY):

  degenerate    every chat reply is "!!!!" and DFlash accepts nothing (#249):
                the canary exits 3 with DEGENERATE ENGINE — the launcher fails
  zero-accept   replies look fine but DFlash accepted 0 drafted tokens over the
                sweep: exit 3 on the acceptance check alone
  canary-off    the degenerate server with GLM53_WARMUP_CANARY=0 keeps the old
                behaviour (exit 0, no canary verdict)

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
  degenerate    chat replies are "!!!!" and DFlash never accepts a draft
  zero-accept   chat replies are fine but DFlash never accepts a draft
/metrics is a counter file next to the log: every scrape reports 100 more
drafted tokens, and 40 more accepted ones unless the mode says 0.
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
record = json.dumps({"path": path, "words": words, "bytes": len(prompt),
                     "sha256": hashlib.sha256(prompt.encode()).hexdigest()}) + "\\n"
fd = os.open(os.environ["WARMUP_FAKE_LOG"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
os.write(fd, record.encode())
os.close(fd)
mode = os.environ["WARMUP_FAKE_MODE"]
if path == "/metrics":
    counter = os.environ["WARMUP_FAKE_LOG"] + ".scrapes"
    n = int(open(counter).read()) + 1 if os.path.exists(counter) else 1
    open(counter, "w").write(str(n))
    accepted = 0 if mode in ("degenerate", "zero-accept") else 40 * n
    print('vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} %d' % (100 * n))
    print('vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} %d' % accepted)
elif path == "/v1/chat/completions":
    content = "!!!!!!!!!!!!!!!!" if mode == "degenerate" else "OK"
    print(json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}))
elif path == "/tokenize":
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
    check("canary: DFlash accepted 40/100 drafted tokens" in done.stdout,
          f"PC canary reports the sweep's draft acceptance (stdout tail={done.stdout.strip()[-160:]!r})")
    wrong = []
    for n in RUNGS:
        want = hashlib.sha256(ladder_prompt(n).encode()).hexdigest()
        for path in ("/tokenize", "/v1/completions"):
            hits = sent(records, path, n)
            if len(hits) != 1 or hits[0]["sha256"] != want:
                wrong.append(f"{path} s={n}: {len(hits)} prompt(s), bytes={[h['bytes'] for h in hits]}")
    check(not wrong,
          f"P3 every ladder/prefill prompt reaches the API byte-exact, incl. s=65536 (wrong={wrong})")


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


def part_degenerate(tmp: Path) -> None:
    print("degenerate: garbage replies and zero DFlash acceptance (#249)")
    done, _ = run("degenerate", tmp)
    check(done.returncode == 3, f"D1 exit 3 (got {done.returncode})")
    check("DEGENERATE ENGINE" in done.stderr and "content:" in done.stderr and "acceptance: 0/100" in done.stderr,
          f"D2 stderr names both failed checks (stderr={done.stderr.strip()[:240]!r})")
    check("may JIT mid-serve" not in done.stderr,
          f"D3 not misreported as missing JIT coverage (stderr={done.stderr.strip()[:160]!r})")


def part_zero_accept(tmp: Path) -> None:
    print("zero-accept: sane replies, but DFlash accepted nothing")
    done, _ = run("zero-accept", tmp)
    check(done.returncode == 3, f"Z1 exit 3 (got {done.returncode})")
    check("acceptance: 0/100" in done.stderr and "content:" not in done.stderr,
          f"Z2 only the acceptance check fires (stderr={done.stderr.strip()[:200]!r})")


def part_canary_off(tmp: Path) -> None:
    print("canary-off: GLM53_WARMUP_CANARY=0 keeps the pre-canary behaviour")
    done, _ = run("degenerate", tmp, "canary-off", GLM53_WARMUP_CANARY="0")
    check(done.returncode == 0, f"O1 exit 0 (got {done.returncode}; stderr={done.stderr.strip()[:160]!r})")
    check("canary:" not in done.stdout and "DEGENERATE" not in done.stderr, "O2 no canary verdict")


def part_launchers(_tmp: Path) -> None:
    """Static: every launcher turns warmup rc 3 into log collection, a stop, and a failed start."""
    print("launchers: rc 3 from the warmup stops the kit and fails the start")
    for name, stop in (("start.sh", "stop_containers"), ("start-tp3.sh", "stop"), ("start-tp4.sh", "stop")):
        text = (ROOT / name).read_text()
        body = text.split("post_ready_warmup() {", 1)[1].split("\n}\n", 1)[0]
        branch = body.split('if [ "$rc" = "3" ]; then', 1)[1].split("fi", 1)[0] if 'if [ "$rc" = "3" ]; then' in body else ""
        order = [branch.find(k) for k in ("collect_failure_logs", f"\n        {stop} ", "die ")]
        check("|| rc=$?" in body and all(i >= 0 for i in order) and order == sorted(order),
              f"L1 {name}: rc captured under set -e, then collect -> {stop} -> die (positions={order})")


def main() -> int:
    if not SCRIPT.is_file():
        raise SystemExit(f"missing {SCRIPT}")
    print(f"warmup script: {SCRIPT}")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        part_pass(tmp)
        part_degenerate(tmp)
        part_zero_accept(tmp)
        part_canary_off(tmp)
        part_launchers(tmp)
        part_mismatch(tmp)
        part_small_context(tmp)
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
