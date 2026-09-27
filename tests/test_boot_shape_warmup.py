#!/usr/bin/env python3
"""Behavioural regression for scripts/boot-shape-warmup.sh — stub curl + loopback.

Two offline harnesses, no GPU, no real kit, never port 8888/8090:

  stub curl   WARMUP_CURL (the script's existing test seam) records every
              request, so the sweep's prompts can be checked byte-exact
  loopback    a real HTTP stub owned by this test on an ephemeral 127.0.0.1
              port, so the canary's parsing, HTTP handling and /metrics reads
              run through the real curl the launchers use

Outcomes pinned here:

  pass            exit 0 and a "24/24 requests ok" summary, with every ladder /
                  prefill prompt arriving at the API byte-exact — n copies of
                  "hello", single spaces, no trailing space, 65536 included
                  (5 ladder + 4 prefill + 15 batch arms at the explicitly
                  configured GLM53_WARMUP_MAX_CONCURRENCY=4)
  mismatch        a rung whose /tokenize count disagrees is reported failed, the
                  rest of the sweep still runs, exit 1 with "23/24"
  small-context   a deployment that refuses the 65536 prefill (HTTP 400) still
                  warms the other 23 shapes and exits 1 — the launcher WARNs
  degenerate      replies "!!!!", DFlash accepts none of 100 drafts (#249):
                  exit 3, both checks named, not filed as missing JIT coverage
  degenerate-lossy the same engine with failing warmup arms: the degenerate
                  verdict still preempts "may JIT mid-serve"
  zero-accept     replies fine, DFlash accepted 0 of 100 drafted: exit 3 on the
                  acceptance check alone
  threshold-hit   the same 0/100 with GLM53_WARMUP_CANARY_MIN_DRAFTS=100 is
                  still degenerate (the minimum is inclusive)
  threshold-miss  the same 0/100 with the minimum at 101 is not judged: exit 0
  low-drafts      0 accepted but only 5 drafted tokens: exit 0, not judged
  no-metrics      /metrics 404: exit 0, acceptance not judged
  metrics-garbage /metrics answers with non-counter text: exit 0, not judged
  drafts-only     only the drafted-token family is exposed: exit 0. A missing
                  accepted counter is "cannot judge", never "accepted nothing"
  metrics-once    the pre-sweep /metrics read fails: exit 0. A post-sweep
                  whole-boot total is not a delta and is never judged
  reset           the counters go backwards (engine restarted mid-sweep): exit 0
  reset-masked    engine 0 resets to 0/0 while engine 1 climbs 100/0 -> 400/10:
                  exit 0. Aggregates hide the reset by cancelling the deltas
                  (0 accepted summed), per-series pairing refuses to judge
  labels-added    the sweep exposes a second spec-decode series: exit 0, the
                  label set changed, so no delta is taken
  labels-removed  the reverse (a series disappeared): exit 0
  unpaired        a drafted series with no accepted series of the same labels:
                  exit 0
  nonfinite       NaN / +Inf / unparseable counter samples: exit 0
  created-changed the counters climb, but a *_created generation sample moved
                  (restart caught up between the samples): exit 0
  reset-catchup   a restart resets and climbs past its pre-restart counters
                  (100/10 -> 300/10 drafted/accepted): exit 0. The counter
                  deltas alone read as a false "accepted 0 of 200 drafted";
                  the moved *_created samples expose the reset
  process-restart the same for process_start_time_seconds: exit 0
  malformed       the chat reply is not JSON: exit 0 with a note, no verdict —
                  transport-level garbage is not evidence of degeneration
  content-substring an HTML error body carrying a `"content": "OK"` substring:
                  exit 0, not judged (the old regex extractor read it as the
                  answer channel and fired)
  content-schema  valid JSON that is not a chat-completions body: exit 0
  escaped-newline the answer is decoded JSON "\nOK": exit 0, c1 is fine — the
                  old extractor turned it into literal backslash-n and judged 3
  escaped-quote   the answer is decoded JSON "\"OK\"": exit 0, same fix
  escaped-unicode the answer carries unicode-escaped and raw UTF-8 text: exit 0
  reasoning       the answer lands on reasoning_content with thinking off:
                  exit 0, not judged (the canary reads the answer channel)
  substring       the reply contains "ok" only inside other words ("Look, this
                  broke ... cookie."): still exit 3 — OK is matched as a token
  empty           a 200 with an empty answer channel: exit 3
  canary-off      GLM53_WARMUP_CANARY=0 on the degenerate server: exit 0, no
                  verdict and no /metrics read at all
  bearer          VLLM_API_KEY reaches /metrics and the sweep (the stub 401s
                  without it)

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
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPT = ROOT / "scripts" / "boot-shape-warmup.sh"
BASE = "http://warmup.invalid"
MODEL = "GLM-5.3-Flash-EXL3"
TOKEN = "canary-test-token"

# LADDER_S + PREFILL_S as the script ships them.
RUNGS = (1, 24, 56, 120, 248, 3584, 7168, 14336, 65536)
# 5 ladder + 4 prefill + 15 batch arms at the configured concurrency of 4.
TOTAL = 24
# One prefill rung is enough to trip the check: the reported count is compared
# against the requested s, so a single disagreement fails that rung only.
MISMATCH_S = 7168
# small-context mode's deployment limit: the 65536 rung is the only one past it.
CONTEXT_LIMIT = 32768

# The two spec-decode counter families and the identity samples a restart moves,
# as the Prometheus text exposition renders them. Every name is spelled out
# literally, independent of how the script under test assembles its own: a
# counter sample carries _total, its creation sample is <stem>_created and
# never <stem>_total_created.
DRAFT = "vllm:spec_decode_num_draft_tokens_total"
ACCEPT = "vllm:spec_decode_num_accepted_tokens_total"
CREATED = ("vllm:spec_decode_num_draft_tokens_created",
           "vllm:spec_decode_num_accepted_tokens_created")
STARTED = "process_start_time_seconds"


def spec(engine: int, drafted: float, accepted: float) -> str:
    """One engine's drafted/accepted counter pair."""
    labels = f'engine="{engine}",model_name="m"'
    return f"{DRAFT}{{{labels}}} {drafted}\n{ACCEPT}{{{labels}}} {accepted}\n"


def generation(created: float, started: float, engine: int = 0) -> str:
    """The *_created / process_start_time_seconds samples of one engine."""
    labels = f'engine="{engine}",model_name="m"'
    return "".join(f"{name}{{{labels}}} {created}\n" for name in CREATED) \
        + f"{STARTED} {started}\n"


GEN = generation(1758100000.5, 1758000000.25)
HEALTHY = (spec(0, 100, 40) + GEN, spec(0, 200, 80) + GEN)
ZERO_ACCEPT = (spec(0, 100, 0) + GEN, spec(0, 200, 0) + GEN)
BROKEN_SAMPLES = (f'{DRAFT}{{engine="1",model_name="m"}} NaN\n'
                  f'{ACCEPT}{{engine="1",model_name="m"}} +Inf\n'
                  f'{DRAFT}{{engine="2",model_name="m"}} abc\n')

# mode -> (before-sweep /metrics body, after-sweep body). Modes absent here use
# HEALTHY; a restart that already climbed past its pre-restart counters is only
# visible through the generation samples, which is why HEALTHY carries them and
# *_changed moves exactly one of them.
METRICS: dict[str, tuple[str, str]] = {
    "degenerate": ZERO_ACCEPT,
    "degenerate-lossy": ZERO_ACCEPT,
    "zero-accept": ZERO_ACCEPT,
    "low-drafts": (spec(0, 100, 5) + GEN, spec(0, 105, 5) + GEN),
    "reset": (spec(0, 100, 40) + GEN, spec(0, 10, 5) + GEN),
    "reset-masked": (spec(0, 100, 10) + spec(1, 100, 0) + GEN,
                     spec(0, 0, 0) + spec(1, 400, 10) + GEN),
    "labels-added": (spec(0, 100, 40) + GEN,
                     spec(0, 200, 80) + spec(1, 200, 80) + GEN),
    "labels-removed": (spec(0, 100, 40) + spec(1, 100, 40) + GEN,
                       spec(0, 200, 80) + GEN),
    "unpaired": (spec(0, 100, 40) + f'{DRAFT}{{engine="1",model_name="m"}} 100\n' + GEN,
                 spec(0, 200, 80) + f'{DRAFT}{{engine="1",model_name="m"}} 150\n' + GEN),
    "nonfinite": (spec(0, 100, 40) + BROKEN_SAMPLES, spec(0, 200, 80) + BROKEN_SAMPLES),
    "created-changed": (spec(0, 100, 40) + generation(1758100000.5, 1758000000.25),
                        spec(0, 200, 80) + generation(1758109999.5, 1758000000.25)),
    # A restart that resets and already climbs past its pre-restart counters:
    # the counter deltas alone read as "accepted 0 of 200 drafted" — a false
    # #249 — and only the moved *_created samples expose the reset.
    "reset-catchup": (spec(0, 100, 10) + generation(1000, 1758000000.25),
                      spec(0, 300, 10) + generation(2000, 1758000000.25)),
    "process-restart": (spec(0, 100, 40) + generation(1758100000.5, 1758000000.25),
                        spec(0, 200, 80) + generation(1758100000.5, 1758099999.25)),
}

FAILURES: list[str] = []

FAKE_CURL = '''#!/usr/bin/env python3
"""Stub curl: records what the warmup script sent, answers like the server.

Modes (WARMUP_FAKE_MODE):
  pass          report {"count": <prompt words>} as the tokenizer would
  mismatch      over-report one prefill rung's token count
  small-context refuse the oversized prefill with the deployment's limit

Replies are sane and the DFlash counters advance, so the canary has nothing to
report here; the loopback harness in this file drives the failing modes.
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
    # One scrape per canary endpoint read; both families must be present.
    counter = os.environ["WARMUP_FAKE_LOG"] + ".scrapes"
    n = int(open(counter).read()) + 1 if os.path.exists(counter) else 1
    open(counter, "w").write(str(n))
    print('vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} %d' % (100 * n))
    print('vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} %d' % (40 * n))
elif path == "/v1/chat/completions":
    print(json.dumps({"choices": [{"message": {"role": "assistant", "content": "OK"}}]}))
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

# --------------------------------------------------------------------------
# loopback stub endpoint
# --------------------------------------------------------------------------


class Stub:
    """Counter state and request log of one loopback server."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.scrapes = 0
        self.requests: list[tuple[str, str | None]] = []
        self.chat_payloads: list[dict] = []
        self.lock = threading.Lock()


def handler_for(stub: Stub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:
            pass

        def handle_one_request(self) -> None:
            try:
                super().handle_one_request()
            except ConnectionResetError:
                self.close_connection = True   # curl closed its side after the response

        def _payload(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return {}

        def _send(self, code: int, body: str, ctype: str = "application/json") -> None:
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if stub.mode != "bearer":
                return True
            return self.headers.get("Authorization") == f"Bearer {TOKEN}"

        def _entry(self, path: str) -> None:
            with stub.lock:
                stub.requests.append((path, self.headers.get("Authorization")))

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            self._entry(path)
            if not self._authorized():
                self._send(401, '{"error": "unauthorized"}')
            elif path == "/v1/models":
                self._send(200, json.dumps({"data": [{"id": MODEL}]}))
            elif path == "/metrics":
                self._metrics()
            else:
                self._send(404, "{}")

        def do_POST(self) -> None:
            payload = self._payload()
            path = urlsplit(self.path).path
            self._entry(path)
            if not self._authorized():
                self._send(401, '{"error": "unauthorized"}')
            elif path == "/tokenize":
                self._send(200, json.dumps({"count": len(str(payload.get("prompt", "")).split())}))
            elif path == "/v1/completions":
                if stub.mode == "degenerate-lossy":
                    self._send(500, '{"error": "internal"}')
                else:
                    self._send(200, json.dumps({"choices": [{"text": "hello"}]}))
            elif path == "/v1/chat/completions":
                with stub.lock:
                    stub.chat_payloads.append(payload)
                self._send(200, self._chat(),
                           "text/html" if stub.mode == "malformed" else "application/json")
            else:
                self._send(404, "{}")

        def _chat(self) -> str:
            mode = stub.mode
            if mode == "degenerate" or mode == "degenerate-lossy":
                content = "!!!!!!!!!!!!!!!!"
            elif mode == "substring":
                content = "Look, this broke while reading the cookie."
            elif mode == "empty":
                content = ""
            elif mode == "escaped-newline":
                content = "\nOK"
            elif mode == "escaped-quote":
                content = '"OK"'
            elif mode == "escaped-unicode":
                content = "OK \u2713 caf\u00e9"
            elif mode == "raw-unicode":
                return json.dumps({"choices": [{"message": {
                    "role": "assistant", "content": "OK \u2713 caf\u00e9"}}]},
                    ensure_ascii=False)
            elif mode == "content-substring":
                # Not JSON at all, but it carries the `"content": "OK"` substring
                # the old regex extractor read as the answer channel.
                return '<html>bad gateway: "content": "OK"</html>'
            elif mode == "content-schema":
                return json.dumps({"content": "OK", "finish_reason": "stop"})
            elif mode == "reasoning":
                return json.dumps({"choices": [{"message": {
                    "role": "assistant", "content": None, "reasoning_content": "OK"}}]})
            elif mode == "malformed":
                return "<html><body>502 Bad Gateway</body></html>"
            else:
                content = "OK"
            return json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]})

        def _metrics(self) -> None:
            mode = stub.mode
            if mode == "no-metrics":
                self._send(404, '{"error": "not found"}')
                return
            if mode == "metrics-garbage":
                self._send(200, "connection pool exhausted\n", "text/plain")
                return
            with stub.lock:
                stub.scrapes += 1
                n = stub.scrapes
            if mode == "metrics-once" and n == 1:
                self._send(500, '{"error": "metrics still warming"}')   # no baseline
                return
            if mode == "metrics-drafts-only":
                self._send(200, f'{DRAFT}{{engine="0"}} {100 * n}\n', "text/plain")
                return
            before, after = METRICS.get(mode, HEALTHY)
            self._send(200, before if n == 1 else after, "text/plain")

    return Handler


def run_loopback(
    mode: str, tmp: Path, extra_env: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess[str], Stub]:
    """Execute the shipped script against the loopback stub with the real curl.

    Allow-listed environment: no proxy, no BASH_ENV, no developer shell.
    """
    stub = Stub(mode)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(stub))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp),
            "LC_ALL": "C",
            "TERM": "dumb",
            "GLM53_WARMUP_MAX_CONCURRENCY": "4",
            "GLM53_WARMUP_REQ_TIMEOUT": "30",
            **(extra_env or {}),
        }
        done = subprocess.run(
            ["bash", str(SCRIPT), f"http://127.0.0.1:{server.server_address[1]}", MODEL],
            env=env, text=True, capture_output=True, timeout=300,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return done, stub


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


def ladder_prompt(n: int) -> str:
    """The bytes mk_ladder_prompt must hand the API for rung s=n."""
    return " ".join(["hello"] * n)


def run(mode: str, tmp: Path) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    """Execute the shipped script with the stub curl; return (process, records).

    Allow-listed environment: only the seam and the stub's own variables.
    """
    stub = tmp / "curl"
    stub.write_text(FAKE_CURL)
    stub.chmod(0o755)
    log = tmp / f"{mode}.jsonl"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp),
        "LC_ALL": "C",
        "TERM": "dumb",
        "WARMUP_CURL": str(stub),
        "GLM53_WARMUP_MAX_CONCURRENCY": "4",
        "WARMUP_FAKE_MODE": mode,
        "WARMUP_FAKE_LOG": str(log),
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


# --------------------------------------------------------------------------
# stub-curl harness: sweep integrity
# --------------------------------------------------------------------------


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
    check("canary: DFlash accepted 40/100 drafted tokens during the sweep" in done.stdout,
          f"P4 the canary reports the sweep's draft acceptance (stdout tail={done.stdout.strip()[-160:]!r})")
    check("DEGENERATE" not in done.stdout + done.stderr,
          f"P5 a healthy sweep keeps the canary quiet (stderr={done.stderr.strip()[:160]!r})")


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
    check("DEGENERATE" not in done.stderr,
          f"M5 an ordinary warmup failure stays nonfatal, not a canary verdict (stderr={done.stderr.strip()[:160]!r})")


def part_small_context(tmp: Path) -> None:
    print(f"small-context: the deployment refuses the 65536 prefill (limit {CONTEXT_LIMIT})")
    done, records = run("small-context", tmp)
    check(done.returncode == 1, f"S1 exit 1 (got {done.returncode})")
    check(summary(done.stdout) == (TOTAL - 1, TOTAL),
          f"S2 summary {TOTAL - 1}/{TOTAL} requests ok (got {summary(done.stdout)})")
    missing = [n for n in RUNGS if n < 65536 and len(sent(records, "/v1/completions", n)) != 1]
    check(not missing, f"S4 the shapes the deployment does hold are still warmed (missing={missing})")


# --------------------------------------------------------------------------
# loopback harness: canary verdicts
# --------------------------------------------------------------------------


@dataclass
class Case:
    """One canary scenario and what the script must say about it."""

    name: str
    mode: str
    rc: int
    stdout: tuple[str, ...] = ()
    stderr: tuple[str, ...] = ()
    nowhere: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    extra: str = ""


SWEEP = "24/24 requests ok"
DEGEN = "boot-shape-warmup: DEGENERATE ENGINE"
ACCEPT_100 = "acceptance: 0/100 drafted tokens accepted during the sweep"

CASES = (
    Case("healthy", "healthy", 0,
         stdout=(SWEEP, "canary: DFlash accepted 40/100 drafted tokens during the sweep"),
         nowhere=(DEGEN, "not judged"),
         extra="c1-shape"),
    Case("degenerate", "degenerate", 3,
         stdout=(SWEEP,),
         stderr=(DEGEN + " — content: c1 (temperature 0, \"Reply with OK.\") answered !!!!!!!!!!!!!!!!",
                 "acceptance: 0/100 drafted tokens accepted during the sweep"),
         nowhere=("may JIT mid-serve",)),
    Case("degenerate-lossy", "degenerate-lossy", 3,
         stdout=("15/24 requests ok",),
         stderr=(DEGEN, "warmup request(s) also failed"),
         nowhere=("may JIT mid-serve",)),
    Case("zero-accept", "zero-accept", 3,
         stdout=(SWEEP,),
         stderr=(DEGEN + " — " + ACCEPT_100,),
         nowhere=("content: c1", "may JIT mid-serve")),
    Case("threshold-hit", "zero-accept", 3,
         stderr=(ACCEPT_100,),
         env={"GLM53_WARMUP_CANARY_MIN_DRAFTS": "100"}),
    Case("threshold-miss", "zero-accept", 0,
         stdout=(SWEEP,),
         stderr=("only 100 drafted tokens (< 101) — acceptance not judged",),
         nowhere=(DEGEN,),
         env={"GLM53_WARMUP_CANARY_MIN_DRAFTS": "101"}),
    Case("low-drafts", "low-drafts", 0,
         stdout=(SWEEP, "canary: DFlash accepted 0/5 drafted tokens during the sweep"),
         stderr=("only 5 drafted tokens (< 64) — acceptance not judged",),
         nowhere=(DEGEN,)),
    Case("no-metrics", "no-metrics", 0,
         stdout=(SWEEP,),
         stderr=("no /metrics sample before the sweep", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("metrics-garbage", "metrics-garbage", 0,
         stdout=(SWEEP,),
         stderr=("drafted counter family is missing before the sweep", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("drafts-only", "metrics-drafts-only", 0,
         stdout=(SWEEP,),
         stderr=("accepted counter family is missing before the sweep", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("metrics-once", "metrics-once", 0,
         stdout=(SWEEP,),
         stderr=("no /metrics sample before the sweep", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted", "whole-boot")),
    Case("reset", "reset", 0,
         stdout=(SWEEP,),
         stderr=("went backwards", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("reset-masked", "reset-masked", 0,
         stdout=(SWEEP,),
         stderr=("went backwards", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("labels-added", "labels-added", 0,
         stdout=(SWEEP,),
         stderr=("label set changed", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("labels-removed", "labels-removed", 0,
         stdout=(SWEEP,),
         stderr=("label set changed", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("unpaired", "unpaired", 0,
         stdout=(SWEEP,),
         stderr=("are unpaired before the sweep", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("nonfinite", "nonfinite", 0,
         stdout=(SWEEP,),
         stderr=("not a finite integer", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("created-changed", "created-changed", 0,
         stdout=(SWEEP,),
         stderr=("generation sample moved", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("reset-catchup", "reset-catchup", 0,
         stdout=(SWEEP,),
         stderr=("generation sample moved", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("process-restart", "process-restart", 0,
         stdout=(SWEEP,),
         stderr=("generation sample moved", "acceptance not judged"),
         nowhere=(DEGEN, "canary: DFlash accepted")),
    Case("malformed", "malformed", 0,
         stdout=(SWEEP,),
         stderr=("not readable JSON", "content not judged"),
         nowhere=(DEGEN,)),
    Case("content-substring", "content-substring", 0,
         stdout=(SWEEP, "canary: DFlash accepted 40/100 drafted tokens during the sweep"),
         stderr=("not readable JSON",),
         nowhere=(DEGEN,)),
    Case("content-schema", "content-schema", 0,
         stdout=(SWEEP, "canary: DFlash accepted 40/100 drafted tokens during the sweep"),
         stderr=("not a chat-completions body",),
         nowhere=(DEGEN,)),
    Case("escaped-newline", "escaped-newline", 0,
         stdout=(SWEEP, "canary: DFlash accepted 40/100 drafted tokens during the sweep"),
         nowhere=(DEGEN, "not judged")),
    Case("escaped-quote", "escaped-quote", 0,
         stdout=(SWEEP, "canary: DFlash accepted 40/100 drafted tokens during the sweep"),
         nowhere=(DEGEN, "not judged")),
    Case("escaped-unicode", "escaped-unicode", 0,
         stdout=(SWEEP, "canary: DFlash accepted 40/100 drafted tokens during the sweep"),
         nowhere=(DEGEN, "not judged")),
    Case("raw-unicode", "raw-unicode", 0,
         stdout=(SWEEP, "canary: DFlash accepted 40/100 drafted tokens during the sweep"),
         nowhere=(DEGEN, "not judged")),
    Case("reasoning", "reasoning", 0,
         stdout=(SWEEP,),
         stderr=("answered on reasoning_content", "content not judged"),
         nowhere=(DEGEN,)),
    Case("substring", "substring", 3,
         stdout=(SWEEP, "canary: DFlash accepted 40/100 drafted tokens during the sweep"),
         stderr=(DEGEN + " — content: c1 (temperature 0, \"Reply with OK.\") answered Look, this broke",),
         nowhere=("acceptance: 0/",)),
    Case("empty", "empty", 3,
         stdout=(SWEEP,),
         stderr=(DEGEN + " — content: c1 returned an empty message on the answer channel",),
         nowhere=("acceptance: 0/",)),
    Case("canary-off", "degenerate", 0,
         stdout=(SWEEP,),
         nowhere=(DEGEN, "canary: "),
         env={"GLM53_WARMUP_CANARY": "0"},
         extra="zero-scrapes"),
    Case("bearer", "bearer", 0,
         stdout=(SWEEP, "canary: DFlash accepted 40/100 drafted tokens during the sweep"),
         nowhere=(DEGEN,),
         env={"VLLM_API_KEY": TOKEN},
         extra="bearer"),
)


def part_canary(tmp: Path) -> None:
    for case in CASES:
        done, stub = run_loopback(case.mode, tmp, case.env)
        note = f"[{case.name}]"
        check(done.returncode == case.rc,
              f"{note} exit {case.rc} (got {done.returncode}; "
              f"stderr={done.stderr.strip()[:200]!r})")
        for want in case.stdout:
            check(want in done.stdout, f"{note} stdout has {want!r}")
        for want in case.stderr:
            check(want in done.stderr, f"{note} stderr has {want!r} "
                                     f"(stderr={done.stderr.strip()[:240]!r})")
        both = done.stdout + done.stderr
        for unwanted in case.nowhere:
            check(unwanted not in both, f"{note} no {unwanted!r}")
        if case.extra == "zero-scrapes":
            check(stub.scrapes == 0, f"{note} the opt-out does not read /metrics (scrapes={stub.scrapes})")
        if case.extra == "c1-shape":
            c1 = [p for p in stub.chat_payloads
                  if " c1-1]" in p["messages"][0]["content"]]
            check(len(c1) == 1, f"{note} exactly one c1 probe was sent (got {len(c1)})")
            if c1:
                body = c1[0]
                check(body.get("temperature") == 0 and body.get("max_tokens") == 24,
                      f"{note} the c1 probe is bounded and deterministic "
                      f"(temperature={body.get('temperature')}, max_tokens={body.get('max_tokens')})")
                check(body.get("chat_template_kwargs", {}).get("enable_thinking") is False,
                      f"{note} the c1 probe runs with thinking off "
                      f"(kwargs={body.get('chat_template_kwargs')})")
        if case.extra == "bearer":
            paths = [p for p, _ in stub.requests]
            check("/metrics" in paths and "/v1/chat/completions" in paths,
                  f"{note} /metrics and the sweep were exercised (paths={sorted(set(paths))})")
            check(all(auth == f"Bearer {TOKEN}" for _, auth in stub.requests),
                  f"{note} VLLM_API_KEY reached every request "
                  f"(missing={[p for p, a in stub.requests if a != f'Bearer {TOKEN}'][:4]})")


def main() -> int:
    if not SCRIPT.is_file():
        raise SystemExit(f"missing {SCRIPT}")
    print(f"warmup script: {SCRIPT}")
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        part_pass(tmp)
        part_mismatch(tmp)
        part_small_context(tmp)
        part_canary(tmp)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("boot-shape-warmup behaviour OK (stub curl + loopback, no live server)")
    return 0


def test_boot_shape_warmup() -> None:
    """pytest entry point (the script form above is what the README documents)."""
    FAILURES.clear()
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
