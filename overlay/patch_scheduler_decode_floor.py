#!/usr/bin/env python3
"""Keep decode from sharing an engine step with a long sparse-MLA prefill.

Issue #6: max_num_batched_tokens=1024 is the whole engine step. A decode
lane needs ~8 tokens (1 + DFlash2 k=7); the leftover ~1016 go to a peer
FLASHINFER_MLA_SPARSE_SM120 prefill chunk (~1.5 s). Decode still runs, but
at ~5 tok/s instead of ~50.

A 128-token mixed cap is not enough on 80k KV: the indexer has a large
per-step cost, so mixed decode stays ~10 tok/s. Default is therefore to
skip scheduling that prefill this step (it resumes when no peer is
decoding). Solo prefill is unchanged (1024).

GLM53_MIXED_PREFILL_CHUNK:
  skip / -1  — do not mix prefill with decode (default)
  N>0        — cap mixed prefill chunks to N tokens (128 still stalls ~10 tok/s)
  0 / off    — disable

Fail closed if the vLLM scheduler anchors drift.

v2 (remaining-prefill threshold bypass + deadline): the gate no longer holds a request whose *uncached*
remainder is <= GLM53_MIXED_PREFILL_WARM_TOKENS (default 3584 = one hybrid block,
the tail that is always recomputed) -- a cached follow-up turn used to wait the
full 15-17 s behind a running generation under `skip` because the old test was
`num_computed_tokens < num_prompt_tokens`, which is always true (the hit is capped
at num_tokens-1). A cold prefill held by `skip` proceeds after
GLM53_MIXED_PREFILL_MAX_WAIT_MS (default 1500; monotonic, stamped on the Request
the first time the gate sees it, so it survives chunking/preemption/requeue)
under GLM53_MIXED_PREFILL_LATE_CAP tokens per step (default 512) -- a time-to-first-service bound, not a
TTFT/completion bound (the request then crawls like cap:N). Both gate sites call one helper. 0 ms = wait forever (v1).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)
MARK = "# [glm53-decode-floor]"

IMPORT_OLD = "import itertools\nimport time\n"
IMPORT_NEW = "import itertools\nimport os\nimport time\n"

HELPER = '''
_GLM53_GATE_CFG = None
_GLM53_FIRST_SEEN_FALLBACK = None   # WeakKeyDictionary used only if Request rejects attribute assignment


def _glm53_gate_config():
    """Parse the v2 gate knobs once (validated; bad values fall back to defaults and are reported once).  [glm53-decode-floor-v2]"""
    global _GLM53_GATE_CFG
    if _GLM53_GATE_CFG is not None:
        return _GLM53_GATE_CFG

    def _int(name, default, lo, hi):
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            v = int(raw.strip(), 10)
        except ValueError:
            print(f"[glm53-decode-floor-v2] {name}={raw!r} is not an integer; using {default}", flush=True)
            return default
        if v < lo or v > hi:
            print(f"[glm53-decode-floor-v2] {name}={v} outside [{lo}, {hi}]; using {default}", flush=True)
            return default
        return v

    cfg = {
        "warm_tokens": _int("GLM53_MIXED_PREFILL_WARM_TOKENS", 3584, 0, 1_000_000),
        "max_wait_ms": _int("GLM53_MIXED_PREFILL_MAX_WAIT_MS", 1500, 0, 600_000),
        "late_cap": _int("GLM53_MIXED_PREFILL_LATE_CAP", 512, 64, 8192),
    }
    print(f"[glm53-decode-floor-v2] gate config: {cfg}", flush=True)
    _GLM53_GATE_CFG = cfg
    return cfg


def _glm53_mixed_prefill_gate(running, current, num_computed_tokens):
    """Return (cap|None) for this prefill step: remaining-prefill threshold bypass + deadline.  [glm53-decode-floor-v2]

    Bypass: remaining uncached prefill <= GLM53_MIXED_PREFILL_WARM_TOKENS (default 3584 = one hybrid block) -> no policy
            (typically <= 2 MNBT steps). This is a size heuristic: it admits cached follow-up tails, short cold prompts
            (the 30-token chat that used to wait 15 s), and the last block of a late-capped prefill alike.
    Late:   waited >= GLM53_MIXED_PREFILL_MAX_WAIT_MS (default 1500) -> proceed under GLM53_MIXED_PREFILL_LATE_CAP
            (default 512) tokens per step -- a bound on time-to-first-service, NOT on TTFT or completion: the request
            then crawls like cap:N and slows running decodes for its duration. 0 ms = wait forever (v1 / issue #6).
    Mutates the Request (first-seen stamp); the decision itself has no other side effects.
    Remainder uses ``num_tokens`` (prompt + generated so far) like the base scheduler, so a preempted request that
    resumes with its prompt cached still replays its output under the policy.
    """
    import time as _t
    cfg = _glm53_gate_config()
    remaining = current.num_tokens - num_computed_tokens
    if remaining <= 0:
        return None
    if remaining <= cfg["warm_tokens"]:
        return None
    cap = _glm53_mixed_prefill_policy(running, current)
    if cap is None or cap > 0:
        return cap
    max_wait_ms = cfg["max_wait_ms"]
    if max_wait_ms <= 0:
        return cap
    # First time this gate saw the request (monotonic; kept on the Request object so it survives chunking, preemption
    # and requeue). If the Request class ever rejects attribute assignment, fall back to a WeakKeyDictionary (entries
    # vanish with the request) and say so once -- never silently degrade to "never late".
    global _GLM53_FIRST_SEEN_FALLBACK
    first_seen = getattr(current, "_glm53_gate_first_seen", None)
    if first_seen is None and _GLM53_FIRST_SEEN_FALLBACK is not None:
        first_seen = _GLM53_FIRST_SEEN_FALLBACK.get(current)
    if first_seen is None:
        first_seen = _t.monotonic()
        try:
            current._glm53_gate_first_seen = first_seen
        except AttributeError:
            import weakref
            if _GLM53_FIRST_SEEN_FALLBACK is None:
                _GLM53_FIRST_SEEN_FALLBACK = weakref.WeakKeyDictionary()
                print("[glm53-decode-floor-v2] Request rejects attributes; using a weak-keyed first-seen map", flush=True)
            _GLM53_FIRST_SEEN_FALLBACK[current] = first_seen
    if (_t.monotonic() - first_seen) * 1000.0 >= max_wait_ms:
        return max(min(cfg["late_cap"], remaining), 1)
    return cap


def _glm53_mixed_prefill_policy(running, current):
    """Mixed-step prefill policy when a peer in `running` is decoding.

    None = no extra policy. 0 = skip this prefill this step. N>0 = cap.
    """
    raw = os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "skip").strip().lower()
    if raw in ("0", "off", "no"):
        return None
    if raw in ("skip", "-1"):
        cap = 0
    else:
        try:
            cap = int(raw)
        except ValueError:
            cap = 0
        if cap <= 0:
            return None
    cur_id = getattr(current, "request_id", None)
    for r in running:
        if r is current or getattr(r, "request_id", None) == cur_id:
            continue
        if r.num_computed_tokens >= r.num_prompt_tokens:
            return cap
    return None


'''

RUNNING_OLD = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # Make sure the input position does not exceed the max model len.
"""

RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_gate(self.running, request, request.num_computed_tokens)  # [glm53-decode-floor] [glm53-decode-floor-v2]
            if mixed_cap is not None:
                num_new_tokens = min(num_new_tokens, mixed_cap)

            # Make sure the input position does not exceed the max model len.
"""

WAITING_OLD = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
"""

WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_gate(self.running, request, num_computed_tokens)  # [glm53-decode-floor] [glm53-decode-floor-v2]
                    if mixed_cap is not None:
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present — skipping")
        return 0
    if "import os\n" not in text.split("import time\n", 1)[0]:
        text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import os")
    if "def _glm53_mixed_prefill_policy(" not in text:
        needle = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
        if text.count(needle) != 1:
            raise SystemExit(f"{P}: helper insert point not unique")
        text = text.replace(needle, HELPER + needle, 1)
    text = replace_once(text, RUNNING_OLD, RUNNING_NEW, "running-prefill")
    text = replace_once(text, WAITING_OLD, WAITING_NEW, "waiting-prefill")
    P.write_text(text)
    cap = os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "skip")
    print(f"patched {P.name} (mixed prefill policy={cap})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
