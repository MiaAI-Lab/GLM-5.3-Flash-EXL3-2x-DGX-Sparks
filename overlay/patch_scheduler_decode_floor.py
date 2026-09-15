#!/usr/bin/env python3
"""Mixed-prefill policy: skip / cap / off / fair (issue #6).

A decode lane on this backend needs ~8 tokens (1 + DFlash2 k=7). The leftover
MNBT budget otherwise goes to a peer FLASHINFER_MLA_SPARSE_SM120 prefill
chunk. Mixed execution leaves the uniform decode FULL-graph path, and a
128-token cap is still ~10 tok/s at 80k KV. Token caps are not a millisecond
budget.

GLM53_MIXED_PREFILL_CHUNK:
  skip / -1  — do not mix prefill with decode. Starves every
               waiting/running prefill while any peer is decoding; there is
               no age limit. Solo prefill is unchanged.
  N>0        — cap mixed prefill chunks to N tokens while a peer decodes.
               The cap is fed into hybrid Mamba alignment so N < block_size
               still makes sub-block progress. 128 still stalls ~10 tok/s.
  0 / off    — disable the extra isolation policy.
  fair       — service-time mixing (TP=2 default since 2026-09-15, v5;
               opt-in on TP=3/TP=4). Decode-only
               steps between prefill turns; at most
               GLM53_FAIR_PREFILL_MAX_CHUNKS chunks per turn (default 1).
               Only prefill that contends with a decoder is charged (solo
               prefill is cost-sampled, not debt). Credit accrues at SHARE
               of accounted engine time. v5 fits a fixed-plus-per-token step
               cost from solo and mixed samples, targets the largest ladder
               rung (128..2048) whose estimated step fits
               GLM53_FAIR_PREFILL_MAX_STEP_MS, and saves credit for that
               rung instead of spending it on small chunks (v4 priced 1024
               tokens off 128-token samples linearly, ~3x too high, then
               stalled at 128-token steps: ~70 tok/s at 20% share). A
               never-served newcomer gets one prompt step-bounded probe;
               afterwards a 2s age override may borrow one such chunk, only
               after all shared debt is repaid. In-flight async prefill
               blocks the next mixed turn. Prefills are selected by last
               completed positive prefill service plus round-robin. Decode
               token/input budget is allocated first by the base scheduler.
               Solo prefill retains the base scheduler's limits. Timing is a
               host busy-time proxy.

Fair knobs (read at runtime; identical on every rank):
  GLM53_FAIR_PREFILL_CHUNK            default 256 (probe size until timing samples exist)
  GLM53_FAIR_PREFILL_SHARE            default 0.20 (credit accrual fraction)
  GLM53_FAIR_PREFILL_MAX_INTERVAL_MS  default 2000
  GLM53_FAIR_PREFILL_MAX_STEP_MS      default 1000 (estimated mixed-step limit)
  GLM53_FAIR_PREFILL_MAX_CHUNKS       default 1

Versioned installer: `# [glm53-decode-floor:v6]`. v1 (no version), v2 and v5
images are unpatched then re-patched. A v3 or v4 marker is refused without
touching the source: no authenticated producer of those intermediate helper
bodies was recovered from public history, so a canonical v3/v4 image cannot be
established, and no body is invented to fill the gap. Fail closed if anchors
drift.

Fail-closed migration: the legacy helper site is validated *before* anything is
removed -- v1/v2/v5 against the published helper text (sha256), v6 against this
installer's own text. The frozen gate sites are inverted, one exact byte range
is removed, and the whole file is round-trip checked: re-adding the same span
and re-applying that version's frozen sites must reproduce the input
byte-for-byte. A drifted, duplicated, decorated, marker-only, v3/v4-marked or
otherwise unattested site is refused with no write.

v6 (opt-in gate, both features OFF by default -- v5 behaviour is preserved):
  GLM53_MIXED_PREFILL_WARM_TOKENS  >0 admits a request whose uncached remainder
      is <= this many tokens (typically one hybrid block, 3584) without waiting
      for the decoders. 0 (default) disables the bypass.
  GLM53_MIXED_PREFILL_MAX_WAIT_MS  >0 releases a request the `skip` hold has
      starved for this long under GLM53_MIXED_PREFILL_LATE_CAP (default 512)
      tokens per step. 0 (default) waits forever, i.e. v1..v5 behaviour. The
      release only makes the request eligible for a LATE_CAP step; it bounds
      neither allocation nor compute service (capacity and the base scheduler's
      own limits still decide), and the request then crawls like cap:N.
  An explicit `cap` keeps its cap (only the warm bypass applies); `fair` is a
  different mechanism and is not touched by either knob.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys
import time
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)
MARK = "# [glm53-decode-floor]"
MARK_V2 = "# [glm53-decode-floor:v2]"
MARK_V3 = "# [glm53-decode-floor:v3]"
MARK_V4 = "# [glm53-decode-floor:v4]"
MARK_V5 = "# [glm53-decode-floor:v5]"
MARK_V6 = "# [glm53-decode-floor:v6]"

IMPORT_OLD = """import itertools
import time
"""
IMPORT_NEW = """import itertools
import os
import time
"""

# v1 helper + insertions (recipe f906ee9 / this overlay before v2).
V1_HELPER_START = "def _glm53_mixed_prefill_policy(running, current):"
V1_RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
            if mixed_cap is not None and request.num_computed_tokens < request.num_prompt_tokens:
                num_new_tokens = min(num_new_tokens, mixed_cap)

            # Make sure the input position does not exceed the max model len.
"""
V1_WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
                    if mixed_cap is not None and num_computed_tokens < request.num_prompt_tokens:
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
"""

# Frozen v2 insertions — used only to unpatch an already-v2 scheduler.
V2_BEGIN_NEW = """        self.current_step += 1
        _GLM53_MIXED.begin_step(self)  # [glm53-decode-floor:v2]
        # NOTE(woosuk) on the scheduling algorithm:
"""
V2_OBS_NEW = """        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        _GLM53_MIXED.observe_output(self, scheduler_output)  # [glm53-decode-floor:v2]
        pooler_outputs = model_runner_output.pooler_output
"""
V2_RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v2]
            if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                num_new_tokens = min(num_new_tokens, mixed_cap)

            # Make sure the input position does not exceed the max model len.
"""
V2_WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v2]
                    if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
"""
V2_ALIGN_NEW = """            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            _align_cap = getattr(self, "_glm53_align_prefill_limit", None)  # [glm53-decode-floor:v2]
            if _align_cap is not None and _align_cap > 0:
                max_prefill_tokens = min(max_prefill_tokens, _align_cap)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
"""
V2_RUNNING_MAMBA_NEW = """            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )
            _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v2]
"""
V2_WAITING_MAMBA_NEW = """                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v2]
                        if num_new_tokens == 0:
                            break
"""


class _Glm53MixedPrefill:  # [glm53-decode-floor:v5]
    """Bound contention using completion feedback, without synchronizing GPUs."""

    LADDER = (128, 256, 512, 768, 1024, 1536, 2048)
    COLD_TOK_S = 1300.0
    FIT_WINDOW = 24

    def __init__(self, now=None):
        self._now = now or time.monotonic
        self.mode = "skip"
        self.legacy_cap = 0
        self.logged_boot = False
        self.logged_gate = False
        self.hist_every = 50
        self._parse()
        self.last_service = {}
        self.arrival = {}
        self.rr_seq = {}
        self.served_tokens = {}
        self.rr_n = 0
        self.steps = 0
        self.credit = 0.0
        self.in_contention = False
        self.inflight = {}
        self.mixed_samples = []
        self.solo_samples = []
        self.last_account_mono = None
        self.last_prefill_turn_mono = 0.0
        self.step_tag = None
        self._sched_id = None
        self._open_rec = None
        self.selected = set()
        self._candidates = []
        self._tried = set()
        self.step_mode = "solo"
        self.defer_reason = "none"
        self.missed_prefill = 0
        self._model_cache = None

    def _e(self, name, default):
        v = os.environ.get(name)
        return default if v is None or not str(v).strip() else str(v).strip()

    def _parse(self) -> None:
        raw = self._e("GLM53_MIXED_PREFILL_CHUNK", "skip").strip().lower()
        self.legacy_cap = 0
        if raw in ("0", "off", "no"):
            self.mode = "off"
        elif raw in ("skip", "-1"):
            self.mode = "skip"
        elif raw == "fair":
            self.mode = "fair"
        else:
            try:
                cap = int(raw)
            except ValueError:
                cap = 0
            if cap <= 0:
                self.mode = "off"
            else:
                self.mode = "cap"
                self.legacy_cap = cap
        try:
            self.chunk = int(self._e("GLM53_FAIR_PREFILL_CHUNK", "256"))
        except ValueError:
            self.chunk = 256
        if self.chunk <= 0:
            self.chunk = 256
        try:
            self.share = float(self._e("GLM53_FAIR_PREFILL_SHARE", "0.20"))
        except ValueError:
            self.share = 0.20
        self.share = min(1.0, max(0.0, self.share))
        try:
            self.interval_s = int(self._e("GLM53_FAIR_PREFILL_MAX_INTERVAL_MS", "2000")) / 1000.0
        except ValueError:
            self.interval_s = 2.0
        if self.interval_s <= 0:
            self.interval_s = 2.0
        try:
            self.max_chunks = int(self._e("GLM53_FAIR_PREFILL_MAX_CHUNKS", "1"))
        except ValueError:
            self.max_chunks = 1
        self.max_chunks = max(1, min(self.max_chunks, 16))
        try:
            self.max_step_s = max(0.001, int(self._e("GLM53_FAIR_PREFILL_MAX_STEP_MS", "1000")) / 1000.0)
        except ValueError:
            self.max_step_s = 1.0
        # v6 gate (warm bypass + deadline). Both features are OFF unless an
        # operator sets the knobs: 0 disables each one and is the default, so a
        # skip/cap deployment keeps exactly the v5 hold behaviour.
        try:
            self.warm_tokens = int(self._e("GLM53_MIXED_PREFILL_WARM_TOKENS", "0"))
        except ValueError:
            self.warm_tokens = 0
        if not 0 <= self.warm_tokens <= 1_000_000:
            self.warm_tokens = 0
        try:
            self.max_wait_ms = int(self._e("GLM53_MIXED_PREFILL_MAX_WAIT_MS", "0"))
        except ValueError:
            self.max_wait_ms = 0
        if not 0 <= self.max_wait_ms <= 600_000:
            self.max_wait_ms = 0
        try:
            self.late_cap = int(self._e("GLM53_MIXED_PREFILL_LATE_CAP", "512"))
        except ValueError:
            self.late_cap = 512
        if not 64 <= self.late_cap <= 8192:
            self.late_cap = 512
        if self.mode == "fair" and not self.logged_boot:
            print(
                f"[glm53-decode-floor] fair v5 probe_chunk={self.chunk} "
                f"ladder={min(self.LADDER)}..{max(self.LADDER)} share={self.share} "
                f"interval_s={self.interval_s} max_step_s={self.max_step_s} "
                f"max_chunks={self.max_chunks}",
                flush=True,
            )
            self.logged_boot = True
        if (self.warm_tokens or self.max_wait_ms) and not self.logged_gate:
            print(
                f"[glm53-decode-floor] gate v6 warm_tokens={self.warm_tokens} "
                f"max_wait_ms={self.max_wait_ms} late_cap={self.late_cap}",
                flush=True,
            )
            self.logged_gate = True


    @staticmethod
    def prefill_remaining(request, computed=None):
        prompt = int(getattr(request, "num_prompt_tokens", 0) or 0)
        if computed is None:
            computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        tokens = int(getattr(request, "num_tokens", prompt) or prompt)
        return max(0, max(prompt, tokens - 1) - int(computed))

    def needs_prefill_compute(self, request, computed=None):
        return self.prefill_remaining(request, computed) > 0

    def _iter_waiting(self, sched):
        for name in ("waiting", "skipped_waiting"):
            q = getattr(sched, name, None)
            if not q:
                continue
            try:
                for r in q:
                    yield r
            except TypeError:
                continue

    def _live_ids(self, sched):
        ids = set()
        for r in list(getattr(sched, "running", None) or []):
            rid = getattr(r, "request_id", None)
            if rid is not None:
                ids.add(rid)
        for r in self._iter_waiting(sched):
            rid = getattr(r, "request_id", None)
            if rid is not None:
                ids.add(rid)
        reqs = getattr(sched, "requests", None)
        if isinstance(reqs, dict):
            ids.update(reqs.keys())
        return ids

    def _prune(self, live):
        for store in (self.last_service, self.arrival, self.rr_seq, self.served_tokens):
            dead = [k for k in store if k not in live]
            for k in dead:
                store.pop(k, None)


    @property
    def inflight_prefill(self):
        return sum(bool(r["prefill_tokens"]) for r in self.inflight.values())

    def _shape(self, decodes, prefills):
        history = max((int(r.num_computed_tokens) for r in decodes), default=0)
        position = max((int(r.num_computed_tokens) for r in prefills), default=0)
        return (len(decodes), (history // 4096).bit_length(),
                (position // 4096).bit_length())

    def _cost_model(self):
        """Fit step cost dt = a + b*n over recent prefill-bearing steps.

        Every prefill-bearing step pays a fixed cost on this kit (~0.3 s host
        time for 82..256 tokens, ~2.7 s for 3584), so a mixed step is close to
        a solo chunk plus a few decode rows: solo samples are pooled for the
        fit. The fit is then scaled so recent mixed samples are not
        underestimated (75th percentile of actual/fit, clamped to [1, 1.5]).
        Needs two distinct chunk sizes; otherwise returns None and the caller
        scales linearly. v4 scaled one sample linearly, priced 1024 tokens off
        128-token samples at ~3x the real cost, and never climbed back.
        """
        if self._model_cache is not None:
            return self._model_cache
        mixed = [(n, dt) for _, n, dt in self.mixed_samples[-self.FIT_WINDOW:]]
        pts = mixed + [(n, dt) for n, dt in self.solo_samples[-self.FIT_WINDOW:]]
        if len(pts) < 2 or len({n for n, _ in pts}) < 2:
            return None
        cnt = float(len(pts))
        sx = float(sum(n for n, _ in pts))
        sy = float(sum(dt for _, dt in pts))
        sxx = float(sum(n * n for n, _ in pts))
        sxy = float(sum(n * dt for n, dt in pts))
        den = cnt * sxx - sx * sx
        b = max(0.0, (cnt * sxy - sx * sy) / den) if den > 0 else 0.0
        a = max(0.0, (sy - b * sx) / cnt)
        if mixed:
            ratios = sorted(dt / max(1e-6, a + b * n) for n, dt in mixed)
            r = ratios[min(len(ratios) - 1, int(0.75 * len(ratios)))]
        else:
            r = 1.1  # decode rows not sampled yet
        r = min(1.5, max(1.0, r))
        self._model_cache = (a * r, b * r)
        return self._model_cache

    def _est_dt(self, tokens, shape=None):
        """Conservative host-time estimate; not a guaranteed execution bound."""
        n = max(1, int(tokens))
        model = self._cost_model()
        if model is not None:
            a, b = model
            return max(0.01, a + b * n)
        samples = self.mixed_samples[-8:]
        if samples:
            # One size only: scale linearly with a fixed-cost floor.
            return max(0.01, max(dt * max(0.5, n / t) for _, t, dt in samples))
        return max(0.05, n / self.COLD_TOK_S)

    def _target(self, remaining, room):
        """Largest rung whose estimated step fits `room` (tokens per accounted
        second rise with size under a fixed per-step cost), or None."""
        rungs = {self.chunk}
        if self.mixed_samples or self._cost_model() is not None:
            rungs.update(self.LADDER)
        if remaining is not None:
            rungs = {min(n, remaining) for n in rungs}
        fitting = [(n, self._est_dt(n)) for n in sorted(rungs)]
        fitting = [(n, cost) for n, cost in fitting if cost <= room + 1e-9]
        return fitting[-1] if fitting else None

    def _credit_limit(self):
        return min(self.max_step_s, self._est_dt(max(self.chunk, max(self.LADDER))))

    def _rank_prefills(self, prefills):
        return sorted(prefills, key=lambda r: (
            self.last_service.get(r.request_id, 0.0),
            self.rr_seq.get(r.request_id, 0), self.arrival[r.request_id], r.request_id))

    def _promote_next(self):
        grants = (self._open_rec or {}).get("grants", {})
        for r in self._candidates:
            if len(self.selected | set(grants)) >= self.max_chunks:
                break
            if r.request_id not in self._tried:
                self.selected.add(r.request_id)

    def _release(self, rid, reason="allocation"):
        rec = self._open_rec
        if rec is None:
            return
        grant = rec["grants"].pop(rid, None)
        if grant:
            self.credit += grant[1]
            if grant[2]:
                rec["borrowed"] = False
        self.selected.discard(rid)
        self._tried.add(rid)
        self.defer_reason = reason
        self._promote_next()

    def note_scheduled(self, request, num_new_tokens):
        # Called after alignment/encoder caps; final allocation is sealed below.
        if num_new_tokens <= 0:
            self._release(request.request_id, "zero_progress")

    def protect_decode(self, request):
        return (self.mode == "fair" and self._open_rec is not None
                and self._open_rec["had_decode"]
                and self.needs_prefill_compute(request))

    def begin_step(self, sched):
        now = self._now()
        sid = id(sched)
        tag = (sid, int(sched.current_step))
        if self.step_tag == tag:
            return
        if self._sched_id is not None and self._sched_id != sid:
            self.inflight.clear()
            self._open_rec = None
            self.credit = 0.0
            self.in_contention = False
            self.last_account_mono = None
        self._sched_id = sid
        # An unfinished/failed schedule has dispatched nothing: refund its grants.
        if self._open_rec:
            self.credit += sum(g[1] for g in self._open_rec["grants"].values())
        self.step_tag = tag
        self.steps += 1
        self._prune(self._live_ids(sched))
        running = list(sched.running)
        waiting = list(self._iter_waiting(sched))
        prefills = list({r.request_id: r for r in running + waiting
                         if self.needs_prefill_compute(r)}.values())
        decodes = [r for r in running if not self.needs_prefill_compute(r)]
        # The base loop checks eligibility and reserves BOTH token and input/draft
        # capacity by executing these requests first. Preserve order within groups.
        if self.mode == "fair":
            sched.running[:] = decodes + [r for r in running
                                         if self.needs_prefill_compute(r)]
        for r in running + waiting:
            self.arrival.setdefault(r.request_id, now)
        self._candidates = self._rank_prefills(prefills)
        self._tried = set()
        self.selected = set()
        sched._glm53_align_prefill_limit = None
        self._open_rec = {
            "step_id": int(sched.current_step), "t_submit": now,
            "had_decode": bool(decodes), "had_prefill_demand": bool(prefills),
            "shape": self._shape(decodes, prefills), "grants": {},
            "borrowed": False,
        }
        # Do not forgive debt while a decoder or its outstanding work remains.
        if not decodes and not any(r["had_decode"] for r in self.inflight.values()):
            self.in_contention = False
            self.credit = 0.0
        self.step_mode = "legacy" if self.mode != "fair" else "solo"
        self.defer_reason = "none"
        if self.mode != "fair" or not prefills or not decodes:
            return
        if not self.in_contention:
            self.credit = min(self.max_step_s, self._est_dt(self.chunk))
            self.in_contention = True
        if self.inflight_prefill:
            self.step_mode = "decode_only"
            self.defer_reason = "async_inflight"
        else:
            self.step_mode = "prefill_turn"
            self._promote_next()
        self._maybe_log()

    def _warm_bypass(self, remaining):
        """Admit a small uncached remainder at once (v6). Off while the knob is 0."""
        return self.warm_tokens > 0 and remaining <= self.warm_tokens

    def _deadline_cap(self, request, remaining):
        """Late admission for a request the skip hold would starve (v6).

        Returns 0 while the deadline is disabled (0 ms = wait forever, v5
        behaviour) or not reached, else LATE_CAP tokens for this step. The
        first-seen stamp is the per-request arrival the fair policy already
        keeps, so chunking, preemption and requeue cannot reset the wait.
        """
        if self.max_wait_ms <= 0:
            return 0
        rid = request.request_id
        first_seen = self.arrival.get(rid)
        if first_seen is None:
            first_seen = self._now()
            self.arrival[rid] = first_seen
        if (self._now() - first_seen) * 1000.0 >= self.max_wait_ms:
            return max(1, min(self.late_cap, remaining))
        return 0

    def cap_for(self, sched, request, computed=None):
        self.begin_step(sched)
        sched._glm53_align_prefill_limit = None
        remaining = self.prefill_remaining(request, computed)
        if remaining <= 0 or self.mode == "off":
            return None
        peer_decode = any(r is not request and not self.needs_prefill_compute(r)
                          for r in sched.running)
        if self.mode == "skip":
            if not peer_decode:
                return None
            if self._warm_bypass(remaining):
                return None
            return self._deadline_cap(request, remaining)
        if self.mode == "cap":
            cap = self.legacy_cap if peer_decode else None
            if cap is not None and self._warm_bypass(remaining):
                cap = None
            sched._glm53_align_prefill_limit = cap
            return cap
        if self.step_mode == "solo":
            return None
        rid = request.request_id
        rec = self._open_rec
        if rid in rec["grants"]:
            return rec["grants"][rid][0]
        if self.step_mode != "prefill_turn" or rid not in self.selected:
            return 0
        reserved = sum(g[1] for g in rec["grants"].values())
        gap_room = max(0.0, self.max_step_s - reserved)
        age = self._now() - self.last_service.get(rid, self.arrival[rid])
        pick = self._target(remaining, gap_room)
        if pick is None:
            self._release(rid, "gap_budget")
            if age >= self.interval_s:
                self.missed_prefill += 1
                self._maybe_log()
            return 0
        # Save credit for the most efficient rung that fits the step budget
        # instead of spending it on a smaller chunk now (v4 did, and stalled
        # at 128-token steps once one expensive sample priced 256 too high).
        cap, cost = pick
        borrowed = False
        if cost > self.credit + 1e-9:
            # A never-served newcomer gets a prompt probe; afterwards service
            # ages. Either may borrow ONE step-bounded chunk globally, only
            # after all shared debt is repaid: queue churn or several aged
            # newcomers cannot repeatedly overdraw the decoder.
            due = rid not in self.last_service or age >= self.interval_s
            if (due and self.credit >= -1e-9 and not rec["grants"]
                    and not rec["borrowed"]):
                borrowed = True
            else:
                self._release(rid, "credit")
                if age >= self.interval_s:
                    self.missed_prefill += 1
                    self._maybe_log()
                return 0
        self.credit -= cost
        rec["grants"][rid] = (cap, cost, borrowed)
        rec["borrowed"] |= borrowed
        self._tried.add(rid)
        sched._glm53_align_prefill_limit = cap
        return cap

    def finish_step(self, sched, scheduler_output):
        """Seal final work before _update_after_schedule advances request state."""
        rec = self._open_rec
        self._open_rec = None
        if rec is None:
            return
        scheduled = {rid: int(n) for rid, n in scheduler_output.num_scheduled_tokens.items()
                     if n > 0}
        requests = getattr(sched, "requests", {})
        prefill = {}
        for rid, n in scheduled.items():
            request = requests.get(rid)
            if request is not None:
                amount = min(n, self.prefill_remaining(request))
                if amount > 0:
                    prefill[rid] = amount
        reserved = 0.0
        for rid, (_, estimate, _) in rec["grants"].items():
            # Never publish/charge a tentative grant that alignment, allocation,
            # encoder caps or preemption removed from the final scheduler output.
            actual_est = min(estimate, self._est_dt(prefill[rid], rec["shape"])) if rid in prefill else 0.0
            self.credit += estimate - actual_est
            reserved += actual_est
        if not scheduled:
            # Empty schedules do not necessarily have a completion callback.
            return
        rec.update(output=scheduler_output, scheduled=scheduled,
                   prefill_tokens=prefill, reserved=reserved)
        del rec["grants"]
        if self.mode == "fair":
            self.inflight[id(scheduler_output)] = rec

    def observe_output(self, sched, scheduler_output):
        rec = self.inflight.pop(id(scheduler_output), None)
        if rec is None or rec["output"] is not scheduler_output:
            return  # Empty, duplicate or unrelated callback; never pop another step.
        now = self._now()
        # Partition observed busy wall time instead of adding overlapping
        # submit-to-completion latencies of queued async steps. Queue residence
        # and engine idle gaps therefore cannot mint decode credit twice.
        start = rec["t_submit"]
        if self.last_account_mono is not None:
            start = max(start, self.last_account_mono)
        dt = max(0.0, now - start)
        self.last_account_mono = max(now, self.last_account_mono or now)
        actual = scheduler_output.num_scheduled_tokens
        served = {rid: min(n, max(0, int(actual.get(rid, 0))))
                  for rid, n in rec["prefill_tokens"].items()
                  if int(actual.get(rid, 0)) > 0}
        for rid, n in served.items():
            self.last_service[rid] = now
            self.rr_n += 1
            self.rr_seq[rid] = self.rr_n
            self.served_tokens[rid] = self.served_tokens.get(rid, 0) + n
        n = sum(served.values())
        if rec["had_decode"]:
            # Reservation was already debited; settle once, including overruns.
            self.credit += rec["reserved"] + self.share * dt - (dt if n else 0.0)
            self.credit = min(self.credit, self._credit_limit())
            if n and dt > 0:
                self.mixed_samples.append((rec["shape"], n, dt))
                self.mixed_samples = self.mixed_samples[-64:]
                self.last_prefill_turn_mono = now
                self._model_cache = None
        elif n and dt > 0:
            self.solo_samples.append((n, dt))
            self.solo_samples = self.solo_samples[-32:]
            self._model_cache = None
        if n and self.hist_every > 0:
            print(f"[glm53-decode-floor] completed_step={rec['step_id']} "
                  f"contention={int(rec['had_decode'])} prefill_tokens={served} "
                  f"reserved_s={rec['reserved']:.3f} accounted_s={dt:.3f} "
                  f"credit={self.credit:.3f} timing=host_busy_proxy", flush=True)
        self._prune(self._live_ids(sched))

    def _maybe_log(self):
        if self.hist_every <= 0 or self.steps % self.hist_every != 1:
            return
        rec = self._open_rec or {}
        remaining = sum(self.prefill_remaining(r) for r in self._candidates)
        target, estimate = (self._target(None, self.max_step_s)
                            or (self.chunk, self._est_dt(self.chunk)))
        rate = target / estimate if estimate > 0 else 0.0
        eta = remaining / (self.share * rate) if self.share * rate > 0 else float("inf")
        model = self._cost_model()
        fit = (f"fit_fixed_s={model[0]:.3f} fit_us_per_tok={model[1] * 1e6:.0f}"
               if model else "fit=none")
        print(f"[glm53-decode-floor] step={rec.get('step_id', self.steps)} "
              f"mode={self.step_mode} defer={self.defer_reason} "
              f"inflight={self.inflight_prefill} credit={self.credit:.3f} "
              f"remaining={remaining} target={target} est_s={estimate:.3f} {fit} "
              f"eta_est_s={eta:.1f} max_step_s={self.max_step_s:.3f} "
              f"missed={self.missed_prefill} timing=host_busy_proxy", flush=True)

    @staticmethod
    def aligned_new_tokens(
        start, num_new, prefill_end, block_size, max_prefill_tokens, policy_cap=None
    ) -> int:
        """Hybrid align clip. policy_cap is an intentional mixed cap, not leftover budget."""
        if policy_cap is not None and policy_cap > 0:
            max_prefill_tokens = min(max_prefill_tokens, policy_cap)
        end = start + num_new
        if end < prefill_end:
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
        return max(0, end - start)



BEGIN_OLD = """        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
"""
BEGIN_NEW = """        self.current_step += 1
        _GLM53_MIXED.begin_step(self)  # [glm53-decode-floor:v4]
        # NOTE(woosuk) on the scheduling algorithm:
"""

OBS_OLD = """        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
"""
OBS_NEW = """        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        _GLM53_MIXED.observe_output(self, scheduler_output)  # [glm53-decode-floor:v4]
        pooler_outputs = model_runner_output.pooler_output
"""

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
            mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v4]
            if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
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
                    mixed_cap = _glm53_mixed_prefill_policy(self, request, num_computed_tokens)  # [glm53-decode-floor:v4]
                    if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request, num_computed_tokens):
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
"""

ALIGN_OLD = """            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
"""
ALIGN_NEW = """            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            _align_cap = getattr(self, "_glm53_align_prefill_limit", None)  # [glm53-decode-floor:v4]
            if _align_cap is not None and _align_cap > 0:
                max_prefill_tokens = min(max_prefill_tokens, _align_cap)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
"""

RUNNING_MAMBA_OLD = """            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )
"""
RUNNING_MAMBA_NEW = """            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )
            _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v4]
"""

WAITING_MAMBA_OLD = """                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        if num_new_tokens == 0:
                            break
"""
WAITING_MAMBA_NEW = """                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v4]
                        if num_new_tokens == 0:
                            if _GLM53_MIXED.mode == "fair":
                                request_queue.pop_request()
                                step_skipped_waiting.prepend_request(request)
                                continue
                            break
"""

FIN_OLD = """        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
"""
FIN_NEW = """        _GLM53_MIXED.finish_step(self, scheduler_output)  # [glm53-decode-floor:v4]
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
"""

RUNNING_ZERO_OLD = """            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
"""
RUNNING_ZERO_NEW = """            if num_new_tokens == 0:
                _GLM53_MIXED.note_scheduled(request, 0)  # [glm53-decode-floor:v4]
                # The request cannot be scheduled because one of the following
"""

WAITING_ZERO_OLD = """                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break
"""
WAITING_ZERO_NEW = """                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            _GLM53_MIXED.note_scheduled(request, 0)  # [glm53-decode-floor:v4]
                            if _GLM53_MIXED.mode == "fair":
                                request_queue.pop_request()
                                step_skipped_waiting.prepend_request(request)
                                continue
                            break
"""

PREFILL_PREEMPT_OLD = """                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
"""
PREFILL_PREEMPT_NEW = """                    # The request cannot be scheduled.
                    if _GLM53_MIXED.protect_decode(request):  # [glm53-decode-floor:v4]
                        break
                    # Preempt the lowest-priority request.
"""

RUNNING_ALLOC_OLD = """            if new_blocks is None:
                # Cannot schedule this request.
                break
"""
RUNNING_ALLOC_NEW = """            if new_blocks is None:
                # Cannot schedule this request.
                _GLM53_MIXED.note_scheduled(request, 0)  # [glm53-decode-floor:v4]
                if _GLM53_MIXED.protect_decode(request):
                    req_index += 1
                    continue
                break
"""

WAITING_ALLOC_OLD = """                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer:"""
WAITING_ALLOC_NEW = """                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    _GLM53_MIXED.note_scheduled(request, 0)  # [glm53-decode-floor:v4]
                    if _GLM53_MIXED.protect_decode(request):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue
                    break

                # KVTransfer:"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


V2_PAIRS = (
    (V2_BEGIN_NEW, BEGIN_OLD, 'v2-begin'),
    (V2_OBS_NEW, OBS_OLD, 'v2-obs'),
    (V2_RUNNING_NEW, RUNNING_OLD, 'v2-running'),
    (V2_WAITING_NEW, WAITING_OLD, 'v2-waiting'),
    (V2_ALIGN_NEW, ALIGN_OLD, 'v2-align'),
    (V2_RUNNING_MAMBA_NEW, RUNNING_MAMBA_OLD, 'v2-running-mamba'),
    (V2_WAITING_MAMBA_NEW, WAITING_MAMBA_OLD, 'v2-waiting-mamba'),
)

V4_PAIRS = (
    (BEGIN_NEW, BEGIN_OLD, 'begin'),
    (OBS_NEW, OBS_OLD, 'obs'),
    (RUNNING_NEW, RUNNING_OLD, 'running'),
    (WAITING_NEW, WAITING_OLD, 'waiting'),
    (ALIGN_NEW, ALIGN_OLD, 'align'),
    (RUNNING_MAMBA_NEW, RUNNING_MAMBA_OLD, 'running_mamba'),
    (WAITING_MAMBA_NEW, WAITING_MAMBA_OLD, 'waiting_mamba'),
    (FIN_NEW, FIN_OLD, 'fin'),
    (RUNNING_ZERO_NEW, RUNNING_ZERO_OLD, 'running_zero'),
    (WAITING_ZERO_NEW, WAITING_ZERO_OLD, 'waiting_zero'),
    (PREFILL_PREEMPT_NEW, PREFILL_PREEMPT_OLD, 'prefill_preempt'),
    (RUNNING_ALLOC_NEW, RUNNING_ALLOC_OLD, 'running_alloc'),
    (WAITING_ALLOC_NEW, WAITING_ALLOC_OLD, 'waiting_alloc'),
)

V1_PAIRS = (
    (V1_RUNNING_NEW, RUNNING_OLD, 'v1-running'),
    (V1_WAITING_NEW, WAITING_OLD, 'v1-waiting'),
)

# v5 and v6 use the same scheduler anchors as v4 with the marker advanced; the
# v4 insertions above stay frozen so a v4 image can be unpatched exactly. v6
# only changes the helper (opt-in warm bypass + deadline, both off by default),
# so an unpatch/re-patch of a v6 image is byte-identical.
V5_PAIRS = tuple((new.replace(MARK_V4, MARK_V5), old, label) for new, old, label in V4_PAIRS)
V6_PAIRS = tuple((new.replace(MARK_V5, MARK_V6), old, label) for new, old, label in V5_PAIRS)

LEGACY_MARK = {1: MARK, 2: MARK_V2, 3: MARK_V3, 4: MARK_V4, 5: MARK_V5, 6: MARK_V6}
LEGACY_PAIRS = {1: V1_PAIRS, 2: V2_PAIRS, 5: V5_PAIRS, 6: V6_PAIRS}  # v3/v4 are refused, not migrated

NEEDLE = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
ADAPTIVE_K_HEAD = "class _Glm53AdaptiveK:"
CLASS_HEAD = "class _Glm53MixedPrefill:"

# ---------------------------------------------------------------------------
# Canonical legacy helper registry.
#
# Every migratable version has exactly one accepted helper site, and the site is
# validated *before* anything is removed. The versions whose helper text is
# published are compared byte-for-byte (sha256 and length). v3 and v4 were
# introduced by 180725a5ce33 as markers for intermediate builds whose helper
# bodies were never recovered from public history: they are refused as
# unsupported rather than migrated on a structural guess, because an invented
# body would not be a canonical image. A site that validates is removed as one
# exact byte range, so text between the helper and its anchors can never be
# dropped by accident.
#
# provenance, public history of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks:
#   v1  f3043c95bbf9 overlay/patch_scheduler_decode_floor.py:HELPER
#       cross-attested byte-identical by cad980f7263:HELPER_V1, which documents
#       itself as "byte-identical to the shipped v1 overlay"
#   v2  9a23b2d8f061 overlay/patch_scheduler_decode_floor.py:HELPER
#   v5  a9bbbadc4c72 overlay/patch_scheduler_decode_floor.py:_helper_text()
#   v3/v4  no public revision carries a body (180725a5ce33 adds only the
#          markers); refused, not migrated (see above)
#
# The published variants that share the unversioned marker but bake a different
# policy default (the "0" default of 83f0067/d9758a6/9a557cf/14b6a9f) are NOT
# accepted: migrating them would silently land a held default change, so they
# are refused like any other drifted site.
# ---------------------------------------------------------------------------
LEGACY_HELPER_SHA256 = {
    1: "c0c10c6385bd7e75bfc0f724378960b9d9cb943cf18c386a5a780cf03cb4c48d",
    2: "e170f2bf6d0c0fe0493a91ff54bd719036f6c4b5ea9fcf2ee1346968a4e3fbed",
    5: "c05769276df6da0a24b3b8c21e251f39ee7aa2a74ee92c6d6ce3d85c38151a53",
}
LEGACY_HELPER_LEN = {1: 784, 2: 13378, 5: 20866}


def _refuse(reason: str):
    raise SystemExit(f"{P}: refusing to rewrite the scheduler: {reason}")


def _site_reasons(span: str, version: int, exact=None) -> str:
    """Return '' when `span` is exactly this installer's canonical v{version} helper site.

    v1/v2/v5 are pinned to their published helper text (sha256 and length); v6 is
    compared against this installer's own `_helper_text()`. A version with no
    published text has no accepted site at all.
    """
    if exact is not None:
        return "" if span == exact else f"helper site is not this installer's v{version} helper text"
    expected = LEGACY_HELPER_SHA256.get(version)
    if expected is None:
        return f"v{version} helper text was never published; there is no accepted site to compare"
    got = hashlib.sha256(span.encode("utf-8")).hexdigest()
    if got != expected or len(span) != LEGACY_HELPER_LEN[version]:
        return (f"helper site is not the published v{version} helper (sha256 {got[:16]}, "
                f"{len(span)} bytes; expected {expected[:16]}, {LEGACY_HELPER_LEN[version]} bytes)")
    return ""


def _legacy_span(text: str, version: int, exact=None):
    """Validate and return (start, end, span) for the unique v{version} helper site.

    Returns None when no helper definition is present at all. Refuses (no write)
    on a missing, duplicated, decorated, drifted or otherwise unattested site.
    """
    head_token = V1_HELPER_START if version == 1 else CLASS_HEAD
    found = text.count(head_token)
    if found == 0:
        return None
    if found != 1:
        _refuse(f"{found} {head_token!r} definitions found; v{version} installs exactly one")
    head = text.find(head_token)
    if head < 2 or text[head - 1] != "\n" or text[head - 2] != "\n":
        _refuse(f"v{version} helper is not preceded by the installer's blank-line boundary "
                f"(indented or decorated site)")
    start = head - 1
    reasons = []
    for anchor in (NEEDLE, ADAPTIVE_K_HEAD):
        end = text.find(anchor, head)
        if end < 0:
            continue
        span = text[start:end]
        why = _site_reasons(span, version, exact)
        if not why:
            return start, end, span
        reasons.append(why)
    _refuse(f"v{version} helper site rejected: " + "; ".join(reasons))


def _unpatch(text: str, version: int, exact=None):
    """Invert version `version`: its frozen gate sites, then its validated helper site."""
    if version not in LEGACY_PAIRS:
        _refuse(f"v{version} is not a migratable version: its helper text was never published")
    for new, old, label in LEGACY_PAIRS[version]:
        if new not in text:
            _refuse(f"{label} insertion is missing or drifted")
        text = replace_once(text, new, old, label)
    site = _legacy_span(text, version, exact)
    if site is None:
        _refuse(f"v{version} helper site is missing")
    start, end, span = site
    clean = text[:start] + text[end:]
    mark = LEGACY_MARK[version]
    leftover = [s for s in (mark, V1_HELPER_START if version == 1 else CLASS_HEAD) if s in clean]
    if leftover:
        _refuse(f"v{version} marker or helper definition still present after unpatch")
    return clean, start, span


def _round_trip(original: str, clean: str, version: int, start: int, span: str) -> None:
    """Refuse unless re-applying version `version` to `clean` reproduces `original`."""
    again = clean[:start] + span + clean[start:]
    for new, old, label in LEGACY_PAIRS[version]:
        again = replace_once(again, old, new, label)
    if again != original:
        _refuse(f"v{version} round trip is not byte-identical; the image is not a canonical "
                f"v{version} install")


def _helper_text() -> str:
    body = inspect.getsource(_Glm53MixedPrefill)
    return (
        "\n"
        + body.replace(MARK_V5, MARK_V6)
        + f"\n_GLM53_MIXED = _Glm53MixedPrefill()  # {MARK_V6}\n\n"
        + f"def _glm53_mixed_prefill_policy(sched, request, computed=None):  # {MARK_V6}\n"
        + "    return _GLM53_MIXED.cap_for(sched, request, computed)\n\n\n"
    )


def apply_v6(text: str) -> str:
    if "import os\n" not in text.split("import time\n", 1)[0]:
        text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import os")
    text = replace_once(text, NEEDLE, _helper_text() + NEEDLE, "helper")
    for new, old, label in V6_PAIRS:
        text = replace_once(text, old, new, label)
    compile(text, str(P), "exec")
    return text


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    original = text
    if MARK_V6 in text:
        # Validate existing anchors and helper instead of trusting the marker: a
        # marker-only or hand-edited patch cannot round-trip and is refused
        # without a write.
        clean, start, span = _unpatch(text, 6, exact=_helper_text())
        if apply_v6(clean) != text:
            _refuse("v6 helper drifted; unpatch/re-patch is not byte-identical")
        print(f"{P.name}: {MARK_V6} already present — verified")
        return 0
    for unsupported in (3, 4):
        if LEGACY_MARK[unsupported] in text:
            _refuse(f"v{unsupported} is refused as unsupported: no authenticated producer of that "
                    f"helper body was recovered from public history, so a canonical v{unsupported} "
                    f"image cannot be established and the source is left unchanged")
    version = next((v for v in (5, 2, 1)
                    if LEGACY_MARK[v] in text or (v == 1 and V1_HELPER_START in text)), None)
    if version is not None:
        clean, start, span = _unpatch(text, version)
        _round_trip(text, clean, version, start, span)
        text = clean
    text = apply_v6(text)
    if MARK_V2 in text or MARK_V3 in text or MARK_V4 in text or MARK_V5 in text:
        raise SystemExit(f"{P}: older marker left after migration")
    if text != original:
        P.write_text(text)
    print(f"patched {P.name} ({MARK_V6})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
