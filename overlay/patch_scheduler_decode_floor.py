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
  fair       — service-time mixing (default on TP=2/3/4 since 2026-09-15,
               v5). Decode-only
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
                completed positive prefill service plus round-robin within each
                request-priority tier when scheduling-policy=priority. Decode
               token/input budget is allocated first by the base scheduler.
               Solo prefill retains the base scheduler's limits. Timing is a
               host busy-time proxy.

Fair knobs (read at runtime; identical on every rank):
  GLM53_FAIR_PREFILL_CHUNK            default 256 (probe size until timing samples exist)
  GLM53_FAIR_PREFILL_SHARE            default 0.30 (credit accrual fraction)
  GLM53_FAIR_PREFILL_MAX_INTERVAL_MS  default 2000
  GLM53_FAIR_PREFILL_MAX_STEP_MS      default 2000 (estimated mixed-step limit)
  GLM53_FAIR_PREFILL_MAX_CHUNKS       default 1

Versioned installer: `# [glm53-decode-floor:v5]`. v1 (no version), v2, v3
and v4 images are unpatched then re-patched. Fail closed if anchors drift.
"""
from __future__ import annotations

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
PRIORITY_MARK = "# [glm53-prefill-priority:v1]"
LEGACY_RANK_PREFIX = """    def _rank_prefills(self, prefills):
        return sorted(prefills, key=lambda r: (
            self.last_service.get(r.request_id, 0.0),
"""
PRIORITY_RANK_PREFIX = """    def _rank_prefills(self, prefills):
        return sorted(prefills, key=lambda r: (
            # [glm53-prefill-priority:v1] Preserve fairness within priority tiers.
            int(getattr(r, "priority", 0) or 0) if getattr(self, "priority_mode", False) else 0,
            self.last_service.get(r.request_id, 0.0),
"""
LEGACY_CANDIDATES = """        self._candidates = self._rank_prefills(prefills)
"""
PRIORITY_CANDIDATES = """        policy = getattr(sched, "policy", None)
        self.priority_mode = getattr(policy, "value", policy) == "priority"
        self._candidates = self._rank_prefills(prefills)
"""

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



# Frozen v3 anchors for migration from the reviewed implementation.
V3_BEGIN_NEW = """        self.current_step += 1
        _GLM53_MIXED.begin_step(self)  # [glm53-decode-floor:v3]
        # NOTE(woosuk) on the scheduling algorithm:
"""
V3_OBS_NEW = """        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        _GLM53_MIXED.observe_output(self, scheduler_output)  # [glm53-decode-floor:v3]
        pooler_outputs = model_runner_output.pooler_output
"""
V3_FIN_OLD = """        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
"""
V3_FIN_NEW = """        # Check if the scheduling constraints are satisfied.
        _GLM53_MIXED.note_schedule_output(self, num_scheduled_tokens)  # [glm53-decode-floor:v3]
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
"""
V3_RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v3]
            if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                num_new_tokens = min(num_new_tokens, mixed_cap)
            num_new_tokens = _GLM53_MIXED.clip_for_decode_reserve(
                num_new_tokens,
                token_budget,
                int(getattr(self, "_glm53_decode_reserve_tokens", 0) or 0),
                _GLM53_MIXED.needs_prefill_compute(request),
            )

            # Make sure the input position does not exceed the max model len.
"""
V3_WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self, request)  # [glm53-decode-floor:v3]
                    if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)
                    num_new_tokens = _GLM53_MIXED.clip_for_decode_reserve(
                        num_new_tokens,
                        token_budget,
                        int(getattr(self, "_glm53_decode_reserve_tokens", 0) or 0),
                        _GLM53_MIXED.needs_prefill_compute(request),
                    )
                    if mixed_cap is not None and _GLM53_MIXED.needs_prefill_compute(request):
                        if num_new_tokens <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                    # chunked prefill has to be enabled explicitly to allow
"""
V3_ALIGN_NEW = """            max_prefill_tokens = self.max_num_scheduled_tokens
            long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            _align_cap = getattr(self, "_glm53_align_prefill_limit", None)  # [glm53-decode-floor:v3]
            if _align_cap is not None and _align_cap > 0:
                max_prefill_tokens = min(max_prefill_tokens, _align_cap)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end
"""
V3_RUNNING_MAMBA_NEW = """            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )
            _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v3]
"""
V3_WAITING_MAMBA_NEW = """                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        _GLM53_MIXED.note_scheduled(request, num_new_tokens)  # [glm53-decode-floor:v3]
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
            self.share = float(self._e("GLM53_FAIR_PREFILL_SHARE", "0.30"))
        except ValueError:
            self.share = 0.30
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
            self.max_step_s = max(0.001, int(self._e("GLM53_FAIR_PREFILL_MAX_STEP_MS", "2000")) / 1000.0)
        except ValueError:
            self.max_step_s = 2.0
        if self.mode == "fair" and not self.logged_boot:
            print(
                f"[glm53-decode-floor] fair v5 probe_chunk={self.chunk} "
                f"ladder={min(self.LADDER)}..{max(self.LADDER)} share={self.share} "
                f"interval_s={self.interval_s} max_step_s={self.max_step_s} "
                f"max_chunks={self.max_chunks}",
                flush=True,
            )
            self.logged_boot = True


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
            # [glm53-prefill-priority:v1] Preserve fairness within priority tiers.
            int(getattr(r, "priority", 0) or 0) if getattr(self, "priority_mode", False) else 0,
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
        policy = getattr(sched, "policy", None)
        self.priority_mode = getattr(policy, "value", policy) == "priority"
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

    def cap_for(self, sched, request, computed=None):
        self.begin_step(sched)
        sched._glm53_align_prefill_limit = None
        remaining = self.prefill_remaining(request, computed)
        if remaining <= 0 or self.mode == "off":
            return None
        peer_decode = any(r is not request and not self.needs_prefill_compute(r)
                          for r in sched.running)
        if self.mode == "skip":
            return 0 if peer_decode else None
        if self.mode == "cap":
            cap = self.legacy_cap if peer_decode else None
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



def _helper_text() -> str:
    body = inspect.getsource(_Glm53MixedPrefill)
    return (
        "\n"
        + body
        + "\n_GLM53_MIXED = _Glm53MixedPrefill()  # [glm53-decode-floor:v5]\n\n"
        + "def _glm53_mixed_prefill_policy(sched, request, computed=None):  # [glm53-decode-floor:v5]\n"
        + "    return _GLM53_MIXED.cap_for(sched, request, computed)\n\n\n"
    )


HELPER = None  # filled at apply time so tests can call _helper_text()


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


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def _strip_helper(text: str, label: str, *, expected: str | None = None) -> str:
    start = text.find("class _Glm53MixedPrefill:")
    if start < 0:
        raise SystemExit(f"{P}: {label} helper start not found")
    if start > 0 and text[start - 1] == "\n":
        start -= 1
    end_ak = text.find("class _Glm53AdaptiveK:", start)
    end_cg = text.find("from vllm.compilation.cuda_graph import CUDAGraphStat\n", start)
    candidates = [i for i in (end_ak, end_cg) if i > start]
    if not candidates:
        raise SystemExit(f"{P}: {label} helper end not found")
    end = min(candidates)
    if expected is not None and text[start:end].strip() != expected.strip():
        raise SystemExit(f"{P}: {label} helper drifted")
    return text[:start] + text[end:]


def _strip_v1_helper(text: str) -> str:
    start = text.find(V1_HELPER_START)
    if start < 0:
        raise SystemExit(f"{P}: v1 helper start not found")
    if start > 0 and text[start - 1] == "\n":
        start -= 1
    end_ak = text.find("class _Glm53AdaptiveK:", start)
    end_cg = text.find("from vllm.compilation.cuda_graph import CUDAGraphStat\n", start)
    candidates = [i for i in (end_ak, end_cg) if i > start]
    if not candidates:
        raise SystemExit(f"{P}: v1 helper end not found")
    end = min(candidates)
    return text[:start] + "\n" + text[end:]


def unpatch_v1(text: str) -> str:
    if V1_RUNNING_NEW in text:
        text = replace_once(text, V1_RUNNING_NEW, RUNNING_OLD, "v1-running")
    elif "_glm53_mixed_prefill_policy(self.running, request)" in text:
        raise SystemExit(f"{P}: v1 running insertion drifted")
    if V1_WAITING_NEW in text:
        text = replace_once(text, V1_WAITING_NEW, WAITING_OLD, "v1-waiting")
    elif "_glm53_mixed_prefill_policy(self.running, request)" in text:
        raise SystemExit(f"{P}: v1 waiting insertion drifted")
    if V1_HELPER_START in text:
        text = _strip_v1_helper(text)
    leftover = [
        "_glm53_mixed_prefill_policy(self.running, request)",
        V1_HELPER_START,
    ]
    for s in leftover:
        if s in text:
            raise SystemExit(f"{P}: v1 leftover after unpatch: {s}")
    return text


def unpatch_v2(text: str) -> str:
    pairs = (
        (V2_BEGIN_NEW, BEGIN_OLD, "v2-begin"),
        (V2_OBS_NEW, OBS_OLD, "v2-obs"),
        (V2_RUNNING_NEW, RUNNING_OLD, "v2-running"),
        (V2_WAITING_NEW, WAITING_OLD, "v2-waiting"),
        (V2_ALIGN_NEW, ALIGN_OLD, "v2-align"),
        (V2_RUNNING_MAMBA_NEW, RUNNING_MAMBA_OLD, "v2-running-mamba"),
        (V2_WAITING_MAMBA_NEW, WAITING_MAMBA_OLD, "v2-waiting-mamba"),
    )
    for new, old, label in pairs:
        if new in text:
            text = replace_once(text, new, old, label)
        elif MARK_V2 in new:
            # Some insertions may already have been removed; fail if marker remains.
            pass
    if "class _Glm53MixedPrefill:" in text:
        text = _strip_helper(text, "v2")
    if MARK_V2 in text:
        raise SystemExit(f"{P}: v2 leftover after unpatch")
    return text


def unpatch_v3(text: str) -> str:
    pairs = (
        (V3_BEGIN_NEW, BEGIN_OLD, "v3-begin"),
        (V3_OBS_NEW, OBS_OLD, "v3-obs"),
        (V3_FIN_NEW, V3_FIN_OLD, "v3-fin"),
        (V3_RUNNING_NEW, RUNNING_OLD, "v3-running"),
        (V3_WAITING_NEW, WAITING_OLD, "v3-waiting"),
        (V3_ALIGN_NEW, ALIGN_OLD, "v3-align"),
        (V3_RUNNING_MAMBA_NEW, RUNNING_MAMBA_OLD, "v3-running-mamba"),
        (V3_WAITING_MAMBA_NEW, WAITING_MAMBA_OLD, "v3-waiting-mamba"),
    )
    for new, old, label in pairs:
        text = replace_once(text, new, old, label)
    text = _strip_helper(text, "v3")
    if MARK_V3 in text:
        raise SystemExit(f"{P}: v3 leftover after unpatch")
    return text


def unpatch_v4(text: str) -> str:
    for new, old, label in V4_PAIRS:
        text = replace_once(text, new, old, label)
    text = _strip_helper(text, "v4")
    if MARK_V4 in text:
        raise SystemExit(f"{P}: v4 leftover after unpatch")
    return text


# v5 uses the same scheduler anchors as v4 with the marker advanced; the v4
# insertions above stay frozen so a v4 image can be unpatched exactly.
V5_PAIRS = tuple((new.replace(MARK_V4, MARK_V5), old, label) for new, old, label in V4_PAIRS)


def unpatch_v5(text: str) -> str:
    for new, old, label in V5_PAIRS:
        text = replace_once(text, new, old, label)
    text = _strip_helper(text, "v5", expected=_helper_text())
    if MARK_V5 in text:
        raise SystemExit(f"{P}: v5 leftover after unpatch")
    return text


def apply_v5(text: str) -> str:
    if "import os\n" not in text.split("import time\n", 1)[0]:
        text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import os")
    needle = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
    text = replace_once(text, needle, _helper_text() + needle, "helper")
    for new, old, label in V5_PAIRS:
        text = replace_once(text, old, new, label)
    compile(text, str(P), "exec")
    return text


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    original = text
    if MARK_V5 in text:
        if PRIORITY_MARK not in text:
            text = replace_once(text, LEGACY_RANK_PREFIX, PRIORITY_RANK_PREFIX, "prefill priority ranking")
            text = replace_once(text, LEGACY_CANDIDATES, PRIORITY_CANDIDATES, "prefill scheduler policy")
        # Validate existing anchors/helper instead of trusting the marker alone.
        # unpatch_v5 checks every v5 insertion occurs exactly once, strips the
        # helper, and rejects leftover markers. The result is discarded: a
        # later overlay may legitimately sit between the helper and the
        # cuda_graph import anchor, so re-applying at that fixed anchor would
        # relocate the helper and fail a byte-compare on a healthy file.
        unpatch_v5(text)
        # The import edit is part of the applied state; the byte-compare used
        # to cover it implicitly.
        if "import os\n" not in text.split("import time\n", 1)[0]:
            raise SystemExit(f"{P}: v5 import drifted")
        compile(text, str(P), "exec")
        if text != original:
            P.write_text(text)
            print(f"{P.name}: migrated prefill-priority ranking ({PRIORITY_MARK})")
        else:
            print(f"{P.name}: {MARK_V5} already present — verified")
        return 0
    if MARK_V4 in text:
        text = unpatch_v4(text)
    elif MARK_V3 in text:
        text = unpatch_v3(text)
    elif MARK_V2 in text:
        text = unpatch_v2(text)
    elif MARK in text or V1_HELPER_START in text:
        text = unpatch_v1(text)
    text = apply_v5(text)
    if MARK_V2 in text or MARK_V3 in text or MARK_V4 in text:
        raise SystemExit(f"{P}: older marker left after migration")
    if text != original:
        P.write_text(text)
    print(f"patched {P.name} ({MARK_V5})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
