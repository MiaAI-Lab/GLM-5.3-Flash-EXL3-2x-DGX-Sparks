#!/usr/bin/env python3
"""Published helper texts of the legacy mixed-prefill installer versions.

Byte-exact artefacts, recovered from the public history of
MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks and used as migration fixtures.
Do not reformat: the installer's registry keys on the sha256 of these strings.

  v1  f3043c95bbf9 overlay/patch_scheduler_decode_floor.py:HELPER
      cross-attested byte-identical by cad980f7263:HELPER_V1, which documents
      itself as "byte-identical to the shipped v1 overlay"
  v2  9a23b2d8f061 overlay/patch_scheduler_decode_floor.py:HELPER
  v5  a9bbbadc4c72 overlay/patch_scheduler_decode_floor.py:_helper_text()

v3 and v4 are absent on purpose: 180725a5ce33 introduced their markers as
migration targets for intermediate builds whose helper bodies were never
recovered from public history, so the installer refuses those markers without
touching the source instead of migrating on a guessed body.
"""
from __future__ import annotations

HELPERS = {}

HELPERS[1] = r'''
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

HELPERS[2] = r'''
class _Glm53MixedPrefill:  # [glm53-decode-floor:v2]
    """Skip / cap / off / fair mixed-prefill policy. CPU-only; no GPU sync."""

    def __init__(self, now=None) -> None:
        self._now = now or time.monotonic
        self.mode = "skip"
        self.legacy_cap = 0
        self.chunk = 256
        self.share = 0.20
        self.interval_s = 2.0
        self.max_chunks = 1
        self.hist_every = 50
        self.logged_boot = False
        self._parse()
        self.last_service = {}
        self.arrival = {}
        self.rr_seq = {}
        self.rr_n = 0
        self.step_prefill_ids = set()
        self.inflight = []
        self.inflight_mixed = 0
        self.history = []
        self.last_schedule_mono = None
        self.last_prefill_turn_mono = 0.0
        self.step_tag = None
        self._sched_id = None
        self.step_mode = "solo"
        self.selected = set()
        self.defer_reason = "none"
        self.granted_this_step = 0
        self.steps = 0

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
        if self.mode == "fair" and not self.logged_boot:
            print(
                f"[glm53-decode-floor] fair chunk={self.chunk} share={self.share} "
                f"interval_s={self.interval_s} max_chunks={self.max_chunks}",
                flush=True,
            )
            self.logged_boot = True

    @staticmethod
    def prefill_remaining(request) -> int:
        """Tokens still requiring prefill/replay compute (not the decode row)."""
        prompt = int(getattr(request, "num_prompt_tokens", 0) or 0)
        computed = int(getattr(request, "num_computed_tokens", 0) or 0)
        num_tokens = int(getattr(request, "num_tokens", prompt) or prompt)
        if computed < prompt:
            return prompt - computed
        prefill_end = max(prompt, num_tokens - 1 if num_tokens > 0 else prompt)
        return max(0, prefill_end - computed)

    def needs_prefill_compute(self, request) -> bool:
        return self.prefill_remaining(request) > 0

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
        for store in (self.last_service, self.arrival, self.rr_seq):
            dead = [k for k in store if k not in live]
            for k in dead:
                store.pop(k, None)

    def _recent_share(self, now):
        window = 8.0
        while self.history and now - self.history[0][0] > window:
            self.history.pop(0)
        prefill_s = sum(dt for _, dt, mix in self.history if mix)
        decode_s = sum(dt for _, dt, mix in self.history if not mix)
        total = prefill_s + decode_s
        if total <= 0:
            return 0.0, 0.0, 0.0
        return prefill_s / total, prefill_s, decode_s

    def _oldest_wait(self, now, prefills):
        oldest = 0.0
        for r in prefills:
            rid = getattr(r, "request_id", None)
            if rid is None:
                continue
            served = self.last_service.get(rid)
            if served is None:
                oldest = max(oldest, now - self.arrival.get(rid, now))
            else:
                oldest = max(oldest, now - served)
        return oldest

    def _select(self, prefills):
        ranked = []
        for r in prefills:
            rid = getattr(r, "request_id", None)
            if rid is None:
                continue
            ranked.append((self.last_service.get(rid, 0.0), self.rr_seq.get(rid, 0), rid, r))
        ranked.sort(key=lambda x: (x[0], x[1], x[2]))
        out = []
        for _, _, rid, r in ranked[: self.max_chunks]:
            out.append(r)
        return out

    def _flush_prior_schedule(self) -> None:
        if self.last_schedule_mono is None:
            return
        ids = set(self.step_prefill_ids)
        self.inflight.append(ids)
        if ids:
            self.inflight_mixed += 1
        self.step_prefill_ids = set()
        self.last_schedule_mono = None

    def begin_step(self, sched) -> None:
        now = self._now()
        sid = id(sched)
        if self._sched_id is not None and self._sched_id != sid:
            self.inflight.clear()
            self.inflight_mixed = 0
            self.step_prefill_ids = set()
            self.last_schedule_mono = None
        self._sched_id = sid
        tag = (sid, int(getattr(sched, "current_step", 0) or 0))
        if tag == self.step_tag:
            return
        self._flush_prior_schedule()
        self.step_tag = tag
        self.steps += 1
        self.granted_this_step = 0
        self.selected = set()
        self.step_prefill_ids = set()
        self.last_schedule_mono = now
        sched._glm53_align_prefill_limit = None
        live = self._live_ids(sched)
        self._prune(live)
        running = list(getattr(sched, "running", None) or [])
        waiting = list(self._iter_waiting(sched))
        for r in running + waiting:
            rid = getattr(r, "request_id", None)
            if rid is not None and rid not in self.arrival:
                self.arrival[rid] = now
        prefills = [r for r in running + waiting if self.needs_prefill_compute(r)]
        decodes = [r for r in running if not self.needs_prefill_compute(r)]
        if self.mode != "fair":
            self.step_mode = "legacy"
            self.defer_reason = "none"
            return
        if not prefills:
            self.step_mode = "solo"
            self.defer_reason = "none"
            return
        if not decodes:
            self.step_mode = "solo"
            self.defer_reason = "none"
            return
        if self.inflight_mixed > 0:
            self.step_mode = "decode_only"
            self.defer_reason = "async_inflight"
            self._maybe_log(now, len(prefills), len(decodes))
            return
        share, _, _ = self._recent_share(now)
        since = now - self.last_prefill_turn_mono if self.last_prefill_turn_mono else self.interval_s
        starved = self._oldest_wait(now, prefills) >= self.interval_s
        want = (
            self.last_prefill_turn_mono == 0.0
            or share < self.share
            or since >= self.interval_s
            or starved
        )
        if not want:
            self.step_mode = "decode_only"
            self.defer_reason = "share"
            self._maybe_log(now, len(prefills), len(decodes))
            return
        chosen = self._select(prefills)
        self.selected = {getattr(r, "request_id", None) for r in chosen}
        self.selected.discard(None)
        self.step_mode = "prefill_turn"
        self.defer_reason = "none"
        self._maybe_log(now, len(prefills), len(decodes))

    def _maybe_log(self, now, n_prefill, n_decode) -> None:
        if self.hist_every <= 0 or self.steps % self.hist_every != 1:
            return
        share, pre_s, dec_s = self._recent_share(now)
        print(
            f"[glm53-decode-floor] step={self.steps} mode={self.step_mode} "
            f"defer={self.defer_reason} inflight={self.inflight_mixed} "
            f"prefill={n_prefill} decode={n_decode} selected={len(self.selected)} "
            f"share={share:.3f} pre_s={pre_s:.3f} dec_s={dec_s:.3f}",
            flush=True,
        )

    def cap_for(self, sched, request):
        self.begin_step(sched)
        if not self.needs_prefill_compute(request):
            sched._glm53_align_prefill_limit = None
            return None
        if self.mode == "off":
            sched._glm53_align_prefill_limit = None
            return None
        running = list(getattr(sched, "running", None) or [])
        rid = getattr(request, "request_id", None)
        peer_decode = False
        for r in running:
            if r is request or getattr(r, "request_id", None) == rid:
                continue
            if not self.needs_prefill_compute(r):
                peer_decode = True
                break
        if self.mode == "skip":
            cap = 0 if peer_decode else None
        elif self.mode == "cap":
            cap = self.legacy_cap if peer_decode else None
        else:
            # fair
            if self.step_mode == "solo":
                cap = None
            elif self.step_mode != "prefill_turn":
                cap = 0
            elif rid not in self.selected:
                cap = 0
            elif self.granted_this_step >= self.max_chunks:
                cap = 0
            else:
                cap = self.chunk
        if cap is not None and cap > 0:
            sched._glm53_align_prefill_limit = cap
            self.step_prefill_ids.add(rid)
            self.granted_this_step += 1
        else:
            sched._glm53_align_prefill_limit = None
        return cap

    def note_scheduled(self, request, num_new_tokens: int) -> None:
        """Keep step_prefill_ids as actual post-alignment prefill tokens only."""
        rid = getattr(request, "request_id", None)
        if rid is None:
            return
        if num_new_tokens <= 0 or not self.needs_prefill_compute(request):
            self.step_prefill_ids.discard(rid)
            return
        self.step_prefill_ids.add(rid)

    def observe_output(self, sched, scheduler_output) -> None:
        now = self._now()
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        if self.inflight:
            ids = self.inflight.pop(0)
            if ids:
                self.inflight_mixed = max(0, self.inflight_mixed - 1)
        elif self.last_schedule_mono is not None:
            ids = set(self.step_prefill_ids)
            self.step_prefill_ids = set()
            self.last_schedule_mono = None
        else:
            ids = set()
        served = set()
        for rid in ids:
            n = int(num_scheduled.get(rid, 0) or 0)
            if n > 0:
                self.last_service[rid] = now
                self.rr_n += 1
                self.rr_seq[rid] = self.rr_n
                served.add(rid)
        mixed = bool(served)
        start = getattr(self, "_observe_prev", None)
        dt = 0.0 if start is None else max(0.0, now - start)
        self._observe_prev = now
        if dt > 0:
            self.history.append((now, dt, mixed))
        if mixed:
            self.last_prefill_turn_mono = now
        live = self._live_ids(sched)
        self._prune(live)

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


_GLM53_MIXED = _Glm53MixedPrefill()  # [glm53-decode-floor:v2]


def _glm53_mixed_prefill_policy(sched, request):  # [glm53-decode-floor:v2]
    return _GLM53_MIXED.cap_for(sched, request)


'''

HELPERS[5] = r'''
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

_GLM53_MIXED = _Glm53MixedPrefill()  # [glm53-decode-floor:v5]

def _glm53_mixed_prefill_policy(sched, request, computed=None):  # [glm53-decode-floor:v5]
    return _GLM53_MIXED.cap_for(sched, request, computed)


'''
