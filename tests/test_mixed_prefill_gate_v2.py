#!/usr/bin/env python3
"""CPU behavior tests for the opt-in mixed-prefill gate (v6: warm bypass + deadline).

The gate is an operator opt-in: ``GLM53_MIXED_PREFILL_WARM_TOKENS`` and
``GLM53_MIXED_PREFILL_MAX_WAIT_MS`` both default to 0, and 0 disables the
feature. These cases pin that default-off contract, the two transitions, the
explicit-cap precedence, and that ``off``/``fair`` are not affected. They run
against the helper text the installer actually writes into scheduler.py
(``_helper_text()``), so a drifted copy cannot pass.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay/patch_scheduler_decode_floor.py"
spec = importlib.util.spec_from_file_location("glm53_gate_v6", PATCH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

HELPER_NS: dict = {"os": os, "time": __import__("time")}
exec(mod._helper_text(), HELPER_NS)
POLICY = HELPER_NS["_Glm53MixedPrefill"]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class Req:
    def __init__(self, rid, prompt=30000, computed=0, decode=False):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self.num_computed_tokens = computed
        self.num_tokens = prompt + int(decode)
        self.spec_token_ids = list(range(7)) if decode else []
        self.num_output_placeholders = 0
        self.next_decode_eligible_step = 0
        self.max_tokens = 4000
        self.has_encoder_inputs = False
        self.is_prefill_chunk = not decode

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)


class Sched:
    def __init__(self, running=(), waiting=()):
        self.running = list(running)
        self.waiting = list(waiting)
        self.skipped_waiting = []
        self.current_step = 1
        self.max_model_len = 850000
        self.num_sampled_tokens_per_step = 1
        self.need_mamba_block_aligned_split = False
        self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=3584)
        self.refresh()

    def refresh(self):
        self.requests = {r.request_id: r for r in self.running + self.waiting + self.skipped_waiting}


class GateV6Tests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.decoder = Req("A", 30000, 30000, decode=True)
        self.prefill = Req("B", 30000, 0)
        self.sched = Sched([self.decoder], [self.prefill])

    def policy(self, **env):
        with patch.dict(os.environ, {"GLM53_MIXED_PREFILL_CHUNK": "skip", **env}), \
                contextlib.redirect_stdout(io.StringIO()):
            p = POLICY(now=self.clock)
        p.hist_every = 0
        return p

    def gate(self, policy, request=None):
        """One gate evaluation in a fresh scheduler step (a requeue/chunk boundary)."""
        self.sched.current_step += 1
        return policy.cap_for(self.sched, request or self.prefill)

    def test_defaults_disable_both_features(self):
        p = self.policy()
        self.assertEqual((p.warm_tokens, p.max_wait_ms, p.late_cap), (0, 0, 512))
        # No held-request release: the skip hold is byte-for-byte the v5 policy
        # even long after any deadline an operator could have set.
        self.assertEqual(self.gate(p), 0)
        self.clock.advance(600.0)
        self.assertEqual(self.gate(p), 0)

    def test_zero_disables_warm_bypass_for_the_smallest_remainder(self):
        # The pathological "bypass when cached >= 0" form would admit everything
        # here; 0 must mean the feature is off, not "everything is warm".
        self.prefill.num_computed_tokens = self.prefill.num_prompt_tokens - 1
        for zero in ("0", "00"):
            p = self.policy(GLM53_MIXED_PREFILL_WARM_TOKENS=zero)
            self.assertEqual(p.warm_tokens, 0)
            self.assertEqual(self.gate(p), 0)
        p = self.policy(GLM53_MIXED_PREFILL_WARM_TOKENS="1")
        self.assertIsNone(self.gate(p))

    def test_warm_bypass_transition_is_inclusive_and_bounded(self):
        p = self.policy(GLM53_MIXED_PREFILL_WARM_TOKENS="3584")
        self.prefill.num_computed_tokens = 30000 - 3584
        self.assertIsNone(self.gate(p))
        self.prefill.num_computed_tokens = 30000 - 3585
        self.assertEqual(self.gate(p), 0)
        # A request that still needs real work is never admitted by the bypass,
        # and a solo prefill is unaffected either way.
        self.sched.running, self.sched.waiting = [self.prefill], []
        self.sched.refresh()
        self.assertIsNone(self.gate(p))

    def test_deadline_releases_a_starved_request_under_late_cap(self):
        p = self.policy(GLM53_MIXED_PREFILL_MAX_WAIT_MS="1500", GLM53_MIXED_PREFILL_LATE_CAP="512")
        self.assertEqual(self.gate(p), 0)
        self.clock.advance(1.4)
        self.assertEqual(self.gate(p), 0)
        self.clock.advance(0.1)
        self.assertEqual(self.gate(p), 512)
        # LATE_CAP is a cap, not a quota: a smaller remainder is not padded.
        self.prefill.num_computed_tokens = 30000 - 100
        self.assertEqual(self.gate(p), 100)

    def test_deadline_zero_waits_forever(self):
        p = self.policy(GLM53_MIXED_PREFILL_MAX_WAIT_MS="0")
        self.assertEqual(self.gate(p), 0)
        self.clock.advance(600.0)
        self.assertEqual(self.gate(p), 0)

    def test_wait_survives_requeue_and_chunking(self):
        # Chunk/preempt boundaries are new steps; the first-seen stamp must not
        # restart, or a requeued request would starve for another full interval.
        p = self.policy(GLM53_MIXED_PREFILL_MAX_WAIT_MS="1500")
        self.assertEqual(self.gate(p), 0)
        self.clock.advance(0.9)
        self.assertEqual(self.gate(p), 0)
        self.prefill.num_computed_tokens = 29000
        self.clock.advance(0.6)
        self.assertEqual(self.gate(p), 512)

    def test_explicit_cap_wins_and_only_gains_the_warm_bypass(self):
        capped = self.policy(GLM53_MIXED_PREFILL_CHUNK="256", GLM53_MIXED_PREFILL_MAX_WAIT_MS="1")
        self.assertEqual(self.gate(capped), 256)
        self.assertEqual(self.sched._glm53_align_prefill_limit, 256)
        self.clock.advance(60.0)
        self.assertEqual(self.gate(capped), 256)
        warmed = self.policy(GLM53_MIXED_PREFILL_CHUNK="256", GLM53_MIXED_PREFILL_WARM_TOKENS="3584")
        self.prefill.num_computed_tokens = 30000 - 100
        self.assertIsNone(self.gate(warmed))
        self.assertIsNone(self.sched._glm53_align_prefill_limit)

    def test_fair_and_off_modes_ignore_the_gate_knobs(self):
        fair = self.policy(GLM53_MIXED_PREFILL_CHUNK="fair",
                           GLM53_MIXED_PREFILL_WARM_TOKENS="1000000",
                           GLM53_MIXED_PREFILL_MAX_WAIT_MS="1")
        fair.begin_step(self.sched)
        fair.credit = -10.0
        self.assertEqual(fair.cap_for(self.sched, self.prefill), 0)
        self.assertEqual(fair.defer_reason, "credit")
        self.assertEqual(self.policy(GLM53_MIXED_PREFILL_CHUNK="off",
                                     GLM53_MIXED_PREFILL_WARM_TOKENS="3584",
                                     GLM53_MIXED_PREFILL_MAX_WAIT_MS="1").cap_for(
                                         Sched([self.decoder], [self.prefill]), self.prefill), None)

    def test_out_of_range_knobs_fall_back_to_disabled(self):
        for env, attr, expected in (
            ({"GLM53_MIXED_PREFILL_WARM_TOKENS": "abc"}, "warm_tokens", 0),
            ({"GLM53_MIXED_PREFILL_WARM_TOKENS": "-1"}, "warm_tokens", 0),
            ({"GLM53_MIXED_PREFILL_WARM_TOKENS": "1000001"}, "warm_tokens", 0),
            ({"GLM53_MIXED_PREFILL_MAX_WAIT_MS": "700000"}, "max_wait_ms", 0),
            ({"GLM53_MIXED_PREFILL_MAX_WAIT_MS": "1500"}, "max_wait_ms", 1500),
            ({"GLM53_MIXED_PREFILL_LATE_CAP": "1"}, "late_cap", 512),
            ({"GLM53_MIXED_PREFILL_LATE_CAP": "8192"}, "late_cap", 8192),
        ):
            p = self.policy(**env)
            self.assertEqual(getattr(p, attr), expected, env)
        # A rejected warm/deadline value leaves the knob's feature off, not half-on.
        self.assertEqual(self.gate(self.policy(GLM53_MIXED_PREFILL_MAX_WAIT_MS="junk")), 0)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
