#!/usr/bin/env python3
"""CPU regressions for prompt progress while a peer keeps generating.

These cases exercise the policy text the overlay actually installs
(``_helper_text()``), so a drifted copy cannot pass. Patch integrity, version
migration, idempotence and refusal live in ``tests/test_scheduler_decode_floor.py``;
set ``GLM53_SCHEDULER_PY_SRC`` to an installed ``scheduler.py`` to run that
matrix against the deployed source as well.
"""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay" / "patch_scheduler_decode_floor.py"
spec = importlib.util.spec_from_file_location("glm53_prefill_v6", PATCH)
OVERLAY = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = OVERLAY
spec.loader.exec_module(OVERLAY)
HELPER_NS: dict = {"os": os, "time": __import__("time")}
exec(OVERLAY._helper_text(), HELPER_NS)
POLICY = HELPER_NS["_Glm53MixedPrefill"]
MIXED_KEYS = ("GLM53_MIXED_PREFILL_CHUNK", "GLM53_MIXED_PREFILL_WARM_TOKENS",
              "GLM53_MIXED_PREFILL_MAX_WAIT_MS", "GLM53_MIXED_PREFILL_LATE_CAP")


class Req:
    def __init__(self, rid, prompt, computed=0, decode=False):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self.num_computed_tokens = computed
        self.num_tokens = prompt + int(decode)
        self.spec_token_ids = list(range(7)) if decode else []
        self.num_output_placeholders = 0
        self.has_encoder_inputs = False
        self.is_prefill_chunk = not decode

    @property
    def num_tokens_with_spec(self):
        return self.num_tokens + len(self.spec_token_ids)


class Sched:
    def __init__(self, running, waiting):
        self.running = list(running)
        self.waiting = list(waiting)
        self.skipped_waiting = []
        self.current_step = 1
        self.max_model_len = 850000
        self.num_sampled_tokens_per_step = 1
        self.need_mamba_block_aligned_split = False
        self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=3584)
        self.requests = {r.request_id: r for r in self.running + self.waiting}


class PrefillProgressTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        for key in MIXED_KEYS:
            os.environ.pop(key, None)
        self.peer = Req("long-generation", 512, 512, decode=True)
        self.newcomer = Req("new-agent", 1200)
        self.sched = Sched([self.peer], [self.newcomer])

    def policy(self, **env):
        with patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            p = POLICY(now=lambda: 1000.0)
        p.hist_every = 0
        return p

    def run_steps(self, policy, steps=64, step_budget=1024, request=None):
        """Offer the newcomer the remaining step budget, honouring the policy cap."""
        request = request or self.newcomer
        for _ in range(steps):
            cap = policy.cap_for(self.sched, request)
            if cap is None or cap > 0:
                budget = step_budget - 8  # the peer's own decode row
                if cap is not None:
                    budget = min(budget, cap)
                request.num_computed_tokens += min(
                    budget, request.num_prompt_tokens - request.num_computed_tokens)
            self.sched.current_step += 1
        return request.num_computed_tokens

    def test_stock_modes_let_the_newcomer_finish_while_the_peer_decodes(self):
        # 0 / off / no = no extra isolation: the base scheduler keeps its budget.
        for mode in ("0", "off", "no"):
            with self.subTest(mode=mode):
                self.newcomer.num_computed_tokens = 0
                self.assertEqual(self.run_steps(self.policy(GLM53_MIXED_PREFILL_CHUNK=mode)),
                                 self.newcomer.num_prompt_tokens)

    def test_skip_hold_defers_the_newcomer_until_the_peer_stops(self):
        # `skip` (TP=4 default) gives the newcomer zero tokens while any peer decodes.
        newcomer = self.newcomer
        self.assertEqual(self.run_steps(self.policy(GLM53_MIXED_PREFILL_CHUNK="skip")), 0)
        self.sched.running = []
        self.sched.requests = {newcomer.request_id: newcomer}
        self.assertEqual(self.run_steps(self.policy(GLM53_MIXED_PREFILL_CHUNK="skip")),
                         newcomer.num_prompt_tokens)

    def test_bounded_chunk_progresses_under_the_cap(self):
        steps = -(-self.newcomer.num_prompt_tokens // 128)  # ceil: 1200 -> 10 steps at 128/step
        self.assertEqual(self.run_steps(self.policy(GLM53_MIXED_PREFILL_CHUNK="128"), steps=steps),
                         self.newcomer.num_prompt_tokens)

    def test_solo_prefill_is_never_capped(self):
        solo = Req("solo", 900)
        self.sched.running, self.sched.waiting = [solo], []
        self.sched.requests = {solo.request_id: solo}
        for mode in ("skip", "128"):
            with self.subTest(mode=mode):
                solo.num_computed_tokens = 0
                self.assertEqual(self.run_steps(self.policy(GLM53_MIXED_PREFILL_CHUNK=mode),
                                                steps=1, request=solo), solo.num_prompt_tokens)

    def test_request_cannot_count_itself_as_a_decoding_peer(self):
        # Only a DIFFERENT request that needs no prefill is a peer: a request that
        # appears in `running` must not be held by its own row.
        self.sched.running, self.sched.waiting = [self.newcomer], []
        self.sched.requests = {self.newcomer.request_id: self.newcomer}
        self.newcomer.num_computed_tokens = 600
        p = self.policy(GLM53_MIXED_PREFILL_CHUNK="skip")
        self.assertIsNone(p.cap_for(self.sched, self.newcomer))
        self.assertEqual(self.run_steps(p, steps=1), self.newcomer.num_prompt_tokens)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
