#!/usr/bin/env python3
"""Regression tests: fair-prefill candidate ranking honors request priority.

Under ``--scheduling-policy priority`` a lower numeric priority must win among
eligible prefills; within a tier the service-age/round-robin order is kept.
Under FCFS, priority hints are ignored. Service-time credit, chunk limits and
running-decoder protection are unchanged.
"""
from __future__ import annotations

import contextlib
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_scheduler_decode_floor as upstream


class PrefillPriorityTests(unittest.TestCase):
    def candidates(self, policy, owner_priority=-10):
        clock = upstream.Clock()
        with patch.dict(os.environ, upstream.FAIR_ENV), contextlib.redirect_stdout(io.StringIO()):
            fair = upstream.POLICY(now=clock)
        decoder = upstream.Req("decoder", prompt=100, computed=100, decode=True)
        family = upstream.Req("family")
        family.priority = 0
        owner = upstream.Req("owner")
        owner.priority = owner_priority
        scheduler = upstream.Sched([decoder], [family])
        scheduler.policy = SimpleNamespace(value=policy)
        fair.begin_step(scheduler)
        clock.advance(0.5)
        scheduler.waiting = [owner, family]
        scheduler.refresh()
        scheduler.current_step += 1
        fair.begin_step(scheduler)
        return [r.request_id for r in fair._candidates], fair.selected

    def test_priority_owner_beats_older_family_prefill(self):
        candidates, selected = self.candidates("priority")
        self.assertEqual(candidates, ["owner", "family"])
        self.assertEqual(selected, {"owner"})

    def test_fcfs_keeps_oldest_prefill_first_despite_priority_hint(self):
        candidates, selected = self.candidates("fcfs")
        self.assertEqual(candidates, ["family", "owner"])
        self.assertEqual(selected, {"family"})

    def test_equal_priority_keeps_existing_fairness(self):
        candidates, _ = self.candidates("priority", owner_priority=0)
        self.assertEqual(candidates, ["family", "owner"])


if __name__ == "__main__":
    unittest.main()
