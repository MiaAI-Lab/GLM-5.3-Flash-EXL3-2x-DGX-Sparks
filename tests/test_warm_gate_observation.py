#!/usr/bin/env python3
"""Inert fixtures for the warm-gate observation rules (no HTTP, no GPU).

The pure capture rules and the actual main entrypoint are exercised with inert
records, bytes, clocks and transports. These cases cover SSE errors, absolute
timestamps, missing peer activity, mixed-invalid verdict precedence and peer
failures discovered only after the observation windows. No socket or device is
opened.
"""
from __future__ import annotations

import contextlib
import io
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tests/validate_warm_gate.py"

# The module reads its knob values at import time; pin them so importing it here
# does not depend on the ambient environment. It never dials out at import.
with patch.dict(os.environ, {"GLM53_MIXED_PREFILL_WARM_TOKENS": "3584",
                             "GLM53_MIXED_PREFILL_MAX_WAIT_MS": "1500"}):
    spec = importlib.util.spec_from_file_location("glm53_warm_gate", VALIDATOR)
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate
    spec.loader.exec_module(gate)

T0 = 1000.0


def capture(offsets=(), ttft=0.5, wall=10.0, finish_reason="stop", error=None, started=True):
    """The record `stream()` leaves behind, from fixed numbers.

    Chunk stamps are absolute (`t0 + offset`), exactly as the live capture
    records them; `ttft` and `wall` are relative to `t0`.
    """
    rec = {"t0": T0, "chunks": [T0 + o for o in offsets], "finish_reason": finish_reason,
           "error": error, "wall": wall}
    if error:
        rec.update(complete=False, void_reason=f"stream error: {error}")
    elif finish_reason in gate.COMPLETE_REASONS and started:
        rec.update(complete=True, void_reason=None)
    else:
        rec.update(complete=False, void_reason=f"stream ended with finish_reason={finish_reason!r}")
    if started:
        rec["ttft"] = ttft
    return rec


def sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n"


class SseErrorObjectTests(unittest.TestCase):
    def test_error_object_is_recorded_and_ends_the_capture(self):
        rec = {"t0": T0, "chunks": [], "finish_reason": None, "error": None, "usage": None}
        gate.consume(iter([
            sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]}),
            sse({"error": {"message": "engine died", "code": 500}}),
            sse({"choices": [{"delta": {"content": "after the error"}, "finish_reason": "stop"}]}),
            b"data: [DONE]\n",
        ]), rec, T0)
        self.assertIn("engine died", rec["error"])
        self.assertEqual(len(rec["chunks"]), 1)          # nothing is recorded after the error
        self.assertIsNone(rec["finish_reason"])          # the error is not a finish reason
        rec["wall"] = 1.0
        reason, verdict = gate.classify(rec, True)
        self.assertTrue(reason)                          # an errored capture can never be held
        self.assertIsNone(verdict)

    def test_clean_stream_is_not_an_error(self):
        rec = {"t0": T0, "chunks": [], "finish_reason": None, "error": None, "usage": None}
        gate.consume(iter([
            sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]}),
            sse({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10}}),
            b"data: [DONE]\n",
        ]), rec, T0)
        self.assertIsNone(rec["error"])
        self.assertEqual(rec["finish_reason"], "stop")
        self.assertEqual(rec["usage"], {"prompt_tokens": 10})


class RateWindowTests(unittest.TestCase):
    def test_capture_relative_window_needs_the_t0_offset(self):
        rec = capture(offsets=(2.0, 4.0, 6.0), ttft=2.0, wall=8.0)
        # Chunk stamps are absolute: a window expressed in the capture's own
        # relative terms matches nothing, which is the defect this pins.
        self.assertEqual(gate.rate(rec, rec["ttft"], rec["wall"]), 0.0)
        self.assertEqual(gate.rate(rec, rec["t0"] + rec["ttft"], rec["t0"] + rec["wall"]), 0.5)

    def test_absolute_window_is_used_as_given(self):
        rec = capture(offsets=(2.0, 4.0, 6.0), ttft=2.0, wall=6.0)
        self.assertEqual(gate.rate(rec, T0 + 2.0, T0 + 4.0), 1.0)

    def test_absent_chunks_do_not_mint_a_rate(self):
        rec = capture(offsets=(), ttft=0.5, wall=4.0, finish_reason=None)
        self.assertEqual(gate.rate(rec, rec["t0"], rec["t0"] + rec["wall"]), 0.0)


class PeerWindowTests(unittest.TestCase):
    def test_window_without_a_peer_chunk_is_void(self):
        peer = capture(offsets=(1.0, 2.0), ttft=0.2, wall=20.0, finish_reason=None)
        self.assertEqual(gate.peer_reason(peer, T0 + 1.0, T0 + 2.0), "")
        self.assertIn("no peer chunk", gate.peer_reason(peer, T0 + 5.0, T0 + 6.0))

    def test_failed_peer_stream_is_void_even_with_a_chunk_in_the_window(self):
        peer = capture(offsets=(1.0,), ttft=0.2, wall=9.0, error="ReadTimeout: dropped")
        self.assertIn("failed", gate.peer_reason(peer, T0, T0 + 9.0))


class VoidPrecedenceTests(unittest.TestCase):
    def test_void_capture_is_never_classified_as_not_met(self):
        void = capture(offsets=(), ttft=0.1, wall=1.0, error="ReadTimeout: dropped")
        reason, verdict = gate.classify(void, False)
        self.assertTrue(reason)
        self.assertIsNone(verdict)

    def test_unfinished_capture_is_void(self):
        rec = capture(offsets=(1.0,), ttft=0.4, wall=3.0, finish_reason=None)
        reason, verdict = gate.classify(rec, True)
        self.assertIn("finish_reason", reason)
        self.assertIsNone(verdict)

    def test_peer_void_voids_the_expectation_it_covers(self):
        warm = capture(offsets=(1.0,), ttft=0.4, wall=2.0)
        peer = capture(offsets=(1.0,), ttft=0.2, wall=9.0, finish_reason=None)
        reason, verdict = gate.classify(warm, False, peer, (T0 + 5.0, T0 + 6.0))
        self.assertEqual(reason, "no peer chunk in the measured window")
        self.assertIsNone(verdict)

    def test_complete_capture_with_a_live_peer_keeps_its_verdict(self):
        warm = capture(offsets=(1.0,), ttft=0.4, wall=2.0)
        peer = capture(offsets=(1.0,), ttft=0.2, wall=9.0, finish_reason=None)
        self.assertEqual(gate.classify(warm, True, peer, (T0, T0 + 2.0)), ("", True))
        self.assertEqual(gate.classify(warm, False, peer, (T0, T0 + 2.0)), ("", False))


class EntrypointVerdictTests(unittest.TestCase):
    def run_capture(self, scenario):
        calls = []

        class Clock:
            now = T0

            @classmethod
            def time(cls):
                return cls.now

            @classmethod
            def sleep(cls, seconds):
                cls.now += seconds

        def stream(text, limit, rec):
            index = len(calls)
            calls.append(index)
            start = Clock.now
            wall, ttft = 2.0, 1.0
            if index == 1:
                wall = 6.0
            elif index == 2:
                wall = 200.0
            elif index == 3 and scenario != "valid":
                wall, ttft = 5.0, 4.0
            elif index == 4:
                wall, ttft = 12.0, 10.0
            elif index == 5:
                wall = 12.0
            rec.update(
                t0=start, chunks=[start + j for j in range(1, int(wall) + 1)],
                complete=True, finish_reason="stop", error=None, void_reason=None,
                wall=wall, ttft=ttft,
                usage={"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 100}},
            )
            if index == 4 and scenario == "mixed-void-unmet":
                rec.update(error="synthetic transport failure", complete=False)
            if index != 2:
                Clock.now += wall

        class Thread:
            def __init__(self, target, args=(), **kwargs):
                self.target, self.args = target, args

            def start(self):
                self.target(*self.args)

            def is_alive(self):
                return False

            def join(self, **kwargs):
                if self.target is stream and scenario == "late-peer-error":
                    self.args[2].update(error="synthetic late peer failure", complete=False)

        output = io.StringIO()
        with patch.multiple(
            gate, time=Clock, threading=type("Threads", (), {"Thread": Thread}),
            stream=stream, require_idle=lambda: None, require_gate_enabled=lambda: None,
            running_now=lambda: 1.0, time_to_service=lambda *args: 10.0,
            MAX_WAIT_S=10.0, WARM_TOKENS=100,
        ), patch.object(sys, "argv", ["validate_warm_gate.py", "fixture"]), contextlib.redirect_stdout(output):
            status = gate.main()
        summary = json.loads(next(
            line.removeprefix("SUMMARY ") for line in output.getvalue().splitlines()
            if line.startswith("SUMMARY ")
        ))
        return status, summary

    def test_valid_run_remains_observable(self):
        self.assertEqual(self.run_capture("valid")[0], 0)

    def test_mixed_invalid_and_unmet_run_is_void(self):
        status, summary = self.run_capture("mixed-void-unmet")
        self.assertEqual(status, 2)
        self.assertTrue(summary["void"])
        self.assertTrue(summary["unmet"])

    def test_late_peer_failure_invalidates_pending_conclusions(self):
        status, summary = self.run_capture("late-peer-error")
        self.assertEqual(status, 2)
        self.assertEqual(summary["unmet"], [])


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
