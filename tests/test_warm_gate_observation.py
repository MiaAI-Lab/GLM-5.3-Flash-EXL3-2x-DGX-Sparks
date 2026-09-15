#!/usr/bin/env python3
"""Inert fixtures for the warm-gate observation rules (no HTTP, no GPU).

`tests/validate_warm_gate.py` runs against a live server, but the rules that
decide what counts as an observation are pure functions over a capture record.
These cases pin them on fixed records and fixed bytes only: SSE error objects,
absolute versus capture-relative chunk stamps, a peer window with no decode, and
the precedence that a void capture is never reported as an expectation that was
not met. Nothing here opens a socket or a device.
"""
from __future__ import annotations

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


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
