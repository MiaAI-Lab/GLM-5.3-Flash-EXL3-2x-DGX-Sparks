#!/usr/bin/env python3
"""CPU-only diagnostics coverage for the logprobs comparison probe.

Guards the two ways compare() must not invent a comparison number: captures of
different lengths and captures with no comparable position.
"""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("bench_logprobs.py")
SPEC = importlib.util.spec_from_file_location("bench_logprobs", MODULE_PATH)
assert SPEC and SPEC.loader
bench_logprobs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench_logprobs)


def pos(*tokens: str) -> dict:
    return {t: {"logprob": -(i + 1) / 10, "rank": i + 1} for i, t in enumerate(tokens)}


def run_compare(a: dict, b: dict) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        a_path, b_path = Path(tmp) / "a.json", Path(tmp) / "b.json"
        a_path.write_text(json.dumps(a))
        b_path.write_text(json.dumps(b))
        buf = io.StringIO()
        with redirect_stdout(buf):
            bench_logprobs.compare(a_path, b_path)
    return buf.getvalue()


class CompareDiagnosticsTest(unittest.TestCase):
    def row(self, out: str, name: str) -> str:
        rows = [line for line in out.splitlines() if line.split() and line.split()[0] == name]
        self.assertTrue(rows, f"no row for {name!r} in:\n{out}")
        return rows[0]

    def test_empty_captures_have_no_comparable_or_excluded_positions(self):
        row = self.row(run_compare({"t": []}, {"t": []}), "t")
        self.assertEqual(tuple(map(int, row.split()[1:3])), (0, 0))

    def test_no_comparable_position_is_unavailable(self):
        out = run_compare({"t": [None, pos("a", "b")]}, {"t": [None, pos("c", "d")]})
        row = self.row(out, "t")
        self.assertEqual(row.split()[-2:], ["unavailable", "unavailable"])
        self.assertNotRegex(row, r"\d+\.\d{4}")

    def test_length_mismatch_is_reported_not_truncated(self):
        out = run_compare({"t": [None, pos("a"), pos("b")]}, {"t": [None, pos("a")]})
        row = self.row(out, "t")
        self.assertIn("unavailable", row)
        self.assertIn("A=3", row)
        self.assertIn("B=2", row)
        self.assertNotRegex(row, r"\d+\.\d{4}")

    def test_identical_captures_still_compare(self):
        capture = {"t": [None, pos("a", "b")]}
        row = self.row(run_compare(capture, capture), "t")
        self.assertNotIn("unavailable", row)
        self.assertEqual(float(row.split()[-1]), 1.0)

    def test_text_present_in_only_one_capture_is_named(self):
        out = run_compare({"t": [None, pos("a")]}, {"t2": [None, pos("a")]})
        named = [ln for ln in out.splitlines() if ln.startswith("not compared") and "t2" in ln]
        self.assertTrue(named, f"extra capture text was not reported in:\n{out}")


if __name__ == "__main__":
    unittest.main()
