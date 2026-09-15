#!/usr/bin/env python3
"""CPU regressions for false-clean and sign errors in the logprob panel."""
import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bench_kda_fp8 as panel


def receipt(first=0.8, second=0.2):
    return {"model": "fixture-model", "texts": {"fixture": {
        "mode": "prompt_logprobs",
        "prompt_sha256": hashlib.sha256(b"fixed fixture").hexdigest(),
        "positions": [None, {"101": {"logprob": math.log(first)}, "102": {"logprob": math.log(second)}}],
    }}}


class LogprobPanelTests(unittest.TestCase):
    def compare(self, a, b):
        with tempfile.TemporaryDirectory() as td:
            left, right = Path(td) / "a.json", Path(td) / "b.json"
            left.write_text(json.dumps(a))
            right.write_text(json.dumps(b))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = panel.compare(str(left), str(right))
            return result, output.getvalue()

    def test_identical_teacher_forced_panel_passes(self):
        self.assertEqual(self.compare(receipt(), receipt())[0], 0)

    def test_nll_delta_is_positive_when_candidate_loses_top_one_mass(self):
        result, output = self.compare(receipt(), receipt(0.4, 0.1))
        self.assertEqual(result, 0)
        row = next(line for line in output.splitlines() if line.startswith("fixture "))
        self.assertAlmostEqual(float(row.split()[-1]), math.log(2), places=3)

    def test_argmax_flip_fails_even_with_small_kl(self):
        self.assertEqual(self.compare(receipt(0.5001, 0.4999), receipt(0.4999, 0.5001))[0], 1)

    def test_large_kl_fails_even_with_matching_argmax(self):
        self.assertEqual(self.compare(receipt(0.99, 0.01), receipt(0.6, 0.4))[0], 1)

    def test_empty_and_missing_records_cannot_pass(self):
        for a, b in (({"texts": {}}, {"texts": {}}), (receipt(), {"texts": {}})):
            with self.subTest(a=a):
                self.assertEqual(self.compare(a, b)[0], 2)

    def test_missing_or_truncated_positions_cannot_pass(self):
        for positions in ([], [None], [None, None], [None, {}, {}]):
            candidate = receipt()
            candidate["texts"]["fixture"]["positions"] = positions
            with self.subTest(positions=positions):
                self.assertEqual(self.compare(receipt(), candidate)[0], 2)

    def test_generation_fallback_and_different_prompts_are_unqualified(self):
        for field, value in (("mode", "generation_logprobs"), ("prompt_sha256", "0" * 64)):
            candidate = receipt()
            candidate["texts"]["fixture"][field] = value
            with self.subTest(field=field):
                self.assertEqual(self.compare(receipt(), candidate)[0], 2)

    def test_invalid_probabilities_cannot_pass(self):
        for value in (float("nan"), 0.1, "not-a-number", True):
            candidate = receipt()
            candidate["texts"]["fixture"]["positions"][1]["101"]["logprob"] = value
            with self.subTest(value=value):
                self.assertEqual(self.compare(receipt(), candidate)[0], 2)

    def test_disjoint_support_is_not_silently_skipped(self):
        candidate = receipt()
        candidate["texts"]["fixture"]["positions"][1] = {"999": {"logprob": 0.0}}
        self.assertEqual(self.compare(receipt(), candidate)[0], 2)

    def test_capture_without_prompt_distributions_does_not_emit_a_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "capture.json"
            with patch.object(panel, "_post", return_value=(200, {"choices": [{}]})):
                self.assertEqual(panel.capture(str(target)), 2)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
