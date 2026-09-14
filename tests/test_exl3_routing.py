#!/usr/bin/env python3
"""CPU-only routing regressions; requires PyTorch, not CUDA or model weights."""
import unittest

import torch

from bench_exl3_thin import N_EXP, TOPK, make_routing


class RoutingTests(unittest.TestCase):
    def assert_valid_topk(self, ids, rows):
        self.assertEqual(tuple(ids.shape), (rows, TOPK))
        self.assertTrue(bool(((ids >= 0) & (ids < N_EXP)).all()))
        ordered = ids.sort(dim=1).values
        self.assertTrue(bool((ordered[:, 1:] != ordered[:, :-1]).all()),
                        'a token must not route to an expert more than once')

    def test_routes_cannot_exceed_the_thin_row_cap(self):
        for mode in ('uniform', 'correlated', 'skewed'):
            with self.subTest(mode=mode):
                ids, _ = make_routing(24, mode, 24, 'cpu')
                self.assert_valid_topk(ids, 24)
                self.assertLessEqual(int(torch.bincount(ids.flatten()).max()), 24)

    def test_correlated_partial_block_remains_valid(self):
        ids, _ = make_routing(9, 'correlated', 9, 'cpu')
        self.assert_valid_topk(ids, 9)
        routed_sets = ids[:8].sort(dim=1).values
        self.assertTrue(bool((routed_sets == routed_sets[0]).all()),
                        'verification siblings should share their routed set')


if __name__ == '__main__':
    unittest.main()
