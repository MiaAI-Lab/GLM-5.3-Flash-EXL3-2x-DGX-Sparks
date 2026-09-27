#!/usr/bin/env python3
"""Production tests for the opt-in SENS8 decode router (TP2/SM121).

Standards (from research qualification):
- ordered expert IDs EXACT (never set-equality);
- routing weights BITWISE vs the frozen candidate kernel, tolerance vs the
  torch reference (CPU/GPU expf differ in the last ulp);
- request isolation across partitions (no global coreset);
- graph replay + warmup-stock->SENS8 regression;
- flag gating (off = stock path literally untouched).

CPU: reference + loader + launcher-adjacent tests run anywhere torch is
present. GPU: kernel parity/replay need CUDA SM12x (skipped otherwise).
vLLM: integration tests need the serving image (skipped otherwise).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "kernel_lab" / "sens8"))

E, C, K, SCALE = 288, 28, 8, 2.5


def torch_reference(logits, bias, blocks=None):
    """Exact-arithmetic mirror of sens8g (fp32). blocks=None -> STOCK."""
    s = torch.sigmoid(logits.float())
    b = s + bias.float()
    m = logits.shape[0]
    if blocks is None:
        return _topk_rows(s, b, [list(range(E))] * m)
    # request-local coreset per block from block rows only.
    parts, out_ids, out_w = [], [], []
    off = 0
    for nb in blocks:
        rows = list(range(off, off + nb))
        off += nb
        util = torch.zeros(E)
        for r in rows:
            br = b[r]
            # rank with index-ascending ties: #{jj: bj>be or (== and jj<e)}
            rank = torch.zeros(E)
            for e in range(E):
                be = br[e]
                rank[e] = ((br > be) | ((br == be)
                                        & (torch.arange(E) < e))).sum()
            util += s[r] / (1.0 + rank)
        order = torch.argsort(util, descending=True, stable=True)[:C]
        coreset = sorted(int(x) for x in order.tolist())
        parts.append((rows, coreset))
    return _topk_rows(s, b, [c for rows, c in parts for _ in rows])


def _topk_rows(s, b, coresets):
    ids_all, w_all = [], []
    for r, core in enumerate(coresets):
        picked, sats = [], []
        pool = list(core)
        for _ in range(K):
            best, bestb = None, None
            for e in pool:
                if e in picked:
                    continue
                if best is None or b[r, e] > bestb or (
                        b[r, e] == bestb and e < best):
                    best, bestb = e, b[r, e]
            picked.append(best)
            sats.append(s[r, best])
        tot = sum(sats)
        ids_all.append(picked)
        w_all.append([x / tot * SCALE for x in sats])
    return (torch.tensor(ids_all, dtype=torch.int32),
            torch.tensor(w_all, dtype=torch.float32))


def need_cuda(tc):
    if not torch.cuda.is_available():
        tc.skipTest("needs CUDA")
    try:
        torch.zeros(1, device="cuda")
    except Exception:
        tc.skipTest("no GPU headroom")
    major, _ = torch.cuda.get_device_capability()
    if major != 12:
        tc.skipTest("needs SM12x")


def get_ext():
    import sens8g_loader as sg
    return sg


class ReferenceTests(unittest.TestCase):
    def test_tie_ordering_index_ascending(self):
        lg = torch.zeros(4, E)
        bi = torch.zeros(E)
        ids, w = torch_reference(lg, bi, (4,))
        self.assertEqual(ids[0].tolist(), list(range(8)))
        self.assertTrue(torch.allclose(
            w.sum(-1), torch.full((4,), SCALE), atol=1e-5))

    def test_request_isolation(self):
        torch.manual_seed(7)
        lg = torch.randn(8, E)
        bi = torch.randn(E) * 2 + 6
        ids_a, _ = torch_reference(lg, bi, (8,))
        ids_b, _ = torch_reference(lg, bi, (5, 3))
        # First block rows share the same rows but different block scope:
        # coresets are request-local, so row<5 top-8 need not match (5,3)'s
        # block-0 vs (8,)'s single block. Isolation = block-1 of (5,3)
        # depends only on rows 5..8.
        lg2 = torch.randn(8, E)
        lg2[5:] = lg[5:]
        ids_c, _ = torch_reference(lg2, bi, (5, 3))
        self.assertEqual(ids_b[5:].tolist(), ids_c[5:].tolist())

    def test_determinism(self):
        torch.manual_seed(0)
        lg = torch.randn(16, E)
        bi = torch.randn(E) * 2 + 6
        self.assertTrue(torch.equal(torch_reference(lg, bi, (8, 8))[0],
                                    torch_reference(lg, bi, (8, 8))[0]))

    def test_weight_normalization(self):
        torch.manual_seed(1)
        lg = torch.randn(6, E)
        bi = torch.randn(E) * 2 + 6
        _, w = torch_reference(lg, bi, (3, 3))
        # weights are s/sum*2.5 per row -> row sums are all SCALE.
        self.assertTrue(torch.allclose(
            w.sum(-1), torch.full((6,), SCALE), atol=1e-5))


class LoaderTests(unittest.TestCase):
    def test_make_meta(self):
        sg = get_ext()
        m = sg.make_meta(1, (8,))
        self.assertEqual(m.tolist(), [1, 1, 8, 0, 0, 0])
        m = sg.make_meta(1, (5, 3))
        self.assertEqual(m.tolist(), [1, 2, 5, 3, 0, 0])
        m = sg.make_meta(0, ())
        self.assertEqual(m.tolist(), [0, 0, 0, 0, 0, 0])
        with self.assertRaises(AssertionError):
            sg.make_meta(1, (8,) * 5)

    def test_fused_forward_gates(self):
        sg = get_ext()
        with self.assertRaises(Exception):
            sg.fused_forward(torch.zeros(8, 100), torch.zeros(288),
                             torch.zeros(6, dtype=torch.int32),
                             torch.zeros(8, 8, dtype=torch.int32),
                             torch.zeros(8, 8), torch.zeros(2, dtype=torch.int32))


class KernelParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need_cuda(cls)
        os.environ.setdefault("SENS8_BUILD_DIR", "/tmp/sens8-pr-test")
        cls.sg = get_ext()
        try:
            cls.sg.get_extension()
        except Exception as e:
            raise unittest.SkipTest(f"no kernel: {e}")

    def _run(self, lg, bi, blocks):
        sg = self.sg
        dev = torch.device("cuda")
        lg32, bi32 = lg.float().to(dev), bi.float().to(dev)
        mode = 1 if blocks is not None else 0
        meta = sg.make_meta(mode, blocks or ())
        ids = torch.empty((lg.shape[0], 8), dtype=torch.int32, device=dev)
        w = torch.empty((lg.shape[0], 8), dtype=torch.float32, device=dev)
        dbg = torch.zeros(2, dtype=torch.int32, device=dev)
        if blocks is not None:
            flat, off = [], 0
            for nb in blocks:
                flat += list(range(off, off + nb))
                off += nb
            lg32 = lg32[flat]
        sg.fused_forward(lg32, bi32, meta.to(dev), ids, w, dbg)
        return ids.cpu(), w.cpu()

    def _check(self, m, blocks, seed=0, bf16=False, ties=False,
               dominant=False):
        torch.manual_seed(seed)
        lg = torch.zeros(m, E) if ties else torch.randn(m, E)
        if dominant:
            lg[:, 0] += 20.0
        bi = (torch.randn(E) * 2 + 6).float()
        if bf16:
            lg = lg.bfloat16()
        e_ids, e_w = torch_reference(lg.float(), bi, blocks)
        k_ids, k_w = self._run(lg, bi, blocks)
        self.assertEqual(k_ids.tolist(), e_ids.tolist(),
                         f"IDs must match exactly m={m} blocks={blocks}")
        self.assertLessEqual(float((k_w - e_w).abs().max()), 1e-5)

    def test_m8_single(self):
        self._check(8, (8,))

    def test_partitions(self):
        self._check(8, (5, 3))
        self._check(8, (3, 5))
        self._check(16, (8, 8))
        self._check(24, (8, 8, 8))
        self._check(32, (8, 8, 8, 8))

    def test_ties_dominant(self):
        self._check(8, (8,), ties=True)
        self._check(8, (8,), dominant=True, seed=3)

    def test_bf16(self):
        self._check(16, (8, 8), bf16=True, seed=5)

    def test_stock_modes(self):
        self._check(64, None)
        self._check(32, None, seed=2)

    def test_strided(self):
        torch.manual_seed(9)
        base = torch.randn(32, E * 2)[:, ::2]
        self.assertFalse(base.is_contiguous())
        bi = (torch.randn(E) * 2 + 6).float()
        e_ids, e_w = torch_reference(base.float(), bi, (8, 8, 8, 8))
        k_ids, k_w = self._run(base, bi, (8, 8, 8, 8))
        self.assertEqual(k_ids.tolist(), e_ids.tolist())
        self.assertLessEqual(float((k_w - e_w).abs().max()), 1e-5)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None
                     and torch.cuda.is_available(), "needs CUDA")
class GraphReplayTests(unittest.TestCase):
    def test_replay_new_logits_meta_partition(self):
        need_cuda(self)
        os.environ.setdefault("SENS8_BUILD_DIR", "/tmp/sens8-pr-test")
        sg = get_ext()
        try:
            sg.get_extension()
        except Exception as e:
            raise unittest.SkipTest(f"no kernel: {e}")
        dev = torch.device("cuda")
        torch.manual_seed(11)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            lg = torch.randn(32, E, device=dev)
            bi = ((torch.randn(E, device=dev) * 2 + 6)).float()
            meta = sg.make_meta(1, (8, 8, 8, 8)).to(dev)
            ids = torch.empty(32, 8, dtype=torch.int32, device=dev)
            w = torch.empty(32, 8, dtype=torch.float32, device=dev)
            dbg = torch.zeros(2, dtype=torch.int32, device=dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                sg.fused_forward(lg, bi, meta, ids, w, dbg)
            # Replay 1: same graph, new SENS8 partition (5,3) first block.
            lg2 = torch.randn(32, E, device=dev)
            lg.copy_(lg2)
            meta.copy_(sg.make_meta(1, (5, 3)).to(dev))
            g.replay()
            torch.cuda.current_stream().synchronize()
            e_ids, e_w = torch_reference(lg2[:8].cpu(), bi.cpu(), (5, 3))
            self.assertEqual(ids[:8].cpu().tolist(), e_ids.tolist())
            self.assertLessEqual(float((w[:8].cpu() - e_w).abs().max()), 1e-5)
            # Replay 2: stock mode through the same graph.
            meta.copy_(sg.make_meta(0, ()).to(dev))
            g.replay()
            torch.cuda.current_stream().synchronize()
            e_ids0, e_w0 = torch_reference(lg2.cpu(), bi.cpu(), None)
            self.assertEqual(ids.cpu().tolist(), e_ids0.tolist())
            self.assertLessEqual(float((w.cpu() - e_w0).abs().max()), 1e-5)


@unittest.skipUnless(importlib.util.find_spec("vllm") is not None,
                     "needs vllm (run in the image container)")
class FlagGatingTests(unittest.TestCase):
    @staticmethod
    def _load_overlay():
        spec = importlib.util.spec_from_file_location(
            "glm53_exl3_overlay_clean", REPO / "overlay" / "exl3.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["glm53_exl3_overlay_clean"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_flag_off_leaves_stock_path_untouched(self):
        import vllm.v1.worker.gpu.model_runner as mr
        import vllm.model_executor.layers.fused_moe.router.grouped_topk_router as gtr
        pre_p = mr.GPUModelRunner.prepare_inputs
        pre_r = gtr.GroupedTopKRouter._compute_routing
        env = dict(os.environ)
        env.pop("GLM53_SENS8_ROUTER", None)
        with mock.patch.dict(os.environ, env, clear=True):
            ox = self._load_overlay()
            self.assertFalse(ox._sens8_installed)
        self.assertIs(mr.GPUModelRunner.prepare_inputs, pre_p)
        self.assertIs(gtr.GroupedTopKRouter._compute_routing, pre_r)

    def test_flag_invalid_rejected(self):
        for v in ("", "yes", "2", " 1", "1 "):
            with self.subTest(v=v):
                with mock.patch.dict(
                        os.environ, {"GLM53_SENS8_ROUTER": v}):
                    ox = self._load_overlay()
                    self.assertFalse(ox._sens8_enabled())


if __name__ == "__main__":
    unittest.main(verbosity=1)
