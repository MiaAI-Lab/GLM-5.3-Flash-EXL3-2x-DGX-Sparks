#!/usr/bin/env python3
"""CPU contracts for the KDA large-M BF16 dispatch (no GPU/vLLM needed).

Executes the real flag, retention and dispatch code from overlay/exl3.py with
narrow device stubs. Optional CPU-PyTorch cases also check eager dequant
values and layouts. These checks do not establish native CUDA compilation,
kernel parity or graph safety; the GPU suite remains a separate gate.

Run:  python3 tests/test_kda_bf16_large_m.py
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "overlay/exl3.py"


def _func_source(name: str) -> str:
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(SRC.read_text(), node) or ""
    raise AssertionError(f"function {name} not found")


def _exec_flag_fn():
    src = _func_source("kda_bf16_large_m_enabled")
    ns: dict = {"os": os}
    exec(compile(ast.Module(body=[ast.parse(src).body[0]], type_ignores=[]),
                 str(SRC), "exec"), ns)
    return ns["kda_bf16_large_m_enabled"]


class LargeMFlagTests(unittest.TestCase):
    def test_values(self):
        fn = _exec_flag_fn()
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_BF16_LARGE_M": "0"}):
            self.assertFalse(fn())
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_BF16_LARGE_M": "1"}):
            self.assertTrue(fn())
        with unittest.mock.patch.dict(os.environ):
            os.environ.pop("GLM53_KDA_BF16_LARGE_M", None)
            self.assertFalse(fn())
        for bad in ("yes", "", " 1", "1 ", "2", "true"):
            with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_BF16_LARGE_M": bad}):
                with self.assertRaises(RuntimeError):
                    fn()


# ---------------------------------------------------------------------------
# Exec-based tests: run the real Glm53DenseFp8Method source against narrow
# torch/vLLM stubs. These verify observable dispatch behavior (which GEMM
# path runs, with which shapes), not source text.
# ---------------------------------------------------------------------------

_N, _K = 12576, 4096


class _FT:
    """Minimal tensor: shape/stride/dtype bookkeeping only."""

    def __init__(self, shape, dtype="bf16", stride=None, device="cuda:0"):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        if stride is None:
            stride = []
            acc = 1
            for d in reversed(self._shape):
                stride.append(acc)
                acc *= d
            stride = tuple(reversed(stride))
        self._stride = tuple(stride)

    @property
    def shape(self):
        return self._shape

    def dim(self):
        return len(self._shape)

    def numel(self):
        n = 1
        for d in self._shape:
            n *= d
        return n

    def stride(self, i=None):
        return self._stride if i is None else self._stride[i]

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        if -1 in shape:
            known = 1
            for d in shape:
                if d != -1:
                    known *= d
            shape[shape.index(-1)] = self.numel() // known
        return _FT(shape, dtype=self.dtype, device=self.device)

    def view(self, *shape):
        return self.reshape(*shape)

    def to(self, dtype=None, **kw):
        return _FT(self._shape, dtype=dtype or self.dtype, device=self.device)


class _FakeTorch(types.SimpleNamespace):
    pass


def _fake_torch(capability=(12, 1), cap_raises=False):
    calls = {"synchronize": 0}

    def get_cap(device):
        if cap_raises:
            raise RuntimeError("no cuda")
        return capability

    def synchronize(*a):
        calls["synchronize"] += 1

    return _FakeTorch(
        float8_e4m3fn="fp8", float32="f32", bfloat16="bf16", float16="f16",
        empty=lambda shape, dtype=None, device=None: _FT(shape, dtype=dtype, device=device),
        zeros=lambda shape, dtype=None, device=None: _FT(shape, dtype=dtype, device=device),
        cuda=types.SimpleNamespace(
            get_device_capability=get_cap, synchronize=synchronize),
        _sync_calls=calls,
    )


def _marlin_stub(rec):
    """Install the vllm marlin_utils_fp8 import chain; returns recorder."""
    def apply_fp8_marlin_linear(**kw):
        rec.append(kw)
        return _FT(kw["input"].shape[:-1] + (kw["size_n"],))
    names = [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.utils",
        "vllm.model_executor.layers.quantization.utils.marlin_utils_fp8",
    ]
    mods = {}
    for name in names:
        m = types.ModuleType(name)
        mods[name] = m
    for parent, child in zip(names, names[1:]):
        setattr(mods[parent], child.rsplit(".", 1)[-1], mods[child])
    mods[names[-1]].apply_fp8_marlin_linear = apply_fp8_marlin_linear
    return mods


_BF16_LINEAR_CALLS: list = []


def _fake_f_linear(x, w, bias=None):
    """Stand-in for torch.nn.functional.linear: records and shape-tracks."""
    _BF16_LINEAR_CALLS.append((x, w))
    return _FT(tuple(x.shape[:-1]) + (w.shape[0],))


def _load_method_class(torch_fake):
    """Exec the real class + helpers from overlay/exl3.py with stubs."""
    source = SRC.read_text()
    tree = ast.parse(source)
    wanted_fns = {
        "kda_bf16_large_m_enabled",
        "kda_bf16_large_m_logical_weight",
        "kda_large_m_dispatch_stats",
        "_kda_large_m_note",
    }
    wanted_assigns = {
        "KDA_BF16_LARGE_M_SHAPES", "KDA_BF16_LARGE_M_MIN_M",
        "KDA_BF16_LARGE_M_CHUNK_ROWS", "KDA_BF16_LARGE_M_DTYPE",
        "_KDA_LARGE_M_DISPATCH_STATS", "_KDA_LARGE_M_STATS_DUMP_EVERY",
    }
    body = [
        ast.ImportFrom(module="__future__",
                       names=[ast.alias(name="annotations")], level=0)
    ]

    def _wanted_assign(node) -> bool:
        """Match both plain and annotated module-level assignments."""
        targets = getattr(node, "targets", None) or [getattr(node, "target", None)]
        return any(
            isinstance(t, ast.Name) and t.id in wanted_assigns for t in targets
        )

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_fns:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and _wanted_assign(node):
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "Glm53DenseFp8Method":
            body.append(node)
    names = {getattr(n, "name", None) for n in body}
    assert "Glm53DenseFp8Method" in names

    class _Base:
        def __init__(self, *a, **k):
            pass

        def apply(self, layer, x, bias=None):
            return ("base_apply", x)

        def process_weights_after_loading(self, layer):
            pass

    env = {
        "os": os,
        "json": json,
        "re": re,
        "torch": torch_fake,
        "F": types.SimpleNamespace(linear=_fake_f_linear),
        "UnquantizedLinearMethod": _Base,
        "logger": types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
        ),
    }
    mod = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(mod, str(SRC), "exec"), env)
    return env


def _lm_layer(n=_N, k=_K):
    return types.SimpleNamespace(
        glm53_fp8_n=n,
        glm53_fp8_k=k,
        weight=_FT((n, k), dtype="fp8"),
        weight_scale=_FT((n,), dtype="bf16"),
        workspace=object(),
    )


class FixedThresholdTests(unittest.TestCase):
    """The dispatch boundary is a fixed qualified constant, not a knob."""

    def test_constant_is_512(self):
        env = _load_method_class(_fake_torch())
        self.assertEqual(env["KDA_BF16_LARGE_M_MIN_M"], 512)


class Bf16LogicalWeightTests(unittest.TestCase):
    """Real dequant source: the exact logical weight, chunk-invariant."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("install CPU PyTorch for tensor-value regressions")
        cls.torch = torch

    def _env(self):
        env = _load_method_class(self.torch)
        env["KDA_BF16_LARGE_M_CHUNK_ROWS"] = 7  # chunk boundaries inside the test
        return env

    def _fp8(self, n=64, k=128, seed=11):
        torch = self.torch
        w = torch.randn(n, k, generator=torch.Generator().manual_seed(seed))
        scales = w.abs().amax(dim=1).clamp_min(1e-12) / 448.0
        fp8 = (w / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return fp8, scales

    def test_matches_the_exact_logical_weight(self):
        torch = self.torch
        fp8, scales = self._fp8()
        stored = scales.to(torch.bfloat16)
        got = self._env()["kda_bf16_large_m_logical_weight"](
            fp8, stored, torch.bfloat16)
        want = (fp8.to(torch.float32) * stored.to(torch.float32).unsqueeze(1)).to(
            torch.bfloat16)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertEqual(tuple(got.shape), tuple(fp8.shape))
        self.assertTrue(torch.equal(got, want))

    def test_chunk_size_does_not_change_a_single_bit(self):
        torch = self.torch
        env = self._env()
        fp8, scales = self._fp8()
        stored = scales.to(torch.bfloat16)
        fn = env["kda_bf16_large_m_logical_weight"]
        small = fn(fp8, stored, torch.bfloat16, 1)
        for chunk in (7, 32, 4096):
            self.assertTrue(torch.equal(small, fn(fp8, stored, torch.bfloat16, chunk)))

    def test_the_fp32_scale_variant_is_a_different_weight(self):
        """An fp32-scale reconstruction is NOT the weight Marlin consumes."""
        torch = self.torch
        fp8, scales = self._fp8()
        fn = self._env()["kda_bf16_large_m_logical_weight"]
        wa = fn(fp8, scales, torch.bfloat16)               # fp32 scales
        wb = fn(fp8, scales.to(torch.bfloat16), torch.bfloat16)  # stored scales
        self.assertFalse(torch.equal(wa, wb))

    def test_copy_residual_far_below_the_old_w8a8_layer_error(self):
        """The copy's own rounding (~2^-9) is ~5x below v1's 0.0266."""
        torch = self.torch
        fp8, scales = self._fp8(n=256, k=512, seed=11)
        stored = scales.to(torch.bfloat16)
        fn = self._env()["kda_bf16_large_m_logical_weight"]
        exact = fp8.to(torch.float32) * stored.to(torch.float32).unsqueeze(1)
        got = fn(fp8, stored, torch.bfloat16).to(torch.float32)
        rel = float(((got - exact).abs() / exact.abs().clamp_min(1e-30)).mean())
        self.assertLess(rel, 0.005)

    def test_memory_math_matches_the_disclosed_cost(self):
        """12576 x 4096 BF16 = 98.25 MiB per layer-rank."""
        torch = self.torch
        self.assertEqual(self._env()["KDA_BF16_LARGE_M_SHAPES"],
                         frozenset({(12576, 4096)}))
        self.assertEqual(self._env()["KDA_BF16_LARGE_M_DTYPE"], torch.bfloat16)
        per_layer = 12576 * 4096 * torch.bfloat16.itemsize
        self.assertEqual(per_layer / 2**20, 98.25)
        self.assertAlmostEqual(per_layer * 34 / 2**30, 3.26, places=2)


class _RealTorchCudaPatch:
    """Patch the two CUDA calls the retention path makes, on real CPU torch."""

    def __init__(self, capability=(12, 1)):
        self.capability = capability

    def __enter__(self):
        import torch

        self._cap = unittest.mock.patch.object(
            torch.cuda, "get_device_capability", lambda *a, **k: self.capability)
        self._sync = unittest.mock.patch.object(
            torch.cuda, "synchronize", lambda *a, **k: None)
        self._cap.start()
        self._sync.start()
        return self

    def __exit__(self, *exc):
        self._cap.stop()
        self._sync.stop()
        return False


_IN_PROJ = "model.layers.0.self_attn.in_proj_qkvbfg_a"


class Bf16RetentionTests(unittest.TestCase):
    """Real _retain_bf16_large_m_weights source: one copy, fail-closed."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("install CPU PyTorch for tensor-value regressions")
        cls.torch = torch

    def _env(self, *, enabled=True, shapes=None):
        env = _load_method_class(self.torch)
        env["KDA_BF16_LARGE_M_SHAPES"] = frozenset(shapes or {(8, 16)})
        env["KDA_BF16_LARGE_M_DTYPE"] = self.torch.bfloat16
        env["kda_bf16_large_m_enabled"] = lambda: enabled
        return env

    def _inputs(self, n=8, k=16, seed=3, scale_dtype=None):
        torch = self.torch
        w = torch.randn(n, k, generator=torch.Generator().manual_seed(seed))
        scales = w.abs().amax(dim=1).clamp_min(1e-12) / 448.0
        fp8 = (w / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return fp8, scales.to(scale_dtype or torch.bfloat16)

    def _method(self, env, group="kda", prefix=_IN_PROJ):
        return env["Glm53DenseFp8Method"](group, prefix)

    def _retain(self, env, *, capability=(12, 1), linear=None, tp_size=2,
                n=8, k=16, group="kda", prefix=_IN_PROJ, scale_dtype=None):
        torch = self.torch
        fp8, stored = self._inputs(n, k, scale_dtype=scale_dtype)
        layer = torch.nn.Module()
        method = self._method(env, group, prefix)
        with _RealTorchCudaPatch(capability):
            if linear is None:
                method._retain_bf16_large_m_weights(
                    layer, fp8, stored, n, k, tp_size)
            else:
                with unittest.mock.patch.object(torch.nn.functional, "linear", linear):
                    method._retain_bf16_large_m_weights(
                        layer, fp8, stored, n, k, tp_size)
        return layer, fp8, stored

    def test_retains_one_bf16_copy(self):
        torch = self.torch
        layer, fp8, stored = self._retain(self._env())
        self.assertTrue(hasattr(layer, "glm53_bf16_lm_w"))
        self.assertEqual(layer.glm53_bf16_lm_w.dtype, torch.bfloat16)
        self.assertEqual(tuple(layer.glm53_bf16_lm_w.shape), (8, 16))
        self.assertEqual(layer.glm53_bf16_lm_n, 8)
        self.assertEqual(layer.glm53_bf16_lm_k, 16)
        want = (fp8.to(torch.float32) * stored.to(torch.float32).unsqueeze(1)).to(
            torch.bfloat16)
        self.assertTrue(torch.equal(layer.glm53_bf16_lm_w, want))

    def test_layer_records_the_fixed_boundary(self):
        """No override exists: retention always stores 512."""
        layer, _, _ = self._retain(self._env())
        self.assertEqual(layer.glm53_bf16_lm_min_m, 512)

    def test_disabled_retains_nothing(self):
        layer, _, _ = self._retain(self._env(enabled=False))
        self.assertFalse(hasattr(layer, "glm53_bf16_lm_w"))

    def test_non_candidate_prefix_retains_nothing_without_raising(self):
        layer, _, _ = self._retain(
            self._env(), prefix="model.layers.0.self_attn.o_proj", tp_size=1)
        self.assertFalse(hasattr(layer, "glm53_bf16_lm_w"))

    def test_wrong_group_retains_nothing_without_raising(self):
        layer, _, _ = self._retain(
            self._env(), group="dense",
            prefix="model.layers.0.mlp.down_proj", tp_size=1)
        self.assertFalse(hasattr(layer, "glm53_bf16_lm_w"))

    def test_wrong_tp_raises(self):
        with self.assertRaises(RuntimeError):
            self._retain(self._env(), tp_size=1)
        with self.assertRaises(RuntimeError):
            self._retain(self._env(), tp_size=3)

    def test_wrong_capability_raises(self):
        with self.assertRaises(RuntimeError):
            self._retain(self._env(), capability=(11, 0))

    def test_wrong_shape_raises(self):
        with self.assertRaises(RuntimeError):
            self._retain(self._env(shapes={(8, 16)}), n=99, k=99)

    def test_non_e4m3_weight_raises(self):
        torch = self.torch
        env = self._env()
        fp8, stored = self._inputs()
        layer = torch.nn.Module()
        method = self._method(env)
        with _RealTorchCudaPatch():
            with self.assertRaises(RuntimeError):
                method._retain_bf16_large_m_weights(
                    layer, fp8.to(torch.bfloat16), stored, 8, 16, 2)
        self.assertFalse(hasattr(layer, "glm53_bf16_lm_w"))

    def test_fp32_stored_scale_raises(self):
        with self.assertRaises(RuntimeError):
            self._retain(self._env(), scale_dtype=self.torch.float32)

    def test_probe_failure_raises(self):
        def boom(*a, **k):
            raise RuntimeError("no cublas for you")

        with self.assertRaises(RuntimeError):
            self._retain(self._env(), linear=boom)


class Bf16ApplyDispatchTests(unittest.TestCase):
    """Real apply() source: the fixed M>512 boundary."""

    def setUp(self):
        self.marlin_calls = []
        self._patcher = unittest.mock.patch.dict(sys.modules, _marlin_stub(self.marlin_calls))
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        _BF16_LINEAR_CALLS.clear()
        self.env = _load_method_class(_fake_torch())
        self.cls = self.env["Glm53DenseFp8Method"]
        self.meth = self.cls("kda", _IN_PROJ)
        self.meth.ready = True

    def _layer(self):
        layer = _lm_layer()
        layer.glm53_bf16_lm_w = _FT((_N, _K), dtype="bf16")
        layer.glm53_bf16_lm_n = _N
        layer.glm53_bf16_lm_k = _K
        layer.glm53_bf16_lm_min_m = 512
        return layer

    def test_fixed_boundary_table(self):
        """M<=512 runs Marlin; M>=513 runs the BF16 copy."""
        layer = self._layer()
        for m in (1, 64, 220, 511, 512):
            self.meth.apply(layer, _FT((m, _K)))
        self.assertEqual(len(_BF16_LINEAR_CALLS), 0)
        self.assertEqual(len(self.marlin_calls), 5)
        for m in (513, 768, 1536):
            out = self.meth.apply(layer, _FT((m, _K)))
            self.assertEqual(out.shape, (m, _N))
        self.assertEqual(len(_BF16_LINEAR_CALLS), 3)
        self.assertEqual(len(self.marlin_calls), 5)

    def test_old_v1_boundary_lengths_stay_marlin(self):
        """M=63/64/65/66 show no dispatch discontinuity under the BF16 path."""
        layer = self._layer()
        for m in (63, 64, 65, 66):
            self.meth.apply(layer, _FT((m, _K)))
        self.assertEqual(len(_BF16_LINEAR_CALLS), 0)
        self.assertEqual(len(self.marlin_calls), 4)

    def test_3d_and_bias_and_wrong_k(self):
        layer = self._layer()
        out = self.meth.apply(layer, _FT((2, 300, _K)))
        self.assertEqual(len(_BF16_LINEAR_CALLS), 1)
        self.assertEqual(_BF16_LINEAR_CALLS[0][0].shape, (600, _K))
        self.assertEqual(out.shape, (2, 300, _N))
        before = len(_BF16_LINEAR_CALLS)
        self.meth.apply(layer, _FT((600, _K)), bias=_FT((_N,)))
        self.meth.apply(layer, _FT((600, _K + 8)))
        self.assertEqual(len(_BF16_LINEAR_CALLS), before)
        self.assertEqual(len(self.marlin_calls), 2)

    def test_missing_retention_falls_through_to_marlin(self):
        layer = _lm_layer()
        self.meth.apply(layer, _FT((4096, _K)))
        self.assertEqual(len(_BF16_LINEAR_CALLS), 0)
        self.assertEqual(len(self.marlin_calls), 1)

    def test_dispatch_counters_track_rows(self):
        stats = self.env["kda_large_m_dispatch_stats"]
        layer = self._layer()
        self.meth.apply(layer, _FT((600, _K)))
        self.meth.apply(layer, _FT((10, _K)))
        self.assertEqual(stats()["bf16_calls"], 1)
        self.assertEqual(stats()["bf16_rows"], 600)
        self.assertEqual(stats()["marlin_calls"], 1)
        self.assertEqual(stats()["marlin_rows"], 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
