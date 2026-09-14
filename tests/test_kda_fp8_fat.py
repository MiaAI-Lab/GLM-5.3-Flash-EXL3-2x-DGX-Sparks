#!/usr/bin/env python3
"""CPU contracts for the hybrid KDA FP8 dispatch (no GPU/vLLM needed).

Executes the real flag, retention and dispatch code with narrow device stubs.
Optional CPU-PyTorch cases also check eager quantization values and layouts.
These checks do not establish native CUDA compilation, kernel parity or
graph safety; the GPU suite remains a separate qualification gate.

Run:  python3 tests/test_kda_fp8_fat.py
"""

from __future__ import annotations

import ast
import os
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


def _class_func_source(cls: str, name: str) -> str:
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    return ast.get_source_segment(SRC.read_text(), sub) or ""
    raise AssertionError(f"{cls}.{name} not found")


def _exec_flag_fn():
    src = _func_source("kda_fp8_fat_enabled")
    ns: dict = {"os": os}
    exec(compile(ast.Module(body=[ast.parse(src).body[0]], type_ignores=[]),
                 str(SRC), "exec"), ns)
    return ns["kda_fp8_fat_enabled"]


class FlagTests(unittest.TestCase):
    def test_values(self):
        fn = _exec_flag_fn()
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_FP8_FAT": "0"}):
            self.assertFalse(fn())
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_FP8_FAT": "1"}):
            self.assertTrue(fn())
        with unittest.mock.patch.dict(os.environ):
            os.environ.pop("GLM53_KDA_FP8_FAT", None)
            self.assertFalse(fn())
        for bad in ("yes", "", " 1", "1 "):
            with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_FP8_FAT": bad}):
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

    def t(self):
        assert self.dim() == 2
        return _FT(self._shape[::-1], dtype=self.dtype,
                   stride=self._stride[::-1], device=self.device)

    def view(self, *shape):
        return self.reshape(*shape)

    # eager rowquant ops (shape-preserving stubs)
    def float(self):
        return _FT(self._shape, dtype="f32", device=self.device)

    def abs(self):
        return self

    def amax(self, dim=None, keepdim=False):
        shape = list(self._shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)
        return _FT(shape, dtype="f32", device=self.device)

    def clamp_min(self, v):
        return self

    def clamp(self, lo, hi):
        return self

    def to(self, dtype=None, **kw):
        return _FT(self._shape, dtype=dtype or self.dtype, device=self.device)

    def __truediv__(self, other):
        return _FT(self._shape, dtype="f32", device=self.device)

    def __rtruediv__(self, other):
        return _FT(self._shape, dtype="f32", device=self.device)


class _FakeTorch(types.SimpleNamespace):
    pass


def _fake_torch(scaled_mm=None, capability=(12, 1), cap_raises=False):
    aten = types.SimpleNamespace()
    if scaled_mm is not None:
        aten._scaled_mm = scaled_mm
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
        ones=lambda shape, dtype=None, device=None: _FT(shape, dtype=dtype, device=device),
        cuda=types.SimpleNamespace(
            get_device_capability=get_cap, synchronize=synchronize),
        ops=types.SimpleNamespace(aten=aten),
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


def _load_method_class(torch_fake, *, triton_available=False, triton_ready=None,
                       rowquant_kernel=None):
    """Exec the real class + helpers from overlay/exl3.py with stubs."""
    source = SRC.read_text()
    tree = ast.parse(source)
    wanted_fns = {"kda_fp8_fat_enabled"}
    wanted_assigns = {"KDA_FP8_FAT_SHAPES", "KDA_FP8_FAT_M_MAX_MARLIN"}
    body = [
        ast.ImportFrom(module="__future__",
                       names=[ast.alias(name="annotations")], level=0)
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_fns:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted_assigns
            for t in node.targets
        ):
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
        "torch": torch_fake,
        "UnquantizedLinearMethod": _Base,
        "_FAT_TRITON_AVAILABLE": triton_available,
        "_FAT_TRITON_READY": [triton_ready],
        "_fat_rowquant_kernel": rowquant_kernel,
        "_warm_fat_triton": lambda device: None,
        "logger": types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
        ),
    }
    mod = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(mod, str(SRC), "exec"), env)
    return env


def _fat_layer(n=_N, k=_K):
    fp8 = _FT((n, k), dtype="fp8")
    return types.SimpleNamespace(
        glm53_fat_wt=fp8.t(),
        glm53_fat_w=fp8,
        glm53_fat_s=_FT((n,), dtype="f32"),
        glm53_fat_sb=_FT((1, n), dtype="f32"),
        glm53_fp8_n=n,
        glm53_fp8_k=k,
        weight=_FT((n, k), dtype="fp8"),
        weight_scale=_FT((n,), dtype="bf16"),
        workspace=object(),
    )


class ApplyDispatchTests(unittest.TestCase):
    """Real apply() source: which GEMM path runs for which input metadata."""

    def setUp(self):
        self.marlin_calls = []
        self.mods = _marlin_stub(self.marlin_calls)
        self._patcher = unittest.mock.patch.dict(sys.modules, self.mods)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
        self.fat_calls = []
        env = _load_method_class(_fake_torch())
        self.cls = env["Glm53DenseFp8Method"]
        self.meth = self.cls("kda")
        self.meth.ready = True

        def spy(layer, x2):
            self.fat_calls.append(x2)
            return _FT((x2.shape[0], _N))

        self.meth._apply_fat = spy

    def apply(self, x, bias=None, layer=None):
        return self.meth.apply(layer or _fat_layer(), x, bias)

    def test_m_boundary(self):
        self.assertEqual(len(self.fat_calls), 0)
        self.apply(_FT((64, _K)))
        self.assertEqual(len(self.fat_calls), 0)
        self.assertEqual(len(self.marlin_calls), 1)
        out = self.apply(_FT((65, _K)))
        self.assertEqual(len(self.fat_calls), 1)
        self.assertEqual(out.shape, (65, _N))

    def test_flattened_m_for_3d(self):
        # Regression: dispatch used x.shape[-2] (40 < 64 -> Marlin) though the
        # flattened M is 80. The real contract is numel/K.
        out = self.apply(_FT((2, 40, _K)))
        self.assertEqual(len(self.fat_calls), 1)
        self.assertEqual(self.fat_calls[0].shape, (80, _K))
        self.assertEqual(out.shape, (2, 40, _N))
        # 2x32 = 64 rows stays Marlin.
        self.apply(_FT((2, 32, _K)))
        self.assertEqual(len(self.fat_calls), 1)
        self.assertEqual(len(self.marlin_calls), 1)

    def test_bias_and_missing_retention_stay_marlin(self):
        self.apply(_FT((512, _K)), bias=_FT((_N,)))
        layer = _fat_layer()
        del layer.glm53_fat_wt
        self.apply(_FT((512, _K)), layer=layer)
        self.apply(_FT((512, _K + 8)))  # wrong K
        self.assertEqual(len(self.fat_calls), 0)
        self.assertEqual(len(self.marlin_calls), 3)

    def test_not_ready_uses_base(self):
        self.meth.ready = False
        out = self.apply(_FT((512, _K)))
        self.assertEqual(out[0], "base_apply")
        self.assertEqual(len(self.fat_calls), 0)


class RowquantTests(unittest.TestCase):
    """Real _fat_rowquant source: Triton only for contiguous-last-dim input."""

    def _method(self, *, ready, kernel):
        env = _load_method_class(
            _fake_torch(), triton_available=True, triton_ready=ready,
            rowquant_kernel=kernel)
        meth = env["Glm53DenseFp8Method"]("kda")
        return meth

    def test_triton_used_for_contiguous(self):
        launched = []

        class _Kernel:
            def __getitem__(self, grid):
                def launch(x, q, s, stride_m, k, block, num_warps):
                    launched.append((grid, stride_m, k, block))
                return launch

        meth = self._method(ready=True, kernel=_Kernel())
        xq, sa = meth._fat_rowquant(_FT((96, _K)))
        self.assertEqual(len(launched), 1)
        self.assertEqual(launched[0][0], (96,))
        self.assertEqual(xq.dtype, "fp8")
        self.assertEqual(sa.shape, (96, 1))

    def test_strided_last_dim_falls_back_to_eager(self):
        # Regression: the kernel indexes x[row*stride_m + col]; a
        # stride(-1) != 1 view would read wrong elements silently.
        launched = []

        class _Kernel:
            def __getitem__(self, grid):
                return lambda *a, **k: launched.append(a)

        meth = self._method(ready=True, kernel=_Kernel())
        strided = _FT((96, _K), stride=(_K * 2, 2))
        xq, sa = meth._fat_rowquant(strided)
        self.assertEqual(launched, [])
        self.assertEqual(xq.dtype, "fp8")
        self.assertEqual(sa.shape, (96, 1))

    def test_eager_when_triton_not_ready(self):
        meth = self._method(ready=False, kernel=None)
        xq, sa = meth._fat_rowquant(_FT((96, _K)))
        self.assertEqual(xq.dtype, "fp8")
        self.assertEqual(sa.shape, (96, 1))


class EagerCpuTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("install CPU PyTorch for tensor-value regressions")
        cls.torch = torch

    def test_zero_and_tiny_rows_use_the_kernel_scale_floor(self):
        torch = self.torch
        method = _load_method_class(torch)["Glm53DenseFp8Method"]("kda")
        x = torch.zeros((3, _K), dtype=torch.bfloat16, device="cpu")
        x[1].fill_(1e-13)
        x[2].fill_(-1e-13)
        quantized, scales = method._fat_rowquant(x)
        self.assertTrue(torch.equal(scales, torch.full((3, 1), 1e-12, device="cpu")))
        expected = torch.zeros((3, _K), dtype=torch.float32, device="cpu")
        expected[1].fill_(13 / 128)
        expected[2].fill_(-13 / 128)
        self.assertTrue(torch.equal(quantized.float(), expected))

    def test_strided_values_do_not_read_interleaved_storage(self):
        torch = self.torch

        class ForbiddenKernel:
            def __getitem__(self, grid):
                raise AssertionError("strided input reached the contiguous-only kernel")

        environment = _load_method_class(
            torch, triton_available=True, triton_ready=True,
            rowquant_kernel=ForbiddenKernel(),
        )
        method = environment["Glm53DenseFp8Method"]("kda")
        storage = torch.full((2, _K * 2), 1000, dtype=torch.bfloat16, device="cpu")
        storage[:, ::2] = 1
        quantized, scales = method._fat_rowquant(storage[:, ::2])
        self.assertTrue(torch.equal(
            quantized.float(), torch.full((2, _K), 448.0, device="cpu")))
        self.assertTrue(torch.equal(
            scales, torch.full((2, 1), 1 / 448, device="cpu")))


class RetentionTests(unittest.TestCase):
    """Real _retain_fat_weights source: every predicate must pass, and the
    _scaled_mm probe must actually launch (has-operator != working kernel)."""

    def _env(self, *, flag="1", scaled_mm=None, capability=(12, 1),
             cap_raises=False):
        tf = _fake_torch(scaled_mm=scaled_mm, capability=capability,
                         cap_raises=cap_raises)
        env = _load_method_class(tf)
        env["_warm_fat_triton"] = lambda device: setattr(
            self, "warmed", True)
        return env["Glm53DenseFp8Method"]("kda"), tf

    def _retain(self, meth, n=_N, k=_K, group_layer=None):
        layer = group_layer or types.SimpleNamespace(orig_dtype="bf16")
        fp8 = _FT((n, k), dtype="fp8")
        scales = _FT((n,), dtype="f32")
        meth._retain_fat_weights(layer, fp8, scales, n, k)
        return layer

    def test_full_predicates_retain(self):
        probe = []
        meth, tf = self._env(scaled_mm=lambda *a: probe.append(a))
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_FP8_FAT": "1"}):
            layer = self._retain(meth)
        self.assertEqual(len(probe), 1)  # real launch attempted
        self.assertIsNotNone(getattr(layer, "glm53_fat_wt", None))
        self.assertEqual(layer.glm53_fat_wt.shape, (_K, _N))
        self.assertEqual(layer.glm53_fat_wt.stride(), (1, _K))  # col-major view
        self.assertEqual(layer.glm53_fat_sb.shape, (1, _N))
        self.assertTrue(getattr(self, "warmed", False))
        self.assertGreater(probe[0][0].shape[0], 64)

    def test_probe_failure_retains_nothing(self):
        def boom(*a):
            raise RuntimeError("no kernel image")
        meth, tf = self._env(scaled_mm=boom)
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_FP8_FAT": "1"}):
            layer = self._retain(meth)
        self.assertFalse(hasattr(layer, "glm53_fat_wt"))

    def test_missing_operator_retains_nothing(self):
        meth, tf = self._env(scaled_mm=None)  # no _scaled_mm attr at all
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_FP8_FAT": "1"}):
            layer = self._retain(meth)
        self.assertFalse(hasattr(layer, "glm53_fat_wt"))

    def test_wrong_capability_retains_nothing(self):
        probe = []
        meth, tf = self._env(scaled_mm=lambda *a: probe.append(a),
                             capability=(12, 0))
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_FP8_FAT": "1"}):
            layer = self._retain(meth)
        self.assertFalse(hasattr(layer, "glm53_fat_wt"))
        self.assertEqual(probe, [])

    def test_flag_off_and_wrong_group_retain_nothing(self):
        probe = []
        meth, tf = self._env(scaled_mm=lambda *a: probe.append(a))
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_FP8_FAT": "0"}):
            layer = self._retain(meth)
        self.assertFalse(hasattr(layer, "glm53_fat_wt"))
        env = _load_method_class(tf)
        dense = env["Glm53DenseFp8Method"]("dense")
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_FP8_FAT": "1"}):
            layer = self._retain(dense)
        self.assertFalse(hasattr(layer, "glm53_fat_wt"))
        self.assertEqual(probe, [])

    def test_wrong_shape_retains_nothing(self):
        probe = []
        meth, tf = self._env(scaled_mm=lambda *a: probe.append(a))
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_FP8_FAT": "1"}):
            layer = self._retain(meth, n=4096, k=4096)
        self.assertFalse(hasattr(layer, "glm53_fat_wt"))
        self.assertEqual(probe, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
