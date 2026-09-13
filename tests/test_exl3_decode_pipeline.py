#!/usr/bin/env python3
"""CPU contracts for the opt-in SM121 thin-decode pipeline.

Covers, without a GPU or vLLM install:
  * `overlay/patch_exl3_decode_pipeline.py` is additive, restricted to the
    K4/N256 SM121 case, refuses to double-apply, and never partially writes;
  * `overlay/exl3.py::build_exl3_fused_state` aliases the up-SUH pointer
    table onto the gate-SUH table only after the load-time shared-SUH flag,
    and fails closed when GLM53_EXL3_MOE_FAST=1 names an image without the
    native fast kernels.

Run:  python3 tests/test_exl3_decode_pipeline.py
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _load_patch_module():
    spec = importlib.util.spec_from_file_location(
        "decode_pipeline_patch", ROOT / "overlay/patch_exl3_decode_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATCHER = _load_patch_module()

# Fixture mirrors the real pinned-source anchors (indentation included).
FIXTURE_KERNEL = '''template<int t_bits, int MOE_TILESIZE_N>
void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS) {
                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );
gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);
}
'''
FIXTURE_HOST = ('#include <set>\n'
                '    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[2 * K + N_off];\n')
FIXTURE_BINDINGS = '    m.def("exl3_moe", &exl3_moe, "exl3_moe");\n'


class NativePatchTests(unittest.TestCase):
    def fixture(self, root: Path):
        (root / "quant/comp_units").mkdir(parents=True)
        (root / "quant/exl3_moe_kernel.cuh").write_text(FIXTURE_KERNEL)
        (root / "quant/exl3_moe.cu").write_text(FIXTURE_HOST)
        (root / "bindings.cpp").write_text(FIXTURE_BINDINGS)

    def test_patch_is_additive_and_restricted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            PATCHER.patch(root)
            # Stock kernel source is byte-identical afterwards.
            self.assertEqual(
                (root / "quant/exl3_moe_kernel.cuh").read_text(), FIXTURE_KERNEL
            )
            fast = (root / "quant/glm53_exl3_moe_fast_kernel.cuh").read_text()
            self.assertIn("glm53_exl3_moe_fast_kernel", fast)
            self.assertIn("bool shared_input", fast)
            self.assertIn("if constexpr (!shared_input)", fast)
            self.assertIn(
                "shared_input ? temp_state_g : temp_state_u", fast
            )
            wrapper = (root / "quant/comp_units/glm53_exl3_moe_fast.cu").read_text()
            self.assertIn("#define MOE_FRAG_STAGES 1", wrapper)
            self.assertIn("#define MOE_SH_STAGES 8", wrapper)
            self.assertIn("glm53_exl3_moe_fast_kernel<4, 256, true>", wrapper)
            self.assertIn("glm53_exl3_moe_fast_kernel<4, 256, false>", wrapper)
            host = (root / "quant/exl3_moe.cu").read_text()
            self.assertIn("K == 4 && N_off == 1", host)
            self.assertIn("major == 12 && minor == 1", host)
            self.assertIn(
                "gate_ptrs_suh.data_ptr() == up_ptrs_suh.data_ptr()", host
            )
            self.assertIn('GLM53_EXL3_MOE_FAST must be 0 or 1', host)
            bindings = (root / "bindings.cpp").read_text()
            self.assertIn('m.def("exl3_moe", &exl3_moe, "exl3_moe");', bindings)
            self.assertIn("glm53_fast_moe_version", bindings)
            # Double apply refuses; host is unchanged by the second attempt.
            with self.assertRaises(RuntimeError):
                PATCHER.patch(root)
            self.assertEqual((root / "quant/exl3_moe.cu").read_text(), host)

    def test_bad_anchor_does_not_partially_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "bindings.cpp").write_text("unknown upstream binding")
            old_host = (root / "quant/exl3_moe.cu").read_text()
            with self.assertRaises(RuntimeError):
                PATCHER.patch(root)
            self.assertEqual((root / "quant/exl3_moe.cu").read_text(), old_host)
            self.assertFalse(
                (root / "quant/glm53_exl3_moe_fast_kernel.cuh").exists()
            )
            self.assertFalse(
                (root / "quant/comp_units/glm53_exl3_moe_fast.cu").exists()
            )


class _FakeTensor:
    _next_ptr = [10**12]

    def __init__(self, shape=()):
        self._shape = tuple(shape)
        self._ptr = _FakeTensor._next_ptr[0]
        _FakeTensor._next_ptr[0] += 1

    @property
    def shape(self):
        return self._shape

    def data_ptr(self):
        return self._ptr

    def numel(self):
        n = 1
        for d in self._shape:
            n *= d
        return n

    def element_size(self):
        return 2


class _FakeTorch:
    int64 = "int64"
    float16 = "float16"

    @staticmethod
    def tensor(values, dtype=None, device=None):
        return _FakeTensor(shape=(len(list(values)),))

    @staticmethod
    def empty(shape, dtype=None, device=None):
        if isinstance(shape, int):
            shape = (shape,)
        return _FakeTensor(shape=tuple(shape))


class _FakeDevice:
    def __init__(self, index=0):
        self.index = index

    def __str__(self):
        return f"cuda:{self.index}"


def _extract_build_fn():
    source = (ROOT / "overlay/exl3.py").read_text()
    tree = ast.parse(source)
    function = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_exl3_fused_state"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    tree = ast.fix_missing_locations(
        ast.Module(body=[future, function], type_ignores=[])
    )
    env = {
        "torch": _FakeTorch(),
        "os": os,
        "temp_rows_fused": lambda: 128,
        "_FUSED_TEMP_CACHE": {},
        "_EXL3_FAT_DIAG": {"fused_temps_allocs": 0, "fused_temps_bytes": 0},
    }
    exec(compile(tree, str(ROOT / "overlay/exl3.py"), "exec"), env)
    return env["build_exl3_fused_state"]


def _fake_layer():
    dev = _FakeDevice(0)
    layer = SimpleNamespace(
        w13_trellis=SimpleNamespace(device=dev),
        _exl3_hidden_size=4096,
        _exl3_intermediate_local=1024,
        _exl3_bits=4,
        _exl3_shared_w13_suh=False,
    )
    inners = []
    for _ in range(2):
        pack = {}
        for name in ("gate", "up", "down"):
            pack[name] = SimpleNamespace(
                trellis=_FakeTensor(), suh=_FakeTensor(), svh=_FakeTensor()
            )
        inners.append(pack)
    return layer, inners


class BuildStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = staticmethod(_extract_build_fn())

    def run_build(self, layer, inners, ext, env):
        with patch.dict(sys.modules, {"exllamav3_ext": ext}), patch.dict(
            os.environ, env, clear=False
        ):
            # Clear only our keys; keep the rest of the environment intact.
            for key in ("GLM53_EXL3_MOE_FAST",):
                if key not in env:
                    os.environ.pop(key, None)
            self.build(layer, inners)

    def test_alias_requires_verified_shared_flag(self):
        ext = SimpleNamespace(exl3_moe_max_concurrency=lambda idx: 6)
        layer, inners = _fake_layer()
        self.run_build(layer, inners, ext, {"GLM53_EXL3_MOE_FAST": "0"})
        self.assertIsNot(
            layer._exl3_ptrs["gate_suh"], layer._exl3_ptrs["up_suh"]
        )
        # svh tables are never aliased.
        self.assertIsNot(
            layer._exl3_ptrs["gate_svh"], layer._exl3_ptrs["up_svh"]
        )
        layer._exl3_shared_w13_suh = True
        self.run_build(layer, inners, ext, {"GLM53_EXL3_MOE_FAST": "0"})
        self.assertIs(layer._exl3_ptrs["gate_suh"], layer._exl3_ptrs["up_suh"])
        self.assertIsNot(
            layer._exl3_ptrs["gate_svh"], layer._exl3_ptrs["up_svh"]
        )

    def test_fast_mode_fails_closed_without_native(self):
        ext = SimpleNamespace(exl3_moe_max_concurrency=lambda idx: 6)
        layer, inners = _fake_layer()
        with self.assertRaisesRegex(RuntimeError, "requires the native"):
            self.run_build(layer, inners, ext, {"GLM53_EXL3_MOE_FAST": "1"})

    def test_fast_mode_rejects_wrong_native_version(self):
        ext = SimpleNamespace(
            exl3_moe_max_concurrency=lambda idx: 6,
            glm53_fast_moe_version=lambda: 2,
        )
        layer, inners = _fake_layer()
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            self.run_build(layer, inners, ext, {"GLM53_EXL3_MOE_FAST": "1"})

    def test_fast_mode_accepts_native_v1(self):
        ext = SimpleNamespace(
            exl3_moe_max_concurrency=lambda idx: 6,
            glm53_fast_moe_version=lambda: 1,
        )
        layer, inners = _fake_layer()
        layer._exl3_shared_w13_suh = True
        self.run_build(layer, inners, ext, {"GLM53_EXL3_MOE_FAST": "1"})
        self.assertIs(layer._exl3_ptrs["gate_suh"], layer._exl3_ptrs["up_suh"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
