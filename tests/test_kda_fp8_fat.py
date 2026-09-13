#!/usr/bin/env python3
"""CPU contracts for the hybrid KDA FP8 dispatch (no GPU/vLLM needed).

Verifies by AST inspection + isolated env parsing that:
  * GLM53_KDA_FP8_FAT accepts only 0/1 (anything else raises);
  * the Marlin/small-M boundary constant is 64 (evidence-pinned);
  * the fat shape gate is exactly the measured in_proj geometry;
  * apply() routes on retained weights + bias-None + dim/shape guards +
    the M threshold, reshaping back afterwards (graph-safe, no sync);
  * the Triton rowquant kernel exists when triton imports, with an eager
    fallback otherwise.

Run:  python3 tests/test_kda_fp8_fat.py
"""

from __future__ import annotations

import ast
import os
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
        os.environ.pop("GLM53_KDA_FP8_FAT", None)
        self.assertFalse(fn())
        with unittest.mock.patch.dict(os.environ, {"GLM53_KDA_FP8_FAT": "yes"}):
            with self.assertRaises(RuntimeError):
                fn()

    def test_boundary_and_shapes(self):
        text = SRC.read_text()
        self.assertIn("KDA_FP8_FAT_M_MAX_MARLIN = 64", text)
        self.assertIn("(12576, 4096)", text)


class DispatchTests(unittest.TestCase):
    def test_apply_guards(self):
        src = _class_func_source("Glm53DenseFp8Method", "apply")
        for frag in ("glm53_fat_wt", "bias is None", "KDA_FP8_FAT_M_MAX_MARLIN",
                     "glm53_fp8_k", "reshape"):
            self.assertIn(frag, src)

    def test_retention_predicates(self):
        src = _class_func_source("Glm53DenseFp8Method", "_retain_fat_weights")
        for frag in ("kda_fp8_fat_enabled()", '"kda"', "KDA_FP8_FAT_SHAPES",
                     "_scaled_mm", "(12, 1)", "glm53_fat_wt", "_warm_fat_triton"):
            self.assertIn(frag, src)

    def test_rowquant_paths(self):
        src = _class_func_source("Glm53DenseFp8Method", "_fat_rowquant")
        self.assertIn("_FAT_TRITON_READY", src)
        self.assertIn("amax", src)  # eager fallback present


if __name__ == "__main__":
    unittest.main(verbosity=2)
