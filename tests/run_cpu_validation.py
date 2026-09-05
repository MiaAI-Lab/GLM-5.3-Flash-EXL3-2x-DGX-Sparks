#!/usr/bin/env python3
"""Run the checkout's CPU checks with explicit skips for integration checks.

Requires CPU PyTorch, NumPy and Jinja2. Does not install dependencies, contact
servers or run GPU tests. Each skipped entry names the omitted coverage.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = {
    "test_exl3_overlay.py": "separate image/GPU self-check",
    "test_hybrid_prefix_hit.py": "requires the installed pinned vLLM source",
    "test_scheduler_decode_floor.py": "requires the installed pinned vLLM source",
    "test_mixed_prefill_decode.py": "separate live-API integration test",
}
OPTIONAL = {
    ("test_xgrammar_termination.py", "test_installed_copy_if_present"),
    ("test_indexer_workspace.py", "test_live_copy_if_present"),
    ("test_indexer_workspace.py", "test_live_container_copy_if_enabled"),
    ("test_kpool_tail_slotmap.py", "test_installed_copy_if_present"),
    ("test_spinwait_patch.py", "test_live_source_if_enabled"),
}


def skipped(label, reason):
    def run():
        raise unittest.SkipTest(reason)
    return unittest.FunctionTestCase(run, description=label)


def suite():
    tests = unittest.TestSuite()
    sys.path.insert(0, str(ROOT / "overlay"))
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        if path.name in BLOCKED:
            tests.addTest(skipped(path.name, BLOCKED[path.name]))
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        tests.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        for name, function in sorted(vars(module).items()):
            if not inspect.isfunction(function) or function.__module__ != module.__name__:
                continue
            if not (name.startswith("test_") or (path.name == "test_ablit.py" and name.startswith("check_"))):
                continue
            label = f"{path.name}::{name}"
            if (path.name, name) in OPTIONAL:
                tests.addTest(skipped(label, "installed-source or live-server coverage is outside this CPU suite"))
            elif path.name == "test_ablit.py" and name == "check_transplant":
                def transplant(fn=function):
                    with tempfile.TemporaryDirectory() as tmp:
                        fn(tmp)
                tests.addTest(unittest.FunctionTestCase(transplant, description=label))
            else:
                tests.addTest(unittest.FunctionTestCase(function, description=label))
    tests.addTest(skipped("test_ablit.py::check_transplant::CUDA subcase", "GPU execution disabled for this CPU suite"))
    return tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    result = unittest.TextTestRunner(verbosity=2).run(suite())
    report = {
        "run": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "failures": len(result.failures), "errors": len(result.errors),
        "skips": [(test.shortDescription() or str(test), reason) for test, reason in result.skipped],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
