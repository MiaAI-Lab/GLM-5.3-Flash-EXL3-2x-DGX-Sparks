#!/usr/bin/env python3
"""Host-only tests for overlay/patch_skip_cudagraph_profile.py."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(p for p in (HERE / "patch_skip_cudagraph_profile.py", ROOT / "overlay" / "patch_skip_cudagraph_profile.py") if p.is_file())
spec = importlib.util.spec_from_file_location("p", PATCH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

FIXTURE = (
    "import vllm.envs as envs\n"
    "class W:\n"
    "    def determine_available_memory(self):\n"
    "        profile_result = None\n"
    + mod.OLD
    + "        applied = (\n"
    "            cudagraph_memory_estimate\n"
    "            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS\n"
    "            else 0\n"
    "        )\n"
    "        return applied\n"
)


def test_apply_and_idempotent():
    assert mod.verified_state(FIXTURE) == "stock"
    out = mod.prepare(FIXTURE)
    assert mod.verified_state(out) == "patched"
    assert "and envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS\n" in out
    assert mod.prepare(out) == out
    compile(out, "gpu_worker.py", "exec")


def test_drift_fails_closed():
    for bad in (FIXTURE.replace("cudagraph_memory_estimate = 0\n", "cudagraph_memory_estimate = 1\n", 1), FIXTURE + FIXTURE):
        try:
            mod.verified_state(bad)
        except SystemExit:
            continue
        raise AssertionError("drift must fail closed")


def test_installed_optin():
    if os.environ.get("GLM53_REQUIRE_TARGET") != "1":
        return
    assert mod.TARGET.is_file()
    assert mod.verified_state(mod.TARGET.read_text()) in ("stock", "patched")


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("test_skip_cudagraph_profile OK")
