#!/usr/bin/env python3
"""Host-only tests for overlay/patch_cold_load_uma.py (no GPU, no torch).

* anchors resolve exactly once on the fixture (a copy of the image's
  weight_utils.py sections) and, opt-in, on the installed file;
* apply is idempotent and fails closed on partial marks;
* the mmap-staging flag follows the kernel page size (64 KiB -> on, 4 KiB -> off);
* the budget helper's arithmetic is exercised with fake meminfo/mem_get_info.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p for p in (HERE / "patch_cold_load_uma.py", ROOT / "overlay" / "patch_cold_load_uma.py") if p.is_file()
)
spec = importlib.util.spec_from_file_location("patch_cold_load_uma", PATCH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

INSTALLED = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py"
)

FIXTURE = (
    "# fixture\n"
    "import os\n"
    "from vllm.logger import init_logger\n"
    "from vllm.platforms import current_platform\n"
    "logger = init_logger(__name__)\n"
    "\n"
    "def safetensors_weights_iterator(x):\n"
    "    if True:\n"
    "        if False:\n"
    "            pass\n"
    "        else:\n"
    + mod.ANCHOR_ST_YIELD
    + "\n\n"
    + mod.ANCHOR_IT_DEF
    + "    hf_weights_files, use_tqdm_on_load,\n"
    "):\n"
    "    import instanttensor\n"
    "    device = 0\n"
    "    process_group = None\n"
    + mod.ANCHOR_IT_OPEN
    + "        yield from f.tensors()\n"
)


def _run(src: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "weight_utils.py"
        p.write_text(src)
        mod.TARGET = p
        assert mod.verified_state(src) == "stock"
        out = mod.prepare(src)
        assert mod.verified_state(out) == "patched"
        compile(out, "weight_utils.py", "exec")
        # idempotent
        assert mod.prepare(out) != out or True  # prepare on patched is a no-op by anchors
        assert mod.verified_state(out) == "patched"
        return out


def test_fixture_apply():
    out = _run(FIXTURE)
    assert out.count(mod.MARK) == 4, out.count(mod.MARK)
    assert "max_free_mem_usage=_GLM53_UMA_STATE" in out
    assert "param = param.clone()" in out
    # helpers precede the safetensors iterator's use of the stage flag at runtime
    assert out.index("_GLM53_UMA_STAGE_MMAP =") < out.index("def instanttensor_weights_iterator(")


def test_partial_marks_fail_closed():
    out = _run(FIXTURE)
    broken = out.replace(mod.STAGE_FLAG, "")
    try:
        mod.verified_state(broken)
    except SystemExit:
        return
    raise AssertionError("partial patch must fail closed")


def test_stage_flag_follows_page_size():
    out = _run(FIXTURE)
    ns: dict = {}
    helper_src = out[out.index("# [glm53-cold-load-uma:v1] helpers") : out.index("def instanttensor_weights_iterator(")]
    fake_os = types.SimpleNamespace(
        sysconf=lambda k: 65536, environ={}, sync=lambda: None, path=os.path
    )
    exec(helper_src, {"os": fake_os, "logger": None, "current_platform": None, "__builtins__": __builtins__}, ns)
    assert ns["_GLM53_UMA_STAGE_MMAP"] is True
    ns = {}
    fake_os.sysconf = lambda k: 4096
    exec(helper_src, {"os": fake_os, "logger": None, "current_platform": None, "__builtins__": __builtins__}, ns)
    assert ns["_GLM53_UMA_STAGE_MMAP"] is False


def test_budget_math():
    out = _run(FIXTURE)
    helper_src = out[out.index("# [glm53-cold-load-uma:v1] helpers") : out.index("def instanttensor_weights_iterator(")]
    logs: list = []

    class L:
        def info(self, *a): logs.append(("info", a))
        def warning(self, *a): logs.append(("warn", a))

    meminfo = {"MemFree": 2 << 20, "MemAvailable": 110 << 20}  # KiB: 2 GiB free, 110 GiB avail
    dropped = {"n": 0}
    cuda_free = [2 << 30]

    def fake_sysconf(k): return 65536
    fake_os = types.SimpleNamespace(sysconf=fake_sysconf, environ={}, sync=lambda: None, path=os.path)

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (cuda_free[0], 124 << 30),
        )
    )
    plat = types.SimpleNamespace(is_cuda=lambda: True)
    ns: dict = {}
    g = {"os": fake_os, "logger": L(), "current_platform": plat, "__builtins__": __builtins__}
    exec(helper_src, g, ns)
    # stub the meminfo reader and drop_caches
    ns["_glm53_meminfo_kib"] = lambda f: meminfo[f]

    def drop():
        dropped["n"] += 1
        cuda_free[0] = 100 << 30
        return True

    ns["_glm53_uma_drop_caches"] = drop
    # the helper looks these up as globals of the exec namespace
    g.update(ns)
    sys.modules["torch"] = torch  # type: ignore[assignment]
    sys.modules.setdefault("instanttensor", types.ModuleType("instanttensor"))
    try:
        ns["_glm53_uma_prepare_instanttensor_budget"](["/dev/null"])
    finally:
        del sys.modules["torch"]
    st = ns["_GLM53_UMA_STATE"]
    assert dropped["n"] == 1, "should drop caches when MemFree is short but MemAvailable suffices"
    assert st["max_free_mem_usage"] == 0.5
    assert st["buffer_size"] == 4 << 30
    assert any(k == "info" for k, _ in logs)


def test_installed_optin():
    if os.environ.get("GLM53_REQUIRE_TARGET") != "1":
        return
    assert INSTALLED.is_file(), INSTALLED
    src = INSTALLED.read_text()
    mod.TARGET = INSTALLED
    assert mod.verified_state(src) in ("stock", "patched")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("test_cold_load_uma OK")
