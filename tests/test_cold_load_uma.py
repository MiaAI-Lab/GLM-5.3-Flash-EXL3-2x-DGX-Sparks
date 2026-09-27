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
        # idempotent: the anchors are consumed, so a second prepare() is a no-op
        assert mod.prepare(out) == out
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


def _budget_ns(logs, meminfo, cuda_free, drop, env=None):
    """Exec the helper slice with fake os/torch/platform; returns the namespace."""
    out = _run(FIXTURE)
    helper_src = out[out.index("# [glm53-cold-load-uma:v1] helpers") : out.index("def instanttensor_weights_iterator(")]

    class L:
        def info(self, *a): logs.append(("info", a))
        def warning(self, *a): logs.append(("warn", a))

    fake_os = types.SimpleNamespace(
        sysconf=lambda k: 65536, environ=env or {}, sync=lambda: None, path=os.path
    )
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
    # stub the meminfo reader and drop_caches; the helper looks these up as
    # globals of the exec namespace
    ns["_glm53_meminfo_kib"] = lambda f: meminfo[f]
    ns["_glm53_uma_drop_caches"] = drop
    g.update(ns)
    sys.modules["torch"] = torch  # type: ignore[assignment]
    sys.modules.setdefault("instanttensor", types.ModuleType("instanttensor"))
    return ns


def test_budget_math():
    logs: list = []
    meminfo = {"MemFree": 2 << 20, "MemAvailable": 110 << 20}  # KiB: 2 GiB free, 110 GiB avail
    dropped = {"n": 0}
    cuda_free = [2 << 30]

    def drop():
        dropped["n"] += 1
        cuda_free[0] = 100 << 30
        return True

    ns = _budget_ns(logs, meminfo, cuda_free, drop)
    try:
        ns["_glm53_uma_prepare_instanttensor_budget"](["/dev/null"])
    finally:
        del sys.modules["torch"]
    st = ns["_GLM53_UMA_STATE"]
    assert dropped["n"] == 1, "should drop caches when MemFree is short but MemAvailable suffices"
    assert st["max_free_mem_usage"] == 0.5
    assert st["buffer_size"] == 4 << 30
    assert any(k == "info" for k, _ in logs)


def test_budget_math_drop_fails_full_cache():
    """Our observed failure: /proc/sys is read-only in the stock container so
    drop_caches fails, cuda free == 2 x the 647100416 B budget we logged, and
    the largest tensor is 1268776960 B. The budget must come from the
    MemAvailable window and the buffer must cover the largest tensor."""
    logs: list = []
    free = 2 * 647100416  # cuda free == host MemFree on UMA
    meminfo = {"MemFree": free // 1024, "MemAvailable": 100 << 20}  # KiB
    dropped = {"n": 0}
    cuda_free = [free]

    def drop():
        dropped["n"] += 1
        return False  # containers cannot drop caches (no CAP_SYS_ADMIN)

    ns = _budget_ns(logs, meminfo, cuda_free, drop)
    largest = 1268776960
    with tempfile.TemporaryDirectory() as td:
        shard = Path(td) / "shard.safetensors"
        with open(shard, "wb") as fh:
            fh.truncate(largest)
        try:
            ns["_glm53_uma_prepare_instanttensor_budget"]([str(shard)])
        finally:
            del sys.modules["torch"]
    st = ns["_GLM53_UMA_STATE"]
    assert dropped["n"] == 1, "the in-container drop attempt stays (harmless)"
    frac = st["max_free_mem_usage"]
    assert frac > 1, frac  # sized against MemAvailable, not cuda free
    assert frac <= 0.9 * (100 << 30) / free, frac
    assert int(free * frac) >= (4 << 30) + largest, int(free * frac)
    assert st["buffer_size"] == 4 << 30  # io_depth stays at the backend default
    assert st["buffer_size"] >= largest
    assert not any(k == "warn" for k, _ in logs), logs


def test_budget_math_discrete_gpu_unchanged():
    meminfo = {"MemFree": 2 << 20, "MemAvailable": 110 << 20}
    dropped = {"n": 0}
    cuda_free = [80 << 30]  # device free nowhere near host MemFree

    def drop():
        dropped["n"] += 1
        return True

    ns = _budget_ns([], meminfo, cuda_free, drop)
    try:
        ns["_glm53_uma_prepare_instanttensor_budget"](["/dev/null"])
    finally:
        del sys.modules["torch"]
    assert dropped["n"] == 0
    assert ns["_GLM53_UMA_STATE"] == {"max_free_mem_usage": None, "buffer_size": None}

    # env overrides pass through unchanged on the discrete path
    ns = _budget_ns(
        [], meminfo, cuda_free, drop,
        env={"INSTANTTENSOR_MAX_FREE_MEM_USAGE": "0.7", "INSTANTTENSOR_BUFFER_SIZE": str(2 << 30)},
    )
    try:
        ns["_glm53_uma_prepare_instanttensor_budget"](["/dev/null"])
    finally:
        del sys.modules["torch"]
    assert ns["_GLM53_UMA_STATE"] == {"max_free_mem_usage": 0.7, "buffer_size": 2 << 30}


def test_budget_math_env_overrides_win_on_uma():
    free = 2 * 647100416
    meminfo = {"MemFree": free // 1024, "MemAvailable": 100 << 20}
    ns = _budget_ns(
        [], meminfo, [free], lambda: False,
        env={"INSTANTTENSOR_MAX_FREE_MEM_USAGE": "1.5", "INSTANTTENSOR_BUFFER_SIZE": str(2 << 30)},
    )
    try:
        ns["_glm53_uma_prepare_instanttensor_budget"](["/dev/null"])
    finally:
        del sys.modules["torch"]
    st = ns["_GLM53_UMA_STATE"]
    assert st["max_free_mem_usage"] == 1.5  # >1 env values are legal now
    assert st["buffer_size"] == 2 << 30


def _run_budget(meminfo, free, env, largest=1268776960):
    logs: list = []
    ns = _budget_ns(logs, meminfo, [free], lambda: False, env=env)
    with tempfile.TemporaryDirectory() as td:
        shard = Path(td) / "shard.safetensors"
        with open(shard, "wb") as fh:
            fh.truncate(largest)
        try:
            ns["_glm53_uma_prepare_instanttensor_budget"]([str(shard)])
        finally:
            del sys.modules["torch"]
    return ns["_GLM53_UMA_STATE"], logs


def test_bare_env_budget_raised_when_page_cache_is_full():
    """#273: the #204/#273 workaround sets only INSTANTTENSOR_MAX_FREE_MEM_USAGE.
    On UMA with the page cache full, 0.8 x cuda free cannot hold the pinned
    4 GiB buffer; the helper must raise it to the MemAvailable-sized fraction
    (with a warning) instead of letting InstantTensor abort."""
    free = 2 * 647100416  # ~1.2 GiB cuda free == MemFree, as in the #230 receipt
    largest = 1268776960
    st, logs = _run_budget(
        {"MemFree": free // 1024, "MemAvailable": 100 << 20}, free,
        {"INSTANTTENSOR_MAX_FREE_MEM_USAGE": "0.8"}, largest,
    )
    assert st["max_free_mem_usage"] > 1, st
    assert st["buffer_size"] == 4 << 30
    assert int(free * st["max_free_mem_usage"]) >= st["buffer_size"] + largest, st
    assert any(k == "warn" and "INSTANTTENSOR_MAX_FREE_MEM_USAGE" in a[0] for k, a in logs), logs


def test_bare_env_budget_kept_when_it_fits():
    free = 60 << 30  # plenty of cuda free: the caller's fraction already holds the load
    st, logs = _run_budget(
        {"MemFree": free // 1024, "MemAvailable": 100 << 20}, free,
        {"INSTANTTENSOR_MAX_FREE_MEM_USAGE": "0.8"},
    )
    assert st["max_free_mem_usage"] == 0.8, st
    assert not any(k == "warn" for k, _ in logs), logs


def test_bare_env_budget_kept_when_it_covers_the_buffer():
    """Between the buffer and the full load window the override already loads
    on main; the fix must not touch it (only the abort case is changed)."""
    free = 6 << 30  # 0.8 x 6 GiB = 4.8 GiB >= the 4 GiB pinned buffer, < the ~9.8 GiB window
    st, logs = _run_budget(
        {"MemFree": free // 1024, "MemAvailable": 100 << 20}, free,
        {"INSTANTTENSOR_MAX_FREE_MEM_USAGE": "0.8"},
    )
    assert st["max_free_mem_usage"] == 0.8, st
    assert st["buffer_size"] == 4 << 30
    assert not any(k == "warn" for k, _ in logs), logs


def test_bare_env_budget_never_lowered():
    free = 2 * 647100416
    st, _ = _run_budget(
        {"MemFree": free // 1024, "MemAvailable": 100 << 20}, free,
        {"INSTANTTENSOR_MAX_FREE_MEM_USAGE": "9.0"},
    )
    assert st["max_free_mem_usage"] == 9.0, st


def test_explicit_pair_untouched_on_full_cache():
    """Setting INSTANTTENSOR_BUFFER_SIZE too is an explicit choice: leave both."""
    free = 2 * 647100416
    st, logs = _run_budget(
        {"MemFree": free // 1024, "MemAvailable": 100 << 20}, free,
        {"INSTANTTENSOR_MAX_FREE_MEM_USAGE": "0.8", "INSTANTTENSOR_BUFFER_SIZE": str(1 << 30)},
    )
    assert st == {"max_free_mem_usage": 0.8, "buffer_size": 1 << 30}, st
    assert not any(k == "warn" for k, _ in logs), logs


# The PR #230 helper exactly as merged (main 75a0e9e / #230 fix-up head 3432f57,
# unchanged through 70b2f33). Frozen as a digest rather than a copy so the bake,
# which copies only this file into the image, needs no extra fixture file.
V1_HELPER_SHA256 = "e6c6122a16ce50d76c560994c76101409deda9548b7d3c7492f44a7cb62b1b47"


def test_v1_helper_matches_frozen_history():
    """HELPER_V1 is derived from HELPER; a future edit outside the budget
    block would silently change both. Pin it to the historical bytes."""
    import hashlib

    assert hashlib.sha256(mod.HELPER_V1.encode()).hexdigest() == V1_HELPER_SHA256


def test_v1_upgrade_through_cli():
    """The real main() upgrades a v1-baked file in place, then is a no-op."""
    v1 = _run(FIXTURE).replace(mod.HELPER, mod.HELPER_V1, 1)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "weight_utils.py"
        p.write_text(v1)
        mod.TARGET = p
        saved = os.environ.pop(mod.ENV_NAME, None)
        try:
            assert mod.main() == 0
            first = p.read_text()
            assert mod.verified_state(first) == "patched" and mod.HELPER in first
            assert mod.main() == 0
            assert p.read_text() == first
        finally:
            if saved is not None:
                os.environ[mod.ENV_NAME] = saved


def test_v1_helper_upgraded_in_place():
    """An image baked with the PR #230 (v1) helper must not fail closed as
    'source drift': prepare() swaps in the current helper, then is a no-op."""
    out = _run(FIXTURE)
    v1 = out.replace(mod.HELPER, mod.HELPER_V1, 1)
    assert v1 != out and mod.HELPER not in v1
    assert mod.verified_state(v1) == "patched-v1"
    up = mod.prepare(v1)
    assert up == out
    assert mod.verified_state(up) == "patched"
    compile(up, "weight_utils.py", "exec")
    assert mod.prepare(up) == up


def test_budget_math_kill_switch():
    """GLM53_COLD_LOAD_UMA=0 at runtime leaves InstantTensor on its own
    defaults (the image is patched at build, so this is the runtime off)."""
    free = 2 * 647100416
    meminfo = {"MemFree": free // 1024, "MemAvailable": 100 << 20}
    dropped = {"n": 0}

    def drop():
        dropped["n"] += 1
        return True

    ns = _budget_ns([], meminfo, [free], drop, env={"GLM53_COLD_LOAD_UMA": "0"})
    try:
        ns["_glm53_uma_prepare_instanttensor_budget"](["/dev/null"])
    finally:
        del sys.modules["torch"]
    assert ns["_GLM53_UMA_STATE"] == {}
    assert dropped["n"] == 0


def test_env_number_parsing():
    out = _run(FIXTURE)
    helper_src = out[out.index("# [glm53-cold-load-uma:v1] helpers") : out.index("def instanttensor_weights_iterator(")]
    warns: list = []

    class L:
        def info(self, *a): pass
        def warning(self, *a): warns.append(a)

    ns: dict = {}
    env = {"INSTANTTENSOR_MAX_FREE_MEM_USAGE": "abc", "INSTANTTENSOR_BUFFER_SIZE": "-5"}
    fake_os = types.SimpleNamespace(sysconf=lambda k: 65536, environ=env, sync=lambda: None, path=os.path)
    exec(helper_src, {"os": fake_os, "logger": L(), "current_platform": None, "__builtins__": __builtins__}, ns)
    fn = ns["_glm53_env_number"]
    assert fn("INSTANTTENSOR_MAX_FREE_MEM_USAGE", float, 0.0, 1.0) is None
    assert fn("INSTANTTENSOR_BUFFER_SIZE", int, 1, None) is None
    assert len(warns) == 2
    env["INSTANTTENSOR_MAX_FREE_MEM_USAGE"] = "0.75"
    assert fn("INSTANTTENSOR_MAX_FREE_MEM_USAGE", float, 0.0, 1.0) == 0.75
    assert fn("MISSING", int, None, None) is None
    # the UMA helper lifts the upper bound: >1 fractions are legal
    env["INSTANTTENSOR_MAX_FREE_MEM_USAGE"] = "1.5"
    assert fn("INSTANTTENSOR_MAX_FREE_MEM_USAGE", float, 0.0, None) == 1.5


def test_installed_optin():
    if os.environ.get("GLM53_REQUIRE_TARGET") != "1":
        return
    assert INSTALLED.is_file(), INSTALLED
    _check_installed(INSTALLED)


def _check_installed(path: Path) -> str:
    """Pre-patch validation of an installed weight_utils.py: stock, current, or
    an image baked with the PR #230 (v1) helper are all supported inputs, and
    applying the patch must leave the file fully patched."""
    src = path.read_text()
    mod.TARGET = path
    state = mod.verified_state(src)
    assert state in ("stock", "patched", "patched-v1"), state
    if state != "patched":
        out = mod.prepare(src)
        assert mod.verified_state(out) == "patched"
        compile(out, str(path), "exec")
    return state


def test_installed_check_accepts_every_supported_state():
    """#273 review: the opt-in check must not reject a v1-baked image that
    the patch itself upgrades."""
    stock = FIXTURE
    patched = _run(FIXTURE)
    v1 = patched.replace(mod.HELPER, mod.HELPER_V1, 1)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "weight_utils.py"
        for src, want in ((stock, "stock"), (v1, "patched-v1"), (patched, "patched")):
            p.write_text(src)
            assert _check_installed(p) == want


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("test_cold_load_uma OK")
