"""CPU-only checks for overlay/patch_display_kv.py and glm53_display_kv.py."""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay" / "patch_display_kv.py"
MOD = ROOT / "overlay" / "display_kv" / "glm53_display_kv.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_patch_is_idempotent_and_fails_closed(tmp_path, monkeypatch):
    p = _load(PATCH, "patch_display_kv")
    vllm = tmp_path / "vllm"
    (vllm / "v1/worker").mkdir(parents=True)
    worker = vllm / "v1/worker/gpu_worker.py"
    worker.write_text("class W:\n    def f(self):\n" + p.WORKER_OLD + "        return 1\n")
    assert p.apply(worker, [(p.WORKER_OLD, p.WORKER_NEW)]) == "patched"
    assert p.apply(worker, [(p.WORKER_OLD, p.WORKER_NEW)]) == "already patched"
    assert p.WORKER_NEW in worker.read_text()
    # duplicated anchor -> refuse
    dup = vllm / "v1/worker/dup.py"
    dup.write_text(p.WORKER_OLD * 2)
    try:
        p.apply(dup, [(p.WORKER_OLD, p.WORKER_NEW)])
    except SystemExit as e:
        assert "exactly one anchor" in str(e)
    else:
        raise AssertionError("expected drift failure")
    # partial marker -> refuse
    part = vllm / "v1/worker/part.py"
    part.write_text(p.MARK + "\n" + p.WORKER_OLD)
    try:
        p.apply(part, [(p.WORKER_OLD, p.WORKER_NEW)])
    except SystemExit as e:
        assert "partial marker" in str(e)
    else:
        raise AssertionError("expected partial-marker failure")


def test_runner_edit_references_plan_and_index():
    p = _load(PATCH, "patch_display_kv2")
    assert "_glm53_dkv.plan([t.size for t in kv_cache_config.kv_cache_tensors])" in p.RUNNER_HEAD_NEW
    assert "for _glm53_i, kv_cache_tensor in enumerate(" in p.RUNNER_HEAD_NEW
    assert p.RUNNER_NEW_A.count("_glm53_i") == 1 and p.RUNNER_NEW_B.count("_glm53_i") == 1
    assert "_glm53_dkv.report()" in p.RUNNER_NEW_B
    # v2 runner path (the one this image uses for GLM-5.3)
    assert "_glm53_dkv.plan(" in p.ATTN_NEW_HEAD and p.ATTN_NEW_HEAD.count("_glm53_i") == 3
    assert "_glm53_dkv.report()" in p.ATTN_NEW_HEAD
    assert any(path.name == "attn_utils.py" for path, _ in p.EDITS)


def test_mode_parsing(monkeypatch):
    m = _load(MOD, "glm53_display_kv")
    for raw, want in (("1", "on"), ("0", "off"), ("auto", "auto"), ("", "off"), ("ON", "on")):
        monkeypatch.setenv("GLM53_DISPLAY_KV", raw)
        assert m._mode() == want, raw
    monkeypatch.setenv("GLM53_DISPLAY_KV", "maybe")
    try:
        m._mode()
    except ValueError:
        pass
    else:
        raise AssertionError("typo'd knob must not pick a mode")
    monkeypatch.setenv("GLM53_DISPLAY_KV_MIB", "1792")
    assert m._pool_bytes() == 1792 << 20
    monkeypatch.setenv("GLM53_DISPLAY_KV_MIB", "1000")
    try:
        m._pool_bytes()
    except ValueError:
        pass
    else:
        raise AssertionError("non-multiple of 16 must be rejected")


def test_plan_first_fit_decreasing(monkeypatch):
    m = _load(MOD, "glm53_display_kv_plan")

    class FakeOwner:
        size = 1792 << 20
        used = 0

    m._owner = FakeOwner()
    MiB = 1 << 20
    sizes = [118272 * 530] * 11 + [2351104 * 530] * 22
    m.plan(sizes)
    chosen = sorted(m._plan)
    # FFD: one 1188 MiB MLA tensor first, then ten of the 60 MiB indexer
    # tensors (1188 + 10*60 = 1788 <= 1792; the eleventh would not fit).
    assert chosen == list(range(0, 10)) + [11], chosen
    planned = sum(sizes[i] for i in chosen)
    assert planned <= FakeOwner.size
    assert (FakeOwner.size - planned) // MiB <= (min(sizes) // MiB) + 1  # stranded <= smallest tensor


def test_off_mode_never_touches_drm(monkeypatch):
    m = _load(MOD, "glm53_display_kv_off")
    monkeypatch.setenv("GLM53_DISPLAY_KV", "0")
    monkeypatch.setenv("GLM53_DISPLAY_KV_LIB", "/nonexistent.so")
    assert m.credit() == 0
    assert m._owner is None
