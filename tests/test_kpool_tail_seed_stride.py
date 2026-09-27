#!/usr/bin/env python3
"""CPU checks for the vLLM #57477 kpool tail-seed stride backport.

The GPU kernel test upstream runs
(``tests/kernels/test_kpool_decode_update_batched.py::
test_prefill_seed_honors_padded_tail_block_stride``) needs Triton and a GPU.
This file ports that test's layout contract, plus the byte windows published
in vLLM PR #57477, onto the pure-Python replica in the overlay.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_kpool_tail_seed_stride.py",
        ROOT / "overlay" / "patch_kpool_tail_seed_stride.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_kpool_tail_seed_stride import (  # noqa: E402
    ANCHOR,
    DENSE_BASE,
    FIXED_BASE,
    GLM53_INDEXER_PAGE_BYTES,
    GLM53_KPOOL_HEAD,
    GLM53_TAIL_BLOCK_ELEMS,
    HEAD_DIM,
    INDEX_KPOOL,
    MARK,
    PATCHED,
    RECIPE_INDEXER_PAGE_BYTES,
    RECIPE_TAIL_BLOCK_ELEMS,
    apply_seed,
    dense_k_element,
    padded_k_element,
    prepare,
    seed_kernel_fixed,
    seed_launch_passes_strides,
    verified_state,
    view_row,
)
from patch_kpool_tail_slotmap import (  # noqa: E402
    MARK as SLOT_MARK,
    TARGET as SLOT_TARGET,
)
from patch_kpool_tail_seed_stride import TARGET as SEED_TARGET  # noqa: E402

PIN_FIXTURE = "kpool_tail_seed_kernel-487ecf187.py.txt"
# The seed kernel + launcher from vLLM db1bfdd4fb0d (the #57477 merge), for the
# genuine already-upstream positive control.
UPSTREAM_FIXTURE = "kpool_tail_seed_kernel-db1bfdd.py.txt"


def _skip(reason: str) -> None:
    """Skip optional installed-source or Triton checks unless explicitly required."""
    if os.environ.get("GLM53_REQUIRE_KERNEL_TESTS") == "1":
        raise AssertionError(f"required test could not run: {reason}")
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(reason)
    print(f"skip {reason}")


def _pin_fixture(name: str = PIN_FIXTURE) -> Path:
    for candidate in (
        ROOT / "tests" / "fixtures" / name,
        HERE / "fixtures" / name,
        Path("/opt/glm53/fixtures") / name,
    ):
        if candidate.is_file():
            return candidate
    raise AssertionError(f"seed-kernel fixture {name} missing")
INSTALLED = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "models/glm5next/nvidia/ops/kpool_compress.py"
)

HEADER = (
    "import torch\n"
    "import triton\n"
    "import triton.language as tl\n"
    "INDEX_HEAD_DIM = 128\n\n"
)


def _module(body: str) -> str:
    return HEADER + body


def _run_patch(target: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_KPOOL_COMPRESS_PY"] = str(target)
    return subprocess.run(
        [sys.executable, str(PATCH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_distinct_from_slotmap() -> None:
    """Slot-map clamp and seed-stride backport edit different files."""
    assert SLOT_TARGET.name == "block_table.py"
    assert SEED_TARGET.name == "kpool_compress.py"
    assert "kpool-tail-slotmap" in SLOT_MARK
    assert "kpool-tail-seed-stride" in MARK
    assert SLOT_MARK not in PATCHED
    assert MARK not in ANCHOR
    assert "block_table" not in str(SEED_TARGET)


def test_anchor_is_pinned_487ecf187() -> None:
    pinned = _pin_fixture().read_text()
    assert ANCHOR == pinned
    assert "TAIL_BLOCK_ELEMS" not in pinned
    assert DENSE_BASE in pinned
    assert "def _kpool_tail_seed_kernel" in pinned


def test_upstream_padded_view_contract() -> None:
    """Port of test_prefill_seed_honors_padded_tail_block_stride (no GPU)."""
    kpool = 4
    num_blocks = 6
    logical_block_elems = 2 * kpool * HEAD_DIM
    padded_block_elems = logical_block_elems + 256
    kpool_head = kpool * HEAD_DIM
    sentinel = -123.0
    block = 3
    ring = 2
    key = [float(i) for i in range(HEAD_DIM)]
    score = [v + 256.0 for v in key]

    backing = [sentinel] * (num_blocks * padded_block_elems)
    apply_seed(
        backing,
        block=block,
        ring=ring,
        key=key,
        score=score,
        kpool=kpool,
        head_dim=HEAD_DIM,
        tail_block_elems=padded_block_elems,
        kpool_head=kpool_head,
        padded=True,
    )
    assert view_row(
        backing, block, 0, ring,
        tail_block_elems=padded_block_elems,
        kpool_head=kpool_head,
        head_dim=HEAD_DIM,
    ) == key
    assert view_row(
        backing, block, 1, ring,
        tail_block_elems=padded_block_elems,
        kpool_head=kpool_head,
        head_dim=HEAD_DIM,
    ) == score
    compact = dense_k_element(block, ring, head_dim=HEAD_DIM, kpool=kpool)
    assert all(v == sentinel for v in backing[compact : compact + HEAD_DIM])

    dense = [sentinel] * (num_blocks * padded_block_elems)
    apply_seed(
        dense,
        block=block,
        ring=ring,
        key=key,
        score=score,
        kpool=kpool,
        head_dim=HEAD_DIM,
        tail_block_elems=padded_block_elems,
        kpool_head=kpool_head,
        padded=False,
    )
    assert view_row(
        dense, block, 0, ring,
        tail_block_elems=padded_block_elems,
        kpool_head=kpool_head,
        head_dim=HEAD_DIM,
    ) != key
    assert any(v != sentinel for v in dense[compact : compact + HEAD_DIM])


def test_published_byte_windows() -> None:
    """Kernel-level windows from vLLM PR #57477 (full tail block, ring 0)."""
    assert GLM53_TAIL_BLOCK_ELEMS == 19008
    assert GLM53_KPOOL_HEAD == 512
    assert GLM53_INDEXER_PAGE_BYTES == 38016
    page = GLM53_INDEXER_PAGE_BYTES
    dense_block = 2 * INDEX_KPOOL * HEAD_DIM * 2  # bytes
    assert dense_block == 2048

    def span(block: int, padded: bool) -> tuple[int, int]:
        if padded:
            start = padded_k_element(block, 0, GLM53_TAIL_BLOCK_ELEMS) * 2
        else:
            start = dense_k_element(block, 0) * 2
        return start, start + dense_block

    assert span(200, padded=False) == (409600, 411648)
    assert span(18, padded=False) == (36864, 38912)
    assert span(200, padded=True) == (7603200, 7605248)
    assert span(18, padded=True) == (684288, 686336)
    # Dense write for block 200 lands in indexer block 10, not block 200.
    dense_start, _dense_end = span(200, padded=False)
    assert dense_start // page == 10
    own_start, own_end = span(200, padded=True)
    assert own_start // page == 200
    assert dense_start < own_start
    assert not (own_start <= dense_start < own_end)


def test_recipe_block_3584_windows() -> None:
    """This recipe's geometry, as measured on a 2x GB10 kit (block 3584, 552
    usable block ids): every dense tail write lands in indexer blocks 0-9."""
    assert RECIPE_INDEXER_PAGE_BYTES == 118272 and RECIPE_TAIL_BLOCK_ELEMS == 59136
    page = RECIPE_INDEXER_PAGE_BYTES
    victims = {dense_k_element(b, 0) * 2 // page for b in range(552)}
    assert victims == set(range(10)), sorted(victims)
    # b=58 lands inside block 1's own tail slice (first 2048 B of its page).
    assert (dense_k_element(58, 0) * 2) // page == 1
    assert (dense_k_element(58, 0) * 2) % page < 2048
    for b in (1, 58, 300, 551):
        own = padded_k_element(b, 0, RECIPE_TAIL_BLOCK_ELEMS) * 2
        assert own // page == b and own % page == 0


def _triton_interpreter_available() -> bool:
    try:
        import importlib.util as u
        return u.find_spec("torch") is not None and u.find_spec("triton") is not None
    except Exception:
        return False


KERNEL_RUN = r"""
import os, sys, json
os.environ["TRITON_INTERPRET"] = "1"
import torch, triton, triton.language as tl
INDEX_HEAD_DIM = 128
src, stride0, patched = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
g = {"torch": torch, "triton": triton, "tl": tl, "INDEX_HEAD_DIM": INDEX_HEAD_DIM}
exec(open(src).read(), g)
seed = g["kpool_seed_tail_cache"]
KPOOL, HD, NB, SENT = 4, 128, 12, -7.0
res = []
for b in (1, 3, 11):
    storage = torch.full((NB * stride0,), SENT, dtype=torch.bfloat16)
    tail = storage.as_strided((NB, 2, KPOOL, HD), (stride0, KPOOL * HD, HD, 1))
    key = (torch.arange(KPOOL * HD, dtype=torch.float32).reshape(KPOOL, HD) % 97 + 1).to(torch.bfloat16)
    score = (-(torch.arange(KPOOL * HD, dtype=torch.float32).reshape(KPOOL, HD) % 89) - 1).to(torch.bfloat16)
    tslot = torch.tensor([b * KPOOL + p for p in range(KPOOL)], dtype=torch.int64)
    seed(tail, key, score, tslot, KPOOL, HD)
    own = bool(torch.equal(tail[b, 0].float(), key.float()) and torch.equal(tail[b, 1].float(), score.float()))
    view = torch.zeros_like(storage, dtype=torch.bool)
    view.as_strided((NB, 2, KPOOL, HD), (stride0, KPOOL * HD, HD, 1))[b] = True
    stray = int(((storage.float() != SENT) & ~view).sum())
    res.append({"b": b, "own_seeded": own, "stray_elems": stray})
print(json.dumps(res))
"""


def test_patched_kernel_under_triton_interpreter() -> None:
    """Run the overlay's real patched Triton seed kernel (and the pinned
    original as a control) on CPU under TRITON_INTERPRET=1, at both upstream's
    and this recipe's stride. Needs torch and triton (the image has both);
    skipped on hosts without them."""
    if not _triton_interpreter_available():
        _skip("test_patched_kernel_under_triton_interpreter (no torch/triton)")
        return
    import json
    import subprocess
    import tempfile

    for label, text in (("patched", PATCHED), ("pinned", ANCHOR)):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / f"seed_{label}.py"
            src.write_text(text)
            for stride0 in (GLM53_TAIL_BLOCK_ELEMS, RECIPE_TAIL_BLOCK_ELEMS):
                env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
                out = subprocess.run(
                    [sys.executable, "-c", KERNEL_RUN, str(src), str(stride0), "1"],
                    env=env, capture_output=True, text=True, timeout=600,
                )
                assert out.returncode == 0, out.stderr[-2000:]
                rows = json.loads(out.stdout.strip().splitlines()[-1])
                if label == "patched":
                    assert all(r["own_seeded"] and r["stray_elems"] == 0 for r in rows), (stride0, rows)
                else:
                    assert all(not r["own_seeded"] and r["stray_elems"] > 0 for r in rows), (stride0, rows)


def test_fixture_apply_idempotent() -> None:
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "kpool_compress.py"
        target.write_text(_module(ANCHOR))
        first = _run_patch(target)
        assert first.returncode == 0, first.stderr
        assert "[glm53-kpool-tail-seed-stride]" in first.stdout
        assert "patched" in first.stdout
        text = target.read_text()
        assert verified_state(text)
        assert MARK in text
        assert FIXED_BASE in text
        assert DENSE_BASE not in text
        second = _run_patch(target)
        assert second.returncode == 0, second.stderr
        assert "already present" in second.stdout
        again, action = prepare(text)
        assert action == "already present"
        assert again == text


def test_already_upstream_without_marker() -> None:
    upstream = _module(PATCHED).replace(MARK, "")
    assert seed_kernel_fixed(upstream)
    again, action = prepare(upstream)
    assert action == "already upstream"
    assert again == upstream


def test_genuine_upstream_fixture_is_already_upstream() -> None:
    upstream = _module(_pin_fixture(UPSTREAM_FIXTURE).read_text())
    assert MARK not in upstream
    assert seed_launch_passes_strides(upstream)
    again, action = prepare(upstream)
    assert action == "already upstream"
    assert again == upstream


LAUNCH_STRIDES = (
    "        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),\n"
    "        KPOOL_HEAD=tail_kv_cache.stride(1),\n"
)


def test_half_fixed_upstream_is_rejected() -> None:
    """An unmarked file whose kernel body looks fixed but whose launch does
    not pass both real strides must not be reported "already upstream"."""
    upstream = _module(_pin_fixture(UPSTREAM_FIXTURE).read_text())
    assert upstream.count(LAUNCH_STRIDES) == 1
    broken = {
        "strides dropped": upstream.replace(LAUNCH_STRIDES, ""),
        "dense block stride": upstream.replace(
            "TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0)",
            "TAIL_BLOCK_ELEMS=2 * kpool * head_dim",
        ),
        "zero plane stride": upstream.replace(
            "KPOOL_HEAD=tail_kv_cache.stride(1)", "KPOOL_HEAD=0"
        ),
        "swapped dims": upstream.replace(
            LAUNCH_STRIDES,
            "        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(1),\n"
            "        KPOOL_HEAD=tail_kv_cache.stride(0),\n",
        ),
        "strides of another tensor": upstream.replace(
            "KPOOL_HEAD=tail_kv_cache.stride(1)", "KPOOL_HEAD=key.stride(1)"
        ),
        "both strides of a non-tail tensor": upstream.replace(
            LAUNCH_STRIDES,
            "        TAIL_BLOCK_ELEMS=key.stride(0),\n"
            "        KPOOL_HEAD=key.stride(1),\n",
        ),
    }
    for label, text in broken.items():
        assert text != upstream, label
        assert not seed_launch_passes_strides(text), label
        assert not seed_kernel_fixed(text), label
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "kpool_compress.py"
            target.write_text(text)
            result = _run_patch(target)
            assert result.returncode != 0, (label, result.stdout)
            assert "preflight failed" in result.stderr, label
            assert target.read_text() == text, label


def test_fail_closed() -> None:
    drifted = _module(ANCHOR).replace(
        "base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM",
        "base = (blk * KPOOL + t % KPOOL) * HEAD_DIM",
        1,
    )
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "kpool_compress.py"
        target.write_text(drifted)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "preflight failed" in result.stderr
        assert "anchor drifted" in result.stderr
        assert target.read_text() == drifted

    partial = _module(ANCHOR).replace(DENSE_BASE, MARK + DENSE_BASE, 1)
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "kpool_compress.py"
        target.write_text(partial)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "partial/inconsistent" in result.stderr


def test_installed_copy_if_present() -> None:
    src = Path(os.environ.get("GLM53_KPOOL_COMPRESS_PY_SRC", INSTALLED))
    if not src.is_file():
        _skip(f"test_installed_copy_if_present (no installed {src})")
        return
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "kpool_compress.py"
        target.write_text(src.read_text())
        result = _run_patch(target)
        assert result.returncode == 0, result.stderr
        text = target.read_text()
        assert seed_kernel_fixed(text)
        assert DENSE_BASE not in text


def main() -> int:
    test_distinct_from_slotmap()
    test_anchor_is_pinned_487ecf187()
    test_upstream_padded_view_contract()
    test_published_byte_windows()
    test_recipe_block_3584_windows()
    test_patched_kernel_under_triton_interpreter()
    test_fixture_apply_idempotent()
    test_already_upstream_without_marker()
    test_genuine_upstream_fixture_is_already_upstream()
    test_half_fixed_upstream_is_rejected()
    test_fail_closed()
    test_installed_copy_if_present()
    print("kpool tail seed-stride patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
