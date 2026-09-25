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
    apply_seed,
    dense_k_element,
    padded_k_element,
    prepare,
    seed_kernel_fixed,
    verified_state,
    view_row,
)
from patch_kpool_tail_slotmap import (  # noqa: E402
    MARK as SLOT_MARK,
    TARGET as SLOT_TARGET,
)
from patch_kpool_tail_seed_stride import TARGET as SEED_TARGET  # noqa: E402

def _pin_fixture() -> Path:
    name = "kpool_tail_seed_kernel-487ecf187.py.txt"
    for candidate in (
        ROOT / "tests" / "fixtures" / name,
        HERE / "fixtures" / name,
        Path("/opt/glm53/fixtures") / name,
    ):
        if candidate.is_file():
            return candidate
    raise AssertionError(f"pinned seed-kernel fixture {name} missing")
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
        return
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "kpool_compress.py"
        target.write_text(src.read_text())
        result = _run_patch(target)
        assert result.returncode == 0, result.stderr
        text = target.read_text()
        assert seed_kernel_fixed(text)
        assert DENSE_BASE not in text


def test_recipe_wiring_if_present() -> None:
    start = ROOT / "start.sh"
    dockerfile = ROOT / "Dockerfile"
    if not start.is_file() or not dockerfile.is_file():
        return
    launcher = start.read_text()
    image = dockerfile.read_text()
    readme = (ROOT / "README.md").read_text()
    assert 'KPOOL_SEED_PATCH_HOST="${KPOOL_SEED_PATCH_HOST:-' in launcher
    order = launcher[
        launcher.index("GLM53_OVERLAY_ORDER=(") : launcher.index(
            ")", launcher.index("GLM53_OVERLAY_ORDER=(")
        )
    ]
    assert "\n    patch_kpool_tail_seed_stride.py\n" in order
    assert order.index("patch_kpool_tail_slotmap.py") < order.index(
        "patch_kpool_tail_seed_stride.py"
    )
    assert (
        "-v '/tmp/patch_kpool_tail_seed_stride.py:"
        "/opt/glm53/patch_kpool_tail_seed_stride.py:ro'" in launcher
    )
    assert (
        '-v "$KPOOL_SEED_PATCH_HOST:'
        '/opt/glm53/patch_kpool_tail_seed_stride.py:ro"' in launcher
    )
    assert 'scp -q -o BatchMode=yes "$KPOOL_SEED_PATCH_HOST"' in launcher
    assert "[glm53-kpool-tail-seed-stride]" in launcher
    assert "COPY overlay/patch_kpool_tail_seed_stride.py" in image
    assert "RUN python3 /opt/glm53/patch_kpool_tail_seed_stride.py" in image
    assert "python3 /opt/glm53/test_kpool_tail_seed_stride.py" in image
    assert "vLLM #57477" in readme
    assert "patch_kpool_tail_seed_stride.py" in readme
    for name in ("start-tp3.sh", "start-tp4.sh"):
        text = (ROOT / name).read_text()
        assert "patch_kpool_tail_seed_stride.py" in text
        assert 'KPOOL_SEED_PATCH_HOST="${KPOOL_SEED_PATCH_HOST:-' in text


def main() -> int:
    test_distinct_from_slotmap()
    test_anchor_is_pinned_487ecf187()
    test_upstream_padded_view_contract()
    test_published_byte_windows()
    test_fixture_apply_idempotent()
    test_already_upstream_without_marker()
    test_fail_closed()
    test_installed_copy_if_present()
    test_recipe_wiring_if_present()
    print("kpool tail seed-stride patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
