#!/usr/bin/env python3
"""Backport vLLM #57477 onto the pinned NVIDIA kpool prefill tail-seed kernel.

The tail cache is an ``as_strided`` view of the indexer allocation. For
GLM-5.3-Flash the indexer page is 38016 B, so ``tail.stride(0)`` is 19008
bf16 elements, not the dense ``2 * kpool * head_dim`` (1024 elements, 2048 B).
``_kpool_tail_seed_kernel`` on pinned vLLM
``487ecf187d3dfe74d2cf6119a92881dba403c219`` still addresses that view as a
dense ``[num_blocks, 2, KPOOL, HEAD_DIM]`` array:

    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM

Every prefill therefore leaves the request's own tail block unseeded and
writes raw bf16 K / gate rows into a different indexer block. The decode
kernel in the same file already takes ``TAIL_BLOCK_ELEMS`` / ``KPOOL_HEAD``.
Only the NVIDIA prefill seed does not. Upstream
https://github.com/vllm-project/vllm/pull/57477 (merged
``db1bfdd4fb0dd7b8226402ee00abc7a987561b7c``) passes
``tail.stride(0)`` and ``tail.stride(1)`` into that seed kernel.

This is not ``overlay/patch_kpool_tail_slotmap.py``. That overlay clamps the
generic paged slot-map in ``vllm/v1/worker/block_table.py`` so a one-block
``KpoolTailSpec`` row cannot be indexed past. It does not edit
``kpool_compress.py`` and does not change seed-kernel strides. This overlay
does not edit ``block_table.py``.

Confirmed missing from the recipe pin on 2026-09-25 by reading
``vllm/models/glm5next/nvidia/ops/kpool_compress.py`` at
``487ecf187d3dfe74d2cf6119a92881dba403c219`` (the base image
``vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce``,
which the published ``:exl3-instanttensor`` tag is built from). That seed
kernel has no ``TAIL_BLOCK_ELEMS`` parameter. No other file under ``overlay/``
rewrites ``kpool_compress.py``. Issue
https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/264

Fail-closed, idempotent, preflights the pinned anchor before writing.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_KPOOL_COMPRESS_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/"
        "models/glm5next/nvidia/ops/kpool_compress.py",
    )
)

# Identity string for start.sh validate_overlay_artifacts. Also the boot log tag.
MARK = "    # [glm53-kpool-tail-seed-stride] vLLM #57477: padded indexer stride.\n"

# bf16 element counts for the GLM-5.3-Flash indexer page (38016 B).
HEAD_DIM = 128
INDEX_KPOOL = 4
GLM53_INDEXER_PAGE_BYTES = 38016
GLM53_TAIL_BLOCK_ELEMS = GLM53_INDEXER_PAGE_BYTES // 2  # stride(0)
GLM53_KPOOL_HEAD = INDEX_KPOOL * HEAD_DIM  # stride(1)

DENSE_BASE = "    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM\n"
FIXED_BASE = "    base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM\n"
FIXED_SCORE = "    tl.store(tail_ptr + base + KPOOL_HEAD + offs, s, mask=m)\n"

# Exact ``_kpool_tail_seed_kernel`` + ``kpool_seed_tail_cache`` from vLLM
# 487ecf187. The decode kernel in the same file already uses TAIL_BLOCK_ELEMS;
# this span is the prefill seed only.
ANCHOR = """@triton.jit
def _kpool_tail_seed_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    \"\"\"Copy token ``i``'s raw K + gate into its request's tail block.

    Token ``i`` is among its request's last KPOOL tokens iff the token KPOOL
    ahead belongs to a different tail block (or is past the batch / padding,
    slot < 0). ``tslot = block * KPOOL + pos % KPOOL``; the destination is
    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.
    \"\"\"
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    if t < 0:
        return
    blk = t // KPOOL  # t >= 0 here, so trunc == floor
    ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
        tl.int64
    )
    # Match the torch semantics exactly: a negative ahead slot floors to a
    # block id that differs from every real block -> token is in the tail.
    # Only divide non-negative slots (Triton int div truncates, torch floors).
    if ahead >= 0 and ahead // KPOOL == blk:
        return
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    base = (blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(tail_ptr + base + KPOOL * HEAD_DIM + offs, s, mask=m)


def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    \"\"\"Seed the paged tail cache from a prefill batch (see the kernel).\"\"\"
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    n = tslot.shape[0]
    if n == 0:
        return
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
"""

PATCHED = """@triton.jit
def _kpool_tail_seed_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    \"\"\"Copy token ``i``'s raw K + gate into its request's tail block.

    Token ``i`` is among its request's last KPOOL tokens iff the token KPOOL
    ahead belongs to a different tail block (or is past the batch / padding,
    slot < 0). ``tslot = block * KPOOL + pos % KPOOL``; the destination is
    ``tail[block, {0:K, 1:score}, pos % KPOOL, :]``.

    The tail cache aliases the indexer cache with the indexer's (padded) block
    stride, so blocks are addressed through ``TAIL_BLOCK_ELEMS`` /
    ``KPOOL_HEAD`` (``tail.stride(0)`` / ``tail.stride(1)``), never as a dense
    ``[num_blocks, 2, KPOOL, HEAD_DIM]`` array.
    \"\"\"
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    if t < 0:
        return
    blk = t // KPOOL  # t >= 0 here, so trunc == floor
    ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
        tl.int64
    )
    # Match the torch semantics exactly: a negative ahead slot floors to a
    # block id that differs from every real block -> token is in the tail.
    # Only divide non-negative slots (Triton int div truncates, torch floors).
    if ahead >= 0 and ahead // KPOOL == blk:
        return
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    # [glm53-kpool-tail-seed-stride] vLLM #57477: padded indexer stride.
    base = blk * TAIL_BLOCK_ELEMS + (t % KPOOL) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(tail_ptr + base + KPOOL_HEAD + offs, s, mask=m)


def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    \"\"\"Seed the paged tail cache from a prefill batch (see the kernel).\"\"\"
    assert tail_kv_cache.dtype == torch.bfloat16
    assert tail_kv_cache.ndim == 4 and tail_kv_cache.shape[1] == 2
    assert tail_kv_cache.stride(3) == 1 and tail_kv_cache.stride(2) == head_dim
    assert key.dtype == torch.bfloat16
    n = tslot.shape[0]
    if n == 0:
        return
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
"""


def dense_k_element(
    block: int, ring: int, *, head_dim: int = HEAD_DIM, kpool: int = INDEX_KPOOL
) -> int:
    """Element offset of K[ring] under the pre-#57477 dense block stride."""
    if block < 0 or ring < 0 or head_dim < 1 or kpool < 1:
        raise ValueError("block, ring, head_dim, kpool must be non-negative")
    return (block * 2 * kpool + ring) * head_dim


def padded_k_element(
    block: int, ring: int, tail_block_elems: int, *, head_dim: int = HEAD_DIM
) -> int:
    """Element offset of K[ring] under ``tail.stride(0) == tail_block_elems``."""
    if block < 0 or ring < 0 or tail_block_elems < 1 or head_dim < 1:
        raise ValueError("block, ring, tail_block_elems, head_dim must be >= 0/1")
    return block * tail_block_elems + ring * head_dim


def apply_seed(
    backing: list[float],
    *,
    block: int,
    ring: int,
    key: list[float],
    score: list[float],
    kpool: int,
    head_dim: int,
    tail_block_elems: int,
    kpool_head: int,
    padded: bool,
) -> None:
    """CPU replica of one token's two stores (K, then gate score)."""
    if len(key) != head_dim or len(score) != head_dim:
        raise ValueError("key and score must be head_dim long")
    if padded:
        k_base = padded_k_element(block, ring, tail_block_elems, head_dim=head_dim)
        s_base = k_base + kpool_head
    else:
        k_base = dense_k_element(block, ring, head_dim=head_dim, kpool=kpool)
        s_base = k_base + kpool * head_dim
    end = s_base + head_dim
    if k_base < 0 or s_base < k_base or end > len(backing):
        raise IndexError(
            f"seed write [{k_base}, {end}) does not fit backing of {len(backing)}"
        )
    backing[k_base : k_base + head_dim] = list(key)
    backing[s_base : s_base + head_dim] = list(score)


def view_row(
    backing: list[float],
    block: int,
    which: int,
    ring: int,
    *,
    tail_block_elems: int,
    kpool_head: int,
    head_dim: int,
) -> list[float]:
    """Read ``tail[block, which, ring, :]`` from a flat element buffer."""
    base = block * tail_block_elems + which * kpool_head + ring * head_dim
    return list(backing[base : base + head_dim])


def seed_kernel_fixed(text: str) -> bool:
    """True when the prefill seed uses the padded stride (marker optional)."""
    return (
        text.count(FIXED_BASE) == 1
        and text.count(FIXED_SCORE) == 1
        and DENSE_BASE not in text
        and "assert tail_kv_cache.ndim == 4 and tail_kv_cache.shape[1] == 2\n" in text
        and (
            "assert tail_kv_cache.stride(3) == 1 "
            "and tail_kv_cache.stride(2) == head_dim\n"
        )
        in text
    )


def verified_state(text: str) -> bool:
    return (
        text.count(ANCHOR) == 0
        and text.count(PATCHED) == 1
        and text.count(MARK) == 1
        and seed_kernel_fixed(text)
    )


def prepare(source: str) -> tuple[str, str]:
    marker_count = source.count(MARK)
    if marker_count:
        if marker_count != 1 or not verified_state(source):
            raise ValueError(
                "partial/inconsistent kpool tail seed-stride patch "
                f"(marker={marker_count})"
            )
        return source, "already present"
    if verified_state(source):
        return source, "already patched"
    if seed_kernel_fixed(source):
        return source, "already upstream"
    n_anchor = source.count(ANCHOR)
    if n_anchor != 1:
        raise ValueError(
            "pinned kpool_compress seed-kernel anchor drifted "
            f"(anchor={n_anchor}; expected vLLM 487ecf187). "
            "This backport is vLLM #57477 (padded tail stride), not the "
            "block_table slot-map clamp."
        )
    patched = source.replace(ANCHOR, PATCHED, 1)
    if not verified_state(patched):
        raise ValueError("kpool tail seed-stride post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kpool-tail-seed.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob("kpool_compress*.pyc"):
        pyc.unlink(missing_ok=True)


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(
            f"kpool tail seed-stride preflight failed: {exc}"
        ) from exc
    compile(patched, str(TARGET), "exec")
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"[glm53-kpool-tail-seed-stride] {TARGET.name}: {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
