#!/usr/bin/env python3
"""Extend the indexer prefill-chunk-metadata warmup to the missing int bucket.

``BuildPrefillChunkMetadataKernel.get_warmup_keys``
(``v1/attention/backends/mla/indexer.py``) enumerates the Triton
specialization space it wants warmed::

    query_slice_start=WarmupIntRange(0, 2),
    query_slice_stop=(1, 2 * max_tokens - 1, 2 * max_tokens),

``WarmupIntRange`` is EXCLUSIVE (``_expand_warmup_values`` ->
``range(start, stop)``), so ``(0, 2)`` yields ``{0, 1}``. Triton 3.7.1
specializes integer args into exactly three buckets -- ``== 1``
(constant-folded), ``% 16 == 0`` ('D'), and everything else ('') -- so
``{0, 1}`` covers 'D' and the folded bucket but never ''. The stop tuple
``(1, 15, 16)`` spans all three, but runtime ``query_slice_start`` values can
land in ''.

``build_prefill_chunk_metadata`` receives ``query_slice`` from
``split_indexer_prefill_chunks``' sub-chunking-on-M path, whose offsets step
by ``max_q = max_logits_elems // chunk_n``. With this deployment's 512 MiB
logits budget (``VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`` default 512 ->
134,217,728 elements), ``index_kpool=4`` and MNBT=7168, sub-chunking first
triggers once a chunk's uncompressed sequence exceeds ~74,897 tokens, and any
non-power-of-two compressed length then yields a ``max_q`` that is neither 1
nor divisible by 16 (e.g. a ~90k-token agent context -> chunk_n=22500 ->
max_q=5965). The first such prefill after a restart pays a cold Triton
load/compile of the ``qs_start=''`` variant, which ``jit_monitor`` reports as
a mid-serve latency spike ("Triton kernel JIT compilation during inference:
BuildPrefillChunkMetadataKernel.kernel").

Observed in the persistent Triton cache (52 entries for this kernel,
2026-08-28..09-02): the ONLY serving-time-created variants are ``qs_start=''``
ones (four entries), while all 24 boot-warmup variants carry boot mtimes -- the
gap is exactly this bucket.

The fix widens the range to ``(0, 3)`` so ``qs_start=2`` -> '' is warmed at
boot. This is a content-only edit INSIDE ``get_warmup_keys``: the
``@triton.jit kernel`` body above it is untouched and no line above it moves,
so the kernel source hash -- and therefore every existing persistent Triton
cache entry -- stays valid. Post-fix warmup covers
{0,1,2} x {1,15,16} x {aligned, misaligned} x CR {1,4} = 36 keys, a superset
of every variant observed in the cache.

Conventions follow ``overlay/patch_kpool_tail_slotmap.py`` (pinned ANCHOR,
MARK sentinel, ``verified_state``, idempotent, atomic replace, pyc clear,
drift => nonzero exit) with ``patch_spinwait.py``'s ``--preflight`` mode
(validate anchors without writing).

Usage::

    python3 patch_indexer_warmup_range.py              # apply
    python3 patch_indexer_warmup_range.py --preflight  # validate anchors only
"""
from __future__ import annotations

import ast
import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_INDEXER_BACKEND_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py",
    )
)

MARK = "            # [glm53-indexer-warmup-range] WarmupIntRange stop is exclusive:\n"

ANCHOR = "            query_slice_start=WarmupIntRange(0, 2),\n"

PATCHED = """            # [glm53-indexer-warmup-range] WarmupIntRange stop is exclusive:
            # (0, 2) expands to {0, 1}, missing the Triton '' int bucket
            # (neither ==1 nor %16==0) for query_slice_start. Runtime
            # sub-chunking offsets (split_indexer_prefill_chunks max_q) hit
            # that bucket on long contexts (seq/4 > ~74,897 tokens under the
            # 512 MiB logits budget), which cold-compiles the variant
            # mid-serve on the first big post-restart prefill. (0, 3) adds
            # qs_start=2 -> ''. Content-only edit below the @triton.jit
            # kernel: the persistent Triton cache stays valid.
            query_slice_start=WarmupIntRange(0, 3),
"""


def warmup_range_calls(text: str) -> set[tuple[int, int]]:
    """All positional ``WarmupIntRange(a, b)`` literals in the module."""
    tree = ast.parse(text)
    out: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "WarmupIntRange"
            and len(node.args) == 2
            and all(
                isinstance(arg, ast.Constant) and isinstance(arg.value, int)
                for arg in node.args
            )
        ):
            out.add((node.args[0].value, node.args[1].value))
    return out


def verified_state(text: str) -> bool:
    return (
        text.count(ANCHOR) == 0
        and text.count(PATCHED) == 1
        and text.count(MARK) == 1
        and text.count("WarmupIntRange(0, 2)") == 0
        and warmup_range_calls(text) == {(0, 3)}
    )


def prepare(source: str) -> tuple[str, str]:
    marker_count = source.count(MARK)
    if marker_count:
        if marker_count != 1 or not verified_state(source):
            raise ValueError(
                "partial/inconsistent indexer warmup-range patch "
                f"(marker={marker_count})"
            )
        return source, "already present"
    n_anchor = source.count(ANCHOR)
    if n_anchor != 1:
        raise ValueError(
            "pinned warmup-range anchor drifted "
            f"(anchor={n_anchor}; expected exactly one "
            "query_slice_start=WarmupIntRange(0, 2))"
        )
    if warmup_range_calls(source) != {(0, 2)}:
        raise ValueError(
            "indexer.py WarmupIntRange literals are not exactly {(0, 2)} — "
            "re-derive the patch"
        )
    patched = source.replace(ANCHOR, PATCHED, 1)
    if not verified_state(patched):
        raise ValueError("indexer warmup-range post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-warmup-range.tmp")
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
    for pyc in cache.glob("indexer*.pyc"):
        pyc.unlink(missing_ok=True)


def main() -> int:
    preflight = "--preflight" in sys.argv[1:]
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"indexer warmup-range preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")
    if preflight:
        print(f"{TARGET.name}: indexer warmup-range preflight OK ({action})")
        return 0
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"{TARGET.name}: indexer warmup-range {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
