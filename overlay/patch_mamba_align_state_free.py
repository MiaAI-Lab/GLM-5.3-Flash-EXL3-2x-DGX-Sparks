#!/usr/bin/env python3
"""Release every superseded Mamba "align" state block; reserve for in-flight ones.

``MambaManager`` (mamba_cache_mode="align") keeps one running-state block per
request. ``remove_skipped_blocks`` releases the block that held the previous
running state once the processed prefix (``num_computed_tokens -
num_in_flight_tokens``, kv_cache_manager.allocate_slots) has moved past it,
because the step that copied it into its successor may still be in flight.
It remembers that block in a single ``last_state_block_idx`` slot, and
``allocate_new_blocks`` overwrites the slot whenever a chunk needs a new state
block. Under async scheduling the processed prefix lags one batch, so when
every chunk crosses a Mamba block (a 7168-token budget against 3584-token
blocks) the slot is overwritten before its block becomes releasable and the
block stays referenced in ``req_to_blocks`` until the request finishes. The
base ``_remove_blocks_in_range`` cannot reclaim it: it walks backwards and
stops at the first null block, and align mode is sparse.

Manager: ``stale_state_block_idxs`` replaces the slot with the per-request
list of superseded state block indices still awaiting release (ascending,
one per unsettled step). Each entry is released under the unchanged
predicate, so no state an in-flight copy may read is freed. The list is
consulted through an allocation-free head check and compacted in place; it
is dropped with the request's blocks (``pop_blocks_for_free``: finish and
preemption).

Spec: ``MambaSpec.max_memory_usage_bytes`` reserved ``2 + num_speculative_blocks``
pages per request in align mode: the running state, its predecessor and the
speculative blocks. Superseded blocks are held for one batch per concurrent
step, so the resident maximum is ``1 + max_concurrent_batches +
num_speculative_blocks`` (``VllmConfig.max_concurrent_batches``: 1 sync, 2
async, PP-size with pipeline parallelism). Modes "all" and "none" are
unchanged.

Usage:
    python3 patch_mamba_align_state_free.py

Idempotent; a partially applied or drifted source fails before writing either
file. Both files are preflighted before either is written.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
STM = Path(
    os.environ.get(
        "GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY",
        f"{_VLLM}/v1/core/single_type_kv_cache_manager.py",
    )
)
KVI = Path(
    os.environ.get("GLM53_KV_CACHE_INTERFACE_PY", f"{_VLLM}/v1/kv_cache_interface.py")
)
MARK = "# [glm53-mamba-align-state-free-v1]"

INIT_OLD = """        if self.mamba_cache_mode == "align":
            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}
"""
INIT_NEW = """        if self.mamba_cache_mode == "align":
            # [glm53-mamba-align-state-free-v1] Per request, the ascending
            # indices of state blocks superseded by a newer state block and
            # not yet released; remove_skipped_blocks frees each once the
            # processed prefix has moved past it. One entry per unsettled
            # step.
            self.stale_state_block_idxs: defaultdict[str, list[int]] = defaultdict(
                list
            )
"""

REMOVE_OLD = """        if self.mamba_cache_mode == "align":
            # `last_state_block_idx` refers to the block index allocated two steps ago.
            # The block allocated in the previous step is used to copy Mamba states
            # into the block allocated in the current step; the earlier block is
            # no longer needed and should be freed here.
            last_state_block_idx = self.last_state_block_idx.get(request_id)
            # Blocks allocated during prefill may be non-contiguous. Use
            # `last_state_block_idx` to free the appropriate block and replace it
            # with a null block.
            if (
                last_state_block_idx is not None
                and last_state_block_idx
                < cdiv(processed_computed_tokens, self.block_size) - 1
            ):
                blocks = self.req_to_blocks[request_id]
                if blocks[last_state_block_idx] != self._null_block:
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block
"""
REMOVE_NEW = """        if self.mamba_cache_mode == "align":
            # [glm53-mamba-align-state-free-v1] A superseded state block is
            # read by the step that copied it into its successor; release it
            # once the processed prefix (committed, not in flight) has moved
            # past the block that held its state. Blocks allocated during
            # prefill may be non-contiguous, so free by recorded index and
            # replace with a null block. Entries are ascending: stop at the
            # first one still needed.
            stale = self.stale_state_block_idxs.get(request_id)
            if stale and stale[0] < (
                releasable_below := cdiv(processed_computed_tokens, self.block_size) - 1
            ):
                blocks = self.req_to_blocks[request_id]
                freed: list[KVCacheBlock] = []
                while stale and stale[0] < releasable_below:
                    block_idx = stale.pop(0)
                    if blocks[block_idx] != self._null_block:
                        freed.append(blocks[block_idx])
                        blocks[block_idx] = self._null_block
                if freed:
                    self.block_pool.free_blocks(freed)
"""

ALLOC_OLD = """                # Record the last state block
                if blocks_allocated:
                    # We always save the running state at the last
                    # (1 + num_speculative_blocks) block
                    self.last_state_block_idx[request_id] = (
                        prev_block_len - 1 - self.num_speculative_blocks
                    )
                elif prev_block_len > 0:
                    # When a new request hits the prefix cache, the last block
                    # saves the hit state.
                    self.last_state_block_idx[request_id] = prev_block_len - 1
"""
ALLOC_NEW = """                # Record the state block this allocation supersedes
                # [glm53-mamba-align-state-free-v1]
                if blocks_allocated:
                    # We always save the running state at the last
                    # (1 + num_speculative_blocks) block
                    self.stale_state_block_idxs[request_id].append(
                        prev_block_len - 1 - self.num_speculative_blocks
                    )
                elif prev_block_len > 0:
                    # When a new request hits the prefix cache, the last block
                    # saves the hit state.
                    self.stale_state_block_idxs[request_id].append(prev_block_len - 1)
"""

POP_OLD = """            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
"""
POP_NEW = """            self._allocated_block_reqs.discard(request_id)
            self.stale_state_block_idxs.pop(request_id, None)
"""

SPEC_OLD = """        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
"""
SPEC_NEW = """        elif vllm_config.cache_config.mamba_cache_mode == "align":
            # [glm53-mamba-align-state-free-v1] The running state, its
            # speculative blocks, and one superseded state block per
            # concurrent batch: a superseded block is released only once the
            # processed prefix (committed, not in flight) has passed it.
            return self.page_size_bytes * (
                1 + vllm_config.max_concurrent_batches + self.num_speculative_blocks
            )
"""
ROW_OLD = """            # Block table rows are position-indexed over the full sequence
            # even though only 2 + num_speculative_blocks state blocks are
            # resident at a time (earlier states are nulled out by
            # remove_skipped_blocks), so the row length must cover max_len
            # rather than max_memory_usage_bytes.
"""
ROW_NEW = """            # Block table rows are position-indexed over the full sequence
            # even though only 1 + max_concurrent_batches +
            # num_speculative_blocks state blocks are resident at a time
            # (earlier states are nulled out by remove_skipped_blocks), so
            # the row length must cover max_len rather than
            # max_memory_usage_bytes.
"""

MANAGER_EDITS = (
    ("align-init", INIT_OLD, INIT_NEW),
    ("align-remove-skipped", REMOVE_OLD, REMOVE_NEW),
    ("align-allocate", ALLOC_OLD, ALLOC_NEW),
    ("align-pop", POP_OLD, POP_NEW),
)
SPEC_EDITS = (
    ("align-reservation", SPEC_OLD, SPEC_NEW),
    ("align-row-comment", ROW_OLD, ROW_NEW),
)


def replace_once(path: Path, text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def verify_complete(path: Path, text: str, edits, stale_names=()) -> list[str]:
    problems = [
        f"{label}: expected exactly one applied block, found {n}"
        for label, _, new in edits
        if (n := text.count(new)) != 1
    ]
    # An applied block may extend its anchor; judge the superseded forms on
    # the text with every applied block removed.
    stripped = text
    for _, _, new in edits:
        stripped = stripped.replace(new, "")
    problems += [
        f"{label}: superseded form still present" for label, old, _ in edits if old in stripped
    ]
    problems += [f"{name}: stale reference remains" for name in stale_names if name in text]
    try:
        compile(text, str(path), "exec")
    except SyntaxError as exc:
        problems.append(f"syntax: {exc.msg}")
    return problems


def prepare(path: Path, edits, stale_names=()) -> tuple[str, str]:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    original = path.read_text()
    text = original
    if MARK not in text:
        for label, old, new in edits:
            text = replace_once(path, text, old, new, label)
    if problems := verify_complete(path, text, edits, stale_names):
        raise SystemExit(f"{path}: incomplete or drifted overlay state: " + "; ".join(problems))
    return original, text


def main() -> int:
    # Preflight both files before writing either.
    updates = (
        (STM, *prepare(STM, MANAGER_EDITS, ("last_state_block_idx",))),
        (KVI, *prepare(KVI, SPEC_EDITS)),
    )
    for path, original, text in updates:
        if text != original:
            path.write_text(text)
        print(f"patched {path.name} ({MARK})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
