#!/usr/bin/env python3
"""Keep hybrid prefix-cache hits when the DFlash2 group would zero them.

Issue #7: OpenClaw follow-ups looked like 0% APC. On this kit the MLA +
mamba groups already hit at the 3584-token hybrid align (README 3584/7760).
Two coordinator bugs then throw the extra block away:

1. ``dflash`` is ``use_eagle()``. GLM never sets ``is_eagle_group`` (that
   annotator is DeepseekV4-only), so HybridKVCacheCoordinator flags EVERY
   group. MLA drops its last scheduler-aligned block (~3584 tokens).
2. The DFlash2 SlidingWindow group still participates in the hybrid min.
   After an EAGLE one-block pop it re-aligns down by a full 3584-token
   scheduler page (block=64, align=3584), which can wipe a longer MLA hit.

KpoolTail already opts out of prefix caching (1-block circular scratch).
Mamba align-mode state materializes only at actual chunk ends. Mamba must
stay in the min: skipping a missing checkpoint is a correctness hole.

This patch: flag only exact SlidingWindowSpec groups as EAGLE, and do not
let that drafter group shrink ``curr_hit_length``. If the drafter window
does not cover the MLA/mamba hit, leave its blocks empty so a fresh
window is allocated (zeros / new pages).

KpoolTail is deliberately per-request scratch and cannot prefix-cache. A
resumed request therefore starts with an empty indexer-tail ring. Replaying
fewer than one complete kpool leaves the ring incomplete and changes sparse
attention selection. Bound the candidate independently of retained drafter
windows so a complete pool is rebuilt before decode. Only participating
groups may veto fine-grained hits. A fine target tail without a draft window
retries the preceding shared checkpoint before backing off a full replay window.

``GLM53_DRAFT_KV_COMPACT=1`` adds a DFlash-specific boundary lookup. The
EAGLE hit rule needs one complete, cached block AFTER the reconciled
boundary (then pops it) because an EAGLE draft KV at position p embeds
token p+1. DFlash context KV at position p is a per-position projection of
the target hidden state at p (``precompute_and_store_context_kv``: row-wise
RMSNorm, fused KV GEMM, K-norm, RoPE at p), so no such block exists in its
dataflow, and with a compact draft block B it cannot exist for any prompt
whose tail past the boundary is shorter than B. The drafter group is then
looked up ending exactly at the boundary (``drop_eagle_block=False``), and
its manager is kept non-EAGLE so the lookahead block is neither hashed
(``cache_blocks``) nor reserved by ``reachable_block_mask``: the retained
tail is exactly the ``cdiv(window - 1, B)`` blocks the lookup consults, one
pool block id fewer per window than the EAGLE form. The complete-window
check, replay clamp and EAGLE flag are unchanged. Default ``0`` keeps the
EAGLE lookahead lookup and retention.

Installation states: pristine source; the legacy hybrid-apc form shipped in
the stock image (with or without the replay stage), which is migrated to the
v3 verification form first; retained compact-boundary source; already-current
source, a byte-identical no-op.
Before writing, every owned stage must be present exactly once in its
supported form; a stage marker alone never counts as an installation.
Fail closed if the vLLM coordinator anchors drift.
"""
from __future__ import annotations

import ast
import dis
import os
import sys
from pathlib import Path
from types import CodeType

P = Path(
    os.environ.get(
        "GLM53_KV_COORDINATOR_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py",
    )
)
MARK = "# [glm53-hybrid-apc]"
DFLASH_REPLAY_MARK = "# [glm53-dflash-swa-replay-v1]"
EAGLE_VERIFY_MARK = "# [glm53-dflash-eagle-verify-v3]"
DFLASH_BOUNDARY_MARK = "# [glm53-dflash-boundary-lookup-v1]"

BASE_HELPER = '''
def _glm53_inner_kv_spec(spec):
    specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(specs, dict) and specs:
        return next(iter(specs.values()))
    return spec


def _glm53_is_draft_swa_spec(spec) -> bool:
    """DFlash2 drafter: exact SlidingWindowSpec, not KpoolTailSpec."""
    return type(_glm53_inner_kv_spec(spec)).__name__ == "SlidingWindowSpec"


'''

DFLASH_REPLAY_HELPER = '''
def _glm53_dflash_swa_replay_tokens(kv_cache_groups) -> int:
    """Fresh tokens required to rebuild the DFlash sliding-attention window."""
    replay = 0
    for group in kv_cache_groups:
        spec = _glm53_inner_kv_spec(group.kv_cache_spec)
        if type(spec).__name__ == "SlidingWindowSpec":
            replay = max(replay, int(getattr(spec, "sliding_window", 0) or 0))
    return replay


def _glm53_dflash_replay_safe_hit(
    hit_length: int,
    max_cache_hit_length: int,
    replay_tokens: int,
    alignment_tokens: int,
) -> int:
    """Move an APC hit back until the fresh suffix rebuilds DFlash SWA."""
    if hit_length <= 0 or replay_tokens <= 0:
        return hit_length
    # max_cache_hit_length is prompt_tokens - 1 because the final prompt token
    # is always recomputed for logits. Include that token in the fresh replay.
    fresh_tokens = max_cache_hit_length + 1 - hit_length
    if fresh_tokens >= replay_tokens:
        return hit_length
    deficit = replay_tokens - fresh_tokens
    pages = (deficit + alignment_tokens - 1) // alignment_tokens
    return max(0, hit_length - pages * alignment_tokens)


'''

EAGLE_OLD = """        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))
"""

EAGLE_NEW = """        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            # [glm53-hybrid-apc] dflash is use_eagle(); GLM has no is_eagle_group
            # annotator. Flag only the drafter SlidingWindowSpec group so MLA /
            # mamba do not drop a whole scheduler page (~3584). MTP with no SWA
            # group keeps the upstream all-groups fallback.
            swa_ids = {
                i
                for i, g in enumerate(kv_cache_config.kv_cache_groups)
                if _glm53_is_draft_swa_spec(g.kv_cache_spec)
            }
            self.eagle_group_ids = swa_ids or set(
                range(len(kv_cache_config.kv_cache_groups))
            )
"""

MIN_OLD = """                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)
"""

MIN_NEW = """                _glm53_draft_swa = _glm53_is_draft_swa_spec(spec)
                if drop_eagle_block:
                    # [glm53-dflash-eagle-verify-v3]
                    # A failed DFlash lookup is an attempted EAGLE pop, not a
                    # verified one. The convergence loop may run again after a
                    # different group shortens the initial candidate; carrying
                    # a failed verification into that pass would suppress the
                    # pop and mistake an ordinary 32-block tail for valid
                    # reconciled-boundary state.
                    if _glm53_draft_swa and _new_hit_length < curr_hit_length:
                        eagle_verified.discard(idx)
                    else:
                        eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                if _glm53_draft_swa:  # [glm53-hybrid-apc]
                    # Drafter SWA must not min() the hybrid hit. Its EAGLE pop
                    # re-aligns by LCM(window block, MLA page) = 3584. If the
                    # cached window does not cover the MLA/mamba hit, leave
                    # blocks empty so a fresh window is allocated; do not
                    # reseed the indexer tail here (KpoolTail already opted out).
                    if _new_hit_length >= curr_hit_length:
                        for group_id, blocks in zip(group_ids, hit_blocks):
                            hit_blocks_by_group[group_id] = blocks
                            hit_length_by_group[group_id] = _new_hit_length
                    continue
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)
"""

# The first shipped hybrid-min form (stock image): the drafter branch guards
# on _glm53_is_draft_swa_spec(spec) directly and every attempted EAGLE
# lookup counts as verified. Migrated to MIN_NEW (v3 verification) before
# the replay/boundary stages, which anchor on the v3 text.
LEGACY_MIN = """                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                if _glm53_is_draft_swa_spec(spec):  # [glm53-hybrid-apc]
                    # Drafter SWA must not min() the hybrid hit. Its EAGLE pop
                    # re-aligns by LCM(window block, MLA page) = 3584. If the
                    # cached window does not cover the MLA/mamba hit, leave
                    # blocks empty so a fresh window is allocated; do not
                    # reseed the indexer tail here (KpoolTail already opted out).
                    if _new_hit_length >= curr_hit_length:
                        for group_id, blocks in zip(group_ids, hit_blocks):
                            hit_blocks_by_group[group_id] = blocks
                            hit_length_by_group[group_id] = _new_hit_length
                    continue
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)
"""

LOG_OLD = """        # Propagate the eagle bit to each manager (default to ``use_eagle=False``).
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True
"""

LOG_NEW = """        # Propagate the eagle bit to each manager (default to ``use_eagle=False``).
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True
        logger.info(  # [glm53-hybrid-apc]
            "hybrid APC groups: %s; eagle_group_ids=%s",
            [
                (
                    type(_glm53_inner_kv_spec(g.spec)).__name__,
                    g.group_ids,
                    getattr(g.manager_cls, "__name__", type(g.manager_cls).__name__),
                    g.use_eagle,
                )
                for g in self.attention_groups
            ],
            sorted(self.eagle_group_ids),
        )
"""

INIT_OLD = """        self.verify_and_split_kv_cache_groups()
"""

INIT_NEW = """        self.verify_and_split_kv_cache_groups()
        # [glm53-dflash-swa-replay-v1] The DFlash drafter needs a complete fresh
        # sliding-attention window after an APC switch. The target KpoolTail is a
        # separate 4-token request-local scratch group and is not this 2048-token
        # replay requirement.
        self.dflash_swa_replay_tokens = _glm53_dflash_swa_replay_tokens(
            kv_cache_config.kv_cache_groups
        )
        logger.info(
            "[glm53-dflash-swa-replay-v1] replay_tokens=%d alignment=%d",
            self.dflash_swa_replay_tokens,
            self._cache_hit_alignment_tokens,
        )
"""

CONVERGE_OLD = """            if curr_hit_length >= hit_length:
                break
"""

CONVERGE_FINAL = """            if curr_hit_length >= hit_length:
                # [glm53-dflash-swa-replay-v2] Reuse the current boundary when
                # every DFlash group actually returned a complete, EAGLE-popped
                # sliding window at exactly that reconciled target length. This
                # avoids backing a valid short-suffix hit up by a full scheduler
                # page. Length equality alone is insufficient: the returned
                # prefix is mostly null blocks, so verify the complete visible
                # window is materialized at its tail.
                draft_replay_ready = True
                draft_groups_seen = 0
                for draft_group_index, (
                    draft_spec,
                    draft_group_ids,
                    _,
                    draft_use_eagle,
                ) in enumerate(self.attention_groups):
                    if not _glm53_is_draft_swa_spec(draft_spec):
                        continue
                    draft_groups_seen += len(draft_group_ids)
                    draft_block_size = self.single_type_managers[
                        draft_group_ids[0]
                    ].block_size
                    required_tail_blocks = (
                        int(getattr(draft_spec, "sliding_window", 0) or 0) - 1
                        + draft_block_size - 1
                    ) // draft_block_size
                    expected_blocks = curr_hit_length // draft_block_size
                    for draft_group_id in draft_group_ids:
                        draft_blocks = hit_blocks_by_group[draft_group_id]
                        if (
                            not draft_use_eagle
                            or draft_group_index not in eagle_verified
                            or required_tail_blocks <= 0
                            or hit_length_by_group[draft_group_id] != curr_hit_length
                            or draft_blocks is None
                            or len(draft_blocks) != expected_blocks
                            or len(draft_blocks) < required_tail_blocks
                            or any(
                                block.is_null
                                for block in draft_blocks[-required_tail_blocks:]
                            )
                        ):
                            draft_replay_ready = False
                            break
                    if not draft_replay_ready:
                        break
                if draft_groups_seen and draft_replay_ready:
                    logger.info(
                        "[glm53-dflash-swa-replay-v2] reusing reconciled "
                        "DFlash boundary hit=%d fresh=%d required=%d",
                        curr_hit_length,
                        max_cache_hit_length + 1 - curr_hit_length,
                        self.dflash_swa_replay_tokens,
                    )
                    break
                replay_safe_hit = _glm53_dflash_replay_safe_hit(
                    curr_hit_length,
                    max_cache_hit_length,
                    self.dflash_swa_replay_tokens,
                    self._cache_hit_alignment_tokens,
                )
                if replay_safe_hit < curr_hit_length:
                    logger.info(
                        "[glm53-dflash-swa-replay-v2] replay clamp hit=%d->%d "
                        "fresh=%d required=%d alignment=%d",
                        curr_hit_length,
                        replay_safe_hit,
                        max_cache_hit_length + 1 - curr_hit_length,
                        self.dflash_swa_replay_tokens,
                        self._cache_hit_alignment_tokens,
                    )
                    hit_length = replay_safe_hit
                    # Cached drafter blocks were looked up at the larger target
                    # hit and cannot be paired with a backed-up target state.
                    # Discard them so the fresh replay rebuilds the complete
                    # DFlash sliding-attention window at the new boundary.
                    for draft_spec, draft_group_ids, _, _ in self.attention_groups:
                        if _glm53_is_draft_swa_spec(draft_spec):
                            for draft_group_id in draft_group_ids:
                                hit_blocks_by_group[draft_group_id] = None
                                hit_length_by_group[draft_group_id] = 0
                    eagle_verified.clear()
                    continue
                break
"""

# ---- [glm53-dflash-boundary-lookup-v1] --------------------------------------
# Applied on top of the hybrid-min and replay edits above (anchors are their
# NEW text), so it installs over pristine and previously patched images alike.

IMPORT_OLD = """from abc import ABC, abstractmethod
from collections.abc import Sequence
"""

IMPORT_NEW = """import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
"""

DFLASH_BOUNDARY_HELPER = '''
def _glm53_dflash_boundary_lookup_enabled() -> bool:
    """GLM53_DRAFT_KV_COMPACT=1: allocator-verified DFlash drafter pages.

    The allocator preflight (patch_glm5_drafter_group.py, at the entry of
    get_kv_cache_groups on every grouping path) fails boot under this flag
    unless every SlidingWindowSpec layer is the DFlash drafter's (speculative
    method and one layer per draft decoder layer), so here every exact
    SlidingWindowSpec group is the DFlash drafter.
    """
    mode = os.environ.get("GLM53_DRAFT_KV_COMPACT", "0")
    if mode not in ("0", "1"):
        raise ValueError("GLM53_DRAFT_KV_COMPACT must be 0 or 1")
    return mode == "1"


'''

BOUNDARY_INIT_OLD = """        logger.info(
            "[glm53-dflash-swa-replay-v1] replay_tokens=%d alignment=%d",
            self.dflash_swa_replay_tokens,
            self._cache_hit_alignment_tokens,
        )
"""

BOUNDARY_INIT_NEW = """        logger.info(
            "[glm53-dflash-swa-replay-v1] replay_tokens=%d alignment=%d",
            self.dflash_swa_replay_tokens,
            self._cache_hit_alignment_tokens,
        )
        # [glm53-dflash-boundary-lookup-v1] KV cache group ids whose window is
        # verified ending exactly at the reconciled boundary instead of one
        # EAGLE lookahead block past it. DFlash context KV at position p is a
        # per-position projection of the target hidden state at p (no token
        # p+1 in its dataflow), so the lookahead block adds nothing; with a
        # compact draft block B it cannot exist for prompts whose tail past
        # the boundary is shorter than B. Only the EAGLE-flagged drafter
        # SlidingWindowSpec groups qualify; the EAGLE flag itself (the group
        # still cannot shrink the hybrid min) and the replay clamp are
        # unchanged.
        self.dflash_boundary_group_ids: frozenset[int] = (
            frozenset(
                i
                for i, g in enumerate(kv_cache_config.kv_cache_groups)
                if i in self.eagle_group_ids
                and _glm53_is_draft_swa_spec(g.kv_cache_spec)
            )
            if _glm53_dflash_boundary_lookup_enabled()
            else frozenset()
        )
        for boundary_group_id in self.dflash_boundary_group_ids:
            # The boundary lookup consults exactly cdiv(window - 1, block)
            # cached blocks ending on the boundary, never the EAGLE lookahead
            # block. The manager's eagle bit only widens what is retained:
            # cache_blocks hashes one block past each aligned boundary and
            # SlidingWindowManager.reachable_block_mask keeps one more block
            # per tail (shifted onto the boundary block). Neither is ever
            # read back under the boundary lookup, so keep the manager
            # non-EAGLE: one fewer pool block id per retained window.
            self.single_type_managers[boundary_group_id].use_eagle = False
        logger.info(
            "[glm53-dflash-boundary-lookup-v1] boundary_group_ids=%s",
            sorted(self.dflash_boundary_group_ids),
        )
"""

BOUNDARY_LOOKUP_OLD = """                drop_eagle_block = use_eagle and idx not in eagle_verified

                _max_length = curr_hit_length
"""

BOUNDARY_LOOKUP_NEW = """                # [glm53-dflash-boundary-lookup-v1] No lookahead block, no
                # margin: the window must end exactly at curr_hit_length.
                _glm53_boundary_lookup = (
                    first_group_id in self.dflash_boundary_group_ids
                )
                drop_eagle_block = (
                    use_eagle
                    and idx not in eagle_verified
                    and not _glm53_boundary_lookup
                )

                _max_length = curr_hit_length
"""

BOUNDARY_VERIFY_OLD = """                _glm53_draft_swa = _glm53_is_draft_swa_spec(spec)
                if drop_eagle_block:
"""

BOUNDARY_VERIFY_NEW = """                _glm53_draft_swa = _glm53_is_draft_swa_spec(spec)
                if _glm53_boundary_lookup:
                    # [glm53-dflash-boundary-lookup-v1] Verified iff the
                    # complete window is cached up to the boundary; the
                    # replay check below still inspects the returned blocks.
                    if _new_hit_length >= curr_hit_length:
                        eagle_verified.add(idx)
                    else:
                        eagle_verified.discard(idx)
                elif drop_eagle_block:
"""


# These independent stages also migrate retained boundary-lookup installations.
FINE_GRAIN_MARK = "# [glm53-participating-fine-hits-v1]"
CAPABILITY_OLD = """                for manager in self.single_type_managers
                if not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size"""
CAPABILITY_NEW = """                for manager, group in zip(  # [glm53-participating-fine-hits-v1]
                    self.single_type_managers, kv_cache_config.kv_cache_groups
                )
                if group.kv_cache_spec.participates_in_prefix_caching
                and not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size"""
KPOOL_INIT_OLD = """        self.dflash_swa_replay_tokens = _glm53_dflash_swa_replay_tokens(
            kv_cache_config.kv_cache_groups
        )
"""
KPOOL_INIT_NEW = KPOOL_INIT_OLD + """        # [glm53-kpool-replay-floor-v1] A retained draft window cannot
        # replace the target's request-local circular scratch.
        self.kpool_replay_tokens = max(
            (
                _glm53_inner_kv_spec(group.kv_cache_spec).block_size
                for group in kv_cache_config.kv_cache_groups
                if not group.kv_cache_spec.participates_in_prefix_caching
                and type(_glm53_inner_kv_spec(group.kv_cache_spec)).__name__
                == "KpoolTailSpec"
            ),
            default=0,
        )
"""
KPOOL_HIT_OLD = """        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
"""
KPOOL_HIT_NEW = """        num_groups = len(self.kv_cache_config.kv_cache_groups)
        # [glm53-kpool-replay-floor-v1] Preserve the true prompt/logits limit
        # for EAGLE lookahead and replay accounting; bound only the candidate.
        hit_length = max(0, min(
            max_cache_hit_length,
            max_cache_hit_length + 1 - self.kpool_replay_tokens,
        ))
"""
COARSE_RETRY_OLD = """                    self._cache_hit_alignment_tokens,
                )
                if replay_safe_hit < curr_hit_length:
"""
COARSE_RETRY_NEW = """                    self._cache_hit_alignment_tokens,
                )
                # [glm53-partial-replay-fallback-v1] A fine Mamba tail may
                # outrun the draft window. Retry the preceding shared boundary
                # before discarding it; the loop still verifies every group.
                if curr_hit_length % self.scheduler_block_size:
                    replay_safe_hit = max(
                        replay_safe_hit,
                        curr_hit_length // self.scheduler_block_size
                        * self.scheduler_block_size,
                    )
                if replay_safe_hit < curr_hit_length:
"""
FINE_EDITS = (
    ("participating capability", CAPABILITY_OLD, CAPABILITY_NEW),
    ("Kpool replay init", KPOOL_INIT_OLD, KPOOL_INIT_NEW),
    ("Kpool replay candidate", KPOOL_HIT_OLD, KPOOL_HIT_NEW),
    ("partial replay fallback", COARSE_RETRY_OLD, COARSE_RETRY_NEW),
)



# Owned stage content that a complete installation carries exactly once, and
# superseded forms it must not carry. A stage marker only selects the
# migration path; this table is what decides whether the result is written.
# Helper blobs are compared stripped: the per-group overlay inserts the
# identical base helper and may separate blobs with different whitespace.
# The boundary stage rewrites the verification preamble inside the v3
# hybrid-min block, so the complete form of that block is MIN_NEW with the
# boundary verification applied.
assert MIN_NEW.count(BOUNDARY_VERIFY_OLD) == 1
MIN_FINAL = MIN_NEW.replace(BOUNDARY_VERIFY_OLD, BOUNDARY_VERIFY_NEW, 1)
REQUIRED_ONCE = (
    ("hybrid-apc helpers", BASE_HELPER.strip()),
    ("eagle-fallback", EAGLE_NEW),
    ("hybrid-min", MIN_FINAL),
    ("group-log", LOG_NEW),
    ("dflash-replay helpers", DFLASH_REPLAY_HELPER.strip()),
    ("dflash-replay-init", INIT_NEW),
    ("dflash-replay-clamp", CONVERGE_FINAL),
    ("dflash-boundary helper", DFLASH_BOUNDARY_HELPER.strip()),
    ("dflash-boundary-init", BOUNDARY_INIT_NEW),
    ("dflash-boundary-lookup", BOUNDARY_LOOKUP_NEW),
    ("dflash-boundary-verify", BOUNDARY_VERIFY_NEW),
)
SUPERSEDED = (
    ("eagle-fallback", EAGLE_OLD),
    ("hybrid-min", MIN_OLD),
    ("legacy-hybrid-min", LEGACY_MIN),
    ("dflash-replay-clamp", CONVERGE_OLD),
    ("dflash-boundary-lookup", BOUNDARY_LOOKUP_OLD),
    ("dflash-boundary-verify", BOUNDARY_VERIFY_OLD),
)
OWNED_HELPERS = {
    node.name
    for node in ast.parse(
        BASE_HELPER + DFLASH_REPLAY_HELPER + DFLASH_BOUNDARY_HELPER
    ).body
    if isinstance(node, ast.FunctionDef)
}


def verify_complete(text: str) -> list[str]:
    """Names of stages whose owned content is missing, duplicated, or stale."""
    # Normalize only exact current stages, so nested older-stage checks remain
    # strict without maintaining a second boundary/replay implementation.
    for label, old, new in FINE_EDITS:
        if text.count(new) != 1:
            return [f"{label}: missing, duplicated, or drifted stage"]
        if old in text.replace(new, "", 1):
            return [f"{label}: competing superseded stage"]
        text = text.replace(new, old, 1)
    problems = [
        f"{label}: expected exactly one owned block, found {n}"
        for label, block in REQUIRED_ONCE
        if (n := text.count(block)) != 1
    ]
    problems += [
        f"{label}: superseded form still present"
        for label, block in SUPERSEDED
        if block in text
    ]
    if sum(line.startswith("import os") for line in text.splitlines()) != 1:
        problems.append("os-import: expected exactly one module-level import")
    try:
        module = ast.parse(text)
    except SyntaxError as exc:
        return problems + [f"coordinator syntax: {exc.msg}"]
    # Remove the one canonical definition of each helper from an analysis-only
    # AST. Let Python's compiler identify any remaining global bindings; this
    # includes pattern captures without guessing at each binding syntax.
    definitions = dict.fromkeys(OWNED_HELPERS, 0)
    remaining = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in definitions:
            definitions[node.name] += 1
        else:
            remaining.append(node)
    problems += [
        f"{name}: expected exactly one module definition, found {count}"
        for name, count in sorted(definitions.items())
        if count != 1
    ]
    module.body = remaining
    try:
        compiled = compile(module, str(P), "exec", dont_inherit=True)
    except SyntaxError as exc:
        return problems + [f"coordinator bindings: {exc.msg}"]
    rebound = set()
    pending = [(compiled, True)]
    while pending:
        code, module_scope = pending.pop()
        for instruction in dis.get_instructions(code):
            if instruction.opname in ("STORE_GLOBAL", "DELETE_GLOBAL") or (
                module_scope and instruction.opname in ("STORE_NAME", "DELETE_NAME")
            ):
                if instruction.argval in OWNED_HELPERS:
                    rebound.add(instruction.argval)
        pending.extend(
            (constant, False) for constant in code.co_consts
            if isinstance(constant, CodeType)
        )
    problems += [f"{name}: competing global binding" for name in sorted(rebound)]
    return problems


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    original = text
    needle = "def _validate_prefix_cache_retention_interval(\n"
    if text.count(needle) != 1:
        raise SystemExit(f"{P}: helper insert point not unique")
    if MARK not in text:
        if "def _glm53_inner_kv_spec(" not in text:
            text = text.replace(needle, BASE_HELPER + needle, 1)
        text = replace_once(text, EAGLE_OLD, EAGLE_NEW, "eagle-fallback")
        text = replace_once(text, MIN_OLD, MIN_NEW, "hybrid-min")
        text = replace_once(text, LOG_OLD, LOG_NEW, "group-log")
    elif EAGLE_VERIFY_MARK not in text:
        # Legacy hybrid-apc source (the stock image, with or without the
        # replay stage): only the recognized first form migrates; anything
        # else is drift and fails before any write.
        text = replace_once(text, LEGACY_MIN, MIN_NEW, "legacy-hybrid-min")
    if DFLASH_REPLAY_MARK not in text:
        if "def _glm53_dflash_swa_replay_tokens(" not in text:
            # Compose with the per-group overlay in a consistent helper order
            # (Kpool helpers precede its retention helpers) regardless of
            # which overlay runs first; whitespace between the blobs may
            # differ, the AST may not.
            replay_needle = (
                "def _glm53_swa_retention_env("
                if "def _glm53_swa_retention_env(" in text
                else needle
            )
            text = text.replace(
                replay_needle, DFLASH_REPLAY_HELPER + replay_needle, 1
            )
        text = replace_once(text, INIT_OLD, INIT_NEW, "dflash-replay-init")
        text = replace_once(
            text, CONVERGE_OLD, CONVERGE_FINAL, "dflash-replay-clamp"
        )
    if DFLASH_BOUNDARY_MARK not in text:
        # patch_apc_per_group_retention.py adds ``import os  # [...]``; either
        # overlay may run first.
        if not any(line.startswith("import os") for line in text.splitlines()):
            text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "os-import")
        if "def _glm53_dflash_boundary_lookup_enabled(" not in text:
            # Same placement rule as the replay helper: ahead of the per-group
            # overlay's helpers, so both application orders yield one AST.
            boundary_needle = (
                "def _glm53_swa_retention_env("
                if "def _glm53_swa_retention_env(" in text
                else needle
            )
            text = text.replace(
                boundary_needle, DFLASH_BOUNDARY_HELPER + boundary_needle, 1
            )
        text = replace_once(
            text, BOUNDARY_INIT_OLD, BOUNDARY_INIT_NEW, "dflash-boundary-init"
        )
        text = replace_once(
            text, BOUNDARY_LOOKUP_OLD, BOUNDARY_LOOKUP_NEW, "dflash-boundary-lookup"
        )
        text = replace_once(
            text, BOUNDARY_VERIFY_OLD, BOUNDARY_VERIFY_NEW, "dflash-boundary-verify"
        )
    if FINE_GRAIN_MARK not in text:
        for label, old, new in FINE_EDITS:
            text = replace_once(text, old, new, label)
    # A marker only chose the path above; the written result must carry every
    # owned stage exactly as supported (stale markers, partial stages, edited
    # verification logic and duplicated stages all stop here, unwritten).
    if problems := verify_complete(text):
        raise SystemExit(f"{P}: incomplete or drifted overlay state: " + "; ".join(problems))
    compile(text, str(P), "exec")
    if text != original:
        import tempfile

        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=P.parent, delete=False) as out:
                temporary = Path(out.name)
                out.write(text)
            temporary.chmod(P.stat().st_mode)
            os.replace(temporary, P)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(
        f"patched {P.name} (hybrid APC + versioned DFlash SWA replay clamp "
        "+ DFlash boundary lookup)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
