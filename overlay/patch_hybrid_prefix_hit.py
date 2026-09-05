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
Mamba align-mode state *does* materialize at 896-token chunk ends, and
3584 is a multiple of 896, so mamba must stay in the min — skipping a
mamba miss is a correctness hole (vLLM #47491 / #43090).

This patch: flag only exact SlidingWindowSpec groups as EAGLE, and do not
let that drafter group shrink ``curr_hit_length``. If the drafter window
does not cover the MLA/mamba hit, leave its blocks empty so a fresh
window is allocated (zeros / new pages).

KpoolTail is deliberately per-request scratch and cannot prefix-cache. A
resumed request therefore starts with an empty indexer-tail ring. Replaying
fewer than one complete kpool leaves the ring incomplete and changes sparse
attention selection. Clamp the shared hit back to an earlier scheduler
boundary when necessary so at least one complete kpool is rebuilt before
decode. This preserves the older MLA/Mamba prefix hit while making the
non-shareable target state deterministic.

Fail closed if the vLLM coordinator anchors drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_KV_COORDINATOR_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py",
    )
)
MARK = "# [glm53-hybrid-apc]"
DFLASH_REPLAY_MARK = "# [glm53-dflash-swa-replay-v1]"
DFLASH_RECONCILE_MARK = "# [glm53-dflash-swa-replay-v2]"
DFLASH_EAGLE_VERIFY_MARK = "# [glm53-dflash-eagle-verify-v3]"

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

CONVERGE_NEW = """            if curr_hit_length >= hit_length:
                replay_safe_hit = _glm53_dflash_replay_safe_hit(  # [glm53-dflash-swa-replay-v1]
                    curr_hit_length,
                    max_cache_hit_length,
                    self.dflash_swa_replay_tokens,
                    self._cache_hit_alignment_tokens,
                )
                if replay_safe_hit < curr_hit_length:
                    logger.info(
                        "[glm53-dflash-swa-replay-v1] replay clamp hit=%d->%d "
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

CONVERGE_RECONCILED = """            if curr_hit_length >= hit_length:
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

EAGLE_VERIFY_V2 = """                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                if _glm53_is_draft_swa_spec(spec):  # [glm53-hybrid-apc]
"""

EAGLE_VERIFY_V3 = """                _glm53_draft_swa = _glm53_is_draft_swa_spec(spec)
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
"""

RECONCILE_LOOP_V2 = """                for (
                    draft_spec,
                    draft_group_ids,
                    _,
                    draft_use_eagle,
                ) in self.attention_groups:
"""

RECONCILE_LOOP_V3 = """                for draft_group_index, (
                    draft_spec,
                    draft_group_ids,
                    _,
                    draft_use_eagle,
                ) in enumerate(self.attention_groups):
"""

RECONCILE_CHECK_V2 = """                            not draft_use_eagle
                            or required_tail_blocks <= 0
"""

RECONCILE_CHECK_V3 = """                            not draft_use_eagle
                            or draft_group_index not in eagle_verified
                            or required_tail_blocks <= 0
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    needle = "def _validate_prefix_cache_retention_interval(\n"
    if text.count(needle) != 1:
        raise SystemExit(f"{P}: helper insert point not unique")
    if MARK not in text:
        if "def _glm53_inner_kv_spec(" not in text:
            text = text.replace(needle, BASE_HELPER + needle, 1)
        text = replace_once(text, EAGLE_OLD, EAGLE_NEW, "eagle-fallback")
        text = replace_once(text, MIN_OLD, MIN_NEW, "hybrid-min")
        text = replace_once(text, LOG_OLD, LOG_NEW, "group-log")
    if DFLASH_REPLAY_MARK not in text:
        if "def _glm53_dflash_swa_replay_tokens(" not in text:
            # Keep byte-identical composition with the per-group overlay in
            # either order: Kpool helpers precede its retention helpers.
            replay_needle = (
                "def _glm53_swa_retention_env("
                if "def _glm53_swa_retention_env(" in text
                else needle
            )
            text = text.replace(
                replay_needle, DFLASH_REPLAY_HELPER + replay_needle, 1
            )
        text = replace_once(text, INIT_OLD, INIT_NEW, "dflash-replay-init")
        text = replace_once(text, CONVERGE_OLD, CONVERGE_NEW, "dflash-replay-clamp")
    if DFLASH_RECONCILE_MARK not in text:
        text = replace_once(
            text,
            CONVERGE_NEW,
            CONVERGE_RECONCILED,
            "dflash-reconciled-boundary-reuse",
        )
    if DFLASH_EAGLE_VERIFY_MARK not in text:
        text = replace_once(
            text, EAGLE_VERIFY_V2, EAGLE_VERIFY_V3, "dflash-eagle-verification"
        )
        text = replace_once(
            text, RECONCILE_LOOP_V2, RECONCILE_LOOP_V3, "dflash-reconcile-index"
        )
        text = replace_once(
            text,
            RECONCILE_CHECK_V2,
            RECONCILE_CHECK_V3,
            "dflash-reconcile-verification",
        )
    # Canonicalize only the two adjacent injected helper boundaries. The two
    # overlays both insert before the upstream validator, so without this the
    # application order can differ by one blank line despite identical code.
    for left, right in (
        (
            'return type(_glm53_inner_kv_spec(spec)).__name__ == "SlidingWindowSpec"',
            "def _glm53_dflash_swa_replay_tokens(",
        ),
        (
            "return max(0, hit_length - pages * alignment_tokens)",
            "def _glm53_swa_retention_env(",
        ),
    ):
        if left in text and right in text:
            left_end = text.index(left) + len(left)
            right_start = text.index(right, left_end)
            if text[left_end:right_start].strip():
                raise SystemExit(f"{P}: unexpected code between injected helpers")
            text = text[:left_end] + "\n\n\n" + text[right_start:]
    P.write_text(text)
    print(
        f"patched {P.name} (hybrid APC + versioned DFlash SWA replay clamp)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
