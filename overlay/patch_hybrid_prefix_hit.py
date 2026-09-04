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
3. Fine-grained 64-token Mamba/MLA hits are disabled when *any* manager with
   a wider block lacks fine-grained lookup support. That check includes
   KpoolTailManager even though KpoolTailSpec explicitly opts out of prefix
   caching, so one transient scratch group forces every reusable group back
   to 3584-token hit alignment.

KpoolTail already opts out of prefix caching (1-block circular scratch).
Mamba align-mode state *does* materialize at 896-token chunk ends, and
3584 is a multiple of 896, so mamba must stay in the min — skipping a
mamba miss is a correctness hole (vLLM #47491 / #43090).

This patch: flag only exact SlidingWindowSpec groups as EAGLE, do not let that
drafter group shrink ``curr_hit_length``, and ignore non-participating cache
groups when checking whether fine-grained lookup is safe. If the drafter
window does not cover the MLA/mamba hit, leave its blocks empty so a fresh
window is allocated (zeros / new pages). Wrong indexer tail state is fatal —
we do not change KpoolTailManager or make its transient state shareable.

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
FINE_MARK = "# [glm53-hybrid-apc-fine]"

HELPER = '''
def _glm53_inner_kv_spec(spec):
    specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(specs, dict) and specs:
        return next(iter(specs.values()))
    return spec


def _glm53_is_draft_swa_spec(spec) -> bool:
    """DFlash2 drafter: exact SlidingWindowSpec, not KpoolTailSpec."""
    return type(_glm53_inner_kv_spec(spec)).__name__ == "SlidingWindowSpec"


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

MIN_NEW = """                if drop_eagle_block:
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

FINE_OLD = """            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager in self.single_type_managers
                if not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }
"""

FINE_NEW = """            unsupported_partial_hit_managers = {  # [glm53-hybrid-apc-fine]
                type(manager).__name__
                for manager, group in zip(
                    self.single_type_managers, kv_cache_config.kv_cache_groups
                )
                # A transient/non-shareable manager cannot constrain prefix-hit
                # granularity because it never participates in the lookup.
                if group.kv_cache_spec.participates_in_prefix_caching
                and not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }
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
    changes: list[str] = []

    # The GHCR image can already contain an older revision of this overlay.
    # Preserve the original idempotence while still allowing additive upgrades.
    if MARK not in text:
        needle = "def _validate_prefix_cache_retention_interval(\n"
        if text.count(needle) != 1:
            raise SystemExit(f"{P}: helper insert point not unique")
        if "def _glm53_inner_kv_spec(" not in text:
            text = text.replace(needle, HELPER + needle, 1)
        text = replace_once(text, EAGLE_OLD, EAGLE_NEW, "eagle-fallback")
        text = replace_once(text, MIN_OLD, MIN_NEW, "hybrid-min")
        text = replace_once(text, LOG_OLD, LOG_NEW, "group-log")
        changes.append("hybrid APC")

    if FINE_MARK not in text:
        text = replace_once(
            text,
            FINE_OLD,
            FINE_NEW,
            "fine-grained manager eligibility",
        )
        changes.append("64-token fine-grained hits")

    if not changes:
        print(f"{P.name}: {MARK} and {FINE_MARK} already present — skipping")
        return 0

    P.write_text(text)
    print(f"patched {P.name} ({', '.join(changes)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
