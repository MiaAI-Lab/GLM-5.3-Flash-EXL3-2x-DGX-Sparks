#!/usr/bin/env python3
"""Per-KV-cache-group prefix-cache retention (stop the DFlash2 drafter evicting APC).

NOT APPLIED. Recipe-overlay style: fail-closed, idempotent, MARK/anchor, matching
overlay/patch_hybrid_prefix_hit.py. See docs/DESIGN-apc-per-group-retention.md for the analysis.

Problem (all line refs = the live fork, read-only):

  The drafter group (exact SlidingWindowSpec, window 2048, block 64, EAGLE) hashes
  ``_contiguous_blocks_for_hit(2048, 64, use_eagle=True) == 33`` block ids at EVERY
  3584-token hit boundary (single_type_kv_cache_manager.py:998-1057), i.e. 33 of the
  38 ids a cached 3584-token segment costs (MLA 1 + mamba 4 + drafter 33). Those
  blocks are freed *cached* by remove_skipped_blocks -> _remove_blocks_in_range ->
  BlockPool.free_blocks (block_pool.py:719-743), which appends hashed blocks to the
  LRU **tail** — the most protected end — while the MLA/mamba blocks that actually
  carry the hit are freed later and sit ahead of them. Priority inversion: 87% of
  the 642-id pool is spent on a group whose hit length is discarded by the hybrid
  min() anyway (kv_cache_coordinator.py:856-868).

  The current mitigation passes ONE VLLM_PREFIX_CACHE_RETENTION_INTERVAL to every
  manager (kv_cache_coordinator.py:154, :316, :753), so buying capacity also
  coarsens the mamba/MLA hit grid to that interval (subagent hits fell to 45%).

This patch: make ``retention_interval`` per group. The EAGLE-exempt drafter group
gets its own explicit value from ``VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA``;
when the variable is unset, every group keeps the global value. No manager-side
change is needed: ``SingleTypeKVCacheManager.cache_blocks`` already takes
retention_interval per call (single_type_kv_cache_manager.py:429-434) and each
reachable_block_mask already honours None / 0 / >0.

Env contract:
  VLLM_PREFIX_CACHE_RETENTION_INTERVAL      unchanged (global; None = dense)
  VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA  ""/unset = inherit the global value
                                            "0"     = reachable boundaries only (recommended)
                                            N       = multiple of scheduler_block_size,
                                                      0 <= N <= 1,000,000

  The raw value is validated **unconditionally** at coordinator init (integer,
  non-negative, scheduler-block multiple, <= 1,000,000) — an unusable value is a
  boot failure even on a model with no sliding-window group.

  It is applied **only** to groups the coordinator itself has singled out as the
  EAGLE-exempt drafter (``_glm53_min_exempt_group_ids``, derived from
  ``eagle_group_ids`` — the narrowing overlay/patch_hybrid_prefix_hit.py performs —
  not from a class name alone). Setting it when no such group exists is a
  fail-closed boot error, not a silent no-op.

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
BP = Path(
    os.environ.get(
        "GLM53_BLOCK_POOL_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py",
    )
)
STM = Path(
    os.environ.get(
        "GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/"
        "single_type_kv_cache_manager.py",
    )
)
MARK = "# [glm53-apc-per-group]"
CONTRACT = "glm53-apc-per-group-contract:explicit-v1"
DIAG_MARK = "# [glm53-apc-retention-diag]"
DIAG_HIT_MARK = "# [glm53-apc-hit-diag]"
PRIORITY_MARK = "# [glm53-apc-drafter-priority]"
BP_PRIORITY_MARK = "# [glm53-apc-drafter-priority]"
PRIOR_HELPER_MARK = "# [glm53-dflash-prior-helper-v1]"
PRIOR_POLICY_MARK = "# [glm53-dflash-prior-policy-v1]"
PRIOR_MANAGER_MARK = "# [glm53-dflash-prior-manager-v1]"

# ---------------------------------------------------------------- anchors ----

IMPORT_OLD = """from abc import ABC, abstractmethod
from collections.abc import Sequence
"""

IMPORT_NEW = """import os  # [glm53-apc-per-group]
from abc import ABC, abstractmethod
from collections.abc import Sequence
"""

# Shared with overlay/patch_hybrid_prefix_hit.py; inserted only if absent, so the
# two overlays may be applied in either order.
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

RETENTION_HELPER = '''
def _glm53_swa_retention_env(  # [glm53-apc-per-group]
    scheduler_block_size,
    max_interval=1_000_000,
):
    """Parse + validate VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA.

    Unset/empty -> None (inherit the global value). Otherwise the raw value is validated
    *unconditionally* — before any group is inspected — so a typo is a boot
    failure rather than a silent no-op on a model without a drafter group:
    integer, non-negative, a multiple of ``scheduler_block_size`` (the base
    cache-hit granularity, so a retained tail lands on a real hit boundary),
    and <= ``max_interval``.
    """
    key = "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA"
    raw = os.environ.get(key, "")
    if raw.strip() == "":
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(
            f"{key} must be an integer (got {raw!r}); "
            "leave it unset to inherit the global value."
        ) from None
    if value < 0:
        raise ValueError(f"{key} ({value}) must be non-negative.")
    if scheduler_block_size and value % scheduler_block_size != 0:
        raise ValueError(
            f"{key} ({value}) must be a multiple of scheduler_block_size "
            f"({scheduler_block_size}); 0 means reachable boundaries only."
        )
    if value > max_interval:
        raise ValueError(
            f"{key} ({value}) exceeds the maximum supported retention "
            f"interval ({max_interval})."
        )
    return value


def _glm53_min_exempt_group_ids(  # [glm53-apc-per-group]
    kv_cache_groups,
    eagle_group_ids,
    is_hybrid_coordinator,
):
    """Group ids a hybrid coordinator treats as EAGLE-exempt drafter groups.

    Derived from coordinator type and state, never from a spec class name alone.
    A group qualifies only when

      (a) the active coordinator is ``HybridKVCacheCoordinator``,
      (b) its inner spec is an *exact* ``SlidingWindowSpec`` (the DFlash2
          drafter; ``KpoolTailSpec`` subclasses it and never prefix-caches), and
      (c) ``eagle_group_ids`` is *exactly* that set of drafter groups.

    (c) is the state overlay/patch_hybrid_prefix_hit.py establishes: it narrows
    ``eagle_group_ids`` from the upstream all-groups fallback down to the drafter
    SWA groups and, driven by the same discriminator, makes those groups skip the
    hybrid hit ``min()``. A base coordinator has no hybrid ``min()`` even when a
    unitary SWA model happens to have matching EAGLE ids, so it is never exempt.
    An undiscriminating all-groups fallback or an annotation over another group
    also returns the empty set.
    """
    if not is_hybrid_coordinator:
        return frozenset()
    eagle = set(eagle_group_ids)
    swa = {
        i
        for i, g in enumerate(kv_cache_groups)
        if _glm53_is_draft_swa_spec(g.kv_cache_spec)
    }
    if not swa or eagle != swa:
        return frozenset()
    return frozenset(swa)


def _glm53_retention_for_group(  # [glm53-apc-per-group]
    spec,
    global_interval,
    swa_interval,
    is_min_exempt,
):
    """Prefix-cache retention interval for one KV cache group.

    Only a group that is both an exact ``SlidingWindowSpec`` **and** EAGLE/
    min-exempt per ``_glm53_min_exempt_group_ids`` diverges from the global
    value. An unset SWA interval preserves the global policy. An explicit value
    applies only to a min-exempt drafter group; the caller validates it and
    refuses the setting outright if no such group exists.
    """
    if (
        swa_interval is None
        or not _glm53_is_draft_swa_spec(spec)
        or not is_min_exempt
    ):
        return global_interval
    return swa_interval


def _glm53_resolve_retention_by_group(  # [glm53-apc-per-group]
    kv_cache_groups,
    eagle_group_ids,
    is_hybrid_coordinator,
    global_interval,
    swa_interval,
):
    """Resolve the per-group retention vector, failing closed on a misdirected
    SWA override.

    An explicit ``VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA`` is only meaningful
    for the EAGLE-exempt drafter group. If the coordinator has not singled such a
    group out, refuse to boot rather than ignore the setting (which would look
    like the knob worked) or apply it by class name (which would sparsify a group
    that is still inside the hit ``min()``).
    """
    min_exempt = _glm53_min_exempt_group_ids(
        kv_cache_groups, eagle_group_ids, is_hybrid_coordinator
    )
    if swa_interval is not None and not min_exempt:
        raise ValueError(
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA is set "
            f"({swa_interval}) but this coordinator has no hybrid-min-exempt drafter "
            "sliding-window group: is_hybrid_coordinator="
            f"{is_hybrid_coordinator}, eagle_group_ids={sorted(set(eagle_group_ids))}. "
            "Applying it would sparsify a group that still determines prefix hits. "
            "Unset the variable to inherit the global retention policy."
        )
    return tuple(
        _glm53_retention_for_group(
            g.kv_cache_spec,
            global_interval,
            swa_interval,
            i in min_exempt,
        )
        for i, g in enumerate(kv_cache_groups)
    )


def _glm53_validate_retention_intervals(  # [glm53-apc-per-group]
    intervals,
    scheduler_block_size,
):
    """Per-group counterpart of _validate_prefix_cache_retention_interval.

    Defence in depth over the resolved vector: every non-``None`` entry is either
    0 or a non-negative multiple of ``scheduler_block_size``. No upper bound here
    — inherited *global* values are upstream's to bound; the 1,000,000 cap
    applies to the SWA value this overlay introduces (``_glm53_swa_retention_env``).
    """
    for group_id, value in enumerate(intervals):
        if value is None or value == 0:
            continue
        if value < 0 or value % scheduler_block_size != 0:
            raise ValueError(
                f"prefix-cache retention interval for KV cache group {group_id} "
                f"({value}) must be non-negative and a multiple of "
                f"scheduler_block_size ({scheduler_block_size})."
            )


def _glm53_format_retention_vector(intervals):  # [glm53-apc-per-group]
    """Compact, greppable rendering of the resolved vector: ``[None,None,0]``."""
    return "[" + ",".join("None" if v is None else str(v) for v in intervals) + "]"


'''

DFLASH_PRIOR_HELPER = '''
def _glm53_dflash_prior_mamba_group_ids(  # [glm53-dflash-prior-helper-v1]
    kv_cache_groups,
    retention_intervals,
    is_hybrid_coordinator,
):
    """Sparse Mamba groups needing one prior state for full DFlash SWA replay."""
    if not is_hybrid_coordinator:
        return frozenset()
    has_dflash_swa = any(
        type(_glm53_inner_kv_spec(g.kv_cache_spec)).__name__ == "SlidingWindowSpec"
        for g in kv_cache_groups
    )
    if not has_dflash_swa:
        return frozenset()
    return frozenset(
        i
        for i, g in enumerate(kv_cache_groups)
        if retention_intervals[i] == 0
        and type(_glm53_inner_kv_spec(g.kv_cache_spec)).__name__ == "MambaSpec"
    )


'''

INIT_OLD = """        self.retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
        _validate_prefix_cache_retention_interval(
            self.retention_interval, self.scheduler_block_size, kv_cache_config
        )
"""

INIT_NEW = """        self.retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL
        _validate_prefix_cache_retention_interval(
            self.retention_interval, self.scheduler_block_size, kv_cache_config
        )
        # [glm53-apc-per-group] Per-group retention. The DFlash2 drafter SWA group
        # hashes cdiv(window - 1, block) + 1 == 33 block ids at EVERY hit boundary
        # and frees them *cached* onto the LRU tail (BlockPool.free_blocks), i.e.
        # 33 of the 38 ids a cached 3584-token segment costs -- for a group that is
        # exempted from the hybrid hit min(). Give it its own sparse interval so
        # MLA and mamba can stay dense and keep the fine hit grid. The raw env
        # value is validated unconditionally; it is applied only when the hybrid
        # coordinator itself flagged the drafter group EAGLE-exempt, else boot fails.
        _glm53_swa_interval = _glm53_swa_retention_env(self.scheduler_block_size)
        _glm53_is_hybrid = isinstance(self, HybridKVCacheCoordinator)
        _glm53_min_exempt = _glm53_min_exempt_group_ids(
            kv_cache_config.kv_cache_groups,
            self.eagle_group_ids,
            _glm53_is_hybrid,
        )
        self.retention_interval_by_group = _glm53_resolve_retention_by_group(
            kv_cache_config.kv_cache_groups,
            self.eagle_group_ids,
            _glm53_is_hybrid,
            self.retention_interval,
            _glm53_swa_interval,
        )
        _glm53_validate_retention_intervals(
            self.retention_interval_by_group, self.scheduler_block_size
        )
        logger.info(  # [glm53-apc-per-group]
            "[glm53-apc-per-group] retention_by_group=%s "
            "(global=%s swa_env=%s eagle_min_exempt=%s)",
            _glm53_format_retention_vector(self.retention_interval_by_group),
            self.retention_interval,
            _glm53_swa_interval,
            sorted(_glm53_min_exempt),
        )
"""

BASE_CACHE_OLD = """        for manager in self.single_type_managers:
            manager.cache_blocks(
                request,
                num_computed_tokens,
                retention_interval=self.retention_interval,
            )
"""

BASE_CACHE_NEW = """        for i, manager in enumerate(self.single_type_managers):
            manager.cache_blocks(
                request,
                num_computed_tokens,
                # [glm53-apc-per-group] per-group, not the single global value
                retention_interval=self.retention_interval_by_group[i],
            )
"""

HYBRID_LOOP_OLD = """        for manager in self.single_type_managers:
            num_tokens_to_cache = aligned_num_computed_tokens
"""

HYBRID_LOOP_NEW = """        for i, manager in enumerate(self.single_type_managers):
            num_tokens_to_cache = aligned_num_computed_tokens
"""

HYBRID_CACHE_OLD = """            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                retention_interval=self.retention_interval,
            )
"""

HYBRID_CACHE_NEW = """            manager.cache_blocks(
                request,
                num_tokens_to_cache,
                # [glm53-apc-per-group] per-group, not the single global value
                retention_interval=self.retention_interval_by_group[i],
            )
"""

DIAG_IMPORT_OLD = """from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
)
"""

DIAG_IMPORT_NEW = """from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_group_id,  # [glm53-apc-retention-diag]
)
"""

DIAG_FREE_OLD = """    def free(self, request_id: str) -> None:
        \"\"\"
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        \"\"\"
        for manager in self.single_type_managers:
            manager.free(request_id)
"""

DIAG_FREE_NEW = """    def free(self, request_id: str) -> None:
        \"\"\"
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        \"\"\"
        def _diag_pool_state():  # [glm53-apc-retention-diag]
            hashes_by_group = [0] * len(self.single_type_managers)
            cached_physical = 0
            for block in self.block_pool.blocks:
                keys = []
                if block.block_hash is not None:
                    keys.append(block.block_hash)
                keys.extend(
                    self.block_pool.cached_block_hashes_by_block.get(
                        block.block_id, ()
                    )
                )
                if keys:
                    cached_physical += 1
                for key in keys:
                    hashes_by_group[get_group_id(key)] += 1
            return (
                self.block_pool.get_num_free_blocks(),
                cached_physical,
                hashes_by_group,
            )

        request_nonnull = []
        request_hashed = []
        for manager in self.single_type_managers:
            blocks = manager.req_to_blocks.get(request_id, ())
            request_nonnull.append(sum(not block.is_null for block in blocks))
            request_hashed.append(
                sum(not block.is_null and block.block_hash is not None for block in blocks)
            )
        before = _diag_pool_state()
        for manager in self.single_type_managers:
            manager.free(request_id)
        after = _diag_pool_state()
        logger.info(
            \"[glm53-apc-retention-diag] request=%s \"
            \"request_nonnull=%s request_hashed=%s \"
            \"before_free=(free=%d,cached_physical=%d,hashes_by_group=%s) \"
            \"after_free=(free=%d,cached_physical=%d,hashes_by_group=%s)\",
            request_id,
            request_nonnull,
            request_hashed,
            *before,
            *after,
        )
"""

DIAG_HIT_OLD = """        return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens
"""

DIAG_HIT_NEW = """        logger.info(  # [glm53-apc-hit-diag]
            \"[glm53-apc-hit-diag] max=%d per_group=%s reconciled=%d \"
            \"longest=%d uncached_common=%d\",
            max_cache_hit_length,
            hit_length_by_group,
            hit_length,
            longest_hit_length,
            num_uncached_common_prefix_tokens,
        )
        return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens
"""

PRIORITY_OLD = """        _glm53_validate_retention_intervals(
            self.retention_interval_by_group, self.scheduler_block_size
        )
        logger.info(  # [glm53-apc-per-group]
            "[glm53-apc-per-group] retention_by_group=%s "
            "(global=%s swa_env=%s eagle_min_exempt=%s)",
            _glm53_format_retention_vector(self.retention_interval_by_group),
            self.retention_interval,
            _glm53_swa_interval,
            sorted(_glm53_min_exempt),
        )
"""

PRIORITY_NEW = """        _glm53_validate_retention_intervals(
            self.retention_interval_by_group, self.scheduler_block_size
        )
        # Explicit boundary-only retention keeps the min-exempt drafter's
        # replay window available for immediate reuse, but those blocks must
        # not evict target MLA/Mamba state under shared-pool pressure.
        self.block_pool.low_priority_cache_group_ids = (  # [glm53-apc-drafter-priority]
            frozenset(_glm53_min_exempt)
            if _glm53_swa_interval == 0
            else frozenset()
        )
        logger.info(  # [glm53-apc-per-group]
            "[glm53-apc-per-group] retention_by_group=%s "
            "(global=%s swa_env=%s eagle_min_exempt=%s low_priority=%s)",
            _glm53_format_retention_vector(self.retention_interval_by_group),
            self.retention_interval,
            _glm53_swa_interval,
            sorted(_glm53_min_exempt),
            sorted(self.block_pool.low_priority_cache_group_ids),
        )
"""

PRIOR_POLICY_OLD = """        self.block_pool.low_priority_cache_group_ids = (  # [glm53-apc-drafter-priority]
            frozenset(_glm53_min_exempt)
            if _glm53_swa_interval == 0
            else frozenset()
        )
        logger.info(  # [glm53-apc-per-group]
            "[glm53-apc-per-group] retention_by_group=%s "
            "(global=%s swa_env=%s eagle_min_exempt=%s low_priority=%s)",
            _glm53_format_retention_vector(self.retention_interval_by_group),
            self.retention_interval,
            _glm53_swa_interval,
            sorted(_glm53_min_exempt),
            sorted(self.block_pool.low_priority_cache_group_ids),
        )
"""

PRIOR_POLICY_NEW = """        self.block_pool.low_priority_cache_group_ids = (  # [glm53-apc-drafter-priority]
            frozenset(_glm53_min_exempt)
            if _glm53_swa_interval == 0
            else frozenset()
        )
        # [glm53-dflash-prior-policy-v1] A cache switch with a short fresh suffix
        # cannot rebuild the DFlash drafter's full 2048-token sliding-attention
        # window. The hybrid lookup backs up one scheduler boundary for that
        # replay. Under Mamba retention=0 the earlier target state would not
        # otherwise exist, so retain exactly one prior Mamba checkpoint. The
        # target KpoolTail is a separate 4-token request-local scratch group.
        self.dflash_replay_prior_group_ids = _glm53_dflash_prior_mamba_group_ids(
            kv_cache_config.kv_cache_groups,
            self.retention_interval_by_group,
            _glm53_is_hybrid,
        )
        for i in self.dflash_replay_prior_group_ids:
            self.single_type_managers[i]._glm53_retain_previous_dflash_boundary = True
        logger.info(  # [glm53-apc-per-group]
            "[glm53-apc-per-group] retention_by_group=%s "
            "(global=%s swa_env=%s eagle_min_exempt=%s low_priority=%s "
            "dflash_prior_mamba=%s)",
            _glm53_format_retention_vector(self.retention_interval_by_group),
            self.retention_interval,
            _glm53_swa_interval,
            sorted(_glm53_min_exempt),
            sorted(self.block_pool.low_priority_cache_group_ids),
            sorted(self.dflash_replay_prior_group_ids),
        )
"""

STM_REACHABLE_OLD = """        reachable_boundaries = [request.num_prompt_tokens - 1]
        if request.shared_prefix_boundary:
            reachable_boundaries.append(request.shared_prefix_boundary)
"""

STM_REACHABLE_NEW = """        reachable_boundaries = [request.num_prompt_tokens - 1]
        if request.shared_prefix_boundary:
            reachable_boundaries.append(request.shared_prefix_boundary)
        if getattr(  # [glm53-dflash-prior-manager-v1]
            self, "_glm53_retain_previous_dflash_boundary", False
        ):
            replay_boundary = (
                (request.num_prompt_tokens - 1)
                // self.scheduler_block_size
                * self.scheduler_block_size
            )
            previous_replay_boundary = replay_boundary - self.scheduler_block_size
            if previous_replay_boundary > 0:
                reachable_boundaries.append(previous_replay_boundary)
"""

BP_INIT_OLD = """        self.metrics_collector = metrics_collector
"""

BP_INIT_NEW = """        self.metrics_collector = metrics_collector
        # Populated only by the hybrid coordinator's explicit drafter policy.
        self.low_priority_cache_group_ids: frozenset[int] = (  # [glm53-apc-drafter-priority]
            frozenset()
        )
"""

BP_FREE_OLD = """        # Identify blocks with hash (LRU cache) and without it (never match APC)
        blocks_to_evict_last = []
        blocks_to_evict_first = []
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is None or not self.enable_caching:
                    # LIFO reuse of non-cached blocks for better GPU locality.
                    blocks_to_evict_first.append(block)
                else:
                    # FIFO reuse of cached blocks for LRU eviction behavior.
                    blocks_to_evict_last.append(block)

        # Blocks to reuse first are prepended to the front of the free queue.
        self.free_block_queue.prepend_n(blocks_to_evict_first)
        # Blocks to reuse last are appended to the end of the free queue.
        self.free_block_queue.append_n(blocks_to_evict_last)
"""

BP_FREE_NEW = """        # Identify blocks with hash (LRU cache) and without it (never match APC).
        # An explicitly min-exempt drafter group may be cached for immediate
        # replay while remaining lower priority than target MLA/Mamba state.
        blocks_to_evict_last = []
        blocks_to_evict_first = []
        blocks_to_evict_before_normal_cache = []  # [glm53-apc-drafter-priority]
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is None or not self.enable_caching:
                    # LIFO reuse of non-cached blocks for better GPU locality.
                    blocks_to_evict_first.append(block)
                else:
                    block_hashes = [block.block_hash]
                    block_hashes.extend(
                        self.cached_block_hashes_by_block.get(block.block_id, ())
                    )
                    if self.low_priority_cache_group_ids and all(
                        get_group_id(key) in self.low_priority_cache_group_ids
                        for key in block_hashes
                    ):
                        blocks_to_evict_before_normal_cache.append(block)
                    else:
                        # FIFO reuse of ordinary cached blocks for LRU behavior.
                        blocks_to_evict_last.append(block)

        # Low-priority cached blocks sit behind immediately reusable unhashed
        # blocks, but ahead of ordinary cached target state.
        self.free_block_queue.prepend_n(blocks_to_evict_before_normal_cache)
        self.free_block_queue.prepend_n(blocks_to_evict_first)
        self.free_block_queue.append_n(blocks_to_evict_last)
"""

# ------------------------------------------------------------------ apply ----


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def main() -> int:
    if not P.is_file() or not BP.is_file() or not STM.is_file():
        raise SystemExit(
            f"missing coordinator/block pool/single-type manager: {P} / {BP} / {STM}"
        )
    text = P.read_text()
    block_pool_text = BP.read_text()
    single_type_text = STM.read_text()
    if MARK not in text:
        needle = "def _validate_prefix_cache_retention_interval(\n"
        if text.count(needle) != 1:
            raise SystemExit(f"{P}: helper insert point not unique")

        if "\nimport os\n" not in text:
            text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import")
        if "def _glm53_inner_kv_spec(" not in text:
            text = text.replace(needle, BASE_HELPER + needle, 1)
        text = text.replace(needle, RETENTION_HELPER + needle, 1)

        text = replace_once(text, INIT_OLD, INIT_NEW, "retention-init")
        text = replace_once(text, BASE_CACHE_OLD, BASE_CACHE_NEW, "base-cache_blocks")
        text = replace_once(text, HYBRID_LOOP_OLD, HYBRID_LOOP_NEW, "hybrid-loop")
        text = replace_once(text, HYBRID_CACHE_OLD, HYBRID_CACHE_NEW, "hybrid-cache_blocks")

    # Versioned migration: images built with the reviewed per-group overlay
    # already carry MARK, but do not yet define the focused DFlash helper.
    if "def _glm53_dflash_prior_mamba_group_ids(" not in text:
        needle = "def _validate_prefix_cache_retention_interval(\n"
        if text.count(needle) != 1:
            raise SystemExit(f"{P}: DFlash helper insert point not unique")
        text = text.replace(needle, DFLASH_PRIOR_HELPER + needle, 1)

    if DIAG_MARK not in text:
        text = replace_once(text, DIAG_IMPORT_OLD, DIAG_IMPORT_NEW, "diagnostic import")
        text = replace_once(text, DIAG_FREE_OLD, DIAG_FREE_NEW, "diagnostic free")
    if DIAG_HIT_MARK not in text:
        text = replace_once(text, DIAG_HIT_OLD, DIAG_HIT_NEW, "diagnostic cache hit")
    if PRIORITY_MARK not in text:
        text = replace_once(text, PRIORITY_OLD, PRIORITY_NEW, "drafter priority config")
    if PRIOR_POLICY_MARK not in text:
        text = replace_once(
            text, PRIOR_POLICY_OLD, PRIOR_POLICY_NEW, "dflash prior-boundary policy"
        )
    if PRIOR_MANAGER_MARK not in single_type_text:
        n = single_type_text.count(STM_REACHABLE_OLD)
        if n != 1:
            raise SystemExit(
                f"{STM}: expected one reachable-boundary target, found {n}"
            )
        single_type_text = single_type_text.replace(
            STM_REACHABLE_OLD, STM_REACHABLE_NEW, 1
        )
    if BP_PRIORITY_MARK not in block_pool_text:
        block_pool_text = replace_once(
            block_pool_text, BP_INIT_OLD, BP_INIT_NEW, "block-pool priority init"
        )
        block_pool_text = replace_once(
            block_pool_text, BP_FREE_OLD, BP_FREE_NEW, "block-pool priority free"
        )

    # Canonical helper spacing shared with patch_hybrid_prefix_hit.py so the
    # rendered coordinator bytes do not depend on overlay application order.
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
    BP.write_text(block_pool_text)
    STM.write_text(single_type_text)
    print(
        f"patched {P.name} (per-group APC retention + temporary occupancy diagnostics; "
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA drives the DFlash2 drafter group)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
