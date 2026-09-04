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
MARK = "# [glm53-apc-per-group]"

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

# ------------------------------------------------------------------ apply ----


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text:
        print(f"{P.name}: {MARK} already present - skipping")
        return 0

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

    P.write_text(text)
    print(
        f"patched {P.name} (per-group APC retention; "
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA drives the DFlash2 drafter group)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
