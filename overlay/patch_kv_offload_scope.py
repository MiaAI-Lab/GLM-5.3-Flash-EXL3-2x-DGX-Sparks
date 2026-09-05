#!/usr/bin/env python3
"""Eligible-group scoping for the KV-offloading connector (port of vllm#54743).

The problem
-----------
``build_offloading_config`` (``vllm/distributed/kv_transfer/kv_connector/v1/
offloading/config.py``) builds an ``OffloadingGroupConfig`` for EVERY KV-cache
group and asserts ``tokens_per_block % tokens_per_hash == 0`` across all of
them, while ``resolve_kv_cache_block_sizes`` derives ``tokens_per_hash`` from
prefix-cacheable groups only. On this deployment the KpoolTailSpec scratch
group (group 1: block_size 4, ``prefix_cacheable=False``) fails the assert
(4 % 64) the moment the OffloadingConnector is enabled — the connector cannot
even boot on the hybrid layout.

Our upstream PR vllm-project/vllm#54743 (branch
fix/offloading-config-scope-prefix-cacheable, commit 899699c) fixes this at
HEAD by scoping the offload group list to prefix-cacheable groups while
preserving ORIGINAL group indices. This overlay ports that scoping onto the
image's older tree (vllm 0.1.dev20051+g487ecf187), which sits between the PR's
base and HEAD: the scheduler-side ``GroupOffloadConfig`` already carries
``group_idx``, but the config boundary does not, and every scheduler pairing
still assumes ``index == group_idx`` identity.

What this overlay does (three files)
------------------------------------
1. ``vllm/v1/kv_offload/config.py`` — ``OffloadingGroupConfig`` gains
   ``group_idx`` (original index into ``KVCacheConfig.kv_cache_groups``;
   trailing default ``-1`` keeps existing constructors valid, and the
   connector path always sets it).
2. ``.../offloading/config.py`` — the groups tuple is scoped to the eligible
   set: prefix-cacheable groups (upstream predicate) MINUS the fork policy
   exclusion (exact-``SlidingWindowSpec`` drafter groups, unless
   ``GLM53_KV_OFFLOAD_DRAFTER=1`` — plan §3.2/§11-C5: the DFlash2 drafter is
   min-exempt with retention 0, so restoring it is pointless and storing it is
   pure waste). Original indices ride in ``group_idx``; the divisibility
   assert still guards every eligible group; one boot log line names the
   eligible set. Final eligible set on this deployment: {0, 2, 3, 4, 5}.
3. ``.../offloading/scheduler.py`` — the group_idx pairing port, [I]-adapted
   from 899699c:
   - ``resolve_mamba_align_size`` / the full-attn alignment scan / the
     kv_group_configs build loop iterate ``spec.config.groups`` (group_idx +
     tokens_per_block from the entry) instead of ``enumerate(
     spec.tokens_per_block)``;
   - ``SchedulerOffloadConfig`` gains ``num_kv_cache_groups`` (the FULL group
     count) and ``from_spec`` fills it;
   - ``RequestOffloadState.group_states`` is sized by the full group count —
     block-id bookkeeping spans all groups (``update_block_id_groups`` asserts
     one entry per KV-cache group), while offload keys are only generated for
     eligible groups;
   - every ``zip(kv_group_configs, group_states)`` pairing indexes
     ``group_states[group_config.group_idx]`` instead;
   - the two ``kv_group_configs[group_idx]`` subscripts go through a new
     ``_group_config_by_idx`` dict;
   - ``update_state_after_alloc`` builds FULL-LENGTH ``group_sizes`` /
     ``block_indices`` (zeros for ineligible groups) and reads
     ``blocks.blocks[group_config.group_idx]`` — the worker's
     ``GPULoadStoreSpec`` layout is built from ALL kv-cache groups
     (``register_kv_caches`` iterates ``kv_cache_config.kv_cache_groups``), so
     ineligible groups keep zero-size entries and the worker layout is
     unchanged;
   - ``_build_store_jobs`` and ``_build_partial_tail_store_jobs`` write their
     group_sizes/block_indices at ``group_idx`` into full-length lists;
   - ``supports_partial_tail`` additionally requires every KV-cache group to
     be eligible (conservative opt-out, exactly HEAD's rule; on this layout it
     is already False via the EAGLE drafter).

Behaviour for models without scratch/policy-excluded groups is unchanged: the
eligible set is then every group and all the new indexing degenerates to the
old identity pairing.

Adaptation notes vs 899699c (per-anchor honesty)
------------------------------------------------
- [I] already has ``group_idx``/``is_eagle_group``/``requires_cow_source`` in
  the scheduler NamedTuple; only the PAIRING is ported, not the fields.
- HEAD's ``kv_connector_extra_config['block_size']`` assert-message reword is
  NOT ported: the branch is unused by this deployment and the semantics are
  already rescoped by the filtered ``groups`` tuple it reads.
- HEAD hoists the ``_current_batch_allocated_block_ids`` sweep over all
  groups in ``update_state_after_alloc``; [I]'s loop is shaped differently but
  the port keeps HEAD's semantics (sweep ALL groups' allocated ids, then walk
  only eligible groups).
- The fork policy exclusion (drafter) is fork-only and NOT part of #54743;
  upstream's predicate alone yields {0,2,3,4,5,6} here.

Conventions follow ``overlay/patch_kv_capacity_log.py``: pinned ANCHORs, MARK
sentinels, ``verified_state``, ``prepare``, idempotent, atomic replace, pyc
clear, drift => nonzero exit. ALL anchors in ALL three files are preflighted
before ANY file is written. Must run BEFORE ``patch_kv_offload_store_local.py``
(that overlay anchors on this one's output in scheduler.py). No anchors are
shared with any other overlay (none touch the kv_offload/connector tree).

Usage::

    python3 patch_kv_offload_scope.py              # apply
    python3 patch_kv_offload_scope.py --preflight  # validate anchors only
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

VLLM_ROOT = os.environ.get(
    "GLM53_VLLM_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"
)

TARGET_KVO_CONFIG = Path(
    os.environ.get("GLM53_KVO_CONFIG_PY", f"{VLLM_ROOT}/v1/kv_offload/config.py")
)
TARGET_CONN_CONFIG = Path(
    os.environ.get(
        "GLM53_KVO_CONN_CONFIG_PY",
        f"{VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/offloading/config.py",
    )
)
TARGET_SCHED = Path(
    os.environ.get(
        "GLM53_KVO_SCHED_PY",
        f"{VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
    )
)

TAG = "[glm53-kv-offload-scope]"
ENV_DRAFTER = "GLM53_KV_OFFLOAD_DRAFTER"


# ---------------------------------------------------------------------------
# Shared helper source: the eligibility predicate.
#
# Injected into offloading/config.py AND exec'd below so host tests drive the
# exact shipped predicate (one implementation, no drifting replica).
# ---------------------------------------------------------------------------
HELPERS_SRC = '''# [glm53-kv-offload-scope] helpers -- see overlay/patch_kv_offload_scope.py
_GLM53_KVO_SCOPE_TAG = "[glm53-kv-offload-scope]"
_GLM53_KVO_DRAFTER_ENV = "GLM53_KV_OFFLOAD_DRAFTER"


def _glm53_kvo_drafter_included() -> bool:
    """Exactly "0" or "1"; unset means "0" (exclude the drafter group).

    Same contract as the launcher's ``_glm53_validate_bool_flag``: "" is a
    value, not an absence; a typo'd knob raises rather than silently choosing
    which groups hit the disk tier.
    """
    import os as _os

    raw = _os.environ.get(_GLM53_KVO_DRAFTER_ENV)
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(
        f"{_GLM53_KVO_DRAFTER_ENV} must be exactly 0 or 1 (got: {raw!r})"
    )


def _glm53_kvo_unwrap_spec(spec):
    """Unwrap a UniformTypeKVCacheSpecs so the group's real spec class is named."""
    specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(specs, dict) and specs:
        return next(iter(specs.values()))
    return spec


def _glm53_kvo_group_eligible(
    kv_cache_group, drafter_included: bool, use_eagle: bool
) -> bool:
    """Upstream predicate (prefix_cacheable) minus the fork policy exclusion.

    - Non-prefix-cacheable groups (KpoolTailSpec scratch) are never offloaded:
      block hashes only cover prefix-cacheable groups, so a scratch group has
      no valid hash granularity (vllm#54743's predicate).
    - Fork policy: DRAFT-model attention groups are excluded unless
      GLM53_KV_OFFLOAD_DRAFTER=1 — the drafter is min-exempt with retention 0
      (#83), so its window is rebuilt by the remainder's prefill and storing
      it is waste (plan §3.2). A group is a drafter group when it carries
      ``is_eagle_group`` OR — only under an EAGLE-like speculative config
      (``use_eagle``) — when its unwrapped spec is EXACTLY SlidingWindowSpec
      (the same exact-class rule as patch_hybrid_prefix_hit's
      ``_glm53_is_draft_swa_spec``; without EAGLE context a SlidingWindowSpec
      group is genuine model SWA and stays eligible).
    """
    spec = kv_cache_group.kv_cache_spec
    # [I] names the flag participates_in_prefix_caching (a property, and on
    # UniformTypeKVCacheSpecs an all-members aggregate); newer trees say
    # prefix_cacheable. Check both, fork name first (same dual-name rule as
    # patch_kv_capacity_log). A spec with neither name caches (legacy).
    for name in ("participates_in_prefix_caching", "prefix_cacheable"):
        value = getattr(spec, name, None)
        if callable(value):
            value = value()
        if isinstance(value, bool):
            if not value:
                return False
            break
    if not drafter_included:
        if bool(getattr(kv_cache_group, "is_eagle_group", False)):
            return False
        if use_eagle:
            inner = _glm53_kvo_unwrap_spec(spec)
            if type(inner).__name__ == "SlidingWindowSpec":
                return False
    return True


def _glm53_kvo_use_eagle(vllm_config) -> bool:
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    if spec_cfg is None:
        return False
    use_eagle = getattr(spec_cfg, "use_eagle", None)
    if callable(use_eagle):
        return bool(use_eagle())
    return bool(use_eagle)


'''


def load_helpers() -> dict:
    """Exec HELPERS_SRC into a fresh namespace (what the host tests drive)."""
    ns: dict = {}
    exec(compile(HELPERS_SRC, "<glm53-kv-offload-scope helpers>", "exec"), ns)
    return ns


_HELPERS = load_helpers()
kvo_drafter_included = _HELPERS["_glm53_kvo_drafter_included"]
kvo_unwrap_spec = _HELPERS["_glm53_kvo_unwrap_spec"]
kvo_group_eligible = _HELPERS["_glm53_kvo_group_eligible"]


# ===========================================================================
# File 1: vllm/v1/kv_offload/config.py — OffloadingGroupConfig.group_idx
# ===========================================================================
MARK_GROUPCFG = "    # [glm53-kv-offload-scope] original index into KVCacheConfig.kv_cache_groups\n"

ANCHOR_GROUPCFG = """@dataclass(frozen=True)
class OffloadingGroupConfig:
    # Total token span covered by one block across all workers
    # (accounts for context parallelism).
    tokens_per_block: int
    # Layer names belonging to this group.
    layer_names: tuple[str, ...]
"""

PATCHED_GROUPCFG = """@dataclass(frozen=True)
class OffloadingGroupConfig:
    # Total token span covered by one block across all workers
    # (accounts for context parallelism).
    tokens_per_block: int
    # Layer names belonging to this group.
    layer_names: tuple[str, ...]
    # [glm53-kv-offload-scope] original index into KVCacheConfig.kv_cache_groups
    # (offload groups are scoped to eligible groups; keys/layouts keep original
    # indices). Required, matching upstream 899699c: the only constructor is
    # build_offloading_config, which always sets it.
    group_idx: int
"""

SITES_KVO_CONFIG = (
    ("OffloadingGroupConfig.group_idx", MARK_GROUPCFG, ANCHOR_GROUPCFG, PATCHED_GROUPCFG),
)


# ===========================================================================
# File 2: offloading/config.py — eligible-group scoping at the boundary
# ===========================================================================
MARK_CONN_HELPERS = "# [glm53-kv-offload-scope] helpers -- see overlay/patch_kv_offload_scope.py\n"

ANCHOR_CONN_HELPERS = """def is_kv_cache_tensor_packed(kv_cache_tensor: "KVCacheTensor") -> bool:
"""

PATCHED_CONN_HELPERS = HELPERS_SRC + ANCHOR_CONN_HELPERS

MARK_CONN_GROUPS = "    # [glm53-kv-offload-scope] scope to eligible groups, keep original indices\n"

ANCHOR_CONN_GROUPS = """    parallel_config = vllm_config.parallel_config
    groups = tuple(
        OffloadingGroupConfig(
            tokens_per_block=(
                group.kv_cache_spec.block_size
                * (
                    parallel_config.decode_context_parallel_size
                    if isinstance(group.kv_cache_spec, AttentionSpec)
                    else 1
                )
            ),
            layer_names=tuple(group.layer_names),
        )
        for group in kv_cache_config.kv_cache_groups
    )
"""

PATCHED_CONN_GROUPS = """    parallel_config = vllm_config.parallel_config
    # [glm53-kv-offload-scope] scope to eligible groups, keep original indices
    # (port of vllm#54743): only prefix-cacheable groups can serve offload
    # hits -- block hashes are computed over prefix-cacheable groups only
    # (resolve_kv_cache_block_sizes), so scratch groups (KpoolTailSpec) have
    # no valid hash granularity and must not be offloaded. The fork policy
    # additionally excludes the exact-SlidingWindowSpec drafter group unless
    # GLM53_KV_OFFLOAD_DRAFTER=1. Original group indices ride in group_idx.
    _glm53_drafter_included = _glm53_kvo_drafter_included()
    _glm53_use_eagle = _glm53_kvo_use_eagle(vllm_config)
    _glm53_eligible = [
        group_idx
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
        if _glm53_kvo_group_eligible(
            group, _glm53_drafter_included, _glm53_use_eagle
        )
    ]
    from vllm.logger import init_logger as _glm53_init_logger

    _glm53_init_logger(__name__).info(
        "%s eligible groups: %s of %d (drafter_included=%s)",
        _GLM53_KVO_SCOPE_TAG,
        _glm53_eligible,
        len(kv_cache_config.kv_cache_groups),
        _glm53_drafter_included,
    )
    groups = tuple(
        OffloadingGroupConfig(
            tokens_per_block=(
                group.kv_cache_spec.block_size
                * (
                    parallel_config.decode_context_parallel_size
                    # Unwrap: [I] wraps per-layer specs in
                    # UniformTypeKVCacheSpecs (KVCacheSpec subclass only), so
                    # the stock isinstance would miss attention groups.
                    if isinstance(
                        _glm53_kvo_unwrap_spec(group.kv_cache_spec), AttentionSpec
                    )
                    else 1
                )
            ),
            layer_names=tuple(group.layer_names),
            group_idx=group_idx,
        )
        for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
        if group_idx in _glm53_eligible
    )
"""

SITES_CONN_CONFIG = (
    ("connector config helpers", MARK_CONN_HELPERS, ANCHOR_CONN_HELPERS, PATCHED_CONN_HELPERS),
    ("eligible-group scoping", MARK_CONN_GROUPS, ANCHOR_CONN_GROUPS, PATCHED_CONN_GROUPS),
)


# ===========================================================================
# File 3: offloading/scheduler.py — group_idx pairing port
# ===========================================================================

# --- site S0: spec-unwrap helper (an [I]-specific adaptation) ---------------
# [I]'s kv-cache groups wrap per-layer specs in UniformTypeKVCacheSpecs
# (subclass of KVCacheSpec only): the module's isinstance dispatch
# (get_sliding_window_size_in_chunks' FullAttentionSpec assert, the MambaSpec
# checks) would crash or silently misclassify every wrapped group. Not part
# of upstream 899699c (HEAD's group specs reach the connector unwrapped).
MARK_S0 = "# [glm53-kv-offload-scope] spec unwrap helper\n"

ANCHOR_S0 = """def get_sliding_window_size_in_chunks(
"""

S0_HELPER = '''# [glm53-kv-offload-scope] spec unwrap helper
def _glm53_kvo_inner_spec(spec):
    """Return the single-type member spec of a UniformTypeKVCacheSpecs.

    [I]'s kv-cache groups wrap their per-layer specs (mixed page sizes) in
    UniformTypeKVCacheSpecs, which subclasses only KVCacheSpec -- the stock
    isinstance() dispatch in this module would assert on it. Unwrap to the
    first member when every member is the same class; otherwise return the
    wrapper unchanged, so the caller's FullAttentionSpec assert fails LOUDLY
    at boot instead of misclassifying a mixed group.
    """
    specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(specs, dict) and specs:
        members = list(specs.values())
        first_cls = type(members[0])
        if all(type(m) is first_cls for m in members):
            return members[0]
    return spec


'''

PATCHED_S0 = S0_HELPER + ANCHOR_S0

# --- site S1: resolve_mamba_align_size iterates config.groups ---------------
MARK_S1 = "    # [glm53-kv-offload-scope] iterate eligible groups (original indices)\n"

ANCHOR_S1 = """    mamba_align_size: int | None = None
    for idx, tokens_per_block in enumerate(spec.tokens_per_block):
        kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode in (
            "align",
            "all",
        ):
            tokens_per_chunk = tokens_per_block * spec.blocks_per_chunk
            assert mamba_align_size is None or mamba_align_size == tokens_per_chunk
            mamba_align_size = tokens_per_chunk
    return mamba_align_size
"""

PATCHED_S1 = """    mamba_align_size: int | None = None
    # [glm53-kv-offload-scope] iterate eligible groups (original indices)
    for group in spec.config.groups:
        kv_spec = _glm53_kvo_inner_spec(
            kv_cache_config.kv_cache_groups[group.group_idx].kv_cache_spec
        )
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode in (
            "align",
            "all",
        ):
            tokens_per_chunk = group.tokens_per_block * spec.blocks_per_chunk
            assert mamba_align_size is None or mamba_align_size == tokens_per_chunk
            mamba_align_size = tokens_per_chunk
    return mamba_align_size
"""

# --- site S2: SchedulerOffloadConfig fields --------------------------------
MARK_S2 = "    # [glm53-kv-offload-scope] total number of KV cache groups\n"

ANCHOR_S2 = """class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    blocks_per_chunk: int
"""

PATCHED_S2 = """class SchedulerOffloadConfig(NamedTuple):
    # One entry per ELIGIBLE KV cache group; group_idx keeps the original
    # index into KVCacheConfig.kv_cache_groups. Ineligible groups (scratch,
    # policy-excluded drafter) are never offloaded and get no entry here.
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    # [glm53-kv-offload-scope] total number of KV cache groups
    # (including ineligible ones); sizes per-group structures shared with the
    # scheduler core and the worker.
    num_kv_cache_groups: int
    blocks_per_chunk: int
"""

# --- site S3: from_spec full-attn alignment scan ---------------------------
MARK_S3 = "        # [glm53-kv-offload-scope] alignment scan over eligible groups\n"

ANCHOR_S3 = """        full_attn_tokens_per_chunk: set[int] = set()
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * spec.blocks_per_chunk)
"""

PATCHED_S3 = """        full_attn_tokens_per_chunk: set[int] = set()
        # [glm53-kv-offload-scope] alignment scan over eligible groups
        for group in spec.config.groups:
            kv_spec = _glm53_kvo_inner_spec(
                kv_cache_config.kv_cache_groups[group.group_idx].kv_cache_spec
            )
            sw = get_sliding_window_size_in_chunks(
                kv_spec, group.tokens_per_block * spec.blocks_per_chunk
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(
                    group.tokens_per_block * spec.blocks_per_chunk
                )
"""

# --- site S4: from_spec kv_group_configs build loop ------------------------
MARK_S4 = "        # [glm53-kv-offload-scope] build configs for eligible groups only\n"

ANCHOR_S4 = """        kv_group_configs_list: list[GroupOffloadConfig] = []
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_cache_group = kv_cache_config.kv_cache_groups[idx]
            kv_spec = kv_cache_group.kv_cache_spec
"""

PATCHED_S4 = """        kv_group_configs_list: list[GroupOffloadConfig] = []
        # [glm53-kv-offload-scope] build configs for eligible groups only
        for group in spec.config.groups:
            idx = group.group_idx
            tokens_per_block = group.tokens_per_block
            kv_cache_group = kv_cache_config.kv_cache_groups[idx]
            kv_spec = _glm53_kvo_inner_spec(kv_cache_group.kv_cache_spec)
"""

# --- site S5: supports_partial_tail + num_kv_cache_groups ------------------
MARK_S5 = "        # [glm53-kv-offload-scope] partial tails need every group eligible\n"

ANCHOR_S5 = """        kv_group_configs = tuple(kv_group_configs_list)
        group_block_sizes = {config.tokens_per_block for config in kv_group_configs}
"""

PATCHED_S5 = """        kv_group_configs = tuple(kv_group_configs_list)
        # [glm53-kv-offload-scope] partial tails need every group eligible
        num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        group_block_sizes = {config.tokens_per_block for config in kv_group_configs}
"""

MARK_S5B = "            and len(kv_group_configs) == num_kv_cache_groups  # [glm53-kv-offload-scope]\n"

ANCHOR_S5B = """        supports_partial_tail = (
            spec.blocks_per_chunk == 1
            and len(group_block_sizes) == 1
            and has_partial_recurrent_group
"""

PATCHED_S5B = """        supports_partial_tail = (
            spec.blocks_per_chunk == 1
            and len(group_block_sizes) == 1
            and len(kv_group_configs) == num_kv_cache_groups  # [glm53-kv-offload-scope]
            and has_partial_recurrent_group
"""

MARK_S5C = "            num_kv_cache_groups=num_kv_cache_groups,  # [glm53-kv-offload-scope]\n"

ANCHOR_S5C = """        return cls(
            num_workers=vllm_config.parallel_config.world_size,
            kv_group_configs=kv_group_configs,
            blocks_per_chunk=spec.blocks_per_chunk,
"""

PATCHED_S5C = """        return cls(
            num_workers=vllm_config.parallel_config.world_size,
            kv_group_configs=kv_group_configs,
            num_kv_cache_groups=num_kv_cache_groups,  # [glm53-kv-offload-scope]
            blocks_per_chunk=spec.blocks_per_chunk,
"""

# --- site S6: RequestOffloadState.group_states full-length -----------------
MARK_S6 = "        # [glm53-kv-offload-scope] one state per KV cache group (original index)\n"

ANCHOR_S6 = """    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
"""

PATCHED_S6 = """    def __post_init__(self) -> None:
        # [glm53-kv-offload-scope] one state per KV cache group (original index)
        # -- block-id bookkeeping spans ALL groups (update_block_id_groups gets
        # one entry per KV cache group from the core), while offload keys are
        # only ever generated for the eligible kv_group_configs.
        self.group_states = tuple(
            RequestGroupState() for _ in range(self.config.num_kv_cache_groups)
        )
"""

# --- site S7: update_offload_keys pairing ----------------------------------
MARK_S7 = "        for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S7\n"

ANCHOR_S7 = """    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
"""

PATCHED_S7 = """    def update_offload_keys(self) -> None:
        for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S7
            group_state = self.group_states[group_config.group_idx]
"""

# --- site S8: advance_stored_idx pairing -----------------------------------
MARK_S8 = "        for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S8\n"

ANCHOR_S8 = """        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.next_stored_chunk_idx = max(
"""

PATCHED_S8 = """        for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S8
            group_state = self.group_states[group_config.group_idx]
            group_state.next_stored_chunk_idx = max(
"""

# --- site S9: update_num_hit_chunks pairing --------------------------------
MARK_S9 = "        for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S9\n"

ANCHOR_S9 = """        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_chunks = (
"""

PATCHED_S9 = """        for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S9
            group_state = self.group_states[group_config.group_idx]
            group_state.num_hit_chunks = (
"""

# --- site S10: scheduler __init__ dict + sort key --------------------------
MARK_S10 = "        # [glm53-kv-offload-scope] original group index -> eligible group config\n"

ANCHOR_S10 = """        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_chunks
            assert val is not None
            return val
"""

PATCHED_S10 = """        # [glm53-kv-offload-scope] original group index -> eligible group config
        self._group_config_by_idx: dict[int, GroupOffloadConfig] = {
            group_config.group_idx: group_config
            for group_config in self.config.kv_group_configs
        }

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self._group_config_by_idx[i].sliding_window_size_in_chunks
            assert val is not None
            return val
"""

# --- site S11: _touch pairing ----------------------------------------------
MARK_S11 = "        for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S11\n"

ANCHOR_S11 = """    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
"""

PATCHED_S11 = """    def _touch(self, req_status: RequestOffloadState):
        for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S11
            group_state = req_status.group_states[group_config.group_idx]
"""

# --- site S12: _lookup_complete_chunks subscript ---------------------------
MARK_S12 = "                group_config: GroupOffloadConfig = self._group_config_by_idx[  # [glm53-kv-offload-scope]\n"

ANCHOR_S12 = """            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
"""

PATCHED_S12 = """            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self._group_config_by_idx[  # [glm53-kv-offload-scope]
                    group_idx
                ]
"""

# --- site S13: _chunks_being_loaded pairing --------------------------------
MARK_S13 = "            for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S13\n"

ANCHOR_S13 = """        if self._chunks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
"""

PATCHED_S13 = """        if self._chunks_being_loaded:
            for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S13
                group_state = req_status.group_states[group_config.group_idx]
"""

# --- site S14: update_state_after_alloc ------------------------------------
MARK_S14 = "        # [glm53-kv-offload-scope] full-length per-group layout, eligible walk\n"

ANCHOR_S14 = """        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state, group_blocks in zip(
            self.config.kv_group_configs,
            req_status.group_states,
            blocks.blocks,
        ):
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            tokens_per_block = group_config.tokens_per_block
"""

PATCHED_S14 = """        # [glm53-kv-offload-scope] full-length per-group layout, eligible walk
        # (GPULoadStoreSpec group_sizes/block_indices are indexed by ORIGINAL
        # group index; ineligible groups never load, so their entries stay 0).
        for group_blocks in blocks.blocks:
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group (indexed by original group index)
        group_sizes: list[int] = [0] * self.config.num_kv_cache_groups
        block_indices: list[int] = [0] * self.config.num_kv_cache_groups
        for group_config in self.config.kv_group_configs:
            group_state = req_status.group_states[group_config.group_idx]
            group_blocks = blocks.blocks[group_config.group_idx]

            tokens_per_block = group_config.tokens_per_block
"""

MARK_S14B = "            group_sizes[group_config.group_idx] = num_pending_gpu_blocks  # [glm53-kv-offload-scope]\n"

ANCHOR_S14B = """            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)
"""

PATCHED_S14B = """            group_sizes[group_config.group_idx] = num_pending_gpu_blocks  # [glm53-kv-offload-scope]
            block_indices[group_config.group_idx] = num_locally_computed_gpu_blocks
"""

# --- site S15: _build_partial_tail_store_jobs ------------------------------
MARK_S15 = "            group_by_key = {  # [glm53-kv-offload-scope] original-index keyed\n"

ANCHOR_S15 = """            group_by_key = {key: idx for idx, key in enumerate(keys)}
            accepted_groups = [group_by_key[key] for key in store_output.keys_to_store]
            group_sizes = [0] * len(self.config.kv_group_configs)
            block_indices = [0] * len(self.config.kv_group_configs)
"""

PATCHED_S15 = """            group_by_key = {  # [glm53-kv-offload-scope] original-index keyed
                key: group.group_idx
                for group, key in zip(self.config.kv_group_configs, keys)
            }
            # GPULoadStoreSpec expects blocks ordered by group index.
            accepted_groups = sorted(
                group_by_key[key] for key in store_output.keys_to_store
            )
            group_sizes = [0] * self.config.num_kv_cache_groups
            block_indices = [0] * self.config.num_kv_cache_groups
"""

MARK_S15B = "            source_blocks = [  # [glm53-kv-offload-scope] via block_id_by_group\n"

ANCHOR_S15B = """            source_blocks = [block_ids[group_idx] for group_idx in accepted_groups]
"""

PATCHED_S15B = """            source_blocks = [  # [glm53-kv-offload-scope] via block_id_by_group
                _glm53_block_id_by_group[group_idx] for group_idx in accepted_groups
            ]
"""

MARK_S15C = "            _glm53_block_id_by_group = {  # [glm53-kv-offload-scope]\n"

ANCHOR_S15C = """            block_ids = [
                cow_blocks[group.group_idx]
                if group.group_idx in self._cow_source_groups
                else req_status.group_states[group.group_idx].block_ids[block_idx]
                for group in self.config.kv_group_configs
            ]
            assert all(block_id != 0 for block_id in block_ids)
"""

PATCHED_S15C = """            _glm53_block_id_by_group = {  # [glm53-kv-offload-scope]
                group.group_idx: (
                    cow_blocks[group.group_idx]
                    if group.group_idx in self._cow_source_groups
                    else req_status.group_states[group.group_idx].block_ids[block_idx]
                )
                for group in self.config.kv_group_configs
            }
            assert all(
                block_id != 0 for block_id in _glm53_block_id_by_group.values()
            )
"""

# --- site S16: _build_store_jobs first pairing (key filter loop) -----------
MARK_S16 = "            for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S16\n"

ANCHOR_S16 = """            new_offload_keys: list[OffloadKey] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
"""

PATCHED_S16 = """            new_offload_keys: list[OffloadKey] = []
            for group_config in self.config.kv_group_configs:  # [glm53-kv-offload-scope] S16
                group_state = req_status.group_states[group_config.group_idx]
"""

# --- site S17: _build_store_jobs source-block walk -------------------------
MARK_S17 = "            group_sizes = [0] * self.config.num_kv_cache_groups  # [glm53-kv-offload-scope]\n"

ANCHOR_S17 = """            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            fenced_block_ids: list[int] = []
            deferred_fence_block_ids: list[int] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
"""

PATCHED_S17 = """            group_sizes = [0] * self.config.num_kv_cache_groups  # [glm53-kv-offload-scope]
            block_indices = [0] * self.config.num_kv_cache_groups
            src_block_ids: list[int] = []
            fenced_block_ids: list[int] = []
            deferred_fence_block_ids: list[int] = []
            for group_config in self.config.kv_group_configs:
                group_state = req_status.group_states[group_config.group_idx]
"""

MARK_S17B = "                group_sizes[group_config.group_idx] = num_group_blocks  # [glm53-kv-offload-scope]\n"

ANCHOR_S17B = """                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
"""

PATCHED_S17B = """                group_sizes[group_config.group_idx] = num_group_blocks  # [glm53-kv-offload-scope]
                block_indices[group_config.group_idx] = start_gpu_block_idx or 0
"""

SITES_SCHED = (
    ("S0 spec unwrap helper", MARK_S0, ANCHOR_S0, PATCHED_S0),
    ("S1 resolve_mamba_align_size", MARK_S1, ANCHOR_S1, PATCHED_S1),
    ("S2 SchedulerOffloadConfig fields", MARK_S2, ANCHOR_S2, PATCHED_S2),
    ("S3 full-attn alignment scan", MARK_S3, ANCHOR_S3, PATCHED_S3),
    ("S4 kv_group_configs build loop", MARK_S4, ANCHOR_S4, PATCHED_S4),
    ("S5 num_kv_cache_groups", MARK_S5, ANCHOR_S5, PATCHED_S5),
    ("S5B supports_partial_tail", MARK_S5B, ANCHOR_S5B, PATCHED_S5B),
    ("S5C from_spec return", MARK_S5C, ANCHOR_S5C, PATCHED_S5C),
    ("S6 group_states sizing", MARK_S6, ANCHOR_S6, PATCHED_S6),
    ("S7 update_offload_keys", MARK_S7, ANCHOR_S7, PATCHED_S7),
    ("S8 advance_stored_idx", MARK_S8, ANCHOR_S8, PATCHED_S8),
    ("S9 update_num_hit_chunks", MARK_S9, ANCHOR_S9, PATCHED_S9),
    ("S10 scheduler init dict", MARK_S10, ANCHOR_S10, PATCHED_S10),
    ("S11 _touch", MARK_S11, ANCHOR_S11, PATCHED_S11),
    ("S12 lookup subscript", MARK_S12, ANCHOR_S12, PATCHED_S12),
    ("S13 chunks_being_loaded", MARK_S13, ANCHOR_S13, PATCHED_S13),
    ("S14 update_state_after_alloc", MARK_S14, ANCHOR_S14, PATCHED_S14),
    ("S14B alloc group sizes", MARK_S14B, ANCHOR_S14B, PATCHED_S14B),
    ("S15C partial-tail block ids", MARK_S15C, ANCHOR_S15C, PATCHED_S15C),
    ("S15 partial-tail group_by_key", MARK_S15, ANCHOR_S15, PATCHED_S15),
    ("S15B partial-tail source blocks", MARK_S15B, ANCHOR_S15B, PATCHED_S15B),
    ("S16 store-jobs key filter", MARK_S16, ANCHOR_S16, PATCHED_S16),
    ("S17 store-jobs block walk", MARK_S17, ANCHOR_S17, PATCHED_S17),
    ("S17B store-jobs group sizes", MARK_S17B, ANCHOR_S17B, PATCHED_S17B),
)


TARGETS = (
    ("kv_offload/config.py", TARGET_KVO_CONFIG, SITES_KVO_CONFIG),
    ("offloading/config.py", TARGET_CONN_CONFIG, SITES_CONN_CONFIG),
    ("offloading/scheduler.py", TARGET_SCHED, SITES_SCHED),
)


def verified_state(text: str, sites) -> bool:
    """Exact post-state: one mark per site, one patched block per site, and
    anchors survive only where the replacement itself contains them."""
    return all(
        text.count(mark) == 1
        and text.count(patched) == 1
        and text.count(anchor) == patched.count(anchor)
        for _name, mark, anchor, patched in sites
    )


def prepare(source: str, sites, label: str) -> tuple[str, str]:
    """Idempotent, fail-closed. Returns (text, action). Nothing written here.

    The already-present check counts MARK sentinels only (each exactly once):
    patch_kv_offload_store_local.py legitimately edits inside this overlay's
    patched scheduler regions afterwards, so full patched-block equality only
    holds between the two applications (and is still enforced post-patch)."""
    marks = sum(source.count(mark) for _n, mark, _a, _p in sites)
    if marks:
        if marks != len(sites) or any(
            source.count(mark) != 1 for _n, mark, _a, _p in sites
        ):
            raise ValueError(
                f"partial/inconsistent kv-offload-scope patch in {label} "
                f"(marks={marks}, expected {len(sites)}) -- refusing to touch "
                "a half-patched file"
            )
        return source, "already present"

    for name, _mark, anchor, _patched in sites:
        n = source.count(anchor)
        if n != 1:
            raise ValueError(
                f"pinned kv-offload-scope anchor '{name}' drifted in {label} "
                f"(found {n}, expected 1)"
            )
    out = source
    for _name, _mark, anchor, patched in sites:
        out = out.replace(anchor, patched, 1)
    if not verified_state(out, sites):
        raise ValueError(f"kv-offload-scope post-patch verification failed in {label}")
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kv-offload-scope.tmp")
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
    for pyc in cache.glob(f"{target.stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]

    if "--check-injected" in argv[1:]:
        # Host-side gate mode: compile the injected helper source standalone
        # (no target access needed) so a truncated/corrupted string literal
        # inside this file fails BEFORE the launcher stops a healthy pair.
        compile(HELPERS_SRC, "<glm53-kv-offload-scope helpers>", "exec")
        compile(S0_HELPER, "<glm53-kv-offload-scope unwrap>", "exec")
        load_helpers()
        print("patch_kv_offload_scope.py: injected sources compile OK")
        return 0

    # Preflight EVERY target before writing ANY (fail-closed: a drifted
    # scheduler must not leave a patched config.py behind).
    prepared: list[tuple[Path, str, str, str]] = []
    for label, target, sites in TARGETS:
        if not target.is_file():
            raise SystemExit(f"missing {target}")
        source = target.read_text()
        try:
            patched, action = prepare(source, sites, label)
        except ValueError as exc:
            raise SystemExit(f"kv-offload-scope preflight failed: {exc}") from exc
        compile(patched, str(target), "exec")
        prepared.append((target, source, patched, action))
        if preflight_only:
            print(f"{target.name}: kv-offload-scope preflight OK ({action})")

    if preflight_only:
        return 0

    for target, source, patched, action in prepared:
        if patched != source:
            replace_file(target, patched)
            clear_pyc(target)
        print(f"{target.name}: kv-offload-scope {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
