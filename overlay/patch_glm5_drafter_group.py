#!/usr/bin/env python3
"""Add a DFlash2 SlidingWindowSpec group to the GLM-5-Next KV layout.

Draft layers share MLA tensors at disjoint block ids, like Mamba groups.
An exact-fit draft page can use the ordinary contiguous view. Otherwise,
pad the draft page to the MLA page size; the default manager block is 64.

GLM53_DRAFT_KV_COMPACT=1 selects the largest 64-token-multiple divisor of
the MLA block that fits its physical page. This reduces draft block-id
demand without increasing prefix-cache alignment or changing cache dtype,
window length, tensor allocation, or the target's groups. Default: 0.
Compact pages are DFlash-only: a preflight at get_kv_cache_groups (every
grouping path, before exact-fit or padded selection) requires that all
sliding-window layers are the DFlash drafter's (speculative method plus
draft layer count), else boot fails. Under the flag,
patch_hybrid_prefix_hit.py looks the drafter group up ending exactly at the
reconciled prefix boundary, which relies on DFlash's per-position context KV.

Padded pages cannot be virtually split into smaller kernel blocks: their
physical stride applies once per manager block. Patch worker/utils.py to
reject unsupported backends during kernel selection, before allocation.
An exact-fit, unpadded page remains splittable.

Usage:
    python3 patch_glm5_drafter_group.py [--kv-file PATH] [--dry-run]

The worker file is resolved beside core/kv_cache_utils.py, in worker/utils.py.
Both files are preflighted before either is written; the guard is written
first. Reapplying is a no-op. Source drift fails before writing.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

DEFAULT_KV_FILE = (
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"
)

MARKER = "DFLASH2-DRAFTER-GROUP"

# ---------------------------------------------------------------------------
# Anchored edits. Every anchor must appear EXACTLY ONCE in the target file.
# ---------------------------------------------------------------------------

# -- _get_kv_cache_groups_glm5_next: partition drafter layers out ------------

EDIT_PARTITION_ANCHOR = """\
    attn_specs = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, (MambaSpec, KpoolTailSpec))
    }
    if not mamba_specs or not all(
        type(s) is MLAAttentionSpec for s in attn_specs.values()
    ):
        return None
"""

EDIT_PARTITION_NEW = """\
    # DFLASH2-DRAFTER-GROUP: a spec-decode drafter (DFlash2) adds plain
    # SlidingWindowSpec layers on top of the GLM-5-Next hybrid. Partition them
    # out (exact type: KpoolTailSpec subclasses SlidingWindowSpec) so they do
    # not disqualify the model from this fast path; they are appended as one
    # extra group below.
    draft_specs = {
        k: v for k, v in kv_cache_spec.items() if type(v) is SlidingWindowSpec
    }
    attn_specs = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, (MambaSpec, KpoolTailSpec))
        and type(v) is not SlidingWindowSpec
    }
    if not mamba_specs or not all(
        type(s) is MLAAttentionSpec for s in attn_specs.values()
    ):
        return None
"""

# -- _get_kv_cache_groups_glm5_next: build + append the drafter group --------

EDIT_GROUPS_RETURN_ANCHOR = """\
    mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for k, name in enumerate(mamba_specs):
        mamba_grouped_names[k % num_groups].append(name)
    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
    )
"""

EDIT_GROUPS_RETURN_NEW = """\
    mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for k, name in enumerate(mamba_specs):
        mamba_grouped_names[k % num_groups].append(name)

    # Drafter group (DFLASH2-DRAFTER-GROUP): one extra group for the spec-
    # decode drafter's SlidingWindowSpec layers, appended LAST so existing
    # group ids stay stable. Exact-fit pages permit virtual splitting;
    # padded pages require matching manager and kernel block sizes.
    draft_group = None
    if draft_specs:
        any_draft = next(iter(draft_specs.values()))
        assert all(spec == any_draft for spec in draft_specs.values()), (
            "drafter SlidingWindowSpec layers must share one spec"
        )
        draft_bytes_per_token = any_draft.page_size_bytes // any_draft.block_size
        mla_block = mla_specs[mla_names[0]].block_size
        fit_block = (
            mla_page // draft_bytes_per_token
            if mla_page % draft_bytes_per_token == 0
            else 0
        )
        if (
            fit_block
            # A 64-divisible manager block is divisible by every int kernel
            # block size the SWA backends register (16/32/64), so
            # select_common_block_size always finds a clean split.
            and fit_block % 64 == 0
            # Keep resolve_kv_cache_block_sizes' scheduler LCM at
            # max(mla_block, fit_block) instead of exploding.
            and (fit_block % mla_block == 0 or mla_block % fit_block == 0)
            and len(draft_specs) <= len(mla_names)
        ):
            # EXACT FIT: the drafter's real page equals the MLA page, so
            # drafter layer i co-owns MLA tensor i at disjoint block ids
            # (like mamba) with a contiguous view: kernel block j of manager
            # block b lands at b * mla_page + j * kernel_page, inside block
            # b's own page. Per-block pool cost unchanged.
            logger.info(
                "DFlash2 drafter KV: exact-fit block=%d mla_page=%d",
                fit_block,
                mla_page,
            )
            new_draft_specs: dict[str, KVCacheSpec] = {
                name: replace(s, block_size=fit_block)
                for name, s in draft_specs.items()
            }
        else:
            # PADDED SLOT-SHARE: 656 vs 4096 cannot exact-fill on this MLA
            # block. Manager 64 matches the SWA kernel, so padding the page
            # to mla_page is a safe strided view (boot 8 OOB was kernel 64
            # inside a 2304-token manager). Layer i co-owns MLA tensor i.
            compact_block = 64
            logger.info(
                "DFlash2 drafter KV: padded slot-share block=%d "
                "mla_page=%d (was block=%d); exact-fit page mismatch "
                "draft_bytes/token=%d",
                compact_block,
                mla_page,
                any_draft.block_size,
                draft_bytes_per_token,
            )
            new_draft_specs = {
                name: replace(
                    s,
                    block_size=compact_block,
                    page_size_padded=mla_page,
                )
                for name, s in draft_specs.items()
            }
        draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)
        assert draft_uniform is not None
        draft_group = KVCacheGroupSpec(list(new_draft_specs), draft_uniform)

    return (
        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
        + ([draft_group] if draft_group is not None else [])
    )
"""

# -- _glm5_next_tensor_layout: return-type annotation ------------------------

EDIT_LAYOUT_ANNOT_ANCHOR = """\
        list[str],
        int,
    ]
    | None
):
"""

EDIT_LAYOUT_ANNOT_NEW = """\
        list[str],
        int,
        KVCacheGroupSpec | None,
    ]
    | None
):
"""

# -- _glm5_next_tensor_layout: docstring Returns -----------------------------

EDIT_LAYOUT_DOC_ANCHOR = """\
      - (attn_group, mamba_groups, mla_names, idx_names, mla_page, idx_page,
         tail_names, tail_page)
"""

EDIT_LAYOUT_DOC_NEW = """\
      - (attn_group, mamba_groups, mla_names, idx_names, mla_page, idx_page,
         tail_names, tail_page, draft_group)
"""

# -- _glm5_next_tensor_layout: detect the drafter group ----------------------

EDIT_LAYOUT_DETECT_ANCHOR = """\
    attn_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    for g in uniform_groups:
        group_inner = cast(UniformTypeKVCacheSpecs, g.kv_cache_spec).kv_cache_specs
        if all(type(s) is MLAAttentionSpec for s in group_inner.values()):
            attn_group = g
        elif all(isinstance(s, KpoolTailSpec) for s in group_inner.values()):
            tail_group = g
"""

EDIT_LAYOUT_DETECT_NEW = """\
    attn_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    draft_group: KVCacheGroupSpec | None = None
    for g in uniform_groups:
        group_inner = cast(UniformTypeKVCacheSpecs, g.kv_cache_spec).kv_cache_specs
        if all(type(s) is MLAAttentionSpec for s in group_inner.values()):
            attn_group = g
        elif all(isinstance(s, KpoolTailSpec) for s in group_inner.values()):
            tail_group = g
        elif group_inner and all(
            type(s) is SlidingWindowSpec for s in group_inner.values()
        ):
            # DFLASH2-DRAFTER-GROUP: the spec-decode drafter's SWA group
            # (validated below once mla_page is known).
            draft_group = g
"""

# -- _glm5_next_tensor_layout: validate the drafter group --------------------

EDIT_LAYOUT_VALIDATE_ANCHOR = """\
    if any(g.kv_cache_spec.page_size_bytes != mla_page for g in mamba_groups):
        return None
    tail_names: list[str] = []
"""

EDIT_LAYOUT_VALIDATE_NEW = """\
    if any(g.kv_cache_spec.page_size_bytes != mla_page for g in mamba_groups):
        return None
    if draft_group is not None:
        # DFLASH2-DRAFTER-GROUP: one uniform page across drafter layers.
        # Padded slot-sharing requires an unsplit manager block;
        # worker-side kernel selection checks the actual backend.
        # page == mla_page means slot-sharing of the MLA tensors
        # (needs one tensor per drafter layer); any other page means
        # standalone drafter tensors.
        draft_inner = cast(
            UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
        ).kv_cache_specs
        draft_pages = {s.page_size_bytes for s in draft_inner.values()}
        if len(draft_pages) != 1:
            return None
        if any(s.page_size_padded is not None for s in draft_inner.values()):
            if any(
                s.block_size != 64 or s.page_size_padded != mla_page
                for s in draft_inner.values()
            ):
                return None
        if (
            draft_pages.pop() == mla_page
            and len(draft_group.layer_names) > len(mla_names)
        ):
            return None
    tail_names: list[str] = []
"""

# -- _glm5_next_tensor_layout: return the drafter group ----------------------

EDIT_LAYOUT_RETURN_ANCHOR = """\
    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_pages.pop(),
        tail_names,
        tail_page,
    )
"""

EDIT_LAYOUT_RETURN_NEW = """\
    return (
        attn_group,
        mamba_groups,
        mla_names,
        idx_names,
        mla_page,
        idx_pages.pop(),
        tail_names,
        tail_page,
        draft_group,
    )
"""

# -- _pool_bytes_per_block: 9-tuple + standalone drafter bytes ---------------

EDIT_POOL_BYTES_ANCHOR = """\
        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5
        return len(mla_names) * mla_page + len(idx_names) * idx_page
"""

EDIT_POOL_BYTES_NEW = """\
        # DFLASH2-DRAFTER-GROUP: an exact-fit drafter (page == mla_page)
        # slot-shares the MLA tensors and adds no bytes; a standalone drafter
        # adds one page per drafter layer.
        _, _, mla_names, idx_names, mla_page, idx_page, _, _, draft_group = glm5
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_group is not None:
            draft_page = next(
                iter(
                    cast(
                        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                    ).kv_cache_specs.values()
                )
            ).page_size_bytes
            if draft_page != mla_page:
                per_block += len(draft_group.layer_names) * draft_page
        return per_block
"""

# -- get_kv_cache_config_from_groups: destructure + drafter mode -------------

EDIT_CONFIG_DESTRUCTURE_ANCHOR = """\
        (
            _,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
        ) = glm5n
"""

EDIT_CONFIG_DESTRUCTURE_NEW = """\
        (
            _,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
            draft_group,
        ) = glm5n
        draft_names: list[str] = []
        draft_page = 0
        draft_shared = False
        if draft_group is not None:
            draft_names = list(draft_group.layer_names)
            draft_page = next(
                iter(
                    cast(
                        UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
                    ).kv_cache_specs.values()
                )
            ).page_size_bytes
            # Exact fit: the drafter's real page equals the MLA page, so it
            # rides the MLA tensors; otherwise it gets standalone tensors.
            draft_shared = draft_page == mla_page
"""

# -- get_kv_cache_config_from_groups: per-block cost (standalone mode) -------

EDIT_CONFIG_PER_BLOCK_ANCHOR = """\
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        num_blocks = available_memory // per_block
"""

EDIT_CONFIG_PER_BLOCK_NEW = """\
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_names and not draft_shared:
            # DFLASH2-DRAFTER-GROUP (standalone): drafter tensors are part of
            # every block's byte cost.
            per_block += len(draft_names) * draft_page
        num_blocks = available_memory // per_block
"""

# -- get_kv_cache_config_from_groups: drafter co-owns MLA tensor i -----------

EDIT_CONFIG_SHARED_BY_ANCHOR = """\
                shared_by=[mla_name]
                + [g.layer_names[i] for g in mamba_groups if i < len(g.layer_names)],
            )
            for i, mla_name in enumerate(mla_names)
"""

EDIT_CONFIG_SHARED_BY_NEW = """\
                shared_by=[mla_name]
                + [g.layer_names[i] for g in mamba_groups if i < len(g.layer_names)]
                # DFLASH2-DRAFTER-GROUP (exact fit): drafter layer i rides MLA
                # tensor i (contiguous view, disjoint block ids), like mamba.
                + ([draft_names[i]] if draft_shared and i < len(draft_names) else []),
            )
            for i, mla_name in enumerate(mla_names)
"""

# -- get_kv_cache_config_from_groups: standalone drafter tensors -------------

EDIT_CONFIG_DRAFT_TENSORS_ANCHOR = """\
            KVCacheTensor(
                size=idx_page * num_blocks,
                shared_by=(
                    [idx_names[i], tail_names[i]] if tail_names else [idx_names[i]]
                ),
            )
            for i in range(len(idx_names))
        ]
"""

EDIT_CONFIG_DRAFT_TENSORS_NEW = """\
            KVCacheTensor(
                size=idx_page * num_blocks,
                shared_by=(
                    [idx_names[i], tail_names[i]] if tail_names else [idx_names[i]]
                ),
            )
            for i in range(len(idx_names))
        ] + [
            # DFLASH2-DRAFTER-GROUP (standalone): compact per-layer drafter
            # tensors; contiguous reshape, safe under kernel block splitting.
            KVCacheTensor(size=draft_page * num_blocks, shared_by=[name])
            for name in ([] if draft_shared else draft_names)
        ]
"""

# -- _max_memory_usage_bytes_from_groups: destructure ------------------------

EDIT_MAXMEM_DESTRUCTURE_ANCHOR = """\
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
        ) = glm5n
"""

EDIT_MAXMEM_DESTRUCTURE_NEW = """\
        (
            attn_group,
            mamba_groups,
            mla_names,
            idx_names,
            mla_page,
            idx_page,
            tail_names,
            _tail_page,
            draft_group,
        ) = glm5n
"""

# -- _max_memory_usage_bytes_from_groups: drafter demand + per-block ---------

EDIT_MAXMEM_BLOCKS_ANCHOR = """\
        if tail_names:
            # Tail: 1 block/req (KpoolTailSpec.max_admission_blocks_per_request
            # == 1), drawn from the shared pool.
            blocks_needed += 1
        return blocks_needed * (len(mla_names) * mla_page + len(idx_names) * idx_page)
"""

EDIT_MAXMEM_BLOCKS_NEW = """\
        if tail_names:
            # Tail: 1 block/req (KpoolTailSpec.max_admission_blocks_per_request
            # == 1), drawn from the shared pool.
            blocks_needed += 1
        per_block = len(mla_names) * mla_page + len(idx_names) * idx_page
        if draft_group is not None:
            # DFLASH2-DRAFTER-GROUP: charge the drafter's window-bounded
            # block-id demand; a standalone drafter also adds its pages to
            # every block's byte cost (an exact-fit one rides the MLA
            # tensors and adds none).
            draft_uniform = draft_group.kv_cache_spec
            assert isinstance(draft_uniform, UniformTypeKVCacheSpecs)
            blocks_needed += draft_uniform.max_memory_usage_pages(vllm_config)
            draft_page = next(
                iter(draft_uniform.kv_cache_specs.values())
            ).page_size_bytes
            if draft_page != mla_page:
                per_block += len(draft_group.layer_names) * draft_page
        return blocks_needed * per_block
"""

EDITS: list[tuple[str, str, str]] = [
    (
        "groups: partition drafter SlidingWindowSpec layers out",
        EDIT_PARTITION_ANCHOR,
        EDIT_PARTITION_NEW,
    ),
    (
        "groups: build + append drafter group (exact-fit / standalone)",
        EDIT_GROUPS_RETURN_ANCHOR,
        EDIT_GROUPS_RETURN_NEW,
    ),
    (
        "layout: return-type annotation gains draft_group",
        EDIT_LAYOUT_ANNOT_ANCHOR,
        EDIT_LAYOUT_ANNOT_NEW,
    ),
    (
        "layout: docstring Returns gains draft_group",
        EDIT_LAYOUT_DOC_ANCHOR,
        EDIT_LAYOUT_DOC_NEW,
    ),
    (
        "layout: detect drafter SWA uniform group",
        EDIT_LAYOUT_DETECT_ANCHOR,
        EDIT_LAYOUT_DETECT_NEW,
    ),
    (
        "layout: validate drafter (uniform page, never padded)",
        EDIT_LAYOUT_VALIDATE_ANCHOR,
        EDIT_LAYOUT_VALIDATE_NEW,
    ),
    (
        "layout: return draft_group (9th element)",
        EDIT_LAYOUT_RETURN_ANCHOR,
        EDIT_LAYOUT_RETURN_NEW,
    ),
    (
        "_pool_bytes_per_block: standalone drafter bytes",
        EDIT_POOL_BYTES_ANCHOR,
        EDIT_POOL_BYTES_NEW,
    ),
    (
        "config: destructure + drafter mode",
        EDIT_CONFIG_DESTRUCTURE_ANCHOR,
        EDIT_CONFIG_DESTRUCTURE_NEW,
    ),
    (
        "config: per-block cost includes standalone drafter",
        EDIT_CONFIG_PER_BLOCK_ANCHOR,
        EDIT_CONFIG_PER_BLOCK_NEW,
    ),
    (
        "config: exact-fit drafter layer i co-owns MLA tensor i",
        EDIT_CONFIG_SHARED_BY_ANCHOR,
        EDIT_CONFIG_SHARED_BY_NEW,
    ),
    (
        "config: standalone drafter tensors",
        EDIT_CONFIG_DRAFT_TENSORS_ANCHOR,
        EDIT_CONFIG_DRAFT_TENSORS_NEW,
    ),
    (
        "max-mem: destructure gains draft_group",
        EDIT_MAXMEM_DESTRUCTURE_ANCHOR,
        EDIT_MAXMEM_DESTRUCTURE_NEW,
    ),
    (
        "max-mem: charge drafter block-id demand + standalone bytes",
        EDIT_MAXMEM_BLOCKS_ANCHOR,
        EDIT_MAXMEM_BLOCKS_NEW,
    ),
]


def _prepare_group(text: str, path: str) -> str:
    if MARKER in text:
        v4_old = (
            "        if any(s.page_size_padded is not None for s in draft_inner.values()):\n"
            "            return None\n"
        )
        v4_new = (
            "        if any(s.page_size_padded is not None for s in draft_inner.values()):\n"
            "            # Padded pages require an unsplit manager block;\n"
            "            # worker-side kernel selection checks the backend.\n"
            "            if any(\n"
            "                s.block_size != 64 or s.page_size_padded != mla_page\n"
            "                for s in draft_inner.values()\n"
            "            ):\n"
            "                return None\n"
        )
        v3_marker = "padded slot-share block=%d"
        if v4_old in text:
            text = text.replace(v4_old, v4_new, 1)
            if v3_marker in text:
                return text
            # v3 grouping not yet present; keep going with mutated text.
        elif v3_marker in text:
            return text

        new_padded = (
            "            # PADDED SLOT-SHARE: 656 vs 4096 cannot exact-fill on this MLA\n"
            "            # block. Manager 64 matches the SWA kernel, so padding the page\n"
            "            # to mla_page is a safe strided view (boot 8 OOB was kernel 64\n"
            "            # inside a 2304-token manager). Layer i co-owns MLA tensor i.\n"
            "            compact_block = 64\n"
            "            logger.info(\n"
            "                \"DFlash2 drafter KV: padded slot-share block=%d \"\n"
            "                \"mla_page=%d (was block=%d); exact-fit page mismatch \"\n"
            "                \"draft_bytes/token=%d\",\n"
            "                compact_block,\n"
            "                mla_page,\n"
            "                any_draft.block_size,\n"
            "                draft_bytes_per_token,\n"
            "            )\n"
            "            new_draft_specs = {\n"
            "                name: replace(\n"
            "                    s,\n"
            "                    block_size=compact_block,\n"
            "                    page_size_padded=mla_page,\n"
            "                )\n"
            "                for name, s in draft_specs.items()\n"
            "            }\n"
        )

        # v3: compact-64 standalone already present (GHCR image + v2).
        v2_compact = (
            "            compact_block = 64\n"
            "            if any_draft.block_size > compact_block:\n"
        )
        if v2_compact in text:
            start = text.find("            # STANDALONE: compact per-layer tensors.")
            end = text.find("        draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)")
            if start < 0 or end < 0 or end <= start:
                raise AssertionError(
                    f"{path}: {MARKER} + compact_block present but cannot "
                    "locate standalone block for padded slot-share v3"
                )
            text = text[:start] + new_padded + text[end:]
            return text

        # v2: shrink standalone DFlash pages off the 1152 MLA manager block.
        old_standalone = (
            "            # STANDALONE: the drafter's geometry cannot exactly fill the MLA\n"
            "            # page; keep its spec as-is and give its layers compact tensors\n"
            "            # of their own (emitted in get_kv_cache_config_from_groups and\n"
            "            # charged in the per-block cost).\n"
            "            new_draft_specs = dict(draft_specs)\n"
        )
        if old_standalone not in text:
            raise AssertionError(
                f"{path}: {MARKER} present but neither padded slot-share, "
                "compact-64, nor keep-as-is standalone block found"
            )
        text = text.replace(old_standalone, new_padded, 1)
        return text

    # Sanity: the file we expect (guards against pointing at the wrong tree).
    for required in (
        "def _get_kv_cache_groups_glm5_next",
        "def _glm5_next_tensor_layout",
        "def _pool_bytes_per_block",
        "SlidingWindowSpec",
        "UniformTypeKVCacheSpecs",
    ):
        assert required in text, (
            f"ANCHOR PRECHECK FAILED: {required!r} not found in kv_cache_utils.py"
        )

    for name, anchor, replacement in EDITS:
        n = text.count(anchor)
        assert n == 1, (
            f"ANCHOR FAILED for edit [{name}]: expected exactly 1 occurrence, "
            f"found {n}. The upstream file has drifted -- re-derive the anchor "
            f"before building.\n--- anchor ---\n{anchor}\n--------------"
        )
        text = text.replace(anchor, replacement, 1)

    return text


COMPACT_BLOCK_HELPER = '''\
def _glm53_draft_kv_compact(vllm_config, kv_cache_spec) -> bool:
    """GLM53_DRAFT_KV_COMPACT preflight; runs on every grouping path.

    Compact pages are DFlash-only: under the flag the prefix-cache
    coordinator verifies the drafter's window ending exactly at the
    reconciled boundary, which is valid only because DFlash context KV at a
    position depends on nothing after that position. So with the flag on,
    every exact SlidingWindowSpec layer must be positively a DFlash draft
    layer (speculative method plus one layer per draft decoder layer), or
    boot fails before any group, exact-fit or padded, is chosen.
    """
    mode = os.environ.get("GLM53_DRAFT_KV_COMPACT", "0")
    if mode not in ("0", "1"):
        raise ValueError("GLM53_DRAFT_KV_COMPACT must be 0 or 1")
    if mode == "0":
        return False
    swa_layers = sum(type(s) is SlidingWindowSpec for s in kv_cache_spec.values())
    spec_config = vllm_config.speculative_config
    if swa_layers and (
        spec_config is None
        or not spec_config.use_dflash()
        or swa_layers != spec_config.draft_model_config.hf_config.num_hidden_layers
    ):
        raise ValueError(
            "GLM53_DRAFT_KV_COMPACT=1 requires the DFlash drafter to own "
            "every sliding-window layer"
        )
    return True


def _glm53_draft_block_size(
    mla_block: int, mla_page: int, bytes_per_token: int, compact: bool
) -> int:
    """Largest 64-token-multiple divisor of the MLA block whose page fits.

    Dividing the MLA block keeps the prefix-cache alignment (the LCM) at the
    MLA block; fitting the page keeps the padded strided view inside its own
    slot; a 64-multiple satisfies every kernel block size the SWA backends
    accept here (prepare_kernel_block_sizes rejects a split padded page). A
    larger block cuts block-id demand: a request holds
    cdiv(window - 1 + in_flight, block) + 1 live ids and a cached boundary
    keeps cdiv(window - 1, block) ids. Off keeps the 64-token page.
    """
    if not compact:
        return 64
    if (
        mla_block <= 0 or mla_block % 64
        or bytes_per_token <= 0 or mla_page < 64 * bytes_per_token
    ):
        raise ValueError("DFlash2 compact KV requires a 64-token page that fits MLA")
    limit = min(mla_block, mla_page // bytes_per_token)
    for block in range(limit // 64 * 64, 0, -64):
        if mla_block % block == 0:
            return block
    raise AssertionError("64 must divide the validated MLA block")


'''

COMPACT_SELECTION_OLD = """\
            # PADDED SLOT-SHARE: 656 vs 4096 cannot exact-fill on this MLA
            # block. Manager 64 matches the SWA kernel, so padding the page
            # to mla_page is a safe strided view (boot 8 OOB was kernel 64
            # inside a 2304-token manager). Layer i co-owns MLA tensor i.
            compact_block = 64
"""
COMPACT_SELECTION_NEW = """\
            # Layer i shares MLA tensor i at disjoint block ids. Padded
            # pages must not be split; prepare_kernel_block_sizes checks
            # the actual attention backends before any cache is allocated.
            # The DFlash-only preflight already ran at get_kv_cache_groups.
            compact_block = _glm53_draft_block_size(
                mla_block,
                mla_page,
                draft_bytes_per_token,
                _glm53_draft_kv_compact(vllm_config, kv_cache_spec),
            )
"""
COMPACT_PREFLIGHT_OLD = """\
        The generated KVCacheGroups
    \"\"\"
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)
"""
COMPACT_PREFLIGHT_NEW = """\
        The generated KVCacheGroups
    \"\"\"
    # Fail closed before any grouping path (uniform, DeepseekV4, GLM-5-Next,
    # generic) can build a sliding-window group the coordinator would treat
    # as a DFlash drafter under GLM53_DRAFT_KV_COMPACT=1.
    _glm53_draft_kv_compact(vllm_config, kv_cache_spec)
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(kv_cache_spec)
"""
COMPACT_VALIDATION_OLD = """\
                s.block_size != 64 or s.page_size_padded != mla_page
"""
COMPACT_VALIDATION_NEW = """\
                s.block_size <= 0 or s.block_size % 64
                or attn_uniform.block_size % s.block_size
                or s.page_size_padded != mla_page
                or s.real_page_size_bytes > mla_page
"""
KERNEL_GUARD_OLD = """\
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)
"""
KERNEL_GUARD_NEW = """\
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            # A padded page has one physical stride per manager block.
            # Splitting it applies that stride to each kernel block (OOB).
            if (
                type(kv_cache_spec) is SlidingWindowSpec
                and kv_cache_spec.page_size_padded is not None
                and selected_kernel_size != kv_manager_block_size
            ):
                raise ValueError(
                    "DFlash2 padded KV pages cannot be split: "
                    f"manager block {kv_manager_block_size}, "
                    f"kernel block {selected_kernel_size}. "
                    "Use a backend supporting the full manager block "
                    "or disable GLM53_DRAFT_KV_COMPACT."
                )
            kernel_block_sizes.append(selected_kernel_size)
"""


def _replace_once(text: str, old: str, new: str) -> str:
    new_count = text.count(new)
    if new_count == 1 and text.count(old) == new.count(old):
        return text
    if new_count or text.count(old) != 1:
        raise AssertionError(f"Expected one patch anchor:\n{old}")
    return text.replace(old, new, 1)


def patch_file(path: str, dry_run: bool = False) -> int:
    kv_path = Path(path)
    worker_path = kv_path.parent.parent / "worker" / "utils.py"
    kv_source = kv_path.read_text()
    worker_source = worker_path.read_text()
    text = _prepare_group(kv_source, path)
    helper_anchor = "def _get_kv_cache_groups_glm5_next("
    if (
        "def _glm53_draft_block_size(" in text or "def _glm53_draft_kv_compact(" in text
    ) and COMPACT_BLOCK_HELPER not in text:
        raise AssertionError("DFlash2 compact block helper has drifted")
    text = _replace_once(text, helper_anchor, COMPACT_BLOCK_HELPER + helper_anchor)
    text = _replace_once(text, COMPACT_SELECTION_OLD, COMPACT_SELECTION_NEW)
    text = _replace_once(text, COMPACT_PREFLIGHT_OLD, COMPACT_PREFLIGHT_NEW)
    text = _replace_once(text, COMPACT_VALIDATION_OLD, COMPACT_VALIDATION_NEW)
    worker = _replace_once(
        worker_source,
        "    MambaSpec,\n    UniformTypeKVCacheSpecs,",
        "    MambaSpec,\n    SlidingWindowSpec,\n    UniformTypeKVCacheSpecs,",
    )
    worker = _replace_once(worker, KERNEL_GUARD_OLD, KERNEL_GUARD_NEW)
    # Preflight both files before writing either. A missing worker guard must
    # never leave a newly enabled larger page on disk.
    updates = ((worker_path, worker_source, worker), (kv_path, kv_source, text))
    for target, _, replacement in updates:
        ast.parse(replacement, filename=str(target))
    for target, original, replacement in updates:
        if original == replacement:
            continue
        if not dry_run:
            target.write_text(replacement)
        print(f"[patch_glm5_drafter_group] {'DRY RUN ' if dry_run else ''}{target}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kv-file", default=DEFAULT_KV_FILE)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate anchors + parse, write nothing",
    )
    args = ap.parse_args()
    return patch_file(args.kv_file, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
