#!/usr/bin/env python3
"""Exclude non-prefix-cachable scratch groups from LMCache's KV layer groups.

THE BUG (measured on GLM-5.3-Flash EXL3)

`build_engine_group_infos()` in lmcache/integration/vllm/kv_cache_groups.py maps
every registered layer to its vLLM engine group. It marks layers absent from all
groups as `EXCLUDED_ENGINE_GROUP` (cross-layer KV-sharing layers), but it assigns
an ordinary engine group id to **every** group that vLLM declares, including
scratch/ring groups that carry no reusable prefix state.

GLM-5.3-Flash's group 1 is exactly that: 12 `KpoolTailSpec` layers. Measured on
the live serve:

    group 1: UniformTypeKVCacheSpecs block_size=4 page_size_bytes=1419264 layers=12
    layer  ...indexer.tail_cache  shape=(507, 2, 4, 128)
                                  stride=(59136, 512, 128, 1) tight=1024

`KVLayerGroupsManager` then builds a kernel group for it and
`resolve_block_stride_and_log_layout` sees stride(0)=59136 against a tight 1024,
computes padding = 58112, and raises:

    ValueError: ... group's probe tensor has dim-0 padding (58112 elements per
    block) but engine_kv_format=... is not a supported dim-0-padded format

That guard is *correct* — the page really is padded and a naive transfer would
read wrong bytes. The error is that this group should never have been built:
`KpoolTailSpec` is a circular scratch buffer with no reusable prefix state.

vLLM already knows this: `participates_in_prefix_caching` is False for such a
spec, and `UniformTypeKVCacheSpecs.participates_in_prefix_caching` returns False
when any member is non-shareable. The information exists — it just is not
plumbed into this function.

THE FIX

Tag every layer of a non-prefix-cachable engine group as
`EXCLUDED_ENGINE_GROUP`, reusing the mechanism that already exists for
cross-layer layers. Those groups then form no LMCache group, exactly as LMCache's
own hybrid-model docs say should happen for the kpool-tail scratch group.

Idempotent.
"""
import sys

TARGET = ("/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/"
          "kv_cache_groups.py")

MARK = "# [glm53-exclude-scratch-groups]"

OLD = """        for engine_group_id, group in enumerate(vllm_groups):
            # The spec's block_size is the logical tokens covered by one of
            # this group's paged chunks (block IDs); the physical slot count
            # per chunk is discovered later from the registered tensors.
            group_tokens_per_block[engine_group_id] = group.kv_cache_spec.block_size
            for name in group.layer_names:
                per_layer_group_idx[layer_to_idx[name]] = engine_group_id
"""

NEW = '''        for engine_group_id, group in enumerate(vllm_groups):
            # The spec's block_size is the logical tokens covered by one of
            # this group's paged chunks (block IDs); the physical slot count
            # per chunk is discovered later from the registered tensors.
            group_tokens_per_block[engine_group_id] = group.kv_cache_spec.block_size
            # [glm53-exclude-scratch-groups] Scratch / ring groups carry no
            # reusable prefix state (GLM-5.3-Flash's KpoolTailSpec is a circular
            # in-progress-pool buffer). vLLM already reports
            # participates_in_prefix_caching=False for them, and their physical
            # pages are padded relative to the logical payload, which the
            # transfer-layout resolver rightly refuses. Exclude them here, using
            # the same EXCLUDED_ENGINE_GROUP mechanism as cross-layer layers, so
            # no kernel group is built for them at all.
            _spec = group.kv_cache_spec
            _participates = getattr(
                _spec, "participates_in_prefix_caching", True
            )
            if _participates is False:
                for name in group.layer_names:
                    per_layer_group_idx[layer_to_idx[name]] = EXCLUDED_ENGINE_GROUP
                continue
            for name in group.layer_names:
                per_layer_group_idx[layer_to_idx[name]] = engine_group_id
'''


def main() -> int:
    with open(TARGET) as fh:
        text = fh.read()

    if MARK in text:
        print("already patched; no change")
        return 0
    if text.count(OLD) != 1:
        print(f"anchor matched {text.count(OLD)} times; refusing", file=sys.stderr)
        return 1

    with open(TARGET, "w") as fh:
        fh.write(text.replace(OLD, NEW, 1))
    print("scratch-group exclusion applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
