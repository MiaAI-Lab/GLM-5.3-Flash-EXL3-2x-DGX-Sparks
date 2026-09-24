#!/usr/bin/env python3
"""Fix LMCache's per-layer KV-cache group edits for heterogeneous uniform groups.

THE BUG (measured on GLM-5.3-Flash EXL3, on gb10a)

`apply_kv_cache_group_edits()` iterates a group's layers but hands every layer
the GROUP's wrapper spec:

    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec          # <- the UniformTypeKVCacheSpecs
        for name in group.layer_names:
            for edit in _EDITS:
                if edit.matches(spec, kv_caches[name]):
                    edited[name] = edit.apply(spec, kv_caches[name], ...)

For GLM-5.3-Flash, group 0 is a single `UniformTypeKVCacheSpecs(block_size=3584)`
holding **24 leaves**: 12 `...attn` (MLA, `compress_ratio=1`) and 12
`...indexer.k_cache` (MLA, `compress_ratio=4`). Measured on the live serve:

    group 0 wrapper : block_size=3584  page_size_bytes=29632512  (SUM of leaves)
      leaf ...attn              : compress_ratio=1  page=2351104  declares=False
      leaf ...indexer.k_cache   : compress_ratio=4  page=118272   declares=True

`_declares_slot_compression()` reads `compress_ratio` off the spec it is given.
The wrapper has no such attribute, so it returns **False** — and the
slot-compressed indexer layers pass the guard and are fed to
`_SubpagedMLAAttentionViewEdit`, whose tiling check then fails:

    ValueError: 56 kernel pages (473088 bytes) do not tile the logical page
                (29632512 bytes)

(8448 B kernel page x 56, measured against the wrapper's 29,632,512 B sum.)

THE FIX

Resolve each layer's OWN leaf spec before matching/applying. Then:
  - indexer.k_cache (compress_ratio=4) -> skipped by the guard, as intended
    (slot-compressed groups belong to `lmcache.v1.kv_layer_groups`)
  - ...attn (compress_ratio=1, page 2351104) -> 64x656x1 fp8 x 56 = 2351104,
    which tiles exactly, so the edit succeeds

This is a correctness fix, not a workaround: a per-layer edit must be driven by
that layer's spec. Passing the group wrapper is only equivalent when every
member shares the wrapper's declared page size, which is not true here.

Idempotent. `--revert` restores the original text.
"""
import sys

TARGET = ("/usr/local/lib/python3.12/dist-packages/lmcache/"
          "integration/vllm/kv_cache_group_edits.py")

MARK = "# [glm53-per-layer-leaf-spec]"

OLD = """    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        for name in group.layer_names:
            for edit in _EDITS:
                if edit.matches(spec, kv_caches[name]):
                    edited[name] = edit.apply(spec, kv_caches[name], layout_hints)
                    counts[edit.name] += 1
                    break
"""

NEW = '''    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        # [glm53-per-layer-leaf-spec] A uniform group can still be
        # heterogeneous per layer (GLM-5.3-Flash: MLA attention with
        # compress_ratio=1 alongside indexer.k_cache with compress_ratio=4).
        # Each per-layer edit must be driven by that LAYER's own leaf spec, or
        # the slot-compression guard reads the wrapper (which has no
        # compress_ratio) and lets compressed layers through.
        _leaves = getattr(spec, "kv_cache_specs", None)
        for name in group.layer_names:
            layer_spec = _leaves.get(name, spec) if isinstance(_leaves, dict) else spec
            for edit in _EDITS:
                if edit.matches(layer_spec, kv_caches[name]):
                    edited[name] = edit.apply(
                        layer_spec, kv_caches[name], layout_hints
                    )
                    counts[edit.name] += 1
                    break
'''


def main() -> int:
    with open(TARGET) as fh:
        text = fh.read()

    if "--revert" in sys.argv:
        if MARK in text:
            text = text.replace(NEW, OLD)
            with open(TARGET, "w") as fh:
                fh.write(text)
            print("reverted")
        else:
            print("not patched")
        return 0

    if MARK in text:
        print("already patched; no change")
        return 0
    if text.count(OLD) != 1:
        print(f"anchor matched {text.count(OLD)} times; refusing", file=sys.stderr)
        return 1
    with open(TARGET, "w") as fh:
        fh.write(text.replace(OLD, NEW, 1))
    print("per-layer leaf-spec fix applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
