#!/usr/bin/env python3
"""patch_kv_offload_groups.py — let OffloadingConnector serve GLM-5.3's
hybrid layout by excluding per-request scratch groups from offload scheduling.

GLM-5.3-Flash (this fork) exposes 7 KV cache groups:
  0 MLA (UniformType, block 3584)  2-5 Mamba/GDN (block 3584, align mode)
  6 DFlash drafter SWA (block 64, EAGLE)   -> all offloadable
  1 indexer/kpool tail (UniformType, block 4, 11 layers)
    -> per-request circular scratch. GPU prefix caching already treats it as
       rebuild-on-hit (it opts out of the hybrid hit min), so external KV
       loads need nothing from it. Its 4-token block cannot align with the
       64-token hash granularity, which stock build_offloading_config asserts
       on. This patch marks such groups as scratch and skips them at every
       scheduler touchpoint instead of failing the boot.

Also unwraps UniformTypeKVCacheSpecs in the window classifier (this fork wraps
group specs) and makes unknown spec classes classify as full-attention instead
of asserting.

Fail-closed: all anchors verified, compile-checked, original restored on error.
Idempotent. Kill switch: GLM53_SKIP_KV_OFFLOAD_GROUPS_PATCH=1.
"""
import os
import sys
from pathlib import Path

MARK = "# [glm53-kv-offload-scratch]"


cfg_edits = [
    (
        "from typing import TYPE_CHECKING",
        "from typing import TYPE_CHECKING\n\nfrom vllm.logger import init_logger",
    ),
    (
        "from vllm.v1.kv_offload.config import (",
        "logger = init_logger(__name__)  # [glm53-kv-offload-scratch-logger]\n\nfrom vllm.v1.kv_offload.config import (",
    ),
    (
        """    _, tokens_per_hash = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    for group in groups:
        assert group.tokens_per_block % tokens_per_hash == 0, (
            f"tokens_per_block={group.tokens_per_block} not divisible by "
            f"tokens_per_hash={tokens_per_hash}. "
            f"Hybrid models (e.g. Mamba+Attention) need "
            f"--enable-prefix-caching to align block sizes."
        )""",
        """    _, tokens_per_hash = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
    # [glm53-kv-offload-scratch] groups that cannot align with the hash
    # granularity are per-request scratch (e.g. GLM kpool tail, block=4).
    # The scheduler excludes them from offload scheduling; do not assert.
    _max_tpb = max(g.tokens_per_block for g in groups)
    for _idx, group in enumerate(groups):
        if (group.tokens_per_block % tokens_per_hash != 0
                or group.tokens_per_block < _max_tpb):
            logger.warning(
                "[kv-offload-scratch] group %d (tokens_per_block=%d, %d layers)"
                " is not hash-aligned (tokens_per_hash=%d) - treating as"
                " per-request scratch, excluded from offloading",
                _idx, group.tokens_per_block, len(group.layer_names),
                tokens_per_hash,
            )""",
    ),
]

sch_edits = [
    # 1) import UniformTypeKVCacheSpecs
    (
        "from vllm.v1.kv_cache_interface import (",
        "from vllm.v1.kv_cache_interface import (\n    UniformTypeKVCacheSpecs,",
    ),
    # 2) unwrap UniformType + tolerate unknown specs in window classifier
    (
        """def get_sliding_window_size_in_chunks(
    kv_cache_spec: KVCacheSpec, tokens_per_chunk: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):""",
        """def get_sliding_window_size_in_chunks(
    kv_cache_spec: KVCacheSpec, tokens_per_chunk: int
) -> int | None:
    # [glm53-kv-offload-scratch] this fork wraps group specs in
    # UniformTypeKVCacheSpecs; classify by the wrapped per-layer spec.
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
    if isinstance(kv_cache_spec, SlidingWindowSpec):""",
    ),
    (
        """    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None""",
        """    if not isinstance(kv_cache_spec, FullAttentionSpec):
        logger.warning_once(
            "[kv-offload-scratch] spec %s treated as full attention",
            type(kv_cache_spec).__name__,
        )
    return None""",
    ),
    # 3) GroupOffloadConfig scratch flag
    (
        """    # True for EAGLE/MTP draft-model attention groups. The trailing chunk
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False""",
        """    # True for EAGLE/MTP draft-model attention groups. The trailing chunk
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False
    # [glm53-kv-offload-scratch] per-request scratch group: contributes
    # nothing to offload stores/loads/lookups (e.g. GLM kpool tail).
    is_scratch: bool = False""",
    ),
    # 4) from_spec: skip scratch in alignment scan
    (
        """        full_attn_tokens_per_chunk: set[int] = set()
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * spec.blocks_per_chunk)""",
        """        full_attn_tokens_per_chunk: set[int] = set()
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            if (tokens_per_block % spec.tokens_per_hash != 0
                    or tokens_per_block < max(spec.tokens_per_block)):
                continue  # [glm53-kv-offload-scratch]
            kv_spec = kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * spec.blocks_per_chunk)""",
    ),
    # 5) from_spec: emit scratch config entries
    (
        """        kv_group_configs_list: list[GroupOffloadConfig] = []
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_cache_group = kv_cache_config.kv_cache_groups[idx]
            kv_spec = kv_cache_group.kv_cache_spec
            sw = get_sliding_window_size_in_chunks(""",
        """        kv_group_configs_list: list[GroupOffloadConfig] = []
        for idx, tokens_per_block in enumerate(spec.tokens_per_block):
            kv_cache_group = kv_cache_config.kv_cache_groups[idx]
            kv_spec = kv_cache_group.kv_cache_spec
            if (tokens_per_block % spec.tokens_per_hash != 0
                    or tokens_per_block < max(spec.tokens_per_block)):
                # [glm53-kv-offload-scratch] per-request scratch group
                kv_group_configs_list.append(
                    GroupOffloadConfig(
                        group_idx=idx,
                        tokens_per_block=tokens_per_block,
                        tokens_per_chunk=tokens_per_block * spec.blocks_per_chunk,
                        hashes_per_chunk=1,
                        sliding_window_size_in_chunks=None,
                        alignment_chunk_count=None,
                        kv_event_group_spec=get_offloading_event_group_spec(
                            kv_cache_group
                        ),
                        is_eagle_group=False,
                        is_scratch=True,
                    )
                )
                continue
            sw = get_sliding_window_size_in_chunks(""",
    ),
    # 6) ctor lookup-group lists skip scratch
    (
        """        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)""",
        """        for group_config in self.config.kv_group_configs:
            if group_config.is_scratch:  # [glm53-kv-offload-scratch]
                continue
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)""",
    ),
    # 7) update_offload_keys skip
    (
        """    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(""",
        """    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            if group_config.is_scratch:  # [glm53-kv-offload-scratch]
                continue
            for req_block_hash in islice(""",
    ),
    # 8) storable_chunks -> 0 for scratch
    (
        """        num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens""",
        """        if group_config.is_scratch:  # [glm53-kv-offload-scratch]
            return 0
        num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens""",
    ),
    # 9) update_num_hit_chunks skip
    (
        """        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_chunks = (
                num_cached_tokens // group_config.tokens_per_chunk
            )""",
        """        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            if group_config.is_scratch:  # [glm53-kv-offload-scratch]
                continue
            group_state.num_hit_chunks = (
                num_cached_tokens // group_config.tokens_per_chunk
            )""",
    ),
    # 10) _touch skip
    (
        """        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_chunks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)""",
        """        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.is_scratch:  # [glm53-kv-offload-scratch]
                continue
            if group_config.sliding_window_size_in_chunks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)""",
    ),
    # 11) update_state_after_alloc: zero contribution
    (
        """            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            tokens_per_block = group_config.tokens_per_block""",
        """            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            if group_config.is_scratch:  # [glm53-kv-offload-scratch]
                group_sizes.append(0)
                block_indices.append(0)
                continue

            tokens_per_block = group_config.tokens_per_block""",
    ),
    # 12) _build_store_jobs key-collection loop skip
    (
        """            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )

                start_chunk_idx = group_state.next_stored_chunk_idx
                if num_chunks <= start_chunk_idx:
                    continue""",
        """            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                if group_config.is_scratch:  # [glm53-kv-offload-scratch]
                    continue
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )

                start_chunk_idx = group_state.next_stored_chunk_idx
                if num_chunks <= start_chunk_idx:
                    continue""",
    ),
    # 13) _build_store_jobs src-block loop: zero contribution
    (
        """            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_chunks is not None
                )""",
        """            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                if group_config.is_scratch:  # [glm53-kv-offload-scratch]
                    group_sizes.append(0)
                    block_indices.append(0)
                    continue
                is_sliding_window = (
                    group_config.sliding_window_size_in_chunks is not None
                )""",
    ),
]


def apply(path: Path, edits, needs_logger: bool = False) -> bool:
    src = path.read_text()
    if MARK in src:
        print(f"[kv-offload-scratch] {path.name} already patched")
        return True
    original = src
    for old, new in edits:
        if old not in src:
            print(f"[kv-offload-scratch] ANCHOR MISSING in {path.name}:\n{old[:120]}",
                  file=sys.stderr)
            return False
        if src.count(old) != 1:
            print(f"[kv-offload-scratch] anchor not unique ({src.count(old)}) in "
                  f"{path.name}: {old[:80]}", file=sys.stderr)
            return False
        src = src.replace(old, new, 1)
    try:
        compile(src, str(path), "exec")
    except SyntaxError as e:
        print(f"[kv-offload-scratch] compile failed for {path.name}: {e}",
              file=sys.stderr)
        path.write_text(original)
        return False
    path.write_text(src)
    print(f"[kv-offload-scratch] applied to {path.name} ({len(edits)} edits)")
    return True



# --- v4: worker-side coherence fixes ---
wrk_edits = [
    # (a) Bypass the packed whole-row fast path. Groups allocate block ids
    # independently over shared rows, so whole-row loads clobber the other
    # groups' live pages in the destination row (proven: corrupted restores).
    # The generic per-layer path copies only each group's own byte ranges.
    (
        """        if packed_kv_cache_tensor is not None:""",
        """        packed_kv_cache_tensor = None  # [glm53-kv-offload-scratch] whole-row
        # fast path is incoherent for multi-block-size hybrids; force the
        # per-layer path below.
        if packed_kv_cache_tensor is not None:""",
    ),
    # (b) Mamba layers in packed layouts are registered as strided views;
    # plain .view() requires contiguity. Build a byte-strided view instead.
    (
        """                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    layer_kv_cache = kv_caches[layer_name]
                    assert layer_kv_cache.dtype == torch.int8
                    tensors_per_block[layer_name] = (
                        layer_kv_cache.view(
                            num_blocks, layer_kv_cache_spec.page_size_bytes
                        ),
                    )""",
        """                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    layer_kv_cache = kv_caches[layer_name]
                    assert layer_kv_cache.dtype == torch.int8
                    # [glm53-kv-offload-scratch] strided-safe view
                    _page = layer_kv_cache_spec.page_size_bytes
                    _raw = torch.empty(
                        0, dtype=torch.int8, device=layer_kv_cache.device
                    ).set_(layer_kv_cache.untyped_storage())
                    _stride0 = (
                        layer_kv_cache.stride(0)
                        if layer_is_packed.get(layer_name)
                        else _page
                    )
                    tensors_per_block[layer_name] = (
                        torch.as_strided(
                            _raw,
                            (num_blocks, _page),
                            (_stride0, 1),
                            layer_kv_cache.storage_offset(),
                        ),
                    )""",
    ),
    # (c) Tolerate certified-mapping size mismatches instead of asserting:
    # fall back to no mapping (direct copies do not use it).
    (
        """                    assert (
                        mapping is None
                        or mapping.local_page_size_bytes
                        == unpadded_page_size_bytes[layer_name]
                    )""",
        """                    if (
                        mapping is not None
                        and mapping.local_page_size_bytes
                        != unpadded_page_size_bytes[layer_name]
                    ):  # [glm53-kv-offload-scratch]
                        logger.warning_once(
                            "[kv-offload-scratch] dropping certified mapping for"
                            " %s (size mismatch)", layer_name)
                        mapping = None""",
    ),
]


# --- v5: NVMe-backed staging region (disk tier that is per-rank correct) ---
sor_edits = [
    (
        '''        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"''',
        '''        _mmap_dir = os.environ.get("GLM53_OFFLOAD_MMAP_DIR")  # [glm53-kv-offload-scratch]
        self._glm53_disk_backed = bool(_mmap_dir)
        if _mmap_dir:
            os.makedirs(_mmap_dir, exist_ok=True)
        else:
            _mmap_dir = "/dev/shm"
        self.mmap_path = f"{_mmap_dir}/vllm_offload_{engine_id}.mmap"''',
    ),
    (
        '''                check_shm_free_space(self.total_size_bytes)
                os.ftruncate(self.fd, self.total_size_bytes)''',
        '''                if not self._glm53_disk_backed:  # [glm53-kv-offload-scratch]
                    check_shm_free_space(self.total_size_bytes)
                os.ftruncate(self.fd, self.total_size_bytes)''',
    ),
    (
        '''        populate_write_fn = _get_populate_write_fn(self.mmap_obj)

        if rank is not None:''',
        '''        if self._glm53_disk_backed:  # [glm53-kv-offload-scratch] sparse file:
            # pre-faulting would materialize the whole region on NVMe.
            populate_write_fn = lambda *_a, **_k: None
            _skip_populate = True
        else:
            populate_write_fn = _get_populate_write_fn(self.mmap_obj)
            _skip_populate = False

        if _skip_populate:
            pass
        elif rank is not None:''',
    ),
]

gwk_edits = [
    (
        '''    if not current_platform.is_cuda_alike():''',
        '''    import os as _os  # [glm53-kv-offload-scratch]
    if _os.environ.get("GLM53_OFFLOAD_MMAP_DIR"):
        logger.info("[kv-offload-scratch] disk-backed staging region: "
                    "skipping cudaHostRegister")
        return
    if not current_platform.is_cuda_alike():''',
    ),
]


# --- v6: bound page cache for disk-backed staging (msync + MADV_DONTNEED) ---
sor_edits.append(
    (
        '''    def create_next_worker_view(self, tensor_page_size: int) -> torch.Tensor:''',
        '''    def glm53_release_page_cache(self) -> None:  # [glm53-kv-offload-scratch]
        """Flush and drop this region's resident pages.

        Disk-backed staging grows the page cache with every store; on GB10 that
        memory is taken from the same pool the GPU allocates out of. msync makes
        the bytes durable, MADV_DONTNEED returns the pages; later reads re-fault
        from NVMe, which is the point of a disk tier.
        """
        if self.mmap_obj is None:
            return
        try:
            self.mmap_obj.flush()
            self.mmap_obj.madvise(mmap.MADV_DONTNEED, 0, self.total_size_bytes)
        except (OSError, ValueError) as exc:
            logger.warning("[kv-offload-scratch] page-cache release failed: %s", exc)

    def create_next_worker_view(self, tensor_page_size: int) -> torch.Tensor:''',
    )
)

gwk_edits.append(
    (
        '''        self._mmap_region = mmap_region
        pin_memory = PIN_MEMORY''',
        '''        self._mmap_region = mmap_region
        import os as _os  # [glm53-kv-offload-scratch]
        self._glm53_release_bytes = int(
            _os.environ.get("GLM53_OFFLOAD_RELEASE_BYTES") or 0
        )
        self._glm53_unreleased = 0
        if self._glm53_release_bytes:
            logger.info(
                "[kv-offload-scratch] releasing staging page cache every %.1f GB",
                self._glm53_release_bytes / 1e9,
            )
        pin_memory = PIN_MEMORY''',
    )
)

gwk_edits.append(
    (
        '''    def get_finished(self) -> list[TransferResult]:
        return self._store_handler.get_finished() + self._load_handler.get_finished()''',
        '''    def get_finished(self) -> list[TransferResult]:
        results = (
            self._store_handler.get_finished() + self._load_handler.get_finished()
        )
        # [glm53-kv-offload-scratch] keep disk-backed staging from filling RAM
        if self._glm53_release_bytes and self._mmap_region is not None:
            for _r in results:
                if _r.transfer_size:
                    self._glm53_unreleased += _r.transfer_size
            if self._glm53_unreleased >= self._glm53_release_bytes:
                self._glm53_unreleased = 0
                self._mmap_region.glm53_release_page_cache()
        return results''',
    )
)

def targets(root: Path | None = None) -> list[tuple[Path, list]]:
    """(file, edits) pairs resolved against the vLLM install root."""
    v = root or Path(
        os.environ.get(
            "GLM53_VLLM_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"
        )
    )
    kvc = v / "distributed/kv_transfer/kv_connector/v1/offloading"
    return [
        (kvc / "config.py", cfg_edits),
        (kvc / "scheduler.py", sch_edits),
        (kvc / "worker.py", wrk_edits),
        (v / "v1/kv_offload/cpu/shared_offload_region.py", sor_edits),
        (v / "v1/kv_offload/cpu/gpu_worker.py", gwk_edits),
    ]


def main() -> int:
    if os.environ.get("GLM53_SKIP_KV_OFFLOAD_GROUPS_PATCH") == "1":
        print("[kv-offload-scratch] skipped via env")
        return 0
    for path, edits in targets():
        if not apply(path, edits):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
