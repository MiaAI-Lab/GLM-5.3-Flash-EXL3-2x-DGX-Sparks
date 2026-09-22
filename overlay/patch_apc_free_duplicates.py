#!/usr/bin/env python3
"""Recycle redundant cache copies at their existing final reference release."""

import argparse
import os
from pathlib import Path
import stat
import sys

MARK = "# [glm53-apc-free-duplicates]"
TARGET = "v1/core/block_pool.py"
MAP_OLD = "    def contain(self, key: BlockHashWithGroupId, block_id: int) -> bool:\n"
MAP_NEW = '''    def has_other_free_block(  # [glm53-apc-free-duplicates]
        self, key: BlockHashWithGroupId, block_id: int
    ) -> bool:
        blocks = self._cache.get(key)
        if isinstance(blocks, dict):
            return any(
                block.block_id != block_id and block.ref_cnt == 0
                for block in blocks.values()
            )
        return (
            blocks is not None
            and blocks.block_id != block_id
            and blocks.ref_cnt == 0
        )

''' + MAP_OLD
FREE_OLD = '''            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is None or not self.enable_caching:
                    # LIFO reuse of non-cached blocks for better GPU locality.
                    blocks_to_evict_first.append(block)
                else:
                    block_hashes = [block.block_hash]
                    block_hashes.extend(
                        self.cached_block_hashes_by_block.get(block.block_id, ())
                    )
                    if self.low_priority_cache_group_ids and all(
'''
FREE_NEW = '''            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is not None and self.enable_caching:
                    block_hashes = [block.block_hash]
                    block_hashes.extend(
                        self.cached_block_hashes_by_block.get(block.block_id, ())
                    )
                    if all(
                        self.cached_block_hash_to_block.has_other_free_block(
                            key, block.block_id
                        )
                        for key in block_hashes
                    ):
                        # Every key survives; retain the existing release timing.
                        self._remove_cached_block_hashes(block)
                if block.block_hash is None or not self.enable_caching:
                    # LIFO reuse of non-cached blocks for better GPU locality.
                    blocks_to_evict_first.append(block)
                else:
                    if self.low_priority_cache_group_ids and all(
'''
EDITS = ((MAP_OLD, MAP_NEW), (FREE_OLD, FREE_NEW))


def prepare(source: str) -> str:
    if MARK in source:
        if not all(source.count(new) == 1 for _, new in EDITS):
            raise ValueError("Duplicate-cache patch is incomplete or drifted")
    else:
        if not all(source.count(old) == 1 for old, _ in EDITS):
            raise ValueError("Duplicate-cache patch anchors drifted")
        for old, new in EDITS:
            source = source.replace(old, new)
    compile(source, TARGET, "exec")
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path("/usr/local/lib/python3.12/dist-packages/vllm"))
    args = parser.parse_args()
    path = Path(os.environ.get("GLM53_BLOCK_POOL_PY", args.source_root / TARGET))
    source = path.read_text()
    patched = prepare(source)
    if patched != source:
        temporary = path.with_name(path.name + ".glm53-dedup.tmp")
        try:
            temporary.write_text(patched)
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        for pyc in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
            pyc.unlink()
    print(f"{MARK} applied and source compiled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
