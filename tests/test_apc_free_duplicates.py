#!/usr/bin/env python3
"""CPU cache ownership and repeated-prefix eviction checks; no GPU execution."""

import argparse
import ast
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from typing import cast
import unittest

HERE = Path(__file__).resolve().parent
OVERLAYS = HERE.parent / "overlay" if (HERE.parent / "overlay").is_dir() else HERE
SOURCE_DIR = Path(os.environ.get(
    "GLM53_VLLM_SRC_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"
)) / "v1/core"
overlay = runpy.run_path(str(OVERLAYS / "patch_apc_free_duplicates.py"))
retention = runpy.run_path(str(OVERLAYS / "patch_apc_per_group_retention.py"))


def load_pool(source_dir, pool_source):
    """Execute the pinned pool and queue classes without importing GPU modules."""
    ns = dict(globals(), BlockHash=bytes, BlockHashWithGroupId=bytes,
              logger=logging.getLogger("cache-test"))
    sources = {
        "kv_cache_utils.py": ((source_dir / "kv_cache_utils.py").read_text(), {
            "KVCacheBlock", "FreeKVCacheBlockQueue", "make_block_hash_with_group_id",
            "get_block_hash", "get_group_id",
        }),
        "block_pool.py": (pool_source, {"BlockHashToBlockMap", "BlockPool"}),
    }
    for filename, (text, names) in sources.items():
        nodes = [n for n in ast.parse(text).body if getattr(n, "name", None) in names]
        if {n.name for n in nodes} != names:
            raise ValueError(f"Missing cache owners in {filename}")
        exec("from __future__ import annotations\n" + ast.unparse(
            ast.Module(body=nodes, type_ignores=[])), ns)
    return ns


def prepare_sources(source_dir):
    source = (source_dir / "block_pool.py").read_text()
    if overlay["MARK"] in source:
        overlay["prepare"](source)
        for old, new in overlay["EDITS"]:
            source = source.replace(new, old)
    if retention["BP_PRIORITY_MARK"] not in source:
        for name in ("BP_INIT", "BP_FREE"):
            source = retention["replace_once"](
                source, retention[name + "_OLD"], retention[name + "_NEW"], name)
    return source, overlay["prepare"](source)


class DuplicateTests(unittest.TestCase):
    ns = None
    source = None

    @classmethod
    def setUpClass(cls):
        if not all((SOURCE_DIR / f).is_file() for f in ("block_pool.py", "kv_cache_utils.py")):
            raise unittest.SkipTest("Set GLM53_VLLM_SRC_ROOT to the pinned vLLM source")
        cls.original, cls.source = prepare_sources(SOURCE_DIR)
        cls.ns = load_pool(SOURCE_DIR, cls.source)

    def setUp(self):
        self.pool = self.ns["BlockPool"](8, True, 64)

    def key(self, name, group=2):
        return self.ns["make_block_hash_with_group_id"](hashlib.sha256(name.encode()).digest(), group)

    def insert(self, block, name, group=2):
        self.pool._insert_block_hash(self.key(name, group), block, 3584)

    def queue(self):
        queue = self.pool.free_block_queue
        cursor = queue.fake_free_list_head.next_free_block
        ids = []
        while cursor is not queue.fake_free_list_tail:
            self.assertNotIn(cursor.block_id, ids)
            self.assertEqual(cursor.ref_cnt, 0)
            self.assertIs(cursor.next_free_block.prev_free_block, cursor)
            ids.append(cursor.block_id)
            cursor = cursor.next_free_block
        self.assertEqual(len(ids), self.pool.get_num_free_blocks())
        return ids

    def test_release_preserves_request_and_copy_pins(self):
        first, second = self.pool.get_new_blocks(2)
        self.insert(first, "shared")
        self.insert(second, "shared")
        self.pool.touch([first])
        self.pool.free_blocks([second])
        self.assertIsNotNone(second.block_hash)
        self.pool.free_blocks([first])
        self.assertEqual(first.ref_cnt, 1)
        self.assertIsNotNone(first.block_hash)
        survivor_order = self.queue()
        self.pool.free_blocks([first])
        self.assertIsNone(first.block_hash)
        self.assertEqual(self.queue(), [first.block_id] + survivor_order)
        self.assertIs(self.pool.get_new_blocks(1)[0], first)
        self.assertIs(self.pool.cached_block_hash_to_block.get_one_block(self.key("shared")), second)

    def test_same_batch_keeps_first_released_copy(self):
        blocks = self.pool.get_new_blocks(3)
        for block in blocks:
            self.insert(block, "shared")
        self.pool.free_blocks(blocks)
        self.assertIsNotNone(blocks[0].block_hash)
        self.assertTrue(all(b.block_hash is None for b in blocks[1:]))
        self.assertEqual(self.queue()[:2], [b.block_id for b in blocks[1:]])
        self.assertIs(self.pool.cached_block_hash_to_block.get_one_block(self.key("shared")), blocks[0])
        # A one-entry dictionary remains after removing the other copies.
        self.assertFalse(self.pool.cached_block_hash_to_block.has_other_free_block(
            self.key("shared"), blocks[0].block_id))

    def test_unique_alias_keeps_the_block(self):
        first, second = self.pool.get_new_blocks(2)
        self.insert(first, "shared")
        self.insert(second, "shared")
        self.insert(second, "unique-alias")
        self.pool.free_blocks([first])
        self.pool.free_blocks([second])
        self.assertIsNotNone(first.block_hash)
        self.assertIsNotNone(second.block_hash)
        self.assertIs(self.pool.cached_block_hash_to_block.get_one_block(self.key("unique-alias")), second)
        self.queue()

    def test_all_aliases_survive_without_removed_events(self):
        first, second, third = self.pool.get_new_blocks(3)
        self.insert(first, "shared")
        self.insert(second, "shared")
        self.insert(second, "alias")
        self.insert(third, "alias")
        self.pool.free_blocks([first, third])
        before = self.queue()
        events = []
        evictions = []
        self.pool._emit_block_removed_events = events.extend
        self.pool.metrics_collector = SimpleNamespace(
            on_block_evicted=lambda b: evictions.append(b.block_id),
            on_block_allocated=lambda b: None)
        self.pool.free_blocks([second])
        self.assertEqual(events, [])
        self.assertEqual(evictions, [])
        self.assertNotIn(second.block_id, self.pool.cached_block_hashes_by_block)
        self.assertEqual(self.queue(), [second.block_id] + before)
        self.assertIs(self.pool.get_new_blocks(1)[0], second)
        self.assertEqual(evictions, [second.block_id])
        self.assertIs(self.pool.cached_block_hash_to_block.get_one_block(self.key("shared")), first)
        self.assertIs(self.pool.cached_block_hash_to_block.get_one_block(self.key("alias")), third)

    def test_groups_and_low_priority_order_are_preserved(self):
        first, second, third = self.pool.get_new_blocks(3)
        self.insert(first, "same", 0)
        self.insert(second, "same", 2)
        self.insert(third, "draft", 6)
        self.pool.low_priority_cache_group_ids = frozenset({6})
        self.pool.free_blocks([first, second, third])
        self.assertTrue(all(b.block_hash is not None for b in (first, second, third)))
        order = self.queue()
        self.assertEqual(order[0], third.block_id)
        self.assertEqual(order[-2:], [first.block_id, second.block_id])

    def test_patch_is_idempotent_and_rejects_partial_or_drifted_source(self):
        prepare = overlay["prepare"]
        self.assertEqual(prepare(self.source), self.source)
        original = self.source
        for old, new in overlay["EDITS"]:
            original = original.replace(new, old)
        self.assertEqual(prepare(original), self.source)
        for old, new in overlay["EDITS"]:
            with self.assertRaises(ValueError):
                prepare(self.source.replace(new, old))
        with self.assertRaises(ValueError):
            prepare(original.replace(overlay["MAP_OLD"], "    def drifted(self):\n"))

    def test_repeated_duplicates_do_not_evict_distinct_histories(self):
        def replay(ns):
            pool = ns["BlockPool"](8, True, 64)
            histories = [self.key(f"history-{i}") for i in range(3)]
            repeated = self.key("repeated")
            blocks = pool.get_new_blocks(4)
            for block, key in zip(blocks, [*histories, repeated]):
                pool._insert_block_hash(key, block, 3584)
            pool.free_blocks(blocks)
            retained = []
            for _ in range(12):
                copies = pool.get_new_blocks(3)
                for block in copies:
                    pool._insert_block_hash(repeated, block, 3584)
                pool.free_blocks(copies)
                retained.append(sum(pool.cached_block_hash_to_block.get_one_block(key)
                                    is not None for key in histories))
            return retained

        self.assertEqual(replay(self.ns), [3] * 12)
        control = replay(load_pool(SOURCE_DIR, self.original))
        self.assertEqual(control[0], 3)
        self.assertEqual(control[1:], [0] * 11)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    args, remaining = parser.parse_known_args()
    SOURCE_DIR = args.source_dir
    for name in ("block_pool.py", "kv_cache_utils.py"):
        if not (SOURCE_DIR / name).is_file():
            parser.error(f"Missing pinned source: {SOURCE_DIR / name}")
    unittest.main(argv=[__file__, *remaining])
