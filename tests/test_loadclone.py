#!/usr/bin/env python3
"""CPU-only: python -m unittest discover -s tests -p test_loadclone.py -v.

Optional exact PR230 composition probe (no downloads during tests):
GLM53_PR230_PATCH=/path/to/bc97cbf1/patch_cold_load_uma.py <same command>.
The source fixture is an attributed extraction, not an installed vLLM module.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import types
import unittest
import weakref
from unittest.mock import patch

import torch
from safetensors.torch import load, safe_open, save_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (Path(__file__).resolve().parent / "fixtures/loadclone_weight_utils.py.txt").read_text()


def module(path):
    spec = importlib.util.spec_from_file_location("loader_patch", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


PATCH_PATH = ROOT / "overlay/patch_loadclone.py"
if not PATCH_PATH.is_file():
    PATCH_PATH = Path(__file__).resolve().parent / "patch_loadclone.py"
OVERLAY = module(PATCH_PATH)


def function_source(src, name):
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(src.splitlines()[node.lineno - 1:node.end_lineno])


def runtime(src=None, fs="ext4"):
    src = OVERLAY.prepare(FIXTURE) if src is None else src
    tree = ast.parse(src)
    # Execute real iterator and overlay, not vLLM imports/CUDA initialization.
    selected = {"safetensors_weights_iterator", "_glm53_original_safetensors_weights_iterator",
                "_glm53_load_options", "_Glm53ShardPrefetch", "_glm53_uma_page_size"}
    tree.body = [n for n in tree.body
                 if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected)
                 or (isinstance(n, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id == "_GLM53_UMA_STAGE_MMAP" for t in n.targets))]
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    ast.fix_missing_locations(tree)
    logger = types.SimpleNamespace(info_once=lambda *a: None, warning_once=lambda *a: None)
    ns = dict(os=os, torch=torch, safe_open=safe_open, load=load, logger=logger,
              DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS=2,
              DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE=1024,
              _natural_sort_key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)],
              _get_fs_type=lambda paths: fs,
              _get_checkpoints_size_bytes=lambda paths: sum(os.path.getsize(p) for p in paths),
              _get_available_ram_bytes=lambda: 1 << 40,
              _prefetch_all_checkpoints=lambda *a, **k: None,
              enable_tqdm=lambda enabled: False, tqdm=lambda items, **k: items,
              _BAR_FORMAT="", should_skip_weight=lambda name, ids: ids == {0} and name == "skip",
              torchao_version_at_least=lambda version: True)
    exec(compile(tree, "cpu-weight-utils", "exec"), ns)
    return ns


class LoaderCPU(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {"GLM53_LOAD_CLONE": "1", "GLM53_LOAD_PREFETCH": "2",
                                          "GLM53_COLD_LOAD_STAGE_MMAP": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.page = patch.object(os, "sysconf", return_value=4096)
        self.page.start()
        self.addCleanup(self.page.stop)
        self.files = []
        for shard in (10, 2, 1):
            path = str(Path(self.tmp.name) / f"shard-{shard}.safetensors")
            save_file({"float": torch.tensor([float("nan"), -0.0, float("inf"), float(shard)]),
                       "bf16": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
                       "int": torch.tensor(shard, dtype=torch.int64),
                       "empty": torch.empty((0, 3), dtype=torch.float16),
                       "skip": torch.tensor([False, True])}, path)
            self.files.append(path)

    def assert_no_workers(self):
        self.assertFalse([t.name for t in threading.enumerate() if t.name.startswith("glm53-load-prefetch")])

    def collect(self, ns, strategy=None, ids=None):
        return list(ns["safetensors_weights_iterator"](self.files, False, strategy, ids))

    def assert_bytes(self, left, right, strategy=None):
        if strategy == "eager":
            # load(bytes) returns a dict in native deserialize iteration order,
            # not a sorted-key contract. Compare exact records within each shard,
            # never flatten to a dict keyed by name: names repeat across shards.
            offset = 0
            for path in sorted(self.files, key=lambda p: int(Path(p).stem.removeprefix("shard-"))):
                with safe_open(path, framework="pt") as shard:
                    count = len(shard.keys())
                def records(items):
                    return [(name, tensor.dtype, tuple(tensor.shape),
                             tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
                            for name, tensor in items]
                self.assertCountEqual(records(left[offset:offset + count]),
                                      records(right[offset:offset + count]), msg=path)
                offset += count
            self.assertEqual(len(left), offset)
            self.assertEqual(len(right), offset)
            return
        self.assertEqual([n for n, _ in left], [n for n, _ in right])
        for (_, a), (_, b) in zip(left, right):
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.reshape(-1).view(torch.uint8).numpy().tobytes(),
                             b.reshape(-1).view(torch.uint8).numpy().tobytes())

    def test_byte_values_order_filter_and_disable(self):
        baseline = self.collect(runtime(FIXTURE), ids={0})
        ns = runtime()
        for depth in (0, 1, 2, 16):
            with self.subTest(depth=depth), patch.dict(os.environ, {"GLM53_LOAD_PREFETCH": str(depth)}):
                self.assert_bytes(baseline, self.collect(ns, ids={0}))
                self.assert_no_workers()
        self.assertEqual([int(t) for n, t in baseline if n == "int"], [1, 2, 10])
        self.assertNotIn("skip", [n for n, _ in baseline])

    def test_one_clone_and_page_safety(self):
        original = torch.Tensor.clone
        for page, optional, expected in ((4096, "0", 0), (4096, "1", 15), (65536, "0", 15), (65536, "1", 15)):
            clones = []
            def clone(tensor, *args, **kwargs):
                clones.append(tensor)
                return original(tensor, *args, **kwargs)
            with self.subTest(page=page, optional=optional), patch.object(os, "sysconf", return_value=page), \
                    patch.dict(os.environ, {"GLM53_LOAD_CLONE": optional}), patch.object(torch.Tensor, "clone", clone):
                self.collect(runtime())
                self.assertEqual(len(clones), expected)
        with patch.object(os, "sysconf", return_value=65536), patch.dict(os.environ, {
                "GLM53_LOAD_CLONE": "0", "GLM53_COLD_LOAD_STAGE_MMAP": "0"}), \
                patch.object(torch.Tensor, "clone", side_effect=AssertionError("cloned")):
            self.collect(runtime())

    def test_early_close_and_consumer_exception(self):
        ns = runtime()
        for consumer_error in (False, True):
            iterator = ns["safetensors_weights_iterator"](self.files, False)
            name, tensor = next(iterator)
            try:
                if consumer_error:
                    raise RuntimeError("consumer")
            except RuntimeError:
                pass
            finally:
                iterator.close()
            self.assert_no_workers()
            # Materialized result remains valid after safe_open and executor close.
            self.assertIsInstance(tensor, torch.Tensor)
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()

    def test_prefetch_and_tensor_errors_propagate_and_join(self):
        ns = runtime()
        cls = ns["_Glm53ShardPrefetch"]
        with patch.object(cls, "read", side_effect=OSError("read failure")):
            with self.assertRaisesRegex(OSError, "read failure"):
                self.collect(ns)
        self.assert_no_workers()
        with patch.dict(os.environ, {"GLM53_LOAD_PREFETCH": "0"}), \
                patch.dict(ns, {"safe_open": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tensor failure"))}):
            with self.assertRaisesRegex(RuntimeError, "tensor failure"):
                self.collect(ns)
        self.assert_no_workers()
        Path(self.files[-1]).write_bytes(b"not safetensors")
        with self.assertRaises(Exception):
            self.collect(ns)
        self.assert_no_workers()

    @unittest.skipUnless(hasattr(os, "posix_fadvise") and Path("/proc/self/fd").is_dir(),
                         "requires Linux file advice")
    def test_completed_shard_advice_waits_for_tensor_release(self):
        for depth, early in ((0, False), (2, False), (2, True)):
            with self.subTest(depth=depth, early=early), \
                    patch.dict(os.environ, {"GLM53_LOAD_PREFETCH": str(depth)}):
                ns = runtime()
                views, advised = {}, []
                class Shard:
                    def __init__(self, path, **kwargs):
                        self.path = path
                        self.shard = safe_open(path, **kwargs)
                    def __enter__(self):
                        self.shard.__enter__()
                        return self
                    def __exit__(self, *args):
                        return self.shard.__exit__(*args)
                    def keys(self):
                        return self.shard.keys()
                    def get_tensor(self, name):
                        value = self.shard.get_tensor(name)
                        views.setdefault(self.path, []).append(weakref.ref(value))
                        return value
                real_advice = os.posix_fadvise
                def advise(fd, offset, length, advice):
                    path = os.readlink(f"/proc/self/fd/{fd}")
                    self.assertIn(path, views)
                    self.assertTrue(all(ref() is None for ref in views[path]),
                                    "cache advice preceded mmap tensor release")
                    advised.append(path)
                    return real_advice(fd, offset, length, advice)
                with patch.dict(ns, {"safe_open": Shard}), patch.object(os, "posix_fadvise", advise):
                    iterator = ns["safetensors_weights_iterator"](self.files, False)
                    if early:
                        values = [next(iterator)]
                        iterator.close()
                    else:
                        values = list(iterator)
                ordered = sorted(self.files, key=ns["_natural_sort_key"])
                self.assertEqual(set(advised), set(ordered[:1] if early else ordered))
                expected = self.collect(runtime(FIXTURE))
                self.assert_bytes(expected[:1] if early else expected, values)
                self.assert_no_workers()

    @unittest.skipUnless(hasattr(os, "posix_fadvise") and Path("/proc/self/maps").is_file(),
                         "requires Linux mmap and file advice")
    def test_filtered_shard_gaps_wait_for_next_tensor_and_release_tail(self):
        paths = []
        for index in range(5):
            path = str(Path(self.tmp.name) / f"filtered-{index}.safetensors")
            save_file({"int" if index in (0, 3) else "skip": torch.tensor(index)}, path)
            paths.append(path)
        real_advice = os.posix_fadvise
        for depth in (0, 2):
            advised = []
            def advise(fd, offset, length, advice):
                path = os.readlink(f"/proc/self/fd/{fd}")
                self.assertNotIn(path, Path("/proc/self/maps").read_text(),
                                 "filtered gap left a source mapping alive")
                advised.append(path)
                return real_advice(fd, offset, length, advice)
            with self.subTest(depth=depth), \
                    patch.dict(os.environ, {"GLM53_LOAD_PREFETCH": str(depth)}), \
                    patch.object(os, "posix_fadvise", advise):
                iterator = runtime()["safetensors_weights_iterator"](paths, False, None, {0})
                first = next(iterator)
                self.assertFalse(advised)
                second = next(iterator)
                self.assertEqual(set(advised), set(paths[:3]))
                self.assertEqual(list(iterator), [])
            self.assertEqual(set(advised), set(paths))
            self.assertEqual((first[0], first[1].item(), second[0], second[1].item()),
                             ("int", 0, "int", 3))
            self.assert_no_workers()

    @unittest.skipUnless(hasattr(os, "posix_fadvise"), "requires file advice")
    def test_advice_failure_preserves_load_error_and_retained_views(self):
        ns = runtime()
        with patch.dict(os.environ, {"GLM53_LOAD_CLONE": "0", "GLM53_LOAD_PREFETCH": "0"}):
            baseline = self.collect(runtime(FIXTURE))
            values = self.collect(ns)
            self.assert_bytes(baseline, values)
            original = ns["safe_open"]
            ordered = sorted(self.files, key=ns["_natural_sort_key"])
            def broken(path, **kwargs):
                if path == ordered[1]:
                    raise RuntimeError("original tensor load failure")
                return original(path, **kwargs)
            with patch.dict(ns, {"safe_open": broken}), \
                    patch.object(os, "posix_fadvise", side_effect=OSError("advice unavailable")), \
                    patch.object(ns["logger"], "warning_once") as warning:
                with self.assertRaisesRegex(RuntimeError, "original tensor load failure"):
                    self.collect(ns)
                warning.assert_called_once()
            with patch.object(os, "posix_fadvise", side_effect=OSError("advice unavailable")) as advice, \
                    patch.object(ns["logger"], "warning_once") as warning:
                self.assert_bytes(baseline, self.collect(ns))
                advice.assert_called_once()
                warning.assert_called_once()
        self.assert_no_workers()

    def test_window_bound_and_no_retained_completed_futures(self):
        ns = runtime()
        prefetch = ns["_Glm53ShardPrefetch"](2)
        reads = []
        with patch.object(prefetch, "read", side_effect=lambda path: reads.append(path)):
            files = prefetch.files([str(i) for i in range(10)], None, False)
            self.assertEqual(next(files), "0")
            self.assertLessEqual(len(reads), 2)
            self.assertEqual(len(prefetch.pending), 1)
            self.assertEqual(next(files), "1")
            self.assertLessEqual(len(reads), 3)
            self.assertEqual(len(prefetch.pending), 1)
            files.close()
            prefetch.close()
        self.assertFalse(prefetch.pending)
        self.assert_no_workers()

    def test_close_cooperatively_stops_reader_and_joins(self):
        ns = runtime()
        entered, release = threading.Event(), threading.Event()
        reads = []
        class Stream:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def readinto(self, buffer):
                reads.append(1)
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test reader gate timeout")
                return 1
        prefetch = ns["_Glm53ShardPrefetch"](1)
        from concurrent.futures import ThreadPoolExecutor
        prefetch.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glm53-load-prefetch")
        with patch("builtins.open", return_value=Stream()):
            future = prefetch.pool.submit(prefetch.read, "unused")
            try:
                self.assertTrue(entered.wait(5))
                prefetch.stop.set()
            finally:
                release.set()
                prefetch.close()
            future.result()
        self.assertEqual(reads, [1])
        self.assert_no_workers()

    def test_strategies_and_nfs_do_not_add_prefetch(self):
        for strategy, fs in (("eager", "ext4"), ("prefetch", "ext4"),
                             (None, "nfs"), (None, "nfs4"), (None, "lustre"), ("lazy", "nfs")):
            ns = runtime(fs=fs)
            with self.subTest(strategy=strategy, fs=fs), \
                    patch.object(ns["_Glm53ShardPrefetch"], "read", side_effect=AssertionError("extra prefetch")), \
                    patch.object(ns["_Glm53ShardPrefetch"], "release", side_effect=AssertionError("extra advice")):
                self.assert_bytes(self.collect(runtime(FIXTURE, fs=fs), strategy), self.collect(ns, strategy), strategy)
            self.assert_no_workers()
        for strategy in ("eager", "prefetch"):
            with patch.object(torch.Tensor, "clone", side_effect=AssertionError("nonlazy clone")):
                self.collect(runtime(), strategy)
        ns = runtime()
        ao = types.ModuleType("torchao.prototype.safetensors.safetensors_support")
        ao.unflatten_tensor_state_dict = lambda state, metadata: (state, {})
        with patch.dict(sys.modules, {ao.__name__: ao}), \
                patch.object(ns["_Glm53ShardPrefetch"], "read", side_effect=AssertionError("extra prefetch")), \
                patch.object(ns["_Glm53ShardPrefetch"], "release", side_effect=AssertionError("extra advice")), \
                patch.object(torch.Tensor, "clone", side_effect=AssertionError("torchao clone")):
            self.assert_bytes(self.collect(runtime(FIXTURE), "torchao"), self.collect(ns, "torchao"))

    def test_explicit_lazy_uses_bounded_prefetch(self):
        ns = runtime()
        original = ns["_Glm53ShardPrefetch"].read
        reads = []
        def read(prefetcher, path):
            reads.append(path)
            return original(prefetcher, path)
        with patch.object(ns["_Glm53ShardPrefetch"], "read", read):
            self.assert_bytes(self.collect(runtime(FIXTURE), "lazy"), self.collect(ns, "lazy"))
        self.assertEqual(sorted(reads), sorted(self.files))
        self.assert_no_workers()

    @unittest.skipUnless((ROOT / "overlay/tp3/vllm/model_executor/model_loader/weight_utils.py").is_file(),
                         "vendored TP3 source is not installed in this image")
    def test_vendored_loader_matches_applied_overlay(self):
        source = (ROOT / "overlay/tp3/vllm/model_executor/model_loader/weight_utils.py").read_text()
        # Read-only mounts must already satisfy the shared patch; no second implementation.
        self.assertEqual(OVERLAY.prepare(source), source)
        for strategy in (None, "lazy", "eager", "prefetch"):
            with self.subTest(strategy=strategy):
                self.assert_bytes(self.collect(runtime(), strategy), self.collect(runtime(source), strategy), strategy)
        self.assert_no_workers()

    @unittest.skipUnless(os.environ.get("GLM53_LOADCLONE_SOURCE"), "set GLM53_LOADCLONE_SOURCE for image byte parity")
    def test_pinned_source_cpu_byte_parity(self):
        source = Path(os.environ["GLM53_LOADCLONE_SOURCE"]).read_text()
        for strategy in (None, "lazy", "eager", "prefetch"):
            with self.subTest(strategy=strategy):
                self.assert_bytes(self.collect(runtime(source), strategy),
                                  self.collect(runtime(OVERLAY.prepare(source)), strategy), strategy)
        self.assert_no_workers()

    def test_invalid_options_and_empty_input(self):
        for key, value in (("GLM53_LOAD_PREFETCH", "-1"), ("GLM53_LOAD_PREFETCH", "17"),
                           ("GLM53_LOAD_PREFETCH", "x"), ("GLM53_LOAD_PREFETCH", ""),
                           ("GLM53_LOAD_CLONE", "true")):
            with self.subTest(key=key, value=value), patch.dict(os.environ, {key: value}):
                with self.assertRaises(ValueError):
                    self.collect(runtime())
        self.assertEqual(list(runtime()["safetensors_weights_iterator"]([], False)), [])
        self.assert_no_workers()


class Patching(unittest.TestCase):
    def test_idempotence_and_instanttensor_unchanged(self):
        out = OVERLAY.prepare(FIXTURE)
        self.assertEqual(OVERLAY.prepare(out), out)
        self.assertEqual(function_source(out, "instanttensor_weights_iterator"),
                         function_source(FIXTURE, "instanttensor_weights_iterator"))
        for src in (FIXTURE.replace(OVERLAY.LOOP, "    for st_file in other(\n        sorted_files,\n"),
                    FIXTURE.replace("                    param = f.get_tensor(name)\n",
                                    "                    param = f.get_tensor(name).clone()\n"),
                    out.replace("self.stop.set()", "self.stop.clear()"),
                    out.replace("clone, safety, depth = _glm53_load_options()", "clone, safety, depth = False, False, 0")):
            with self.assertRaises(ValueError):
                OVERLAY.prepare(src)

    def test_check_mode_accepts_readonly_patch_and_refuses_stock_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weight_utils.py"
            for src, accepted in ((FIXTURE, False), (OVERLAY.prepare(FIXTURE), True)):
                target.write_text(src)
                with patch.object(OVERLAY, "TARGET", target), patch.object(sys, "argv", ["patch", "--check"]), \
                        patch.object(Path, "write_text", side_effect=AssertionError("check attempted a write")):
                    if accepted:
                        OVERLAY.main()
                    else:
                        with self.assertRaises(SystemExit):
                            OVERLAY.main()
                self.assertEqual(target.read_text(), src)

    def test_failed_publication_preserves_loader_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "weight_utils.py"
            target.write_text(FIXTURE)
            target.chmod(0o644)
            with patch.object(OVERLAY, "TARGET", target), patch.object(sys, "argv", ["patch"]):
                with patch.object(OVERLAY.os, "replace", side_effect=OSError("publish failed")):
                    with self.assertRaisesRegex(OSError, "publish failed"):
                        OVERLAY.main()
                self.assertEqual(target.read_text(), FIXTURE)
                self.assertEqual(set(Path(tmp).iterdir()), {target})
                OVERLAY.main()
                self.assertEqual(target.read_text(), OVERLAY.prepare(FIXTURE))
                self.assertEqual(target.stat().st_mode & 0o777, 0o644)
                self.assertEqual(set(Path(tmp).iterdir()), {target})

    @unittest.skipUnless(os.environ.get("GLM53_LOADCLONE_SOURCE"), "set GLM53_LOADCLONE_SOURCE for full-image source probe")
    def test_full_source_patch_and_optional_pr_composition(self):
        source = Path(os.environ["GLM53_LOADCLONE_SOURCE"]).read_text()
        out = OVERLAY.prepare(source)
        self.assertEqual(OVERLAY.prepare(out), out)
        self.assertEqual(function_source(out, "instanttensor_weights_iterator"),
                         function_source(source, "instanttensor_weights_iterator"))
        if os.environ.get("GLM53_PR230_PATCH"):
            pr = module(Path(os.environ["GLM53_PR230_PATCH"]))
            pr.verified_state(source)
            left = OVERLAY.prepare(pr.prepare(source))
            pr.verified_state(out)
            right = pr.prepare(out)
            self.assertEqual(pr.verified_state(right), "patched")
            self.assertEqual(left, right)
            self.assertEqual(OVERLAY.prepare(right), right)

    @unittest.skipUnless(os.environ.get("GLM53_PR230_PATCH"), "set GLM53_PR230_PATCH for exact pinned composition probe")
    def test_exact_pr230_both_orders_and_reapplication(self):
        pr = module(Path(os.environ["GLM53_PR230_PATCH"]))
        # Require the caller's exact pinned artifact, not an arbitrary adapter.
        self.assertEqual(hashlib.sha256(Path(os.environ["GLM53_PR230_PATCH"]).read_bytes()).hexdigest(),
                         PR230_SHA256)
        def apply_pr(src):
            pr.verified_state(src)
            out = pr.prepare(src)
            self.assertEqual(pr.verified_state(out), "patched")
            return out
        left = OVERLAY.prepare(apply_pr(FIXTURE))
        right = apply_pr(OVERLAY.prepare(FIXTURE))
        self.assertEqual(left, right)
        for out in (left, right):
            self.assertEqual(OVERLAY.prepare(apply_pr(OVERLAY.prepare(apply_pr(out)))), out)
            self.assertEqual(function_source(out, "instanttensor_weights_iterator"),
                             function_source(apply_pr(FIXTURE), "instanttensor_weights_iterator"))
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "one.safetensors")
                save_file({"x": torch.tensor([1.0, -0.0])}, path)
                real_clone = torch.Tensor.clone
                for page in (4096, 65536):
                    for optional in ("0", "1"):
                        clones = []
                        def clone(tensor, *args, **kwargs):
                            clones.append(1)
                            return real_clone(tensor, *args, **kwargs)
                        with patch.object(os, "sysconf", return_value=page), patch.dict(os.environ, {
                                "GLM53_LOAD_CLONE": optional, "GLM53_LOAD_PREFETCH": "0",
                                "GLM53_COLD_LOAD_STAGE_MMAP": "1"}), patch.object(torch.Tensor, "clone", clone):
                            ns = runtime(out)
                            values = list(ns["safetensors_weights_iterator"]([path], False))
                            self.assertEqual(len(clones), int(page != 4096 or optional == "1"))
                            self.assertTrue(torch.equal(values[0][1], torch.tensor([1.0, -0.0])))


# overlay/patch_cold_load_uma.py at PR230 head bc97cbf1d5ab06df778b79b99efa92997bb4f153.
PR230_SHA256 = "b52b45325d2525e52f7c0fc9f82edf7b2eb564fe2b4cacc5433fd4a32c0734c9"

if __name__ == "__main__":
    unittest.main()
