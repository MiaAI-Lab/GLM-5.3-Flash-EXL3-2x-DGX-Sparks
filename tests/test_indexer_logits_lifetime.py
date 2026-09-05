#!/usr/bin/env python3
"""Patch drift/idempotence and Python allocation-lifetime regression tests."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
import weakref

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / 'overlay/patch_indexer_logits_lifetime.py'
if not PATCH.exists():
    PATCH = Path('/opt/glm53/patch_indexer_logits_lifetime.py')
spec = importlib.util.spec_from_file_location('logits_patch', PATCH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

SOURCE = '''def run(chunks, make_logits, torch, topk_dst, index_kpool=1):
    if chunks:
        for chunk in chunks:
            logits = make_logits(chunk)
            torch.ops._C.top_k_per_row_prefill(logits, topk_dst)
            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
    # A following decode allocation must not overlap the final prefill logits.
    decode = make_logits([19, 7])
    return topk_dst
'''


class TestLogitsLifetime(unittest.TestCase):
    def test_idempotence(self):
        once = patch.patched_source(SOURCE)
        self.assertEqual(once, patch.patched_source(once))

    def test_anchor_drift(self):
        for source in (SOURCE.replace('pool_ids =', 'changed ='), SOURCE + SOURCE):
            with self.assertRaises(RuntimeError):
                patch.patched_source(source)

    def test_atomic_file_and_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'indexer.py'
            target.write_text(SOURCE)
            target.chmod(0o640)
            patch.apply(target)
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertEqual(target.read_text(), patch.patched_source(SOURCE))
            patch.apply(target)

    def test_consumed_chunks_do_not_overlap(self):
        class Logits:
            def __init__(self, values):
                self.values = values
        def execute(source, chunks):
            refs, live_before, output = [], [], []
            def make(values):
                live_before.append(sum(ref() is not None for ref in refs))
                logits = Logits(values)
                refs.append(weakref.ref(logits))
                return logits
            def topk(logits, dst):
                dst.append(sorted(logits.values, reverse=True)[:2])
            torch = types.SimpleNamespace(ops=types.SimpleNamespace(
                _C=types.SimpleNamespace(top_k_per_row_prefill=topk)))
            env = {}
            exec(compile(source, '<indexer-fixture>', 'exec'), env)
            env['run'](chunks, make, torch, output)
            return output, live_before
        for chunks in ([], [[1]], [[3, 1, 2], [-7, -1, -3], [0, 0, 1]]):
            before, overlap = execute(SOURCE, chunks)
            after, live = execute(patch.patched_source(SOURCE), chunks)
            self.assertEqual(before, after)
            self.assertEqual(live, [0] * len(live))
            if chunks:
                self.assertIn(1, overlap)


if __name__ == '__main__':
    unittest.main()
