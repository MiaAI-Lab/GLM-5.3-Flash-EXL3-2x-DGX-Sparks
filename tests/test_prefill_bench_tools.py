"""CPU-only structural tests; these do not validate CUDA arithmetic or memory."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


memory = load('bench_indexer_logits_memory')
kernels = load('bench_exl3_prefill_kernels')

# Original tiny test fixture, not a copy of the vLLM loop.
SOURCE = '''def sparse_attn_indexer_kpool():
    if enabled:
        for chunk in prefill_metadata.chunks:
            logits = helper(chunk)
            topk_indices_buffer.append(logits)
'''
PATCHED = SOURCE + memory.INSERT


class BenchmarkToolsTest(unittest.TestCase):
    def test_help_without_cuda_dependencies(self):
        for filename in ('bench_exl3_prefill_kernels.py', 'bench_indexer_logits_memory.py'):
            result = subprocess.run([sys.executable, '-I', str(HERE / filename), '--help'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)

    def test_fixed_workloads(self):
        self.assertEqual(kernels.MS, (129, 256, 512, 1024, 2048, 4096, 7168))
        self.assertEqual(kernels.PAIR_CASES, (
            ('tail', 173, 4096, 1024), ('medium', 637, 4096, 1024),
            ('large', 2305, 4096, 1024), ('full', 7168, 4096, 1024),
            ('wide', 1025, 2048, 1536)))
        self.assertEqual(memory.CASES, ((1024, 32768), (1777, 75008), (769, 49152)))
        self.assertEqual((memory.SEED, memory.REPEATS), (20260719, 3))

    def test_cli_defaults(self):
        self.assertEqual(kernels.build_parser().parse_args([]).panel, 'all')
        self.assertEqual(kernels.build_parser().parse_args([]).projection_arm, 'both')
        self.assertFalse(memory.build_parser().parse_args([]).require_patch)
        self.assertTrue(memory.build_parser().parse_args(['--require-patch']).require_patch)

    def test_upstream_mode_requires_explicit_work(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            kernels.parse_args(['--validation', 'upstream'])
        args = kernels.parse_args(['--validation', 'upstream', '--panel', 'kernels'])
        self.assertEqual(args.panel, 'kernels')
        args = kernels.parse_args(['--validation', 'upstream', '--projection-arm', 'copies'])
        self.assertEqual((args.panel, args.projection_arm), ('all', 'copies'))

    def test_candidate_never_accepts_missing_new_symbols(self):
        extension = SimpleNamespace(exl3_fat_gemm=lambda: None,
                                    exl3_fat_gemm_scatter=lambda: None)
        module = SimpleNamespace(_check_fat_kernel=lambda: None)
        self.assertEqual(kernels.required_checks(extension, module, 'upstream'),
                         ('_check_fat_kernel',))
        with self.assertRaisesRegex(RuntimeError, 'exl3_fat_gemm_pair'):
            kernels.required_checks(extension, module, 'candidate')
        extension.exl3_fat_gemm_pair = lambda: None
        extension.exl3_fat_swiglu = lambda: None
        with self.assertRaisesRegex(RuntimeError, '_check_fat_pair'):
            kernels.required_checks(extension, module, 'candidate')

    def test_upstream_still_requires_direct_scatter(self):
        with self.assertRaisesRegex(RuntimeError, 'exl3_fat_gemm_scatter'):
            kernels.required_checks(SimpleNamespace(exl3_fat_gemm=lambda: None),
                                    SimpleNamespace(_check_fat_kernel=lambda: None), 'upstream')

    def test_unpatched_reference_is_identity(self):
        self.assertEqual(memory.reference_source(SOURCE), (SOURCE, False))

    def test_exact_patch_removal_only(self):
        self.assertEqual(memory.reference_source(PATCHED), (SOURCE, True))
        self.assertIsInstance(memory.extract_loop(PATCHED), ast.For)

    def test_marker_drift_rejected(self):
        for source in (PATCHED.replace('del logits', 'del logits # changed'),
                       PATCHED + memory.INSERT,
                       PATCHED.replace('del logits', 'pass')):
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                memory.reference_source(source)

    def test_additional_deletion_rejected(self):
        with self.assertRaises(RuntimeError):
            memory.reference_source(PATCHED + '            del logits\n')

    def test_release_outside_loop_rejected(self):
        source = (SOURCE + '    if enabled:\n        if enabled:\n' + memory.INSERT
                  + '            pass\n')
        with self.assertRaises(RuntimeError):
            memory.reference_source(source)

    def test_ambiguous_function_and_loop_rejected(self):
        for source in (SOURCE + SOURCE, 'def other():\n    pass\n',
                       SOURCE + '        for x in prefill_metadata.chunks:\n            pass\n'):
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                memory.extract_loop(source)

    def test_compile_preserves_real_helpers_and_ast(self):
        calls = []

        def helper(chunk):
            calls.append(chunk)
            return chunk * 2

        module = SimpleNamespace(__file__='synthetic_fixture.py', helper=helper)
        inputs = dict(prefill_metadata=SimpleNamespace(chunks=[1, 3, 5]),
                      topk_indices_buffer=[])
        loop = memory.extract_loop(PATCHED)
        before = ast.dump(loop)
        function, namespace = memory.compile_loop(loop, module, inputs, 'test_loop')
        try:
            self.assertIs(namespace['helper'], helper)
            self.assertIs(function(), inputs['topk_indices_buffer'])
            self.assertEqual(inputs['topk_indices_buffer'], [2, 6, 10])
            self.assertEqual(calls, [1, 3, 5])
            self.assertEqual(ast.dump(loop), before)
        finally:
            namespace.clear()

    def test_global_collision_rejected(self):
        module = SimpleNamespace(__file__='synthetic_fixture.py', helper=lambda x: x)
        with self.assertRaises(RuntimeError):
            memory.compile_loop(memory.extract_loop(SOURCE), module, {'helper': None}, 'loop')

    def test_normal_selfcheck_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test_exl3_overlay.py'
            path.write_text('from dataclasses import dataclass\n'
                            '@dataclass\nclass Imported:\n    value: int = 7\n')
            previous = sys.modules.get('overlay_selfcheck')
            try:
                module = kernels.load_selfcheck(path)
                self.assertEqual(module.Imported().value, 7)
                self.assertIs(sys.modules['overlay_selfcheck'], module)
            finally:
                sys.modules.pop('overlay_selfcheck', None)
                if previous is not None:
                    sys.modules['overlay_selfcheck'] = previous

    def test_failed_selfcheck_import_not_stubbed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test_exl3_overlay.py'
            path.write_text('raise RuntimeError("missing real dependency")\n')
            with self.assertRaisesRegex(RuntimeError, 'missing real dependency'):
                kernels.load_selfcheck(path)
            self.assertNotIn('overlay_selfcheck', sys.modules)


if __name__ == '__main__':
    unittest.main()
