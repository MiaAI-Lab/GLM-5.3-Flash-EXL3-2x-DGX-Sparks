#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'overlay/patch_exl3_fat_kernel.py'
if not path.exists():
    path = Path('/opt/glm53/patch_exl3_fat_kernel.py')
spec = importlib.util.spec_from_file_location('fat_patch', path)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

SOURCE = '''#include "quant/exl3_moe.cuh"
void bind() {
    m.def("exl3_moe", &exl3_moe, "exl3_moe");
}
'''
NAMES = ('exl3_fat_gemm', 'exl3_fat_gemm_pair', 'exl3_fat_swiglu', 'exl3_fat_gemm_scatter')

class TestFatPatch(unittest.TestCase):
    def test_fresh_and_repeat(self):
        once = patch.patch_bindings(SOURCE)
        self.assertEqual(once, patch.patch_bindings(once))
        for name in NAMES:
            self.assertEqual(once.count(f'm.def("{name}"'), 1)

    def test_upgrade_original_patch(self):
        old = patch.patch_bindings(SOURCE)
        for name in ('exl3_fat_gemm_pair', 'exl3_fat_swiglu'):
            old = old.replace(f'    m.def("{name}", &{name}, "{name}");\n', '')
        upgraded = patch.patch_bindings(old)
        for name in NAMES:
            self.assertEqual(upgraded.count(f'm.def("{name}"'), 1)
        self.assertEqual(upgraded, patch.patch_bindings(upgraded))

    def test_drift_and_duplicates(self):
        for broken in (SOURCE.replace('m.def', 'changed'),
                       patch.patch_bindings(SOURCE) +
                       '    m.def("exl3_fat_gemm", &exl3_fat_gemm, "exl3_fat_gemm");'):
            with self.assertRaises(RuntimeError):
                patch.patch_bindings(broken)

if __name__ == '__main__':
    unittest.main()
