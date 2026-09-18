"""Regression tests for legacy hybrid-overlay upgrades to EAGLE verification v3.

Exercises the installer (overlay/patch_hybrid_prefix_hit.py) against a
self-contained coordinator fixture via subprocess, without loading vLLM or
starting a model. Covers: legacy-to-v3 migration, byte-idempotent repeated
application, and fail-closed rejection of unknown, duplicate or corrupted
verification regions. These tests check migration mechanics only; they are
not a substitute for the exact-source replay checks in
tests/test_hybrid_prefix_hit.py.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PATCH = Path(__file__).resolve().parents[1] / "overlay/patch_hybrid_prefix_hit.py"
V3 = "# [glm53-dflash-eagle-verify-v3]"
LEGACY = """                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                if _glm53_is_draft_swa_spec(spec):  # [glm53-hybrid-apc]
"""
FIXTURE = (
    "# [glm53-hybrid-apc]\n# [glm53-dflash-swa-replay-v1]\n"
    "def _validate_prefix_cache_retention_interval(\n):\n    pass\n"
    "class HybridKVCacheCoordinator:\n"
    "    def find_longest_cache_hit(self):\n"
    "        for idx in []:\n"
    "            if True:\n"
    + LEGACY + "                    pass\n"
)


class UpgradeTests(unittest.TestCase):
    def apply(self, source):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "coordinator.py"
            target.write_text(source)
            result = subprocess.run(
                [sys.executable, "-S", str(PATCH)],
                env={**os.environ, "GLM53_KV_COORDINATOR_PY": str(target)},
                capture_output=True, text=True,
            )
            return result.returncode, target.read_text()

    def test_legacy_marker_does_not_skip_v3(self):
        code, upgraded = self.apply(FIXTURE)
        self.assertEqual(code, 0)
        self.assertIn(V3, upgraded)
        self.assertIn("eagle_verified.discard(idx)", upgraded)
        compile(upgraded, "coordinator.py", "exec")
        code, repeated = self.apply(upgraded)
        self.assertEqual(code, 0)
        self.assertEqual(repeated, upgraded)

    def test_unknown_legacy_region_is_not_written(self):
        source = FIXTURE.replace("eagle_verified.add(idx)", "eagle_verified.add(idx + 1)")
        code, after = self.apply(source)
        self.assertNotEqual(code, 0)
        self.assertEqual(after, source)

    def test_marker_only_cannot_claim_current_verification(self):
        source = FIXTURE + "\n" + V3 + "\n"
        code, after = self.apply(source)
        self.assertNotEqual(code, 0)
        self.assertEqual(after, source)

    def test_duplicate_legacy_regions_are_not_written(self):
        source = FIXTURE.replace(LEGACY, LEGACY + "                    pass\n" + LEGACY)
        code, after = self.apply(source)
        self.assertNotEqual(code, 0)
        self.assertEqual(after, source)

    def test_current_region_with_leftover_legacy_is_not_written(self):
        code, upgraded = self.apply(FIXTURE)
        self.assertEqual(code, 0)
        source = upgraded + "\n" + LEGACY
        code, after = self.apply(source)
        self.assertNotEqual(code, 0)
        self.assertEqual(after, source)

    def test_corrupted_current_region_is_not_written(self):
        code, upgraded = self.apply(FIXTURE)
        self.assertEqual(code, 0)
        source = upgraded.replace("eagle_verified.discard(idx)", "eagle_verified.add(idx)")
        code, after = self.apply(source)
        self.assertNotEqual(code, 0)
        self.assertEqual(after, source)


if __name__ == "__main__":
    unittest.main()
