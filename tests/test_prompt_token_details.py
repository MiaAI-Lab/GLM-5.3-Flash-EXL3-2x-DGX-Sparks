#!/usr/bin/env python3
"""CPU-only checks of the actual head/worker base argv; never run a launcher."""
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAG = "--enable-prompt-tokens-details"


class PromptTokenDetailsTests(unittest.TestCase):
    def test_all_topologies_and_ranks(self):
        for name in ("start.sh", "start-tp3.sh", "start-tp4.sh"):
            source = (ROOT / name).read_text()
            blocks = re.findall(r"^ARGS=\(\n(.*?)^\)", source, re.M | re.S)
            self.assertEqual(len(blocks), 2, name)
            for rank, block in zip(("head", "worker"), blocks):
                with self.subTest(launcher=name, rank=rank):
                    # Only evaluate the base array, never the surrounding script.
                    argv = subprocess.check_output(
                        ["bash", "--noprofile", "--norc", "-c",
                         'ARGS=(\n' + block + '\n)\nprintf "%s\\0" "${ARGS[@]}"'],
                        env={"PATH": os.defpath},
                    ).decode().split("\0")[:-1]
                    self.assertEqual(argv.count(FLAG), 1)
                    self.assertNotIn("--enable-prompt-token-details", argv)
                    self.assertNotIn("--no-enable-prompt-tokens-details", argv)
                    self.assertEqual("--headless" in argv, rank == "worker")


if __name__ == "__main__":
    unittest.main()
