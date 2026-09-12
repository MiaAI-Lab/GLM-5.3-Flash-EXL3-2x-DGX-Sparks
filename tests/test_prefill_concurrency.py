#!/usr/bin/env python3
"""CPU regressions for prompt progress during a peer's long generation.

Set GLM53_SCHEDULER_PY_SRC to an installed scheduler.py to also exercise
upgrade/idempotence against the deployed runtime, without loading a model.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay" / "patch_scheduler_decode_floor.py"
OVERLAY = runpy.run_path(str(PATCH))


def policy_from(source: str):
    module = ast.parse(source)
    fn = next(node for node in module.body if isinstance(node, ast.FunctionDef)
              and node.name == "_glm53_mixed_prefill_policy")
    namespace = {"os": os}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "scheduler.py", "exec"), namespace)
    return namespace[fn.name]


def request(name: str, prompt: int, computed: int = 0):
    return SimpleNamespace(request_id=name, num_prompt_tokens=prompt,
                           num_computed_tokens=computed)


class PrefillProgressTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("GLM53_MIXED_PREFILL_CHUNK", None)
        self.policy = policy_from(OVERLAY["HELPER"])

    def test_new_agent_finishes_prefill_while_peer_keeps_decoding(self):
        peer = request("long-generation", 512, 512)
        new_agent = request("new-agent", 48_000)
        # The stock scheduler offers its remaining per-step token budget.
        # The overlay must not suppress that progress for an entire response.
        for _ in range(64):
            peer.num_computed_tokens += 8
            available = 1024 - 8
            cap = self.policy([peer, new_agent], new_agent)
            if cap is not None:
                available = min(available, cap)
            new_agent.num_computed_tokens += min(
                available, new_agent.num_prompt_tokens - new_agent.num_computed_tokens)
        self.assertEqual(new_agent.num_computed_tokens, new_agent.num_prompt_tokens)
        self.assertGreater(peer.num_computed_tokens, peer.num_prompt_tokens)

    def test_default_and_explicit_stock_modes_allow_both_scheduler_paths(self):
        peer = request("decode", 100, 200)
        newcomer = request("prefill", 1000)
        for mode in (None, "0", "off", "no"):
            with self.subTest(mode=mode):
                if mode is None:
                    os.environ.pop("GLM53_MIXED_PREFILL_CHUNK", None)
                else:
                    os.environ["GLM53_MIXED_PREFILL_CHUNK"] = mode
                self.assertIsNone(self.policy([peer], newcomer))  # waiting admission
                self.assertIsNone(self.policy([peer, newcomer], newcomer))  # running prefill

    def test_explicit_decode_protection_is_still_available(self):
        peer = request("decode", 100, 200)
        newcomer = request("prefill", 1000)
        for mode in ("skip", "-1"):
            with self.subTest(mode=mode):
                os.environ["GLM53_MIXED_PREFILL_CHUNK"] = mode
                self.assertEqual(self.policy([peer], newcomer), 0)
                self.assertEqual(self.policy([peer, newcomer], newcomer), 0)

    def test_positive_cap_applies_only_when_another_request_decodes(self):
        os.environ["GLM53_MIXED_PREFILL_CHUNK"] = "128"
        newcomer = request("new", 1000)
        self.assertEqual(self.policy([request("peer", 100, 200)], newcomer), 128)
        self.assertIsNone(self.policy([request("peer", 100, 50)], newcomer))
        self.assertIsNone(self.policy([newcomer], newcomer))
        self.assertIsNone(self.policy([], newcomer))

    def test_request_cannot_count_itself_as_a_decoding_peer(self):
        os.environ["GLM53_MIXED_PREFILL_CHUNK"] = "skip"
        current = request("same", 100, 200)
        self.assertIsNone(self.policy([current], current))
        self.assertIsNone(self.policy([request("same", 100, 200)], current))

    def test_launcher_and_example_enable_progress(self):
        # Execute the real launcher's assignment, including shell defaulting.
        line = next(line for line in (ROOT / "start.sh").read_text().splitlines()
                    if line.startswith('GLM53_MIXED_PREFILL_CHUNK='))
        result = subprocess.check_output(
            ["bash", "-c", line + '\nprintf "%s" "$GLM53_MIXED_PREFILL_CHUNK"'],
            text=True, env=os.environ.copy())
        peer, newcomer = request("peer", 100, 200), request("new", 1000)
        os.environ["GLM53_MIXED_PREFILL_CHUNK"] = result
        self.assertIsNone(self.policy([peer], newcomer))
        example = next(line.partition("=")[2] for line in (ROOT / ".env.example").read_text().splitlines()
                       if line.startswith("GLM53_MIXED_PREFILL_CHUNK="))
        os.environ["GLM53_MIXED_PREFILL_CHUNK"] = example
        self.assertIsNone(self.policy([peer], newcomer))


class InstalledSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ.get("GLM53_SCHEDULER_PY_SRC",
                      "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"))
        if not source.is_file():
            raise unittest.SkipTest("set GLM53_SCHEDULER_PY_SRC for installed-runtime integration")
        cls.source = source.read_text()

    def apply(self, path: Path, *, succeeds=True):
        env = os.environ.copy()
        env["GLM53_SCHEDULER_PY"] = str(path)
        result = subprocess.run([sys.executable, str(PATCH)], env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)

    def test_existing_image_default_is_upgraded_and_reapply_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "scheduler.py"
            legacy = self.source.replace(
                'os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "0")',
                'os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "skip")')
            dst.write_text(legacy)
            self.apply(dst)
            updated = dst.read_text()
            compile(updated, str(dst), "exec")
            with patch.dict(os.environ):
                os.environ.pop("GLM53_MIXED_PREFILL_CHUNK", None)
                self.assertIsNone(policy_from(updated)([request("peer", 100, 200)], request("new", 1000)))
            self.apply(dst)
            self.assertEqual(dst.read_text(), updated)
            self.assertEqual(updated.count(OVERLAY["MARK"]), 2)

    def test_ambiguous_legacy_upgrade_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "scheduler.py"
            legacy = self.source.replace(
                'os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "0")',
                'os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "skip")')
            ambiguous = legacy + '\nother = os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "skip")\n'
            dst.write_text(ambiguous)
            self.apply(dst, succeeds=False)
            self.assertEqual(dst.read_text(), ambiguous)


if __name__ == "__main__":
    unittest.main()
