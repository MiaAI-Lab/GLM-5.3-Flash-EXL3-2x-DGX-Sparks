#!/usr/bin/env python3
"""Unit-test the fail-closed DFlash2 verification-width runtime patch."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent
PATCH = HERE.parent / "overlay" / "patch_dflash2_verify_width.py"

FIXTURE = '''from __future__ import annotations

import numpy as np
import torch


class DraftTokensHandler:
    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
        if not input_batch.has_structured_output_reqs:
            self.draft_tokens_np = None
            return
        self.draft_tokens_np = draft_tokens
'''


class FakeTensor:
    def __init__(self, rows: list[list[int]]):
        self.rows = rows

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.rows[0])

    def __getitem__(self, key):
        row_sel, col_sel = key
        assert row_sel == slice(None)
        return FakeTensor([row[col_sel] for row in self.rows])


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "utils.py"
        dst.write_text(FIXTURE)
        env = os.environ.copy()
        env["GLM53_DRAFT_UTILS_PY"] = str(dst)
        env["DFLASH_VERIFY_TOKENS"] = "3"
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        text = dst.read_text()
        assert text.count("[glm53-dflash2-verify-width]") == 1
        assert 'draft_tokens = draft_tokens[:, :verify_tokens]' in text
        assert 'self.num_draft_tokens = draft_tokens.shape[1]' in text
        compile(text, str(dst), "exec")

        # Execute the patched method with dependency stubs. Ordinary batches
        # publish a uniform prefix; structured batches retain full width.
        fake_np = types.ModuleType("numpy")
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = FakeTensor
        old_np = sys.modules.get("numpy")
        old_torch = sys.modules.get("torch")
        sys.modules["numpy"] = fake_np
        sys.modules["torch"] = fake_torch
        try:
            namespace: dict[str, object] = {}
            exec(compile(text, str(dst), "exec"), namespace)
        finally:
            if old_np is None:
                del sys.modules["numpy"]
            else:
                sys.modules["numpy"] = old_np
            if old_torch is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = old_torch

        handler_cls = namespace["DraftTokensHandler"]
        tensor = FakeTensor([list(range(7)), list(range(10, 17))])
        ordinary = types.SimpleNamespace(
            req_ids=["a", "b"], has_structured_output_reqs=False
        )
        old_verify = os.environ.get("DFLASH_VERIFY_TOKENS")
        os.environ["DFLASH_VERIFY_TOKENS"] = "3"
        try:
            handler = handler_cls()
            handler.set_draft_tokens(ordinary, tensor)
            assert handler.num_draft_tokens == 3

            os.environ["DFLASH_VERIFY_TOKENS"] = "0"
            handler = handler_cls()
            handler.set_draft_tokens(ordinary, tensor)
            assert handler.num_draft_tokens == 7

            os.environ["DFLASH_VERIFY_TOKENS"] = "8"
            handler = handler_cls()
            try:
                handler.set_draft_tokens(ordinary, tensor)
            except RuntimeError as exc:
                assert "proposal width (7)" in str(exc)
            else:
                raise AssertionError("out-of-range verification width was accepted")

            os.environ["DFLASH_VERIFY_TOKENS"] = "not-an-integer"
            handler = handler_cls()
            try:
                handler.set_draft_tokens(ordinary, tensor)
            except RuntimeError as exc:
                assert "must be an integer" in str(exc)
            else:
                raise AssertionError("non-integer verification width was accepted")

            os.environ["DFLASH_VERIFY_TOKENS"] = "3"
            structured = types.SimpleNamespace(
                req_ids=["a", "b"], has_structured_output_reqs=True
            )
            handler = handler_cls()
            handler.set_draft_tokens(structured, tensor)
            assert handler.num_draft_tokens == 7
            assert handler.draft_tokens_np.shape == (2, 7)
        finally:
            if old_verify is None:
                del os.environ["DFLASH_VERIFY_TOKENS"]
            else:
                os.environ["DFLASH_VERIFY_TOKENS"] = old_verify

        # Applying the overlay twice must not duplicate it.
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        assert dst.read_text().count("[glm53-dflash2-verify-width]") == 1

        # A stray marker must not bypass validation of the complete patch.
        incomplete = Path(tmp) / "incomplete.py"
        incomplete.write_text(text.replace("        self.req_ids = input_batch.req_ids\n", "", 1))
        env["GLM53_DRAFT_UTILS_PY"] = str(incomplete)
        result = subprocess.run(
            [sys.executable, str(PATCH)], env=env, capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "incomplete or drifted existing patch" in result.stderr

        # Anchor drift must fail rather than silently run without the limit.
        drifted = Path(tmp) / "drifted.py"
        drifted.write_text("import numpy as np\nimport torch\n")
        env["GLM53_DRAFT_UTILS_PY"] = str(drifted)
        result = subprocess.run(
            [sys.executable, str(PATCH)], env=env, capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "expected one DraftTokensHandler target" in result.stderr

    print("DFlash2 verification-width patch OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
