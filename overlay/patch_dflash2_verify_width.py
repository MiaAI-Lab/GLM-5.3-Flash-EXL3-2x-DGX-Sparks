#!/usr/bin/env python3
"""Decouple DFlash2 proposal width from target verification width.

DFlash2 is trained to propose a fixed parallel block.  Low-acceptance workloads
can benefit from keeping that trained proposal intact while publishing only a
batch-uniform prefix to the target verifier.  Standard rejection sampling is
unchanged; this only limits how many proposal positions the scheduler sees.

DFLASH_VERIFY_TOKENS:
  0 / unset  — publish the full proposal (stock behavior)
  N>0        — publish the first N proposal tokens

The patch is deliberately fail-closed if the vLLM source anchors drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


P = Path(
    os.environ.get(
        "GLM53_DRAFT_UTILS_PY",
        "/usr/local/lib/python3.12/dist-packages/"
        "vllm/v1/worker/gpu/spec_decode/utils.py",
    )
)
MARK = "# [glm53-dflash2-verify-width]"

IMPORT_OLD = "import numpy as np\nimport torch\n"
IMPORT_NEW = "import os\n\nimport numpy as np\nimport torch\n"

SET_OLD = """    ) -> None:
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
"""

SET_NEW = """    ) -> None:
        raw_limit = os.environ.get("DFLASH_VERIFY_TOKENS", "0").strip()
        try:
            verify_tokens = int(raw_limit or "0")
        except ValueError as exc:
            raise RuntimeError(
                f"DFLASH_VERIFY_TOKENS must be an integer, got {raw_limit!r}"
            ) from exc
        proposal_tokens = draft_tokens.shape[1]
        if verify_tokens < 0 or verify_tokens > proposal_tokens:
            raise RuntimeError(
                "DFLASH_VERIFY_TOKENS must be 0 or between 1 and the DFlash2 "
                f"proposal width ({proposal_tokens}), got {verify_tokens}"
            )
        # Grammar validation can shorten different requests by different
        # amounts. Keep stock full width there so this overlay never introduces
        # a new ragged batch on a UNIFORM_BATCH-only attention backend.
        if verify_tokens and not input_batch.has_structured_output_reqs:
            draft_tokens = draft_tokens[:, :verify_tokens]  # [glm53-dflash2-verify-width]
        self.req_ids = input_batch.req_ids
        self.num_draft_tokens = draft_tokens.shape[1]
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text:
        checks = {
            "marker": text.count(MARK) == 1,
            "patched import": text.count(IMPORT_NEW) == 1,
            "patched DraftTokensHandler": text.count(SET_NEW) == 1,
            "stock DraftTokensHandler removed": SET_OLD not in text,
        }
        failed = [label for label, ok in checks.items() if not ok]
        if failed:
            raise SystemExit(
                f"{P}: incomplete or drifted existing patch: {', '.join(failed)}"
            )
        print(f"{P.name}: {MARK} already present — skipping")
        return 0
    text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import block")
    text = replace_once(text, SET_OLD, SET_NEW, "DraftTokensHandler")
    P.write_text(text)
    limit = os.environ.get("DFLASH_VERIFY_TOKENS", "0") or "0"
    print(f"patched {P.name} (DFlash2 verify tokens={limit})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
