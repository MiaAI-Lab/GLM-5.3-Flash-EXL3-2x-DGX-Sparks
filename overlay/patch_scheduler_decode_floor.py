#!/usr/bin/env python3
"""Optional protection from mixing decode with a long sparse-MLA prefill.

Issue #6: max_num_batched_tokens=1024 is the whole engine step. A decode
lane needs ~8 tokens (1 + DFlash2 k=7); the leftover ~1016 go to a peer
FLASHINFER_MLA_SPARSE_SM120 prefill chunk (~1.5 s). Decode still runs, but
at ~5 tok/s instead of ~50.

A 128-token mixed cap is not enough on 80k KV: the indexer has a large
per-step cost, so mixed decode stays ~10 tok/s. However, skipping every
prefill while a peer decodes can starve another agent for the entire
response. Default to the stock chunked-prefill scheduler so both make
progress. Decode protection remains an explicit throughput tradeoff.

GLM53_MIXED_PREFILL_CHUNK:
  skip / -1  — do not mix prefill with decode; new prompts may starve
  N>0        — cap mixed prefill chunks to N tokens (128 still stalls ~10 tok/s)
  0 / off    — stock chunked-prefill scheduling (default)

Fail closed if the vLLM scheduler anchors drift.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)
MARK = "# [glm53-decode-floor]"

IMPORT_OLD = "import itertools\nimport time\n"
IMPORT_NEW = "import itertools\nimport os\nimport time\n"

HELPER = '''
def _glm53_mixed_prefill_policy(running, current):
    """Mixed-step prefill policy when a peer in `running` is decoding.

    None = no extra policy. 0 = skip this prefill this step. N>0 = cap.
    """
    raw = os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "0").strip().lower()
    if raw in ("0", "off", "no"):
        return None
    if raw in ("skip", "-1"):
        cap = 0
    else:
        try:
            cap = int(raw)
        except ValueError:
            cap = 0
        if cap <= 0:
            return None
    cur_id = getattr(current, "request_id", None)
    for r in running:
        if r is current or getattr(r, "request_id", None) == cur_id:
            continue
        if r.num_computed_tokens >= r.num_prompt_tokens:
            return cap
    return None


'''

RUNNING_OLD = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )

            # Make sure the input position does not exceed the max model len.
"""

RUNNING_NEW = """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
            mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
            if mixed_cap is not None and request.num_computed_tokens < request.num_prompt_tokens:
                num_new_tokens = min(num_new_tokens, mixed_cap)

            # Make sure the input position does not exceed the max model len.
"""

WAITING_OLD = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
"""

WAITING_NEW = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    mixed_cap = _glm53_mixed_prefill_policy(self.running, request)  # [glm53-decode-floor]
                    if mixed_cap is not None and num_computed_tokens < request.num_prompt_tokens:
                        if mixed_cap <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, mixed_cap)

                    # chunked prefill has to be enabled explicitly to allow
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text or "_glm53_mixed_prefill_policy" in text:
        legacy_helper = HELPER.replace(
            'os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "0")',
            'os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "skip")',
        )
        # Validate every owned helper/gate before upgrading or writing.
        helpers = [
            node for node in ast.parse(text).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_glm53_mixed_prefill_policy"
        ]
        canonical_helpers = {
            ast.dump(ast.parse(body).body[0], include_attributes=False)
            for body in (HELPER, legacy_helper)
        }
        helper_count = text.count(HELPER) + text.count(legacy_helper)
        if (
            helper_count != 1
            or text.count("def _glm53_mixed_prefill_policy(") != 1
            or text.count(MARK) != 2
            or text.count(RUNNING_NEW) != 1
            or len(helpers) != 1
            or ast.dump(helpers[0], include_attributes=False) not in canonical_helpers
            or text.count("_glm53_mixed_prefill_policy") != 3
            or text.count(WAITING_NEW) != 1
        ):
            raise SystemExit(f"{P}: mixed-prefill patch regions drifted")
        updated = text.replace(legacy_helper, HELPER, 1)
        compile(updated, str(P), "exec")
        if updated != text:
            P.write_text(updated)
            print(f"{P.name}: upgraded mixed-prefill default to stock scheduling")
        else:
            print(f"{P.name}: {MARK} already present — validated")
        return 0
    if "import os\n" not in text.split("import time\n", 1)[0]:
        text = replace_once(text, IMPORT_OLD, IMPORT_NEW, "import os")
    if "def _glm53_mixed_prefill_policy(" not in text:
        needle = "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
        if text.count(needle) != 1:
            raise SystemExit(f"{P}: helper insert point not unique")
        text = text.replace(needle, HELPER + needle, 1)
    text = replace_once(text, RUNNING_OLD, RUNNING_NEW, "running-prefill")
    text = replace_once(text, WAITING_OLD, WAITING_NEW, "waiting-prefill")
    compile(text, str(P), "exec")
    P.write_text(text)
    cap = os.environ.get("GLM53_MIXED_PREFILL_CHUNK", "0")
    print(f"patched {P.name} (mixed prefill policy={cap})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
