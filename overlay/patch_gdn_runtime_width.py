#!/usr/bin/env python3
"""Preserve active runtime-K width in GDN speculative metadata.

The pinned vLLM build allocates GDN state metadata for the server's maximum
speculative width and accidentally exposes that full width after selecting a
shorter runtime verification horizon.  GLM-5.3-Flash reads the exposed width
as ``max_query_len`` in each KDA layer, so max K7 / runtime K3 retains K7
kernel metadata.

This is a fail-closed runtime backport of vllm-project/vllm#53542 commit
e981ca592d9c91a7d28464c6fc7256495bcb8642 for the v0.1.dev20051+g487ecf187
source shipped by the Mia image. The backing staging allocation remains
max-width; only the selected/copied/returned view narrows.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


P = Path(
    os.environ.get(
        "GLM53_GDN_ATTN_PY",
        "/usr/local/lib/python3.12/dist-packages/"
        "vllm/v1/attention/backends/gdn_attn.py",
    )
)
MARK = "# [glm53-gdn-runtime-width]"

INIT_OLD = """        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
"""
INIT_NEW = """        spec_sequence_masks_cpu: torch.Tensor | None = None
        active_spec_width = 0  # [glm53-gdn-runtime-width]
        if (
"""

QUERY_OLD = """            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
"""
QUERY_NEW = """            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            # Keep max-K storage, but expose only the widest active query in
            # this step. This is CPU metadata and introduces no GPU sync.
            active_spec_width = int(
                query_lens_cpu[spec_sequence_masks_cpu].max().item()
            )
            if not 1 < active_spec_width <= self.num_spec + 1:
                raise RuntimeError(
                    "invalid active GDN speculative width "
                    f"{active_spec_width}; expected 2..{self.num_spec + 1}"
                )

            # Use CPU tensors to avoid CPU-GPU sync
"""

SELECT_OLD = """                    spec_sequence_masks_cpu, : self.num_spec + 1
"""
SELECT_NEW = """                    spec_sequence_masks_cpu, :active_spec_width
"""

STAGE_OLD = """            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)
"""
STAGE_NEW = """            assert spec_sequence_masks is not None
            if active_spec_width <= 0:
                raise RuntimeError("active GDN speculative width was not initialized")
            self.spec_state_indices_tensor[
                :num_spec_decodes, :active_spec_width
            ].copy_(spec_state_indices_tensor, non_blocking=True)
            spec_state_indices_tensor = self.spec_state_indices_tensor[
                :batch_size, :active_spec_width
            ]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)
"""


def replace_exact(text: str, old: str, new: str, count: int, label: str) -> str:
    found = text.count(old)
    if found != count:
        raise SystemExit(f"{P}: expected {count} {label} target(s), found {found}")
    return text.replace(old, new, count)


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text:
        checks = {
            "marker": text.count(MARK) == 1,
            "initialization": text.count(INIT_NEW) == 1,
            "query width": text.count(QUERY_NEW) == 1,
            "state selections": text.count(SELECT_NEW) == 2,
            "FULL-graph staging": text.count(STAGE_NEW) == 1,
            "stock state selection removed": SELECT_OLD not in text,
            "stock FULL-graph staging removed": STAGE_OLD not in text,
        }
        failed = [label for label, ok in checks.items() if not ok]
        if failed:
            raise SystemExit(
                f"{P}: incomplete or drifted existing patch: {', '.join(failed)}"
            )
        print(f"{P.name}: {MARK} already present — skipping")
        return 0

    text = replace_exact(text, INIT_OLD, INIT_NEW, 1, "initialization")
    text = replace_exact(text, QUERY_OLD, QUERY_NEW, 1, "query-width")
    text = replace_exact(text, SELECT_OLD, SELECT_NEW, 2, "state selection")
    text = replace_exact(text, STAGE_OLD, STAGE_NEW, 1, "FULL-graph staging")
    P.write_text(text)
    print(f"patched {P.name} (active runtime-K GDN metadata width)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
