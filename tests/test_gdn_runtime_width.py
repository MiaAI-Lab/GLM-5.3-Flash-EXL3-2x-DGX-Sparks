#!/usr/bin/env python3
"""Unit-test the fail-closed active runtime-K GDN metadata backport."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
PATCH = HERE.parent / "overlay" / "patch_gdn_runtime_width.py"

FIXTURE = '''class Builder:
    def build(self):
        spec_sequence_masks_cpu: torch.Tensor | None = None
        if (
            enabled
        ):
            pass

        if spec_sequence_masks is None:
            pass
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            assert spec_sequence_masks_cpu is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

            # Use CPU tensors to avoid CPU-GPU sync
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]

            if pure_spec:
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]
            else:
                spec_state_indices_tensor = block_table_tensor[
                    spec_sequence_masks_cpu, : self.num_spec + 1
                ]

        if full_graph:
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=True
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(NULL_BLOCK_ID)
'''


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "gdn_attn.py"
        dst.write_text(FIXTURE)
        env = os.environ.copy()
        env["GLM53_GDN_ATTN_PY"] = str(dst)

        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        text = dst.read_text()
        assert text.count("[glm53-gdn-runtime-width]") == 1
        assert text.count(":active_spec_width") == 4
        assert text.count(": self.num_spec + 1") == 0
        assert "query_lens_cpu[spec_sequence_masks_cpu].max().item()" in text
        assert ":num_spec_decodes, :active_spec_width" in text
        assert ":batch_size, :active_spec_width" in text
        compile(text, str(dst), "exec")

        # Applying twice is safe and does not duplicate the backport.
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        assert dst.read_text().count("[glm53-gdn-runtime-width]") == 1

        # A stray marker must not bypass validation of the complete backport.
        incomplete = Path(tmp) / "incomplete.py"
        incomplete.write_text(
            text.replace("                :batch_size, :active_spec_width\n", "", 1)
        )
        env["GLM53_GDN_ATTN_PY"] = str(incomplete)
        result = subprocess.run(
            [sys.executable, str(PATCH)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "incomplete or drifted existing patch" in result.stderr

        # Any source drift fails rather than silently leaving max-K metadata.
        drifted = Path(tmp) / "drifted.py"
        drifted.write_text("spec_sequence_masks_cpu = None\n")
        env["GLM53_GDN_ATTN_PY"] = str(drifted)
        result = subprocess.run(
            [sys.executable, str(PATCH)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "expected 1 initialization target" in result.stderr

    print("GDN active runtime-K width patch OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
