#!/usr/bin/env python3
"""Use resolved scheduler checkpoints without starving small prefill grants.

Shared checkpoints use the scheduler LCM, not the engine's smallest cache page
or the largest Mamba page. Private state in a smaller Mamba page must stop at
that page's next boundary before its slot can be published under a full hash.
Only participating non-drafter EAGLE groups require checkpoint backoff.

The decode-floor overlay owns per-request caps; this overlay owns alignment,
positive-grant progress and checkpoint backoff. Their anchors do not overlap,
so either application order is supported. Decode-floor versions before v5 are
rejected because they do not preserve per-request alignment caps.

Pristine sources and exact retained v1 installations migrate atomically.
Partial or drifted installations fail before writing.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_SCHEDULER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py",
    )
)
MARK = "# [glm53-mamba-align-chunking-v2]"
LEGACY_MARK = "# [glm53-mamba-align-chunking-v1]"
DECODE_FLOOR_MARK = "# [glm53-decode-floor"
DECODE_FLOOR_V5 = "# [glm53-decode-floor:v5]"

IMPORT_OLD = """from vllm.v1.kv_cache_interface import KVCacheConfig
"""
IMPORT_NEW = """from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
"""
INIT_OLD = """        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
"""
INIT_V1 = """        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        # [glm53-mamba-align-chunking-v1] The block whose boundaries carry
        # cacheable SSM states is the Mamba groups' own; cache_config.block_size
        # is the smallest prefix-caching block (the drafter's page here).
        self.mamba_align_block_size = max(
            (
                group.kv_cache_spec.block_size
                for group in kv_cache_config.kv_cache_groups
                if isinstance(group.kv_cache_spec, MambaSpec)
            ),
            default=self.cache_config.block_size,
        )
        # Back off the last cacheable position only when full attention
        # (group 0) really drops its last matching block, i.e. it is one of
        # the coordinator's EAGLE groups.
        self.mamba_align_eagle_backoff = self.use_eagle and 0 in getattr(
            self.kv_cache_manager.coordinator, "eagle_group_ids", {0}
        )
"""
INIT_NEW = INIT_OLD + """        # [glm53-mamba-align-chunking-v2] Smaller private state pages can
        # finish before the shared LCM checkpoint. Cache their sizes off-path.
        self.mamba_align_sub_block_sizes = tuple(sorted({
            group.kv_cache_spec.block_size
            for group in kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, MambaSpec)
            and group.kv_cache_spec.block_size != self.block_size
        }))
        # Drafter-only lookahead does not prune a target checkpoint. Non-SWA
        # EAGLE participants, including the MTP fallback, still require backoff.
        self.mamba_align_eagle_backoff = self.use_eagle and any(
            i in self.kv_cache_manager.coordinator.eagle_group_ids
            and group.kv_cache_spec.participates_in_prefix_caching
            and type(group.kv_cache_spec).__name__ != "SlidingWindowSpec"
            for i, group in enumerate(kv_cache_config.kv_cache_groups)
        )
"""
SPLIT_OLD = """        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
"""
SPLIT_V1 = """        block_size = self.mamba_align_block_size  # [glm53-mamba-align-chunking-v1]
        # The last block-aligned position whose state can be cached. When
        # full attention prunes its last matching block (EAGLE), back off one
        # block to avoid a Mamba cache miss.
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.mamba_align_eagle_backoff:
            last_cache_position = max(last_cache_position - block_size, 0)
"""
SPLIT_NEW = """        block_size = self.block_size  # [glm53-mamba-align-chunking-v2]
        # A reusable shared checkpoint must satisfy every participating page.
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.mamba_align_eagle_backoff:
            last_cache_position = max(last_cache_position - block_size, 0)
"""
GRANT_OLD = """        if end < prefill_end:
            max_prefill_tokens = self.max_num_scheduled_tokens
"""
GRANT_NEW = """        # [glm53-mamba-align-chunking-v2] A sub-page grant cannot
        # reach a full checkpoint from an aligned start. From a private start,
        # the mandatory stops below catch its next shared or smaller boundary.
        # Only grants spanning a full page may enter full-page rounding.
        if end < prefill_end and num_new_tokens >= block_size:
            max_prefill_tokens = self.max_num_scheduled_tokens
"""
STOP_OLD = """        next_block_boundary = (start // block_size + 1) * block_size
"""
STOP_NEW = """        next_block_boundary = (start // block_size + 1) * block_size
        # A mid-page private state cannot be hashed as a complete smaller page.
        for state_block in self.mamba_align_sub_block_sizes:
            if start % state_block:
                next_block_boundary = min(
                    next_block_boundary, (start // state_block + 1) * state_block
                )
"""
EDITS = (
    ("import", IMPORT_OLD, IMPORT_NEW),
    ("init", INIT_OLD, INIT_NEW),
    ("split", SPLIT_OLD, SPLIT_NEW),
    ("grant-bounded rounding", GRANT_OLD, GRANT_NEW),
    ("private-state stop", STOP_OLD, STOP_NEW),
)
LEGACY_EDITS = (
    ("import", IMPORT_OLD, IMPORT_NEW),
    ("init", INIT_OLD, INIT_V1),
    ("split", SPLIT_OLD, SPLIT_V1),
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def verify_complete(text: str, edits=EDITS) -> list[str]:
    problems = [
        f"{label}: expected exactly one applied block, found {n}"
        for label, _, new in edits
        if (n := text.count(new)) != 1
    ]
    stripped = text
    for _, _, new in edits:
        stripped = stripped.replace(new, "")
    problems += [
        f"{label}: superseded form still present"
        for label, old, _ in edits if old in stripped
    ]
    if LEGACY_MARK in stripped or MARK in stripped:
        problems.append("unrecognized alignment stage remains")
    try:
        compile(text, str(P), "exec")
    except SyntaxError as exc:
        problems.append(f"syntax: {exc.msg}")
    return problems


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    original = text = P.read_text()
    if DECODE_FLOOR_MARK in text and DECODE_FLOOR_V5 not in text:
        raise SystemExit(
            f"{P}: patch_scheduler_decode_floor.py older than v5 present; its "
            "per-request cap is not applied to the Mamba alignment"
        )
    if LEGACY_MARK in text:
        if problems := verify_complete(text, LEGACY_EDITS):
            raise SystemExit(f"{P}: drifted retained overlay: " + "; ".join(problems))
        for label, old, new in LEGACY_EDITS:
            text = replace_once(text, new, old, label)
    if MARK not in text:
        for label, old, new in EDITS:
            text = replace_once(text, old, new, label)
    if problems := verify_complete(text):
        raise SystemExit(f"{P}: incomplete or drifted overlay state: " + "; ".join(problems))
    if text != original:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=P.parent, delete=False) as out:
                temporary = Path(out.name)
                out.write(text)
            temporary.chmod(P.stat().st_mode)
            os.replace(temporary, P)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"patched {P.name} ({MARK})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
