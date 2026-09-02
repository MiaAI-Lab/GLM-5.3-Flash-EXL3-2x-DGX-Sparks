#!/usr/bin/env python3
"""Right-size the sparse-indexer prefill gather workspace (opt-in).

``get_max_prefill_buffer_size`` returns ``max_model_len * 40`` entries
(``v1/attention/backends/mla/indexer.py``). Each entry is 128 fp8 bytes plus a
4-byte scale = 132 bytes, so at ``max_model_len=1_000_000`` that is 40,000,000
entries = 5,280,000,000 B. The kpool indexer requests it (plus the 1 MiB radix
top-k scratch) during the *memory profile* — so it is subtracted from the KV
pool — and then it is locked for the life of the process
(``v1/worker/gpu_model_runner.py`` -> ``v1/worker/workspace.py``).

Live receipt, head Spark, Stage A boot 2026-09-01 with ``VLLM_DEBUG_WORKSPACE=1``::

    [WORKSPACE DEBUG] Resized workspace from
      'sparse_attn_indexer_kpool.py:284:sparse_attn_indexer_kpool':
      0.00 MB -> 5036.40 MB (ubatch 0)

5036.40 MiB is 5,281,048,576 B == 40,000,000*132 + 1 MiB radix scratch, to the
byte. The over-allocation is measured, not inferred.

Why it is over-sized
--------------------
The splitter that consumes the workspace (``split_indexer_prefill_chunks``) is
fed COMPRESSED seq lens: the metadata builder divides by ``compress_ratio``
before calling it, and for GLM-5.3-Flash ``compress_ratio == index_kpool == 4``
(``models/glm5next/nvidia/attention.py`` sets ``compress_ratio=index_kpool`` on
the indexer kv-cache spec). The workspace is therefore sized in TOKENS while it
is indexed in POOLS. Upstream already knows this: ``models/deepseek_v4/
attention.py`` divides ``get_max_prefill_buffer_size(vllm_config)`` by
``compress_ratio`` at its call site; ``models/glm5next/nvidia/attention.py``
does not.

What ``rightsize`` computes
---------------------------
The LEGAL MAXIMUM a single step can ask for, not a tuned guess:

    entries = min(max_num_seqs, max_num_batched_tokens)
              * cdiv(max_model_len + num_speculative_tokens, compress_ratio)

* ``min(max_num_seqs, max_num_batched_tokens)`` bounds the prefill requests in
  one step: the scheduler admits at most ``max_num_seqs`` sequences, and every
  prefill row contributes at least one query token to the ``max_num_batched_
  tokens`` budget.
* ``+ num_speculative_tokens`` covers ``seq_lens_cpu_upper_bound``, which the
  builder documents as "an upper bound for async-spec extend rows" and which is
  the tensor actually handed to the splitter.
* ``cdiv`` (not floor) because the consumer floors; cdiv is the upper bound.

Because the result is >= the largest total the splitter can ever be asked to
pack, the ``new_n <= workspace_size`` constraint inside
``split_indexer_prefill_chunks`` can never bind that the stock 40x value did not
already bind. The chunk list is therefore IDENTICAL to stock for every batch the
scheduler can legally form -- see ``split_prefill_chunks`` below and
``tests/test_indexer_workspace.py::test_chunking_is_identical_to_stock``, which
proves it by exhaustion over randomized legal batches rather than asserting it.

The result is also clamped to the stock value, so this patch can only ever
narrow. If a config's legal maximum exceeds ``max_model_len * 40`` the stock
value is returned unchanged and a warning is logged: stock is already under the
legal maximum there, which is an upstream latent risk this overlay does not
inherit and does not fix.

At the live config (M=1e6, kpool=4, max_num_seqs=16, MNBT=2048, k=7):
4,000,032 entries = 528,004,224 B = 503.5 MiB, against 5035.4 MiB. ~4.43 GiB
returns to the KV pool.

Fail-closed, at boot, at the authoritative source
-------------------------------------------------
The sizing runs in ``get_max_prefill_buffer_size``, which only has
``vllm_config``, so it reads ``hf_text_config.index_kpool``. The value the
runtime actually uses is ``self.kv_cache_spec.compress_ratio``, resolved later
in ``DeepseekV32IndexerMetadataBuilder.__init__``. Those are the same number by
construction, but "by construction" is not a check, so the patch installs one:
the builder raises at init if the two disagree, and logs the computed sizing on
every TP rank so a rank mismatch is visible in the boot receipts.

That comparison is unconditional whenever ``rightsize`` is active, and raises in
BOTH directions. It is not guarded on ``self.compress_ratio > 1``: the direction
that actually under-sizes is ``index_kpool=4`` with ``compress_ratio=1``, which
sizes for ~4M compressed entries while the runtime may present ~16M token
entries, and that direction is precisely the one such a guard would skip.

Activation
----------
``GLM53_INDEXER_WORKSPACE=stock``      (default, and the value of an UNSET var)
    ``get_max_prefill_buffer_size`` returns ``max_model_len * 40``, byte-for-byte
    the stock expression. The image may be built with this patch applied and
    still serve exactly stock.
``GLM53_INDEXER_WORKSPACE=rightsize``  opt in.
Any other value raises at boot rather than picking a serving mode from a typo.
The match is literal, so ``""``, ``" rightsize "`` and ``RIGHTSIZE`` all raise:
identical semantics to ``start.sh``'s ``_glm53_validate_enum``, which defaults
only on an unset var and then compares the value as-is.

Conventions follow ``overlay/patch_kpool_tail_slotmap.py``: pinned ANCHOR, MARK
sentinel, ``verified_state``, ``prepare``, idempotent, atomic replace, pyc
clear, drift => nonzero exit. All three sites are preflighted before any is
written.

Usage::

    python3 patch_indexer_workspace.py              # apply
    python3 patch_indexer_workspace.py --preflight  # validate anchors only
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_INDEXER_BACKEND_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py",
    )
)

ENV_NAME = "GLM53_INDEXER_WORKSPACE"
STOCK_MULTIPLIER = 40
BYTES_PER_ENTRY = 132  # 128 fp8 value bytes + 4 scale bytes (FP8 indexer cache)


# ---------------------------------------------------------------------------
# The injected sizing helpers, as source.
#
# This string is BOTH what gets written into indexer.py AND what the host tests
# exercise: it is exec'd below to produce module-level callables. There is one
# implementation of the formula, so a test can never pass against a replica that
# has drifted from the shipped code.
# ---------------------------------------------------------------------------
HELPERS_SRC = '''

# [glm53-indexer-workspace] Opt-in right-sizing of the prefill gather
# workspace. See overlay/patch_indexer_workspace.py and
# docs/DESIGN-indexer-workspace.md in the recipe. Default is stock.
_GLM53_WORKSPACE_ENV = "GLM53_INDEXER_WORKSPACE"


def _glm53_workspace_mode() -> str:
    """Exactly "stock" or "rightsize"; an UNSET var means "stock".

    Same contract as the launcher: start.sh runs
    ``_glm53_validate_enum GLM53_INDEXER_WORKSPACE "${GLM53_INDEXER_WORKSPACE-stock}"
    stock rightsize``, which substitutes the default only when the var is
    unset and then compares the value literally. So no normalisation here
    either -- no strip, no lower, and "" is a value, not an absence.
    "RIGHTSIZE", " rightsize " and "" are operator errors on both sides of the
    container boundary, and must not silently select a serving mode.
    """
    raw = os.environ.get(_GLM53_WORKSPACE_ENV)
    if raw is None:
        return "stock"
    if raw not in ("stock", "rightsize"):
        raise ValueError(
            f"{_GLM53_WORKSPACE_ENV} must be exactly one of: stock rightsize "
            f"(got: {raw!r})"
        )
    return raw


def _glm53_stock_workspace_entries(max_model_len: int) -> int:
    return max_model_len * 40


def _glm53_indexer_compress_ratio(vllm_config) -> int:
    """Indexer pool width from config only. 1 == no compression == no-op."""
    hf = getattr(vllm_config.model_config, "hf_text_config", None)
    raw = getattr(hf, "index_kpool", None) if hf is not None else None
    try:
        ratio = int(raw or 1)
    except (TypeError, ValueError):
        return 1
    return ratio if ratio > 1 else 1


def _glm53_rightsized_workspace_entries(vllm_config) -> int:
    """Legal maximum compressed N a single prefill step can require.

    Never larger than the stock value: this is a narrowing only.
    """
    max_model_len = int(vllm_config.model_config.max_model_len)
    stock = _glm53_stock_workspace_entries(max_model_len)
    ratio = _glm53_indexer_compress_ratio(vllm_config)
    if ratio <= 1:
        return stock
    sched = vllm_config.scheduler_config
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    num_spec = int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0)
    # Every prefill row costs at least one query token, so a step can carry at
    # most this many prefill requests.
    max_prefill_reqs = max(
        1, min(int(sched.max_num_seqs), int(sched.max_num_batched_tokens))
    )
    # seq_lens_cpu_upper_bound may run ahead of max_model_len by the draft
    # length on async-spec extend rows; cdiv because the consumer floors.
    span = max_model_len + max(0, num_spec)
    per_req = -(-span // ratio)
    return min(per_req * max_prefill_reqs, stock)
'''

_HELPER_NS: dict = {"os": os}
exec(compile(HELPERS_SRC, "<glm53-indexer-workspace helpers>", "exec"), _HELPER_NS)

workspace_mode = _HELPER_NS["_glm53_workspace_mode"]
stock_workspace_entries = _HELPER_NS["_glm53_stock_workspace_entries"]
indexer_compress_ratio = _HELPER_NS["_glm53_indexer_compress_ratio"]
rightsized_workspace_entries = _HELPER_NS["_glm53_rightsized_workspace_entries"]


def split_prefill_chunks(
    compressed_seq_lens: list[int],
    query_lens: list[int],
    workspace_size: int,
    max_logits_bytes: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Host replica of ``split_indexer_prefill_chunks`` (indexer.py).

    Used only by the tests, to prove that a right-sized workspace produces the
    same chunk list as the stock 40x one for every batch the scheduler can
    legally form. Slices are returned as ``(start, stop)`` pairs so the result
    compares by value.
    """
    chunks: list[tuple[tuple[int, int], tuple[int, int]]] = []
    n = len(compressed_seq_lens)
    max_logits_elems = max_logits_bytes // 4
    end = 0
    while end < n:
        start, chunk_m, chunk_n = end, 0, 0
        while end < n:
            q, s = query_lens[end], compressed_seq_lens[end]
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break
        if end == start:
            chunk_m, chunk_n = query_lens[end], compressed_seq_lens[end]
            end += 1
        req_slice = (start, end)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else max(1, chunk_m)
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, (q_off, q_off + sub_m)))
    return chunks


# ---------------------------------------------------------------------------
# Site 1 -- module-scope ``import os``
# ---------------------------------------------------------------------------
MARK_IMPORT = "import os  # [glm53-indexer-workspace] workspace mode env\n"

ANCHOR_IMPORT = """# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
"""

PATCHED_IMPORT = """# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os  # [glm53-indexer-workspace] workspace mode env
from dataclasses import dataclass
"""


# ---------------------------------------------------------------------------
# Site 2 -- the sizing function itself
# ---------------------------------------------------------------------------
MARK_SIZE = (
    "    # [glm53-indexer-workspace] The workspace is sized in TOKENS "
    "but indexed in\n"
)

ANCHOR_SIZE = """def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40
"""

PATCHED_SIZE = (
    HELPERS_SRC.lstrip("\n")
    + """

def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    # [glm53-indexer-workspace] The workspace is sized in TOKENS but indexed in
    # POOLS: split_indexer_prefill_chunks above is fed seq_lens // compress_ratio
    # by the metadata builder, and models/deepseek_v4/attention.py already
    # divides this return value by compress_ratio at its call site --
    # models/glm5next/nvidia/attention.py does not. At max_model_len=1e6 the
    # stock value is 40,000,000 entries; a 2026-09-01 VLLM_DEBUG_WORKSPACE boot
    # measured the resulting lock at 5036.40 MB, inside the memory profile and
    # therefore straight out of the KV pool.
    #
    # "rightsize" returns the LEGAL MAXIMUM a single step can need, so the
    # workspace constraint in the splitter binds exactly where it already did
    # and the chunk list is unchanged. It never exceeds the stock value.
    # Default is stock: an unset env var is byte-for-byte the expression below.
    _glm53_stock = _glm53_stock_workspace_entries(max_model_len)
    if _glm53_workspace_mode() != "rightsize":
        return _glm53_stock
    _glm53_entries = _glm53_rightsized_workspace_entries(vllm_config)
    if _glm53_entries >= _glm53_stock:
        # The legal maximum is already above stock: stock is under-sized here
        # to begin with. Return it unchanged rather than growing the profile.
        logger.warning(
            "[glm53-indexer-workspace] legal max %d entries >= stock %d; "
            "keeping stock sizing",
            _glm53_entries,
            _glm53_stock,
        )
        return _glm53_stock
    logger.info(
        "[glm53-indexer-workspace] rightsize: compress_ratio=%d "
        "max_num_seqs=%d max_num_batched_tokens=%d -> %d entries "
        "(stock %d, ~%.1f MiB reclaimed at 132 B/entry)",
        _glm53_indexer_compress_ratio(vllm_config),
        vllm_config.scheduler_config.max_num_seqs,
        vllm_config.scheduler_config.max_num_batched_tokens,
        _glm53_entries,
        _glm53_stock,
        (_glm53_stock - _glm53_entries) * 132 / (1024 * 1024),
    )
    return _glm53_entries
"""
)


# ---------------------------------------------------------------------------
# Site 3 -- fail-closed cross-check at the authoritative runtime source
# ---------------------------------------------------------------------------
MARK_GUARD = "        # [glm53-indexer-workspace] The sizing above ran against\n"

ANCHOR_GUARD = """        # KV compression. Default to 1 for no compression.
        self.compress_ratio = 1
        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio
        if self.dcp_world_size > 1 and self.compress_ratio > 1:
            raise NotImplementedError(
                "DCP is not supported with sparse indexer KV compression "
                f"(compress_ratio={self.compress_ratio})."
            )
"""

PATCHED_GUARD = """        # KV compression. Default to 1 for no compression.
        self.compress_ratio = 1
        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            self.compress_ratio = self.kv_cache_spec.compress_ratio
        if self.dcp_world_size > 1 and self.compress_ratio > 1:
            raise NotImplementedError(
                "DCP is not supported with sparse indexer KV compression "
                f"(compress_ratio={self.compress_ratio})."
            )
        # [glm53-indexer-workspace] The sizing above ran against
        # hf_text_config.index_kpool, because get_max_prefill_buffer_size only
        # receives vllm_config. THIS is the authoritative ratio: the runtime
        # divides seq_lens by self.compress_ratio before handing them to the
        # splitter. They are the same number by construction
        # (models/glm5next/nvidia/attention.py replaces the indexer spec with
        # compress_ratio=index_kpool), but construction is not a check. Fail at
        # init, on every TP rank, rather than at the first oversized step.
        #
        # The comparison is UNCONDITIONAL in rightsize mode, in both
        # directions. Guarding it on self.compress_ratio > 1 would skip the one
        # case that actually under-sizes: index_kpool=4 with
        # kv_cache_spec.compress_ratio=1 sizes the workspace for ~4M compressed
        # entries while the runtime hands the splitter up to ~16M token
        # entries. The mirror case (index_kpool absent/1 with compress_ratio>1)
        # sizes at stock and so cannot under-run, but it still means the two
        # config sources disagree about the model being served, which is not a
        # state to serve rightsize from.
        if _glm53_workspace_mode() == "rightsize":
            _glm53_cfg_ratio = _glm53_indexer_compress_ratio(self.vllm_config)
            if _glm53_cfg_ratio != self.compress_ratio:
                raise ValueError(
                    "[glm53-indexer-workspace] compress-ratio disagreement: "
                    f"hf_text_config.index_kpool={_glm53_cfg_ratio} but "
                    f"kv_cache_spec.compress_ratio={self.compress_ratio}. "
                    "Refusing to serve on a workspace sized from the wrong "
                    f"ratio; unset {_GLM53_WORKSPACE_ENV} to use stock sizing."
                )
            _glm53_need = _glm53_rightsized_workspace_entries(self.vllm_config)
            _glm53_stock = _glm53_stock_workspace_entries(
                self.vllm_config.model_config.max_model_len
            )
            if self.max_prefill_buffer_size < min(_glm53_need, _glm53_stock):
                raise ValueError(
                    "[glm53-indexer-workspace] workspace "
                    f"{self.max_prefill_buffer_size} entries is below the legal "
                    f"per-step maximum {_glm53_need} (compress_ratio="
                    f"{self.compress_ratio}, max_num_seqs="
                    f"{self.vllm_config.scheduler_config.max_num_seqs})."
                )
            logger.info(
                "[glm53-indexer-workspace] builder: compress_ratio=%d "
                "workspace=%d entries (legal max %d, stock %d)",
                self.compress_ratio,
                self.max_prefill_buffer_size,
                _glm53_need,
                _glm53_stock,
            )
"""


SITES = (
    ("os import", MARK_IMPORT, ANCHOR_IMPORT, PATCHED_IMPORT),
    ("workspace sizing", MARK_SIZE, ANCHOR_SIZE, PATCHED_SIZE),
    ("builder cross-check", MARK_GUARD, ANCHOR_GUARD, PATCHED_GUARD),
)


def verified_state(text: str) -> bool:
    """Exact post-state check.

    Two of the three replacements are insertions that legitimately keep their
    anchor, so ``anchor == 0`` would reject them and ignoring the anchor would
    stop catching a half-applied site. Expect exactly as many surviving anchors
    as the replacement itself contains.
    """
    return all(
        text.count(mark) == 1
        and text.count(patched) == 1
        and text.count(anchor) == patched.count(anchor)
        for _name, mark, anchor, patched in SITES
    )


def prepare(source: str) -> tuple[str, str]:
    """Idempotent, fail-closed. Returns ``(text, action)``."""
    marks = sum(source.count(mark) for _n, mark, _a, _p in SITES)
    if marks:
        if marks != len(SITES) or not verified_state(source):
            raise ValueError(
                "partial/inconsistent indexer workspace patch "
                f"(marks={marks}, expected {len(SITES)}) -- refusing to touch "
                "a half-patched file"
            )
        return source, "already present"

    out = source
    for name, _mark, anchor, patched in SITES:
        n = out.count(anchor)
        if n != 1:
            raise ValueError(
                f"pinned indexer anchor '{name}' drifted (found {n}, expected 1)"
            )
        out = out.replace(anchor, patched, 1)

    if not verified_state(out):
        raise ValueError("indexer workspace post-patch verification failed")
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-indexer-workspace.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(f"{target.stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]

    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"indexer workspace preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")

    if preflight_only:
        print(f"{TARGET.name}: indexer workspace preflight OK ({action})")
        return 0

    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    # Report the value as set, not as normalised: this script runs at image
    # build time where the knob is usually unset, and printing "stock" for an
    # explicitly empty value would hide the error the runtime will raise.
    mode = os.environ.get(ENV_NAME, "stock (unset)")
    print(f"{TARGET.name}: indexer workspace {action} ({ENV_NAME}={mode!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
