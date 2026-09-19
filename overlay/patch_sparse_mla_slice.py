#!/usr/bin/env python3
"""Opt-in bounded final sparse-MLA attention call for TP=4 (SM120/SM121).

``VLLM_SM120_SPARSE_MLA_SLICE_TOKENS`` unset, empty or ``0`` leaves the shipped
``flashinfer_mla_sparse_sm120.py`` byte-identical. ``64`` rewrites the final
``flashinfer_trtllm_batch_decode_with_kv_cache_mla`` call so it runs in slices
of at most 64 query rows (the existing H16 decode/merge kernels); the outer
2048-token batch, EXL3 kernels, DFlash and NCCL plumbing are untouched. Any
other value is refused. Inside the patched backend the slice is still gated by
the same variable at call time and fails closed on unqualified geometry: BF16
``[T,16,576]`` query, ``[64,656]`` packed KV pages, physical top-k 2048 and at
least 33,685,504 bytes of decode workspace.

The preimage and the result are both pinned by SHA-256, so the rewrite cannot
silently apply to another vLLM revision. The pinned preimage is the file shipped
in ``ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3`` (vLLM
``0.1.dev20051+g487ecf187``); ``tests/fixtures/flashinfer_mla_sparse_sm120-487ecf187.py.txt``
is that file.

Mitigation for the four-rank stall in #128 / #159; see #223. The transformation
(``OLD_CALL`` / ``NEW_CALL`` / ``CONSTANTS`` and both hash pins) is carried
verbatim from ``recipe/patch_sparse_mla_slice.py`` in
https://github.com/punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark (MIT License,
Copyright (c) 2026 punkjazz-labs), where it was qualified on a 4x GB10 TP=4
kit (153-minute mixed soak, FP32 reference comparison at 1/64/65/129/193/2048
tokens, max abs diff 0.00185609). It does not identify or fix the underlying
race; it bounds the call in which progress was observed to stop.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

ENV_NAME = "VLLM_SM120_SPARSE_MLA_SLICE_TOKENS"
SLICE_TOKENS = 64
TARGET = Path(
    os.environ.get(
        "GLM53_SPARSE_SLICE_TARGET",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
        "flashinfer_mla_sparse_sm120.py",
    )
)

ORIGINAL_SHA256 = "d665ef2109b0183d48e3541ecd24e9fa8e1dc3e410983bc29b8d997af9d7cd01"
PATCHED_SHA256 = "f1854c0cce9d749d5ce67dd5132f6541c9d01dec4c2bb414267815ed6fd620c6"

OLD_CALL = """        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.kernel_qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=topk_lengths,
            max_seq_len=sparse_topk_capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=sparse_topk_capacity,
            kv_scale_format=self.kv_scale_format,
        )
        out = out.squeeze(1)
"""
NEW_CALL = """        slice_tokens = int(os.getenv(_SLICE_ENV, "0"))
        if slice_tokens not in (0, _SLICE_TOKENS):
            raise ValueError(
                f"{_SLICE_ENV} must be 0 or {_SLICE_TOKENS}; got {slice_tokens}"
            )
        call_kwargs = dict(
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.kernel_qk_rope_head_dim,
            max_seq_len=sparse_topk_capacity,
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=sparse_topk_capacity,
            kv_scale_format=self.kv_scale_format,
        )
        if slice_tokens == 0:
            out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
                query=q.unsqueeze(1),
                block_tables=topk_indices_physical.unsqueeze(1),
                seq_lens=topk_lengths,
                out=output.unsqueeze(1),
                **call_kwargs,
            ).squeeze(1)
        else:
            if q.dtype != torch.bfloat16 or tuple(q.shape[1:]) != (16, 576) or self.num_heads != 16:
                raise RuntimeError(f"SM120 attention slicing requires BF16 query [T,16,576]; got dtype={q.dtype}, shape={tuple(q.shape)}, heads={self.num_heads}")
            if kv_c_and_k_pe_cache.shape[-2:] != (64, 656):
                raise RuntimeError("SM120 attention slicing requires packed KV pages [64,656]")
            if sparse_topk_capacity != _SLICE_REQUIRED_TOPK:
                raise RuntimeError(
                    "SM120 attention slicing requires physical sparse-top-k "
                    f"{_SLICE_REQUIRED_TOPK}; got {sparse_topk_capacity}"
                )
            workspace_bytes = (
                self._workspace_buffer.numel() * self._workspace_buffer.element_size()
            )
            if workspace_bytes < _SLICE_REQUIRED_WORKSPACE_BYTES:
                raise RuntimeError(
                    "SM120 attention slicing requires at least "
                    f"{_SLICE_REQUIRED_WORKSPACE_BYTES} workspace bytes; got "
                    f"{workspace_bytes}"
                )
            for begin in range(0, num_actual_toks, _SLICE_TOKENS):
                end = min(begin + _SLICE_TOKENS, num_actual_toks)
                flashinfer_trtllm_batch_decode_with_kv_cache_mla(
                    query=q[begin:end].unsqueeze(1),
                    block_tables=topk_indices_physical[begin:end].unsqueeze(1),
                    seq_lens=topk_lengths[begin:end],
                    out=output[begin:end].unsqueeze(1),
                    **call_kwargs,
                )
            out = output
"""
CONSTANTS = """_SLICE_ENV = "VLLM_SM120_SPARSE_MLA_SLICE_TOKENS"
_SLICE_TOKENS = 64
_SLICE_REQUIRED_TOPK = 2048
_SLICE_REQUIRED_WORKSPACE_BYTES = 33_685_504


"""

IMPORT_ANCHOR = "from typing import TYPE_CHECKING, cast"
CONSTANTS_ANCHOR = "def _kv_scale_format_for_model"


def parse_slice_tokens(raw: str | None) -> int | None:
    """``None``/``""``/``"0"`` -> ``None`` (stock). ``"64"`` -> 64. Else ValueError."""
    if raw is None or raw == "" or raw == "0":
        return None
    if raw == str(SLICE_TOKENS):
        return SLICE_TOKENS
    raise ValueError(f"{ENV_NAME} must be empty, 0 or {SLICE_TOKENS} (got: {raw!r})")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(source: str, slice_tokens: int | None) -> tuple[str, str]:
    """Return ``(new_source, action)``; never touches disk.

    action is ``"stock"`` (nothing requested), ``"already present"`` (source is
    the pinned patched text) or ``"patched"``. Drift fails closed: a source that
    is neither the pinned preimage nor the pinned result raises RuntimeError.
    """
    digest = sha256(source)
    if digest == PATCHED_SHA256:
        return source, "already present"
    if digest != ORIGINAL_SHA256:
        raise RuntimeError(
            f"unrecognized sparse MLA backend preimage {digest[:16]}; "
            f"expected {ORIGINAL_SHA256[:16]} (stock) or {PATCHED_SHA256[:16]} (patched)"
        )
    if slice_tokens is None:
        return source, "stock"
    for anchor in (IMPORT_ANCHOR, CONSTANTS_ANCHOR, OLD_CALL):
        if source.count(anchor) != 1:
            raise RuntimeError(f"anchor not found exactly once: {anchor.strip().splitlines()[0]!r}")
    result = source.replace(IMPORT_ANCHOR, "import os\n" + IMPORT_ANCHOR, 1)
    result = result.replace(CONSTANTS_ANCHOR, CONSTANTS + CONSTANTS_ANCHOR, 1)
    result = result.replace(OLD_CALL, NEW_CALL, 1)
    ast.parse(result)
    if sha256(result) != PATCHED_SHA256:
        raise RuntimeError("sparse MLA slice transformation hash mismatch")
    return result, "patched"


def replace_file(target: Path, source: str) -> None:
    mode = stat.S_IMODE(target.stat().st_mode)
    temporary = target.with_name(target.name + ".glm53-slice-tmp")
    temporary.write_text(source)
    temporary.chmod(mode)
    os.replace(temporary, target)


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if cache.is_dir():
        for stale in cache.glob(target.stem + ".*.pyc"):
            stale.unlink()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    target = Path(argv[1]) if len(argv) > 1 else TARGET
    try:
        slice_tokens = parse_slice_tokens(os.environ.get(ENV_NAME))
    except ValueError as exc:
        print(f"[glm53-sparse-mla-slice] {exc}", file=sys.stderr)
        return 2
    if not target.is_file():
        print(f"[glm53-sparse-mla-slice] target missing: {target}", file=sys.stderr)
        return 2
    source = target.read_text()
    try:
        result, action = prepare(source, slice_tokens)
    except RuntimeError as exc:
        print(f"[glm53-sparse-mla-slice] {exc}", file=sys.stderr)
        return 2
    if result != source:
        replace_file(target, result)
        clear_pyc(target)
    print(
        json.dumps(
            {
                "patch": "glm53-sparse-mla-slice",
                "target": str(target),
                "slice_tokens": slice_tokens or 0,
                "action": action,
                "sha256": sha256(result),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
