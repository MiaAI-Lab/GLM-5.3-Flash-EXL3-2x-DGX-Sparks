#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture full-vocabulary logits of our GLM-5.3-Flash serving stack, per window.

Runs *inside* ``vllm/vllm-backport:cmp170hx`` (see capture-glm-awq.sh). The
engine computes prompt-position logits because every request asks for
``prompt_logprobs``; ``logit_dump_hook`` (mounted via ``sitecustomize``) saves
the whole matrix before top-k truncation. This driver only drives requests and
assembles the per-window ``[positions, vocab]`` float32 tensors plus a sealed
capture receipt whose field names match the published capture schema, so a
teacher capture from any other stack can be diffed against it.

Deliberate protocol choices:
  * ``enforce_eager`` + ``max_num_seqs=1`` + no speculative decoding: the KLD
    protocol requires a deterministic pipeline (batch-composed reductions and
    CUDA-graph replay both perturb the tail of the KLD distribution).
  * ``logprobs_mode=raw_logits``: store raw logits, recompute log-softmax in
    float64 at analysis time.
  * one window per forward: chunked prefill would split a window across dumps;
    ``max_num_batched_tokens`` is sized to hold a whole window.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from kld_core import (
    CAPTURE_SCHEMA,
    LOGITS_FIELD,
    load_token_panel,
    prepare_empty_destination,
    seal,
    sha256_file,
    write_json,
)

ENV_DUMP_DIR = "GLM_KLD_DUMP_DIR"


class Window:
    def __init__(self, window_id: str, document_id: str, domain: str, role: str,
                 token_ids: list[int], token_sha: str, mask_sha: str, positions: int):
        self.window_id = window_id
        self.document_id = document_id
        self.domain = domain
        self.role = role
        self.token_ids = token_ids
        self.token_sha = token_sha
        self.mask_sha = mask_sha
        self.positions = positions


def load_panel_jsonl(path: Path) -> list[Window]:
    windows: list[Window] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        token_ids = [int(token) for token in row["token_ids"]]
        if len(token_ids) < 2:
            raise ValueError(f"window {row.get('window_id')} is too short")
        windows.append(
            Window(
                window_id=str(row["window_id"]),
                document_id=str(row.get("document_id", row["window_id"])),
                domain=str(row.get("domain", "unknown")),
                role=str(row.get("role", "final")),
                token_ids=token_ids,
                token_sha=_int_sha(token_ids),
                mask_sha=_int_sha([1] * len(token_ids)),
                positions=len(token_ids) - 1,
            )
        )
    if not windows:
        raise ValueError(f"no windows in {path}")
    return windows


def _int_sha(values: list[int]) -> str:
    from kld_core import sha256_bytes

    return sha256_bytes(np.asarray(values, dtype=np.int64).tobytes())


def read_panel(args: argparse.Namespace, vocab_size: int) -> list[Window]:
    if args.panel_receipt:
        _, windows = load_token_panel(args.panel_receipt, roles=("final",), vocab_size=vocab_size)
        return [
            Window(w.window_id, w.document_id, w.domain, w.role,
                   [int(t) for t in w.token_ids], w.token_ids_sha256,
                   w.attention_mask_sha256, w.prediction_positions)
            for w in windows
        ]
    if not args.panel_jsonl:
        raise SystemExit("either --panel-receipt or --panel_jsonl is required")
    return load_panel_jsonl(Path(args.panel_jsonl))


def newest_dump(dump_dir: Path, after: int) -> tuple[int, list[Path]]:
    """Return (highest index, new dump files in order) beyond ``after``."""
    found: list[int] = []
    for path in dump_dir.glob("dump-*.pt"):
        try:
            index = int(path.stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        if index > after:
            found.append(index)
    return (max(found) if found else after), [dump_dir / f"dump-{index:06d}.pt" for index in sorted(found)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="checkpoint path inside the container")
    parser.add_argument("--panel-receipt", default=None, help="published token-panel receipt (.npy artifacts)")
    parser.add_argument("--panel-jsonl", default=None, help="local pilot panel: one JSON window per line")
    parser.add_argument("--output", required=True)
    parser.add_argument("--served-label", default="awq-w4a16")
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--pp-partition", default="14,12,12,7", help="AWQ needs 14,12,12,7 (rank3 takes 7 layers)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-dumps", action="store_true")
    args = parser.parse_args()

    dump_dir = Path(os.environ.get(ENV_DUMP_DIR, "/work/dumps"))
    if not dump_dir.is_dir():
        raise SystemExit(f"{ENV_DUMP_DIR}={dump_dir} must exist and be writable by the engine worker")
    output = prepare_empty_destination(Path(args.output))
    logits_dir = output / "logits"
    logits_dir.mkdir()

    from safetensors.torch import save_file
    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.inputs import TokensPrompt

    config = json.loads((Path(args.model) / "config.json").read_text())
    text_config = config.get("text_config", config)
    vocab_size = int(text_config["vocab_size"])
    windows = read_panel(args, vocab_size)
    longest = max(len(window.token_ids) for window in windows)
    if args.max_num_batched_tokens < longest:
        raise SystemExit("--max-num-batched-tokens must hold a whole window to keep chunking out of the capture")

    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_MLA_SPARSE")
    os.environ.setdefault("VLLM_PP_LAYER_PARTITION", args.pp_partition)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    started = time.monotonic()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        pipeline_parallel_size=args.pp,
        enable_expert_parallel=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        enforce_eager=True,
        logprobs_mode="raw_logits",
        seed=args.seed,
        disable_log_stats=True,
        attention_config={"sparse_mla_force_mqa": True},
        enable_prefix_caching=False,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1,
                            output_kind=RequestOutputKind.FINAL_ONLY)

    high_water = 0
    records: list[dict] = []
    for position, window in enumerate(windows, 1):
        before, _ = newest_dump(dump_dir, high_water)
        llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=list(window.token_ids))],
            sampling_params=params,
            use_telemetry=False,
        )
        high_water, new_files = newest_dump(dump_dir, before)
        if not new_files:
            raise RuntimeError(f"window {window.window_id}: engine produced no full-vocab logit dump")
        import torch as _torch

        chunks = [_torch.load(path, map_location="cpu", weights_only=True) for path in new_files]
        rows = int(sum(chunk.shape[0] for chunk in chunks))
        if rows != len(window.token_ids):
            raise RuntimeError(
                f"window {window.window_id}: dumped {rows} rows for {len(window.token_ids)} tokens "
                "(chunking or batching leaked into the capture)"
            )
        matrix = chunks[0] if len(chunks) == 1 else _torch.cat(chunks, dim=0)
        selected = matrix[: window.positions].to(_torch.float32).contiguous()
        if tuple(selected.shape) != (window.positions, vocab_size):
            raise RuntimeError(f"window {window.window_id}: logit geometry {tuple(selected.shape)}")
        logit_path = (logits_dir / f"window-{position - 1:04d}.safetensors").resolve()
        save_file(
            {LOGITS_FIELD: selected},
            logit_path,
            metadata={
                "capture_role": "engine_student",
                "student_label": args.served_label,
                "window_id": window.window_id,
                "token_ids_sha256": window.token_sha,
                "attention_mask_sha256": window.mask_sha,
                "engine": "vllm/vllm-backport:cmp170hx",
                "logits_mode": "raw_logits",
            },
        )
        records.append(
            {
                "window_id": window.window_id,
                "document_id": window.document_id,
                "domain": window.domain,
                "role": window.role,
                "token_ids_sha256": window.token_sha,
                "attention_mask_sha256": window.mask_sha,
                "prediction_positions": window.positions,
                "path": str(logit_path),
                "bytes": logit_path.stat().st_size,
                "sha256": sha256_file(logit_path),
            }
        )
        del chunks, matrix, selected
        if not args.keep_dumps:
            for path in new_files:
                path.unlink(missing_ok=True)
                path.with_suffix(".json").unlink(missing_ok=True)
        print(f"[glm53-kld] {position}/{len(windows)} {window.window_id} "
              f"{window.positions} positions", flush=True)

    receipt = {
        "schema": CAPTURE_SCHEMA,
        "capture_role": "engine_student",
        "student_label": args.served_label,
        "model_path": args.model,
        "engine": "vllm/vllm-backport:cmp170hx",
        "engine_flags": {
            "pipeline_parallel_size": args.pp,
            "pp_layer_partition": args.pp_partition,
            "enable_expert_parallel": True,
            "enforce_eager": True,
            "max_num_seqs": 1,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "logprobs_mode": "raw_logits",
            "speculative_decoding": "off",
            "prefix_caching": False,
        },
        "weight_dtype": "compressed-tensors AWQ W4A16 (pack-quantized, group 128)",
        "logits_dtype": "float32",
        "kld_direction": "teacher_to_student",
        "prediction_positions": sum(record["prediction_positions"] for record in records),
        "vocab_size": vocab_size,
        "logit_files": records,
        "elapsed_seconds": time.monotonic() - started,
    }
    seal(receipt, "receipt_sha256")
    write_json(output / "capture-receipt.json", receipt)
    print(json.dumps({"ok": True, "receipt_sha256": receipt["receipt_sha256"],
                      "windows": len(records),
                      "prediction_positions": receipt["prediction_positions"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
