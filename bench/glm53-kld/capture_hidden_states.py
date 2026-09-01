#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture GLM-5.3-Flash hidden states for the sealed fidelity windows.

Teacher-forced, one 2048-token context per generate call, ``max_tokens=1``, no
sampling and no speculative decoding: the captured tensor is the pre-sampling
state, so MTP acceptance jitter and draft-tower quantization stay out of the
number. Hidden states are grabbed inside the engine by ``logit_dump_hook`` at the
suite's own semantic point (``after_final_rmsnorm_before_lm_head``) and written
out as ``hidden_<index>.safetensors`` with the same [2047, 4096] BF16 shape the
published lanes use, alongside a ``capture-manifest-shard.json`` in the suite's
shard format, so ``score_hidden_kld.py`` reads our lane exactly like theirs.

Every context must produce exactly one dump. Zero dumps is a hard failure, not a
warning - the plausible cause is a CUDA-graph replay that never re-entered Python,
which would otherwise show up as a quietly short lane. ``--enforce-eager`` is the
lever if that fires.

Before touching the GPUs it checks the thing that makes the whole comparison
meaningful: our checkpoint's ``lm_head`` must be bit-identical to the suite's
shared head. If the repack requantized the head, every logit row shifts and KLD
measures the head, not the experts.

Run it through ``capture-glm-awq.sh`` (which owns the container flags); the module
docstring there covers scheduling and card hand-off.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from kld_core import (
    CAPTURE_MANIFEST_SCHEMA,
    SEMANTIC_POINT,
    SUITE_TENSOR_KEY,
    load_distribution_suite,
    load_replay_head,
    sha256_file,
    seal,
    write_json,
)

STAGING_PREFIX = "hidden"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="/srv/models/fidelity-suites/GLM-5.3-Flash-fidelity-suite-v1")
    parser.add_argument("--model", required=True, help="model path as seen inside the container")
    parser.add_argument("--out", required=True, help="lane directory to write")
    parser.add_argument("--dump-dir", default=os.environ.get("GLM_KLD_DUMP_DIR", "/work/dumps"))
    parser.add_argument("--indices", default=None, help="comma list; default = whole lane")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=4)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    # Eager by default: a CUDA-graph replay never re-enters Python, so a module
    # forward hook can silently stop firing. The per-context dump check would then
    # abort the run, but eager removes the whole class of failure for a ~2x cost we
    # can afford on 512 prefill steps.
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-engine-json", default=None,
                        help="JSON object of extra LLM kwargs (production parity flags)")
    parser.add_argument("--skip-head-check", action="store_true")
    return parser.parse_args()


def staged_dumps(staging: Path) -> list[Path]:
    if not staging.is_dir():
        return []
    return sorted(staging.glob(f"{STAGING_PREFIX}-*.safetensors"))


def verify_head_matches_checkpoint(model_dir: Path, head_path: Path) -> dict:
    """Bit-compare our checkpoint's lm_head with the suite's shared head."""
    import torch
    from safetensors import safe_open

    from kld_core import load_head

    ours = load_head(head_path)
    index = model_dir / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"{index}: cannot locate the weight index to find lm_head")
    weight_map = json.loads(index.read_text())["weight_map"]
    candidates = [name for name in weight_map if name.endswith("lm_head.weight")]
    if not candidates:
        raise KeyError("no *lm_head.weight in the checkpoint index - is the head tied?")
    report = {}
    for name in candidates:
        with safe_open(str(model_dir / weight_map[name]), framework="pt") as handle:
            if name not in handle.keys():
                continue
            tensor = handle.get_tensor(name)
        same_shape = tuple(tensor.shape) == tuple(ours.shape)
        report[name] = {
            "path": str(model_dir / weight_map[name]),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bitwise_equal_to_suite_head": bool(same_shape and torch.equal(tensor, ours)),
        }
    if not any(row["bitwise_equal_to_suite_head"] for row in report.values()):
        raise ValueError(
            "checkpoint lm_head is not bit-identical to the suite's shared head; KLD would measure "
            f"the head, not the quantized experts: {json.dumps(report)}"
        )
    return report


def main() -> int:
    args = parse_args()
    # apply_model ships a Python callable to every worker. vLLM's default msgpack
    # serializer refuses callables, so this run opts into pickle serialization -
    # acceptable because the only callable is ours, and it is the reason the tap is
    # attached to the module rather than monkey-patched around a class name.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    staging = Path(args.dump_dir, "hidden")
    staging.mkdir(parents=True, exist_ok=True)

    # Windows come from the sealed suite manifest; the teacher lane is not read
    # here (our lane is created by this run, so it cannot be the loader's input).
    indices = None if not args.indices else [int(x) for x in args.indices.split(",") if x]
    meta, windows = load_distribution_suite(args.suite, indices=indices, limit=args.limit,
                                            require_lane_files=False)
    rows = meta["scored_positions_per_context"]
    print(f"suite {meta['schema']} contexts={len(windows)} rows/context={rows} "
          f"hidden_size={meta['hidden_size']}", flush=True)

    model_dir = Path(args.model)
    head_report = None
    if not args.skip_head_check:
        head_path, head_meta = load_replay_head(args.suite)
        head_report = verify_head_matches_checkpoint(model_dir, head_path)
        print(f"lm_head check: bitwise equal ({list(head_report)[0]})", flush=True)

    engine_kwargs = {
        "model": args.model,
        "dtype": args.dtype,
        "tokenizer": args.model,
        "trust_remote_code": False,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "kv_cache_dtype": args.kv_cache_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": max(args.max_num_batched_tokens, meta["context_length"] + 8),
        "max_num_seqs": 1,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "disable_log_stats": True,
        "skip_mm_profiling": True,
        "enable_prefix_caching": False,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    if args.extra_engine_json:
        engine_kwargs.update(json.loads(args.extra_engine_json))
    if args.served_model_name:
        engine_kwargs["served_model_name"] = args.served_model_name
    if meta["context_length"] + 8 > engine_kwargs["max_model_len"]:
        raise ValueError("max_model_len must exceed the suite context length")

    # Fail here, not inside vLLM after a 178 GB weight path has been opened:
    # LLM(**kwargs) only accepts EngineArgs fields, and the CLI accepts more than
    # the dataclass does (num_scheduler_steps, tool_call_parser, ...). Anything we
    # pass that this build does not understand is named, not silently ignored.
    import dataclasses

    from vllm.engine.arg_utils import EngineArgs

    supported = {field.name for field in dataclasses.fields(EngineArgs)}
    unknown = sorted(set(engine_kwargs) - supported)
    if unknown:
        raise ValueError(
            f"engine kwargs this build does not accept: {unknown}; "
            "drop them or pass the equivalent CLI flag")
    # Dry-run the whole config build (reads config.json; no weights, no GPU).
    EngineArgs(**engine_kwargs).create_engine_config()
    print("engine config built ok", flush=True)

    llm = LLM(**engine_kwargs)
    from logit_dump_hook import install_capture

    taps = llm.apply_model(install_capture)
    print("worker taps:", taps, flush=True)
    if not any(str(row).startswith("hooked:") for row in taps):
        raise RuntimeError(f"no worker installed the hidden tap: {taps}")

    sampler = SamplingParams(max_tokens=1, temperature=0.0)
    done = {int(path.stem.split("_")[1]) for path in out.glob("hidden_*.safetensors")}
    started = time.time()
    captured = 0
    for position, window in enumerate(windows, start=1):
        if window.index in done:
            continue
        before = {path.name for path in staged_dumps(staging)}
        llm.generate(
            [TokensPrompt(prompt_token_ids=[int(token) for token in window.tokens])],
            sampler,
        )
        fresh = [path for path in staged_dumps(staging) if path.name not in before]
        if len(fresh) != 1:
            raise RuntimeError(
                f"context {window.index}: expected exactly one hidden dump, got {len(fresh)} "
                f"({[p.name for p in fresh]}). Zero dumps usually means the forward pass ran inside "
                "a CUDA graph and never re-entered the Python hook - rerun with --enforce-eager."
            )
        target = out / f"hidden_{window.index:04d}.safetensors"
        shutil.move(str(fresh[0]), target)
        captured += 1
        if captured % 16 == 0 or position == len(windows):
            rate = captured / max(time.time() - started, 1e-9)
            print(f"  {position}/{len(windows)} contexts, {rate:.2f} ctx/s, "
                  f"eta {(len(windows) - position) / max(rate, 1e-9):.0f}s", flush=True)

    manifest_captures = []
    for path in sorted(out.glob("hidden_*.safetensors")):
        manifest_captures.append({
            "index": int(path.stem.split("_")[1]),
            "sha256": sha256_file(path),
            "shape": [rows, meta["hidden_size"]],
            "file": path.name,
        })
    shard_manifest = {
        "schema": CAPTURE_MANIFEST_SCHEMA,
        "semantic_point": SEMANTIC_POINT,
        "tensor_key": SUITE_TENSOR_KEY,
        "lane": f"captured on cmp170hx from {args.model}",
        "filter": "all",
        "contexts": len(manifest_captures),
        "complete": True,
        "index_range": [min(row["index"] for row in manifest_captures),
                        max(row["index"] for row in manifest_captures)] if manifest_captures else None,
        "captures": manifest_captures,
    }
    write_json(out / "capture-manifest-shard.json", shard_manifest)

    receipt = seal({
        "schema": "cmp170hx.glm53-hidden-capture.v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "semantic_point": SEMANTIC_POINT,
        "model": args.model,
        "engine_kwargs": {key: value for key, value in engine_kwargs.items()},
        "vllm_version": _vllm_version(),
        "suite_token_sha256": meta["suite_token_sha256"],
        "context_length": meta["context_length"],
        "rows_stored": rows,
        "contexts": len(manifest_captures),
        "newly_captured": captured,
        "lm_head_check": head_report,
        "lane": str(out),
    }, "receipt_sha256")
    digest = write_json(out / "capture-receipt.json", receipt)
    print(f"captured {captured} contexts into {out} (receipt_sha256={digest})")
    return 0


def _vllm_version() -> str:
    try:
        import vllm

        return getattr(vllm, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
