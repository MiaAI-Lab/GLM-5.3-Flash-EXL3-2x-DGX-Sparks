"""Correctness and model-agnostic SM121 tactic sweep for direct EXL3 CUDA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .abi import MCG_MARKER_SIGNED_INT32, PackedExl3Metadata, PackedExl3Weight
from .backend import DirectExllamaBackend, KernelTactic, inspect_capability
from .interfaces import KernelKey, Representation
from .receipts import (
    TacticMeasurement,
    build_atlas_receipt,
    build_catalog,
    git_commit,
)
from .reference import (
    matmul_materialized_reference,
    matmul_streaming_reference,
    pack_trellis,
)
from .shapes import (
    DECODE_M_SWEEP,
    GENERIC_WORKLOADS,
    LEGACY_FIXTURES,
    GemmWorkload,
    load_workload_file,
    parse_shape_spec,
)


def synthetic_weight(
    k: int,
    n: int,
    *,
    bits: int = 3,
    seed: int,
    pack_codes: bool,
) -> PackedExl3Weight:
    metadata = PackedExl3Metadata(in_features=k, out_features=n, bits=bits)
    rng = np.random.default_rng(seed)
    if pack_codes:
        encoded = rng.integers(
            0, 1 << metadata.bits, size=metadata.encoded_shape, dtype=np.int16
        )
        trellis = np.ascontiguousarray(pack_trellis(encoded, metadata.bits))
    else:
        # Every bit pattern is a valid packed trellis payload. This avoids a
        # slow CPU pack before a large GPU-only performance sweep.
        trellis = rng.integers(
            -32768, 32767, size=metadata.trellis_shape, dtype=np.int16
        )
    signs_k = rng.choice(np.array([-1.0, 1.0], dtype=np.float16), size=k)
    signs_n = rng.choice(np.array([-1.0, 1.0], dtype=np.float16), size=n)
    suh = np.ascontiguousarray(signs_k * np.float16(0.1))
    svh = np.ascontiguousarray(signs_n * np.float16(0.1))
    mcg = np.array([MCG_MARKER_SIGNED_INT32], dtype=np.int32)
    return PackedExl3Weight(metadata, trellis, suh, svh, mcg)


def run_cpu_oracle_check(*, seed: int = 0, bits: int = 3) -> dict[str, Any]:
    """Small deterministic identity check suitable for non-CUDA hosts."""

    weight = synthetic_weight(128, 128, bits=bits, seed=seed, pack_codes=True)
    x = np.random.default_rng(seed + 1).standard_normal((2, 128), dtype=np.float32)
    started = perf_counter()
    materialized = matmul_materialized_reference(x, weight)
    streaming = matmul_streaming_reference(x, weight)
    elapsed_ms = (perf_counter() - started) * 1000.0
    delta = np.abs(materialized - streaming)
    return {
        "evidence": "measured",
        "representation": {
            "format": "exl3",
            "bits": bits,
            "codebook": "mcg",
        },
        "shape": {"m": 2, "n": 128, "k": 128},
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "elapsed_ms": elapsed_ms,
        "passed": bool(np.allclose(materialized, streaming, rtol=2e-5, atol=2e-5)),
    }


def _time_cuda(fn: Callable[[], Any], *, warmup: int, iterations: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop) / iterations)


def _classify_bottleneck(
    direct_ms: float, dense_ms: float, reconstruct_ms: float
) -> str:
    if direct_ms <= dense_ms * 1.25:
        return "compute_or_launch"
    if reconstruct_ms >= direct_ms * 0.75:
        return "reconstruction"
    return "memory_bandwidth_or_pipeline"


def _baseline_measurements(linear, x, *, warmup: int, iterations: int):
    """Time diagnostic baselines outside every direct-kernel event range."""

    import torch

    reconstructed = linear.get_weight_tensor().half()
    dense_output = torch.matmul(x, reconstructed).float()
    dense_ms = _time_cuda(
        lambda: torch.matmul(x, reconstructed), warmup=warmup, iterations=iterations
    )
    slower_iterations = max(1, iterations // 5)
    reconstruct_ms = _time_cuda(
        linear.get_weight_tensor, warmup=1, iterations=slower_iterations
    )
    naive_ms = _time_cuda(
        lambda: torch.matmul(x, linear.get_weight_tensor().half()),
        warmup=1,
        iterations=slower_iterations,
    )
    return dense_output, dense_ms, reconstruct_ms, naive_ms


def run_sm121_workload(
    *,
    workload: GemmWorkload,
    bits: int,
    m: int,
    seed: int,
    warmup: int,
    iterations: int,
    runtime_commit: str,
    command: list[str],
    tactic_sweep: bool = True,
) -> list[dict[str, Any]]:
    """Measure every compatible direct tactic and return canonical receipts."""

    import torch

    packed = synthetic_weight(
        workload.k, workload.n, bits=bits, seed=seed, pack_codes=False
    )
    backend = DirectExllamaBackend()
    key = KernelKey(
        (12, 1),
        workload.phase,
        Representation.exl3_mcg(bits),
        m,
        workload.n,
        workload.k,
        grouped=workload.grouped_moe,
        model_id=workload.model_id,
        operator=workload.operator,
        tp_world_size=workload.tp_world_size,
    )
    supported, reason = backend.supports(key)
    if not supported:
        raise RuntimeError(reason)
    linear = backend.build(packed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 1)
    x = torch.randn(
        (m, workload.k), generator=generator, dtype=torch.float16, device="cuda"
    )
    dense_output, dense_ms, reconstruct_ms, naive_ms = _baseline_measurements(
        linear, x, warmup=warmup, iterations=iterations
    )
    tactics = backend.tactics(key)
    if not tactic_sweep:
        tactics = (tactics[0],)

    capability = inspect_capability()
    generated_at = datetime.now(timezone.utc)
    bytes_per_call = packed.payload_nbytes + 2 * m * workload.k + 4 * m * workload.n
    flops = 2 * m * workload.n * workload.k
    receipts: list[dict[str, Any]] = []
    for tactic in tactics:
        output = torch.empty(
            (m, workload.n), dtype=torch.float32, device="cuda"
        )
        x_hadamard = torch.empty_like(x)
        direct_fn = lambda: backend.run_tactic(
            x,
            linear,
            tactic,
            output=output,
            x_hadamard=x_hadamard,
        )
        direct_output, selected_shape = direct_fn()
        direct_ms = _time_cuda(direct_fn, warmup=warmup, iterations=iterations)
        difference = (direct_output - dense_output).abs().float()
        passed = bool(
            torch.allclose(direct_output, dense_output, rtol=2e-2, atol=2e-2)
        )
        measurement = TacticMeasurement(
            latency_ms=direct_ms,
            effective_bandwidth_gbps=bytes_per_call / (direct_ms * 1.0e6),
            achieved_tflops=flops / (direct_ms * 1.0e9),
            reconstruction_overhead_pct=100.0 * reconstruct_ms / naive_ms,
            max_abs_error=float(difference.max().item()),
            mean_abs_error=float(difference.mean().item()),
            bottleneck=_classify_bottleneck(direct_ms, dense_ms, reconstruct_ms),
            selected_shape=selected_shape,
            passed=passed,
        )
        receipts.append(
            build_atlas_receipt(
                generated_at=generated_at,
                capability=capability,
                metadata=packed.metadata,
                workload=workload,
                m=m,
                tactic=tactic,
                measurement=measurement,
                runtime_commit=runtime_commit,
                command=command,
                seed=seed,
            )
        )
    return receipts


def _resolve_workloads(args: argparse.Namespace) -> tuple[GemmWorkload, ...]:
    workloads: list[GemmWorkload] = []
    for path in args.workload_file or ():
        workloads.extend(load_workload_file(path))
    for fixture in args.fixture or ():
        workloads.append(LEGACY_FIXTURES[fixture])
    for shape in args.shape or ():
        workloads.append(
            parse_shape_spec(
                shape,
                model_id=args.model_id,
                tp_world_size=args.tp,
                phase=args.phase,
                grouped_moe=args.grouped_moe,
            )
        )
    return tuple(workloads) if workloads else GENERIC_WORKLOADS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu-oracle", "sm121"), default="cpu-oracle")
    parser.add_argument(
        "--shape",
        action="append",
        help="arbitrary workload as OPERATOR:K:N; repeat as needed",
    )
    parser.add_argument(
        "--workload-file",
        action="append",
        type=Path,
        help="JSON list of arbitrary workload objects",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        choices=tuple(LEGACY_FIXTURES),
        help="optional reproducibility fixture; not a supported-model registry",
    )
    parser.add_argument("--model-id", default="unbound")
    parser.add_argument("--phase", default="decode")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--grouped-moe", action="store_true")
    parser.add_argument("--bits", nargs="+", type=int, default=[3])
    parser.add_argument("--m", nargs="+", type=int, default=list(DECODE_M_SWEEP))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--no-tactic-sweep", action="store_true")
    parser.add_argument("--runtime-commit")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    if any(bits < 1 or bits > 8 for bits in args.bits):
        raise SystemExit("--bits values must be in [1, 8]")
    workloads = _resolve_workloads(args)
    if args.backend == "cpu-oracle":
        report: dict[str, Any] = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "backend": "cpu-oracle",
            "capability": inspect_capability().as_dict(),
            "cases": [
                run_cpu_oracle_check(seed=args.seed, bits=bits) for bits in args.bits
            ],
        }
    else:
        embedded_commit = os.environ.get("RUNTIME_SOURCE_COMMIT", "")
        runtime_commit = args.runtime_commit or (
            embedded_commit if len(embedded_commit) == 40 else None
        )
        runtime_commit = runtime_commit or git_commit(Path(__file__).parents[2])
        command = [sys.executable, "-m", "kernel_lab.exl3.benchmark", *raw_argv]
        receipts: list[dict[str, Any]] = []
        for workload in workloads:
            for bits in args.bits:
                for m in args.m:
                    receipts.extend(
                        run_sm121_workload(
                            workload=workload,
                            bits=bits,
                            m=m,
                            seed=args.seed,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            runtime_commit=runtime_commit,
                            command=command,
                            tactic_sweep=not args.no_tactic_sweep,
                        )
                    )
        report = build_catalog(receipts)

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.backend == "cpu-oracle":
        return 0 if all(case["passed"] for case in report["cases"]) else 1
    return 0 if all(row["run_status"] == "passed" for row in report["receipts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
