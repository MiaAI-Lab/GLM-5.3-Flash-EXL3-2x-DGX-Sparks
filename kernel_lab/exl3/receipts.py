"""Canonical Atlas receipt generation for direct EXL3 measurements."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .abi import PackedExl3Metadata
from .backend import CapabilityReceipt, KernelTactic
from .shapes import GemmWorkload

ATLAS_RECEIPT_SCHEMA = "atlas.kernel-benchmark/v1"
ATLAS_CATALOG_SCHEMA = "atlas.kernel-catalog/v1"
RUNTIME_REPOSITORY = (
    "https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks"
)
EXLLAMAV3_REPOSITORY = "https://github.com/turboderp-org/exllamav3"


@dataclass(frozen=True)
class TacticMeasurement:
    latency_ms: float
    effective_bandwidth_gbps: float
    achieved_tflops: float
    reconstruction_overhead_pct: float
    max_abs_error: float
    mean_abs_error: float
    bottleneck: str
    selected_shape: int
    passed: bool


def git_commit(path: str | Path) -> str:
    """Return the exact runtime commit or fail instead of inventing provenance."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(path),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError("unable to resolve the runtime source commit") from exc
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError("runtime source commit is not a full Git SHA")
    return commit


def build_atlas_receipt(
    *,
    generated_at: datetime,
    capability: CapabilityReceipt,
    metadata: PackedExl3Metadata,
    workload: GemmWorkload,
    m: int,
    tactic: KernelTactic,
    measurement: TacticMeasurement,
    runtime_commit: str,
    command: list[str],
    seed: int,
) -> dict[str, Any]:
    """Build one exact-shape, direct-packed, measured receipt."""

    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    if not capability.available:
        raise ValueError("a measured receipt requires an available CUDA capability")
    if not capability.device_name or not capability.compute_capability:
        raise ValueError("a measured receipt requires hardware identity")
    if not capability.cuda_version or not capability.driver_version:
        raise ValueError("a measured receipt requires CUDA and driver versions")
    if len(runtime_commit) != 40 or any(
        char not in "0123456789abcdef" for char in runtime_commit
    ):
        raise ValueError("runtime_commit must be a full Git SHA")

    identity = {
        "at": generated_at.isoformat(),
        "device": capability.device_name,
        "model": workload.model_id,
        "operator": workload.operator,
        "m": m,
        "n": workload.n,
        "k": workload.k,
        "bits": metadata.bits,
        "tactic": tactic.name,
        "runtime_commit": runtime_commit,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    cc = f"{capability.compute_capability[0]}.{capability.compute_capability[1]}"
    kernel_name = _resolved_kernel_name(tactic, measurement.selected_shape)
    return {
        "schema_version": ATLAS_RECEIPT_SCHEMA,
        "receipt_id": f"exl3-sm121-{digest}",
        "generated_at": generated_at.isoformat(),
        "evidence_kind": "measured",
        "run_status": "passed" if measurement.passed else "failed",
        "hardware": {
            "vendor": "nvidia",
            "device_name": capability.device_name,
            "compute_capability": cc,
            "device_uuid": capability.device_uuid,
            "cuda_version": capability.cuda_version,
            "driver_version": capability.driver_version,
        },
        "representation": {
            "format": "exl3",
            "abi_name": metadata.abi_name,
            "abi_version": metadata.abi_version,
            "bits_per_weight": float(metadata.bits),
            "codebook": metadata.codebook,
            "fused_transform": True,
            "full_precision_materialized": False,
        },
        "workload": {
            "model_id": workload.model_id,
            "operator": workload.operator,
            "phase": workload.phase,
            "m": m,
            "n": workload.n,
            "k": workload.k,
            "tp_world_size": workload.tp_world_size,
            "grouped_moe": workload.grouped_moe,
        },
        "backend": {
            "name": "exllamav3_direct_exl3",
            "kernel_name": kernel_name,
            "repository": EXLLAMAV3_REPOSITORY,
            "commit": metadata.exllamav3_commit,
            "execution_path": "direct_packed",
        },
        "metrics": {
            "latency_ms": measurement.latency_ms,
            "effective_bandwidth_gbps": measurement.effective_bandwidth_gbps,
            "achieved_tflops": measurement.achieved_tflops,
            "reconstruction_overhead_pct": measurement.reconstruction_overhead_pct,
            "max_abs_error": measurement.max_abs_error,
            "mean_abs_error": measurement.mean_abs_error,
            "bottleneck": measurement.bottleneck,
        },
        "provenance": {
            "producer_schema": ATLAS_RECEIPT_SCHEMA,
            "source_repository": RUNTIME_REPOSITORY,
            "source_commit": runtime_commit,
            "command": command,
            "seed": seed,
            "source_receipt_sha256": None,
            "notes": [
                "dense and reconstruct baselines were timed separately from direct latency",
                f"extension selected EXL3 kernel shape {measurement.selected_shape}",
            ],
        },
    }


def build_catalog(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": ATLAS_CATALOG_SCHEMA, "receipts": receipts}


def _resolved_kernel_name(tactic: KernelTactic, selected_shape: int) -> str:
    if tactic.force_shape_idx > 0:
        return tactic.kernel_name
    if selected_shape == 90:
        return "exl3_gemv.qtip_small_m"
    if selected_shape == 0:
        return "exl3_gemv.specialized"
    return f"exl3_gemm.shape{selected_shape}.autotuned"
