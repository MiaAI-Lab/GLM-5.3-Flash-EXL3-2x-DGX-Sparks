"""Model-agnostic GEMM workload descriptions and optional legacy fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GemmWorkload:
    """One exact ``A[M,K] @ W[K,N]`` operator shape.

    Model and operator names are provenance only. Kernel compatibility is
    determined by the numerical shape and runtime/representation fields.
    """

    operator: str
    k: int
    n: int
    model_id: str = "unbound"
    tp_world_size: int = 1
    phase: str = "decode"
    grouped_moe: bool = False

    def __post_init__(self) -> None:
        if not self.operator:
            raise ValueError("operator must be non-empty")
        if self.k <= 0 or self.n <= 0:
            raise ValueError("K and N must be positive")
        if self.k % 16:
            raise ValueError("EXL3 K must be divisible by 16")
        if self.n % 128:
            raise ValueError("the direct EXL3 kernel requires N divisible by 128")
        if self.tp_world_size < 1:
            raise ValueError("tp_world_size must be positive")
        if not self.phase:
            raise ValueError("phase must be non-empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GemmWorkload:
        allowed = {
            "operator",
            "k",
            "n",
            "model_id",
            "tp_world_size",
            "phase",
            "grouped_moe",
        }
        extra = set(payload) - allowed
        if extra:
            raise ValueError(f"unknown workload fields: {sorted(extra)}")
        return cls(**payload)


def parse_shape_spec(
    value: str,
    *,
    model_id: str = "unbound",
    tp_world_size: int = 1,
    phase: str = "decode",
    grouped_moe: bool = False,
) -> GemmWorkload:
    """Parse ``OPERATOR:K:N`` without consulting a model registry."""

    try:
        operator, raw_k, raw_n = value.rsplit(":", 2)
        k, n = int(raw_k), int(raw_n)
    except (TypeError, ValueError) as exc:
        raise ValueError("shape must use OPERATOR:K:N") from exc
    return GemmWorkload(
        operator=operator,
        k=k,
        n=n,
        model_id=model_id,
        tp_world_size=tp_world_size,
        phase=phase,
        grouped_moe=grouped_moe,
    )


def load_workload_file(path: str | Path) -> tuple[GemmWorkload, ...]:
    """Load either a JSON list or ``{"workloads": [...]}`` document."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("workloads") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("workload file must contain a non-empty workload list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("each workload must be a JSON object")
    return tuple(GemmWorkload.from_dict(row) for row in rows)


# Defaults are hardware-oriented shape classes, not a supported-model list.
GENERIC_WORKLOADS = (
    GemmWorkload("linear.square", 4096, 4096),
    GemmWorkload("linear.expand", 4096, 11008),
    GemmWorkload("linear.contract", 11008, 4096),
)


# Kept only as reproducible fixtures for the original M0 measurements.
LEGACY_FIXTURES = {
    "glm52.tp2.gate": GemmWorkload(
        "experts.gate", 6144, 1024, "zai-org/GLM-5.2", 2
    ),
    "glm52.tp2.down": GemmWorkload(
        "experts.down", 1024, 6144, "zai-org/GLM-5.2", 2
    ),
    "hy4.tp2.gate": GemmWorkload(
        "experts.gate", 6144, 1024, "tencent/Hy4-preview", 2
    ),
    "hy4.tp2.down": GemmWorkload(
        "experts.down", 1024, 6144, "tencent/Hy4-preview", 2
    ),
}


DECODE_M_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
