"""Model-agnostic extension seams for Atlas runtime kernel work.

Only single-matrix decode is implemented in Milestone 0.  These typed keys
prevent that prototype from hard-coding assumptions that would block grouped
MoE, prefill, mixed representation dispatch, or runtime adapters later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ServingPhase:
    """Convenience phase names, not an allowlist."""

    DECODE = "decode"
    PREFILL = "prefill"


@dataclass(frozen=True)
class Representation:
    """The exact packed ABI consumed by a kernel candidate."""

    format: str
    abi_name: str
    abi_version: int
    bits_per_weight: float
    codebook: str | None = None

    @classmethod
    def exl3_mcg(cls, bits: int) -> Representation:
        return cls("exl3", "exllamav3.exl3.mcg", 1, float(bits), "mcg")


@dataclass(frozen=True)
class KernelKey:
    compute_capability: tuple[int, int]
    phase: str
    representation: Representation
    m: int
    n: int
    k: int
    grouped: bool = False
    model_id: str = "unbound"
    operator: str = "linear"
    tp_world_size: int = 1


@dataclass(frozen=True)
class KernelCandidate:
    name: str
    backend: str
    tactic: dict[str, Any]


class KernelOracle(Protocol):
    def candidates(self, key: KernelKey) -> tuple[KernelCandidate, ...]: ...

    def select(self, key: KernelKey) -> KernelCandidate: ...


class RuntimeBackend(Protocol):
    name: str

    def supports(self, key: KernelKey) -> tuple[bool, str]: ...

    def run(self, *args: Any, **kwargs: Any) -> Any: ...
