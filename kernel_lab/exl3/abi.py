"""Versioned contract for the existing ExLlamaV3 EXL3/MCG tensor ABI.

This module does not define a new quantization format.  It makes the exact
layout already consumed by ``overlay/exl3.py`` and ExLlamaV3 explicit and
machine-checkable for the kernel lab.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

ABI_NAME = "exllamav3.exl3.mcg"
ABI_VERSION = 1
TILE_SIZE = 16
DEFAULT_BITS = 3
MCG_MULTIPLIER = 0xCBAC1FED
MCG_MARKER_SIGNED_INT32 = -877912083
PINNED_EXLLAMAV3_COMMIT = "c5d9c657966ffeeaa9353f0cc899f18629da4a13"
SUPPORTED_BITS = tuple(range(1, 9))


@dataclass(frozen=True)
class PackedExl3Metadata:
    """Serializable metadata for one K-major ``A @ W`` packed matrix."""

    in_features: int
    out_features: int
    bits: int = DEFAULT_BITS
    codebook: str = "mcg"
    abi_name: str = ABI_NAME
    abi_version: int = ABI_VERSION
    tile_size: int = TILE_SIZE
    trellis_dtype: str = "int16"
    scale_dtype: str = "float16"
    marker_dtype: str = "int32"
    matrix_layout: str = "k_major_16x16_tensor_core_permuted"
    exllamav3_commit: str = os.environ.get(
        "EXLLAMAV3_COMMIT", PINNED_EXLLAMAV3_COMMIT
    )

    def __post_init__(self) -> None:
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if self.in_features % self.tile_size:
            raise ValueError("in_features must be divisible by 16")
        if self.out_features % 128:
            raise ValueError("out_features must be divisible by 128 for the direct kernel")
        if self.bits not in SUPPORTED_BITS:
            raise ValueError("EXL3 trellis bits must be in [1, 8]")
        if self.codebook != "mcg":
            raise ValueError("the portable kernel-lab oracle currently supports MCG only")
        if self.abi_name != ABI_NAME or self.abi_version != ABI_VERSION:
            raise ValueError("unsupported EXL3 ABI name or version")
        if self.tile_size != TILE_SIZE:
            raise ValueError("EXL3 trellis tiles must be 16x16")

    @property
    def trellis_shape(self) -> tuple[int, int, int]:
        return (
            self.in_features // self.tile_size,
            self.out_features // self.tile_size,
            self.bits * self.tile_size,
        )

    @property
    def encoded_shape(self) -> tuple[int, int, int]:
        return (*self.trellis_shape[:2], self.tile_size * self.tile_size)

    @property
    def trellis_payload_bytes(self) -> int:
        return int(np.prod(self.trellis_shape, dtype=np.int64)) * np.dtype(np.int16).itemsize

    @property
    def total_payload_bytes(self) -> int:
        return self.trellis_payload_bytes + 2 * (self.in_features + self.out_features) + 4

    @property
    def effective_bpw(self) -> float:
        return 8.0 * self.total_payload_bytes / (self.in_features * self.out_features)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> PackedExl3Metadata:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise TypeError("EXL3 metadata JSON must contain an object")
        return cls(**payload)


@dataclass(frozen=True)
class PackedExl3Weight:
    """Validated NumPy view of one serialized EXL3 matrix."""

    metadata: PackedExl3Metadata
    trellis: np.ndarray
    suh: np.ndarray
    svh: np.ndarray
    mcg: np.ndarray

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        expected = self.metadata.trellis_shape
        if self.trellis.shape != expected or self.trellis.dtype != np.int16:
            raise ValueError(f"trellis must be int16 with shape {expected}, got {self.trellis.dtype} {self.trellis.shape}")
        if self.suh.shape != (self.metadata.in_features,) or self.suh.dtype != np.float16:
            raise ValueError("suh must be float16 with one entry per input feature")
        if self.svh.shape != (self.metadata.out_features,) or self.svh.dtype != np.float16:
            raise ValueError("svh must be float16 with one entry per output feature")
        if self.mcg.shape not in {(), (1,)} or self.mcg.dtype != np.int32:
            raise ValueError("mcg must be a scalar/length-one int32 marker")
        if int(self.mcg.reshape(-1)[0]) != MCG_MARKER_SIGNED_INT32:
            raise ValueError("mcg marker does not select the pinned MCG codebook")
        for name, tensor in (("trellis", self.trellis), ("suh", self.suh), ("svh", self.svh), ("mcg", self.mcg)):
            if not tensor.flags.c_contiguous:
                raise ValueError(f"{name} must be contiguous")

    @property
    def payload_nbytes(self) -> int:
        return sum(t.nbytes for t in (self.trellis, self.suh, self.svh, self.mcg))

    def tensor_manifest(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "nbytes": tensor.nbytes}
            for name, tensor in (
                ("trellis", self.trellis),
                ("suh", self.suh),
                ("svh", self.svh),
                ("mcg", self.mcg),
            )
        }
