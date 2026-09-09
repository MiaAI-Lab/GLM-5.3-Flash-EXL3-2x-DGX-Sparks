"""Stable EXL3 ABI, reference oracle, and direct-runtime adapter."""

from .abi import MCG_MARKER_SIGNED_INT32, PackedExl3Metadata, PackedExl3Weight
from .reference import (
    matmul_materialized_reference,
    matmul_streaming_reference,
    pack_trellis,
    reconstruct_weight,
    unpack_trellis,
)

__all__ = [
    "MCG_MARKER_SIGNED_INT32",
    "PackedExl3Metadata",
    "PackedExl3Weight",
    "matmul_materialized_reference",
    "matmul_streaming_reference",
    "pack_trellis",
    "reconstruct_weight",
    "unpack_trellis",
]
