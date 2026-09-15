"""Deterministic, dependency-light EXL3 pack/unpack and correctness oracle.

The implementation is intentionally slow and explicit.  It is an independent
CPU oracle for tests and diagnostics, not a serving implementation.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .abi import MCG_MULTIPLIER, TILE_SIZE, PackedExl3Weight


def _as_u16(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array).view(np.uint16)


def pack_trellis(encoded: np.ndarray, bits: int = 3) -> np.ndarray:
    """Pack ExLlamaV3 tensor-core-ordered states exactly as ``pack.cu``.

    Each 256-state tile is split into 16 independently aligned spans.  Values
    are written most-significant-bit first, then adjacent uint16 words are
    swapped to match the CUDA kernel's little-endian ``SWAP16`` store.
    """

    if encoded.dtype != np.int16 or encoded.ndim < 1 or encoded.shape[-1] != 256:
        raise ValueError("encoded must be an int16 array ending in 256 states")
    if bits < 1 or bits > 8:
        raise ValueError("EXL3 trellis bits must be in [1, 8]")
    values = _as_u16(encoded)
    if np.any(values > ((1 << bits) - 1)):
        raise ValueError("encoded state exceeds the requested trellis bitrate")

    flat = values.reshape(-1, 256)
    raw = np.empty((flat.shape[0], TILE_SIZE * bits), dtype=np.uint16)
    mask = (1 << bits) - 1
    for row_index, row in enumerate(flat):
        for span in range(TILE_SIZE):
            accumulator = 0
            start = span * TILE_SIZE
            for value in row[start : start + TILE_SIZE]:
                accumulator = (accumulator << bits) | (int(value) & mask)
            for word in range(bits):
                shift = 16 * (bits - 1 - word)
                raw[row_index, span * bits + word] = (accumulator >> shift) & 0xFFFF

    packed = raw.copy()
    packed[:, 0::2] = raw[:, 1::2]
    packed[:, 1::2] = raw[:, 0::2]
    return packed.reshape(*encoded.shape[:-1], TILE_SIZE * bits).view(np.int16)


def unpack_trellis(packed: np.ndarray, bits: int | None = None) -> np.ndarray:
    """Inverse of :func:`pack_trellis`, preserving the official state order."""

    if packed.dtype != np.int16 or packed.ndim < 1:
        raise ValueError("packed must be an int16 array")
    if bits is None:
        if packed.shape[-1] % TILE_SIZE:
            raise ValueError("packed word count is not divisible by 16")
        bits = packed.shape[-1] // TILE_SIZE
    if bits < 1 or bits > 8 or packed.shape[-1] != TILE_SIZE * bits:
        raise ValueError("packed shape does not match the requested bitrate")

    words = _as_u16(packed).reshape(-1, TILE_SIZE * bits)
    raw = words.copy()
    raw[:, 0::2] = words[:, 1::2]
    raw[:, 1::2] = words[:, 0::2]
    decoded = np.empty((words.shape[0], 256), dtype=np.uint16)
    mask = (1 << bits) - 1
    for row_index, row in enumerate(raw):
        for span in range(TILE_SIZE):
            accumulator = 0
            for word in row[span * bits : (span + 1) * bits]:
                accumulator = (accumulator << 16) | int(word)
            base = span * TILE_SIZE
            for index in range(TILE_SIZE):
                shift = bits * (TILE_SIZE - 1 - index)
                decoded[row_index, base + index] = (accumulator >> shift) & mask
    return decoded.reshape(*packed.shape[:-1], 256).view(np.int16)


def _lop3(a: np.ndarray, b: int, c: int, lut: int) -> np.ndarray:
    """Portable PTX ``lop3.b32`` truth-table evaluation."""

    a = np.asarray(a, dtype=np.uint32)
    b_u = np.uint32(b)
    c_u = np.uint32(c)
    out = np.zeros_like(a, dtype=np.uint32)
    for index in range(8):
        if not ((lut >> index) & 1):
            continue
        a_term = a if (index & 4) else ~a
        b_term = b_u if (index & 2) else ~b_u
        c_term = c_u if (index & 1) else ~c_u
        out |= a_term & b_term & c_term
    return out


def decode_mcg(states: np.ndarray) -> np.ndarray:
    """Decode uint16 trellis states with the pinned MCG procedural codebook."""

    state_u32 = np.asarray(states, dtype=np.uint16).astype(np.uint32)
    product = (state_u32.astype(np.uint64) * MCG_MULTIPLIER) & 0xFFFFFFFF
    words = _lop3(product.astype(np.uint32), 0x8FFF8FFF, 0x3B603B60, 0x6A)
    low = (words & 0xFFFF).astype(np.uint16).view(np.float16)
    high = (words >> 16).astype(np.uint16).view(np.float16)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.add(low, high, dtype=np.float16)


@lru_cache(maxsize=1)
def tensor_core_permutation() -> tuple[np.ndarray, np.ndarray]:
    """Return the official row-major→MMA permutation and its inverse."""

    permutation = np.empty(256, dtype=np.int64)
    for thread in range(32):
        r0 = (thread % 4) * 2
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = thread // 4
        cols = (c0, c0 + 8)
        values = (
            rows[0] * 16 + cols[0],
            rows[1] * 16 + cols[0],
            rows[2] * 16 + cols[0],
            rows[3] * 16 + cols[0],
            rows[0] * 16 + cols[1],
            rows[1] * 16 + cols[1],
            rows[2] * 16 + cols[1],
            rows[3] * 16 + cols[1],
        )
        permutation[thread * 8 : (thread + 1) * 8] = values
    return permutation, np.argsort(permutation)


def decode_tiles(packed: np.ndarray, bits: int = 3) -> np.ndarray:
    """Decode packed tiles to row-major ``[..., 16, 16]`` FP16 values."""

    states = unpack_trellis(packed, bits)
    values_tc = decode_mcg(states)
    _, inverse = tensor_core_permutation()
    return values_tc[..., inverse].reshape(*packed.shape[:-1], 16, 16)


@lru_cache(maxsize=8)
def hadamard(dimension: int) -> np.ndarray:
    """Normalized Sylvester Hadamard used by EXL3 for the 128-wide blocks."""

    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    matrix = np.ones((1, 1), dtype=np.float32)
    while matrix.shape[0] < dimension:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / math.sqrt(dimension)


def _right_hadamard_blocks(array: np.ndarray, block: int = 128) -> np.ndarray:
    if array.shape[-1] % block:
        raise ValueError("feature dimension must be divisible by the Hadamard block")
    original_shape = array.shape
    view = np.asarray(array, dtype=np.float32).reshape(-1, original_shape[-1] // block, block)
    result = view @ hadamard(block)
    return result.reshape(original_shape)


def _left_hadamard_blocks(array: np.ndarray, block: int = 128) -> np.ndarray:
    if array.shape[0] % block:
        raise ValueError("input dimension must be divisible by the Hadamard block")
    matrix = np.asarray(array, dtype=np.float32)
    result = np.empty_like(matrix)
    transform = hadamard(block)
    for start in range(0, matrix.shape[0], block):
        result[start : start + block] = transform @ matrix[start : start + block]
    return result


def reconstruct_inner(weight: PackedExl3Weight) -> np.ndarray:
    """Materialize the MCG-decoded inner matrix before Hadamard/scales."""

    tiles = decode_tiles(weight.trellis, weight.metadata.bits)
    return tiles.transpose(0, 2, 1, 3).reshape(
        weight.metadata.in_features, weight.metadata.out_features
    )


def reconstruct_weight(weight: PackedExl3Weight) -> np.ndarray:
    """Materialize the complete reference matrix in FP32."""

    inner = reconstruct_inner(weight)
    transformed = _left_hadamard_blocks(inner)
    transformed *= weight.suh.astype(np.float32)[:, None]
    transformed = _right_hadamard_blocks(transformed)
    transformed *= weight.svh.astype(np.float32)[None, :]
    return transformed


def matmul_materialized_reference(x: np.ndarray, weight: PackedExl3Weight) -> np.ndarray:
    """Independent dense correctness oracle (allowed to materialize weights)."""

    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != weight.metadata.in_features:
        raise ValueError("activation K does not match packed weight metadata")
    return x @ reconstruct_weight(weight)


def matmul_streaming_reference(x: np.ndarray, weight: PackedExl3Weight) -> np.ndarray:
    """Direct packed oracle that never constructs a full dequantized weight.

    This mirrors the serving dataflow at CPU-oracle speed: transform the input,
    decode one 16x16 trellis tile, consume it immediately, and discard it.
    """

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != weight.metadata.in_features:
        raise ValueError("x must be [M, in_features]")
    xh = _right_hadamard_blocks(x * weight.suh.astype(np.float32)[None, :])
    output = np.zeros((x.shape[0], weight.metadata.out_features), dtype=np.float32)
    for tile_k in range(weight.trellis.shape[0]):
        k0 = tile_k * TILE_SIZE
        k1 = k0 + TILE_SIZE
        for tile_n in range(weight.trellis.shape[1]):
            n0 = tile_n * TILE_SIZE
            n1 = n0 + TILE_SIZE
            tile = decode_tiles(weight.trellis[tile_k, tile_n], weight.metadata.bits)
            output[:, n0:n1] += xh[:, k0:k1] @ tile.astype(np.float32)
    output = _right_hadamard_blocks(output)
    output *= weight.svh.astype(np.float32)[None, :]
    return output
