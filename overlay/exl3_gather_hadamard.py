# SPDX-License-Identifier: Apache-2.0
"""Gather fat-expert token rows directly into the pre-scaled Hadamard-128."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_hadamard(
    x, indices, scale, out, n_elements,
    HIDDEN: tl.constexpr, SOURCE_ROWS: tl.constexpr,
    INPUT_STRIDE: tl.constexpr, INDEX_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr, LANES: tl.constexpr,
):
    lane = tl.arange(0, LANES)
    offset = (tl.program_id(0).to(tl.int64) * LANES + lane) * 4
    row, col = offset // HIDDEN, offset % HIDDEN
    valid = offset < n_elements
    token = tl.load(indices + row * INDEX_STRIDE, valid, other=0)
    in_bounds = (token >= 0) & (token < SOURCE_ROWS)
    tl.device_assert((~valid) | in_bounds, "token index out of bounds")
    columns = col[:, None] + tl.arange(0, 4)[None, :]
    value = tl.load(x + token[:, None] * INPUT_STRIDE + columns,
                    valid[:, None] & in_bounds[:, None], other=0).to(tl.float32)
    prescale = tl.load(scale + columns).to(tl.float32)
    # Native had_r_128 multiplies in half before its FP32 butterflies.
    values = (value * prescale).to(tl.float16).to(tl.float32)
    even, odd = tl.split(tl.reshape(values, (LANES, 2, 2)))
    v0, v2 = tl.split(even)
    v1, v3 = tl.split(odd)
    s0, d0 = v0 + v1, v0 - v1
    s1, d1 = v2 + v3, v2 - v3
    h0, h1 = s0 + s1, d0 + d1
    h2, h3 = s0 - s1, d0 - d1
    for shift in tl.static_range(5):
        bit = 1 << shift
        peer = lane ^ bit
        p0 = tl.gather(h0, peer, axis=0)
        p1 = tl.gather(h1, peer, axis=0)
        p2 = tl.gather(h2, peer, axis=0)
        p3 = tl.gather(h3, peer, axis=0)
        high = (lane & bit) != 0
        h0 = tl.where(high, -h0, h0) + p0
        h1 = tl.where(high, -h1, h1) + p1
        h2 = tl.where(high, -h2, h2) + p2
        h3 = tl.where(high, -h3, h3) + p3
    values = tl.reshape(tl.join(tl.join(h0, h2), tl.join(h1, h3)), (LANES, 4))
    tl.store(out + row[:, None] * OUTPUT_STRIDE + columns,
             values * 0.088388347648, valid[:, None])


def gather_hadamard(x, indices, scale, out):
    """Equivalent to index_select(x, 0, indices), then had_r_128(..., scale).

    Inputs/output are FP16 with contiguous columns; row padding is allowed.
    Indices are valid int64 token IDs (possibly strided/repeated). The caller
    must supply output storage that does not overlap any input. No host reads
    of the indices are introduced. Invalid IDs are outside this internal ABI;
    debug Triton builds assert their bounds, and loads are always bounds-masked.
    """
    assert x.ndim == out.ndim == 2 and indices.ndim == scale.ndim == 1
    rows, hidden = out.shape
    assert hidden > 0 and hidden % 128 == 0 and x.shape[1] == hidden
    assert indices.numel() == rows and indices.dtype == torch.long
    assert x.dtype == scale.dtype == out.dtype == torch.float16
    assert scale.shape == (hidden,) and scale.stride(0) == 1
    assert x.stride(1) == out.stride(1) == 1
    assert x.is_cuda and x.device == indices.device == scale.device == out.device
    if rows == 0:
        return
    with torch.cuda.device(x.device):
        _gather_hadamard[(triton.cdiv(rows * hidden, 1024),)](
            x, indices, scale, out, rows * hidden,
            HIDDEN=hidden, SOURCE_ROWS=x.shape[0], INPUT_STRIDE=x.stride(0),
            INDEX_STRIDE=indices.stride(0), OUTPUT_STRIDE=out.stride(0),
            LANES=256, num_warps=8, enable_fp_fusion=False,
        )
