# SPDX-License-Identifier: Apache-2.0
"""Clamped SwiGLU for the EXL3 fat-expert FP32 gate/up output.

Keep the two products in FP32, then cast once to FP16 for the down Hadamard.
Unlike the thin MoE kernel, E2's gate/up GEMM materializes this intermediate.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _fat_swiglu_kernel(
    gate_up, out, n_elements,
    INTERMEDIATE: tl.constexpr,
    INPUT_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // INTERMEDIATE
    col = offsets % INTERMEDIATE
    valid = offsets < n_elements
    gate = tl.load(gate_up + row * INPUT_STRIDE + col, valid, other=0)
    up = tl.load(gate_up + row * INPUT_STRIDE + INTERMEDIATE + col, valid, other=0)
    gate = tl.minimum(gate, LIMIT, propagate_nan=tl.PropagateNan.ALL)
    up = tl.maximum(
        tl.minimum(up, LIMIT, propagate_nan=tl.PropagateNan.ALL),
        -LIMIT, propagate_nan=tl.PropagateNan.ALL,
    )
    # Match torch.sigmoid's FP32 exp/divide, not the exp2 approximation.
    sigmoid = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    activation = (sigmoid * gate) * up
    tl.store(out + row * OUTPUT_STRIDE + col, activation, valid)


def fat_swiglu(gate_up: torch.Tensor, out: torch.Tensor, limit: float) -> None:
    """Write clamp(gate)*sigmoid(clamp(gate))*clamp(up) to FP16 ``out``.

    Input is the E2 row-major concatenated gate/up FP32 buffer. Row strides
    may exceed the logical width; the input is not modified.
    """
    rows, intermediate = out.shape
    assert gate_up.shape == (rows, 2 * intermediate)
    assert gate_up.dtype == torch.float32 and out.dtype == torch.float16
    assert gate_up.stride(1) == out.stride(1) == 1
    assert gate_up.device == out.device and gate_up.is_cuda
    if rows == 0:
        return
    _fat_swiglu_kernel[(triton.cdiv(rows * intermediate, 1024),)](
        gate_up, out, rows * intermediate,
        INTERMEDIATE=intermediate,
        INPUT_STRIDE=gate_up.stride(0),
        OUTPUT_STRIDE=out.stride(0),
        LIMIT=limit,
        BLOCK=1024,
        enable_fp_fusion=False,
    )
