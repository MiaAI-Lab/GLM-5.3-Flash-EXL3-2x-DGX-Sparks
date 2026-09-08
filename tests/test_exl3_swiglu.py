#!/usr/bin/env python3
"""GPU reference checks for EXL3 fat activation; runnable without vLLM import."""
import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "exl3_swiglu", Path(__file__).resolve().parents[1] / "overlay/exl3_swiglu.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def reference(x, limit):
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return ((torch.sigmoid(gate) * gate) * up).half()


def main():
    torch.manual_seed(93715)
    cases = 0
    max_abs = 0.0
    for rows, width in [(0, 1024), (1, 17), (129, 1024), (257, 512), (1024, 1024), (7168, 1024), (13, 2048)]:
        for limit in (1.0, 10.0, 20.0):
            for padded in (False, True):
                padding = 11 if padded else 0
                x = (torch.randn(rows, 2 * width + padding, device='cuda') * 15)[:, :2 * width]
                out_base = torch.full((rows, width + padding), 123.0, device='cuda', dtype=torch.float16)
                out = out_base[:, :width]
                original = x.clone()
                expected = reference(x, limit)
                mod.fat_swiglu(x, out, limit)
                # FP32 sigmoid implementations can round differently at half-ULP boundaries.
                torch.testing.assert_close(out, expected, rtol=0.001, atol=0.000002)
                assert torch.equal(x, original), 'input must not be modified'
                assert torch.all(out_base[:, width:] == 123.0), 'row padding overwritten'
                if rows:
                    max_abs = max(max_abs, (out - expected).abs().max().item())
                cases += 1
    # Finite saturation, signed zeros, infinities and NaNs follow torch clamp/sigmoid.
    v = torch.tensor([-float('inf'), -100., -20., -10., -1., -0., 0., 1., 10., 20., 100., float('inf'), float('nan')], device='cuda')
    g,u = torch.meshgrid(v, v, indexing='ij')
    x = torch.cat((g, u), dim=1).contiguous()
    out = torch.empty_like(g, dtype=torch.float16)
    mod.fat_swiglu(x, out, 10.0)
    torch.testing.assert_close(out, reference(x, 10.0), rtol=0.001, atol=0.000002, equal_nan=True)
    # Real decode/prefill use graph capture; validate graph output on changed data.
    x = torch.randn(257, 2048, device='cuda')
    out = torch.empty(257, 1024, device='cuda', dtype=torch.float16)
    mod.fat_swiglu(x, out, 10.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mod.fat_swiglu(x, out, 10.0)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(out, reference(x, 10.0), rtol=0.001, atol=0.000002)
    print(f'PASS {cases + 2} activation reference cases, max_abs={max_abs}')


if __name__ == '__main__':
    main()
