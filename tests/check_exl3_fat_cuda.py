#!/usr/bin/env python3
"""Run fat-kernel numerical checks and nondefault-stream graph replay tests.

Run inside the candidate CUDA image, optionally under Compute Sanitizer.
No model weights or live server are needed.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def check_stream_graph(torch, ext, device):
    torch.manual_seed(781)
    for m, k, n in ((1, 16, 128), (65, 80, 256), (173, 4096, 1024)):
        a = torch.randn(m, k, dtype=torch.float16, device=device)
        q = torch.randint(-30000, 30000, (k // 16, n // 16, 64), dtype=torch.int16, device=device)
        qr = torch.randint_like(q, -30000, 30000)
        scale = torch.randn(n, dtype=torch.float16, device=device)
        scale_r = torch.randn_like(scale)
        indices = torch.randperm(m + 11, device=device)[:m].contiguous()
        route = torch.randn(m, dtype=torch.float16, device=device)

        def buffers():
            return (
                torch.empty(m, n, dtype=torch.float32, device=device),
                torch.empty(m, 2 * n, dtype=torch.float32, device=device),
                torch.empty(m + 11, n, dtype=torch.float32, device=device),
            )

        expected, actual = buffers(), buffers()

        def launch(outputs):
            direct, paired, scatter = outputs
            scatter.zero_()
            ext.exl3_fat_gemm(a, q, direct, scale, 4, True, False)
            ext.exl3_fat_gemm_pair(a, q, qr, paired, scale, scale_r, 4, True, False)
            ext.exl3_fat_gemm_scatter(a, q, scatter, scale, indices, route, 4, True, False)

        default = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(default)
        with torch.cuda.stream(side):
            for _ in range(3):
                launch(actual)
        side.synchronize()
        launch(expected)
        default.synchronize()
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            launch(actual)
        # Update the same input storage between replays. A stale capture must
        # not pass just because all replays read identical data.
        for factor in (0.5, -1.0, 1.5):
            a.mul_(factor)
            launch(expected)
            side.wait_stream(default)
            with torch.cuda.stream(side):
                graph.replay()
            side.synchronize()
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        print(f"PASS nondefault-stream graph M={m} K={k} N={n}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck-path", type=Path, default=Path("/opt/glm53/test_exl3_overlay.py"))
    args = parser.parse_args()
    import torch
    import exllamav3_ext as ext

    device = torch.device("cuda:0")
    for name in ("exl3_fat_gemm", "exl3_fat_gemm_scatter", "exl3_fat_gemm_pair", "exl3_fat_swiglu"):
        if not callable(getattr(ext, name, None)):
            raise RuntimeError(f"candidate symbol missing: {name}")
    spec = importlib.util.spec_from_file_location("fat_selfcheck", args.selfcheck_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {args.selfcheck_path}")
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    for name in ("_check_fat_kernel", "_check_fat_pair", "_check_fat_k_tails", "_check_fat_swiglu"):
        print(f"CHECK {name}", flush=True)
        getattr(checks, name)(device)
        torch.cuda.synchronize(device)
    check_stream_graph(torch, ext, device)
    print("PASS all fat numerical and stream/graph checks", flush=True)


if __name__ == "__main__":
    main()
