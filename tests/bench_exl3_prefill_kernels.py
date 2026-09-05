#!/usr/bin/env python3
"""Reproduce the historical E2 kernel and projection-copy workloads on CUDA.

No deployment, model load, source patches or reduced-work quick mode.
See docs/exl3-prefill-validation.md for scope and measurement limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import statistics
import sys
from pathlib import Path

MS = (129, 256, 512, 1024, 2048, 4096, 7168)
PAIR_CASES = (
    ('tail', 173, 4096, 1024), ('medium', 637, 4096, 1024),
    ('large', 2305, 4096, 1024), ('full', 7168, 4096, 1024),
    ('wide', 1025, 2048, 1536),
)
SEED = 618


def packed(k: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Fixed pseudo-random packed values prevent compressible/cache-special data.
    g = torch.Generator(device="cpu").manual_seed(k * 10000 + n)
    trellis = torch.randint(
        -30000, 30000, (k // 16, n // 16, 64), dtype=torch.int16, generator=g
    ).cuda()
    svh = torch.where(
        torch.rand(n, generator=g) > 0.5, torch.tensor(1.0), torch.tensor(-1.0)
    ).half().cuda()
    return trellis, svh


def elapsed_ms(call, warmups: int = 3, repeats: int = 7) -> float:
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        call()
        end.record()
        end.synchronize()
        values.append(float(begin.elapsed_time(end)))
    return statistics.median(values)


def direct_case(m: int, k: int, n: int) -> float:
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    q, svh = packed(k, n)
    out = torch.empty((m, n), dtype=torch.float32, device="cuda")
    value = elapsed_ms(lambda: ext.exl3_fat_gemm(a, q, out, svh, 4, True, False))
    del a, q, svh, out
    torch.cuda.empty_cache()
    return value


def scatter_case(m: int, k: int, n: int) -> float:
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    q, svh = packed(k, n)
    out = torch.zeros((m + 17, n), dtype=torch.float32, device="cuda")
    token_idx = torch.randperm(m + 17, device="cuda")[:m].contiguous()
    route = torch.full((m,), 1e-4, dtype=torch.float16, device="cuda")
    value = elapsed_ms(
        lambda: ext.exl3_fat_gemm_scatter(
            a, q, out, svh, token_idx, route, 4, True, False
        )
    )
    del a, q, svh, out, token_idx, route
    torch.cuda.empty_cache()
    return value


def default_selfcheck_path() -> Path:
    installed = Path('/opt/glm53/test_exl3_overlay.py')
    return installed if installed.is_file() else Path(__file__).with_name('test_exl3_overlay.py')


def load_selfcheck(path: Path):
    """Normal file-module import, not AST extraction or dependency stubbing."""
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("overlay_selfcheck", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    print(f'SELFCHECK path={path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}', flush=True)
    return module


def required_checks(extension, module, validation: str) -> tuple[str, ...]:
    symbols = ['exl3_fat_gemm', 'exl3_fat_gemm_scatter']
    checks = ['_check_fat_kernel']
    if validation == 'candidate':
        symbols += ['exl3_fat_gemm_pair', 'exl3_fat_swiglu']
        checks += ['_check_fat_pair', '_check_fat_swiglu', '_check_fat_k_tails']
    elif validation != 'upstream':
        raise ValueError(f'Unknown validation mode: {validation}')
    for owner, names in ((extension, symbols), (module, checks)):
        missing = [name for name in names if not callable(getattr(owner, name, None))]
        if missing:
            raise RuntimeError(f'{validation} validation missing required symbols: {missing}')
    return tuple(checks)


def correctness(path: Path, validation: str) -> None:
    module = load_selfcheck(path)
    checks = required_checks(ext, module, validation)
    device = torch.device('cuda')
    for name in checks:
        getattr(module, name)(device)
    print(f'CORRECTNESS validation={validation} checks={checks} PASS', flush=True)
    if validation == 'upstream':
        print('SCOPE upstream baseline only; pair/SwiGLU/K-tail candidate checks '
              'not requested, NOT candidate validation', flush=True)


def kernel_panel() -> None:
    # Historical activation/route randomness was unseeded; pin it for reruns.
    # Packed weights retain the original independent CPU generator seed.
    torch.manual_seed(SEED)
    rows: dict[int, tuple[float, float]] = {}
    # Production TP2 shapes: combined gate+up is 4096x2048; down is 1024x4096.
    for m in MS:
        direct = direct_case(m, 4096, 2048)
        scatter = scatter_case(m, 1024, 4096)
        rows[m] = (direct, scatter)
        print(
            f"KERNEL m={m} direct_ms={direct:.6f} scatter_ms={scatter:.6f} "
            f"pair_ms={direct + scatter:.6f}",
            flush=True,
        )
    panel = sum(direct + scatter for direct, scatter in rows.values())
    normalized = [
        (direct + scatter) / m for m, (direct, scatter) in rows.items()
    ]
    row_geomean_us = math.exp(sum(math.log(v * 1000) for v in normalized) / len(normalized))
    print(f"METRIC fat_panel_ms={panel:.6f}")
    print(f"METRIC fat_row_geomean_us={row_geomean_us:.6f}")
    for m, (direct, scatter) in rows.items():
        print(f"METRIC m{m}_direct_ms={direct:.6f}")
        print(f"METRIC m{m}_scatter_ms={scatter:.6f}")


def projection_panel(paired: bool) -> None:
    # Reset for each arm so operands match the historical seed=618 sequence.
    torch.manual_seed(SEED)
    results = {}
    for label, m, k, n in PAIR_CASES:
        a = torch.randn(m, k, device='cuda', dtype=torch.float16)
        left = torch.randint(-32768, 32767, (k//16, n//16, 64), device='cuda', dtype=torch.int16)
        right = torch.randint_like(left, -32768, 32767)
        sl = torch.randn(n, device='cuda', dtype=torch.float16)
        sr = torch.randn_like(sl)
        packed = torch.empty(k//16, 2*n//16, 64, device='cuda', dtype=torch.int16)
        scales = torch.empty(2*n, device='cuda', dtype=torch.float16)
        out = torch.empty(m, 2*n, device='cuda', dtype=torch.float32)

        def call():
            if paired:
                ext.exl3_fat_gemm_pair(a, left, right, out, sl, sr, 4, True, False)
            else:
                packed[:, :n//16].copy_(left)
                packed[:, n//16:].copy_(right)
                scales[:n].copy_(sl)
                scales[n:].copy_(sr)
                ext.exl3_fat_gemm(a, packed, out, scales, 4, True, False)

        for _ in range(5):
            call()
        torch.cuda.synchronize()
        times = []
        for _ in range(15):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                call()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / 5)
        if not bool(torch.isfinite(out).all()):
            raise AssertionError(f'{label}: nonfinite projection output')
        results[label] = statistics.median(times)
        print(f'CASE {label} M={m} K={k} N={n} paired={paired} ms={results[label]:.6f}', flush=True)
    arm = 'paired' if paired else 'copies'
    print(f'METRIC {arm}_projection_panel_ms={sum(results.values()):.6f}')
    for name, value in results.items():
        print(f'METRIC {arm}_{name}_ms={value:.6f}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', choices=('candidate', 'upstream'), default='candidate',
                        help='candidate requires all new symbols/checks; explicit upstream '
                             'baseline requires only direct/scatter and rejects paired work')
    parser.add_argument('--panel', choices=('all', 'kernels', 'projection'), default='all',
                        help='fixed historical workload panel (default: all)')
    parser.add_argument('--projection-arm', choices=('both', 'copies', 'paired'), default='both',
                        help='projection arm; copies includes all four copies inside timing')
    parser.add_argument('--selfcheck-path', type=Path, default=None,
                        help='normal import of self-check file; default: installed /opt/glm53/'
                             'test_exl3_overlay.py if present, otherwise sibling test file')
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.validation == 'upstream' and args.panel != 'kernels'
            and args.projection_arm != 'copies'):
        parser.error('upstream mode requires --panel kernels or --projection-arm copies; '
                     'paired work is never silently omitted')
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f'MODE validation={args.validation} panel={args.panel} '
          f'projection_arm={args.projection_arm if args.panel != "kernels" else "not-requested"}',
          flush=True)
    global torch, ext
    import torch
    import exllamav3_ext as ext

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('Historical target requires a CUDA GB10 / SM121 device')
    print(f'DEVICE name={torch.cuda.get_device_name()!r} torch={torch.__version__} '
          f'cuda={torch.version.cuda} seed={SEED}', flush=True)
    print(f'SOURCE benchmark_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()} '
          f'extension={ext.__file__} '
          f'extension_sha256={hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest()}', flush=True)
    print('ENV ' + repr({k: v for k, v in sorted(os.environ.items())
                        if k.startswith(('EXL3_', 'GLM53_'))}), flush=True)
    correctness(args.selfcheck_path or default_selfcheck_path(), args.validation)
    if args.panel in ('all', 'kernels'):
        kernel_panel()
    if args.panel in ('all', 'projection'):
        if args.projection_arm in ('both', 'copies'):
            projection_panel(False)
        if args.projection_arm in ('both', 'paired'):
            projection_panel(True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
