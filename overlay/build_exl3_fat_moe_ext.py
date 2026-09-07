#!/usr/bin/env python3
"""Build the additive `exl3_fat_moe_ext` module (E3 grouped fat-expert kernels).

The kernel source is compiled inside a copy of the installed exllamav3_ext
source tree so its relative includes (../ptx.cuh, exl3_dq.cuh,
hadamard_inner.cuh, ...) resolve exactly as they would in a full
patch_exl3_fat_kernel.py image build. This is the layered-image path: it
does not recompile exllamav3_ext.

Flags: -O3 -lineinfo, sm_121a. Deliberately WITHOUT --use_fast_math (which
exllamav3's setup.py uses): the gate/up epilogue must reproduce torch.sigmoid's
full-precision expf/division to keep the E2 rounding boundaries.

Usage:
  build_exl3_fat_moe_ext.py --src <dir with exl3_fat_moe.cu/.cuh> --out <build dir>
                            [--install <site-packages dir>] [--arch 121a]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

EXT_DEFAULT = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
MODULE = "exl3_fat_moe_ext"
BINDING = """
#include <torch/extension.h>
#include "quant/exl3_fat_moe.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exl3_fat_moe_gather", &exl3_fat_moe_gather, "exl3_fat_moe_gather");
    m.def("exl3_fat_moe_gateup", &exl3_fat_moe_gateup, "exl3_fat_moe_gateup");
    m.def("exl3_fat_moe_down", &exl3_fat_moe_down, "exl3_fat_moe_down");
    m.def("exl3_fat_moe_tile_rows_gateup", &exl3_fat_moe_tile_rows_gateup, "exl3_fat_moe_tile_rows_gateup");
    m.def("exl3_fat_moe_tile_rows_down", &exl3_fat_moe_tile_rows_down, "exl3_fat_moe_tile_rows_down");
}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", default=EXT_DEFAULT)
    ap.add_argument("--install", default="")
    ap.add_argument("--arch", default="121a")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import torch  # noqa: F401  (libc10 must load before any ext)
    from torch.utils.cpp_extension import load

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tree = out / "ext"
    if tree.is_dir():
        shutil.rmtree(tree)
    shutil.copytree(args.ext, tree, ignore=shutil.ignore_patterns("*.so", "__pycache__"))
    for name in ("exl3_fat_moe.cu", "exl3_fat_moe.cuh"):
        source = src / name
        if not source.is_file():
            raise SystemExit(f"missing source: {source}")
        shutil.copyfile(source, tree / "quant" / name)
    binding = tree / f"{MODULE}_binding.cpp"
    binding.write_text(BINDING)
    cu13 = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
    for var in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
        os.environ[var] = cu13 + (":" + os.environ[var] if os.environ.get(var) else "")
    t0 = time.time()
    mod = load(
        name=MODULE,
        sources=[str(binding), str(tree / "quant" / "exl3_fat_moe.cu")],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            f"-gencode=arch=compute_{args.arch},code=sm_{args.arch}",
            "-Xcudafe", "--diag_suppress=177",
            "-Xcudafe", "--diag_suppress=20012",
        ],
        extra_include_paths=[str(tree)],
        build_directory=str(out),
        verbose=args.verbose,
    )
    print(f"built {MODULE} in {time.time() - t0:.0f}s: {mod.__file__}", flush=True)
    assert int(mod.exl3_fat_moe_tile_rows_gateup()) > 0
    so = Path(mod.__file__)
    if args.install:
        dest = Path(args.install) / so.name
        shutil.copyfile(so, dest)
        print(f"installed {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
