#!/usr/bin/env python3
"""Install the additive EXL3 fat-GEMM source and pybind entries."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one binding anchor: {old!r}")
    return text.replace(old, new, 1)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_exl3_fat_kernel.py EXLLAMAV3_EXT SOURCE_DIR")
    ext_root = Path(sys.argv[1]).resolve()
    source_dir = Path(sys.argv[2]).resolve()
    quant = ext_root / "quant"
    bindings = ext_root / "bindings.cpp"
    if not quant.is_dir() or not bindings.is_file():
        raise RuntimeError(f"invalid extension root: {ext_root}")

    for name in ("exl3_fat_gemm.cu", "exl3_fat_gemm.cuh"):
        source = source_dir / name
        if not source.is_file():
            raise RuntimeError(f"missing additive source: {source}")
        shutil.copyfile(source, quant / name)

    text = bindings.read_text()
    text = replace_once(
        text,
        '#include "quant/exl3_moe.cuh"',
        '#include "quant/exl3_moe.cuh"\n#include "quant/exl3_fat_gemm.cuh"',
    )
    text = replace_once(
        text,
        '    m.def("exl3_moe", &exl3_moe, "exl3_moe");',
        '    m.def("exl3_moe", &exl3_moe, "exl3_moe");\n'
        '    m.def("exl3_fat_gemm", &exl3_fat_gemm, "exl3_fat_gemm");\n'
        '    m.def("exl3_fat_gemm_scatter", &exl3_fat_gemm_scatter, "exl3_fat_gemm_scatter");',
    )
    bindings.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
