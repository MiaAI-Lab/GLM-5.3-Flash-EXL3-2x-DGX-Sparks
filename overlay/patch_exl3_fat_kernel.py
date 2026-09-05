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


def patch_bindings(text: str) -> str:
    include = '#include "quant/exl3_fat_gemm.cuh"'
    if text.count(include) > 1:
        raise RuntimeError("duplicate fat-GEMM include")
    if include not in text:
        text = replace_once(text, '#include "quant/exl3_moe.cuh"',
                            '#include "quant/exl3_moe.cuh"\n' + include)
    names = ("exl3_fat_gemm", "exl3_fat_gemm_pair", "exl3_fat_swiglu",
             "exl3_fat_gemm_scatter")
    missing = []
    for name in names:
        line = f'    m.def("{name}", &{name}, "{name}");'
        if text.count(line) > 1:
            raise RuntimeError(f"duplicate binding: {name}")
        if line not in text:
            missing.append(line)
    anchor = '    m.def("exl3_moe", &exl3_moe, "exl3_moe");'
    if text.count(anchor) != 1:
        raise RuntimeError("expected exactly one MoE binding anchor")
    if missing:
        text = replace_once(text, anchor, anchor + "\n" + "\n".join(missing))
    return text


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_exl3_fat_kernel.py EXLLAMAV3_EXT SOURCE_DIR")
    ext_root = Path(sys.argv[1]).resolve()
    source_dir = Path(sys.argv[2]).resolve()
    quant = ext_root / "quant"
    bindings = ext_root / "bindings.cpp"
    if not quant.is_dir() or not bindings.is_file():
        raise RuntimeError(f"invalid extension root: {ext_root}")

    text = patch_bindings(bindings.read_text())
    sources = [source_dir / name for name in ("exl3_fat_gemm.cu", "exl3_fat_gemm.cuh")]
    for source in sources:
        if not source.is_file():
            raise RuntimeError(f"missing additive source: {source}")
    for source in sources:
        shutil.copyfile(source, quant / source.name)
    bindings.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
