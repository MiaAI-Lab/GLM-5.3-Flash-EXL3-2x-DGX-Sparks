#!/usr/bin/env python3
"""Install the overlay exl3.py (with the optional dense-FP8 Marlin path) and, when
GLM53_DENSE_FP8 is on, let the KDA and MLA constructors keep the quant config so
their projections reach Exl3Config.get_quant_method (idempotent, fail closed).

GLM53_DENSE_FP8=off (default): only the module file is refreshed (its new code
is unreachable: get_quant_method returns UnquantizedLinearMethod for every
LinearBase, exactly as before). Anything else: also patch kda.py / model.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SITE = Path(os.environ.get("GLM53_SITE", "/usr/local/lib/python3.12/dist-packages/vllm"))
OPT = Path(os.environ.get("GLM53_OPT", "/opt/glm53"))
MARK = "# [glm53-dense-fp8]"

KDA_OLD = """        saved_quant_config = vllm_config.quant_config
        vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config
"""
KDA_NEW = """        saved_quant_config = vllm_config.quant_config
        if getattr(saved_quant_config, "get_name", lambda: "")() != "exl3":  # [glm53-dense-fp8]
            vllm_config.quant_config = None
        super().__init__(config, vllm_config, prefix)
        vllm_config.quant_config = saved_quant_config
"""
MLA_OLD = """                quant_config=None,  # MLA projections are BF16 in checkpoint
                prefix=f"{prefix}.self_attn",
"""
MLA_NEW = """                quant_config=(quant_config if getattr(quant_config, "get_name", lambda: "")() == "exl3" else None),  # [glm53-dense-fp8]
                prefix=f"{prefix}.self_attn",
"""


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if MARK in text:
        print(f"{path.name}: {MARK} already present — skipping")
        return
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one {label} target, found {n}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched {path.name} ({label})")


def main() -> int:
    src = OPT / "exl3.py"
    dst = SITE / "model_executor/layers/quantization/exl3.py"
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    if not dst.is_file():
        raise SystemExit(f"missing {dst}")
    if dst.read_text() != src.read_text():
        dst.write_text(src.read_text())
        print(f"installed {src} -> {dst}")
    else:
        print(f"{dst.name}: already current")
    mode = os.environ.get("GLM53_DENSE_FP8", "off").strip().lower()
    if mode in ("", "off", "0", "no", "none"):
        print("GLM53_DENSE_FP8=off — constructors untouched")
        return 0
    kda = SITE / "models/glm5next/nvidia/kda.py"
    if not kda.is_file():
        kda = SITE / "model_executor/models/glm5next/nvidia/kda.py"
    model = kda.parent / "model.py"
    replace_once(kda, KDA_OLD, KDA_NEW, "kda quant_config")
    replace_once(model, MLA_OLD, MLA_NEW, "mla quant_config")
    print(f"dense fp8 groups: {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
