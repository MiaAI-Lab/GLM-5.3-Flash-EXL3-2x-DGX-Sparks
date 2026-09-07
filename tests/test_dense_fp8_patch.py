#!/usr/bin/env python3
"""overlay/patch_dense_fp8.py on copies of kda.py / model.py, plus the allow-list classifier."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = ROOT / "overlay" / "patch_dense_fp8.py"
SITE = Path("/usr/local/lib/python3.12/dist-packages/vllm")
KDA_SRC = Path(os.environ.get("GLM53_KDA_PY_SRC", SITE / "models/glm5next/nvidia/kda.py"))
MODEL_SRC = Path(os.environ.get("GLM53_GLM5_MODEL_PY_SRC", SITE / "models/glm5next/nvidia/model.py"))


def classifier_tests() -> None:
    text = (ROOT / "overlay" / "exl3.py").read_text()
    start = text.index("_GLM53_DENSE_FP8_SUFFIXES = {")
    end = text.index("class Glm53DenseFp8Method(")
    ns = {"os": os, "re": __import__("re")}
    exec(text[start:end], ns)
    f = ns["_glm53_dense_fp8_group"]
    lt = ["linear_attention"] * 3 + ["deepseek_sparse_attention"] + ["linear_attention"] * 41
    all_g = {"shared", "dense", "kda", "mla"}
    assert f("model.layers.0.mlp.gate_up_proj", set(), lt) is None, "off"
    assert f("model.layers.0.mlp.gate_up_proj", all_g, lt) == "dense"
    assert f("model.layers.5.mlp.shared_experts.gate_up_proj", all_g, lt) == "shared"
    assert f("model.layers.5.mlp.shared_experts.down_proj", {"dense"}, lt) is None
    assert f("model.layers.5.mlp.experts", all_g, lt) is None
    assert f("model.layers.0.self_attn.in_proj_qkvbfg_a", all_g, lt) == "kda"
    assert f("model.layers.0.self_attn.q_conv1d", all_g, lt) is None
    assert f("model.layers.0.self_attn.o_proj", all_g, lt) == "kda"
    assert f("model.layers.3.self_attn.o_proj", all_g, lt) == "mla"
    assert f("model.layers.3.self_attn.o_proj", {"kda"}, lt) is None
    assert f("model.layers.3.self_attn.kv_b_proj", all_g, lt) is None
    assert f("model.layers.3.self_attn.fused_qkv_a_proj", {"mla"}, lt) == "mla"
    assert f("model.layers.45.mtp_block.self_attn.o_proj", all_g, lt) is None
    assert f("model.layers.0.self_attn.o_proj", all_g, None) is None, "no layer types -> no attention groups"
    assert f("draft_model.layers.0.mlp.gate_up_proj", all_g, lt) is None


def main() -> int:
    classifier_tests()
    for src in (KDA_SRC, MODEL_SRC):
        if not src.is_file():
            raise SystemExit(f"missing {src}")
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp) / "site"
        (site / "models/glm5next/nvidia").mkdir(parents=True)
        (site / "model_executor/layers/quantization").mkdir(parents=True)
        shutil.copyfile(KDA_SRC, site / "models/glm5next/nvidia/kda.py")
        shutil.copyfile(MODEL_SRC, site / "models/glm5next/nvidia/model.py")
        (site / "model_executor/layers/quantization/exl3.py").write_text("stale\n")
        opt = Path(tmp) / "opt"; opt.mkdir()
        shutil.copyfile(ROOT / "overlay" / "exl3.py", opt / "exl3.py")
        env = os.environ.copy(); env["GLM53_SITE"] = str(site); env["GLM53_OPT"] = str(opt)
        env["GLM53_DENSE_FP8"] = "off"
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        assert (site / "model_executor/layers/quantization/exl3.py").read_text() == (ROOT / "overlay" / "exl3.py").read_text()
        assert "[glm53-dense-fp8]" not in (site / "models/glm5next/nvidia/kda.py").read_text(), "off leaves constructors alone"
        env["GLM53_DENSE_FP8"] = "shared,kda"
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        kt = (site / "models/glm5next/nvidia/kda.py").read_text(); mt = (site / "models/glm5next/nvidia/model.py").read_text()
        assert kt.count("[glm53-dense-fp8]") == 1 and mt.count("[glm53-dense-fp8]") == 1
        compile(kt, "kda.py", "exec"); compile(mt, "model.py", "exec")
        subprocess.check_call([sys.executable, str(PATCH)], env=env)  # idempotent
        assert (site / "models/glm5next/nvidia/kda.py").read_text() == kt
    print("dense-fp8 patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
