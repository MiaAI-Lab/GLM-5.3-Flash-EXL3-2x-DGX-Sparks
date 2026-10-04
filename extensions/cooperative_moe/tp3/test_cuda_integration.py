#!/usr/bin/env python3
"""Explicit maintenance-only GPU gate. Never downloads weights or starts serving.

Frozen screens: peak-relative <= .003, per-row peak-relative <= .05, relative
L2 <= .05, finite output, exact zero for empty-local rows. Graph replays must
be bitwise equal to eager candidate output, including changed inputs/routes.
These are screening tolerances, not a model-quality or speedup claim.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

if os.environ.get("GLM53_COOP_MAINTENANCE_TEST") != "1":
    raise RuntimeError("requires explicit maintenance test authorization")
os.environ.update(EXL3_FUSED_MOE="1", EXL3_FAT_GROUPED="1", EXL3_FAT_KERNEL="0",
                  EXL3_TEMP_ROWS_FUSED="64", EXL3_FAT_EXPERT_LOG="0")
sys.path.insert(0, os.environ.get("GLM53_COOP_TEST_HELPERS", "/opt/glm53"))
import torch
import test_exl3_overlay as helpers
from vllm.model_executor.layers.quantization import exl3

BUNDLE = Path(os.environ["GLM53_COOP_BUNDLE"])
spec = importlib.util.spec_from_file_location("glm53_coop_test_runtime", BUNDLE / "runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
if not getattr(exl3, "_glm53_coop_installed", False):
    runtime.install(exl3, str(BUNDLE), enabled=True)
stock = exl3._glm53_coop_original_apply
wrapper = exl3.apply_exl3_fused_moe
ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20, 21, 24, 25, 30, 32, 35, 40, 48, 56, 64)
PEAK_TOL, ROW_PEAK_TOL, REL_L2_TOL = .003, .05, .05
EP_RANK = int(os.environ.get("GLM53_COOP_EP_RANK", "1"))
if EP_RANK not in (0, 1, 2):
    raise ValueError("EP rank must be 0, 1, or 2")
EP_START = EP_RANK * 96
SANITIZER = os.environ.get("GLM53_COOP_SANITIZER") == "1"
if SANITIZER:
    ROWS = (1, 7, 32, 35, 64)
records = []


def compare(actual, ref, label):
    torch.cuda.synchronize()
    if not bool(torch.isfinite(actual).all() and torch.isfinite(ref).all()):
        raise AssertionError(f"nonfinite output: {label}")
    delta = (actual.float() - ref.float()).abs()
    metrics = {"peak_rel": float(delta.max() / ref.abs().max().clamp_min(1e-30)),
               "row_rel": float((delta.amax(1) / ref.abs().amax(1).clamp_min(1e-30)).max()),
               "rel_l2": float(delta.norm() / ref.float().norm().clamp_min(1e-30))}
    passed = metrics["peak_rel"] <= PEAK_TOL and metrics["row_rel"] <= ROW_PEAK_TOL and metrics["rel_l2"] <= REL_L2_TOL
    record = {"stage": "compare", "label": label, "pass": passed, **metrics}
    records.append(record)
    print(json.dumps(record), flush=True)
    if not passed:
        raise AssertionError(record)


def routes(rows, pattern):
    if pattern == "concentrated":
        ids = (torch.tensor([0, 1, 2, 3, 4, 5, 94, 95], device="cuda") + EP_START).repeat(rows, 1)
    else:
        ids = torch.stack([torch.randperm(96, device="cuda")[:8] + EP_START for _ in range(rows)])
    weights = torch.rand(rows, 8, device="cuda").softmax(-1)
    if pattern == "mixed":
        ids[0].fill_(((EP_RANK + 1) % 3) * 96)  # valid global expert, nonlocal on the selected EP rank
        if rows > 1:
            ids[1].fill_(-1)
        if rows > 2:
            ids[2].fill_(2**40)
        if rows > 3:
            weights[3].zero_()
    return ids, weights


def run_case(owner, rows, kind, pattern, limit=10.):
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16) * .1
    ids, weights = routes(rows, pattern)
    native = owner._glm53_coop_native
    calls = []
    original_launch = native.launch
    def counted(*args):
        calls.append(rows)
        return original_launch(*args)
    native.launch = counted
    def run(impl):
        exl3.apply_exl3_fused_moe = impl
        return exl3.apply_exl3_experts(x.float(), ids, weights, owner, fused=True, limit=limit)
    try:
        ref = run(stock)
        actual = run(wrapper).clone()
        torch.cuda.synchronize()
        selected = 1 <= rows <= 64 and limit == 10.0
        if bool(calls) != selected:
            raise AssertionError(f"wrong physical row selection {rows}: {calls}")
        label = f"{kind}/{pattern}/rows={rows}/limit={limit}"
        compare(actual, ref, label)
        if pattern == "mixed" and bool(torch.any(actual[:min(rows, 4)])):
            raise AssertionError(f"nonzero empty-local rows {label}")
        if not selected:
            if rows > 64 and pattern == "concentrated" and owner._exl3_last_fat_fallback != "grouped":
                raise AssertionError(f"E3 fallback missing: {owner._exl3_last_fat_fallback}")
            return
        # Warm up on a side stream; graph capture is ordered with all prior scratch use.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(wrapper)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(wrapper)
        for iteration in range(3):
            if iteration:
                x.copy_(torch.randn_like(x) * .1)
                new_ids, new_weights = routes(rows, "mixed" if iteration == 2 else "concentrated")
                ids.copy_(new_ids)
                weights.copy_(new_weights)
                ref = run(stock)
                actual = run(wrapper).clone()
                compare(actual, ref, label + f"/mutation={iteration}")
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(captured, actual):
                raise AssertionError(f"graph/eager mismatch {label} iteration={iteration}")
        print(json.dumps({"stage": "graph", "label": label, "replays": 3, "pass": True}), flush=True)
    finally:
        native.launch = original_launch
        exl3.apply_exl3_fused_moe = wrapper


def attach_map(layer):
    mapping = torch.full((288,), -1, dtype=torch.long, device="cuda")
    mapping[EP_START:EP_START + 96] = torch.arange(96, device="cuda")
    layer.expert_map = mapping
    return layer


def real_expert_layer(device, n_exp=96):
    """Layer built from existing checkpoint layer 3 and the selected 96-expert EP range.

    Uses the head node's HF cache when mounted; returns None otherwise.
    """
    import glob
    import json

    import torch
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod

    snaps = glob.glob("/root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/*/model.safetensors.index.json")
    if not snaps:
        return None
    try:
        from safetensors import safe_open
    except Exception:
        return None
    if len(snaps) != 1:
        raise RuntimeError(f"expected exactly one local snapshot, got {snaps}")
    index_path = snaps[0]
    root = index_path.rsplit("/", 1)[0]
    wmap = json.load(open(index_path))["weight_map"]
    prefix = "model.language_model.layers.3.mlp.experts"
    tensors = {}
    files = {}
    for e in range(n_exp):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suf in ("trellis", "suh", "svh", "mcg"):
                name = f"{prefix}.{e + EP_START}.{proj}.{suf}"
                files.setdefault(wmap[name], []).append(name)
    for fname, names in files.items():
        with safe_open(f"{root}/{fname}", framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name)
    g0 = tensors[f"{prefix}.{EP_START}.gate_proj.trellis"]
    hidden = int(g0.shape[0] * 16)
    inter = int(g0.shape[1] * 16)
    import types

    moe = types.SimpleNamespace(swiglu_limit=10.0)
    method = Exl3MoEMethod(moe, Exl3Config())
    layer = torch.nn.Module()
    method.create_weights(layer, num_experts=n_exp, hidden_size=hidden,
                          intermediate_size_per_partition=inter, params_dtype=torch.float16)
    with torch.no_grad():
        for e in range(n_exp):
            for si, proj in ((0, "gate_proj"), (1, "up_proj")):
                layer.w13_trellis[e, si].copy_(tensors[f"{prefix}.{e + EP_START}.{proj}.trellis"])
                layer.w13_suh[e, si].copy_(tensors[f"{prefix}.{e + EP_START}.{proj}.suh"])
                layer.w13_svh[e, si].copy_(tensors[f"{prefix}.{e + EP_START}.{proj}.svh"])
                layer.w13_mcg[e, si].copy_(tensors[f"{prefix}.{e + EP_START}.{proj}.mcg"].reshape(-1)[:1])
            layer.w2_trellis[e].copy_(tensors[f"{prefix}.{e + EP_START}.down_proj.trellis"])
            layer.w2_suh[e].copy_(tensors[f"{prefix}.{e + EP_START}.down_proj.suh"])
            layer.w2_svh[e].copy_(tensors[f"{prefix}.{e + EP_START}.down_proj.svh"])
            layer.w2_mcg[e].copy_(tensors[f"{prefix}.{e + EP_START}.down_proj.mcg"].reshape(-1)[:1])
    layer = layer.to(device)
    method.process_weights_after_loading(layer)
    return layer, hidden, inter


def main():
    torch.manual_seed(20260916)
    print(json.dumps({"stage": "tolerances_before_test", "peak": PEAK_TOL,
                      "row_peak": ROW_PEAK_TOL, "rel_l2": REL_L2_TOL,
                      "graph": "bitwise", "rows": ROWS, "ep_rank": EP_RANK}), flush=True)
    _, synthetic = helpers._tiny_layer(torch.device("cuda"), n_exp=96, hidden=4096, inter=2048)
    attach_map(synthetic)
    native = synthetic._glm53_coop_native
    print(json.dumps({"stage": "resources", "specializations": native.resources}), flush=True)
    # Verify the eager adapter itself makes no CUDA tensor allocations after prepare.
    probe_x = torch.randn(64, 4096, device="cuda", dtype=torch.float32)
    probe_ids, probe_weights = routes(64, "random")
    wrapper(probe_x, probe_ids, probe_weights, synthetic, synthetic._exl3_inners, synthetic.expert_map, 10.)
    torch.cuda.synchronize()
    before_alloc = torch.cuda.memory_stats()["allocation.all.allocated"]
    wrapper(probe_x, probe_ids, probe_weights, synthetic, synthetic._exl3_inners, synthetic.expert_map, 10.)
    torch.cuda.synchronize()
    after_alloc = torch.cuda.memory_stats()["allocation.all.allocated"]
    if after_alloc != before_alloc:
        raise AssertionError(f"candidate allocated {after_alloc - before_alloc} CUDA tensors in eager hot path")
    print(json.dumps({"stage": "allocation", "eager_cuda_tensor_allocations": 0, "pass": True}), flush=True)
    for rows in ROWS:
        for pattern in (("concentrated", "mixed") if SANITIZER else ("random", "concentrated", "mixed")):
            run_case(synthetic, rows, "synthetic", pattern)
    for rows in (65, 96):
        run_case(synthetic, rows, "synthetic", "concentrated")
    for limit in (0., .5):
        run_case(synthetic, 64, "synthetic", "random", limit)
    # This helper opens existing local safetensors only; missing data is a hard gate.
    real_result = real_expert_layer(torch.device("cuda"), n_exp=96)
    if real_result is None:
        raise RuntimeError("real-weight fixture unavailable: mount existing HF snapshot read-only")
    real, hidden, intermediate = real_result
    if (hidden, intermediate) != (4096, 2048):
        raise AssertionError(f"wrong real-weight shape {(hidden, intermediate)}")
    attach_map(real)
    if real._glm53_coop_native is not native:
        raise AssertionError("scratch was replicated per layer")
    for rows in ROWS:
        run_case(real, rows, "real96", "random")
        run_case(real, rows, "real96", "mixed")
    # Alternating layers and capacities exercises persistent counter ownership.
    for layer, rows in ((synthetic, 32), (real, 64), (synthetic, 1), (real, 35), (synthetic, 64)):
        run_case(layer, rows, "alternating", "concentrated")
    print(json.dumps({"stage": "complete", "pass": True, "comparisons": len(records),
                      "real_weight_experts": 96, "distributed_serving_verified": False,
                      "quality_eval_verified": False}), flush=True)


if __name__ == "__main__":
    main()
