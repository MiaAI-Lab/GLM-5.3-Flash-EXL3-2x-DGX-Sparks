#!/usr/bin/env python3
"""Experimental E3 activation boundary regression (not a full GEMM test).

Run in the CUDA13/exllamav3 build environment, with both serving models stopped:
  python tests/test_exl3_e3_activation.py --out results.json --build-dir /tmp/e3-test

Compiles the actual overlay CU, not a copied/extracted activation expression.
Requires exact FP16 bits (including signed zero); NaN payloads are ignored.
FTZ diagnostics never excuse failures. Hadamard coverage is explicitly limited:
we compare build modes and a Torch-activation-fed native epilogue, not an
independently qualified Torch shuffle-order oracle. No Hadamard accuracy claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback


WRAPPER = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include "quant/exl3_fat_moe.cu"

// One warp per 128-column row, same float4 lane layout as gateup.
__global__ void probe_kernel(const float* g, const float* u, const half* suh,
                             const half* supplied, float* cg, float* cu,
                             half* activation, half* scaled, half* final,
                             int64_t rows, float limit) {
    int lane = threadIdx.x;
    int64_t row = blockIdx.x;
    if (row >= rows) return;
    int64_t i = row * 128 + lane * 4;
    float a[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        // Deliberately reproduce existing CUDA clamps, including NaN suppression.
        float x = fminf(g[i+j], limit);
        float y = fminf(fmaxf(u[i+j], -limit), limit);
        cg[i+j] = x; cu[i+j] = y;
        a[j] = fm_swiglu_fp32(x, y);  // actual helper in included overlay
    }
    half4 h = supplied ? *reinterpret_cast<const half4*>(supplied + i)
        : half4(__floats2half2_rn(a[0], a[1]), __floats2half2_rn(a[2], a[3]));
    *reinterpret_cast<half4*>(activation+i) = h;
    half4 s = *reinterpret_cast<const half4*>(suh+i);
    h.x = __hmul2(h.x, s.x); h.y = __hmul2(h.y, s.y);
    *reinterpret_cast<half4*>(scaled+i) = h;
    float4 v = make_float4(__low2float(h.x), __high2float(h.x),
                           __low2float(h.y), __high2float(h.y));
    fm_had_row(v, lane);  // actual helper, NOT a reconstructed butterfly
    fm_store_half4(final+i, v);
}

std::vector<at::Tensor> probe(at::Tensor g, at::Tensor u, at::Tensor suh,
                             double limit, at::Tensor supplied) {
    TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kFloat &&
                g.is_contiguous() && g.dim() == 2 && g.size(1) == 128 &&
                g.size(0) > 0, "g must be nonempty contiguous CUDA float [rows,128]");
    TORCH_CHECK(u.device() == g.device() && u.scalar_type() == at::kFloat &&
                u.is_contiguous() && u.sizes() == g.sizes(), "invalid u");
    TORCH_CHECK(suh.device() == g.device() && suh.scalar_type() == at::kHalf &&
                suh.is_contiguous() && suh.sizes() == g.sizes(), "invalid suh");
    TORCH_CHECK(supplied.device() == g.device() && supplied.scalar_type() == at::kHalf &&
                supplied.is_contiguous() && (supplied.numel() == 0 ||
                supplied.sizes() == g.sizes()), "invalid supplied activation");
    c10::cuda::CUDAGuard guard(g.device());
    auto cg = at::empty_like(g), cu = at::empty_like(u);
    auto a = at::empty_like(suh), s = at::empty_like(suh), h = at::empty_like(suh);
    probe_kernel<<<g.size(0), 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        g.data_ptr<float>(), u.data_ptr<float>(), (half*)suh.data_ptr<at::Half>(),
        supplied.numel() ? (half*)supplied.data_ptr<at::Half>() : nullptr,
        cg.data_ptr<float>(), cu.data_ptr<float>(), (half*)a.data_ptr<at::Half>(),
        (half*)s.data_ptr<at::Half>(), (half*)h.data_ptr<at::Half>(), g.size(0), limit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {cg, cu, a, s, h};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("probe", &probe); }
'''


def build(args, root, report):
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parents[1] / "overlay"
    # Keep the installed package untouched. Each invocation owns its copied tree.
    tree = root / "ext"
    shutil.copytree(args.ext, tree, ignore=shutil.ignore_patterns("*.so", "__pycache__"))
    hashes = {}
    for name in ("exl3_fat_moe.cu", "exl3_fat_moe.cuh"):
        shutil.copyfile(source / name, tree / "quant" / name)
        hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
    wrapper = tree / "activation_probe.cu"
    wrapper.write_text(WRAPPER)
    hashes["wrapper"] = hashlib.sha256(WRAPPER.encode()).hexdigest()
    hashes["installed_hadamard_inner.cuh"] = hashlib.sha256(
        (tree / "quant" / "hadamard_inner.cuh").read_bytes()).hexdigest()
    report["source_sha256"] = hashes
    cu13 = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
    for var in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
        os.environ[var] = cu13 + (":" + os.environ[var] if os.environ.get(var) else "")
    os.environ.setdefault("MAX_JOBS", "1")
    flags = ["-O3", "-lineinfo", "-gencode=arch=compute_121a,code=sm_121a",
             "-Xcudafe", "--diag_suppress=177", "-Xcudafe", "--diag_suppress=20012"]
    modules = {}
    report["cuda_flags"] = {}
    for mode in ("precise", "fast"):
        out = root / mode
        out.mkdir()
        mode_flags = flags + (["--use_fast_math"] if mode == "fast" else [])
        report["cuda_flags"][mode] = mode_flags
        modules[mode] = load(name="e3_activation_probe_" + mode,
                             sources=[str(wrapper)], extra_cuda_cflags=mode_flags,
                             extra_include_paths=[str(tree)], build_directory=str(out),
                             verbose=args.verbose)
    return modules


def fixtures(torch):
    rng = torch.Generator().manual_seed(0xE3AC7)
    # Normal, uniform, and raw bit-pattern sampling cover very different exponents.
    n = 131072
    for name, g, u in (
        ("normal", torch.randn(n, generator=rng) * 12, torch.randn(n, generator=rng) * 12),
        ("uniform", torch.rand(n, generator=rng) * 220 - 110,
         torch.rand(n, generator=rng) * 200 - 100),
        ("fp32_bits", torch.randint(-(2**31), 2**31, (n,), generator=rng,
                                     dtype=torch.int32).view(torch.float32),
         torch.randint(-(2**31), 2**31, (n,), generator=rng,
                       dtype=torch.int32).view(torch.float32)),
    ):
        yield name, g, u
    all_half = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16).float()
    for up in (-10., -1.3162293434143066, -1., -0., 0., 2**-24, 1., 7.076262474060059, 10.):
        yield f"all_half_up_{up!r}", all_half, torch.full_like(all_half, up)
    special = torch.tensor([0., -0., float("inf"), -float("inf"), float("nan"),
                            2**-149, -2**-149, 2**-126, -2**-126,
                            2**-24, -2**-24, 65504., -65504.])
    yield "exception_cartesian", special.repeat_interleave(len(special)), special.repeat(len(special))
    pairs = torch.tensor([[-1.7404592037200928, -1.3162293434143066],
                          [-1.0463237762451172, 10.], [.2645307779312134, -10.],
                          [-15.316059112548828, 7.076262474060059],
                          [-7.231855869293213, -10.]])
    yield "rounding_sensitive", pairs[:, 0], pairs[:, 1]
    centers = torch.tensor([-104., -90., -88.72283935546875, -87.3365478515625,
                            -10., -7., -1., 0., 1., 7., 10., 88.72283935546875])
    values = [centers]
    lo, hi = centers, centers
    for _ in range(64):
        lo = torch.nextafter(lo, torch.full_like(lo, -float("inf")))
        hi = torch.nextafter(hi, torch.full_like(hi, float("inf")))
        values.extend((lo, hi))
    g = torch.cat(values)
    ups = torch.tensor([-10., -1., -0., 0., 1., 7.076262474060059, 10.])
    yield "clamp_exp_neighborhoods", g.repeat_interleave(len(ups)), ups.repeat(len(g))


def compare(torch, actual, expected):
    """Frozen predicate, valid for both FP16 and FP32 tensors; no tolerances."""
    integer = torch.int16 if actual.dtype == torch.float16 else torch.int32
    nan_a, nan_b = torch.isnan(actual), torch.isnan(expected)
    bits = actual.view(integer) != expected.view(integer)
    bad = (nan_a != nan_b) | (~nan_a & ~nan_b & bits)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    zeros = (actual == 0) & (expected == 0)
    idx = bad.flatten().nonzero().flatten()[:8]
    return {"elements": actual.numel(), "mismatches": int(bad.sum()),
            "finite_bit_mismatches": int((finite & bits).sum()),
            "nan_mask_mismatches": int((nan_a != nan_b).sum()),
            "signed_zero_mismatches": int((zeros & bits).sum()),
            "max_finite_abs_error": float((actual.float() - expected.float()).abs()[finite].max())
            if bool(finite.any()) else None,
            "examples": [{"index": int(i), "actual_bits": int(actual.view(integer).flatten()[i]),
                          "expected_bits": int(expected.view(integer).flatten()[i])} for i in idx]}


def run(torch, modules, report):
    empty = torch.empty(0, device="cuda", dtype=torch.float16)
    checks = report["checks"] = []
    report["hadamard_limitation"] = (
        "No independently qualified Torch shuffle-order reference: exact build-mode and "
        "Torch-activation-fed native epilogue comparisons only. These share fm_had_row and "
        "cannot detect a common Hadamard implementation bug. No arbitrary tolerance used.")
    report["scope"] = "Post-gate/up-Hadamard FP32 inputs only; not GEMM, SVH, or clamp parity with Torch."

    def record(name, actual, expected, diagnostic=False):
        result = compare(torch, actual, expected)
        result.update(name=name, diagnostic=diagnostic)
        checks.append(result)
        if result["mismatches"]:
            print(f"{'DIAGNOSTIC' if diagnostic else 'FAIL'} {name}: {result}", flush=True)

    scales = torch.tensor([1., -1., 0., -0., .5, 2., .333251953125, 2**-24,
                           2**-14, 65504., -2**-24], dtype=torch.float16)
    for name, gate, up in fixtures(torch):
        # Repeat padding rather than dropping any special or exhaustive pattern.
        size = ((gate.numel() + 127) // 128) * 128
        g = gate.repeat((size + gate.numel() - 1) // gate.numel())[:size].reshape(-1, 128).cuda()
        u = up.repeat((size + up.numel() - 1) // up.numel())[:size].reshape(-1, 128).cuda()
        suh = scales.repeat((size + len(scales) - 1) // len(scales))[:size].reshape_as(g).cuda()
        for limit in (0., 1., 7., 10.):
            outputs = {}
            for mode, mod in modules.items():
                out = mod.probe(g, u, suh, limit, empty)
                cg, cu, a, scaled, final = out
                oracle = (torch.sigmoid(cg) * cg * cu).half()
                oracle_scaled = (oracle * suh).half()
                prefix = f"{name}/limit{limit}/{mode}"
                record(prefix + "/activation", a, oracle)
                record(prefix + "/scaled", scaled, oracle_scaled)
                baseline = mod.probe(g, u, suh, limit, oracle)
                record(prefix + "/native_epilogue_from_torch_activation", final, baseline[4])
                # FP64 exponential is contextual evidence, NEVER the pass/fail oracle.
                fp64 = ((1. / (1. + torch.exp(-cg.double()))) * cg.double() * cu.double()).half()
                record(prefix + "/fp64_diagnostic", a, fp64, diagnostic=True)
                outputs[mode] = out
            for j, boundary in enumerate(("clamped_g", "clamped_u", "activation", "scaled", "hadamard")):
                # Global FTZ may alter subnormal FP32 clamp outputs even in
                # the unchanged kernel. Diagnose those inputs; the activation,
                # scaled, and Hadamard FP16 boundaries remain strict gates.
                record(f"{name}/limit{limit}/build_modes/{boundary}",
                       outputs["fast"][j], outputs["precise"][j], diagnostic=j < 2)
            # Explicitly quantify FTZ-prone strata without excluding them above.
            tiny = torch.finfo(torch.float32).tiny
            cg, cu = outputs["precise"][:2]
            ftz = ((cg.abs() < tiny) & (cg != 0)) | ((cu.abs() < tiny) & (cu != 0))
            ftz |= torch.isfinite(cg) & (cg < -87.)
            report.setdefault("ftz_diagnostics", []).append({
                "case": f"{name}/limit{limit}", "definition": "subnormal clamped input or finite gate < -87",
                "elements": int(ftz.sum()),
                "activation": compare(torch, outputs["fast"][2][ftz], outputs["precise"][2][ftz])})

    # Capture on a nondefault stream; mutate all input buffers before replay.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        g = torch.full((8, 128), -1., device="cuda")
        u = torch.ones_like(g)
        suh = torch.ones_like(g, dtype=torch.float16)
        for mode, mod in modules.items():
            for _ in range(3):
                mod.probe(g, u, suh, 10., empty)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = mod.probe(g, u, suh, 10., empty)
            for replay in range(3):
                g.fill_(-1.0463237762451172 + replay * .125)
                u.fill_(10. - replay)
                suh.fill_(-.5 if replay % 2 else 2.)
                graph.replay()
                eager = mod.probe(g, u, suh, 10., empty)
                for j, label in enumerate(("g", "u", "activation", "scaled", "hadamard")):
                    record(f"graph/{mode}/{replay}/{label}", captured[j], eager[j])
                oracle = (torch.sigmoid(captured[0]) * captured[0] * captured[1]).half()
                record(f"graph/{mode}/{replay}/oracle_activation", captured[2], oracle)
                record(f"graph/{mode}/{replay}/oracle_scaled", captured[3], (oracle * suh).half())
            del graph, captured
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    report["failed_checks"] = sum(c["mismatches"] != 0 and not c["diagnostic"] for c in checks)
    report["status"] = "fail" if report["failed_checks"] else "pass"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--build-dir", type=Path,
                        help="parent for a unique temporary copied source/build tree (removed on exit)")
    parser.add_argument("--ext", type=Path,
                        default=Path("/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    report = {"status": "error", "seed": 0xE3AC7,
              "predicate": "exact non-NaN bits; NaN masks equal; signed zero required; no FTZ exemptions"}
    code = 1
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for E3 activation regression (not skipped)")
        report.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
        if args.build_dir:
            args.build_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="e3-activation-", dir=args.build_dir) as tmp:
            modules = build(args, Path(tmp), report)
            run(torch, modules, report)
        code = 0 if report["status"] == "pass" else 1
    except Exception as exc:
        report.update(status="error", error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"E3 activation: {report['status']}; failed checks={report.get('failed_checks', 'n/a')}")
    return code


if __name__ == "__main__":
    sys.exit(main())
