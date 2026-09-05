#!/usr/bin/env python3
"""Screen the installed kpool prefill loop's CUDA allocation peak and parity.

Run in the vLLM CUDA environment; --help needs only the Python standard library.
No model load, source patching, fake kernels, or saved tensor fixtures are used.
The installed vLLM loop (Apache-2.0, copyright vLLM contributors) is compiled
in memory only; no third-party source is vendored or written to disk.
See docs/exl3-prefill-validation.md for provenance and limitations.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace


CASES = ((1024, 32768), (1777, 75008), (769, 49152))
SEED = 20260719
REPEATS = 3
# Exact INSERT from overlay/patch_indexer_logits_lifetime.py.
# Used ONLY to remove the patch from the reference, never to modify the candidate.
INSERT = (
    "            # [glm53-release-prefill-logits] Top-k is the final consumer.\n"
    "            del logits\n\n"
)
MARKER = "[glm53-release-prefill-logits]"


def extract_loop(source: str) -> ast.For:
    tree = ast.parse(source)
    functions = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn_indexer_kpool"
    ]
    if len(functions) != 1:
        raise RuntimeError("Expected one sparse_attn_indexer_kpool source function")
    loops = [
        node for node in ast.walk(functions[0])
        if isinstance(node, ast.For)
        and any(
            isinstance(part, ast.Attribute)
            and part.attr == "chunks"
            and isinstance(part.value, ast.Name)
            and part.value.id == "prefill_metadata"
            for part in ast.walk(node.iter)
        )
    ]
    if len(loops) != 1:
        raise RuntimeError(f"Expected one prefill_metadata.chunks loop, got {len(loops)}")
    return loops[0]


def reference_source(source: str) -> tuple[str, bool]:
    if MARKER not in source:
        # An unpatched installed module benchmarks itself; do not insert the fix.
        return source, False
    if source.count(MARKER) != 1 or source.count(INSERT) != 1:
        raise RuntimeError("Lifetime INSERT drift: refusing an ambiguous reference")
    reference = source.replace(INSERT, "", 1)
    candidate_loop = copy.deepcopy(extract_loop(source))
    reference_loop = extract_loop(reference)

    class RemoveRelease(ast.NodeTransformer):
        count = 0

        def visit_Delete(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "logits"):
                self.count += 1
                return None
            return node

    stripper = RemoveRelease()
    stripped = stripper.visit(candidate_loop)
    if stripper.count != 1 or ast.dump(stripped) != ast.dump(reference_loop):
        raise RuntimeError("Reference must differ by only the inserted del logits")
    return reference, True


def compile_loop(loop: ast.For, module, inputs: dict, name: str):
    # The ONLY executable body is the original loop plus returning its output.
    # A copy of module globals retains the real helper objects and CUDA dispatch.
    function = ast.FunctionDef(
        name=name,
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                           kw_defaults=[], defaults=[]),
        body=[copy.deepcopy(loop), ast.Return(ast.Name("topk_indices_buffer", ast.Load()))],
        decorator_list=[],
    )
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    if ast.dump(function.body[0]) != ast.dump(loop):
        raise RuntimeError("Extracted loop AST changed")
    namespace = dict(vars(module))
    collisions = set(inputs).intersection(namespace)
    if collisions:
        raise RuntimeError(f"Input globals unexpectedly override module helpers: {collisions}")
    namespace.update(inputs)
    exec(compile(tree, f"{module.__file__}:{name}", "exec"), namespace)
    return namespace[name], namespace


def make_inputs(torch, module, m: int, n: int, device, case_id: int) -> dict:
    generator = torch.Generator(device=device).manual_seed(SEED + case_id)
    fp8 = module.current_platform.fp8_dtype()
    if fp8 != torch.float8_e4m3fn:
        raise RuntimeError(f"Expected CUDA e4m3fn FP8, got {fp8}")
    q = torch.randn((3 * m, 64, 128), device=device, generator=generator).clamp_(-8, 8).to(fp8)
    k = torch.randn((n, 128), device=device, generator=generator).clamp_(-8, 8).to(fp8)
    k_scale = torch.rand((n, 1), device=device, generator=generator, dtype=torch.float32) + 0.5
    weights = torch.rand((3 * m, 64), device=device, generator=generator, dtype=torch.float32) / (64 * 128**0.5)
    positions = torch.arange(4 * n - 3 * m, 4 * n, device=device, dtype=torch.int64)
    # Exclusive compressed causal end: only completed 4-token pools are visible.
    ends = ((positions + 1) // 4).to(torch.int32)
    chunks = [
        SimpleNamespace(
            total_seq_lens=n,
            skip_kv_gather=True,
            token_start=i * m,
            token_end=(i + 1) * m,
            cu_seqlen_ks=torch.zeros(m, dtype=torch.int32, device=device),
            cu_seqlen_ke=ends[i * m:(i + 1) * m],
        )
        for i in range(3)
    ]
    assert 0 < int(ends.min()) <= int(ends.max()) <= n
    return dict(
        q_quant=q, q_scale=None, k_quant_full=k, k_scale_full=k_scale,
        weights=weights, prefill_metadata=SimpleNamespace(chunks=chunks),
        positions=positions, index_kpool=4, topk_tokens=2048,
        topk_indices_buffer=torch.full((3 * m, 2051), -1, device=device, dtype=torch.int32),
        use_fp4_cache=False, short_prefill=False,
    )


def output_cpu(torch, function, buffer):
    # The enclosing production function initializes the buffer before its loop.
    buffer.fill_(-1)
    result = function()
    if result is not buffer:
        raise RuntimeError("Extracted function did not return the supplied buffer")
    # Explicit copy also prevents aliasing if this helper is exercised on CPU.
    return result.to(device="cpu", copy=True)


def recover_pools(torch, row, pool_count, kpool, begin, end, label):
    """Recover retained pool IDs from expansion's complete-pool prefix."""
    groups = row[:pool_count * kpool].reshape(pool_count, kpool).to(torch.int64)
    ids = groups[:, 0] // kpool
    reconstructed = ids[:, None] * kpool + torch.arange(kpool)[None, :]
    if not torch.equal(groups, reconstructed):
        print(f"DIAGNOSTIC {label} malformed_pool_groups={groups.tolist()}", flush=True)
        raise AssertionError(f"{label}: incomplete/malformed pool expansion")
    if not bool(((ids >= begin) & (ids < end)).all()) or len(set(ids.tolist())) != pool_count:
        print(f"DIAGNOSTIC {label} pool_ids={ids.tolist()} valid_range=[{begin},{end})", flush=True)
        raise AssertionError(f"{label}: invalid/duplicate pools or wrong cardinality")
    return ids


def diagnose_tied_row(torch, module, inputs, row_id, actual, expected, label):
    """Prove exact boundary ties with real MQA scores, outside measurement.

    The source selects 512 pools but expands only its first 511. Therefore we
    validate the boundary of the 511 *retained* pools, not an assumed 512-wide
    output ordering. Any non-boundary substitution remains a hard failure.
    """
    chunk = next(c for c in inputs["prefill_metadata"].chunks
                 if c.token_start <= row_id < c.token_end)
    offset = row_id - chunk.token_start
    ks = chunk.cu_seqlen_ks[offset:offset + 1]
    ke = chunk.cu_seqlen_ke[offset:offset + 1]
    begin, end = int(ks.item()), int(ke.item())
    kpool = inputs["index_kpool"]
    pool_count = inputs["topk_tokens"] // kpool - 1
    actual_ids = recover_pools(torch, actual, pool_count, kpool, begin, end, f"{label} actual")
    expected_ids = recover_pools(torch, expected, pool_count, kpool, begin, end, f"{label} reference")
    actual_set, expected_set = set(actual_ids.tolist()), set(expected_ids.tolist())
    added, removed = sorted(actual_set - expected_set), sorted(expected_set - actual_set)
    print(f"DIAGNOSTIC {label} row={row_id} pool_range=[{begin},{end}) "
          f"retained_pools={pool_count} actual_only={added} reference_only={removed}", flush=True)
    if not added or len(added) != len(removed):
        raise AssertionError(f"{label}: mismatch is not an equal-cardinality pool substitution")
    if not torch.equal(actual[pool_count * kpool:], expected[pool_count * kpool:]):
        print(f"DIAGNOSTIC {label} actual_tail={actual[pool_count * kpool:].tolist()} "
              f"reference_tail={expected[pool_count * kpool:].tolist()}", flush=True)
        raise AssertionError(f"{label}: tail or sentinel columns differ")

    # Reconstruct both expansions using the actual installed helper: do not
    # assume the details of its tail policy, padding, or duplicate-tail handling.
    device = inputs["q_quant"].device
    q_seq = inputs["positions"][row_id:row_id + 1].to(torch.int32) + 1
    for name, ids, row in (("actual", actual_ids, actual), ("reference", expected_ids, expected)):
        expanded = module.expand_pools_and_append_tail(ids[None, :].to(device), q_seq, kpool)
        rebuilt = torch.full_like(row, -1)
        rebuilt[:expanded.shape[-1]] = expanded.to(device="cpu", copy=True)[0]
        if not torch.equal(rebuilt, row):
            print(f"DIAGNOSTIC {label} {name} expansion mismatch", flush=True)
            raise AssertionError(f"{label}: output does not match real pool/tail expansion")
        del expanded

    # Recompute ONLY the selected Q row, with byte-identical FP8 Q/K, weights,
    # scales and original causal metadata. No alternate arithmetic or oracle.
    logits = module.fp8_fp4_mqa_logits(
        (inputs["q_quant"][row_id:row_id + 1], None),
        (inputs["k_quant_full"][:chunk.total_seq_lens],
         inputs["k_scale_full"][:chunk.total_seq_lens].view(torch.float32).squeeze(-1)),
        inputs["weights"][row_id:row_id + 1], ks, ke, clean_logits=True,
    )
    scores = logits.to(device="cpu", copy=True)[0]
    del logits
    valid_scores = scores[begin:end]
    if not bool(torch.isfinite(valid_scores).all()):
        print(f"DIAGNOSTIC {label} nonfinite_valid_scores={valid_scores.tolist()}", flush=True)
        raise AssertionError(f"{label}: nonfinite recomputed logits")
    if scores.dtype != torch.float32:
        raise AssertionError(f"{label}: expected FP32 real MQA logits, got {scores.dtype}")
    for name, ids in (("actual_only", added), ("reference_only", removed)):
        print(f"DIAGNOSTIC {label} {name}_scores=" + repr([
            {"pool": p, "value": float(scores[p]), "hex": float(scores[p]).hex(),
             "fp32_bits": f"0x{int(scores[p].view(torch.int32)) & 0xffffffff:08x}"}
            for p in ids
        ]), flush=True)
    # Exact equality, no epsilon or rounded printing comparison. Prove each
    # substituted score sits at the retained-set boundary: every selected pool
    # is >= it and every unselected valid pool is <= it, in BOTH outputs.
    boundary = scores[added[0]]
    equal_scores = bool((scores[added + removed] == boundary).all())
    boundary_valid = True
    for name, ids in (("actual", actual_ids), ("reference", expected_ids)):
        selected_scores = scores[ids]
        omitted = torch.ones(end - begin, dtype=torch.bool)
        omitted[ids - begin] = False
        min_selected = selected_scores.min()
        max_omitted = valid_scores[omitted].max()
        valid_boundary = bool((min_selected == boundary) & (max_omitted <= boundary))
        boundary_valid &= valid_boundary
        print(f"DIAGNOSTIC {label} {name} min_selected={float(min_selected)!r} "
              f"max_omitted={float(max_omitted)!r} boundary={float(boundary)!r} "
              f"valid_retained_boundary={valid_boundary}", flush=True)
    if not equal_scores or not boundary_valid:
        raise AssertionError(f"{label}: nonidentical scores or non-boundary pool substitution; "
                             f"equal_scores={equal_scores} boundary_valid={boundary_valid}")
    print(f"DIAGNOSTIC {label} row={row_id}: PROVEN exact-score retained-boundary tie; "
          "valid unique pools, equal cardinality, identical tail, real expansion parity", flush=True)


def compare_outputs(torch, actual, expected, positions_cpu, label: str, module, inputs) -> str:
    if (actual.shape != expected.shape or actual.dtype != torch.int32
            or expected.dtype != torch.int32 or actual.device.type != "cpu"
            or expected.device.type != "cpu"):
        raise AssertionError(f"{label}: output shape/dtype/device mismatch")
    if actual.data_ptr() == expected.data_ptr():
        raise AssertionError(f"{label}: actual and stored baseline alias")
    for name, output in (("actual", actual), ("reference", expected)):
        if bool(((output < -1) | (output > positions_cpu[:, None])).any()):
            raise AssertionError(f"{label}: {name} invalid or noncausal selected token")
        if not bool((output >= 0).any(dim=1).all()):
            raise AssertionError(f"{label}: {name} empty token selection")
    if torch.equal(actual, expected):
        mode = "exact"
    elif torch.equal(actual.sort(dim=1).values, expected.sort(dim=1).values):
        mode = "row-sorted (same token multiset; top-k ordering differs)"
    else:
        bad_rows = (actual.sort(dim=1).values != expected.sort(dim=1).values).any(dim=1)
        row_ids = bad_rows.nonzero().flatten().tolist()
        print(f"DIAGNOSTIC {label}: differing_rows={row_ids}", flush=True)
        device = inputs["q_quant"].device
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        for row_id in row_ids:
            diagnose_tied_row(torch, module, inputs, row_id, actual[row_id], expected[row_id], label)
        # All diagnostic device tensors were function-local and are gone before
        # the next loop baseline/reset. Do not carry diagnostic allocations into
        # any measured invocation (reserved allocator cache is not this metric).
        gc.collect()
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_allocated(device)
        if after != before:
            raise RuntimeError(f"{label}: diagnostic retained GPU allocations: {before} -> {after}")
        mode = f"proven exact-score boundary ties ({len(row_ids)} rows)"
    print(f"CORRECTNESS {label}: PASS {mode}; rows={actual.shape[0]} "
          f"selected_entries={int((actual >= 0).sum())}", flush=True)
    return mode


def measure(torch, function, buffer, device) -> tuple[int, float, int]:
    buffer.fill_(-1)
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    peak_delta = torch.cuda.max_memory_allocated(device) - baseline
    if result is not buffer or peak_delta <= 0:
        raise RuntimeError("Invalid output identity or nonpositive measured allocation peak")
    # The caller copies output to CPU only after this peak counter was read.
    return peak_delta, elapsed_ms, baseline


def run_case(torch, module, candidate_loop, ref_loop, device, case_id, m, n):
    inputs = make_inputs(torch, module, m, n, device, case_id)
    candidate, candidate_globals = compile_loop(candidate_loop, module, inputs, "measured_prefill_loop")
    reference, reference_globals = compile_loop(ref_loop, module, inputs, "reference_prefill_loop")
    buffer = inputs["topk_indices_buffer"]
    pos_cpu = inputs["positions"].cpu()
    label = f"M={m},N={n}"
    try:
        # Warm BOTH real paths, including JIT, allocator, top-k and expansion.
        expected = output_cpu(torch, reference, buffer)
        actual = output_cpu(torch, candidate, buffer)
        compare_outputs(torch, actual, expected, pos_cpu, f"{label} warmup", module, inputs)
        for fn in (candidate, reference):
            output_cpu(torch, fn, buffer)
        samples = {"candidate": [], "reference": []}
        for repeat in range(REPEATS):
            order = (("candidate", candidate), ("reference", reference))
            if repeat % 2:
                order = tuple(reversed(order))
            for name, fn in order:
                peak, elapsed, baseline = measure(torch, fn, buffer, device)
                samples[name].append((peak, elapsed))
                print(f"SAMPLE {label} path={name} repeat={repeat} "
                      f"baseline_bytes={baseline} peak_delta_bytes={peak} "
                      f"loop_ms={elapsed:.6f}", flush=True)
                compare_outputs(torch, buffer.to(device="cpu", copy=True), expected, pos_cpu,
                                f"{label} {name} repeat={repeat}", module, inputs)
        # Conservative peak across repeated calls, not an allocator/noise minimum.
        candidate_peak = max(x[0] for x in samples["candidate"])
        reference_peak = max(x[0] for x in samples["reference"])
        loop_ms = statistics.median(x[1] for x in samples["candidate"])
        ref_ms = statistics.median(x[1] for x in samples["reference"])
        print(f"CASE {label} candidate_peak_mb={candidate_peak / 2**20:.6f} "
              f"reference_peak_mb={reference_peak / 2**20:.6f} "
              f"saved_mb={(reference_peak - candidate_peak) / 2**20:.6f} "
              f"loop_ms={loop_ms:.6f} reference_loop_ms={ref_ms:.6f}", flush=True)
        return candidate_peak / 2**20, reference_peak / 2**20, loop_ms, ref_ms
    finally:
        # exec creates function/global reference cycles; do not retain previous
        # case GPU inputs in a cycle and inflate the next case's baseline.
        candidate_globals.clear()
        reference_globals.clear()
        inputs.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-patch', action='store_true',
                        help='fail if the installed source lacks the exact lifetime patch; '
                             'otherwise unpatched source is explicitly a self-comparison')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    import torch

    # Required import order: avoid GLM module/indexer circular initialization.
    importlib.import_module("vllm.models.glm5next.nvidia.model")
    module = importlib.import_module("vllm.model_executor.layers.sparse_attn_indexer_kpool")
    if not torch.cuda.is_available() or not module.current_platform.is_cuda():
        raise RuntimeError("This benchmark requires real NVIDIA CUDA and installed vLLM kernels")
    device = torch.device("cuda", torch.cuda.current_device())
    source = inspect.getsource(module)
    ref_source, patched = reference_source(source)
    if args.require_patch and not patched:
        raise RuntimeError('Installed source has no lifetime patch; refusing self-comparison')
    if not patched:
        print('WARNING unpatched installed source: candidate/reference are identical; '
              'this cannot establish a lifetime-patch benefit', flush=True)
    candidate_loop = extract_loop(source)
    ref_loop = extract_loop(ref_source)
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    loop_hash = hashlib.sha256(ast.dump(candidate_loop).encode()).hexdigest()
    print(f"SOURCE path={module.__file__} sha256={source_hash} "
          f"loop_ast_sha256={loop_hash} lifetime_patch_present={patched}", flush=True)
    print(f'BENCHMARK sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}', flush=True)
    print('ENV ' + repr({k: v for k, v in sorted(os.environ.items())
                        if k.startswith(('EXL3_', 'GLM53_'))
                        or k in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF',
                                 'MALLOC_ARENA_MAX', 'CUDA_VISIBLE_DEVICES')}), flush=True)
    print(f"DEVICE name={torch.cuda.get_device_name(device)!r} torch={torch.__version__} "
          f"cuda={torch.version.cuda} seed={SEED} repeats={REPEATS}", flush=True)
    print("SCOPE actual installed prefill chunk loop; staged FP8 cache; "
          "allocation-peak/correctness screen, NOT end-to-end performance. "
          "Units: *_mb are MiB (2**20 bytes); timing includes host loop and final CUDA sync.",
          flush=True)
    results = []
    with torch.inference_mode():
        for case_id, (m, n) in enumerate(CASES):
            results.append(run_case(torch, module, candidate_loop, ref_loop, device, case_id, m, n))
            gc.collect()
            torch.cuda.synchronize(device)
    print(f"METRIC indexer_peak_mb={sum(r[0] for r in results):.6f}")
    print(f"METRIC indexer_peak_max_mb={max(r[0] for r in results):.6f}")
    print(f"METRIC loop_ms={sum(r[2] for r in results):.6f}")
    print(f"METRIC reference_peak_mb={sum(r[1] for r in results):.6f}")
    print(f"METRIC reference_loop_ms={sum(r[3] for r in results):.6f}")
    print("CORRECTNESS all cases/repeats PASS", flush=True)


if __name__ == "__main__":
    main()  # Uncaught import, compilation, CUDA and parity exceptions exit nonzero.
