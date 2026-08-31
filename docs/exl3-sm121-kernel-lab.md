# Atlas EXL3 SM121 Kernel Lab — engineering log

Last updated: 2026-08-31

## Status

The first optimization stage is implemented but not yet timed on an isolated
SM121 device:

- reuse ExLlamaV3's packed ABI and direct CUDA kernels;
- accept arbitrary model/operator shapes rather than a model registry;
- validate MCG K1-K8 with independent materialized and streaming CPU oracles;
- enumerate ExLlamaV3's built-in tile shapes and SM quotas;
- emit native Atlas receipts for every exact measured tactic;
- keep the production ExLlamaV3 pin unchanged while enabling an isolated
  current-upstream comparison image.

No speedup is claimed until the checked-in sweep runs on SM121.

## Repository and architecture boundary

CUDA/runtime work belongs in this serving repository. Atlas owns profiling,
candidate decisions, receipt import, and promotion gates. This lab is the
bridge: it produces `atlas.kernel-benchmark/v1` measurements that Atlas can
rank without importing CUDA infrastructure.

The surrounding deployment remains GLM-specific, but the kernel-lab API is
not. `model_id`, `operator`, and `phase` are receipt provenance strings.
Compatibility is based on hardware/software identity, representation ABI,
M/N/K, tensor-parallel world size, grouped state, and backend commit.

The five unrelated untracked download/cutover scripts discovered in this
checkout were preserved and are not part of the kernel-lab branch.

## ABI decision

The lab reuses ExLlamaV3's serialized MCG ABI without modification:

| Field | Dtype | Shape |
|---|---|---|
| `trellis` | int16 | `[K/16, N/16, 16*bits]` |
| `suh` | fp16 | `[K]` |
| `svh` | fp16 | `[N]` |
| `mcg` | int32 | scalar marker `0xCBAC1FED` / `-877912083` |

The matrix convention is `A[M,K] @ W[K,N]`. Trellis tiles are 16×16 in
ExLlamaV3 tensor-core lane order. The direct MCG adapter now accepts every
integer upstream bitrate K1-K8. Fractional model-wide rates remain allocations
across integer-rate tensors, not an invented fractional tensor format.

Production remains pinned to ExLlamaV3
`c5d9c657966ffeeaa9353f0cc899f18629da4a13`. The Docker build argument can
select a different immutable upstream commit for an isolated experiment, and
the selected commit is embedded into benchmark receipts.

## Reference and direct paths

`kernel_lab/exl3/reference.py` implements the authoritative bit order,
tensor-core permutation, MCG procedural codebook, and 128-wide
Hadamard/scaling transforms in NumPy.

Two correctness paths are intentionally independent:

1. decode and materialize the complete weight, then multiply;
2. transform activations, decode one packed 16×16 tile, consume it immediately,
   and discard it.

The SM121 direct path calls ExLlamaV3 `exl3_gemm` with packed trellis tensors.
Full reconstruction is used only for separately timed correctness and
diagnostic baselines. It is outside the direct CUDA event range and receipts
set `full_precision_materialized=false` for the measured execution path.

## First SM121 tactic surface

The pinned extension already exposes controls that production normally hides:

- automatic dispatch/autotuning;
- four compiled GEMM tile shapes, filtered by exact K/N compatibility;
- an explicit number of participating SMs.

The lab now measures automatic dispatch plus each compatible tile at quarter,
half, and full-device SM quotas. This yields a hardware-derived dispatch table
instead of rules tied to a model name.

The current upstream ExLlamaV3 candidate at
`0c49587a7c235e6303a6bbedc8b665272ad3a2ea` additionally contains a QTIP-style
small-M FP16 GEMV and a fused mul1 int8-activation GEMV. Those are candidate
experiments, not silently adopted production code.

SM121 is consumer Blackwell. NVIDIA's current CuTe implementation directs
SM120/121 kernels to warp-level `mma.sync`, not datacenter-Blackwell
`tcgen05`. Therefore the first experiment stays on ExLlamaV3's CUDA
`mma.sync` path. CuTe/CUTLASS remains useful for layout and medium-M scheduling
once measurements demonstrate a gap.

## Workload input

The benchmark accepts any exact shape as `OPERATOR:K:N` or a JSON workload
list. No code change is needed for a future model. Defaults are generic
square/expand/contract shape classes. Early GLM-5.2 and Hy4 shapes remain only
under explicit `--fixture` names for reproducibility.

The default M sweep is `1,2,4,8,16,32,64,128`. The first performance gate
should focus on M=1/2/4/8; wider M values locate the direct-to-reconstruction
crossover.

## Evidence so far

### Measured — portable host

- Focused kernel-lab suite: 19 passed.
- K1-K8 deterministic pack/unpack: passed.
- K1-K8 streaming matmul versus materialized oracle: passed.
- Worst maximum absolute error across that sweep: `1.08e-6`.
- No dense tensor exists in the streaming oracle.

### Previously measured — GB10 compatibility

- NumPy K3 packing matched the official ExLlamaV3 CUDA packer byte-for-byte on
  deterministic and randomized fixtures.
- NumPy MCG decoding matched the CUDA decoder for all 65,536 states.

### Not measured

- Direct tactic latency on an isolated SM121.
- Effective bandwidth/TFLOP/s and reconstruction overhead on target shapes.
- Pinned 0.0.43 versus current-upstream small-M GEMV.
- mul1 int8-activation crossover.

The current development host exposes no NVIDIA CUDA device. Earlier available
Sparks were occupied by an unrelated two-node service, so no disruptive timing
run was attempted.

## Decision gate

Run the automatic path and tactic sweep on an isolated Spark, import the
generated catalog into Atlas, then choose the next patch from measured data:

- `reconstruction`: optimize bit extraction/register prefetch and K splitting;
- `memory_bandwidth_or_pipeline`: prioritize packed layout, coalescing, and
  multi-matrix fusion;
- `compute_or_launch`: reduce cooperative synchronization/launch work or move
  next to grouped MoE;
- direct loses to reconstruct-then-GEMM: investigate a medium-M tile-local
  decode/MMA kernel or move the crossover rather than forcing direct execution.

Only after that gate should a new CUDA schedule be promoted into the runtime.

## Upstream primary references

- ExLlamaV3 production pin:
  <https://github.com/turboderp-org/exllamav3/tree/c5d9c657966ffeeaa9353f0cc899f18629da4a13>
- ExLlamaV3 current small-M kernel:
  <https://github.com/turboderp-org/exllamav3/blob/0c49587a7c235e6303a6bbedc8b665272ad3a2ea/exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh>
- ExLlamaV3 current int8-activation kernel:
  <https://github.com/turboderp-org/exllamav3/blob/0c49587a7c235e6303a6bbedc8b665272ad3a2ea/exllamav3/exllamav3_ext/quant/exl3_gemv_int8_kernel.cuh>
- QTIP: <https://arxiv.org/abs/2406.11235>
- FLUTE: <https://arxiv.org/abs/2407.10960>
- CUTLASS SM121 GEMM:
  <https://github.com/NVIDIA/cutlass/blob/dc45f979ae336a235da1676b311f35efeb30149a/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu>
- CUTLASS consumer-Blackwell MMA guidance:
  <https://github.com/NVIDIA/cutlass/blob/dc45f979ae336a235da1676b311f35efeb30149a/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py>
- FlashInfer SM12x W4A16 MoE:
  <https://github.com/flashinfer-ai/flashinfer/blob/2cc51dcf67ee71aade7074c64e84f13b7b7b117b/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py>
