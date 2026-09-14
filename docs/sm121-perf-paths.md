# SM121 performance paths (EXL3 decode, KDA FP8 prefill)

Two independently guarded, opt-in optimizations for GB10 / SM121. Both default
off. An explicitly requested EXL3 fast implementation must load successfully;
unsupported KDA-fat retention falls back to Marlin.

Performance, compiler and GPU-numerical results below are author-reported
measurements of an earlier candidate. CPU hardening does not requalify them.
The updated candidate still needs the latest completed TheGrill for native
builds, kernel/graph parity, serving correctness, latency and memory capacity.

## 1. EXL3 thin-decode fast path — `GLM53_EXL3_MOE_FAST=1`

Specializes the thin / small-M routed-expert kernel (`exl3_moe`) for K4 /
N256 with a shallower register pipeline (1 stage instead of 3) and deeper
shared-memory staging (8 stages instead of 3). Stock geometry (K32 tiles,
N256, eight blocks per expert) is unchanged.

* Native change: `overlay/patch_exl3_decode_pipeline.py` adds two K4/N256
  kernels (shared vs independent gate/up input transform) to `exllamav3_ext`
  at image build time, plus a `glm53_fast_moe_version` symbol. The stock
  kernels are untouched.
* Dispatch (native, per call): K == 4, N256-compatible dimensions
  (`hidden % 256 == intermediate % 256 == 0`), `GLM53_EXL3_MOE_FAST=1`,
  SM121. Transform reuse applies only when the gate/up SUH pointer tables
  are the identical allocation; `overlay/exl3.py` aliases them at load only
  after an all-expert `torch.equal` proof on the packed scales. Anything
  else keeps the stock kernel, including the E3 fat-prefill path.
* Fail-closed: requesting the flag on an image built without the patch
  raises at model load instead of silently running stock. The same
  failure surfaces when the fused `exl3_moe` path itself is unavailable
  (EXL3_FUSED_MOE=0, missing symbol, or any fused-state build error):
  FAST=1 never degrades to the Python loop. Both optimization flags require
  literal `0` or `1`: the launcher rejects other values, including explicit
  empty and surrounding whitespace, before stopping services. Load-time
  validation also remains in place.
* Compiler effect (SM121 `ptxas -v`): 128 regs / 84 B spill stores /
  188 B spill loads / 88 B stack → 127 regs / zero spills / 16 B stack.
* Measured (2x GB10 serving, same image, flag-only switch): C1 code
  +8.6%, C3 code +6.5–7.5%, C6 code +8.8%, mixed decode +11.5% at identical
  speculative acceptance; cold prefill unchanged. Numerical battery:
  fast-vs-stock rel RMSE ~4e-8 (repeat-noise level), graph replay clean.
* Tests: `tests/test_exl3_decode_pipeline.py` (CPU: patch anchors, alias
  and version gates), `tests/test_exl3_thin_fast_gpu.py` +
  `tests/compare_thin_fast.py` (parity, streams, graphs, fallback),
  `tests/bench_exl3_thin.py` (isolated dispatch bench).

## 2. KDA FP8 hybrid dispatch — `GLM53_KDA_FP8_FAT=1`

Requires `GLM53_DENSE_FP8=dense,kda`. Keeps FP8-Marlin for small-M decode
and routes large-M prefill of the KDA `in_proj` through torch native FP8
`_scaled_mm` on the same logical FP8 weights. No BF16 copies.

* Dispatch (`Glm53DenseFp8Method.apply`, no sync — M is tensor metadata):
  retained fat weights present (load-time proof below), no bias,
  matching K, flattened `M = numel/K > 64` → fat path; otherwise Marlin. Per-capture-size
  CUDA graphs bake the branch taken at capture. `o_proj`/`f_b`/`g_b` and
  dense projections stay Marlin by measurement.
* Boundary evidence (SM121): Marlin wins every measured M ≤ 64; the fat
  path wins every measured M ≥ 65 (1.5x at 65, 2.7x at 220, 3.2–3.8x at
  1536–3683). Serving Ms split cleanly (decode ≤ 220, prefill ≥ 1536).
* Load-time retention (fail-closed): only for `kda` group layers with the
  measured `[12576, 4096]` geometry on SM121 with a `_scaled_mm` proven
  by a real M=65 production-branch launch probe on each retained layer at
  load (has-operator is not working-kernel);
  keeps the raw `[N,K]` e4m3 tensor plus fp32 per-channel scales (~51.6 MB
  per layer-rank, ~1.75 GB/rank over 34 KDA layers). The retained
  row-major transpose is exactly the col-major operand `_scaled_mm`
  requires — zero copy, no runtime repacking.
* Activation quantization: the fused Triton kernel requires contiguous last
  dimensions. The eager fallback handles strided inputs and uses the same
  fp32 division and scale floor `max(row_amax / 448, 1e-12)`, including zero
  and tiny rows. CPU tensor-value checks cover the eager path. Updated GPU
  parity assertions require bit identity; that GPU check is still deferred.
* Measured (same image, flag-only switch): decode within ±1.5% of Marlin
  (M≤64 never leaves Marlin); cold prefill +20% at 16k and 100k; mixed
  decode +4.9%, mixed prefill +8.1%. Hybrid-vs-Marlin KL ≤ 5.4e-3 with
  flips only at low-confidence positions; generation canaries pass.
* Memory cost: ~1.75 GB/rank (~3.5 GB cluster) of retained FP8; KV
  capacity, max context, and concurrency unchanged in the tested
  configuration (KV stays pinned).
* Tests: `tests/test_kda_fp8_fat.py` (CPU dispatch/retention guards and actual
  CPU tensor values), `tests/test_kda_fp8_fat_gpu.py` (device retention,
  numerics and graphs), `tests/test_kda_logprob_compare.py` (CPU screening
  correctness), `tests/bench_kda_fp8.py` (shared-top-k screening panel),
  `tests/bench_fp8_fat.py` (kernel crossover bench).

The logprob panel now requires matching teacher-forced prompt fingerprints,
matching record/position coverage and usable distributions. Empty data,
generation-fallback receipts, malformed probabilities and missing support
return an unqualified result rather than a clean panel. The top-1 NLL delta
sign is corrected. Old receipts need fresh collection; passing shared-top-k
KL/argmax screening alone is not full numerical or model qualification.

## Serving flags

```bash
# stock behavior (defaults)
GLM53_EXL3_MOE_FAST=0
GLM53_KDA_FP8_FAT=0
# fully optimized path
GLM53_EXL3_MOE_FAST=1
GLM53_DENSE_FP8=dense,kda
GLM53_KDA_FP8_FAT=1
```

`start.sh` forwards both flags to head and worker. Rejected alternatives
on record: N128 tiles, smaller expert groups, K16 tiles, private-output
accumulation, CUTLASS/Inductor FP8 GEMMs on SM121 (no usable kernel in
this toolchain), and BF16 duplicate weights (unnecessary).

## Reproducing the serving numbers

Combined upstream-vs-candidate A/B/A2 campaign (2026-09-13, 2x GB10,
fresh builds from `upstream/main` vs this branch, all other settings
identical: TP=2, E3 grouped, adaptive-k ema, KV fp8 pinned 14 GiB):

Baseline A/A2: upstream image, `GLM53_DENSE_FP8=off`, both new flags 0.
Candidate B: this branch, `FAST=1` + `dense,kda` + `FAT=1`.

| case | A (upstream) | B (candidate) | A2 (upstream) | B vs A |
| --- | --- | --- | --- | --- |
| C1 code tok/s | 67.2 | 81.2 | 66.3 | **+20.8%** |
| C3 code agg | 155.9 / 151.0 | 184.2 / 184.3 | — | **+20.6%** |
| C6 code agg | 307.0 / 306.5 | 361.3 / 363.0 | — | **+18.0%** |
| C1 prose tok/s | 28.5 (1.94) | 35.5 (2.07) | — | accept-driven |
| cold prefill 16k | 1497.2 | 1492.6 / 1634.0 | — | volatile, -0 to +9% |
| cold prefill 100k | 1611.3 | 1659.1 | 1606.5 | **+3.0%** |
| mixed dec / pre | 56.8 / 662.8 | 76.2 / 794.6 | 61.9 / 695.3 | **+34% / +20%** |

Acceptance 7.0 on all code runs; coherence true both arms; KLD
candidate-vs-upstream within the validated envelope (flips only at
p<0.44 low-confidence positions). The 16k prefill sample is noisy
(single ~12 s runs); 100k replicated. Decode gains combine the EXL3
(+8.6%) and kda-FP8 (+10.3%) effects measured in the isolated
campaigns — the combined number above is measured, not added.

```bash
GLM53_BENCH_BASE=http://127.0.0.1:8000 python3 tests/bench_decode.py \
  --phase c1-code --structured --runs 3 --max-tokens 200 --out c1.json
GLM53_BENCH_BASE=... python3 tests/bench_decode_conc.py \
  --streams 3 --max-tokens 200 --prompt structured --runs 2 --out c3.json
GLM53_BENCH_BASE=... python3 tests/_run_cold_prefill.py \
  --only '~16k' --out prefill.json
python3 tests/bench_kda_fp8.py capture --out kld.json
python3 tests/bench_kda_fp8.py compare a.json b.json
```
