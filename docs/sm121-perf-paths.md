# SM121 performance paths (EXL3 decode, KDA FP8 prefill)

Two independently guarded, opt-in optimizations for GB10 / SM121. Both
default off and fail closed to the stock paths.

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
  raises at model load instead of silently running stock.
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
  matching K, `M > 64` → fat path; otherwise Marlin. Per-capture-size
  CUDA graphs bake the branch taken at capture. `o_proj`/`f_b`/`g_b` and
  dense projections stay Marlin by measurement.
* Boundary evidence (SM121): Marlin wins every measured M ≤ 64; the fat
  path wins every measured M ≥ 65 (1.5x at 65, 2.7x at 220, 3.2–3.8x at
  1536–3683). Serving Ms split cleanly (decode ≤ 220, prefill ≥ 1536).
* Load-time retention (fail-closed): only for `kda` group layers with the
  measured `[12576, 4096]` geometry on SM121 with a working `_scaled_mm`;
  keeps the raw `[N,K]` e4m3 tensor plus fp32 per-channel scales (~51.6 MB
  per layer-rank, ~1.75 GB/rank over 34 KDA layers). The retained
  row-major transpose is exactly the col-major operand `_scaled_mm`
  requires — zero copy, no runtime repacking.
* Activation quantization: fused Triton rowwise kernel compiled once at
  load (bit-exact vs fp32 reference); eager torch fallback if Triton is
  unavailable. No new dependency.
* Measured (same image, flag-only switch): decode within ±1.5% of Marlin
  (M≤64 never leaves Marlin); cold prefill +20% at 16k and 100k; mixed
  decode +4.9%, mixed prefill +8.1%. Hybrid-vs-Marlin KL ≤ 5.4e-3 with
  flips only at low-confidence positions; generation canaries pass.
* Memory cost: ~1.75 GB/rank (~3.5 GB cluster) of retained FP8; KV
  capacity, max context, and concurrency unchanged in the tested
  configuration (KV stays pinned).
* Tests: `tests/test_kda_fp8_fat.py` (CPU: flag/shape/threshold/dispatch
  guards), `tests/test_kda_fp8_fat_gpu.py` (retention, dispatch spies,
  numerics, graphs), `tests/bench_kda_fp8.py` (KLD panel),
  `tests/bench_fp8_fat.py` (kernel crossover bench).

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
