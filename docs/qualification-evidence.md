# Exact-head qualification evidence (PR #182)

Candidate: `fe71e6d` (+ test-only `abc4ee4`, runtime-identical) on branch
`pr/sm121-exl3-kda-perf`, fast-forwarded from reviewed `f91fe07`
(which integrates main `b3bd1f6`). Image `glm53eval/pr182:fe71e6d`
(`sha256:80166afa...`), base
`vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293...`,
exllamav3 pin `c5d9c65`, vLLM `0.1.dev20051+g487ecf187`, torch
`2.13.0+cu130`, CUDA 13.0, 2x GB10 SM121 TP=2
(head `antonius`, worker `spark-8352`).
Model `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw@25a44fd`.
Both optimization defaults remain OFF. Earlier-head receipts are
historical and labeled as such below.

## 1. Standalone harness fix (Phase 1)

`tests/bench_fp8_fat.py` and `tests/test_kda_fp8_fat_gpu.py` died in
`process_weights_after_loading` (`AssertionError: tensor model parallel
group is not initialized`, reproduced with `_TP=None`). Both now run
inside `tests/_vllm_tp1.py::single_rank_model_parallel` — a real NCCL
process group + real `GroupCoordinators` via `init_distributed_environment`
/ `initialize_model_parallel` under a default `VllmConfig`, destroyed
exception-safely (leak = hard error). No mocks. `Tp1LifecycleTests`
pins the original failure, real TP=1 init, leak refusal, clean restore:
19/19 CPU tests pass in-image; the GPU harness passes end to end.

## 2. Native kernel qualification (exact-head image)

- EXL3 parity battery (maintainer-corrected fixtures: 32 experts,
  distinct top-8, correlated/uniform/skewed, shared+independent SUH,
  rows 1–128, streams, graphs with changed data+routing, fallback):
  **119/119 PASS**, fast-vs-stock rel RMSE ~1e-8.
- EXL3 layer latency (load clocks ~2.3 GHz): **+8 to +17%** fast vs stock
  across rows/routing.
- KDA GPU harness: all PASS (retention 49.2 MiB/layer, M<=64 all-Marlin,
  M>64 all-fat, eager-fallback bitwise, graph replay bitwise-exact on both
  branches, FAT=0 clean, bad-flag raises).
- Kernel crossover (true-Marlin control, production fat path with fused
  Triton quant): fat-prod wins **1.8x at M=65 → 6.2x at M=3584**;
  Marlin kept at M<=64. Both have identical rel RMSE vs BF16 (0.0266).
- `compute-sanitizer --tool memcheck`: KDA harness **17 PASS, 0 errors**;
  EXL3 battery (FAST=1) **119 cases, 0 errors**, tensors compare ALL PASS.
- Overlay GPU suite: 5 OK under both flag states.

## 3. SASS audit (host CUDA 13.0 toolkit, exact-head `.so`)

- Fast `glm53_exl3_moe_fast_kernel<4,256>` (both SUH variants):
  REG 127, STACK 16, LOCAL 0; SASS has **6 STL / 0 LDL** — write-only
  stack slots in wait-loop plumbing, **no spilled value ever reloaded**.
- Stock `exl3_moe_kernel<4,256>`: REG 128, STACK 88;
  SASS has **37 STL / 39 LDL** genuine spill traffic.
- Precise claim: zero local-memory reloads in the fast kernel
  (16 B frame) vs 39 reloads in stock (88 B frame). Occupancy class
  unchanged (127 vs 128 regs, same 1024 B smem).
- `nvdisasm`/`cuobjdump`/`compute-sanitizer` were available on the host
  toolkit; no system packages were installed.

## 4. Numerical admission (frozen `bench_kda_fp8.py` panel)

- Stock-vs-stock on exact-head FAILS the as-written gates (argmax 0.938
  structured A1/A2; 0.972 prose A1/A3): the gate tests near-tie noise.
  5 stock runs → 13-position fragile pool; all flips at top-1 <= 0.5
  (one 0.77 position that also flips within stock).
- THIN (3 B-runs vs 5 A-runs): B-runs disagree with each other at every
  "extra"; B3 has zero positions beyond the stock pool. No position shows
  B-consensus against A-consensus. No high-confidence divergence.
  Outcome: noise (C) + gate-measures-ties (D). No bisection target exists.
- FAT (3 F-runs vs 2 A-runs): F2/F3 agree with stock at every suspect
  except dense:52 (3v2, margins <= 0.12, F3 weakest at 0.315). Same verdict.
- Admission: CONDITIONAL PASS for both — no systematic cross-arm flips;
  all differences at tie/flat positions; kernel parity 1e-8 (thin) and
  identical BF16-relative error (fat). The frozen argmax>=0.99 gate cannot
  pass on this model for any numerics change, including stock-vs-stock.

## 5. Serving studies (TheGrill v0.2.0, CI artifact verified)

Collector `3cb208b6...`, source `4166e9f...`; workloads
`glm-routine-decode-v3` (`6d075fb8...`), prefill (`f0ddfda5...`),
mixed (`886c96c9...`); predeclared gates G1–G6 (see study plan).

- **THIN A/B/A2** (decode, FAT off, dense off): B vs A +4.8%..+8.4%
  all cells (structured-1 +8.3%); A2 vs A within ±2.4%. GATES PASS.
- **FAT A/B/A2** (decode, dense,kda both arms, FAST off): within ±1.4%;
  A2 within ±1.2%. GATES PASS (decode non-regression).
- **Prefill/mixed TheGrill workloads UNQUALIFIED** for this backend:
  they require `reported-prefix-zero` cache accounting; the server
  returns `prompt_tokens_details: null` (retained INVALID, never
  rerun weakened). Prefill claim rests on the qualified kernel table
  plus engineering cold-prefill A/B/A2 below.
- **FAT engineering** (dense,kda both arms): c1 tied 73.8/73.8;
  prefill 16k 1369→1615 **+18.0%**, 100k 1405→1668 **+18.7%**;
  mixed dec +5.6% / pre +10.0%.
- **COMBINED A/B/A2** (decode, dense,kda held both arms):
  B vs A +5.7%..+9.9% (structured-1 +9.5%, code +9.9%);
  A2 within ±2.1%. Engineering prefill: 16k +11.9%, 100k +20.0%.
- All captures 8/8 complete, `first_failure: null`, text admitted
  (thinking-off control); one early overlap (4 min kld during grill
  setup/warmup) disclosed; no captures replaced or rerun-to-pass.

## 6. Startup probe, memory, soak, scope

- M=65 probe: **0.54 ms/layer** standalone (~18 ms over 34 layers;
  unmeasurable in ~330 s load); probe tensors ~266 KB transient;
  failures log + retain nothing (CPU-tested).
- Controlled memory (worker rank, fresh boots, same config):
  model 79.65 GiB (FAT=0) vs **81.28 GiB (FAT=1): +1.63 GiB/rank**
  (== 34 x 49.2 MiB predicted); KV pinned **14.0 GiB both arms**;
  100k context + C6 verified usable. "Capacity unchanged" is replaced
  by these numbers.
- Bounded soak on combined-B (~25 min traffic + prior grill captures
  on the same config): C3/C6/prefill/mixed/text/cancel-probe clean,
  health 200 throughout, no restarts/OOM/CUDA errors. Two soak-probe
  artifacts of mine (default-thinking token budget; short-answer length
  assert) investigated and cleared — model outputs correct
  (37*41=1517 verified).
- Scope: homogeneous SM121/GB10 only. Fail-closed verified by
  construction: native per-device cap check (stock kernel otherwise),
  Python cap check + live M=65 probe (Marlin otherwise), no persisted
  probe cache (per-process verdict). Non-SM121 hardware untested here.

## 7. What is NOT claimed

- No `decide` PASS/REGRESSION envelope (descriptive + predeclared gates
  instead; prefill workloads backend-unqualified).
- No SASS claim beyond §3 numbers; no "zero spills" absolute
  (6 write-only STL exist).
- No combined headline percentage (cells never pooled).
- Historical (old-head) numbers in `sm121-perf-paths.md` stay labeled
  historical until rerun.
