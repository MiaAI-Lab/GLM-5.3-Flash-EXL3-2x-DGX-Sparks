# SM121 thin-decode fast path (EXL3 routed experts)

One opt-in optimization for GB10 / SM121, default off. An explicitly requested
EXL3 fast implementation must load successfully: requesting it on an image built
without the native kernels raises at model load instead of silently running the
stock kernel.

Speed numbers below are separated into *exact-head kernel evidence* and
*historical author-measured serving A/B*. The numerical qualification is a
separate, hash-frozen study with its own verdict and limits; it is cited, not
paraphrased.

## `GLM53_EXL3_MOE_FAST=1` — thin / small-M decode

Specializes the thin / small-M routed-expert kernel (`exl3_moe`) for K4 / N256
with a shallower register pipeline (1 stage instead of 3) and deeper
shared-memory staging (8 stages instead of 3). Stock geometry (K32 tiles, N256,
eight blocks per expert) is unchanged.

* Native change: `overlay/patch_exl3_decode_pipeline.py` adds two K4/N256
  kernels (shared vs independent gate/up input transform) to `exllamav3_ext` at
  image build time, plus a `glm53_fast_moe_version` symbol. The stock kernels
  are untouched; the patch refuses to apply twice and validates every source
  anchor before writing anything.
* Dispatch (native, per call): K == 4, N256-compatible dimensions
  (`hidden % 256 == intermediate % 256 == 0`), `GLM53_EXL3_MOE_FAST=1`, SM121.
  Transform reuse applies only when the gate/up SUH pointer tables are the
  identical allocation; `overlay/exl3.py` aliases them at load only when the
  flag is `1` and after an all-expert `torch.equal` proof on the packed scales
  (so the aliased table is content-identical). With the flag off, the pointer
  tables and the kernels are exactly what the stock path builds and runs.
* Fail-closed: requesting the flag on an image built without the patch raises at
  model load instead of silently running stock. The same failure surfaces when
  the fused `exl3_moe` path itself is unavailable (`EXL3_FUSED_MOE=0`, missing
  symbol, or any fused-state build error): `FAST=1` never degrades to the
  Python loop. The flag requires a literal `0` or `1`: the launcher rejects
  other values, including explicit empty and surrounding whitespace, before
  stopping services; load-time validation remains in place as well. `start.sh`
  forwards it to both ranks; the TP3 launcher keeps unsetting it.
* Scope: homogeneous SM121 / GB10 only. Non-SM121 hardware is untested here.

### Exact-head kernel evidence

* Compiler effect (SASS audit, host CUDA 13.0 toolkit on
  `glm53_exl3_moe_fast_kernel<4,256>` vs stock `exl3_moe_kernel<4,256>`):
  fast 127 regs / 16 B frame / 6 STL / 0 LDL (write-only stack slots, no spilled
  value ever reloaded) vs stock 128 regs / 88 B frame / 37 STL / 39 LDL genuine
  spill traffic. Occupancy class unchanged (same 1024 B smem).
* Layer latency: **+8% to +17%** fast vs stock across rows/routing (same
  hardware, load clocks ~2.3 GHz).
* Parity battery (`tests/test_exl3_thin_fast_gpu.py` +
  `tests/compare_thin_fast.py`, 32 experts, distinct top-8, correlated /
  uniform / skewed routing, shared + independent SUH, rows 1–128, non-default
  streams, CUDA-graph capture and replay with changed data and routing, and an
  N_off fallback geometry that must stay on the stock path): **119/119 PASS**,
  fast-vs-stock relative RMSE ~1e-8.
* `compute-sanitizer --tool memcheck` on the same battery with `FAST=1`:
  119 cases, 0 errors.

Provenance: every number in this section and in the historical serving section
below is an **author-reported** measurement recorded in the parent effort
(PR #182, branch head `4c0f422`) — SASS on the exact-head `.so`, the in-image
parity battery, and the earlier TheGrill A/B/A2 campaign. This branch does not
carry that qualification record and does not re-derive those numbers. The
current-`main` measurement section below is separate: it was measured here, on
this branch's own rebased image.

### Measured on current `main` (maintainer-measured at head `3d6ffbd`)

TheGrill v0.3.0, profile `glm-routine-decode-v3` (workload sha256
`6d075fb8…`), collector binary sha256 `2b48429f…`, image
`glm53-thin:3d6ffbd` = `sha256:f267534bbe52b792dbcaa3d1439bb4f5aaa02ccde7349eb7bc7eea5becac5116`
on upstream main `0437c09`, same image in all three arms, fresh two-node boot
per arm, `GLM53_DENSE_FP8=kda` and `GLM53_KDA_FP8_FAT=0` in every arm,
thinking-off, no tools, literal-loopback endpoint. Descriptive A/B/A2 — no
PASS envelope is claimed, and this says nothing about numerical quality.

Decode tokens/s (median) per cell, A (`FAST=0`) / B (`FAST=1`) / A2 (`FAST=0`):

| cell | A | B | A2 | B vs A | A2 vs A (noise) |
|---|---|---|---|---|---|
| structured-1 | 72.74 | 78.36 | 72.52 | **+7.71%** | −0.31% |
| code-1 | 67.83 | 73.86 | 68.04 | **+8.88%** | +0.30% |
| json-1 | 51.32 | 58.81 | 49.50 | **+14.59%** | −3.54% |
| structured-2 | 68.86 | 78.07 | 71.67 | withheld (raw +13.4%) | +4.07% |
| prose-1 | 31.93 | 34.91 | 33.40 | withheld (raw +9.3%) | +4.62% |

Accepted-with-resolved-ranges cells also improve on achieved completion
throughput (+7.93%, +9.13%, +14.36%) and wave latency (−7.34%, −8.37%,
−12.56%). Two cells are **withheld by the tool's own range-overlap rule**;
their raw medians are labelled raw and no change is claimed for them. Every arm
published 20 waves (15 measured, 15 eligible), with 0 invalid captures, 0
retries and 0 replacement runs, and every response reports `cached_tokens=0` on
the declared-cold legs.

Disclosed caveats: the server-side generation override caps every trial at 256
output tokens (identical in all arms, below the workload's 400-token cap); the
A2-vs-A noise control is what the B-vs-A deltas are read against; arm A (the
`FAST=0` baseline) was re-entered through the operator's reuse path after an
operator bug, with no request served before its capture; and the run
declarations recorded a placeholder image digest, with the image identity
independently evidenced by the arm containers, the image inspect and the
in-image gate — this collector does not attest declarations against the server.

### Historical serving evidence (earlier head, author-measured)

An independent [2026-09-22 operator field A/B](benchmarks/fast-moe-field-ab-20260922/README.md)
used a customized TP2 fair-prefill profile at recipe `ca85576`, not current
`main`. Its eight matched mixed-request repetitions and two bursts are additional
evidence, not a resolution of the slow-run reports in #227.

* TheGrill A/B/A2 decode, `FAST=1` vs stock, `GLM53_DENSE_FP8=off` in both
  arms: **+4.8% to +8.4%** across all cells (structured-1 +8.3%); the A2 stock
  repeat sat within **±2.4%**.
* These are historical measurements at the pre-rebase head. The current-`main`
  measurement above supersedes them for the rebased branch; do not quote this
  older pair as current.

### Numerical qualification (sealed study, cite by hash)

Prospective study `pr182q-20260917`, closed 2026-09-18, hash-frozen protocol
`protocol-numerical-FROZEN.md` sha256 `588c5393a0f4385a…` (full digest in that
study's `FREEZE.json`), calibration addendum `calibration-addendum.json` sha256
`c9539c22172d7c1d…` — m\*=1.5 nats (grid cap), stock consensus-flip-rate 95%
upper bound r_up=7.47e-4, tau_C=0.241 nats, classes CONF 5,783 / TIE 3,768 /
UNSTABLE 1,196; stock null shared-top-k KL mean 0.0128 / max 0.0131 (all 35
balanced splits 0.0127–0.0131).

Window 2 ran 8 stock + 6 thin captures on E material with interleaved boots:

* **T1 consensus flips: 0** (critical value 11 overall; 5 in the primary scope).
* **T2 reproducible shifts: 0** (max observed gap 0.054 nats vs bar 0.529).
* **T3b relative KL: pass every family** (means inside the stock null).
* **T4 tie-flip rate: F1 0.0887 < F0 0.0902** (p=0.42; A2 exhaustive p=0.477).
* **A2 exhaustive randomization** (all 3,003 balanced relabelings, per scope):
  T1 p=1.0, T4 p=0.48 — no capture-exchangeability violation.
* **Formal verdict: INCONCLUSIVE** via the predeclared control self-test only —
  the absolute T3a gate (KL ≤ 0.01) fails on stock-vs-stock itself (stock null
  0.0142–0.0146 > 0.01 on every split), and one bimodal long-context stock
  position pushed the tau-transfer bound (tau_E 0.5288 vs 0.500). Both control
  failures are instrument-calibration properties of the material, not candidate
  behaviour. This is not a pass and not a clearance.

Citation sentence: *"Under a prospective, hash-frozen protocol with
calibration-derived thresholds, the thin-decode optimization
(GLM53_EXL3_MOE_FAST) showed zero consensus flips (critical 11), zero
reproducible distribution shifts (tau=0.53 nats), tie-flip rate below stock's
own, and mean KL within the stock null in every family; the study's only failing
gate fails identically on unchanged stock repeats (absolute KL gate 0.01 < stock
null 0.0142–0.0146), so the formal outcome is inconclusive-with-clean-relative-
evidence rather than pass."*

Honest limits of that study: single deployment (2×GB10, TP=2, dflash-7,
dense=kda); teacher-forced positions, no multi-step decode; a formal PASS
requires a successor study id with the absolute gate re-derived and a
quantile-based tau transfer.

### What this change does not claim

* No pooled headline percentage and no PASS envelope: the serving comparison
  above is descriptive A/B/A2 only, read against its own A2-vs-A noise control.
* Client tool / structured-tool workloads were **not** validated on the surface
  that #215 (`tool_choice:none` decode guard) introduced; nothing here asserts
  behaviour for them.
* Composition with the cooperative-MoE path is untested.
* CPU-side contracts are covered by the tests below; the native kernels are
  only compiled and exercised in the image, so an image build with the patch is
  required before any GPU claim applies.

## Tests

* `tests/test_exl3_decode_pipeline.py` (CPU: patch anchors, double-apply and
  partial-write refusal, alias gate, version/`FAST=1` fail-closed paths).
* `tests/test_exl3_routing.py` (CPU routing fixtures: a token never routes to
  the same expert twice, which the thin-kernel row cap depends on).
* `tests/test_exl3_thin_fast_gpu.py` + `tests/compare_thin_fast.py` (GPU
  receipt battery and its CPU comparator; stock/candidate/stock order).
* `tests/bench_exl3_thin.py` (isolated dispatch A/B bench).
* `tests/test_launcher_rank_parity.py` (both ranks receive the same flag value;
  the launcher rejects anything but `0`/`1` before it stops a running pair).

```bash
GLM53_EXL3_MOE_FAST=0 python3 tests/test_exl3_thin_fast_gpu.py --out /tmp/thin_fast0.pt
GLM53_EXL3_MOE_FAST=0 python3 tests/test_exl3_thin_fast_gpu.py --out /tmp/thin_fast0b.pt
GLM53_EXL3_MOE_FAST=1 python3 tests/test_exl3_thin_fast_gpu.py --out /tmp/thin_fast1.pt
python3 tests/compare_thin_fast.py /tmp/thin_fast0.pt /tmp/thin_fast0b.pt /tmp/thin_fast1.pt
EXL3_TEMP_ROWS_FUSED=128 GLM53_EXL3_MOE_FAST=0 python3 tests/bench_exl3_thin.py
EXL3_TEMP_ROWS_FUSED=128 GLM53_EXL3_MOE_FAST=1 python3 tests/bench_exl3_thin.py
```

Nothing routed through this path is enabled by default: `GLM53_EXL3_MOE_FAST`
stays `0` unless an operator exports it.
