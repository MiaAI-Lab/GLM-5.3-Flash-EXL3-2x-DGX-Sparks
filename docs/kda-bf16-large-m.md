# KDA large-M BF16 prefill optimization

Status: implementation PR document. Opt-in via `GLM53_KDA_BF16_LARGE_M=1`;
default is `0` (stock behavior everywhere). Research provenance: PR #182.

## Motivation

FP8-Marlin W8A16 performs well at small M but is slow for the observed KDA
prefill geometry. On TP2/2xGB10, the KDA `in_proj` (TP2-local `[12576x4096]`)
runs decode at M=1..220 and prefill at M>=768, with the band 221-511 empty in
every measured step. A large-M GEMM for that projection recovers 11-14% of
cold/mixed prefill TTFT while leaving decode untouched.

## Architecture

```
                  ┌── M <= 512 ──> stock FP8-Marlin W8A16 (unchanged)
KDA in_proj ──────┤
                  └── M >  512 ──> F.linear(x_bf16, w_bf16), cuBLAS
```

`w_bf16` is a load-time BF16 copy of **the same logical FP8 weight** the stock
path already computes: the e4m3 tensor times the stored per-output-channel
scale (the BF16-rounded scale Marlin multiplies by). The pre-quantization
master is never restored, so FP8 weight-quantization semantics stay constant.
Activations stay BF16. Construction uses 512-row chunks (~8 MiB FP32 scratch,
no second full-size FP32 allocation); a tiny-M cuBLAS probe at load fails a
broken path during load, not on the first prefill. Dispatch reads only tensor
metadata (no host sync). The dispatch boundary is fixed at M > 512: it was
selected from the measured TP2 runtime distribution (decode stayed at
M<=220, the 221-511 band was empty in the measured workloads, prefill was
dominated by much larger M), it is stored on the layer at load so CUDA-graph
capture and replay cannot disagree, and it is intentionally not
user-configurable -- 512 is the boundary actually measured and qualified.

## Why not W8A8

PR #182 explored a large-M W8A8 path (rowwise BF16->FP8 activation
quantization x FP8 weight via `aten._scaled_mm`). It was fast but produced
reproducible full-model probability shifts at its M>64 boundary. Follow-up
attribution isolated the cause: activation QDQ alone explains the ~0.0266
layer error vs stock Marlin, while the `_scaled_mm` backend contributes only
~0.002-0.003. This PR removes the activation-quantization term instead of
fixing the backend. See #182 for the full research record; no W8A8 code is
carried here.

## Numerical evidence

Layer level, real layer-0 weights, rel RMSE vs Marlin (M>=768):

| reconstruction | vs Marlin |
|---|---|
| old W8A8 path | ~0.0266 |
| fp8 x fp32 scale | 0.002938 |
| **fp8 x stored bf16 scale (shipped)** | **0.002526** |
| Marlin itself vs an fp32-exact logical weight | 0.00005 |

The shipped copy is ~10.5x closer to stock than the W8A8 path, and Marlin is
essentially exact against the fp32-exact logical weight, so the residual is
the BF16 rounding of the copy. M=63/64/65/66 shows exact continuity (there is
no dispatch switch there; everything <=512 is Marlin).

Prospective full-model A/B/C study (stock / W8A8 positive control / BF16 v2,
15 frozen texts, 33,005 scored positions per capture): the W8A8 mechanism
reproduced at the old boundary (M=65: 7.99x stock-null mean KL, 23 coordinated
flips; M=66: 6.62x, 6 flips; a stock-reference token collapsed 0.778->0.0051),
while the BF16 path did not (M=65: 1.10x, 0 flips; M=66: 0.88x, 0 flips; the
same position stayed 0.768-0.782 alongside stock). The remaining BF16-vs-stock
differences sit inside measured stock/runtime variation. This is engineering
evidence, not an equivalence claim: no mathematical or bitwise equivalence is
asserted.

## Performance (TP2 / 2xGB10)

KDA `in_proj` large-M, median of 20 with A/B/A drift check:

* BF16 ~3.1-3.7x faster than Marlin for M>=768
* at M=7168: Marlin ~27.21 ms, old W8A8 ~12.57 ms, BF16 ~7.36 ms

Serving, same image, flag-only arms:

* cold ~16k TTFT: 11.41 s -> 10.12 s (-11.4%)
* cold ~100k TTFT: 70.86 s -> 62.12 s (-12.3%)
* mixed C3 + cold ~100k prefill TTFT: 76.21 s -> 65.43 s (-14.1%)
* decode: unchanged (literally the same Marlin code path)

Do not combine these percentages; each is a separate bounded measurement.

## Memory cost

* Retained copy: 12576 x 4096 x 2 bytes = 98.25 MiB/layer/rank, 34 KDA layers
* Theoretical +3.26 GiB/rank; measured model load 79.65 -> 82.94 GiB
  (**+3.29 GiB/rank**)
* KV reservation and the tested 850k context remain unchanged (883,552 KV
  tokens, 552 blocks), while idle headroom is reduced approximately
  7.7 GiB -> 4.7 GiB on this unified-memory platform.

## Validation

* `tests/test_kda_bf16_large_m.py`: flag validation, fixed-512 constant,
  chunk-invariant construction, stored-scale-vs-fp32 regression,
  fail-closed retention, the fixed M<=512 Marlin / M>512 BF16 dispatch table
  (including the old 63/64/65/66 boundary), counters.
* `tests/test_kda_bf16_large_m_gpu.py`: retention, dispatch counters,
  BF16-vs-Marlin numerics, graph capture/replay on both branches, fail-closed.
* `tests/bench_kda_bf16_large_m.py`: Marlin-vs-BF16 numerics and crossover
  timing receipt.
* Launcher: both-rank parity and pre-stop validation for the two new knobs.

## Scope and limitations

* Qualification scope: TP=2 homogeneous GB10/SM121. TP3 is intentionally out
  of scope: it has different KDA sharding/head-padding geometry and was not
  testable on this cluster. The loader fails closed on TP != 2 rather than
  silently running Marlin.
* Not claimed: mathematical/bitwise equivalence, universal quality
  preservation, zero memory cost, all-workload improvement, TP3 support, or
  formal numerical certification.
* The exact maintainer-only study texts were not available locally; the
  campaign reconstructed the mechanism prospectively with the W8A8 path as a
  positive control. Running the maintainer's exact cases against this draft is
  valuable independent follow-up evidence.

## Relationship to #182

#182 remains the historical research/evidence thread (W8A8 exploration,
attribution, A/B/C campaign). This PR contains only the BF16 large-M
implementation. Decode work from #217 is already separate and untouched here.
