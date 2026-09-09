# E3 FP32 activation: validation and reproduction

## What changes

E3 already fuses gate/up GEMM, clamped SwiGLU, and down-input Hadamard.
PR #140's separate E2 activation helper is not called by this path. This E3
change instead replaces the epilogue's double-precision exponential with the
accurate FP32 libdevice entry `__nv_expf`, retaining explicitly rounded
sigmoid/multiplication operations and all FP16 storage boundaries.

This is not approximate `__expf`, does not route E3 through E2, and does not
change the default expert cap or any serving knob by itself. A subsequent
[default-startup correction](default-startup-validation.md) sets the prefill batch
to 5120 and qualifies startup with graph estimation enabled; the 7168/CG0 timing
below is historical evidence, not the current-default measurement.
The kernel change requires a native image
rebuild; mounting a Python overlay alone cannot install a CUDA kernel change.

The arithmetic targets Torch's FP32 sigmoid, not bit identity with the former
FP64-exp implementation. In an initial 1,048,576-input probe, five activation
FP16 results differed from the old implementation; all five matched Torch.
Fast-math's existing FP32 flush-to-zero behavior remains in effect.

## Correctness gates

Tested on both GB10 nodes, CUDA13, the repository's Python3.12 image layout:

- Actual-source boundary wrapper built with and without `--use_fast_math`:
  **582 strict checks passed per node**. Covers seeded FP32 values, every FP16
  gate bit pattern with representative up values, clamp/overflow neighborhoods,
  NaNs, infinities, signed zeros, subnormals, four activation limits, and
  nondefault-stream CUDA-graph replay with changed inputs.
- FP16 activation and post-SUH boundaries require exact non-NaN bits against
  Torch; NaN masks and signed zeros are checked explicitly. No FTZ exemptions
  are applied to these FP16 gates.
- Both complete native baseline/candidate images passed the existing full GPU
  integration suite on both nodes, including real-checkpoint and graph tests,
  with frozen tolerances and no real-checkpoint skips.

The boundary wrapper uses production helpers, but is **not** the full GEMM.
Its Hadamard comparison shares the native Hadamard implementation and therefore
cannot establish independent Hadamard accuracy. It also consumes native-clamped
inputs and does not establish Torch clamp parity. These limitations are why the
full-kernel integration suite remains mandatory.

With inference stopped, run in this repository's CUDA build environment:

```bash
python tests/test_exl3_e3_activation.py \
  --out /tmp/e3-boundaries.json --build-dir /tmp/e3-boundary-build
python tests/test_exl3_overlay.py
```

The first script targets `sm_121a` and the project's CUDA13/Python3.12 include
layout. It fails, rather than silently skipping, when CUDA is unavailable.
For real-checkpoint coverage, mount the existing checkpoint read-only at the
cache path expected by `tests/test_exl3_overlay.py` and verify that its report's
`real` section is populated. A skipped real-checkpoint test is not a pass.

## Preliminary synthetic timing — not serving throughput

TP2 per-rank geometry: hidden4096, intermediate1024, experts288, top8, cap32.
Same inputs and source toolchain, four alternating-order rounds and 40 timings
per variant/case; full MoE apply, not an isolated activation timing:

| 7,168 tokens, routing skew | Baseline ms | Candidate ms | Throughput change |
|---|---:|---:|---:|
| uniform | 32.920 | 31.039 | +6.06% |
| 1.0 | 32.658 | 31.028 | +5.25% |
| 1.5 | 35.555 | 33.696 | +5.52% |

At 1,024 tokens the fast-math candidate was flat to about 1.3% slower. These
synthetic routing distributions are not a live serving trace. Neither these
numbers nor the earlier E2 measurements establish an E3 serving improvement.

## Serving protocol

**Completed native E3 A/B/A/B:** size-ladder throughput geometric mean improved
**3.32%**, with paired improvements of **4.58%** and **2.08%**. All three holdouts
improved in both pairs. This is evidence from two boots per variant on one
cluster, not a universal speedup or a statistical confidence interval.

| Fixture | A1 tok/s | B1 tok/s | A2 tok/s | B2 tok/s | Combined change |
|---|---:|---:|---:|---:|---:|
| 4K | 1433.03 | 1445.69 | 1439.49 | 1448.52 | +0.76% |
| 16K | 1567.15 | 1646.23 | 1624.24 | 1658.08 | +3.55% |
| 64K | 1570.52 | 1695.03 | 1645.67 | 1704.15 | +5.72% |
| docs-short | 1431.25 | 1473.97 | 1481.01 | 1512.15 | +2.54% |
| code-medium | 1536.04 | 1583.66 | 1550.80 | 1588.68 | +2.77% |
| mixed-long | 1551.93 | 1612.45 | 1546.48 | 1623.34 | +4.43% |

Each cell is the median of three measured repetitions. Combined change is
`sqrt(B1 * B2 / (A1 * A2)) - 1`; the headline additionally takes the geometric
mean across the three size fixtures. A2 was faster than A1, particularly at 64K,
so retain both paired results rather than presenting the best pair alone.

All 100 fixture requests completed, including 28 warmup/discarded requests.
Token usage matched fixture-by-fixture across all arms, and every arm had zero
prefix-hit delta. Both ranks remained running with zero restarts and no OOMs;
rank logs contained no ERROR/Traceback. All four arms passed 22/25 canary checks
(including vision), failing only the same original-code prompt in three repetitions.
The runner therefore correctly exited 2 rather than claiming fully-green semantics.

Minimum sampled system MemAvailable was **1.89 GiB on the head** and 4.35 GiB on
the worker. The 1.5 GiB/three-sample guard did not trip, but headroom was limited;
this is not evidence for arbitrary concurrency or full 850K populated contexts.
Test ranks were stopped, worker staging restored, and observability left running.

Raw per-request rows, per-boot medians, image IDs, state summaries, and pairing
calculations are in [e3-activation-results.json](e3-activation-results.json).

Four fresh boots: upstream baseline A1, candidate B1, baseline A2, candidate B2.
Two GB10 nodes, TP2, E3/cap32, chunk7168, maxseq4, DFlash2 k7, vision/image100,
850000 context, GPU utilization0.85. Both arms use **`CG_ESTIMATE=0`**, retaining
CUDA graphs. Untouched defaults failed KV-budget initialization on both upstream
and candidate in this cluster; this is a disclosed compatibility override, not
an untouched-default pass or an 850K populated-context stress test.

Use identical installed dependencies, checkpoint/drafter bytes, fabric, clocks,
launch settings, and an idle server. Run strict semantic and vision canaries on
each boot, then the same 25 fixtures: one short warmup, three size ladders and
three document/code holdouts, each with one discarded and three measured runs.
Measure before/after prefix-hit counters and reject any nonzero delta. Compare
all token usage across arms. Report each boot separately; repetitions within a
boot are not independent deployment trials.

### Portable fixture replay

`tests/fixtures/e3-prefill.json.gz` contains the frozen fixtures (including salts).
The decompressed SHA256 must be:

```text
90b7f1d9fe2517f0aff3d82e4062b7d1fdbdf72fe95c68c15b61ae8f6ea321fd
```

Use a clean sparkDash checkout at
`e03b9d624e7135d6e82b4c8fc94ea0ddcf300547` with its dependencies installed.
The replay script uses its unmodified timing implementation, preserving the
measurement protocol while removing site-specific paths. It does not automate
launches, canaries, memory monitoring, or cleanup; those are separate required
parts of the surrounding campaign. The measured campaign
used the equivalent site-local harness; this portable wrapper is not itself a
separate measured serving trial.

```bash
SPARKDASH_ROOT=/path/to/sparkDash \
LLM_BASE_URL=http://127.0.0.1:8888 \
RESULT_DIR=/tmp/e3-a1 \
node tests/bench_e3_serving.mjs
```

Optional `VLLM_API_KEY` supplies bearer authentication. Restart the server before
each invocation: reusing these fixed salts on the same loaded server invalidates
the cold-prefill comparison. Keep output directories separate and preserve all
raw rows, including warmups and outliers.

## Known baseline caveat

The strict prompt `PR kernel validation Python: What does sum([3, 5, 7]) return?
Answer with only the integer.` returned `10` rather than expected `15` in all four E3 serving arms and prior
baseline, candidate, and speculation-off controls. The bare Python question and
equivalent addition returned `15`. Do not replace the failing prompt or weaken
its expected answer. This unresolved behavior prevents an unqualified claim that
all semantic canaries pass; it has not been isolated to this E3 change.
