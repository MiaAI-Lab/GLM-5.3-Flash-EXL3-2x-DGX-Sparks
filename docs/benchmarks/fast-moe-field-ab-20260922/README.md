# FAST MoE field A/B on two DGX Sparks — 2026-09-22

This is a descriptive operator measurement for the opt-in `GLM53_EXL3_MOE_FAST=1` path, following the slow-run report in [issue #227](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/227). Under one customized TP2 serving profile, eight length-matched mixed-request repetitions improved median whole-request decode by 9.8%. Both eight-request bursts improved aggregate throughput. No paired FAST repetition was slower in this window. **This small, sequential A/B does not rule out the slow-run mode reported in #227.**

## Fixed setup and method

- Two GB10 DGX Sparks, GLM-5.3-Flash-EXL3 TP2, runtime recipe commit `ca8557665bffa6529758f2c330ba8fb44c1e801a`, same local image digest `sha256:147531595e8c2e26e4c79e8735b44e875f58d0c244c196c312820d356cc8811a` on both ranks and both arms. This is **not** a measurement of current upstream `main` or a stock recipe.
- The installed scheduler hash was `5196a64da447ae591845db914e9452e14db59b72ae8243a450abbda5d11a571e` on both ranks and both arms; the fair-prefill overlay hash was `0870fab903a4e6a7323037834e5c4843d1abe1503b1b7b2dcc5f85de54a524df`. The full container environment was checked against the pretrial snapshot: between isolated arms only `GLM53_EXL3_MOE_FAST` changed from `0` to `1`. The public API was rebound to loopback for both arms.
- Context `262144`, `MAX_NUM_SEQS=8`, batch tokens `7168`, fair-prefill share `0.50`, adaptive draft lengths 2/4/7. The GLM weights, drafter, scheduler, image and request parameters stayed fixed. After three idle samples, the order was stock (`FAST=0`) then FAST (`FAST=1`), with a full restart and a five-case quality smoke check for each arm.
- Mixed test: eight repetitions per arm. A cold approximately 8.25K-token coding request generated 1024 tokens. Once it began generating, a separate cold approximately 16.7K-token retrieval request was submitted; overlap and retrieval correctness were checked. Distinct nonces varied prompt content, while prompt lengths matched within 0.5% for each arm pair. Whole-request decode is completion tokens divided by first-to-last visible content time; it includes scheduler interruptions and is **not** an isolated MoE-kernel rate.
- Burst test: two repetitions per arm, each with eight simultaneous cold approximately 8.28K-token prompts, 256 output tokens each. Aggregate rate includes prefill and completion time. The burst p95 is computed from just eight requests per repetition and is **not** a production p95 estimate.

## Results

| Metric | FAST=0 | FAST=1 | Change |
|---|---:|---:|---:|
| Mixed incumbent decode, median tok/s | 21.753 | 23.878 | +9.8% |
| Mixed newcomer TTFT, median s | 23.913 | 23.398 | −2.2% |
| Largest mixed content-event gap, s | 1.657 | 1.607 | −3.0% |
| Eight-request aggregate, tok/s | 21.326 | 24.060 | +12.8% |
| Eight-request median TTFT, s | 56.789 | 49.927 | −12.1% |
| Eight-request p95 TTFT, s | 84.427 | 74.159 | −12.2% |
| Eight-request completion time, s | 96.046 | 85.164 | −11.3% |

The per-repetition decode rates below are useful because a median alone can conceal the slow mode described in #227. Pairs have matched lengths, not identical token sequences.

| Repetition | FAST=0 tok/s | FAST=1 tok/s | Paired change |
|---:|---:|---:|---:|
| 1 | 20.48 | 25.10 | +22.5% |
| 2 | 22.41 | 23.59 | +5.3% |
| 3 | 21.31 | 24.21 | +13.6% |
| 4 | 22.86 | 23.40 | +2.4% |
| 5 | 22.53 | 24.30 | +7.9% |
| 6 | 21.36 | 22.57 | +5.7% |
| 7 | 22.00 | 22.98 | +4.4% |
| 8 | 21.51 | 24.16 | +12.3% |

Each arm passed bounded weather/reminder/add-tool, code and image/JSON smoke cases. The before/after vLLM counters showed no additional error, abort or preemption events in measured windows. Thermal checks were snapshots, not continuous profiling. The trial kept FAST because its prespecified gates required at least 5% median decode gain, no paired run more than 10% slower, no more than 5% TTFT or burst-makespan regression, matched token counts and clean quality/error checks. [decision.json](decision.json) contains all per-repetition numeric records, both burst repetitions and the gate outcomes; it contains no private addresses, credentials or user prompts.

The synthetic request runners are preserved at the pinned commit of [draft PR #246](https://github.com/beastllama/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/dc702b4fca1c1eda58f04555e570a3db5193a8af/docs/benchmarks/prefill-starvation-20260921) (`share_bench.py`, `lane_bench.py` and their local helpers). This PR does not include the private operator restart controller or the full event traces.

## Interpretation and limits

The result supports testing FAST on profiles like this one, especially where concurrent cold prefill interrupts decode. It does **not** establish a universal speedup: the schedule was stock-then-FAST without a closing stock arm, the prompts used distinct nonces, the two bursts are a small sample, and this customized fair-prefill profile differs from the direct single-stream profile in #227. The five-case quality suite is a smoke check, not a broad agent-quality or numerical-equivalence test. The private event traces remain with the operator; this PR publishes the numeric run summaries, not the full SSE capture.

The operator promoted FAST after the test. Both production ranks showed `FAST=1`, the public health endpoint returned HTTP 200, the same five quality cases passed, and the liveness timer was active. These post-restart checks establish the state at trial completion, not long-term production performance. No default launcher change is proposed here.
