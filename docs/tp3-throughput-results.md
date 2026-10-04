# TP3 cooperative / combined-profile evaluation

This is a contribution for maintainer evaluation, with separable kernel,
FlashKDA, integration, and evidence commits. No default TP2 behavior is changed.
The measurements below belong to an earlier customized TP3 recipe. They are
**not measurements of this newly assembled upstream-based branch**, nor
isolated cooperative-kernel gains. This branch starts at upstream `6961fa0`.

## Components

- `extensions/cooperative_moe/tp3/`: local-I2048 / EP96 cooperative adaptation,
  32/64-row ABI, collision-free route ordering, per-device shared scratch,
  graph-safe staging, manifest verification and measured per-row dispatch.
- `overlay/patch_flashkda_tp3.py`: optional prefill patch for the pinned
  14-argument FlashKDA ABI and FP32 recurrent state. Default disabled.
- `overlay/ablit_runtime.py`: optional transplant donor end-padding compatibility
  for TP3; TP2 shape rejection remains unchanged. No donor files are included.
- `examples/tp3-throughput.env`: optional eight-request profile, all 25 graph
  rows for adaptive DFlash 2/4/7, shared/MLA FP8, InstantTensor and fair-prefill
  settings. Requires a separately prepared, qualified cooperative bundle.
- The launcher stages the complete ABI2 manifest bundle on every rank. The
  existing two-file TP2 adapter path is retained. Missing ABI2 manifests are
  rejected during preflight, before a restart can stop an existing engine. Latest upstream SWA retention
  support is preserved; selecting a different retention policy is a separate
  behavioral change.

## Historical environment and methodology

Three GB10 devices, TP=3/EP=3, 96 experts per rank, 66 padded attention heads;
CUDA 13 SM121a, vLLM `487ecf187`, EXL3 K4 MCG, FP8 KV. Target:
the EXL3 4bpw checkpoint used for that run.
Drafter: `incoai/GLM-5.3-Flash-DFlash2@9a5c86e3b48179cfdb6e5a7d1ed701b00a9c8fa5`,
k=7, adaptive 2/4/7, draft TP=1, TP3 padding 36/9. Quant and drafter were
unchanged between measured arms. Existing donor edits at layers 15–45,
alpha 3 including MTP, were held fixed. No donor data is included here.

The baseline was a customized `a35eaab` deployment, not stock upstream main.
Both arms used 1M request context, 32 GiB KV/rank (3,005,093 logical shared
KV tokens), 4096-token chunks, long-prefill threshold 2048, and global
retention 0 without explicit SWA override. The combined arm changed multiple
components: cooperative adapter, FlashKDA, FP8 scope dense/kda ->
dense/kda/shared/mla, four -> eight active requests, 32 -> 64 fused rows,
expanded graph coverage, InstantTensor 0.2.0 and fair share .20 -> .30 with
maximum step 1000 -> 2000 ms. Do not attribute the bundle's gain to one item.

### Matched C1 core fixtures

Five repetitions per arm and prompt, alternating prompt order, temperature 0,
thinking off, forced 512 output tokens. Metric is
`(completion_tokens - 1)/(last visible SSE chunk - first visible SSE chunk)`;
this is a chunk-timing approximation, not token interarrival time. All ten
corresponding sample pairs improved. No closing baseline rerun was performed.

| Synthetic workload | Baseline median tok/s | Combined median tok/s | Change |
|---|---:|---:|---:|
| Python async queue, retries/logging/cancellation/tests | 41.4575 | 48.8793 | +17.90% |
| Essay about bandwidth, arithmetic intensity and batching | 32.1122 | 38.8015 | +20.83% |

The code-generation fixture times generation; it does not execute or grade
its output. Sample-level numeric records: [matched-c1.json](benchmarks/tp3-combined/matched-c1.json).

### Historical same-script comparisons

These comparisons have differing nonces and are directional, not fresh
matched A/B/A. C4/C8 are medians of three batch-level end-to-end aggregate
rates. C8 includes increasing active slots from four to eight.

| Workload | Prior | Combined | Change |
|---|---:|---:|---:|
| LRU coding, tok/s | 58.8836 | 68.4341 | +16.22% |
| History essay, tok/s | 32.4325 | 37.5403 | +15.75% |
| C4 aggregate, tok/s | 75.7799 | 77.4341 | +2.18% |
| C8 aggregate, tok/s | 76.4132 | 123.0554 | +61.04% |
| C8 median TTFT, seconds | 10.6277 | 0.5189 | -95.12% |
| C8 batch completion, seconds | 53.6033 | 33.2858 | -37.90% |
| **C8 per-stream decode, tok/s** | **21.5802** | **17.7965** | **-17.53%** |
| **7.6K effective cold prefill, prompt tok/s** | **1549.9** | **1411.6** | **-8.92%** |
| 95K effective cold prefill, prompt tok/s | 1647.7 | 1704.9 | +3.47% |

Effective prefill is prompt tokens / TTFT, including transport, scheduling and
first decode, not an isolated kernel measurement. The short-prefill regression
was not resolved. More active C8 streams start sooner and finish the batch
sooner, but each gets less decode throughput while all run concurrently.

### Exact upstream benchmark protocol

Unmodified `tests/bench_decode.py` at `bc68f310`, with endpoint/model rebound:
five repetitions, temperature 0, thinking off, 400-token ceiling, natural EOS
allowed (all measured outputs reached 400). Upstream metric:
`(completion_tokens-1)/(stream_end-first_visible)`.

| Workload | Combined median | Observed range | Median TTFT |
|---|---:|---:|---:|
| Counting 1–200 | 101.655 tok/s | 75.357–103.770 | 0.235 s |
| Explain hash maps | 44.151 tok/s | 42.685–44.718 | 0.238 s |

Median draft-token acceptance was 95.88% / 49.62%, respectively. Counting is
not coding. Against the published September 14 TP3 figures 87.8 / 39.6,
these are nominal +15.78% / +11.49%, with different environment and repetition
counts; they are not a controlled comparison with the maintainer's kit.
Sanitized per-run data: [mia-script.json](benchmarks/tp3-combined/mia-script.json).

### Startup and capacity

| Observation | Baseline | Combined |
|---|---:|---:|
| Target weight load, rank 0 | 324.40 s | 55.49 s |
| Target weight load, rank 1 | 371.56 s | 55.56 s |
| Target weight load, rank 2 | 364.56 s | 55.57 s |
| Native API ready | 601.30 s | first observed at 321.5 s |
| Full launcher return | 656.32 s | observed by 412 s |

These are individual observations, with different graph/artifact work, not
matched repeated cold boots. Loading weights is not total startup.
InstantTensor is already upstream; these measurements do not claim invention
of the loader or another incremental gain over current main.

- 999,485-token multi-position retrieval passed (cold; TTFT 718.747 s).
- 131,059-token context retained 130,560 tokens, TTFT 1.124 s.
- 524,283-token context retained 522,240 tokens, TTFT 3.409 s.
- 24/24 API correctness gates and 50/50 shape warm-up requests passed.
- Selected client tool continuation and reasoning on/off checks passed.
- No full-hour soak, p95/p99 qualification, closing baseline or component
  ablation was completed. These remain limitations, not implied passes.

## Kernel-specific evidence

Historical component screens used fixed tolerances (peak relative <=.003,
row-peak relative <=.05, relative L2 <=.05), exact candidate eager/graph replay,
real weights, all three EP ranges, invalid/nonlocal/zero-weight routes and
counter reuse. Geometry-1 full qualification had 394 comparisons per rank;
450 raw-output geometry profile cases passed. Final mixed policy checks
covered 27 shapes per EP range. Supplemental geometry-2 checks recorded 342
comparisons and 342 exact graph replays across three GPUs, with zero candidate
memcheck/racecheck errors. Stock-reference race warnings from an earlier broad
run were excluded from the later candidate-filtered check, not declared fixed.

Activation limit 0.5 failed the frozen screen; the adapter is restricted to
10.0 and falls back for other limits. Row 64 did not meet the conservative 3%
worst-case speed margin and uses stock despite ABI support. These numerical
screens are not model-wide quality certification.

## Reproduction / adoption

1. Use a separate checkout and compatible image. Keep existing weights,
   drafter, transport and target edits fixed for comparisons.
2. Build and qualify the TP3 bundle as described in
   [the extension README](../extensions/cooperative_moe/tp3/README.md).
3. Bind the reprofiled policy with `select_policy.py --bundle BUILD LOGS...`
   (nine complete logs), then run final policy and sanitizer checks.
4. Run `prepare_profile.py --stock overlay/exl3.py --bundle BUILD`. Set
   `EXL3_OVERLAY_HOST` in a configured `.env.tp3` to the returned absolute
   `exl3-tp3.py` path. The launcher stages all manifest-listed artifacts.
5. Review and append `examples/tp3-throughput.env` to that configuration.
   This profile requests 32 GiB KV/rank and eight slots; measure available
   memory on the actual kit. Its 1M setting is not a guarantee on every image.
   No custom weights or alternative quant download is part of this PR.
6. On a separately reserved test window, launch and rerun API, correctness,
   context, code/prose and concurrency checks. A current-main image may differ
   from the historical runtime. FlashKDA anchor mismatch fails deliberately.
7. For synthetic C1 comparisons use `tests/bench_tp3_matched.py --base-url
   ENDPOINT/v1 --model MODEL --nonce-seed SAME_NEW_SEED --output NEW.jsonl`.
   Use a fresh seed for each A/B series; confirm cached-token counts instead
   of assuming all requests are cold. The public runner reconstructs the
   historical core protocol; it is not the original campaign harness.
   Use upstream `tests/bench_decode.py --help` for the distinct Mia protocol.

No services were restarted and no new GPU measurements were run while
preparing this branch. CPU validation status is recorded in
[validation.json](benchmarks/tp3-combined/validation.json). Full boot/GPU
validation of the upstream-based branch is pending.

## Attribution

- MiaAI-Lab: base EXL3 serving recipe, TP2 cooperative integration, scheduler,
  adaptive verification, FP8 facilities and upstream benchmark script.
- Turboderp / ExLlamaV3: cooperative kernel lineage (`58d4d732` as identified
  upstream), pinned vendor headers at `02aef45c`; original MIT notices retained.
- NNNtrance: FlashKDA TP3 patch at `7f1bd29c`; original integration by
  JaredforReal in [vLLM PR #55737](https://github.com/vllm-project/vllm/pull/55737).
  The adapted patch is marked as modified; Apache-2.0 text is included.
- FlyCockpit: existing upstream TP3 shape/EP plumbing on which this work relies.
- InstantTensor 0.2.0: existing fast loader; no wheel/binary is redistributed.
- sovereignbrah: TP3/64-row adaptation, measured dispatch integration,
  compatibility checks, combined-profile evaluation and this contribution.

Existing source licenses remain applicable; the included MIT and Apache
notices do not replace the repository's existing license notices.
