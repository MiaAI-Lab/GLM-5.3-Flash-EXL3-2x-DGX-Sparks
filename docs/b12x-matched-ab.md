# Matched native B12X experiment on two DGX Sparks

On 2026-08-30, we compared the existing FlashInfer compatibility attention
path with the native B12X `GLM_NEXT` path on the same two GB10 systems. The
candidate kept the target weights, DFlash2 weights, prompts, sampling, chat
template, speculative width, and declared context fixed.

This is a site-specific matched A/B test. It is not a replacement for the
official results in the main README. The current recipe uses a different
1M-context and draft-TP2 shape, so its published concurrency figures are not
directly comparable with the measurements below.

## License and attribution

These measurements used the ShapleyMcg-based EXL3/TR3 checkpoint created by
[Brandon M. Music](https://github.com/brandonmmusic-max/shapleymcg).
ShapleyMcg uses the attribution-required
[ShapleyMcg License v1.0](https://github.com/brandonmmusic-max/shapleymcg/blob/main/LICENSE),
which includes a named exclusion. The required attribution notice and citation
are in [Schedule B of that license](https://github.com/brandonmmusic-max/shapleymcg/blob/main/LICENSE#schedule-b--attribution-notice-and-citation).

```bibtex
@misc{music2026shapleymcg,
  author = {Music, Brandon M.},
  title = {ShapleyMCG: An Auditable Calibration-to-Encoding Pipeline for Low-Bit Mixture-of-Experts Models},
  year = {2026},
  url = {https://github.com/brandonmmusic-max/shapleymcg},
  note = {ShapleyMcg License v1.0}
}
```

The DFlash2 draft checkpoint belongs to Inco AI and is licensed under
[CC BY-NC-ND 4.0](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2#license)
for research and evaluation. This report distributes no model weights,
checkpoint fragments, donor tensors, or modified DFlash2 artifacts.

## Test scope

| Item | Fixed value |
|---|---|
| Hardware | 2x DGX Spark, NVIDIA GB10, 200 GbE CX7 link |
| Target checkpoint | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at `61e26e1484e16d7a603f77040cda9b43cc4a31d6` |
| Draft checkpoint | `incoai/GLM-5.3-Flash-DFlash2` at `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| Control recipe | MiaAI-Lab commit `02e46f2a5f28e2003655c5cf916b72cec0efbb37` |
| Context declaration | 750,000 tokens |
| Speculation | DFlash2 K=7, target verification K=7, draft TP1 |
| Decode request | Temperature 0, thinking off, 400 generated tokens |
| Concurrency request | Four independent structured prompts, 400 generated tokens each |
| Other head workload | A co-resident service remained loaded and used about 4.6 GiB |

Both profiles mounted the same target and draft snapshots read-only. No
replacement weights were downloaded, and nothing was requantized. A site-local
load-time `o_proj` transplant was enabled in both profiles. The transplant was
held constant, is not included here, and is not part of the measured change.
Runtime logs recorded the same 30 tensor edits on both ranks. We did not run a
separate behavioral evaluation of that profile.

## What changed

The control used MiaAI-Lab's FlashInfer compatibility path. The candidate used
the native B12X sparse-MLA implementation and its native KV record:

```text
--attention-backend B12X_MLA_SPARSE
--kv-cache-dtype fp8_ds_mla
--kv-cache-dtype-skip-layers sliding_window
--kv-cache-memory 6600000000
--gpu-memory-utilization 0.75
--max-model-len 750000
--max-num-seqs 4
--max-num-batched-tokens 1024
--block-size 2304
--mm-processor-cache-gb 0
--cudagraph-capture-sizes 1 2 4 8 16 24 32
--speculative-config '{"method":"dflash","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'
```

The complete candidate profile produced the measured result. The experiment
does not isolate the attention backend by itself. It also changes the runtime
stack, explicit KV allocation, processor-cache budget, and startup admission
setting. The 0.75 memory-utilization value is an admission ceiling. The
explicit 6.6 GB setting controls the KV allocation.

The multimodal processor cache setting disables reuse of processed multimodal
inputs. It does not disable image or video inputs. The candidate kept the
existing multimodal limits and skipped dummy multimodal profiling.

The candidate used these pinned sources:

| Component | Pin |
|---|---|
| Entrpi image | `ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark@sha256:284142c5833cbfd540ad42bb8f32cb340451db05a3b84029eaebae54579e9135` |
| Entrpi vLLM fork | [`59c1a0c4c3142a303b6f46ef5f7784731c11074d`](https://github.com/Entrpi/vllm-glm-5.3-flash-spark/commit/59c1a0c4c3142a303b6f46ef5f7784731c11074d) |
| Entrpi B12X backport | [`cd5e8a50ac106b7e32a7d90965d600dd71eb7131`](https://github.com/Entrpi/sparkinfer-glmrt/commit/cd5e8a50ac106b7e32a7d90965d600dd71eb7131) |
| Entrpi deployment recipe | [`3a7c13fa5a5b9373bb91651e61c810ce1912bada`](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark/commit/3a7c13fa5a5b9373bb91651e61c810ce1912bada) |

The published image predates the complete `GLM_NEXT` path used here. We
mounted a bounded set of Python files from the pinned vLLM revision, mounted
the pinned B12X package, and loaded Entrpi's standalone `persistent_topk`
extension. The B12X master revision available during the test no longer
accepted the `exl3_trellis_mcg` source format, so the experiment used Entrpi's
EXL3-compatible `glm-next-backport` branch.

Checkpoint loading on the head used a throttled read-only NFS mount from the
worker. A temporary 48 GiB load-only swap file covered the transient unified
memory peak. The launcher removed that file before any benchmark. Per-shard
`POSIX_FADV_DONTNEED`, Python garbage collection, and `malloc_trim(0)` ran only
during checkpoint loading. They were inactive during inference.

## Results

Decode speed is `(completion_tokens - 1) / (end - first_content_token)`.
Structured and prose results are medians of three runs. Four-client wall
throughput is total completion tokens divided by elapsed wall time for the
whole request group.

| Matched test | Control | Native B12X K=7 | Change |
|---|---:|---:|---:|
| Structured count-prompt decode | 62.56 tok/s | 68.53 tok/s | +9.5% |
| Prose decode | 27.27 tok/s | 28.77 tok/s | +5.5% |
| Four-client wall throughput | 87.16 tok/s | 101.53 tok/s | +16.5% |

The individual decode runs were:

| Workload | Control tok/s | Native B12X tok/s |
|---|---|---|
| Structured | 62.5619, 62.5858, 61.5633 | 68.5288, 68.6500, 66.6238 |
| Prose | 27.2724, 27.4151, 24.2763 | 31.3479, 27.7043, 28.7709 |

The four-client comparison is one matched first run per profile, not a median.
The control completed 1,600 tokens in 18.3567 seconds. The candidate completed
1,600 tokens in 15.7586 seconds. An immediately repeated candidate run with a
fresh run tag reached 138.06 tok/s. We excluded it from the change calculation
because there was no matching run-tagged control repeat.

The sum of overlapping per-request post-TTFT decode rates changed from 174.81
to 165.33 tok/s. That sum fell by 5.4%, but it double-counts simultaneous time
and depends on completion order. We retain it as a diagnostic rather than use
it as a server-capacity result.

The sanitized source data is in
[`docs/_b12x_matched_ab.json`](_b12x_matched_ab.json). Future four-client runs
can use [`tests/bench_concurrency.py`](../tests/bench_concurrency.py):

```bash
python3 tests/bench_concurrency.py \
  --clients 4 --runs 3 --max-tokens 400 \
  --out /tmp/glm53-concurrency.json
```

Three decode runs and one matched concurrency run are enough to report this
deployment result, but not enough to claim a general GB10 speedup. An upstream
evaluation should repeat both profiles with five or more decode runs and at
least three cache-unique concurrency runs.

## Validation

The final candidate passed these checks:

- Arithmetic response
- Native GLM tool call
- Strict JSON schema response
- Prefix-cache reuse
- Coherence probes and NaN scan
- API health before and after the benchmark
- Native `B12X_MLA_SPARSE` and `GLM_NEXT` backend selection on both ranks
- Full K=7 target and draft execution
- No CUDA graph, NCCL, KV geometry, cooperative launch, or OOM errors

The runtime measured a 779,702-token KV pool, which exceeded the 750,000-token
service declaration. One post-gate snapshot reported 5.22 GiB available and
7.34 GiB in the normal swap file. The temporary load-only swap file was absent
during all reported measurements.

The test did not send a 750K-token request. The 750K statement covers the
declared limit and measured allocation capacity, not long-context retrieval or
coherence at that length. The validation set also did not include KLD,
MATH-500, GPQA, vision, refusal, or broad tool-use evaluation. "All checks
passed" means only the checks listed above.

The public manifest binds the report to the chat-template hash, site-local
transplant-manifest hash, overlay hash manifest, OS, kernel, driver, and Docker
versions. The transplant manifest carried a legacy donor label but no donor
commit. Upstream later corrected that attribution. We therefore treat the
transplant as a fixed site-local input, publish no donor identity claim, and
recommend an `ABLIT=0` rerun before proposing this profile as an upstream
default.

## Rejected profiles

We kept failed and inconclusive profiles out of production:

| Profile | Structured | Prose | Four-client wall throughput | Decision |
|---|---:|---:|---:|---|
| B12X K=7, batch 1024, 11.6 GB KV | +9.8% | +3.8% | +4.3% | Missing control template; prose gate missed |
| B12X K=7, batch 8192, 11.6 GB KV | +8.1% | +5.6% | +1.2% | Steady-state memory rejected |
| B12X target K=3, physical draft K=7 | -30.6% | +13.3% | -14.0% first, +17.0% repeat | Structured speed rejected |
| B12X K=7, batch 1024, 6.6 GB KV | +9.5% | +5.5% | +16.5% | Promoted locally |

Static K=3 is not a safe default for mixed workloads. It improved this prose
prompt but cut structured speed by almost one third. Its four-client runs also
varied from 74.98 to 102.00 tok/s. We did not deploy the generic adaptive
controller because it had no calibrated GB10 cost model for this batch shape.

Memory boundary tests also mattered. A 7.0 GB KV allocation produced only
746,341 tokens and missed the declared context. A 7.2 GB allocation exceeded
750K but left too little steady-state headroom. The final 6.6 GB layout used a
smaller 1,024-token prefill arena and produced the measured 779,702-token pool.

## Reproducing the comparison

1. Start from the pinned control recipe and record its structured, prose, and
   four-client results.
2. Keep the target checkpoint, draft checkpoint, chat template, K=7 width,
   draft TP1, sampling inputs, and 750K context fixed.
3. Build the pinned Entrpi B12X backport and `persistent_topk` extension for
   `sm_121a`.
4. Select `B12X_MLA_SPARSE`, `fp8_ds_mla`, and the settings listed above.
5. Verify backend selection, the actual KV pool, and a post-gate memory
   snapshot from runtime logs.
6. Remove all load-only resources, then run the semantic and speed checks.
7. Compare medians and whole-group wall throughput. Do not sum overlapping
   per-request decode rates as a server-capacity metric.

This report documents an optional experimental path. It does not change the
repository's default image, launch script, context declaration, or draft TP.

## Credits

- MiaAI-Lab for the EXL3 two-Spark recipe and benchmark harness
- Entrpi for the native B12X `GLM_NEXT` runtime, vLLM fork, and deployment work
- brandonmusic for the EXL3/TR3 checkpoint
- IncoAI for the DFlash2 draft checkpoint
