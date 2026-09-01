# Non-Blackwell KLD lane (4x CMP 170HX / sm80, vLLM fork, PP4)

The suite's own capture tooling in this repo is SM120-only (`Dockerfile`,
`overlay/exl3.py`). This adds a second engine -- vLLM 0.11.2 fork, PP4 on Ampere
-- and one measured row, so a non-Blackwell box can place its own candidate on the
same yardstick instead of extrapolating from a Blackwell-only panel.

Host assumptions baked into `capture-glm-awq.sh` (all env-overridable except the
`PY` fallback): docker image `vllm/vllm-backport:cmp170hx`, suite at
`/srv/models/fidelity-suites/GLM-5.3-Flash-fidelity-suite-v1`, weights at
`/srv/models/wtdcode/GLM-5.3-Flash-AWQ-W4A16`, PP partition `14,12,12,7`, host
conda env `vllm26` for the bf16 head replay. Nothing here touches the SM120 path.

# GLM-5.3-Flash quantization fidelity (KLD)

Answers one question GSM8K cannot: **how far is our production quantization from
the BF16 teacher, in nats?** GSM8K-100 gives AWQ 99 and NVFP4 97 - saturated, and
it cannot tell "half the probability mass is on the wrong class" from a harmless
tie-break. This is full-vocabulary KL(teacher || student) over sealed text
windows, teacher-forced, with no sampling anywhere.

**Status (2026-09-01):** harness validated against the published protocol, and our
production row is measured: **AWQ W4A16 = 0.0770 nats [0.0696, 0.0854]** over 108
contexts / 221,076 positions (see "Our measured row"). Suite lives in
`/srv/models/fidelity-suites/GLM-5.3-Flash-fidelity-suite-v1`; our captured lane is
`cmp170hx-awq-w4a16/` beside the published lanes (112 contexts, ~1.8 GB).

### Validation runs already done

| run | result |
|---|---|
| `glm53-kld-runs/self-canary.json` | reference lane vs itself: mean KLD **exactly 0.000000**, JSD 9e-9 (the float32 subtraction floor), top-1 1.000000, anchor gate PASS |
| `glm53-kld-runs/fp8-anchor-bf16.json` | FP8 lane vs reference, 30 contexts: median per-context relative delta vs the published report **5.89e-5**, scope-mean drift **3.49e-5** - our scorer reproduces their numbers |
| `glm53-kld-runs/fp8-anchor-111ctx.json` | 48 complete triples: mean **0.028057966** vs the published **0.028058384** for the same indices - scope-mean drift **1.49e-5**, median per-context delta 5.68e-5, all gates pass |
| `glm53-kld-runs/fp8-anchor.json` | the same run in float32: delta 1.17e-2. Kept as the negative control that pinned the dtype rule below |
| `selftest.py` | 13 named CPU checks, all pass |
| lm_head bit-check | our AWQ `lm_head.weight` is bit-identical to the suite head |
| lane geometry control | per-token RMS of our captured lane (min 0.670 / median 1.369 / max 1.445) against the teacher lane (0.660 / 1.370 / 1.445), the FP8 lane (0.616 / 1.369 / 1.445) and the published cut-point stats (0.784 / 1.380 / 1.442). Re-applying the norm would pin every row near ‖norm.weight‖ = 1.4315; capturing before it would show a much wider, depth-growing spread. The shape of the distribution is the evidence, not the mean |

## The suite that is actually on the hub

`malaiwah/GLM-5.3-Flash-fidelity-suite-v1`, schema
`glm53flash-distribution-fidelity/6`, lands in
`/srv/models/fidelity-suites/GLM-5.3-Flash-fidelity-suite-v1`:

| path | what |
|---|---|
| `suite/suite-manifest.json` | 5,120 sealed windows: index, stratum, partition, source cluster, token count, per-window `token_sha256`, plus corpus/windowing/contamination provenance |
| `suite/tokens/context-NNNN.json` | the sealed token ids |
| `head/head.safetensors` | shared `lm_head.weight`, `[154880, 4096]` BF16, sha `47eaf729…` |
| `head/final_norm.safetensors` | shipped **for verification only** - see the cut point below |
| `reference-bf16-shard0/hidden_NNNN.safetensors` | teacher lane, one `[2047, 4096]` BF16 tensor per context |
| `as-served-fp8-shard0/…` | same shapes for the official FP8 model |
| `reports/*.json` | the published rows, per-context values, determinism and cross-check reports |

Two things about the geometry, both verified from the shipped bytes:

* **Only shard 0 of 10 is distributed.** `capture-manifest-full.json` says
  `complete: true, contexts: 5120` in a directory holding **512** files, and even
  ships `capture-manifest-shard.json` telling you so ("any coverage or CI computed
  from `captures[]` is wrong by 10x"). We hold windows **0-511**, i.e. 512 × 2047
  = **1,048,064 scored positions**. Any number we publish must name that scope.
* **Geometry**: `context_length` 2048, `scored_positions_per_context` 2047,
  `first_scored_position_index` 0. Row *r* is the state after consuming
  `tokens[r]`; its target is `tokens[r+1]`.

## The cut point is the trap

`<lane>/capture-cut-point.json`:

> `semantic_point: "after_final_rmsnorm_before_lm_head"` … these hidden states are
> **already past the model's final RMSNorm**. `head/final_norm.safetensors` is
> shipped so a reader can verify the norm weights, **NOT** so they can be applied
> again before the head. Replay applies `head/head.safetensors` ONLY.

Their evidence is per-token RMS of the published tensors (min 0.784, median 1.380,
max 1.442) against `rms(final_norm.weight) = 1.4315`. Re-applying the norm is a
silent-deadly bug: it rescales every logit row like a temperature change and
inflates KLD while the run looks completely healthy. The loader refuses a lane
whose cut point claims `applied_at_replay`, and `selftest.py` proves the refusal
fires (`suite_scorer_path`).

Consequence for our capture: the worker tap is a **forward hook on the output of
`language_model.model.norm`** (last PP stage only - `PPMissingLayer` elsewhere),
not a pre-hook on its input. Only calls with exactly 2048 rows are kept, and the
last row is dropped, so our lane has the same `[2047, 4096]` shape as theirs.

## dtype is part of the protocol (measured, not assumed)

Replaying the shared head in **float32 does not reproduce the published rows**.
On the first 30 contexts:

| replay dtype | median per-context relative delta | scope-mean drift |
|---|---|---|
| float32 | 1.17e-2 | 1.00e-2 |
| **bfloat16** | **5.89e-5** | **3.49e-5** |

So the suite multiplies `[2047,4096] × [4096,154880]` in **bf16**, exactly the
dtype the tensors are stored in, and the residual 1e-4-level differences are GEMM
accumulation order. The mechanism for the fp32 gap is real and measurable: bf16
logits carry ~0.08 nats of rounding noise, which *inflates* KL by ~0.003 nats
(about 10% at this KL, verified with a synthetic same-distribution pair). Using
fp32 for our own row would therefore make AWQ look ~1-4% better than every
published row for a reason that has nothing to do with AWQ. The harness defaults
to bf16 and the anchor gate is what proves the choice.

## Head equality

`reports/head-equality-fp8.json`: `head_equal: true`, `final_norm_equal: true` -
both lanes share `lm_head` byte-for-byte, which is what licenses replaying through
one head. Verified on our side too: `lm_head.weight` lives in
`model-00009-of-00009.safetensors` and is **bit-identical to the suite head**
(`torch.equal` = True, max abs diff 0.0). `capture_hidden_states.py` enforces this
before it will run. If a repack ever quantizes the head, KLD would measure the
head instead of the experts.

## Download mechanics (why `fetch-fidelity-suite.sh` is curl, not huggingface_hub)

`huggingface.co` is DNS-poisoned from this box (AliDNS returns Twitter IPs,
DNSPod a Facebook IP), DoH is blocked, and `hf-api.gitee.com` wants a token.
`hf-mirror.com` answers but at 25-80 s per request, and `huggingface_hub` 1.28
fetches the *tree listing* straight from `huggingface.co` regardless of
`HF_ENDPOINT`, so the python client is useless here. The mirror's `/resolve/`
redirect points at `us.aws.cdn.hf.co`, which is directly reachable and honours
ranges (~0.8 MB/s per connection).

Two failure modes, both measured, both survived:

1. **stalls** - a connection sits at 0 B/s for minutes, small files as often as
   big ones (`--connect-timeout 15 --speed-limit 4096 --speed-time 25`, then retry);
2. **silent corruption on resume** - after an aborted stall, resuming from the
   local breakpoint produced files with HTTP 200 and *wrong bytes*: a 14 KB token
   window unparseable at char 4096, a file at exactly 4096 bytes, a zero-byte file.
   13 of the first 154 files were corrupt. Nothing is trusted: `SHA256SUMS` is
   fetched first, every file is hashed before it is put in place, and after two bad
   resumes the partial is deleted and the transfer restarts clean.

```bash
bash fetch-fidelity-suite.sh --verify --purge-bad   # hash what is present
bash fetch-fidelity-suite.sh SELECT=ref PARALLEL=10 # 512 reference shards, 8.6 GB
```

## Our measured row (2026-09-01)

```
candidate      cmp170hx-awq-w4a16  (wtdcode GLM-5.3-Flash-AWQ-W4A16, our vLLM fork, PP4 14,12,12,7)
scope          108 contexts scored of 112 captured (indices 0-114; 63/68/90/113 have no
                 reference shard - the reference-lane download is partial, see below),
                 221,076 positions, 97 source clusters
token_mean_kld 0.077026 nats      bootstrap 95% [0.069561, 0.085387]
mean_jsd_bits  0.024241           top1_agreement 0.9085
p95 / p999 / max  0.314 / 3.167 / 17.29 nats
strata         literary 0.0884 > code 0.0848 > encyclopedic 0.0803 > scientific 0.0695 > multilingual 0.0586
report         ../glm53-kld-runs/awq-row-112ctx.json  report_sha256=c8135e5623dd966bfe038e3d88f5f06020c280dab60dca32d3f9323c09a9bbda
```

Against the published rows on the **same 108 indices**: FP8 = 0.026898, so **AWQ measures 2.86x the
FP8 row** and ranks below the published EXL3 K3 (0.0505) on the full 5,120-context scale
(0.0137 K6 / 0.0255 TR3-4bpw / 0.0273 K4 / 0.0281 FP8 / 0.0505 K3 / **0.077 ours** / 0.155 K2).

**Why 108 of 112**: 112 windows were captured (indices 0-114), but four (63, 68, 90, 113) have no
reference shard on this host -- the reference-lane fetch stopped partway (384 shards present, index
range 0-428, not contiguous; the FP8 lane has 243). The loss is scattered across indices rather than
concentrated in a stratum, and finishing the fetch extends the row to 112 by re-scoring on CPU, no
cards needed.

`d(AWQ, official-FP8-lane) = 0.0836` over the 49 shared source clusters -- larger than
`d(AWQ, BF16) = 0.0770`, i.e. our error is *independent* of FP8's, not an FP8-like perturbation.

**What the number includes**: quantization **plus** our engine (PP4 + mHC handoff, eager capture),
against a teacher captured on a different engine. The published engine-drift figure for a native
BF16 recapture is 0.0115 nats (vcruz2, full scope). We hold no BF16 GLM-5.3-Flash weights, so we
cannot subtract an engine term: **0.077 is an upper bound on the AWQ quantization error**, not a
decomposition. GSM8K-100 = 99/100 on the same weights: 9.2% of positions disagree with the
teacher's argmax and the tail (p999 = 3.17 nats) carries most of the distance, which is why a
pass@1 benchmark does not see it.

## Running it

```bash
MODE=self     bash capture-glm-awq.sh   # reference vs itself: mean must be exactly 0.0
MODE=anchor   bash capture-glm-awq.sh   # FP8 lane vs reference: reproduce the published row
MODE=capture  LIMIT=8 bash capture-glm-awq.sh   # smoke: 8 contexts through our engine
MODE=capture  bash capture-glm-awq.sh   # teacher-force our AWQ (takes all four cards)
MODE=score    bash capture-glm-awq.sh   # our lane vs reference -> the row
```

`MODE=self` and `MODE=anchor` need no capture; both run on the host (the head is
1.27 GiB in bf16, and they fit in the ~5 GiB the production worker leaves on a
card - pass `CUDA_VISIBLE_DEVICES=3`). The anchor runs `--compare-report
reports/report-fp8-vs-bf16.json`, which compares **per context** with their
`per_context[]` - a mean can match by accident; 512 per-context values cannot. If
the median relative delta exceeds 1e-3 the gate fails and nothing is published.
`--expect-mean` (the published global 0.028104) is deliberately **off** by default:
that mean belongs to the whole 5,120-context suite, and we hold 512; comparing a
partial mean against it measures sampling noise. Set `EXPECT_MEAN=1` if you ever
score the full published scope.

`MODE=capture` refuses to start while a container holds the cards (stop the worker
through the orchestrator first). Engine flags mirror the AWQ production profile
(`VLLM_PP_LAYER_PARTITION=14,12,12,7`, EP on, `attention_config
{sparse_mla_force_mqa: true}`, GMU 0.95) with two disclosed differences:
**speculative decoding off** (a draft tower would put its own quantization into
the hidden states) and `max_model_len` 4096. Both are recorded in the lane's
`capture-receipt.json`, along with the lm_head check. Smoke it with `LIMIT=8`
before the full pass: a hook that never fires (CUDA-graph replay bypassing Python)
or a row-count mismatch shows up there in minutes, and the runner aborts unless
every context produced exactly one dump.

The kwargs are pre-checked against `dataclasses.fields(EngineArgs)` and the whole
`VllmConfig` is built once before the weights are opened - `LLM(**kwargs)` accepts
fewer options than the CLI does (`num_scheduler_steps`, `tool_call_parser` and
friends are CLI-only), and a typo there otherwise surfaces after a 178 GB load.

One parity note worth recording: **`VLLM_ATTENTION_BACKEND` is not a recognised
environment variable in this build** (vLLM logs `Unknown vLLM environment
variable` and `AttentionConfig.backend` stays `None`). `launch-glm.sh` sets it to
`TRITON_MLA_SPARSE`; the fork has moved backend selection to `attention_config`,
so the variable is inert in production and in the capture alike. Parity holds
because both are equally inert - but do not describe the serving backend as pinned
by that variable. What *is* effective and verified reaching the config is
`sparse_mla_force_mqa=True`.

## Reading the number

The published rows, from the suite's own `reports/` (teacher = the sealed BF16
lane; candidates other than FP8 were served through exllamav3, so they carry an
engine-delta term, and so does ours):

| row | bits | mean KLD (nats) | gate |
|---|---|---|---|
| EXL3 K6 | ~6 | **0.013723** | pass |
| TurboDev exl3 mul1 | 4.05 (head 6-bit) | **0.025526** | pass |
| EXL3 TR3 mcg | 4.0 | **0.025503** | pass |
| Official FP8 (same stack) | 8 | **0.028104** (CI 0.02721-0.02898) | - |
| EXL3 K4 | ~4 | **0.027263** | pass |
| EXL3 K3 | 3.0 | **0.050501** | pass |
| EXL3 K2 | 2.0 | **0.155210** | **fail** |
| NVFP4 (our stack, deleted) | 4 | 0.0605 (logits protocol) | - |

Context for interpretation:

* **our own repeat-noise floor** - `reports/determinism-noise-*.json`: the same
  model captured twice gives **0.00087** (BF16) / **0.00063** (FP8) nats on the 32
  sentinel contexts. Differences below ~1e-3 are jitter.
* **engine floor** - `reports/vcruz-k2-2bpw-attributable.json` computes
  `attributable = quant_mean - floor_mean`, with `floor_mean = 0.011506` from a
  *native BF16* recapture against the sealed teacher. Any number we publish is
  `quant + engine drift`, and we cannot measure our own floor without a BF16
  capture of this architecture on this stack (we hold no BF16 weights). So our row
  is reported raw, with the floor disclosure, and **not** as `attributable`.
* FP8 at 0.028104 against a 4-bit TR3 at 0.025503 is the honest scale of the
  question we care about: our AWQ row is interesting precisely because it lands in
  this band, where GSM8K cannot see anything.

## Determinism rules baked in

* teacher-forced, one context per generate call, `max_tokens=1`, no sampling,
  `enable_prefix_caching=False`, `seed` fixed, no speculative decoding;
* every context must produce exactly one dump: zero dumps aborts (a CUDA-graph
  replay that never re-enters Python would otherwise look like a short lane) - run
  with `--enforce-eager` if that fires;
* the loader verifies each window's sealed `token_sha256` (sha256 of
  `json.dumps(token_ids)`), so a different tokenizer snapshot or a corrupt
  download cannot score;
* every lane file is checked against `capture-manifest-shard.json` digests;
* reported CI is a **source-cluster bootstrap** (837 clusters, 10,000 draws),
  matching the published method: windows from one document are not independent, so
  bootstrapping contexts or positions would understate the interval;
* `--offset-audit` scores the candidate shifted by ±1 row: top-1 agreement must
  collapse (0.996 → 0.016 in the published audit). That is the alignment proof;
* reports are sealed (`report_sha256` over the payload minus the seal field).

## Files

| file | role |
|---|---|
| `kld_core.py` | sealing primitives, suite loader, cluster bootstrap, KLD/JSD |
| `logit_dump_hook.py` | worker-side taps: post-norm hidden state (primary), `Sampler.gather_logprobs` logits spill (secondary); inert unless `GLM_KLD_DUMP_DIR` is set |
| `sitecustomize.py` | installs the logits tap in every engine process |
| `capture_hidden_states.py` | teacher-forced capture + lm_head bit-check + lane manifest |
| `capture-glm-awq.sh` | `MODE=self\|anchor\|capture\|score`, card guard |
| `score_hidden_kld.py` | head-only replay, per-context metrics, strata, bootstrap, offset audit, anchor gate |
| `fetch-fidelity-suite.sh` | verified resumable download |
| `probe_container.py` | engine preflight without loading weights |
| `selftest.py` | 13 named CPU checks: `python selftest.py` |

## Validation performed

* `selftest.py`: 13/13 named checks pass, including the cut-point refusal, the sealed-token
  refusal, head-only replay matching an independent float64 reference, the
  reference-vs-itself exact zero, the offset audit, and the tap's row gate;
* `probe_container.py` inside `vllm/vllm-backport:cmp170hx`: tap installs,
  `LLM.apply_model` present, every engine kwarg accepted;
* published-suite files verified against `SHA256SUMS`.
