<h1 align="center">GLM-5.3 Flash EXL3 for 2x DGX Sparks</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

OpenAI-compatible vLLM serve of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) as
**[Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)**
— a byte-identical public mirror of
[brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
snapshot `5ab363a8…` (uniform-K4 EXL3/TR3 routed-experts, 4 bpw, ~164 GiB, 120 shards)
so this recipe stays fetchable if the upstream Hub id moves. On a **2× NVIDIA GB10**
kit: tensor-parallel size 2 over CX7, native `sm_121a` cubins, API on `:8888`.
Served model id: **`GLM-5.3-Flash-EXL3`**. EXL3/TR3 quant by
[brandonmusic](https://huggingface.co/brandonmusic).

This is **EXL3 weights + fp8 KV** on GB10. Do not pass `--moe-backend marlin`.
The Hub card on brandonmusic (TP2/EP2/DCP2 + calibrated NVFP4 MLA KV) is the SM120 B12X
image (`verdictai/glm53-flash-exl3-k4:…-v84-dflash2`), not this overlay. Target KV
stays packed **`fp8_ds_mla`**. Speculator is **DFlash2 k=7**
([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2));
draft attention is **FLASH_ATTN** (do not pin `TRITON_ATTN` — that mask is causal
inside the draft block on this image and collapses later-position accept).

## Decode (this kit, 2026-08-28)

Official numbers: sparkDash Decode bench, DFlash2 k=7, **Structured** (count 1→200) and **Code** (`clamp_00`…`clamp_49`) — same high-accept regime. Temp **0**, thinking **off**, 400 tokens, CUDA graphs, fused EXL3 MoE. Prompt types, not grammar / schema. Stream tok/s is per request; aggregate is all streams.

| Concurrency | TTFT | Stream tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| **×1** | **719 ms** | **62.9** | **62.9** |
| **×2** | 6.62 s | 51.7 | 103.3 |
| **×4** | 6.30 s | 37.1 | 146.5 |

That 2026-08-28 decode serve used `--max-model-len 1000000` with a **1,754,237-token** KV pool. These runs are warm / empty KV — they do not need a filled 1M cache.

Lab `tests/bench_decode.py` on the same protocol (median of 5 × 400, 2026-08-30 C4, `DFLASH_DRAFT_TP=2`): Structured **65.1** tok/s (0.959 accept / 6.71 per step); Prose (hash-map) **27.1** (0.341 / 2.39). Prior TP=1 lab: 61.7 / 26.9. Long context / mixed (~60–100k KV) 24–27. MTP k=2 baseline ~24.6.

Structured per-pos (lab median): **0.98 / 0.98 / 0.94 / 0.94 / 0.91 / 0.83 / 0.83**.
Prose per-pos: **0.75 / 0.58 / 0.41 / 0.28 / 0.16 / 0.09 / 0.06**.
Pinning `attention_backend=TRITON_ATTN` dropped structured to ~29 tok/s / 0.31 accept
(pos0 healthy, later positions collapsed).

Re-measure:

```bash
# structured (count 1→200)
python3 tests/bench_decode.py --phase structured --structured --runs 5 --max-tokens 400 --skip-coherence --out /tmp/glm53-structured.json
# prose (hash-map explanation)
python3 tests/bench_decode.py --phase prose --runs 5 --max-tokens 400 --skip-coherence --out /tmp/glm53-prose.json
```

## E2 fat-expert prefill — [PR77](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/77) (2026-09-01)

PR77 adds purpose-built direct/scatter CUDA kernels for the routed “fat”
experts and lifts fully uncached long-context prefill by about **20–21%**.
The controlled promotion ran E2 on → legacy off → E2 on at MNBT 2048, with
five unique-salt cold samples per rung per boot:

| Fully uncached rung | Legacy mean tok/s | PR77 pooled mean tok/s | Gain |
|---|---:|---:|---:|
| ~8K | 941.04 | 1132.32 | **+20.33%** |
| ~100K | 1023.20 | 1241.71 | **+21.36%** |
| ~300K | 995.05 | 1201.02 | **+20.70%** |

All 45 observations passed the cold gate, with complete separation at every
rung. A second 2× DGX Spark deployment reproduced **+21.0% at 100K** and
**+20.4% at 300K**. These repeated results establish E2 as the production
prefill path; the decode path is unchanged. They replace earlier preliminary
figures that included APC hits.

On the independent `MAX_NUM_SEQS=16` geometry, MNBT 2048 delivered the best
measured balance of prefill throughput and KV capacity. MNBT 7168 remains the
current maintainer default for `MAX_NUM_SEQS=4`, pending a repeated same-kit
comparison.

## Quality (KLD)

Independent teacher-logit panel from
[malaiwah on the 4bpw discussion](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/discussions/1#6a9144846b0bdba943bfe86f):
KLD(teacher ‖ model), five cold runs, 25 sealed windows (51,175 positions). This
scores the **weights**, not this GB10 overlay. We serve the **4bpw** row.

| Model | Mean KLD (nats) | Size |
|---|---:|---:|
| TR3 K6 (6bpw) | 0.013723 | 254 GB |
| Official FP8 (cross-stack) | 0.020615 | 328 GB |
| **This checkpoint — EXL3 4bpw** | **0.024555** | **176 GB** |
| Official FP8 (brandonmusic stack, v44) | 0.024629 | 328 GB |
| NVFP4 (brandonmusic stack, v44) | 0.060535 | ~180 GB |

On the same stack, 4bpw matches official FP8 (~1.00× KLD) at **54%** of the bytes.
K6 (`malaiwah/GLM-5.3-Flash-TR3-6bpw`) is a different checkpoint. Padded DFlash
slot-share is an allocator change only — target KV stays packed `fp8_ds_mla`,
same path as the compact-64 fp8 serve (not NVFP4 KV).

## What runs

| Layer | Runtime |
|---|---|
| API | vLLM OpenAI (`/v1/chat/completions`) on the head, port **8888**. Open by default; set `VLLM_API_KEY` for optional Bearer auth |
| Weights | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (mirror of `brandonmusic/…` snapshot `5ab363a8…`) |
| Model id | `GLM-5.3-Flash-EXL3` (`--served-model-name`) |
| Image | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` FROM `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` (arm64, CUDA 13.0) |
| Executor | `mp`, `--nnodes 2`, `--tensor-parallel-size 2` |
| Head | this machine, `HEAD_IP=10.0.0.1`, container `glm53-exl3-head` |
| Worker | `WORKER_USER@WORKER_IP` (this kit: `zurih@10.0.0.2`), `--headless`, `glm53-exl3-worker` |
| Fabric | CX7 QSFP: `enp1s0f1np1`/`rocep1s0f1` ↔ `enp1s0f0np0`/`rocep1s0f0`. Image NCCL (`USE_HOST_NCCL=0`) |
| Attention | `FLASHINFER_MLA_SPARSE_SM120` (NoPE MLA padded into GLM_NSA 576-wide) |
| KV | `--kv-cache-dtype fp8` → packed **`fp8_ds_mla`** (target). Draft DFlash2 KV is `auto`/bf16. Latest validated 7168/rightsize pool: **1,243,902** tokens / **1.24×** at 1M. `--enable-prefix-caching` (block-aligned hits; see Prefix caching) |
| Experts | packed trellis + suh + svh + mcg, codebook MCG, **one fused `exllamav3_ext.exl3_moe` launch per layer** |
| Dense / shared / attn / embed / lm_head | native (unquantized) |
| Spec | **DFlash2 k=7** (`incoai/GLM-5.3-Flash-DFlash2`); draft KV `auto`/bf16, draft TP=2, FLASH_ATTN. Rollback `SPEC_METHOD=mtp` |
| Context | **1M** (`MAX_MODEL_LEN=1000000`). Latest validated 7168/rightsize pool: **1,243,902** tokens / **1.24×**. Pool size varies with MNBT, activation/graph reservations and hybrid block geometry |
| Tools / reasoning | `--tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45` |
| Graphs | on (`ENFORCE_EAGER=0`) — MTP capture `1 2 3 4 6 8 12`; DFlash2 capture `1 2 4 8 16 24 32` |
| Vision | on (`LANGUAGE_MODEL_ONLY=0`) — image + video, `--limit-mm-per-prompt {image:4,video:1}`, `--skip-mm-profiling` |
| Ablit | **off** (`ABLIT=0`). Stock `o_proj`. Set `ABLIT=1` to enable; see [Abliteration](#abliteration-ablit1) |

Kernels: `TORCH_CUDA_ARCH_LIST=12.1a`. ExLlamaV3 pin `c5d9c657` (0.0.43) exposes
`exl3_moe` / `exl3_moe_max_concurrency`; aarch64 CPU allreduce stubs in
`overlay/patch_exl3_ext_aarch64.py`.

## Abliteration (`ABLIT=1`)

**Off by default.** Unset or `ABLIT=0` serves stock `o_proj`. `ABLIT=1` is
opt-in refusal-direction ablation at weight-load on top of the EXL3 checkpoint
— nothing on disk is rewritten.

Artifacts live in `ablit/` (from
[drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock),
method `dealign-oproj-transplant`: layers **15–45** edited, **0–14 stock**
safety anchors, MTP block included).

**Method — `ABLIT_METHOD=transplant` (default via `auto`).** The published
checkpoint's `o_proj` L15–45 are byte-copied from that recipe
(same 120-shard layout; o_proj is native BF16 in both). Since this EXL3
checkpoint's o_proj is byte-identical to the NVFP4 body's, replacing those
31 tensors at load reproduces the published edit exactly. Fetched once (~2.7 GiB,
range-requests only the o_proj tensors — no full checkpoint download):

```bash
python3 ablit/fetch_transplant.py      # donor is public; HF token optional
ABLIT=1 ./start.sh restart
```

Fidelity checks (logged per layer): mean `rel_l2` vs stock ≈ the published
fingerprint (`Mean Δrel 0.126`; their L44 outlier `0.74` matches our measured
`0.737`). Their own gate: `refusal32` → 32/32 bypass on the NVFP4 stack.

**Why not direction orthogonalization (`ABLIT_METHOD=proj`)?** Measured: the
shipped direction vectors (`refusal_direction_glm53_*.pt`) are statistically
random against the stock o_proj (max |rᵀW| 0.084 vs 0.084 for random unit
vectors; mean identical). The publisher's own table shows projection variants
(Blackfrost V / Dealign V SVD) topping out at 9/32 bypass — only the byte-copy
hits 32/32. `proj` stays available for custom directions you extract yourself:

```text
W' = (I - alpha * r rᵀ) W      # ABLIT_ALPHA, default 3.0 (alpha_ref)
```

Mechanics (both methods): o_proj is native BF16 and `RowParallelLinear` shards
only the input dim, so edits are per-rank row-space ops with no collectives —
identical result on both TP ranks. Applied by `overlay/ablit_runtime.py` at
the end of `Glm5NextModel.load_weights` / `Glm5NextMTP.load_weights`, before
CUDA-graph capture. The DFlash2 drafter is never touched (for the checkpoint's
own MTP block, `ABLIT_INCLUDE_MTP=1` transplants its o_proj too — the
publisher found a stock MTP draft head keeps proposing refusals).

Disable with `ABLIT=0` (or unset) — hook is a no-op, stock weights. No
rebuild: artifacts + hook are bind-mounted into both containers every start.

| Knob | Default | What |
|---|---|---|
| `ABLIT` | `0` (off) | `1` = apply the o_proj edit at load (both ranks). Default and unset = stock weights |
| `ABLIT_METHOD` | `auto` | `auto` = transplant when `ablit/transplant/` is populated, else `proj` \| `transplant` \| `proj` |
| `ABLIT_LAYERS` | `15-45` | inclusive range; `45` is the checkpoint MTP block |
| `ABLIT_ALPHA` | `3.0` | proj-only: projection scale (`1.0` = plain projection) |
| `ABLIT_DIRECTION` | `dealign` | proj-only: `dealign` \| `bf_oproj` \| path to a custom direction `.pt` |
| `ABLIT_INCLUDE_MTP` | `1` | also edit the MTP block's o_proj when it loads (`SPEC_METHOD=mtp`) |

Caveats: the KLD quality panel above was measured **without** ablit; expect
behavioral drift and re-run `tests/bench_decode.py` after enabling (DFlash2
acceptance can shift). Donor licensing: Dealign weights carry their own
terms — see the donor card before redistributing anything derived.

## Why the overlay exists

Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` loads this checkpoint and dies on
the first forward: `pe_dim must be 64 for fp8_ds_mla`. GLM-5.3-Flash is **NoPE MLA**
(`qk_rope_head_dim=0`, `kv_lora_rank=512`). On SM12x the only sparse-MLA backend is
`FLASHINFER_MLA_SPARSE_SM120`, whose packed record is 512 NoPE + 16 B scales + 128 B
RoPE (656 B). The overlay zero-pads the 512-d latent into that GLM_NSA geometry
(RoPE pad is zeros; the QK dot is unchanged) and registers a real EXL3 method so
routed experts stay packed instead of expanding to BF16.

Registering the name `"exl3"` is not enough. Experts must stay **trellis + suh +
svh + mcg** and run Trellis/MCG. Shared experts, attention, embeddings, and
`lm_head` stay native. TP=2 shards gate/up **column-wise** and down **row-wise**;
the MoE runner all-reduces once per layer.

DFlash2 on this fork also needs three GLM-specific hooks the stock image lacks:
EAGLE3 aux capture at mHC (`hc_post` then `hc_contract` → 4096-wide, taps log as
`(6, 15, 25, 34, 43)`), drafter SWA **padded slot-share** onto the MLA tensors
(`block_size=64`, `page_size_padded` equal to the MLA page so drafter layer i
co-owns MLA tensor i; the layout validator allows that one padded case), and
checkpoint `is_causal: false` so draft attention is bidirectional inside the
block. Draft KV is forced `auto` because dense DFlash2 cannot use the target's
`fp8_ds_mla` layout and SM121 has no FA3/FA4 for plain FP8.

The pinned vLLM `487ecf187` also predates two merged XGrammar speculative-decode
fixes. `overlay/patch_xgrammar_termination.py` source-exactly backports
[vLLM PR #52805](https://github.com/vllm-project/vllm/pull/52805)
([commit `12f64b39`](https://github.com/vllm-project/vllm/commit/12f64b39d29282437e35be9aa5db432fb2a1a6e6))
and [vLLM PR #53046](https://github.com/vllm-project/vllm/pull/53046)
([commit `c6e19b3`](https://github.com/vllm-project/vllm/commit/c6e19b3be24338759a443e03c8325d76da9ee202)).
The first stops `accept_tokens()` and `validate_tokens()` at the first
terminating token, ignores later advances after termination, and clears the
cached flag on reset. The second validates drafts produced before a mid-window
reasoning-end marker before advancing the newly active grammar, avoiding a
spurious `Failed to advance FSM` error for invalid drafts. The fail-closed,
idempotent script preflights both source files before writing and is already
mounted and run on both ranks. These address issue #19's matcher-error paths;
they do not reinterpret a client request that combines GLM XML tool output
with a JSON response schema, nor do they remove cold-prefill queue time.

`overlay/patch_kpool_tail_slotmap.py` clamps the generic paged slot-map kernel
so `KpoolTailSpec`'s one-block circular scratch cannot index past its single
block-table entry. Without that clamp, every token at `pos >= block_size`
fills the mapping with adjacent memory and the kpool seed/update kernels
write through it — long generations (~2k tokens) crash or silently corrupt
another layer's indexer. The clamp is identity for every other KV group.
Fail-closed, idempotent, mounted and run on both ranks.

`overlay/patch_glm_video_placeholders.py` routes Glm5Next video timestamps through
the glm46v path and aligns placeholder blocks to encoder `grid_t`. The overlay
also disables GB10 `persistent_topk` so long-history decode uses
`top_k_per_row_decode`.

## KV cache (this kit, 2026-08-29)

`--kv-cache-dtype fp8` is required. The SM12x sparse-MLA kernel only accepts packed
`fp8_ds_mla`. **bf16 KV has no sparse kernel** on this arch. Metrics report
`cache_dtype=fp8`; that is the **target** path. The 2026-08-29
**1,754,237-token** receipt is hybrid BlockPool accounting, not uniform fp8 tensors.

| Piece | Dtype / layout | Notes |
|---|---|---|
| Target MLA (12 layers) | packed **`fp8_ds_mla`**, 656 B/token/layer | `FLASHINFER_MLA_SPARSE_SM120` |
| Indexer / kpool tail | follows the GLM-5-Next hybrid groups | kernel block 64 |
| Mamba (33 layers, 3 groups) | `mamba_cache_dtype=auto` | window / state, mostly length-independent |
| DFlash2 draft (5 SWA layers) | **`auto`/bf16**, 2048 B/token this boot | no MLA FP8 backend on SM121 |

With DFlash2 + vision + util **0.87**, the pool is leftover UMA after weights and
CUDA graphs. The 2026-08-29 boot below records the padded-slot-share allocator
state:

| | |
|---|---|
| GPU KV cache size | **1,754,237** tokens |
| Max concurrency at 1M | **1.75×** (same 1.75M pool; was 1.95× at 900k) |
| GPU blocks | **690** (`block_size=64`, `mamba_block_size=16`) |
| Available KV memory | **18.67 GiB** |
| `kv_cache_max_concurrency` | 1.949… |
| Boot line | `padded slot-share block=64 mla_page=2351104 (was block=16); draft_bytes/token=2048` |

Latest validated boot (2026-09-01, MNBT 7168, `MAX_NUM_SEQS=4`, E2 and
indexer rightsizing on): **1,243,902 KV tokens / 1.24× at 1M**.

DFlash2 cannot exact-fit the 656 B MLA page, so the five SWA layers **padded
slot-share** the MLA tensors: manager `block_size=64` (indexer kernel size; not
the 3584-token mamba-aligned MLA manager) and `page_size_padded` equal to the
MLA page. Drafter layer *i* co-owns MLA tensor *i* at window-bounded BlockPool
IDs, like mamba. Per-block pool bytes unchanged. This is an **allocator**
change — target attention is still the same `fp8_ds_mla` kernel and scales as
the compact-64 fp8 serve. It is not NVFP4 KV.

Inheriting the MLA manager block (1152, later 3584) made each of 5 draft layers
tens of MiB per pool block and pinned logged concurrency near 1×
`max-model-len`. Compact-64 without slot-share still burned unique IDs per
draft layer.

Live occupancy, temp **0**, thinking **off**, unique pads, `max_tokens=8`:

| Load | HTTP | Peak KV | Wall / TTFT | Notes |
|---|---|---:|---:|---|
| ~36k ×1 (compact-64, no slot-share) | 200 | **44.6%** | — | five standalone DFlash ID sets |
| ~36k ×1 (padded slot-share) | 200 | **~16%** | — | one shared ID set |
| ~36k ×3 concurrent | **3× 200** | **21%** (two in flight) | 54 / 96 / 137 s | `GLM53_MIXED_PREFILL_CHUNK=skip` still serializes prefills (`Running: 1`, others wait on capacity, then deferred) |
| ~256k ×3 concurrent | **3× 200** | **29.5%** (two in flight) | 305 / 608 / 916 s | live on the 900k boot; 256,013 prompt tokens each, gen `OK`; third waited (skip) |
| ~300k ×1 streamed | **200** | **26.0%** | **356 s** TTFT (~840 tok/s) | 299,213 prompt tokens, gen `OK`; MNBT=1024 remeasure **323 s** / ~928 tok/s; production MNBT=2048 **319 s** / **941** tok/s |

Live **3×256k** held (the original failure). Prefills still serialize under skip; two 256k contexts were in KV at once at 29.5%. One 256k sat ~25%. Hybrid occupancy is a large length-independent floor (mamba + DFlash window) plus MLA pages that scale: 36k → 16%, 256k → ~25%, 300k → 26%.
Default is **1M**. Do **not** drop `MAX_MODEL_LEN` to 256k to “free” slots —
logged tokens ≈ concurrency × that cap, and the hybrid floor then shrinks the
pool.

Keep **`SKIP_MM_PROFILING=1`** — a max-size image+video dummy profile OOMs this UMA.
`LIMIT_MM={"image":4,"video":1}`.

**NVFP4 KV is not available here.** FlashInfer’s SM12x NVFP4 kernels are dense MHA,
not sparse MLA. Do not confuse that with NVFP4 **weights** (`--moe-backend marlin`).

## Prefix caching (this kit, 2026-08-30)

`--enable-prefix-caching` is on. The OpenAI API is **stateless**: the client
resends the full history each turn; vLLM hashes that prefix. Concurrent chats
do **not** mix activations. `--max-num-seqs 4` is four **in-flight** generations,
not four parked sessions. MLA `KpoolTailManager` disables **fine-grained**
hits — only **block-aligned** tokens count (3584-token hybrid align).
`KpoolTail` already opts out of the hybrid min (1-block circular scratch).

`dflash` is `use_eagle()`. GLM never sets `is_eagle_group` (that annotator is
DeepseekV4-only), so stock HybridKVCacheCoordinator flagged **every** group.
MLA dropped its last 3584-token page, and the DFlash2 SlidingWindow group
re-aligned the min by another scheduler page. Overlay
`patch_hybrid_prefix_hit.py` flags only the drafter SWA group as EAGLE and
does **not** let that group shrink the MLA+mamba hit. Mamba stays in the min
(skipping a mamba miss is a correctness hole). Do not raise
`--max-num-batched-tokens` to “fix” APC.

**Historical pre-E2 receipts** (thinking off, temp 0, unique pads), 1M serve, **`MAX_NUM_BATCHED_TOKENS=2048`** (P1 keep; 3584/4096 reverted), **`DFLASH_DRAFT_TP=2`**. Idle 8k/16k/100k are the 2026-08-30 C4 keep A/B. 12k/256k/300k and the concurrent follow-ups are the prior TP=1 production ladder (chunk size does not change those hit counts). The MNBT=1024 baseline is `docs/cold-prefill.md`. Details: `docs/improve-prefill.md`.

| Turn | Hits | Compute | Prompt tok | TTFT | Prefill tok/s |
|---|---:|---:|---:|---:|---:|
| ~8k cold | 0 | 7995 | 7995 | **8.53 s** | **938** |
| ~8k follow-up | **7168** | 836 | 8004 | **1.30 s** | 6177 |
| ~12k cold | 0 | 11995 | 11995 | 12.96 s | **926** |
| ~12k follow-up | **10752** | 1263 | 12015 | **1.94 s** | — |
| ~16k cold | 0 | 15995 | 15995 | **16.45 s** | **972** |
| ~16k follow-up | **14336** | 1679 | 16015 | **2.18 s** | — |
| ~100k cold | 0 | 99995 | 99995 | **100.3 s** | **997** |
| ~256k cold | 0 | 255995 | 255995 | 263.2 s | **973** |
| ~300k cold | 0 | 299995 | 299995 | 318.9 s | **941** |
| 4× ~7.5k concurrent follow-ups | **7168 each** (28672 total) | rest | 7515 each | **1.86–2.50 s** | — |

An ~8k follow-up still reuses **7168 / 8004 ≈ 90%** of the prompt, not 46%. MNBT=2048 vs the 1024 ladder (draft TP=1): ~8k 10.36 s / 772 → 8.93 s / 895; ~100k 105.6 s / 947 → 102.5 s / 975; ~256k 273 s / 936 → **263 s / 973**; ~300k 323 s / 928 → **319 s / 941**. C4 keep (`DFLASH_DRAFT_TP=2`): ~8k **8.53 s / 938**; ~16k **16.45 s / 972**; ~100k **100.3 s / 997**. Coarser decode interleave (2k-token chunks vs 1k).

**Default from this checkout:** E2 fat kernel on (`EXL3_FAT_KERNEL=1`) and `MAX_NUM_BATCHED_TOKENS=7168`. The table above is the pre-E2 C4 keep at 2048; use the linked PR77 table for the current E2 cold-prefill baseline.
Hits work **below** UserHIJ’s 14,336-token floor (that floor is 896-chunk ×
2048-align LCM on a different geometry; this kit’s 3584 is 4×896). Isolation
held (`STILL_READY_S` / `STILL_C0`…`C3`). Idle chats are not reserved; after
the pool drains, a later turn of an old window prefills again. Concurrent
colds still serialize under `GLM53_MIXED_PREFILL_CHUNK=skip` (`Deferred`).

A later pre-E2 1M boot measured **1,670,157** tokens / **1.67×** / 638 GPU
blocks (padded slot-share still applied). The 900k process measured 1,754,237 /
690 blocks on the same recipe; the delta is leftover UMA, not a slot-share collapse.

Re-measure (see also `tests/bench_prefix_cache.py`):

```bash
# unique-content cold/warm pairs; hit ratio from the vllm:prefix_cache_*
# counter deltas; hit_efficiency scores against the 3584-token page model
python3 tests/bench_prefix_cache.py --runs 3
```

Note the page math: hits are **block-aligned to the 3584-token hybrid MLA
page**, so a warm prompt only ever reuses `floor(tokens / 3584) × 3584`
tokens — the 7168 / 10752 / 14336 hit rows above are exactly 2 / 3 / 4 full
pages. And since this build exposes **no cache-reset endpoint**, the bench
salts its filler content per invocation so every cold is genuinely cold.

## Quick start (2× Spark)

```bash
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git
cd GLM-5.3-Flash-EXL3-2x-DGX-Sparks
cp .env.example .env          # edit HEAD_IP / WORKER_IP / WORKER_USER if needed
./download.sh                 # optional: EXL3 + DFlash2 into the head HF cache only
./start.sh                    # pull public GHCR :exl3, download if missing, rsync, launch TP=2
```

First run of `./start.sh` copies `.env.example` → `.env` if missing. Prefix env
wins over `.env` (`SPEC_METHOD=dflash SKIP_DOWNLOAD=1 ./start.sh restart`).

`./start.sh` downloads weights automatically when the HF cache is incomplete
(120 shards of `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, falling back to
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` if the mirror is incomplete, plus DFlash2 when
`SPEC_METHOD=dflash`). `./download.sh` is the same Hub fetch **on this machine
only** — no docker, no SSH, no worker rsync. Use it to stage ~164 GiB before
the worker is ready. `REFRESH_WEIGHTS=1 ./download.sh` re-fetches.
Already present: both scripts skip. `./start.sh` still rsyncs the cache to the
worker unless `SKIP_SYNC=1`.

DFlash2 (`incoai/GLM-5.3-Flash-DFlash2`, ~2.3 GiB BF16, CC BY-NC-ND 4.0 research/eval)
is the default. Rollback:

```bash
SPEC_METHOD=mtp ./start.sh restart      # MTP k=2
```

`./start.sh` will:

1. Preflight docker/ssh/disk on both nodes
2. `docker pull` `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` (public; no login) on the head, then the same pull on the worker if GHCR is reachable — **unless** the local image's `glm53.recipe.stamp` does not match this checkout (Dockerfile/overlay change after `git pull`), in which case it rebuilds from this Dockerfile once. If the worker cannot pull, `docker save --platform linux/arm64 | ssh docker load`. `SKIP_PULL=1` keeps a local copy. `SKIP_BUILD=1` keeps GHCR even when the stamp drifts. `SKIP_SHIP=1` never copies.
3. Download the TR3 EXL3 repo into `$HF_HOME` / `~/.cache/huggingface` (~164 GiB, 120 shards) if missing. Same job as `./download.sh`, which stops here (head only).
4. `rsync` that cache to `${WORKER_HOME}/.cache/huggingface`
5. Start rank 1 `--headless` on the worker, rank 0 + API on the head
6. Poll `/health` (weight load + warmup is slow; `READY_TIMEOUT` default 3600s), then a **nonfatal** DFlash2/sampler shape sweep so the first client is not the first JIT on TP=2. `GLM53_BOOT_SHAPE_WARMUP=0` skips it.

The worker does not need GHCR access — start.sh pulls there when it can, otherwise it ships a single-platform tar over SSH.

```bash
./download.sh                              # head HF cache only (no worker); same as ./start.sh download
SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh     # weights already local on both nodes
SKIP_PULL=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart  # keep local image, no GHCR
# overlay/Dockerfile change after git pull rebuilds once; SKIP_BUILD=1 skips that
BUILD=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 ./start.sh restart  # force rebuild overlay + ship
./start.sh status
./start.sh logs                # head
./start.sh logs worker
./start.sh stop                # or ./stop.sh
```

### Experimental: 4× Spark (TP=4)

Untested here (no 4-Spark kit). Optional sibling of `./start.sh` — same image
and weights, does not change the supported 2× path. First run copies
`.env.tp4.example` → `.env.tp4` (gitignored). Stop with `./start-tp4.sh stop`;
`./start.sh stop` does not know ranks 2/3.

```bash
# edit WORKER2_IP / WORKER3_IP / CX7 pins in .env.tp4
./start-tp4.sh
./start-tp4.sh stop
./start-tp4.sh logs            # head; logs 1|2|3 for a worker rank
```

Do not pull `glm53-flash-sm121:v8` — that is the older NVFP4/Ray kernel.

API: `http://127.0.0.1:8888/v1` (LAN: `http://10.0.0.1:8888/v1`).
`/v1` is unauthenticated unless you set `VLLM_API_KEY` in `.env` (opt-in;
empty = no auth). vLLM reads the env var natively so the key never lands in
argv. `/health` and `/metrics` stay unguarded. Restart after setting it.
Clients then send `Authorization: Bearer <key>` on `/v1` (warmup already
picks it up via `GLM53_WARMUP_BEARER`).

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5.3-Flash-EXL3",
    "messages": [{"role": "user", "content": "hello!"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'

# with auth:
# curl ... -H "Authorization: Bearer $VLLM_API_KEY" ...
```

Thinking defaults on. Disable it with the **top-level** JSON field
`"chat_template_kwargs": {"enable_thinking": false}`. This closes the empty
thinking block in the generation prompt and omits the reasoning-effort hint.
Do not send a literal nested `extra_body` object over raw HTTP; `extra_body` is
an OpenAI Python SDK option that merges its contents into the top-level request.
The Hub `generation_config.json` stamps `temperature=1.0` / `top_p=0.95` unless
the request overrides. The launcher sets
`--chat-template /opt/glm53/chat_template.jinja` (checkpoint jinja is language-only).

Needs: Docker (no sudo) on both nodes, passwordless SSH head → worker,
`hf` / `huggingface-cli` + `curl` + `rsync` on the head, ~180 GiB free per
node for the first download. The GHCR image is public; login is only needed
if you hit anonymous pull rate limits (`GHCR_TOKEN` + `GHCR_USER`).
Mixed OS accounts: set `WORKER_USER` (this kit uses `zurih` on spark2).

NCCL cannot use the `10.0.0.x` loopback aliases — leave the CX7 pins unless
your cabling differs. `ncclCommInitRank` hangs without them.

## Running on a different 2×Spark kit

Independently reproduced on a second GB10 pair (2026-08-28) — decode within the
same bands (structured 38–62, prose 27.1) after three kit-specific adjustments
that are now documented/enforced:

- **NIC names differ per kit.** Set all four of `HEAD_CX7_IF/IB`,
  `WORKER_CX7_IF/IB` in `.env` (some pairs use the same names on both nodes,
  e.g. `enP2p1s0f1np1`/`roceP2p1s0f1`). Exporting generic
  `NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA` does **not** override the per-node values.
- **The RoCEv2 GID index is per-NIC** — each node needs the index carrying its
  own `::ffff:<ip>` entry, and an all-zero entry kills that rank ~60 s in with
  `ibv_modify_qp` errno 61. Preflight validates each rank against its own device
  and dumps both GID tables (0–7) when it refuses. If one index is valid on both
  nodes, set `NCCL_IB_GID_INDEX`; if the nodes need different indices, set
  `HEAD_GID` / `WORKER_GID` per rank (both default to `NCCL_IB_GID_INDEX`).

- **`GPU_MEM_UTIL=0.87` needs ≥105.9 GiB free *after* vLLM's own ~9 GiB
  init.** Nodes running resident services (dashboards, TTS, desktop) can miss
  it by well under 1 GiB and fail the startup memory check; `GPU_MEM_UTIL=0.86`
  with `MAX_MODEL_LEN=800000` (the previously published pair) fits with margin.
  If :8888 is taken on your head node, `PORT` moves the API cleanly.

## .env

| Knob | Default | What |
|---|---|---|
| `HEAD_IP` | `10.0.0.1` | this node, NCCL/vLLM master |
| `WORKER_IP` | `10.0.0.2` | other Spark |
| `WORKER_USER` | *(unset = `$USER`)* | SSH user on the worker |
| `WORKER_HOME` | `$HOME` if same user, else `/home/$WORKER_USER` | worker HF cache |
| `MODEL` | `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | Hub repo into the HF cache (mirror) |
| `MODEL_FALLBACK` | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` | Used if the mirror 404s or has fewer than 120 shards |
| `SERVED_MODEL_NAME` | `GLM-5.3-Flash-EXL3` | OpenAI `model` id (`/v1/models`) |
| `IMAGE` | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` | public GHCR tag. Rebuilt when the overlay recipe stamp drifts (`BUILD=1` forces; `SKIP_BUILD=1` keeps GHCR). `SKIP_PULL=1` skips pull |
| `GHCR_TOKEN` / `GHCR_USER` | *(unset)* | optional login if anonymous GHCR pull is rate-limited |
| `PORT` | `8888` | OpenAI API on the head |
| `VLLM_API_KEY` | *(unset)* | opt-in Bearer token for `/v1`. Empty = open API. `/health` stays keyless |
| `ABLIT` | `0` (off) | opt-in. `1` = apply o_proj edit at load (both ranks). Unset = stock weights |
| `ABLIT_METHOD` | `auto` | `auto` = transplant when `ablit/transplant/` is populated, else `proj` |
| `ABLIT_LAYERS` | `15-45` | inclusive range; `45` is the checkpoint MTP block |
| `ABLIT_DIRECTION` | `dealign` | proj-only: `dealign` \| `bf_oproj` \| path to a custom `.pt` |
| `ABLIT_ALPHA` | `3.0` | proj-only: projection scale |
| `ABLIT_INCLUDE_MTP` | `1` | also edit the MTP block's o_proj when it loads |
| `TP` / `NNODES` | `2` / `2` | do not change for this recipe |
| `QUANTIZATION` | `exl3` | overlay method; never `marlin` |
| `MTP_TOKENS` | `2` | MTP speculative tokens (`SPEC_METHOD=mtp`) |
| `SPEC_METHOD` | `dflash` | `dflash` / `mtp` / `none`. Rollback: `SPEC_METHOD=mtp ./start.sh restart` |
| `DFLASH_MODEL` | `incoai/GLM-5.3-Flash-DFlash2` | DFlash2 draft Hub repo (~2.3 GiB BF16) |
| `DFLASH_TOKENS` | `7` | DFlash2 speculative tokens (trained block 8) |
| `DFLASH_DRAFT_TP` | `2` | shard DFlash2 across TP (C4 keep: 8k 938 / decode 65.1). `1` = rank 0 only. Empty = inherit TP |
| DFlash2 draft KV | `auto` (bf16) | target stays `fp8`/`fp8_ds_mla`; dense draft has no MLA FP8 backend on SM121 |
| DFlash2 attention | *(unset)* | SM121 picks FLASH_ATTN for non-causal SWA. Do not pin `TRITON_ATTN` |
| `ENFORCE_EAGER` | `0` | CUDA graphs; MTP capture `1 2 3 4 6 8 12`, DFlash2 `1 2 4 8 16 24 32` |
| `EXL3_FUSED_MOE` | `1` | `exl3_moe` per layer; `0` = LinearEXL3 loop |
| `EXL3_FAT_KERNEL` | `1` | [PR77 E2 fat-expert prefill kernel](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/77) (implies batched+sorted). `0` = legacy fat path |
| `KV_CACHE_DTYPE` | `fp8` | packed `fp8_ds_mla`; not `nvfp4`, not bf16 |
| `GPU_MEM_UTIL` | `0.87` | GB10 UMA budget. Latest validated 7168/rightsize pool: **1,243,902 tokens / 1.24×** at 1M |
| `MAX_MODEL_LEN` | `1000000` | default context. Pool size varies with MNBT, activation/graph reservations and hybrid block geometry |
| `MAX_NUM_SEQS` | `4` | decode batch; MTP adds k+1 tokens/seq |
| `MAX_NUM_BATCHED_TOKENS` | `7168` | current maintainer default at `MAX_NUM_SEQS=4`. MNBT 2048 was the clean PR77 A/B configuration and the best measured balance on an independent `MAX_NUM_SEQS=16` geometry. Tune per deployment; change after a repeated same-kit comparison |
| `GLM53_MIXED_PREFILL_CHUNK` | `skip` | do not mix a peer prefill into a decode step (issue #6). `N>0` = cap tokens; `0` = off. Solo prefill stays MNBT (7168) |
| `GLM53_SUPPRESS_STOPS_IN_REASONING` | `1` | ignore client `stop` strings until `</think>` (thinking-on default) |
| `GLM53_INDEXER_WORKSPACE` | `stock` | sparse-indexer prefill gather workspace. `stock` = `max_model_len * 40` entries (**5036.40 MB** locked at 1M — measured, `VLLM_DEBUG_WORKSPACE=1`). `rightsize` = the legal per-step maximum `min(MAX_NUM_SEQS, MNBT) * cdiv(MAX_MODEL_LEN + k, index_kpool)` = 126 MB at `MAX_NUM_SEQS=4` / 504 MB at 16, so **~+26–28% KV**. Opt-in; see [docs/DESIGN-indexer-workspace.md](docs/DESIGN-indexer-workspace.md) |
| `GLM53_SPINWAIT_MS` | `stock` | SpinCondition reader busy-loop window. `stock` preserves vLLM's 1 s default; `1..1000` selects milliseconds. A frozen TP=2 sweep selected `16` (+0.95% median decode vs stock, 85.3% less active EngineCore CPU) |
| `GLM53_BOOT_SHAPE_WARMUP` | `1` | after `/health`, burn DFlash2 BLOCK / sampler / kpool shapes (nonfatal) |
| `TRITON_HOST_CACHE` / `TILELANG_HOST_CACHE` | `$CACHE_ROOT/triton` / `tilelang` | persist JIT caches across container recreate |
| `LANGUAGE_MODEL_ONLY` | `0` | load vision tower (image + video) |
| `SKIP_MM_PROFILING` | `1` | skip max-size MM dummy at init (OOM otherwise) |
| `LIMIT_MM` | `{"image":4,"video":1}` | `--limit-mm-per-prompt` |
| `HEAD_CX7_IF` / `WORKER_CX7_IF` | `enp1s0f1np1` / `enp1s0f0np0` | NCCL sockets |
| `HEAD_CX7_IB` / `WORKER_CX7_IB` | `rocep1s0f1` / `rocep1s0f0` | NCCL HCAs |
| `USE_HOST_NCCL` | `0` | image nvidia-nccl; host preload duplicates DeepEP |

## Image / overlay

```bash
docker build -t glm53-flash-sm121:local .
# or: BUILD=1 ./start.sh
```

`./start.sh` **rebuilds** from this Dockerfile when the image label `glm53.recipe.stamp` does not match the current overlay/Dockerfile hash — that is what makes a `git pull` pick up `exl3_fat_gemm` instead of staying on the public GHCR tag (which predates E2). `SKIP_BUILD=1` keeps GHCR. `BUILD=1` forces a rebuild. `SKIP_PULL=1` skips `docker pull` only.

After CUDA compile, Python overlay edits (`overlay/exl3.py`, tests) are a cheap layer so they do not rebuild `exllamav3_ext`.

| Path | Role |
|---|---|
| `Dockerfile` | NoPE sparse-MLA patches + EXL3 install (`sm_121a`) + self-check |
| `overlay/exl3.py` | `Exl3Config` / packed load / TP shard / fused `exl3_moe` apply / E2 fat kernel |
| `overlay/exl3_fat_gemm.cu` | additive `exl3_fat_gemm` / `_scatter` compiled into `exllamav3_ext` |
| `overlay/patch_exl3_ext_aarch64.py` | stub AVX CPU allreduce so the ext builds on GB10 |
| `overlay/patch_model_overrides.py` | `"exl3"` in ModelConfig overrides |
| `tests/test_exl3_overlay.py` | registry, TP shard, `sm_121a` cubin, fused vs loop GEMM, `EXL3_FUSED_MOE=0` |
| `tests/bench_decode.py` | streaming decode + coherence; `--structured` is the count-1→200 median |
| `start.sh` / `stop.sh` / `download.sh` | 2-node launch; Hub fetch on the head only |
| `start-tp4.sh` / `.env.tp4.example` | experimental 4-node TP=4 launch; knobs stay out of `.env` |
| `files/chat_template.jinja` | GLM-5.3 MM template (`<|image|>` / `<|video|>`); checkpoint jinja is language-only |
| `overlay/qwen3_dflash2.py` | DFlash2 draft (grouped conv + candidate selector) |
| `overlay/dflash2_speculator.py` | DFlash2 selector walk (V2 speculator) |
| `overlay/patch_dflash2.py` | registry + `decoder_layer_cls` + speculator dispatch + draft KV `auto` on MLA/FP8 |
| `overlay/patch_glm_eagle3.py` | Glm5Next EAGLE3 aux-hidden layers (mHC `hc_post` + contract) |
| `overlay/patch_glm5_drafter_group.py` | GLM KV fast path + DFlash2 padded slot-share (`block=64`, `page_size_padded=mla_page`); runtime-mounted by `start.sh` (`DRAFTER_PATCH_HOST`) |
| `overlay/patch_glm_video_placeholders.py` | align video timestamp blocks to encoder `grid_t` |
| `overlay/patch_suppress_stops_in_reasoning.py` | fail-closed detokenizer guard: client `stop` dormant until `</think>` |
| `overlay/patch_scheduler_decode_floor.py` | skip (or cap) peer prefill while another seq is decoding |
| `overlay/patch_xgrammar_termination.py` | source-exact vLLM #52805/#53046 backports; stop at termination and validate post-reasoning speculative drafts before FSM advance |
| `tests/test_xgrammar_termination.py` | exact two-file patch, idempotence, cross-file fail-closed drift, termination/rollback/reset and post-reasoning draft behavior, launcher wiring |
| `overlay/patch_kpool_tail_slotmap.py` | clamp KpoolTail one-block circular slot mapping; identity for other KV groups |
| `tests/test_kpool_tail_slotmap.py` | circular addressing math, exact kernel patch, idempotence, fail-closed drift, launcher wiring |
| `overlay/patch_indexer_workspace.py` | opt-in `GLM53_INDEXER_WORKSPACE=rightsize`: size the sparse-indexer prefill workspace to the legal per-step maximum instead of `max_model_len * 40`; boot-time compress-ratio cross-check |
| `tests/test_indexer_workspace.py` | sizing formula (MNBT/`max_num_seqs`/spec-token edge cases, stock clamp), chunk-list equivalence vs stock by exhaustion, exact three-site patch, idempotence, fail-closed drift, launcher wiring |
| `overlay/patch_spinwait.py` | opt-in numeric `GLM53_SPINWAIT_MS`: fail-closed runtime patch of SpinCondition's reader busy-loop window on both ranks |
| `tests/test_spinwait_patch.py` | numeric contract, exact patch, idempotence, drift rejection, mode preservation, pyc cleanup, and launcher/build wiring |
| `overlay/ablit_runtime.py` | load-time o_proj transplant / projection (`ABLIT=1`); no-op when off |
| `overlay/patch_ablit.py` | install the load_weights hook; bind-mounted and run on both ranks |
| `ablit/` | direction vectors + `LAYER_MAP.json` from drowzeys' published recipe; `fetch_transplant.py` + `transplant/` for the donor o_proj byte-copy |
| `tests/test_ablit.py` | recipe integrity, orthogonalization math, TP-shard equivalence, transplant byte-copy + TP slice, hook gating |
| `scripts/boot-shape-warmup.sh` | post-`/health` DFlash2 k=7 BLOCK ladder + sampler/kpool arms |

Image-build runs `EXL3_SELFCHECK_GPU=0`. `./start.sh` runs the GPU self-check
(`docker run --gpus all`) before shipping unless `SKIP_OVERLAY_VERIFY=1`.

## Do not

- Destroy HF weights, requantize, or `docker rm` HF caches. `REFRESH_WEIGHTS=1 ./download.sh` only if you intend to re-fetch
- `--moe-backend marlin`, NVFP4 weights, or `glm53-flash-sm121:v8` as this serve
- qemu / amd64 / `cstechdev/vllm:glm53-flash-nope-sm120-*` / verdictai SM120 B12X
- `--kv-cache-dtype nvfp4` or bf16 (no sparse-MLA kernel)
- `"attention_backend": "TRITON_ATTN"` in speculative-config (causal-in-block on this image)
- Change TP, CX7 pins, or `USE_HOST_NCCL` unless you are re-plumbing NCCL
- Force-push

## License

This repository (serve scripts, overlay, docs) is **MIT**. The EXL3/TR3
checkpoint stays [ShapleyMCG License 1.0](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/blob/main/LICENSE)
(unmodified upstream LICENSE; also on
[brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)).
DFlash2 stays [CC BY-NC-ND 4.0](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).

## Credits

- **EXL3/TR3 weights:** [brandonmusic](https://huggingface.co/brandonmusic) —
  [GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
  (uniform-K4 routed-experts, ShapleyMCG License 1.0). Public mirror for this
  recipe: [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
- **EXL3 format / kernels:** [turboderp](https://github.com/turboderp-org/exllamav3) (ExLlamaV3)
- **Base model:** [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
- **DFlash2 drafter:** [IncoAI](https://huggingface.co/incoai) —
  [GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (CC BY-NC-ND 4.0, research/eval)
- **KLD panel:** [malaiwah](https://huggingface.co/malaiwah) —
  [discussion #1](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/discussions/1#6a9144846b0bdba943bfe86f)
- **Abliteration recipe / direction artifacts:** [drowzeys](https://huggingface.co/drowzeys) —
  [keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock)
