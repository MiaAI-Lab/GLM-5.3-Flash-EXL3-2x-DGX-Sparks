<h1 align="center">GLM-5.3 Flash EXL3 for 2-4x DGX Sparks</h1>

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
A **3×** sibling is `./start-tp3.sh` on the same image and weights (see
[3× Spark (TP=3)](#3x-spark-tp3)). Served model id: **`GLM-5.3-Flash-EXL3`**. EXL3/TR3 quant by
[brandonmusic](https://huggingface.co/brandonmusic).

Optional TP3 contribution for evaluation: [cooperative ABI2, 64-row support,
FlashKDA and combined-profile measurements](docs/tp3-throughput-results.md).
Historical measurements and pending validation of the upstream-based branch
are documented separately; existing defaults are unchanged.

This is **EXL3 weights + fp8 KV** on GB10. Do not pass `--moe-backend marlin`.
The Hub card on brandonmusic (TP2/EP2/DCP2 + calibrated NVFP4 MLA KV) is the SM120 B12X
image (`verdictai/glm53-flash-exl3-k4:…-v84-dflash2`), not this overlay. Target KV
stays packed **`fp8_ds_mla`**. Speculator is **DFlash2 k=7**
([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2));
draft attention is **FLASH_ATTN** (do not pin `TRITON_ATTN` — that mask is causal
inside the draft block on this image and collapses later-position accept).

Release notes from the initial 1.0.0 recipe through **1.6.0** are in
[CHANGELOG.md](CHANGELOG.md).

## Cold prefill (E3 grouped MoE, this kit, 2026-09-07)

`EXL3_FAT_GROUPED=1` (launcher default since 2026-09-07) replaces the E2 per-expert host loop for "fat" experts with three
GPU-driven launches per MoE layer (gather, gate/up + SwiGLU, down + scatter) built from
device-side segment tables — no per-expert launches, no weight repacking, no host sync
(`overlay/exl3_fat_moe.cu`, Fable's design, E2 rounding boundaries restored).

**Latest run — 2026-09-07, measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash).**
Shipped E3 image `glm53-flash-sm121:e3-20260907` at `MAX_MODEL_LEN=900000`, `GPU_MEM_UTIL=0.86`,
`GLM53_INDEXER_WORKSPACE=rightsize`, `EXL3_TEMP_ROWS_FUSED=32`, MNBT 7168, `MAX_NUM_SEQS=4`,
DFlash2 k=7 draft TP=2, C4, thinking off.

| Prompt | Prompt tok | TTFT | Prefill tok/s |
|---|---:|---:|---:|
| ~8k | 8,221 | 5.51 s | **1,492.1** |
| ~16k | 16,411 | 10.56 s | **1,553.7** |
| ~32k | 32,797 | 22.96 s | **1,428.2** |
| ~64k | 65,566 | 41.31 s | **1,587.0** |
| ~128k | 131,101 | 83.95 s | **1,561.7** |
| ~256k | 262,173 | 172.84 s | **1,516.8** |

**A/B receipt vs E2 (2026-09-06 overnight run).** Measured
on this kit: **MAX_MODEL_LEN=500000**, `GPU_MEM_UTIL=0.84`, `GLM53_INDEXER_WORKSPACE=rightsize`,
`EXL3_TEMP_ROWS_FUSED=32`, MNBT 7168, DFlash2 k=7 draft TP=2, C4, thinking off, unique cold
prompts (0 prefix hits). Prefill tok/s = prompt tokens / TTFT (client side).

| Prompt | Prompt tok | TTFT | Prefill tok/s | E2 before (tok/s) | Gain |
|---|---:|---:|---:|---:|---:|
| ~8k (first request after boot) | 8,221 | 8.23 s | 999 | 1,112 warm / 418 first | one-time allocation; warm ~1,500 |
| ~16k | 16,417 | 10.41 s | **1,578** | 1,155 | **+37%** |
| ~32k | 32,798 | 20.20 s | **1,624** | ~1,160 | **+40%** |
| ~64k | 65,565 | 40.06 s | **1,637** | ~1,170 | **+40%** |
| ~128k | 131,101 | 80.46 s | **1,629** | 1,148–1,183 (100k) | **+38–42%** |
| ~256k | 262,173 | 166.4 s | **1,576** | 1,087 | **+45%** |

E2 column: the 2026-09-01 E2 receipts (`logs/pr77-build-kernel-20260901`) and the matched
E2 baselines of the 2026-09-06 overnight run (`logs/overnight-20260906T164059Z/exp500k`,
A/B/A at 500k: 8k 1,112 / 16k 1,154 / 100k 1,183, repeats within 0.5%); 32k/64k interpolated.
Decode is unchanged (E3 never runs on decode-sized steps; structured / prose / code within
run-to-run spread). Isolated layer bench: 7168-token MoE layer 77–91 ms (E2) → 31 ms (E3),
numerically indistinguishable from E2 vs the LinearEXL3 reference. Full receipts, gates and
the memory caveat: `logs/overnight-20260906T164059Z/report.md`.

**Context and headroom:** the shipped default is now **850k / util 0.85 / rightsize** (boots with ~0.5 GiB of KV margin on the head; a 256k prefill ran at 900k / 0.87 with driver retries but no failure). E3 keeps a 560 MiB fat-row scratch that vLLM's profile run charges to
the KV budget, so at 1M / util 0.87 the pool no longer fits one 1M request on this kit
(needs 14.52 GiB). 500k needs 10.98 GiB and boots reliably at 0.84 (1.2× at 500k). Prompts
≥ ~100k are near the head's host-memory limit at any setting (a 256k prefill at util 0.87
with zero MemAvailable crashed the head on 2026-09-06); the 256k row above ran at 0.84.

## Decode (this kit, 2026-08-28)

Official numbers: sparkDash Decode bench, DFlash2 k=7, **Structured** (count 1→200) and **Code** (`clamp_00`…`clamp_49`) — same high-accept regime. Temp **0**, thinking **off**, 400 tokens, CUDA graphs, fused EXL3 MoE. Prompt types, not grammar / schema. Stream tok/s is per request; aggregate is all streams.

| Concurrency | TTFT | Stream tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| **×1** | **719 ms** | **62.9** | **62.9** |
| **×2** | 6.62 s | 51.7 | 103.3 |
| **×4** | 6.30 s | 37.1 | 146.5 |

That 2026-08-28 decode serve used `--max-model-len 1000000` with a **1,754,237-token** KV pool. These runs are warm / empty KV — they do not need a filled 1M cache.

**Prose** (sparkDash Decode bench, prose prompt type, 2026-09-17, thinking
**off**) on this 2× kit (`GLM53_ADAPTIVE_K=ema`, `GLM53_DENSE_FP8=dense,kda`,
cooperative MoE overlay, 850k context, KV pool capped at 14 GiB). Stream is
per request; aggregate is all streams.

| Concurrency | TTFT | Stream tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| **×1** | **333 ms** | **36.1** | **37.1** |
| **×2** | 365 ms | 25.0 | 51.1 |
| **×3** | 405 ms | 22.3 | 65.8 |
| **×4** | 401 ms | 19.4 | 75.3 |

The 2026-09-08 adaptive-k + dense-FP8 table (no coop overlay in that write-up)
was ×1 **32.1** / ×2 **22.1** stream (**41.2** agg), TTFT 268 / 399 ms.

### Opt-in cooperative decode MoE

An optional decode-only cooperative EXL3 MoE overlay lives in
[`extensions/cooperative_moe/`](extensions/cooperative_moe/). It specializes
Turboderp's two-stage kernel for this recipe (H=4096, TP2 local I=1024, top-k 8,
K4 MCG, 1–32 rows) and does **not** replace E3 prefill. Default image, launcher,
and `overlay/exl3.py` stay stock until you select a generated overlay with
`EXL3_OVERLAY_HOST`.

Do not load the DS4.1 cooperative `.so` here. Serving measurements vs the
tables above (prose ×1 **37.1** / ×2 **51.1** agg, structured ×1 62.9) are recorded
after the GPU gate in [`docs/cooperative-moe.md`](docs/cooperative-moe.md).
Live operator handoff (geometry 1, rollback, pins):
[`docs/cooperative-moe-handoff.md`](docs/cooperative-moe-handoff.md).
The two-node opt-in and rollback sequence is
[`docs/cooperative-moe-quickstart.md`](docs/cooperative-moe-quickstart.md).

### Faster prose decode (opt-in, 2026-09-08)

Two decode speed-ups ship in the overlay, both **off by default** (matched A/B/A at 131k and 850k, 8 runs per prompt, bootstrap 95 % CI; receipts in `logs/overnight-decode-20260907T224521Z/`):

- **Adaptive verification length** (`GLM53_ADAPTIVE_K=ema`): the DFlash2 drafter still proposes 7 tokens, but the scheduler verifies only a per-step prefix (2, 4 or 7) chosen from a running average of how many drafts have been surviving, batch-uniform so every decode step keeps its FULL CUDA graph. Lossless at temperature 0. Measured vs stock k=7 (8 runs/prompt, 131k and 850k): Silk Road essay +21 %, sky/sunset +13 %, hash-map +10 %, code +5–15 %, counting unchanged.
- **FP8 weight-only dense projections** (`GLM53_DENSE_FP8=dense,kda`): KDA and dense-MLP projections quantised per output channel to FP8 at load and run through the Marlin kernel, ~11 ms less per step on everything (+10 % on counting, prose +12–19 % alone, **+37 % on hard prose stacked with adaptive-k**). PROVISIONAL: it changes target numerics by FP8 rounding (KL proxy vs stock 0.002–0.013 nats/position, argmax agreement 94–100 %; no full KLD panel yet).

Turn on (no rebuild; the patches apply at container start on both nodes):

```bash
# .env
GLM53_ADAPTIVE_K=ema
GLM53_ADAPTIVE_K_SET=2,4,7
GLM53_DENSE_FP8=dense,kda            # drop this line to keep BF16 dense weights (lossless config)
EXTRA_ARGS="--kv-cache-memory-bytes 15032385536"
```

For DFlash with adaptive-k enabled (`ema`/`on`/`1`, case-insensitive, surrounding
whitespace ignored), the launcher supplies the capture-size list automatically.
It combines stock captures with multiples of the configured `GLM53_ADAPTIVE_K_SET`
query lengths (`k + 1`, bounded by `DFLASH_TOKENS + 1`), including the full draft
length, through `MAX_NUM_SEQS`. Explicit `--cudagraph-capture-sizes` in `EXTRA_ARGS`
always wins; eager mode and non-DFlash capture defaults are unchanged.

then `./start.sh restart`. The capture-size list is required for adaptive-k (multiples of 3, 5 and 8 up to 4 requests; the stock `1 2 4 8 16 24 32` misses the 3- and 5-token shapes). The KV cap turns FP8's freed GPU memory into host headroom instead of a bigger pool: uncapped, the head dropped to ~1.5 GiB MemAvailable at 850k. 14 GiB leaves an 883,552-token pool (1.04x of 850k) and ~5 GiB free; 15 GiB buys 1.11x but measured only 0.8-2.2 GiB free under load, which is not enough margin on this UMA. Do not go much lower at 850k either — the boot refuses a pool that cannot hold one max-length request (13 GiB is ~820k tokens). Verify after boot: `docker logs glm53-exl3-head | grep -a "adaptive-k\|dense fp8"` should show `uniform decode graph query lens: [3, 5, 8]` and `dense fp8 groups: dense,kda`, and with `ABLIT=1` the line `ABLIT_METHOD=auto -> transplant` (a missing `ablit/transplant/` silently falls back to the projection edit, which garbles sampled output). A running server can be retuned without a reboot through `~/.cache/vllm-glm53-flash/glm53_adaptive_k.json` (`{"mode":"ema","set":"2,4,7","margin":1.0}`; `{"mode":"off"}` restores k=7). Live sparkDash prose numbers with both on are in the table above.

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

For keyed servers, export `VLLM_API_KEY` before running the decode benchmark.
It sends Bearer auth on completion requests; a non-empty `API_KEY` takes
precedence over `VLLM_API_KEY`. Unset or empty values fall through, and no
header is sent when both are unset or empty. `/health` and `/metrics` stay keyless.

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
| Image | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` FROM `vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c0293…` (arm64, CUDA 13.0). InstantTensor baked in; `--load-format instanttensor` is the default. Wheel-less `:exl3` still exists |
| Executor | `mp`, `--nnodes 2`, `--tensor-parallel-size 2` |
| Head | this machine, `HEAD_IP=10.0.0.1`, container `glm53-exl3-head` |
| Worker | `WORKER_USER@WORKER_IP` (this kit: `zurih@10.0.0.2`), `--headless`, `glm53-exl3-worker` |
| Fabric | CX7 QSFP: `enp1s0f1np1`/`rocep1s0f1` ↔ `enp1s0f0np0`/`rocep1s0f0`. Image NCCL (`USE_HOST_NCCL=0`) |
| Attention | `FLASHINFER_MLA_SPARSE_SM120` (NoPE MLA padded into GLM_NSA 576-wide) |
| KV | `--kv-cache-dtype fp8` → packed **`fp8_ds_mla`** (target). Draft DFlash2 KV is `auto`/bf16. Latest validated 7168/rightsize pool: **1,243,902** tokens / **1.24×** at 1M. `--enable-prefix-caching` (block-aligned hits; see Prefix caching) |
| Context | **850k** default since 2026-09-07 (E3; `EXL3_FAT_GROUPED=0` fits 1M again). KV pool is **~1M tokens** (measured 989,010 at 900k / util 0.85, and 1,023,626-1,084,615 at 900k / util 0.86; pool size varies ±0.4 GiB between identical boots, and the 850k default is not yet measured). Pre-E3 1M receipts: live pool **1,754,237** tokens (1.75×) / 690 GPU blocks / 18.67 GiB at MNBT 2048; latest validated 7168/rightsize E2 pool at 1M **1,243,902** tokens / **1.24×**. Pool size varies with MNBT, activation/graph reservations and hybrid block geometry. Padded slot-share is why 1M allocates; the old 900k cap was 1.95× on this same pool. Do not drop to 256k to “free” slots (hybrid mamba + DFlash window block-id demand is mostly length-independent) |
| Experts | packed trellis + suh + svh + mcg, codebook MCG, **one fused `exllamav3_ext.exl3_moe` launch per layer** |
| Dense / shared / attn / embed / lm_head | native (unquantized) |
| Tools / reasoning | `--tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45` |
| Graphs | on (`ENFORCE_EAGER=0`) — MTP capture `1 2 3 4 6 8 12`; DFlash2 capture `1 2 4 8 16 24 32` |
| Spec | **DFlash2 k=7** (`incoai/GLM-5.3-Flash-DFlash2`); draft KV `auto`/bf16, draft TP=2, FLASH_ATTN. Rollback `SPEC_METHOD=mtp` |
| Vision | on (`LANGUAGE_MODEL_ONLY=0`) — image + video, `--limit-mm-per-prompt {image:48,video:1}`, `--mm-processor-kwargs {max_image_tokens:2048}`, `--mm-processor-cache-gb 1`, `--skip-mm-profiling`. Stateless chat clients resend every image each turn, so once a session holds more images than this ceiling, later turns are rejected — see `docs/images-per-request.md` |
| Ablit | **off** (`ABLIT=0`). Stock `o_proj`. Set `ABLIT=1` to enable; see [Abliteration](#abliteration-ablit1) |

Kernels: `TORCH_CUDA_ARCH_LIST=12.1a`. ExLlamaV3 pin `c5d9c657` (0.0.43) exposes
`exl3_moe` / `exl3_moe_max_concurrency`; aarch64 CPU allreduce stubs in
`overlay/patch_exl3_ext_aarch64.py`.

## Abliteration (`ABLIT=1`)

**Off by default.** With the default checkpoint, unset or `ABLIT=0` serves
stock `o_proj`. The flag only controls the load-time edit; it does not identify
whether a selected checkpoint was edited before download. `ABLIT=1` applies
refusal-direction ablation at weight load. Nothing on disk is rewritten.

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

Disable with `ABLIT=0` or leave it unset. The hook then leaves the selected
checkpoint unchanged. No rebuild is needed; the launcher bind-mounts the
artifacts and hook into both containers on every start.

| Knob | Default | What |
|---|---|---|
| `ABLIT` | `0` (off) | `1` = apply the o_proj edit at load on both ranks. `0` leaves the selected checkpoint unchanged |
| `ABLIT_METHOD` | `auto` | `auto` = transplant when `ablit/transplant/` is populated, else `proj` \| `transplant` \| `proj` |
| `ABLIT_LAYERS` | `15-45` | inclusive range; `45` is the checkpoint MTP block |
| `ABLIT_ALPHA` | `3.0` | proj-only: projection scale (`1.0` = plain projection) |
| `ABLIT_DIRECTION` | `dealign` | proj-only: `dealign` \| `bf_oproj` \| path to a custom direction `.pt` |
| `ABLIT_INCLUDE_MTP` | `1` | also edit the MTP block's o_proj when it loads (`SPEC_METHOD=mtp`) |

### Prebuilt abliterated checkpoint

To use the published EXL3 checkpoint instead of editing weights at load time:

```bash
./start-abliterated.sh download
./start-abliterated.sh restart
```

The preset selects
[`bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit`](https://huggingface.co/bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit)
at commit `14858211ed81d7fa773f8a0db02f38f36d230252`. It sets the fallback to
the same repository, so an incomplete download fails instead of starting the
original model. It also forces `ABLIT=0`; these weights already contain the
Keys transplant at layers 15-43 and MTP layer 45, while layer 44 remains from
the EXL3 parent. Applying the runtime edit again would produce a different
model.

The preset does not rewrite `.env`. DFlash2, vision, context, API naming, and
other serve settings still come from the regular launcher. Run
`./start.sh restart` to return to the model configured in `.env`.

The preset is fail-closed on both nodes. It requires the pinned snapshot's
published inventory — 120 safetensors shards, each link followed to a real
blob file, plus `config.json` and the tensor index — and never falls back to
`refs/main` or to another complete revision in the cache. A caller
`EXPECTED_SHARDS` cannot lower that requirement, and `SKIP_SYNC=1` still
checks the worker's copy. With `NFS_SHARE=1` the ranks read the head's
verified tree instead of a local copy.

The author reported that a staged-checkpoint TP=2 deployment at runtime commit
`6599585` passed all 10 functional checks. These included a
synthetic-image request, 9,128-token retrieval, three concurrent requests,
streaming, tools, and active DFlash drafting. The donor's Refusal32 script
reported 32/32 bypass, 0 refuse, and 0 garble in one greedy thinking-off run.
The model card records the method, integrity checks, and measured quality
tradeoffs. Teacher KLD was higher than the original model, so this is an
alternative checkpoint, not a quality-equivalent replacement.
Those historical observations do not qualify this integrated launcher or
checkpoint quality/boot behavior. The preset's local file-count and sidecar
checks do not verify shard hashes, tensor contents, or checkpoint integrity.

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
Default is **850k** (E3; was 1M). Do **not** drop `MAX_MODEL_LEN` to 256k to “free” slots —
logged tokens ≈ concurrency × that cap, and the hybrid floor then shrinks the
pool.

Keep **`SKIP_MM_PROFILING=1`** — a max-size image+video dummy profile OOMs this UMA.
The cost of that is permanent: **nothing is reserved for the vision tower**, so every
multimodal token is encoded out of memory the model has already spent. `LIMIT_MM` is a
validation ceiling only, and it has to be a ceiling this kit can actually encode.

On **2026-09-14** it was not. An 11.9 MB video attached in a chat reached vLLM as ~33
separate full-res *image* items — clients decompose video, and vLLM never splits one
video into multiple encoder items. At the checkpoint's `max_image_tokens=8000` each
frame cost ~7.2k tokens, the prompt reached **236,544 tokens** of vision encode, Node 0
fell to **44 MB free**, and the kernel killed `VLLM::Worker_TP` (`EngineCore encountered
a fatal error` → engine dead). File size is not the signal: 11.9 MB of H.264 is ~33
decoded frames, and a frame costs the same as a full-page image.

Three caps bound it, and all three are tunable:

| knob | default | effect |
|---|---|---|
| `MM_IMAGE_TOKENS` | `2048` | per-image budget. A 1080p frame measures **2691** tokens uncapped, **2040** at 2048, **1008** at 1024. Also fixes a latent hazard: the checkpoint's 8000 exceeds `MAX_NUM_BATCHED_TOKENS=7168`, which is vLLM's encoder cache size, so a full-res image could not fit the cache. |
| `LIMIT_MM` | `{"image":48,...}` | worst case 48 × 2048 = **98k** tokens, vs 800k before. Past the cap the API returns HTTP 500 `At most N image(s) may be provided in one prompt` — an error, not a dead engine. |
| `MM_PROCESSOR_CACHE_GB` | `1` | vLLM holds 4 GiB of host RAM for processed media by default; on UMA that is 4 GiB the model cannot have. `0` disables it. |

A 33-frame attachment now costs **67,320** tokens and serves. Raise `MM_IMAGE_TOKENS`
toward 7168 if you need document-grade detail from single images, and watch
`MemAvailable` on the head while you do.

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
pages. The bench POSTs `/reset_prefix_cache` between colds when that route is
enabled (`GLM53_EXPOSE_CACHE_RESET=1`; opt-in, see the API surface notes
below) and salts its filler content per invocation on top — repeated runs
stay genuinely cold even with the reset route off.

## API surface notes (this build, 2026-08-29)

Two things that cost us time (#31), documented so the next person does not
chase ghosts:

**Cache reset.** `overlay/patch_cache_reset.py` mounts the upstream dev
cache router on the head API server, so a genuinely cold prefix cache no
longer needs a container restart. It is **opt-in**: the patch is always
applied, but the router is only attached when `GLM53_EXPOSE_CACHE_RESET=1`
is exported for the launcher (default `0`):

```bash
GLM53_EXPOSE_CACHE_RESET=1 ./start.sh restart   # then:
curl -s -X POST http://127.0.0.1:8888/reset_prefix_cache    # -> {"success": true}
```

It returns `{"success": bool}` and reports `false` while blocks are still
held (running requests, in-flight async KV offload) — retry after they
drain. Unset (the default) leaves the stock surface, where a restart is the
only reset; `VLLM_SERVER_DEV_MODE=1` still mounts the whole dev set
(`/sleep`, `/rlhf`, `/rpc`, `/server_info`) if ever needed.
Auth caveat: the bearer middleware only guards `/v1`, `/v2`, `/inference`,
`/cohere` (upstream `GUARDED_PREFIX`), so root-mounted routes — the stock
`/tokenize` / `/detokenize` and the cache-reset routes — answer without the
key even with `VLLM_API_KEY` set. That is why the exposure is opt-in: leave
`GLM53_EXPOSE_CACHE_RESET` unset (0) on kits that serve untrusted clients.

**Tokenize.** It is mounted at the **root** (`/v1/tokenize` is 404) and the
request validates `prompt` (or `messages` for the chat shape), not `text`:

```bash
curl -s http://127.0.0.1:8888/tokenize -H 'Content-Type: application/json' \
     -d '{"model": "GLM-5.3-Flash-EXL3", "prompt": "hello world"}'
# -> {"count":2,"max_model_len":1000000,"tokens":[14978,1879],"token_strs":null}
```

### Optional sparse retention and DFlash replay

`GLM53_APC_RETENTION_INTERVAL_SWA` controls the DFlash2 drafter separately
from target retention. Empty inherits the global retention policy and keeps
ordinary eviction priority. Explicit `0` retains reachable drafter boundaries
and makes cached draft-only blocks lower priority than target cache blocks.
Sparse target Mamba state also retains the prior replay boundary; a cached
DFlash window is reused only after successful EAGLE verification, otherwise
the request backs up to rebuild its window.

**Sparse retention is not a general performance upgrade.** On one 2× Spark
deployment, explicit global/SWA `0/0` retained four independent 210K histories
for ~2.5 s revisits. In a matched 128K comparison, however, an edit at 90%
took 112.49 s instead of 14.70 s, and a branch at 90% took 99.89 s instead of
3.35 s: the sparse policy reused zero tokens where the old runtime reused
111,104. All tested answers were correct. These are sequential histories,
not four simultaneously active 210K streams.

The global launcher spelling is `GLM53_APC_RETENTION_INTERVAL` (leave unset
for a dense MLA/mamba grid). `start.sh`, `start-tp3.sh`, and `start-tp4.sh`
all forward `GLM53_APC_RETENTION_INTERVAL_SWA`.

Both retention knobs remain unset by default. Keep that default unless the
tradeoff fits the workload. SWA-only sparse retention with a dense target is
a separate configuration; the all-zero results do not qualify it. See the
[protocol, raw measurements, and limitations](docs/apc-retention-qualification.md)
before selecting a policy or a cache budget for another kit.

## Existing installs: pull the InstantTensor image

`main` now defaults to
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`.
**`git pull` does not switch the running containers or a leftover `.env`.**
If `IMAGE` is still `:exl3`, or `.env` has `SKIP_PULL=1`, you stay on the
wheel-less image and InstantTensor will not load.

1. Update the kit:

```bash
git pull
```

2. Set these lines in `.env` (already the default in `.env.example`; do
   the same in `.env.tp3` / `.env.tp4` if you use those):

```
IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor
LOAD_FORMAT=instanttensor
```

3. Pull that tag on the head and restart. `SKIP_BUILD=1` keeps the published
   GHCR image (do not let a recipe-stamp mismatch rebuild from this
   Dockerfile). `./start.sh` then pulls on the worker when GHCR is reachable,
   otherwise it ships the digest over SSH:

```bash
SKIP_BUILD=1 ./start.sh restart
```

If `.env` has `SKIP_PULL=1`, override it for this restart:

```bash
SKIP_PULL=0 SKIP_BUILD=1 ./start.sh restart
```

Manual pull, then the same restart:

```bash
docker pull ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor
```

Keep the wheel-less tag only if you also clear the loader:
`IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3` and `LOAD_FORMAT=`.

## Quick start (2× Spark)

```bash
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks.git
cd GLM-5.3-Flash-EXL3-2x-DGX-Sparks
cp .env.example .env          # edit HEAD_IP / WORKER_IP / WORKER_USER if needed
./download.sh                 # optional: EXL3 + DFlash2 into the head HF cache only
./start.sh                    # pull public GHCR :exl3-instanttensor, download if missing, share or rsync weights, launch TP=2
```

First run of `./start.sh` copies `.env.example` → `.env` if missing. Prefix env
wins over `.env` (`SPEC_METHOD=dflash SKIP_DOWNLOAD=1 ./start.sh restart`) for
every key, including an explicitly empty export (`KEY= ./start.sh`). Empty
values still follow each knob's own defaulting and validation rules; for example,
an empty `GLM53_DEFAULT_REASONING_EFFORT` disables the `.env` setting, while an
empty `GLM53_INDEXER_WORKSPACE` is rejected.

`./start.sh` downloads weights automatically when the HF cache is incomplete
(120 shards of `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, falling back to
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` if the mirror is incomplete, plus DFlash2 when
`SPEC_METHOD=dflash`). `./download.sh` is the same Hub fetch **on this machine
only** — no docker, no SSH, no worker rsync. Use it to stage ~164 GiB before
the worker is ready. `REFRESH_WEIGHTS=1 ./download.sh` re-fetches.
Already present: both scripts skip. With `NFS_SHARE=1` (this kit) `./start.sh`
exports the head cache over NFSv4 instead of rsyncing a copy; with `NFS_SHARE=0`
it rsyncs unless `SKIP_SYNC=1`. See [Sharing weights from the head](#sharing-weights-from-the-head-nfs_share1).

DFlash2 (`incoai/GLM-5.3-Flash-DFlash2`, ~2.3 GiB BF16, CC BY-NC-ND 4.0 research/eval)
is the default. Rollback:

```bash
SPEC_METHOD=mtp ./start.sh restart      # MTP k=2
```

`./start.sh` will:

1. Preflight docker/ssh/disk on both nodes
2. `docker pull` `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` (public; no login) on the head, then the same pull on the worker if GHCR is reachable — **unless** the local image's `glm53.recipe.stamp` does not match this checkout (Dockerfile/overlay change after `git pull`), in which case it rebuilds from this Dockerfile once. If the worker cannot pull, `docker save --platform linux/arm64 | ssh docker load`. `SKIP_PULL=1` keeps a local copy. `SKIP_BUILD=1` keeps GHCR even when the stamp drifts. `SKIP_SHIP=1` never copies. Existing kits: see [Existing installs: pull the InstantTensor image](#existing-installs-pull-the-instanttensor-image) — `git pull` alone does not replace `:exl3`.
3. Download the TR3 EXL3 repo into `$HF_HOME` / `~/.cache/huggingface` (~164 GiB, 120 shards) if missing. Same job as `./download.sh`, which stops here (head only).
4. Put the cache on the worker: **`NFS_SHARE=1`** (this kit) mounts the head's
   HF cache read-only over NFSv4 on ConnectX; otherwise `rsync` a full copy to
   `${WORKER_HOME}/.cache/huggingface`
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

Concurrent lifecycle commands on the same checkout are serialized by a `flock`
on `logs/cluster.lock`: `start`/`restart` refuse immediately when another
lifecycle command owns it, and `stop` waits up to 30 s for it and then exits 1
**without stopping anything** — retry once the running command exits. No PID is
ever signalled to break the lock. The lock is per checkout and covers TP=2
only: another clone, manual `docker rm`, and the `start-tp3.sh` /
`start-tp4.sh` stacks are not serialized by it.

### Sharing weights from the head (`NFS_SHARE=1`)

**On this kit this is on.** `.env` sets `NFS_SHARE=1`; `start-tp3.sh` sources
`.env` then `.env.tp3`, so TP=3 inherits it unless `.env.tp3` overrides.
`.env.example` still defaults to `0` (a full rsync copy) so a clone without an
NFS exporter still boots.

This is **NFSv4 over ConnectX**, not ZFS send/recv. These Sparks have no ZFS
pool; the head exports `$HF_HOME` / `~/.cache/huggingface` and each worker
mounts it read-only as a docker volume (`files/nfs-share.sh`). Only the head
stores the ~164 GiB checkpoint. The DFlash2 drafter comes along with it — the
export root is the whole HF cache, so a worker sees `hub/<repo>/snapshots/<rev>`
at the same paths the container already uses.

Read-only is safe: the serve container runs `HF_HUB_OFFLINE=1` /
`TRANSFORMERS_OFFLINE=1`, and its writable Triton/TileLang/vLLM caches are
separate node-local mounts. An exporter already serving that cache
(`vllm-fn-nfs`, `glm53fp8-nfs`, `dsv41-nfs`) is reused and its client list
merged rather than fought with — a second kernel nfsd will not start.

Each worker must mount the head's **ConnectX** address on its own cable, never
`10.0.0.x` — those are loopback aliases, and mounting over the management link
turns a 164 GiB load into an overnight job. The launcher autodetects with
`ip route get` toward that rank's fabric IP and **refuses** a loopback result;
pin `NFS_SERVER_IP_<rank>` if autodetect cannot see your cabling. This kit pins
`NFS_SERVER_IP_1=10.0.22.1` and `NFS_SERVER_IP_2=10.0.23.1` in `.env.tp3`.

```bash
NFS_SHARE=1              # .env (TP=2); TP=3 inherits unless .env.tp3 sets it
./start.sh share         # re-export + remount, no restart
./start-tp3.sh share     # same for both worker ranks
```

`./start.sh stop` / `./start-tp3.sh stop` remove the worker docker volumes and
leave the exporter running. Pattern borrowed from
`~/NewModels/DS4.1/files/nfs-share.sh` on the same kit.

### 3x Spark (TP=3)

Boots on this kit (2026-09-14). Optional sibling of `./start.sh` — same image
and weights, does not change the supported 2× path. `start.sh` never reads
`.env.tp3`. First run copies `.env.tp3.example` → `.env.tp3` (gitignored).
Containers are `glm53-exl3-tp3-*` so a TP=2 serve is not reused. Port is still
`:8888`; stop TP=2 before starting TP=3.

```bash
# edit WORKER2_IP / CX7 pins / SOCKET_IFNAME in .env.tp3
./start-tp3.sh
./start-tp3.sh status
./start-tp3.sh logs            # head; logs 1|2 for a worker rank
./start-tp3.sh share           # re-export + remount weights, no restart
./stop.sh                      # running stack(s); or ./stop.sh tp3
```

Layout (mp executor, not Ray): rank 0 `HEAD_IP` (API), rank 1 `WORKER_IP`,
rank 2 `WORKER2_IP` — `--tensor-parallel-size 3 --nnodes 3`.

**TP=3 is not TP=2 plus one node, and not TP=4 with one node removed.** Almost
nothing in this model divides by three. `overlay/tp3/` (FlyCockpit, MIT; used
only by `start-tp3.sh`) plus these flags are what make it load:

| what | stock | fix |
|---|---|---|
| attention / KV heads | 64 | `--hf-overrides` pads both to **66** (3 × 22). Knob `TP3_HEAD_OVERRIDE` |
| routed `moe_intermediate_size` | 2048 | do **not** pad (EXL3 trellis is packed 2048-wide); `--enable-expert-parallel` gives each rank **96** of 288 experts. Knob `ENABLE_EXPERT_PARALLEL` |
| vision tower | — | `--mm-encoder-tp-mode data`. Knob `MM_ENCODER_TP_MODE` |
| DFlash2 drafter | 32 heads / 8 KV | neither divides by 3 → `DFLASH_DRAFT_TP=1` (rank 0 only), plus `pad-tp3-config.py` GQA 32/8 → 36/9 |
| vocab / shared-expert / A_log | 154880 / 2048 / 64 | `patch_tp3_glm.py` pads at load (`overlay/tp3/README.md`) |

Do **not** pad heads to 96 (one rank becomes all dummy KDA heads and logits
collapse). `intermediate_size` 12288 / 3 = 4096 and `n_routed_experts` 288 / 3
= 96 divide cleanly, which is why expert parallel covers the MoE.

The decode stack is the same as TP=2 (`GLM53_ADAPTIVE_K=ema`,
`GLM53_DENSE_FP8=dense,kda`, E3 grouped). Those knobs **must reach every rank**
— a rank that captures different CUDA graphs hangs NCCL after PIECEWISE.
`start-tp3.sh` forwards them the same way `start.sh` does. At TP=3, Marlin
cannot take KDA `f_b_proj` / `g_b_proj` (row pitch is not 8-aligned at 22 local
heads); the overlay keeps those two BF16 and FP8-quantizes the rest. Look for
`[glm53-dense-fp8] TP=3 keeps KDA f_b_proj/g_b_proj in BF16` at boot.

This kit's `.env.tp3` also pins:

| Knob | This kit | Why |
|---|---|---|
| `NFS_SHARE` | inherited `1` | one 164 GiB copy on the head; neither worker has room for another |
| `MAX_MODEL_LEN` | `1000000` | three nodes hold ~55.75 GiB of weights each |
| `GPU_MEM_UTIL` | `0.80` | UMA headroom; raise only with `MemAvailable` in front of you |
| `--kv-cache-memory-bytes` | 40 GiB (`42949672960`) | `.env`'s 14 GiB cap is a TP=2 number; this flag ignores `GPU_MEM_UTIL` |
| `EXL3_FAT_GROUPED` | `1` (from `.env`) | E3 grouped fat-expert prefill |
| `SOCKET_IFNAME` | management LAN (`enP7s7`) | gloo/NCCL bootstrap; the ring has no single CX7 IF that reaches both peers |
| `NCCL_CROSS_NIC` / `NCCL_IB_SUBNET_AWARE_ROUTING` | `1` | directed dual-port ring |

Live KV on this kit (`/metrics` `vllm:cache_config_info`, 2026-09-15, 40 GiB
cap, `MAX_MODEL_LEN=1000000`): **3,230,656 tokens**, **3.23×** concurrency at
1M (`num_gpu_blocks=2213`, `block_size=64`, `cache_dtype=fp8`). Occupancy
moves with load; the pool size does not.

Wire the three boxes as a **directed ring** — each node's Port0 (cage next to
the RJ45) to the *next* node's Port1. NCCL pairs NIC index to NIC index per
channel, so a mirrored ring leaves one pair that can never connect, and
reordering `NCCL_IB_HCA` does not help (NCCL enumerates devices in system
order). Put the control plane on the management LAN (`SOCKET_IFNAME`); keep
data on RoCE via `NCCL_IB_HCA`.

**Prose** (sparkDash Decode bench, 2026-09-17, thinking **off**, 512 tok,
1–4 concurrent) on this 3× kit:

| Concurrency | TTFT | Stream tok/s | Aggregate tok/s |
|---|---:|---:|---:|
| **×1** | **255 ms** | **40.1** | **40.1** |
| **×2** | 411 ms | 28.7 | 56.6 |
| **×3** | 323 ms | 25.5 | 75.5 |
| **×4** | 351 ms | 22.8 | 88.4 |

Earlier lab medians on this kit (2026-09-14, temp 0, thinking off, 400 tok,
median of 3; count / hashmap / LRU-code): structured **87.8**, code **54.9**,
prose **39.6**, TTFT **0.25 s**. Same prompts, TP=2: 73.4 / 45.0 / 32.9 / 0.33 s.
jspark3's 3× stack is still ahead on structured (~95 tok/s) —
`DFLASH_DRAFT_TP=1` is the divisibility tax.

Shape overlays and the two EP loader traps: [`overlay/tp3/README.md`](overlay/tp3/README.md).
The flags and overlays come from
[FlyCockpit's 3× recipe](https://github.com/FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks)
(MIT), by way of [jakejharris/jspark3](https://github.com/jakejharris/jspark3)
and [outstandly/glm53-flash-3x-dgx-spark](https://github.com/outstandly/glm53-flash-3x-dgx-spark).
Those run a different launcher (`fleetctl.py`); orchestration, E3, adaptive-k
and the FP8 path stay this repo's.

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

**Stall mitigation (opt-in).** `VLLM_SM120_SPARSE_MLA_SLICE_TOKENS=64` in `.env.tp4` (or
exported before `./start-tp4.sh`) applies `overlay/patch_sparse_mla_slice.py` on every rank at
container start: the final sparse-MLA attention call runs in slices of at most 64 query rows,
the point where the 4-node stalls in #128 / #159 were seen to stop making progress (#223).
The rewrite is pinned by SHA-256 to the backend shipped in this image and refuses anything else;
the default `0` is byte-identical stock. It was qualified on another 4x GB10 kit with
`DFLASH_TOKENS=3`, mixed prefill `off` and `--enforce-eager`; treat other combinations as
unqualified until soaked. It does not identify or fix the underlying race.

Do not pull `glm53-flash-sm121:v8` — that is the older NVFP4/Ray kernel.

**Measured on a 4-Spark kit (2026-09-02).** Four DGX Sparks (two at 200G, two
at 100G, CRS812 switch), this image and overlay at 493cb88, DFlash2 draft TP=4,
1M context, launched per rank with the same `docker run` shape as `start.sh`.
TP=4 vs the 2-node baseline on the same production-mix bench (temperature 0,
30-min soak): decode 1.45x, mixed phases 20-35 % shorter, **cold prefill only
+29 %** at 282k tokens (1162 vs 901 tok/s: a 4-node TP job is fabric-bound on
prefill). One caveat found by a 150-minute soak on 2026-09-03: with the DFlash2
draft on, a 96k chunked prefill sharing steps with 6-7 speculative decode
streams hangs all four ranks (3/3 runs, 31-78 min; independent of the E2
fat-expert kernel). The draft was not the cause: the draft-off soak that first
passed clean stalled an hour later on a lone 282k cold prefill, and the
packet-loss-only explanation that followed was withdrawn as well (later stalls
reproduced with the RoCE loss counters flat). On that kit the failure was still
reproducing in September 2026, where an H16 sparse-attention progress failure
was localized at the same step (the internal race is not identified); the
mitigation in use there is a bounded final sparse-attention call (at most 64
query rows), not the knobs below. Kit-scoped attribution, updated 2026-09-09:
<https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/115#issuecomment-5599489540>
(details and receipts in the linked repo). The defaults above are tuned for 2 nodes; on 4 nodes an autoresearch
loop (one knob per relaunch, hard reliability gates) settled on the values now
in `.env.tp4.example`: `GPU_MEM_UTIL=0.75` (0.85 left <2 GiB host memory per
rank and preceded two engine deaths), `MAX_NUM_SEQS=8` (135 vs 84 tok/s
aggregate at 8 streams, worst first token 1.0 s vs 27 s), `DFLASH_TOKENS=3`
(53 % accepted vs 30 % at 7; prose decode 31-33 vs 26-28 tok/s),
`MAX_NUM_BATCHED_TOKENS=2048` (larger chunks do not prefill faster on TP=4 and
hurt first-token latency), `GLM53_MIXED_PREFILL_CHUNK=off`. Full recipe,
launcher, watchdog, benchmark and every receipt:
<https://github.com/punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark>.

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
thinking block in the generation prompt. The `Reasoning Effort:` line itself
renders unconditionally since #63 (prefix-cache stability), thinking on or off.

Do not send a literal nested `extra_body` object over raw HTTP; `extra_body` is
an OpenAI Python SDK option that merges its contents into the top-level request.
The Hub `generation_config.json` stamps `temperature=1.0` / `top_p=0.95` unless
the request overrides. The launcher sets
`--chat-template /opt/glm53/chat_template.jinja` (checkpoint jinja is language-only).

### Client request defaults

The launcher tunes the server; these are the request-side knobs the chat
template reads. None of them need a restart, and none are enforced by
`.env` — a route that drifts here is invisible from the server side.

| Field | Send | Why |
|---|---|---|
| `reasoning_effort` | `high` for reasoning work | Unset = **Max** (`files/chat_template.jinja:7`); `low` is the model card's lightest simple-Q&A mode. Keep it constant per route — see below |
| `max_tokens` | ≥ `32768` with thinking on | Max-effort reasoning runs well past 8k output tokens. Too small a cap truncates mid-thought and the reply comes back with empty `content` |
| `chat_template_kwargs.clear_thinking` | `true` for multi-turn agents | Replaces earlier turns' reasoning with `<think></think>` (`chat_template.jinja:154`), keeping the current tool-call chain. Cuts context, not answer quality |
| `top_p` / `temperature` | leave unset | `generation_config.json` already supplies `0.95` / `1.0`; the boot log prints the override line. Sending `top_p=1.0` explicitly overrides that and is worse |

Read the reply from **`reasoning`**, not `reasoning_content`. The response
models carry `reasoning` only (`ChatMessage`, `DeltaMessage`); vLLM accepts the
deprecated name on *input* and renames it, but never emits it. A client reading
`reasoning_content` gets nothing and the thinking looks like it leaked into
`content` — it did not:

```console
$ curl -s $BASE/v1/chat/completions -d '{...,"reasoning_effort":"low"}' | jq '.choices[0].message | keys'
["annotations","audio","content","function_call","reasoning","refusal","role"]
```

**Do not vary `reasoning_effort` per request within a conversation.** The effort
word lands at char 39 of the prompt, so changing it is a full prefix-cache miss
on an otherwise-warm conversation, not a partial one.
`tests/test_chat_template.py` pins that shape.

Needs: Docker (no sudo) on both nodes, python3 on the head plus a host Python with Jinja2 (verifies mounted inputs before `restart` stops anything), passwordless SSH head → worker,
`hf` / `huggingface-cli` + `curl` + `rsync` on the head, ~180 GiB free on the
head for the first download. With `NFS_SHARE=1` (this kit) workers do not need
another 164 GiB copy; with `NFS_SHARE=0` they do. The GHCR image is public; login is only needed
if you hit anonymous pull rate limits (`GHCR_TOKEN` + `GHCR_USER`).
Mixed OS accounts: set `WORKER_USER` (this kit uses `zurih` on spark2).

Chat-template validation tries `python3` from the caller's `PATH`, then
`python3.12`, `python3.11`, and `/usr/bin/python3`, selecting the first that can
import Jinja2. To pin the validator, set `GLM53_VALIDATE_PYTHON` to one executable
name (resolved on `PATH`) or path, without command-line arguments; paths containing
spaces are supported. When this variable is set, it is the **only** candidate:
an empty value, missing/non-executable interpreter, or missing Jinja2 fails closed
with exit status 2, without falling back. Unset it to restore automatic discovery.
The selected interpreter must still parse the template successfully, with loop
controls enabled; parse failures never trigger interpreter fallback. These failures
abort `start`/`restart` before either rank is stopped. Python-overlay and JSON
validation still use the caller's `python3`; this knob changes no other checks.

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
| `IMAGE` | `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` | public GHCR tag with InstantTensor 0.2.0. Existing kits must pull this tag — `git pull` does not replace a leftover `:exl3` or `SKIP_PULL=1` ([Existing installs](#existing-installs-pull-the-instanttensor-image)). Rebuilt when the overlay recipe stamp drifts (`BUILD=1` forces; `SKIP_BUILD=1` keeps GHCR). `SKIP_PULL=1` skips pull. Wheel-less fallback: `:exl3` |
| `LOAD_FORMAT` | `instanttensor` when `IMAGE` contains `instanttensor`; else empty | `--load-format`. Direct-I/O safetensors. Explicit empty (`LOAD_FORMAT=`) restores vLLM auto. Required empty on the wheel-less `:exl3` tag |
| `GHCR_TOKEN` / `GHCR_USER` | *(unset)* | optional login if anonymous GHCR pull is rate-limited |
| `PORT` | `8888` | OpenAI API on the head |
| `VLLM_API_KEY` | *(unset)* | opt-in Bearer token for `/v1`. Empty = open API. `/health` stays keyless |
| `GLM53_EXPOSE_CACHE_RESET` | `0` (off) | opt-in. `1` attaches the upstream cache-reset dev routes (`/reset_prefix_cache`, `/reset_mm_cache`, `/reset_encoder_cache`, #31) on the head API server. This flag does not enable other dev routes; independent `VLLM_SERVER_DEV_MODE` retains precedence and can enable the full dev surface. Root routes are outside the bearer guard—leave this flag off where clients are untrusted. Takes effect on restart. TP=2 `start.sh` only; `start-tp3.sh` / `start-tp4.sh` are unchanged |
| `ABLIT` | `0` (off) | opt-in. `1` = apply o_proj edit at load on both ranks. Unset leaves checkpoint weights unchanged |
| `GLM53_ADAPTIVE_K` | `off` | `ema` = adaptive verification length (prose +13–21 %); needs the capture-size list in `EXTRA_ARGS`. See *Faster prose decode* |
| `GLM53_ADAPTIVE_K_SET` | `2,4,7` | candidate draft lengths; graphs are captured for each length + 1 |
| `GLM53_DENSE_FP8` | `off` | `dense,kda` = FP8 weight-only (Marlin) dense projections, ~-11 ms/step; PROVISIONAL numerics. Groups: `shared,dense,kda,mla` |
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
| `EXL3_FAT_GROUPED` | `1` | E3 grouped fat-expert tier (`overlay/exl3_fat_moe.cu`): one gather + gate/up + down launch per layer from device-side tables, no host sync. **+37–45% cold prefill** measured (see *Cold prefill (E3)*). Default on since 2026-09-07 with the launcher defaults 850k / util 0.85 / rightsize / cap 32. Needs an image with the E3 kernels (fails closed at load otherwise). `0` = E2 kernel path (then set `MAX_MODEL_LEN=1000000`, cap 256 is picked automatically). 1M does not fit with E3 at util ≤ 0.87 on this kit (560 MiB scratch charged to the KV budget) |
| `EXL3_TEMP_ROWS_FUSED` | `32` with E3, `256` with E2 (launcher picks by `EXL3_FAT_GROUPED` unless set) | fused `exl3_moe` rows per expert; experts above it are "fat". Keep ≥ `MAX_NUM_SEQS × (DFLASH_TOKENS+1)` so decode stays one graph-safe launch. E2 wants 256 (its per-expert loop is host-bound) |
| `MAX_NUM_SEQS` | `4` | decode batch; MTP adds k+1 tokens/seq |
| `MAX_NUM_BATCHED_TOKENS` | `7168` | current maintainer default at `MAX_NUM_SEQS=4`. MNBT 2048 was the clean PR77 A/B configuration and the best measured balance on an independent `MAX_NUM_SEQS=16` geometry. Tune per deployment; change after a repeated same-kit comparison |
| `MAX_MODEL_LEN` | `850000` | default context since 2026-09-07 (E3 default; 1M fits again with `EXL3_FAT_GROUPED=0`). 1M allocates on the 1.75M padded-slot-share pool. Do not drop to 256k to “free” KV — logged tokens ≈ concurrency × this cap; hybrid block-id overhead then shrinks the pool. At MNBT 7168 one 1M request needs **14.52 GiB** KV (10.98 GiB at 500k; ~7.4 GiB fixed + 7.1 GiB per 1M); the E3 recipe runs 500k |
| `GPU_MEM_UTIL` | `0.85` | GB10 UMA budget (default lowered from 0.87 on 2026-09-07: each 0.01 is 1.2 GiB of host headroom, and long prefills need it — see *Cold prefill (E3)*). E3 at 900k / 0.85: pool ~1.05M tokens / 1.17× (0.87: 16.2 GiB / 1,051,648 tokens). Pre-E3 receipts at 1M / 0.87: 1,754,237 tokens / 18.67 GiB (MNBT 2048); 1,243,902 tokens / 1.24× (7168, rightsize, E2) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` when unset | TP=2 `start.sh` passes the effective value to both ranks. An explicit empty value disables this option; caller exports, including empty, override `.env`. Changing allocator settings requires a restart and separate memory/connector qualification; TP=4 is unchanged |
| `KV_CACHE_DTYPE` | `fp8` | packed `fp8_ds_mla`; not `nvfp4`, not bf16 |
| `DEFAULT_MAX_NEW_TOKENS` | `65536` | Omitted-only output-token default (`1..1000000`) for chat and completion requests, implemented by `overlay/patch_default_max_new_tokens.py`. Explicit `max_tokens`/`max_completion_tokens` overrides this default; independent server, platform and remaining-context caps still apply. Empty preserves stock model/server defaults and caps. Does not reserve admission capacity or fix long-prefill contention; admission is chunk-based. Caller exports (including empty) override `.env`. TP=2 launcher only; `start-tp4.sh` is unchanged. |
| `GLM53_APC_RETENTION_INTERVAL_SWA` | *(unset)* | DFlash2 drafter retention on TP=2/3/4. Empty inherits global retention with ordinary priority; explicit `0` keeps reachable boundaries and enables draft-only eviction priority; positive values must be multiples of 3584, at most 1,000,000. Requires `SPEC_METHOD=dflash` and the hybrid prefix overlay. Qualify retention, branching, and draft acceptance for the chosen global/SWA pair; see [measurements](docs/apc-retention-qualification.md) |
| `GLM53_APC_NO_STORE` | `1` | honour a client's per-request GPU prefix-cache **no-store** flag (overlay `patch_apc_no_store.py`; see [Opting a request out of the prefix cache](#opting-a-request-out-of-the-prefix-cache)). Requests never opt in on their own, so `1` changes nothing until a client sends the flag. `0` = ignore the flag (logged once); malformed values are rejected either way. Exactly `0` or `1`; the launcher refuses anything else before `restart` stops the pair |
| `GLM53_KV_CAPACITY_LOG` | `1` | after vLLM's `GPU KV cache size: N tokens` boot line (N = max_concurrency × max_model_len, **not** a pool size) log one line per KV-cache group and a summary with the usable block ids, the ids one aligned cached segment costs across groups and the resulting cached-conversation capacity (overlay `patch_kv_capacity_log.py`; see [What the KV cache boot line means](#what-the-kv-cache-boot-line-means)). `0` = off (one line saying so). Log-only, no serving change either way. Exactly `0` or `1`; the launcher refuses anything else before `restart` stops the pair |
| `GLM53_MIXED_PREFILL_CHUNK` | `fair` (`start.sh`, `start-tp3.sh`, `start-tp4.sh`, `.env.example`, `.env.tp3.example`, `.env.tp4.example`) | Mixed-prefill policy while a peer decodes. **`skip` starves prefills until decode ends** (the reported multi-minute newcomer freeze). `N>0` caps mixed chunks with hybrid alignment support; `0`/`off` disables isolation (admits newcomers in ~1 s but collapses the incumbent 10–36× on TP=2). `fair` v5 allocates decodes first, charges only prefill that contends with a decoder, fits a fixed-plus-per-token step cost, runs the largest chunk that fits `GLM53_FAIR_PREFILL_MAX_STEP_MS`, and gives a newcomer one prompt probe. Measured on TP=2 (reporter recipe, thinking essay at ~24 tok/s): 2k newcomer first token ~12 s, 30k newcomer ~164 s while the essay still streams, incumbent keeps ~80–90% of its in-run rate. TP=3 same recipe: 2k in 8.7 s, 30k in 110 s, both during the essay, incumbent ~83–87%. TP=4 inherits the same default; that topology was not re-measured. See [receipts](docs/diditfix.md) and [design](docs/astra-fix.md). |
| `GLM53_FAIR_PREFILL_CHUNK` | `256` | Probe chunk until timing samples exist. Afterwards fair v5 fits a fixed-plus-per-token step cost from solo and mixed samples and targets the largest ladder rung (128..2048) whose estimated step fits `GLM53_FAIR_PREFILL_MAX_STEP_MS`, saving credit for it instead of spending on small chunks (every prefill-bearing step costs ~0.3 s fixed on this kit, so 128-token steps ran at ~70 tok/s under v4). Base scheduler token/input and long-prefill caps still apply. |
| `GLM53_FAIR_PREFILL_SHARE` | `0.30` | Credit accrual fraction of accounted busy engine wall time (`0..1`), using a host timing proxy. Whole mixed-step cost is charged; queued async spans are counted once. 0.30 vs 0.20: a cold 30k newcomer during decode waits ~145 s instead of ~220 s; the incumbent is ~1.3–1.4× slower while that prefill runs (vs ~1.16× at 0.20). Drop to 0.20 if you want the person already answering to keep more of the engine. Do not shorten `MAX_INTERVAL_MS` first |
| `GLM53_FAIR_PREFILL_MAX_INTERVAL_MS` | `2000` | Age at which an already-served prefill may borrow one step-bounded chunk after all shared debt is repaid; a never-served newcomer gets that probe promptly. Resource, credit, and step limits can defer service beyond this age. |
| `GLM53_FAIR_PREFILL_MAX_STEP_MS` | `2000` | Limit on estimated aggregate mixed-step duration (`1..600000` ms), forwarded to all ranks. It also bounds the target chunk and any borrowed probe, so it is the one knob for how long the incumbent may pause per mixed step. 2000 ms lets the ladder reach 2048 when a 256-token mixed step costs ~1 s (1000 ms deadlocks that fitter). On this kit 1000 ms already selected 1024; 2000 ms allows 2048. Estimates do not guarantee client delivery gaps; overrun debt must be repaid. |
| `GLM53_FAIR_PREFILL_MAX_CHUNKS` | `1` | Maximum distinct prefills per turn, sharing one aggregate credit and step budget. Increasing this does not multiply one request's chunk. |
| `GLM53_EXTRA_ENV` | (empty) | space-separated `NAME=VALUE` list of extra container env for both ranks, for diagnostics (e.g. `VLLM_DEBUG_WORKSPACE=1`, `VLLM_LOGGING_LEVEL=DEBUG`). Names and values validated; launcher-owned names (everything the launcher forwards, plus `NCCL_*`/`HF_*`/`GLM53_*`/`EXL3_*`/`FLASHINFER_*`) are rejected instead of adding a duplicate `-e`. Only names are logged, and a rejected entry is reported by position only, never echoed; caller export wins over `.env`. Validation occurs during launch: invalid entries can fail after existing containers have been removed. |
| `GLM53_SUPPRESS_STOPS_IN_REASONING` | `1` | ignore client `stop` strings until `</think>` (thinking-on default) |
| `GLM53_DEFAULT_REASONING_EFFORT` | *(empty)* | `low` / `high` / `max` via `--default-chat-template-kwargs` on both ranks. Empty sends no flag, so omitted effort renders Max. Per-request `chat_template_kwargs.reasoning_effort` overrides the default; `medium` is rejected because the template maps it to Max |
| `GLM53_INDEXER_WORKSPACE` | `rightsize` (default since 2026-09-07; was `stock`) | sparse-indexer prefill gather workspace. `stock` = `max_model_len * 40` entries (**5036.40 MB** locked at 1M — measured, `VLLM_DEBUG_WORKSPACE=1`). `rightsize` = the legal per-step maximum `min(MAX_NUM_SEQS, MNBT) * cdiv(MAX_MODEL_LEN + k, index_kpool)` = 126 MB at `MAX_NUM_SEQS=4` / 504 MB at 16, so **~+26–28% KV**. Opt-in; see [docs/DESIGN-indexer-workspace.md](docs/DESIGN-indexer-workspace.md) |
| `GLM53_SPINWAIT_MS` | `stock` | SpinCondition reader busy-loop window. `stock` preserves vLLM's 1 s default; `1..1000` selects milliseconds. A frozen TP=2 sweep selected `16` (+0.95% median decode vs stock, 85.3% less active EngineCore CPU) |
| `GLM53_BOOT_SHAPE_WARMUP` | `1` | after `/health`, burn DFlash2 BLOCK / sampler / kpool shapes (nonfatal) |
| `TRITON_HOST_CACHE` / `TILELANG_HOST_CACHE` | `$CACHE_ROOT/triton` / `tilelang` | persist JIT caches across container recreate |
| `NFS_SHARE` | `0` in `.env.example`; this kit's `.env` is `1` | `1` = workers mount the head's HF cache over NFSv4 instead of an rsync copy — see [Sharing weights from the head](#sharing-weights-from-the-head-nfs_share1). TP=3 inherits `.env` unless `.env.tp3` overrides |
| `NFS_SERVER_IP_<rank>` | *(autodetect)* | head ConnectX address that rank mounts; a `10.0.0.x` result is refused |
| `LANGUAGE_MODEL_ONLY` | `0` | load vision tower (image + video) |
| `SKIP_MM_PROFILING` | `1` | skip max-size MM dummy at init (OOM otherwise) |
| `LIMIT_MM` | `{"image":48,"video":1}` | `--limit-mm-per-prompt` (validation ceiling; nothing reserved) |
| `MM_IMAGE_TOKENS` | `2048` | `--mm-processor-kwargs {"max_image_tokens":N}`; empty = checkpoint's 8000 |
| `VIDEO_NUM_FRAMES` | (empty) | `--media-io-kwargs {"video":{"num_frames":N}}`; empty = vLLM's 32 |
| `MM_PROCESSOR_CACHE_GB` | `1` | `--mm-processor-cache-gb` (vLLM default 4 GiB of host RAM) |
| `HEAD_CX7_IF` / `WORKER_CX7_IF` | `enp1s0f1np1` / `enp1s0f0np0` | NCCL sockets |
| `HEAD_CX7_IB` / `WORKER_CX7_IB` | `rocep1s0f1` / `rocep1s0f0` | NCCL HCAs |
| `USE_HOST_NCCL` | `0` | image nvidia-nccl; host preload duplicates DeepEP |
| `GLM53_EXL3_MOE_FAST` | `0` | opt-in SM121 K4/N256 **thin-decode** kernels for routed experts (`overlay/patch_exl3_decode_pipeline.py`; see [docs/sm121-perf-paths.md](docs/sm121-perf-paths.md)). `1` needs an image built with that patch and requires the fused `exl3_moe` path — otherwise model load raises instead of silently degrading (also under `EXL3_FUSED_MOE=0`). Exactly `0` or `1`; the launcher refuses anything else, including explicit empty, before `restart` stops the pair. TP=2 only; the TP3 launcher keeps unsetting it |
| `VLLM_SM120_SPARSE_MLA_SLICE_TOKENS` | `0` (`start-tp4.sh`, `.env.tp4.example`) | **TP=4 only.** `64` slices the final sparse-MLA attention call into <=64 query rows on every rank (`overlay/patch_sparse_mla_slice.py`, hash-pinned to this image's backend); `0` keeps the backend byte-identical. Opt-in mitigation for the all-rank stall in #128 / #159 (#223); bounds the call where progress stopped, does not fix the race. Any other value is refused. |

`DEFAULT_MAX_NEW_TOKENS` preserves omitted completion limits through Pydantic normalization; an explicit `max_tokens: null` retains the pinned runtime's native normalization to 16. The overlay validates the limiter, completion caller, and protocol validator before writing any target. The CPU regression (`python3 tests/test_gen_defaults.py`) requires Pydantic v2 and exercises its real before-validator, not fabricated field-set metadata.

Native thin-kernel qualification uses `tests/test_exl3_thin_fast_gpu.py` and `tests/compare_thin_fast.py` with stock/candidate/stock receipts before timing `tests/bench_exl3_thin.py`. The fixtures route each token to eight distinct experts from a 32-expert pool. Uniform, correlated and hot-expert cases must never route a token to the same expert twice; `tests/test_exl3_routing.py` checks this on CPU. Receipts from the older with-replacement fixtures do not establish parity for valid top-k routing. Maintainer-measured TheGrill v0.3.0 A/B/A2 on current `main` (`glm-routine-decode-v3`, image `sha256:f267534b…`, fresh boot per arm, descriptive — no PASS envelope) gives **+7.71% / +8.88% / +14.59%** median decode rate in the three cells whose ranges resolve, with an A2-vs-A drift of −0.31% / +0.30% / −3.54%; two cells are withheld by the tool's range-overlap rule. These checks alone do not qualify full-model quality or speed, and the hash-frozen numerical study for this path is formally **inconclusive**: both predeclared control self-tests fail on unchanged stock repeats (the absolute-KL gate and the tau-transfer bound), so no claim here is a pass — see [docs/sm121-perf-paths.md](docs/sm121-perf-paths.md) for the recorded numbers and limits.

**Default from this checkout:** E2 fat kernel on (`EXL3_FAT_KERNEL=1`) and `MAX_NUM_BATCHED_TOKENS=7168`; the E3 grouped tier is the launcher default (`EXL3_FAT_GROUPED=1`, see *Cold prefill (E3)*). The pre-E2 C4 keep was 2048; the current E2 cold-prefill baseline is the linked PR77 table; E2 at 7168 is ~1,150–1,185 tok/s cold, E3 ~1,580–1,640.
## Opting a request out of the prefix cache

`BlockPool.free_blocks` puts **hashed** blocks at the back of the free queue (LRU) and unhashed
ones at the front (LIFO). A one-off batch/eval request therefore stores its blocks *behind* the
owner's idle 80K conversation and the owner's blocks are what gets evicted next — the batch job
re-orders the LRU in its own favour. `cache_salt` namespaces and still stores; vLLM's
`skip_reading_prefix_cache` is read-side only. Overlay `patch_apc_no_store.py` adds the write-side
opt-out: `SamplingParams.skip_writing_prefix_cache`, reachable on `/v1/chat/completions`,
`/v1/completions` and `/v1/responses` through `vllm_xargs` (no entrypoint edits):

```bash
curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-EXL3", "messages": [{"role": "user", "content": "classify: ..."}],
  "max_tokens": 64, "cache_salt": "batch-lane-07",
  "vllm_xargs": {"skip_writing_prefix_cache": 1}}'
```

Send the integer `1` (`"1"` and JSON `true` also work: `vllm_xargs` is typed
`dict[str, str | int | float | list]` and pydantic coerces a JSON boolean to `1`/`0` — verified on
pydantic 2.13). Any other value (`1.0`, `"yes"`, `2`, …) is rejected with HTTP 400 naming the field
(validated in `SamplingParams.__post_init__`, i.e. in the API server — never a silent no-op). The
request then:

- is **still allowed to read** the cache (a lane that shares the system prompt gets the free prefix;
  reading touches blocks, i.e. refreshes their LRU position — the flag is write-only);
- inserts **no** block hash in any KV-cache group (MLA, mamba partial tails, drafter SWA) and emits no
  `BlockStored` event; all allocation bookkeeping (`num_cached_block`, partial-hit CoW for what it
  *read*) proceeds exactly as for a normal request — the guards sit at the two `_insert_block_hash`
  sites in `BlockPool`, not at `allocate_slots`, because `num_cached_block` doubles as the
  running-request sentinel for SWA/drafter allocation;
- has its blocks freed to the **front** of the free queue, so they are the next ids recycled and the
  resident conversation is not displaced;
- if preempted, resumes from whatever it could *read* — nothing, when its prefix was cold — so prefer
  it for short lanes, ideally with a low `priority`.

Server-log receipts (each once per process): `[glm53-apc-no-store] first request resolved
skip_writing_prefix_cache=1` (the flag reached the engine) and `[glm53-apc-no-store] suppressing
prefix-cache store (full site)` / `(partial site)` (a store was actually cut; the partial site needs the
runtime's fine-grained partial-tail producer, which the coordinator vetoes for this model's
`KpoolTailManager` — see [Prefix caching](#prefix-caching-this-kit-2026-08-30) — so on this kit it is
normally the full site that fires). If the first line never appears, the flag did not reach the
engine — do not trust an A/B measured without it. Not covered:
KV connectors / CPU offload (none on this kit), pooling requests. Kill switch: `GLM53_APC_NO_STORE=0`.
Design + receipts protocol: `docs/DESIGN-apc-no-store.md`.

## What the KV cache boot line means

vLLM logs once per boot (`v1/core/kv_cache_utils.py`, `update_kv_cache_capacity`):

```
GPU KV cache size: 1,553,140 tokens, Maximum concurrency for 1,000,000 tokens per request: 1.55x
```

The first number is `int(max_concurrency × max_model_len)`, with `max_concurrency = num_blocks /
num_blocks_per_request` and `num_blocks_per_request` a **sum over KV-cache groups** of
`cdiv(spec.max_memory_usage_bytes, spec.page_size_bytes)`. It is a concurrency figure in token units, not the
size of a prefix cache. On this hybrid model (MLA + kpool tail + 4 mamba + DFlash2 drafter SWA, one shared
`BlockPool` with globally unique block ids) the pool behind that line was **643 block ids** (642 usable; id 0 is
the null block) and one cached 3584-token segment costs **38** of them (1 MLA + 4 mamba + 33 drafter), so
about **57K tokens** of cached conversation fit with nothing running — not 1.5M. Neither `num_blocks` nor
`num_blocks_per_request` is logged by stock vLLM (feature request
[vllm-project/vllm#54662](https://github.com/vllm-project/vllm/issues/54662)).

Overlay `patch_kv_capacity_log.py` (`GLM53_KV_CAPACITY_LOG=1`, default) keeps that line byte-identical and adds,
from the same config objects, one line per group and one summary — on the 643-id boot:

```
[glm53-kv-capacity-log] group 0: MLAAttentionSpec layers=11 block_size=3584 page_size=… B blocks/request@1,000,000=280 prefix_caching=yes
[glm53-kv-capacity-log] group 1: KpoolTailSpec layers=11 block_size=4 page_size=… B blocks/request@1,000,000=1 prefix_caching=no (scratch) window=4 eagle=no
[glm53-kv-capacity-log] group 2: MambaSpec layers=9 block_size=3584 page_size=… B blocks/request@1,000,000=9 prefix_caching=yes mamba_cache_mode=align
… groups 3-5 identical to group 2 …
[glm53-kv-capacity-log] group 6: SlidingWindowSpec layers=1 block_size=64 page_size=… B blocks/request@1,000,000=97 prefix_caching=yes window=2048 eagle=yes
[glm53-kv-capacity-log] usable block ids: 642 (num_blocks=643 incl. the null block; 414 ids per 1,000,000-token request => 1.55x); ids per 3584-token cached segment across groups: 38 (per group: [1, 0, 1, 1, 1, 1, 33]); cached-conversation capacity at this alignment ≈ 57,344 tokens = 16 segments (aligned dense-retention prefix-cache upper bound: nothing running, every reachable block hashed, block-aligned hits). The 'GPU KV cache size' line above is max_concurrency x max_model_len, not this figure.
```

`blocks/request` is the same `cdiv` expression the stock line is built from (its column sum, 414, is the stock
denominator). "Ids per cached segment" is the dense-retention cost with block-aligned hits — what each manager's
`reachable_block_mask` hashes with no retention interval: every block for `FullAttentionSpec` / `MLAAttentionSpec`
and for `MambaSpec` in `align`/`all` mode (read from the spec the manager acts on),
`min(cdiv(window − 1, 64) + 1 (EAGLE), 3584 / 64)` = 33 for the drafter (`SlidingWindowSpec` /
`SlidingWindowMLASpec`), 0 for the kpool tail (opts out of prefix caching). Exact class names only: any other
spec — subclasses such as `SinkFullAttentionSpec` included, since they come with their own manager rule — is
reported as unmodelled and the capacity is withheld rather than guessed, as it is under DCP/PCP > 1 (block sizes
are rescaled per rank there). The alignment is the lcm of the group block sizes (the coordinator's scheduler
block). The figure is an upper bound: a running request holds its own blocks, and PR #83's per-group retention
lowers the drafter's cost to boundary tails only (not modelled here — the summary states its assumption).

The numbers move with the boot, the arithmetic does not: the figures elsewhere in this README (690 blocks /
1,754,237 tokens / 1.75× at 1M) are the same quantity on the reference kit's boot (690 / 1.75 ≈ 394 ids per
1M-token request under that config); our 2026-08-31 boot had 643 ids at 414 per request, and the current
rightsized-workspace boot 820 (`usable block ids: 819 … ≈ 75,264 tokens = 21 segments`). Read your own boot's
summary line rather than any number in this section.

## Image / overlay

```bash
docker build -t glm53-flash-sm121:local .
# or: BUILD=1 ./start.sh
```

`./start.sh` **rebuilds** from this Dockerfile when the image label `glm53.recipe.stamp` does not match the current overlay/Dockerfile hash — that is what makes a `git pull` pick up `exl3_fat_gemm` instead of staying on the public GHCR tag (which predates E2). `SKIP_BUILD=1` keeps GHCR. `BUILD=1` forces a rebuild. `SKIP_PULL=1` skips `docker pull` only. To take the InstantTensor default without a local rebuild, set `IMAGE` to `:exl3-instanttensor` and restart with `SKIP_BUILD=1` ([Existing installs](#existing-installs-pull-the-instanttensor-image)).

After CUDA compile, Python overlay edits (`overlay/exl3.py`, tests) are a cheap layer so they do not rebuild `exllamav3_ext`.

| Path | Role |
|---|---|
| `Dockerfile` | NoPE sparse-MLA patches + EXL3 install (`sm_121a`) + self-check |
| `overlay/exl3.py` | `Exl3Config` / packed load / TP shard / fused `exl3_moe` apply / E2 fat kernel |
| `overlay/exl3_fat_gemm.cu` | additive `exl3_fat_gemm` / `_scatter` compiled into `exllamav3_ext` |
| `overlay/exl3_fat_moe.cu` / `.cuh` | E3 grouped fat-expert kernels (gather / gate-up+SwiGLU / down+scatter); compiled into `exllamav3_ext` by the full build, or as the additive `exl3_fat_moe_ext` module by `Dockerfile.e3-layer` + `overlay/build_exl3_fat_moe_ext.py` |
| `tests/bench_e3_microbench.py` | CUDA-event MoE-layer timing E2 vs E3 + production-geometry parity (isolated, serving stopped) |
| `overlay/patch_exl3_ext_aarch64.py` | stub AVX CPU allreduce so the ext builds on GB10 |
| `overlay/patch_exl3_decode_pipeline.py` | additive SM121 K4/N256 thin-decode kernels (shared / independent gate-up transform) + `glm53_fast_moe_version` in `exllamav3_ext`; stock kernels untouched, every anchor validated before anything is written |
| `tests/test_exl3_decode_pipeline.py` | CPU: patch anchors, double-apply and partial-write refusal, gate/up SUH alias gate, `GLM53_EXL3_MOE_FAST` 0/1 contract, and the load-time fail-closed paths (no native symbol, wrong version, fused path unavailable) |
| `overlay/patch_model_overrides.py` | `"exl3"` in ModelConfig overrides |
| `tests/test_exl3_overlay.py` | registry, TP shard, `sm_121a` cubin, fused vs loop GEMM, `EXL3_FUSED_MOE=0`, E2 diag schema, E3 grouped tables/parity/graph-replay/fallback checks |
| `tests/test_apc_per_group_retention.py` | host: overlay apply/idempotence, min-exemption derivation, routing, env validation, composition with `patch_hybrid_prefix_hit.py` in both orders, id-cost/capacity arithmetic (needs `GLM53_KV_COORDINATOR_PY_SRC` + `_PRISTINE` copies of the fork's coordinator) |
| `overlay/patch_apc_no_store.py` | per-request GPU prefix-cache no-store (`skip_writing_prefix_cache` / `vllm_xargs`): strict 0/1 validation in `SamplingParams.__post_init__`, never-raising resolver on `Request`, guards at the two `_insert_block_hash` sites in `BlockPool`; transactional, fail-closed; kill switch `GLM53_APC_NO_STORE` |
| `tests/test_apc_no_store.py` | host: apply / idempotence / drift with nothing written / partial-application refusal; resolver accept-reject and kill-switch behavior; on CPU vLLM (`GLM53_VLLM_SRC_ROOT`, mandatory in the image): real `BlockPool` free-queue policy, chunked-prefill bookkeeping parity, hybrid partial-tail producer/reader/CoW, seven-group fork layout, env-driven retention and cache lookup, preemption, `skip_reading`+`skip_writing`, and log receipts |
| `tests/test_launcher_rank_parity.py` | launcher (CPU-only, docker/ssh stubbed): retention and kill-switch validation, pre-stop artifact checks, ordered hybrid/per-group/no-store overlays (kv-capacity-log after its shared-file drafter-group patch and before xgrammar), and matching rank environments and mounts including no-store, KV-capacity-log, thin-decode, and cache-reset values |
| `tests/bench_decode.py` | streaming decode + coherence; `--structured` is the count-1→200 median |
| `tests/test_start_overrides.py` | CPU-only caller precedence: `.env` keys, empty exports, shell assignments, and child inheritance |
| `tests/test_launcher_extra_env.py` | launcher (CPU-only, docker/ssh stubbed): non-owned `GLM53_EXTRA_ENV` diagnostics reach both ranks as `-e` pairs, launcher-owned names fail the launch before any container starts, values stay out of the log, and malformed entries are rejected by position without echoing any fragment |
| `start.sh` / `stop.sh` / `download.sh` | 2-node launch; Hub fetch on the head only. `./stop.sh` also stops TP=3 when that stack is up (`tp2` / `tp3` / `all`) |
| `start-abliterated.sh` | pinned, fail-closed preset for the pre-edited EXL3 checkpoint |
| `start-tp3.sh` / `.env.tp3.example` | 3-node TP=3 launch (head padding + expert parallel + `overlay/tp3/`); knobs stay out of `.env`. Boots this kit 2026-09-14 |
| `overlay/tp3/` | TP=3-only shape overlays (FlyCockpit MIT): head/vocab/shared-expert/A_log pads, EP loader, SM120 decode pad. Not on the TP=2 path |
| `files/nfs-share.sh` / `files/nfs-server/` | `NFS_SHARE=1`: workers mount the head's HF cache over NFSv4 instead of holding a copy. Used by `start.sh` and `start-tp3.sh`. This kit has it on |
| `start-tp4.sh` / `.env.tp4.example` | experimental 4-node TP=4 launch; knobs stay out of `.env` |
| `kernel_lab/exl3/` | model-agnostic EXL3 K1-K8 oracle, SM121 tactic sweep, and Atlas receipt producer (development-only; not in the production image) |
| `docs/exl3-sm121-kernel-lab.md` | kernel ABI, measurements, upstream sources, decisions, and next gate |
| `files/chat_template.jinja` | GLM-5.3 MM template (`<|image|>` / `<|video|>`); checkpoint jinja is language-only |
| `overlay/qwen3_dflash2.py` | DFlash2 draft (grouped conv + candidate selector) |
| `overlay/dflash2_speculator.py` | DFlash2 selector walk (V2 speculator) |
| `overlay/patch_dflash2.py` | registry + `decoder_layer_cls` + speculator dispatch + draft KV `auto` on MLA/FP8 |
| `overlay/patch_glm_eagle3.py` | Glm5Next EAGLE3 aux-hidden layers (mHC `hc_post` + contract) |
| `overlay/patch_glm5_drafter_group.py` | GLM KV fast path + DFlash2 padded slot-share (`block=64`, `page_size_padded=mla_page`); runtime-mounted by `start.sh` (`DRAFTER_PATCH_HOST`) |
| `overlay/patch_glm_video_placeholders.py` | align video timestamp blocks to encoder `grid_t` |
| `overlay/patch_suppress_stops_in_reasoning.py` | fail-closed detokenizer guard: client `stop` dormant until `</think>` |
| `overlay/patch_scheduler_decode_floor.py` | skip / cap / off / `fair` mixed-prefill; v5 fixed-cost step fit + largest step-fitting chunk + bounded contention credit, decode first; versioned installer |
| `tests/test_scheduler_decode_floor.py` | v5 migration from v1/v2/v3/v4; cost fit, ladder climb-back, prompt probe, async accounting, bounded credit, alignment and actual scheduler budget regressions |
| `overlay/patch_xgrammar_termination.py` | source-exact vLLM #52805/#53046 backports; stop at termination and validate post-reasoning speculative drafts before FSM advance |
| `tests/test_xgrammar_termination.py` | exact two-file patch, idempotence, cross-file fail-closed drift, termination/rollback/reset and post-reasoning draft behavior, launcher wiring |
| `overlay/patch_cache_reset.py` | mount only the upstream cache-reset dev router (`/reset_prefix_cache` et al., #31) when `GLM53_EXPOSE_CACHE_RESET=1`; runtime-mounted by `start.sh` (`CACHE_RESET_PATCH_HOST`) |
| `tests/test_cache_reset_endpoint.py` | exact `build_app` anchor, flag semantics (off / on / dev precedence), fail-closed drift, idempotence, installed copy |
| `overlay/patch_kpool_tail_slotmap.py` | clamp KpoolTail one-block circular slot mapping; identity for other KV groups |
| `tests/test_kpool_tail_slotmap.py` | circular addressing math, exact kernel patch, idempotence, fail-closed drift, launcher wiring |
| `overlay/patch_kv_capacity_log.py` | log-only: after the stock `GPU KV cache size` line (kept byte-identical) log per-group `blocks/request` (the stock line's own denominator) and the usable-block-ids / ids-per-aligned-segment / cached-conversation-capacity summary; unmodelled spec kinds withhold the figure; knob `GLM53_KV_CAPACITY_LOG` (0/1); two pinned anchors, preflighted before either is written, atomic, idempotent |
| `tests/test_kv_capacity_log.py` | CPU-only execution of the shipped derivation helpers against hybrid, single-group, uniform-type, unsupported-spec, and null-block cases; patch application and idempotence on a pinned fixture, fail-closed anchor drift, and installed-source preflight when available (required in the image). Launcher flag validation belongs to `tests/test_numeric_config.py`. |
| `overlay/patch_indexer_workspace.py` | opt-in `GLM53_INDEXER_WORKSPACE=rightsize`: size the sparse-indexer prefill workspace to the legal per-step maximum instead of `max_model_len * 40`; boot-time compress-ratio cross-check |
| `tests/test_indexer_workspace.py` | sizing formula (MNBT/`max_num_seqs`/spec-token edge cases, stock clamp), chunk-list equivalence vs stock by exhaustion, exact three-site patch, idempotence, fail-closed drift, launcher wiring |
| `overlay/patch_spinwait.py` | opt-in numeric `GLM53_SPINWAIT_MS`: fail-closed runtime patch of SpinCondition's reader busy-loop window on both ranks |
| `tests/test_spinwait_patch.py` | numeric contract, exact patch, idempotence, drift rejection, mode preservation, pyc cleanup, and launcher/build wiring |
| `overlay/patch_tool_choice_none.py` | honor `tool_choice:"none"` at decode time: keep tools in the prompt, mask the `<tool_call>` opener via `bad_words` (glm47 parser hook); see `docs/tool-choice-none.md` |
| `tests/test_tool_choice_none.py` | exact patch, idempotence, fail-closed drift, none+tools masking, client `bad_words` union, auto/required/no-tools/Responses untouched |
| `overlay/ablit_runtime.py` | load-time o_proj transplant / projection (`ABLIT=1`); no-op when off |
| `overlay/patch_ablit.py` | install the load_weights hook; bind-mounted and run on both ranks |
| `ablit/` | direction vectors + `LAYER_MAP.json` from drowzeys' published recipe; `fetch_transplant.py` + `transplant/` for the donor o_proj byte-copy |
| `tests/test_ablit.py` | recipe integrity, orthogonalization math, TP-shard equivalence, transplant byte-copy + TP slice, hook gating |
| `tests/test_default_reasoning_effort.sh` | `GLM53_DEFAULT_REASONING_EFFORT` enum guard (`""`/`low`/`high`/`max`; `medium` rejected) and the `--default-chat-template-kwargs` flag at both rank sites, sliced out of `start.sh` and evaluated |
| `scripts/boot-shape-warmup.sh` | post-`/health` DFlash2 k=7 BLOCK ladder + sampler/kpool arms |
| `tests/test_boot_shape_warmup.py` | the shipped warmup script end-to-end with `WARMUP_CURL` stubbed: all 9 ladder/prefill prompts (65536 rung included) arrive byte-exact, the 24-request tally holds, and the tokenize-mismatch / smaller-context runs exit 1 while still warming the rest |
| `scripts/tool-choice-none-preflight.py` | pre-generation gates for the `tool_choice:none` live test: identity/launch flags, enforcement-chain digests, tokenizer opener-mask dry run (`docs/tool-choice-none.md`) |

Image-build runs `EXL3_SELFCHECK_GPU=0`. `./start.sh` runs the GPU self-check
(`docker run --gpus all`) before shipping unless `SKIP_OVERLAY_VERIFY=1`.

The kernel lab is development-only source: the production Dockerfile does not
copy or install `kernel_lab/`, so the served image does not contain it. The lab
accepts arbitrary `OPERATOR:K:N` shapes or a JSON workload file; GLM/Hy4 names
are optional historical fixtures, not a supported-model list. See
[`kernel_lab/README.md`](kernel_lab/README.md) for the portable correctness
gate and isolated SM121 tuning command.

## Do not

- Destroy HF weights, requantize, or `docker rm` HF caches. `REFRESH_WEIGHTS=1 ./download.sh` only if you intend to re-fetch
- `--moe-backend marlin`, NVFP4 weights, or `glm53-flash-sm121:v8` as this serve
- qemu / amd64 / `cstechdev/vllm:glm53-flash-nope-sm120-*` / verdictai SM120 B12X
- `--kv-cache-dtype nvfp4` or bf16 (no sparse-MLA kernel)
- `"attention_backend": "TRITON_ATTN"` in speculative-config (causal-in-block on this image)
- Change TP / CX7 pins / `USE_HOST_NCCL` in `.env` unless you are re-plumbing NCCL. Three nodes is `./start-tp3.sh`, not `TP=3` in `.env`
- Force-push

## License

This repository (serve scripts, overlay, docs) is **[AGPL-3.0](LICENSE)**.
If you run a modified version as a network service, the AGPL requires you to
offer its source to users of that service. Contributions made before
2026-09-07 were licensed MIT; that notice is retained in
[`LICENSE.MIT`](LICENSE.MIT). The EXL3/TR3
checkpoint stays [ShapleyMCG License 1.0](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/blob/main/LICENSE)
(unmodified upstream LICENSE; also on
[brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)).
The [prebuilt derivative](https://huggingface.co/bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit)
retains that license and the parent's third-party notices. DFlash2 stays [CC BY-NC-ND 4.0](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).

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
- **Prebuilt EXL3 derivative:** [bullerwins](https://huggingface.co/bullerwins) published
  [GLM-5.3-Flash-exl3-4bpw-ablit](https://huggingface.co/bullerwins/GLM-5.3-Flash-exl3-4bpw-ablit),
  using the [Keys L15-43/MTP-L45 transplant](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-43-mtp-l45)
- **Abliteration recipe / direction artifacts:** [drowzeys](https://huggingface.co/drowzeys) —
  [keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock)
- **TP=3 shape recipe** (`start-tp3.sh` only): `--hf-overrides` padding the 64
  attention/KV heads to 66, `--enable-expert-parallel` for the MoE,
  `--mm-encoder-tp-mode data` for the vision tower, a drafter left at
  `draft_tensor_parallel_size=1`, and the load-time pads in `overlay/tp3/`
  (vendored from
  [FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks](https://github.com/FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks),
  MIT). Reached this kit by way of
  [jakejharris/jspark3](https://github.com/jakejharris/jspark3)
  (Apache-2.0) and
  [outstandly/glm53-flash-3x-dgx-spark](https://github.com/outstandly/glm53-flash-3x-dgx-spark)
  (MIT). Those projects run their own launcher (`fleetctl.py`);
  orchestration, E3, adaptive-k and the FP8 path stay this repo's.

## Concurrency ladder (2026-08-31, 1M ctx, MNBT 2048, MAX_NUM_SEQS 16, `GLM53_MIXED_PREFILL_CHUNK=512`)

Both tables are historical measurements of the configuration named in their own
heading, not of current main. Only the 2026-09-01 run has raw cells checked in
(`docs/ladder-final-2026-09-01.json`).

`tests/bench_concurrency.py` runs N simultaneous streams per level (modes `code` / `data` / `chat`; optional cached context per lane),
counts tokens from the server's `usage`, and writes per-cell JSON (`agg_tps`, `stream_tps_median`, TTFT/ITL p50/p95/p99, cache hit
ratio, preemptions) plus a live `logs/status.json` (gitignored; `--status` moves it). Canonical run (idle server):
`python3 tests/bench_concurrency.py --levels 1,2,4,8,12,16 --modes code,data,chat --ctx 0,50000,100000 --reps 3 --out logs/ladder.json`.

`tests/bench_live.html` renders that live file by default, or a checked-in receipt via `?src=` (paths resolve under `tests/`).
It must be served over loopback HTTP — `fetch()` from a `file://` page is blocked and the page only shows "waiting": from the repo
root run `python3 -m http.server 8765 --bind 127.0.0.1`, then open `http://127.0.0.1:8765/tests/bench_live.html`, e.g.
`.../tests/bench_live.html?src=../docs/ladder-final-2026-09-01.json` for the 2026-09-01 receipt.

| job (temp 0) | ×1 | ×2 | ×4 | ×8 | ×12 | ×16 |
|---|---:|---:|---:|---:|---:|---:|
| code — aggregate tok/s | 41.9 | 52.3 | 77.9 | 93.8 | 116.2 | 127.2 |
| code — per stream | 44.3 | 28.8 | 21.0 | 13.9 | 11.9 | 10.2 |
| data (JSON/CSV) — aggregate | 31.6 | 53.0 | 72.0 | 83.2 | – | – |
| chat — aggregate | 18.1 | 26.3 | 36.1 | 55.9 | 66.4 | 76.7 |
| chat — per stream | 18.5 | 14.1 | 10.0 | 7.7 | 6.1 | 5.2 |
| TTFT median (s), any mode | 0.4–0.6 | 0.7–1.0 | 0.8–1.0 | 1.1–1.3 | 1.1–1.6 | 1.3–1.6 |

Warm-context ladder (2026-09-01, `docs/ladder-final-2026-09-01.json`), recorded on the
then-unmerged overlay stack per-group retention + fine-grained hits + gate v2
(PRs #83 / #84 / #80) — none of which is current main. #83 was closed unmerged;
main carries per-KV-cache-group retention via merged #130, where an empty
`GLM53_APC_RETENTION_INTERVAL_SWA` inherits the global retention interval instead of
applying #83's automatic rule. #84's overlay patch and the #80 gate-v2 knobs are not
in main. Current TP=2 main defaults to `GLM53_MIXED_PREFILL_CHUNK=fair` (v5);
the historical gate-v2 measurements below do not qualify that policy.

| ctx 50K per lane (distinct prefixes, verified warm) | ×1 | ×2 | ×4 | ×8 | ×16 |
|---|---:|---:|---:|---:|---:|
| code — aggregate tok/s | 35 | 50 | 67 | 8.5–29 | 6.8 |
| chat — aggregate tok/s | 18.5 | 26 | 35 | 5–7 | 5.2 |
| cache hit | 1.0 | 1.0 | 0.999 | 0.25–0.75 | 0.19 |
| TTFT p95 (s) | 0.4 | 0.8 | 1.2–1.4 | 173–454 | 899–963 |

What it shows on that stack: **up to 4 concurrent 50K-context lanes ran fully warm** (hits 0.999, TTFT ≤1.4 s,
aggregate equal to the same run's ctx-0 cells); at 8×50K the recorded cached working set (~400K tokens + in-flight) exceeds the
~455K-token budget measured then (642 block ids / ~5 per 3584-token segment) and hit rates collapse. The ladder alone does not
establish that budget as the cause — later whole-stack comparisons (#174) neither confirm the attribution nor isolate a single
root cause. Speed is set by the DFlash2 drafter's acceptance (code ~35–44 tok/s solo, prose chat ~18), not by temperature
(0 vs 0.7 within noise) or thinking on/off; the interactive knee is ~4 lanes. Under the original `skip` policy the second stream
waited 15–17 s (requests served one at a time) — fixed by the mixed-prefill gate v2 knobs on this stack.
