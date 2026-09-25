# Changelog

All notable changes to this GLM-5.3-Flash EXL3 serve recipe are documented here.

Versions **1.0.0–1.5.0** are retrospective SemVer labels over merged `main` history.
There were no git tags for 1.0.0–1.4.0; 1.5.0 is the first cut named as a release.
**1.6.0** is this kit's current cut. Dates are merge dates on
`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`.

## [Unreleased]

### Added

- Auto/lazy safetensors staging (`GLM53_LOAD_CLONE=1`) and optional bounded
  local-shard read-ahead (`GLM53_LOAD_PREFETCH=0`, decimal `0..16`) on TP2/TP3/TP4.
  Preserve InstantTensor selection, rank parity, PR230 safety and compact-draft
  settings. The shard window is not a host-memory limit; GPU loader performance
  and KV-capacity gains are not established for this combined candidate.
  Motivated by [Alexbob0's mmap-load measurements](https://github.com/Alexbob0/glm53-flash-vllm-upstream-sm121/blob/bc3891aed74a1f4ccd679e5205ab9bd2605cf283/README.md).
- `examples/tp2-long-coding.env`: the maintainer's TP=2 long-coding profile
  (262k context, two sequences, 1,024-token prefill batches) with each
  default-off option it enables, its measured benefit, and its cost. Not
  sourced automatically; defaults are unchanged.
- Compact DFlash2 KV pages (`GLM53_DRAFT_KV_COMPACT`): derive a
  page-fitting divisor of the MLA block to reduce draft block-ID demand
  without changing precision or backing allocations. Now ON by default under
  `SPEC_METHOD=dflash` (the default method) and OFF otherwise, so
  `SPEC_METHOD=mtp`/`none` is unaffected; the default applies only when the
  variable is unset — an explicit `0` opts out and an explicitly empty value
  is rejected at launch (was: default `0` everywhere).
  Reject padded-page kernel splitting during backend setup. DFlash-only:
  an allocator preflight on every grouping path verifies the method and
  matching draft-layer count before exact-fit or padded selection.
  TP2/TP3/TP4 launchers independently reject `1` unless `SPEC_METHOD=dflash`.
  Under `1`,
  `overlay/patch_hybrid_prefix_hit.py` looks
  the DFlash drafter group up ending exactly at the reconciled prefix
  boundary instead of requiring one complete draft block past it
  (`# [glm53-dflash-boundary-lookup-v1]`), so a same-prompt reuse whose tail
  is shorter than the draft block no longer loses one MLA page to the replay
  clamp; DFlash context KV is a per-position projection of the target hidden
  state, so no lookahead block is needed. At `9a3aca4`, 61 focused CPU
  tests pass without skips. Fresh custom TP2 OFF/ON/OFF qualification
  completed 153 requests, 90 correct reference answers and three rollover
  checks without preemptions or safety stops. Repeat TTFT was 30.46% lower
  versus mean OFF; prose and C2 throughput were lower, not universally faster.
  The prefix overlay migrates the stock image's legacy hybrid-apc
  coordinator form (with or without the published replay stage) to the
  current verification form before installing the boundary lookup; the
  result is byte-identical to a pristine install for either flag value.
  Before writing, the overlay verifies that every owned stage (helpers,
  EAGLE narrowing, v3 hybrid-min verification, replay init/clamp, boundary
  init/lookup/verification) is present exactly once in its supported form;
  a stale stage marker, a partial stage, edited verification logic, a
  duplicated stage, a competing module binding for an owned helper, or
  unrecognized legacy drift exits non-zero with the file untouched.
  At `9a3aca4`, stock installation succeeded, but full stock qualification
  was incomplete: automatic-budget OFF could not fit the configured 850k
  context; automatic-budget ON answered the 814,571-token cold prompt but
  recorded two preemptions in that group; the repeat was not completed.
  A separate fixed-14-GiB comparison hit preemption, memory-floor
  and OFF sequence-reference stops; failed runs are retained. Default remains
  off. TP=4 GPU execution and tensor-level numerical parity remain
  unqualified. A 2026-09-23 TP=3 boot on this kit, with the flag set on all
  three ranks, selected the derived 640-token padded page (MLA block 2560,
  not the TP=2 896) and logged boundary lookup on drafter group 6. That is
  geometry confirmation, not the TP=2 reservation or repeat-TTFT result.
  `.env.tp3.example` documents the opt-in and leaves it commented.
  See README for both profiles, receipt hash, baseline drift and limitations.
- Experimental TP2/SM121 KDA large-M BF16 prefill path
  (`GLM53_KDA_BF16_LARGE_M`, default `0`): uses retained FP8-derived BF16
  weights for scheduled M > 512, with approximately 3.26 GiB extra retained
  weights per rank. BF16-only eligibility fails closed at load; smaller
  matrices keep Marlin. See `docs/kda-bf16-large-m.md` for qualification,
  numerical uncertainty, and memory/cache tradeoffs. (#233)
- Same opt-in on TP=3: `start-tp3.sh` validates `0`/`1` and forwards the
  flag to all three ranks (FAST/FAT stay unset). Overlay retains the
  TP3-local `[8726x4096]` in_proj after the 64→66 head pad (~2.26 GiB/rank
  theoretical; live boot 34/34 layers at +68.2 MiB/rank). Fail-closed on
  any other world size or shape. Serving-speed receipts remain TP2-only.
- Opt-in bounded final sparse-MLA attention call for TP=4
  (`VLLM_SM120_SPARSE_MLA_SLICE_TOKENS`, default `0`): `overlay/patch_sparse_mla_slice.py`
  slices the final `flashinfer_trtllm_batch_decode_with_kv_cache_mla` call into
  <=64 query rows on every rank, hash-pinned to the shipped backend; `start-tp4.sh`
  validates (`0`/`64`), stages and applies it. Mitigation for the all-rank stall
  in #128 / #159, ported with attribution from the MIT recipe qualified on a 4x
  GB10 kit; it does not fix the underlying race. `start.sh` / `start-tp3.sh`
  untouched. (#223)
- Opt-in SM121 **thin-decode** kernels for the EXL3 routed experts
  (`GLM53_EXL3_MOE_FAST`, default `0`): `overlay/patch_exl3_decode_pipeline.py`
  adds two K4/N256 fast kernels (shared / independent gate-up input transform)
  and a `glm53_fast_moe_version` symbol to `exllamav3_ext` at image build time,
  and the patched native host dispatch selects them only for K == 4,
  N256-compatible dimensions and SM121 when the flag is literally `1`.
  `overlay/exl3.py` adds the load-time gate and, in fast mode only, the gate/up
  SUH pointer alias that carries the transform-reuse proof; the alias needs the
  all-expert `torch.equal` comparison of the packed SUH scales. With the flag
  off the module builds the same pointer tables and runs the same stock kernels
  as before. `FAST=1` fails closed at model load when the image has no native
  kernels or the fused `exl3_moe` path is unavailable, and the launcher rejects
  anything but `0`/`1` before `restart` stops the pair. TP=2 `start.sh` only;
  `start-tp3.sh` keeps unsetting it. Split out of #182 and rebased onto the
  current `main`.
  Kernel evidence (exact head, author-reported under #182): 119/119 parity
  cases at rel RMSE ~1e-8, 0 `compute-sanitizer` errors, 0 local-memory reloads
  in the fast kernel vs 39 in stock, +8–17% layer latency. Serving speed is
  measured twice: historically at the pre-rebase head (+4.8–8.4% decode A/B,
  A2 within ±2.4%) and now on current `main` — maintainer-measured TheGrill
  v0.3.0 A/B/A2 at head `3d6ffbd` (image `sha256:f267534b…`, fresh boot per arm,
  `glm-routine-decode-v3`) gives **+7.71% / +8.88% / +14.59%** median decode
  rate in the three range-resolved cells against A2-vs-A drift of −0.31% /
  +0.30% / −3.54%, with two cells withheld by the tool's range-overlap rule
  (descriptive; no PASS envelope). The hash-frozen numerical study is formally
  **inconclusive** — both predeclared control self-tests (the absolute-KL gate
  and the tau-transfer bound) fail on unchanged stock repeats — so this stays
  opt-in; see `docs/sm121-perf-paths.md`.

### Changed

- Correct hybrid APC checkpoint alignment to use the resolved scheduler LCM
  (3584 in the tested layout), rather than the minimum group's 64-token size,
  while retaining bounded small-step progress and the hash-grain prompt tail.
  Apply the scheduler EAGLE backoff only when a participating non-SWA group
  needs it: SWA-only drafting preserves the last full target checkpoint under
  production 7168-token grants; no-SWA MTP retains upstream behavior. The
  extra prefill step versus improved short-suffix hits has unmeasured GPU cost.
- Limit the partial-hit capability veto to prefix-participating groups:
  nonparticipating KpoolTail scratch no longer blocks compatible target states.
  No unsafe SWA exemption is introduced; incompatible participating SWA still vetoes.
  Preserve a reusable preceding coarse checkpoint before larger draft replay
  backoff, require four fresh prompt tokens for Kpool scratch, and retain SWA
  retention policy. `PREFIX_MATCH_UNIT` stays empty by default; explicit `64` is
  supported for the tested geometry. CPU real-function metadata/composition
  checks are not GPU state/logit parity or TTFT measurements; GPU qualification
  remains outstanding. Reproduction uses the Dockerfile-pinned source probe
  described in [README](README.md#reproduce-the-pinned-source-cpu-probe).
  The four-token Kpool replay floor remains conservative and kernel-unverified;
  coarse-only lookup can lose a whole page within three tokens of a boundary.
- TP3/TP4 now preserve an explicitly exported `LOAD_FORMAT=` through shared and
  topology env files, so callers can select auto without changing loader defaults.
- Repinned the cooperative-MoE profile generators' `overlay/exl3.py` digest
  (`extensions/cooperative_moe/prepare_profile.py` and
  `extensions/cooperative_moe/tp3/prepare_profile.py`) after reviewing the
  thin-decode additions above. With `GLM53_EXL3_MOE_FAST` unset — the launcher
  default, and what `start-tp3.sh` enforces by unsetting it — the module builds
  the same pointer tables and takes the same paths it did before, so a generated
  TP2/TP3 adapter is unchanged, and both generators still refuse any other
  content. `docs/cooperative-moe-handoff.md` records the new pin.
- Repinned those generators again after the KDA large-M path gained the
  TP3-local `[8726x4096]` shape. With `GLM53_KDA_BF16_LARGE_M` unset the
  module still takes the existing Marlin/base paths.

### Fixed

- `start.sh`, `start-tp3.sh`, and `start-tp4.sh`: a GHCR pull could replace a
  local image whose recipe stamp already matched the repo, then launch the
  published image (no locally compiled artifacts; `GLM53_EXL3_MOE_FAST=1`
  then fails closed at load). The pull is now held aside and adopted only
  when its stamp matches too. A mismatch restores the local tag and does not
  rebuild, so a later restart does not build on every boot. Workers are
  shipped that local image instead of pulling the mismatched GHCR tag.
  `SKIP_BUILD=1` still keeps GHCR. Supersedes the rebuild-after-pull approach
  in #248.
- `overlay/patch_mamba_align_state_free.py`: release every superseded
  Mamba "align" state block. The manager tracked one superseded block per
  request and released it only once the processed prefix (computed minus
  in-flight) had passed it; with async scheduling and chunks larger than the
  Mamba block the tracker was overwritten first and the block stayed
  referenced for the request's lifetime, accumulating throughout long
  prefills despite bounded live-state needs. Superseded indices are now
  a bounded per-request list released under the unchanged predicate.
  `MambaSpec.max_memory_usage_bytes`
  reserves `1 + max_concurrent_batches + num_speculative_blocks` pages in
  align mode (10 here, was 9), matching the resident peak.
- `overlay/patch_mamba_align_chunking.py` is the sole owner of Mamba checkpoint
  alignment and EAGLE back-off. Use the resolved scheduler LCM rather than the
  drafter's smaller page, preserve smaller private Mamba state boundaries, and
  let positive sub-page grants progress without changing decode-floor policy.
  Back off only for participating non-SWA EAGLE groups. Migrate the retained
  alignment overlay without stacking a second scheduler implementation.
- Exclude nonparticipating scratch groups from fine-hit capability checks.
  Preserve compact DFlash boundary lookup while reserving four fresh Kpool
  tokens and retrying the preceding shared checkpoint when a partial target
  hit lacks a reusable drafter window. Existing PR238 GPU receipts below do
  not qualify these additional fine-grained APC changes.
- Under `GLM53_DRAFT_KV_COMPACT=1`, the DFlash drafter manager drops the
  extra EAGLE lookahead block from each retained window; the KV-capacity
  log costs the boundary lookup accordingly (`lookup=boundary`). The
  logger migrates only the exact published prior helper revision and
  refuses unknown edits without overwriting them.
- Follow-up qualification of these fixes: 53 focused CPU tests passed
  without skips, plus source composition and native CLI checks on both
  supported images. The custom OFF/ON/OFF matrix, stock ON full workload
  and fixed regression holdout completed 500 requests, 140 reference
  answers and four rollover checks without preemptions or safety stops.
  Stock ON completed both 814,571-token cold and repeated prompts; stock
  OFF still cannot fit the unchanged defaults (13.56 GiB required versus
  10.77 GiB available). The initial approximately 3.3% prose/long-C2
  regressions are retained; the fixed holdout measured 0.93% lower prose
  throughput and 1.18% higher cold long-C2 wall time. No universal decode
  speedup or VRAM reduction is claimed. Default remains off; see README
  for the full matrix, cache-residency limits and source-pinned receipt.

## [1.6.0] — 2026-09-17

TP3 ABI2 cooperative MoE and opt-in FlashKDA, ABLIT off, and new sparkDash
prose decode tables for this 2× and 3× kit.

### Added

- Opt-in TP3 expert-parallel cooperative kernel (ABI2, 32/64-row) and
  FlashKDA prefill adaptation (`HAREM_KDA_FLASHKDA`, default 0) on
  `start-tp3.sh` only. Incomplete ABI2 bundles are refused before restart
  teardown. (#208, merged as #212)
- Staging of verified TP3 bundles on all ranks; optional
  `examples/tp3-throughput.env` profile.

### Changed

- `start-tp3.sh` forces `ABLIT=0` after sourcing `.env` (opt in with
  `ABLIT=1 ./start-tp3.sh` or `ABLIT=1` in `.env.tp3`). It does not inherit
  the TP2 `EXL3_OVERLAY_HOST` or FAST/FAT flags. #207 SWA retention stays.
- `start.sh` also forces `ABLIT=0` after `.env`; `ABLIT=1 ./start.sh` still
  opts in.
- README sparkDash prose decode (thinking off, this kit, 2026-09-17):
  **TP2** ×1 37.1 / 36.1 str (TTFT 333 ms), ×2 51.1 / 25.0 (365 ms),
  ×3 65.8 / 22.3 (405 ms), ×4 75.3 / 19.4 (401 ms).
  **TP3** 512 tok ×1 40.1 (255 ms), ×2 56.6 / 28.7 (411 ms),
  ×3 75.5 / 25.5 (323 ms), ×4 88.4 / 22.8 (351 ms).

## [1.5.0] — 2026-09-17

Cooperative decode MoE (geometry 1) plus DFlash prefix-cache retention on TP3/TP4.

### Added

- Opt-in cooperative EXL3 MoE overlay for decode (`extensions/cooperative_moe/`,
  geometry 1: H=4096, local I=1024, top-k 8, K4 MCG, 1–32 rows). Stock
  `overlay/exl3.py` and the image stay unchanged until `EXL3_OVERLAY_HOST` points
  at a generated overlay. E3 grouped prefill is unchanged. (#202)
- Operator notes: `docs/astra-results.md`, `docs/cooperative-moe-handoff.md`.

### Changed

- `start-tp3.sh` / `start.sh` copy `runtime.py` and `cooperative_moe.so` into every
  rank's vLLM cache so a generated overlay does not die on
  `FileNotFoundError: /root/.cache/vllm/cooperative_moe/runtime.py`. (#202)

### Fixed

- DFlash skipped-window LRU inversion on TP3 and TP4. `start-tp3.sh` and
  `start-tp4.sh` apply `patch_apc_per_group_retention.py` and forward
  `GLM53_APC_RETENTION_INTERVAL_SWA`. Explicit `0` keeps only the drafter window
  boundary so a finished long chat is not evicted by hashed skipped DFlash
  blocks. MLA/mamba stay dense. Examples set `GLM53_APC_RETENTION_INTERVAL_SWA=0`.
  (#207)

### Decode (this 2× Spark kit)

Matched A/B/A serving at 850k context, 14 GiB / 883,552-token FP8 KV:

| Job | Stock | Geometry 1 | Gain |
|---|---:|---:|---:|
| Structured ×1 | 72.03 tok/s (70.63–72.92) | 77.29 tok/s (75.66–79.01) | **+7.3%** |
| Structured ×2 aggregate | 113.91 tok/s | 124.46 tok/s | **+9.3%** |
| Isolated 32-row kernel | 1.183 ms | 0.997 ms | faster |

Further geometries did not beat geometry 1.

---

## [1.4.0] — 2026-09-16

Fair mixed-prefill as the TP2 default, 3-node NFS serve, InstantTensor image, and
a batch of community launcher/ops PRs.

### Added

- Fair v5 mixed-prefill scheduler (`overlay/patch_scheduler_decode_floor.py`):
  service-time share, largest step-fitting chunk, decode first. Opt-in then
  defaulted on TP2; later defaulted on TP3/TP4 as well. (#186, #188)
- 3-node launcher `start-tp3.sh` with NFS weight share and a 35 GiB KV cap after
  a head OOM. (#184)
- InstantTensor 0.2.0 baked into the overlay image; launchers default `IMAGE` to
  GHCR `:exl3-instanttensor` and pin NCCL channels / load format. (#200, #201)
- Omitted-only `DEFAULT_MAX_NEW_TOKENS` (legacy completion default 16 is covered;
  explicit limits still win). (#51)
- Cache-reset endpoint. (#37)
- Extra launcher environment forwarding. (#81)
- Concurrency ladder harness and receipts. (#82)
- APC no-store gate. (#95)
- Prebuilt abliterated-model preset. (#137)
- Tool-concurrency bench and `spark_doctor.sh`. (#41)
- EXL3 SM121 kernel lab. (#75)
- KV capacity boot log. (#94)
- Unified-memory preflight. (#39)

### Changed

- Fair share default 0.30 and mixed-step cap 2000 ms. (#194)
- Title/docs: 2–4× DGX Spark support. (#189)
- Long-prefill warmup and linear prompt construction. (#170)
- Dated TP4 measurements; dropped a withdrawn DFlash2/packet-loss caveat. (#115)

### Fixed

- Decode-floor v5 verify is position-independent, so a healthy patched scheduler
  survives container restart when `patch_adaptive_k.py` sits between the helper
  and the old cuda_graph import anchor. (#198)
- Mixed-prefill skip restored as the TP3/TP4 default until fair was re-enabled
  on those launchers. (#187, then #188)
- Scheduler test overlay path beside the image copy. (#192)
- `pipefail`-safe container health. (#42)
- GB10 UVM livelock runbook. (#70)

---

## [1.3.0] — 2026-09-12

Decode-path knobs, per-group APC retention, and launcher hardening.

### Added

- Opt-in adaptive-k (ema 2, 4, 7) plus FP8 dense projections, env-gated.
  (#139) Later enabled safely by default. (#169)
- Per-group APC retention / DFlash replay-free ordering
  (`patch_apc_per_group_retention.py`, `GLM53_APC_RETENTION_INTERVAL[_SWA]`).
  (#130)
- Opt-in default reasoning effort. (#158)

### Changed

- Long-prefill token threshold is an explicit opt-in (empty omits the flag).
  (#157)
- Default per-prompt image cap raised 4 → 100, then per-image tokens capped so
  a chat video cannot OOM the host. (#146, #183)
- Recommend a 14 GiB KV cap for the FP8 opt-in, not 15. (#146)

### Fixed

- Reasoning Effort head line gated on thinking again so thinking-off stays
  prefix-cache stable. (#150)
- Caller exports preserved across dotenv load. (#161)
- `PYTORCH_CUDA_ALLOC_CONF` overridable. (#175)
- Host python with jinja2 for chat-template validation. (#173)
- RoCE GID validated on every listed CX7 HCA. (#172)
- Decode bench sends Bearer auth on keyed runs. (#136)
- Long-prefill metadata kernels included in boot shape warmup. (#170-era warmup)

---

## [1.2.0] — 2026-09-07

E2 then E3 fat-expert prefill, experimental TP4, AGPL-3.0.

### Added

- E2 fat-expert prefill kernel (`EXL3_FAT_KERNEL`), MNBT 7168, rebuild on
  overlay drift. (#77)
- E3 grouped fat-expert MoE (`EXL3_FAT_GROUPED`, now the launcher default):
  three GPU-driven launches per MoE layer (gather, gate/up + SwiGLU, down +
  scatter) from device-side segment tables — no per-expert host loop, no weight
  repacking, no host sync. (`overlay/exl3_fat_moe.cu`, `overlay/exl3.py`)
- Experimental `start-tp4.sh` / `.env.tp4`. (#105)
- Indexer-workspace rightsizing. (#86)
- Cold-prefill harness. (#71)
- Issue/PR templates. (#126)

### Changed

- License MIT → AGPL-3.0. (#134)
- Shipped context **1M → 900k then 850k**, `GPU_MEM_UTIL` 0.87 → 0.85,
  `GLM53_INDEXER_WORKSPACE=rightsize`, `EXL3_TEMP_ROWS_FUSED` 128 → 32 (E3) /
  256 (E2).
- Stop re-shipping the GHCR image to the worker every run. (#134 follow-up)
- Spin-wait 16 ms. (#96)

### Prefill (this 2× GB10 kit, E3 vs E2)

Cold prefill **+37–45%**; decode unchanged (E3 never runs on decode-sized steps).

| Prompt | E2 tok/s | E3 tok/s | Gain |
|---|---:|---:|---:|
| ~16k | 1,155 | 1,578 | **+37%** |
| ~128k | ~1,150 | 1,629 | **+38–42%** |
| ~256k | 1,087 | 1,576 | **+45%** |

Isolated 7,168-token MoE layer: **77–91 ms (E2) → 31 ms (E3)**. E3 error vs the
LinearEXL3 reference matches E2 (remaining E3/E2 delta is atomic accumulation
order). Receipts: `logs/overnight-20260906T164059Z/`.

E3 charges a ~560 MiB fat-row scratch to the KV budget, which is why 1M / util
0.87 no longer fits one full-length request on this kit.

### Fixed

- Reasoning Effort emitted unconditionally so the prefix cache does not break.
  (#63)
- Numeric knob validation. (#38)

---

## [1.1.0] — 2026-08-30

Bring-up robustness, prefix cache correctness, and first multi-kit knobs.

### Added

- DFlash2 draft TP default 2 (drafter shards across tensor parallel). (#48)
- Optional `VLLM_API_KEY` Bearer auth. (#30)
- Other-kits NCCL GID preflight. (#16)
- Per-rank GID. (#26)
- CUDA-graph capture-size estimate opt-out. (#25)
- Bring-up robustness. (#34)
- Prefix-cache bench. (#33)
- MNBT=2048 cold-prefill receipts. (#40)
- C4 idle-prefill keep documented as the live recipe. (#49)

### Fixed

- Hybrid APC: keep MLA prefix hits when DFlash2's EAGLE drop would zero them.
  Hits remain 3584-token aligned. (#18)
- K-pool tail slot mapping pinned to the one-block circular scratch. (#50)
- Do not re-ship the GHCR image when the worker already has it. (#9)
- `MAX_NUM_SEQS` inline override. (#28)
- xgrammar structured-output issue. (#21)
- Abliterated overlay restored onto a dedicated path, then the AblitBrench
  ping/sync dropped from this recipe. (#45, #46)

---

## [1.0.0] — 2026-08-28

Initial public recipe: GLM-5.3-Flash EXL3 4 bpw on 2× NVIDIA GB10 (SM121).

### Added

- `start.sh` / `stop.sh` two-node serve over CX7, native `sm_121a` cubins,
  OpenAI API on `:8888`, served id `GLM-5.3-Flash-EXL3`.
- Public GHCR image pull and Mia-AiLab Hub mirror of
  `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (uniform-K4 EXL3/TR3, 4 bpw).
- DFlash2 k=7 speculator (`incoai/GLM-5.3-Flash-DFlash2`), FLASH_ATTN draft.
- CUDA graphs on fused EXL3.
- Image/video placeholders, GB10 long-prefill chunk size, glm46v video timestamps.
- Head-only `download.sh`.
- Independent KLD panel for the 4 bpw checkpoint.
- sparkDash Structured/Code decode receipt (~62.9 tok/s at ×1 on the day).

### Changed

- Default context 900k (util 0.87 → ~982k-token KV pool), then 1M once padded
  DFlash2/MLA slot-share allocated it so three long sessions fit.

### Fixed

- Thinking-off chat template. (#1)
- Client stop strings dormant until `</think>`. (#2, #4)
- Persist Triton/TileLang caches and warm DFlash2 shapes after `/health`. (#3)

Weights keep their own terms. The serve recipe later moved from MIT to AGPL-3.0
in 1.2.0.
