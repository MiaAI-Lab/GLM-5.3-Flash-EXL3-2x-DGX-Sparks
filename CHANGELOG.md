# Changelog

All notable changes to this GLM-5.3-Flash EXL3 serve recipe are documented here.

Versions **1.0.0–1.5.0** are retrospective SemVer labels over merged `main` history.
There were no git tags for 1.0.0–1.4.0; 1.5.0 is the first cut named as a release.
Dates are merge dates on `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`.

## [Unreleased]

### Changed

- `start.sh` and `start-tp3.sh` default `ABLIT=0` and ignore `ABLIT=1` in `.env`.
  Opt in with `ABLIT=1 ./start.sh` or `ABLIT=1` in `.env.tp3`. Other `.env`
  knobs are unchanged, including `GLM53_APC_RETENTION_INTERVAL_SWA` from #207.

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
