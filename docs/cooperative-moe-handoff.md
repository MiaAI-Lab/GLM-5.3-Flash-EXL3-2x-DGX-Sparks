# Cooperative MoE handoff — 2026-09-16

**Status, 16:30 local (checkpoint 5):** C1 is live on the two-node TP2 serve.
`/health` 200 at http://127.0.0.1:8888, model `GLM-5.3-Flash-EXL3`. Both ranks
log `geometry=1 (A-wide/B-wide)`, `eligible=42`. E3 grouped prefill is still
on. KV still 883,552 tokens / 14 GiB. Default image and recipe
`overlay/exl3.py` are unchanged; this is an opt-in overlay. Results:
[astra-results.md](astra-results.md). C1 **beats matched stock**; later
kernels **did not** beat the initial geo1 `.so`.

This is the operator note. Shape constants and the GPU gate live in
[cooperative-moe.md](cooperative-moe.md). The first-time build/deploy steps
are [cooperative-moe-quickstart.md](cooperative-moe-quickstart.md).

## What landed

A decode-only cooperative EXL3 MoE path, specialized for this recipe, modeled
on [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks#8](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/pull/8)
(`~/NewModels/DeepSeek-V4.1-Flash-EXL3-2.9bpw/extensions/cooperative_moe/`).

The DS4.1 `.so` is compile-time locked to DeepSeek-V4.1 (H=5120, I=1152,
top-k 6, K2/K3 mul1) and is **not** loaded here. GLM talks to a new
`glm53_coop_*` ABI.

| Piece | Value |
|---|---|
| Hidden / local I / top-k | 4096 / **1024** (packed 2048, TP2 shard) / 8 |
| Quant | K4 MCG, cb=1, no mul1 |
| Decode rows | 1–32 (DFlash2 k=7 → 8/seq; `MAX_NUM_SEQS=4` → 32) |
| Prefill / fat | stock E3 (`tokens > EXL3_TEMP_ROWS_FUSED`) |
| Geometry | **1 = A-wide / B-wide** (not Blackwell auto) |
| Image | `glm53-flash-sm121:e3-20260907-itensor` |
| Activation | `EXL3_OVERLAY_HOST` → generated overlay; `install(..., enabled=True)` |

`GLM53_COOPERATIVE_MOE=1` alone does not deploy the binary or select the
overlay.

## Live layout

| Where | Path |
|---|---|
| Recipe | `~/NewModels/glm-5.3-flash-sm120` |
| Overlay (`.env`) | `/home/mia/.cache/vllm-glm53-flash/cooperative_moe/exl3-cooperative.py` |
| Native + adapter (head) | `/home/mia/.cache/vllm-glm53-flash/cooperative_moe/{cooperative_moe.so,runtime.py}` |
| Same files (worker) | `/home/zurih/.cache/vllm-glm53-flash/cooperative_moe/` |
| Container mount | host vLLM cache → `/root/.cache/vllm`; overlay → `/opt/glm53/exl3.py` |
| Rollback `.env` | `logs/cooperative-moe.eFKgLt/original.env` (mode 600) |
| Decode receipts | `logs/coop-decode-{structured,prose}-geo{1,2}.json` |

Pins (`extensions/cooperative_moe/prepare_profile.py`):

| Artifact | sha256 |
|---|---|
| stock `overlay/exl3.py` | `fe07cf3cd1928d0a189e793579a7d2dd529f75617a55f620ee14a0a9d3b20121` |
| `cooperative_moe.so` (unchanged geo1 kernel) | `aa3fe5e9387c7e0d42d685fb2ca8a5fb959ad956600236baac078a9076c17a1c` |
| `runtime.py` C1 (live) | `9427f6a65def09ebdbea231e42f735236e145f3d02c19cf5e5276c2e704ce1ca` |
| `runtime.py` original geo1 adapter | `71111c230c2519d473cc77f215703fbb12b1be2659cde66aa166c98ea3952fc6` |
| generated overlay (stock + footer) | `5f28f5543629043117c7506cd3cf47cc8fa66fcd1153ddf67c696bf3e5484c4d` |

The rows above describe the historical C1 deployment (2026-09-16), not the
current build acceptance policy. The TP2 generator now pins stock overlay
`241ebf56a01ddf940ba7015f31355a1dad810b391f9d77ef79fa978f42ccbab3`,
the ExLlamaV3 commit and combined source tree, and each native/runtime source
hash. It deliberately does not pin the complete ELF: CUDA/host toolchain
updates may change it. `build-manifest.json` carries compiler provenance, and
runtime validates ABI/layout/occupancy. Regenerate from current
`overlay/exl3.py`; an older generated overlay lacks the dense loader. TP3
remains for ordinary packs only. This source compatibility review is not a new
GPU qualification of the combined dense/cooperative profile.

The overlay footer `run_path`s `/root/.cache/vllm/cooperative_moe/runtime.py`.
Changing geometry is a `runtime.py` copy to **both** caches plus restart. It
does not require regenerating the overlay or rebuilding the `.so` (all three
geometries are already in the binary). `prepare_profile.py` will refuse a
`runtime.py` whose digest is not the pinned `ADAPTER_SHA`.

`start.sh` requires the `Exl3LinearMethod` identity and the stock `Exl3Config`
closer `        )` in the generated overlay; these guards reject stale or truncated copies.

## Why the first boot looked like “no benefit”

The first live overlay used **geometry 2** (Blackwell auto: A-wide, B-narrow)
because `I/16 = 64`. That sizes the down grid as `rows * topk * (H/32)` —
8192 blocks at DFlash2 ×1 — versus 2048 with wide B (`H/128`).

Stock fused `exl3_moe` on this kit already streams near the measured copy
ceiling (~210–240 GB/s vs ~231 GB/s). Overnight decode profiling put fused
MoE at ~50 ms of a ~121 ms cycle. DFlash2 ×1 is 8 tokens × 8 top-k into 288
local experts: almost no expert-weight reuse, so grouping cannot buy much
bandwidth. Extra narrow-B blocks were overhead on top of that. DS4.1’s
working serve hard-coded geometry 1 (both wide) even though auto would have
picked narrow B there too.

GPU util during hashmap decode stayed ~96% (kernel-bound, not ctypes/launch).
Coop **was** on the decode path; the tile choice was wrong.

Do not expect DS4.1’s +25–33%. That stack was K2/K3 against a weaker fused
kernel. GLM K4 fused is already good.

## Decode numbers (this kit, 2026-09-16)

`tests/bench_decode.py`, temp 0, thinking off, adaptive-k + dense FP8, E3 on,
CUDA graphs. Two warmed runs. Structured 200 tok; prose (hash-map) 400 tok.

| Workload | Stock | Coop geo 2 (auto) | Coop geo 1 (live) |
|---|---:|---:|---:|
| Structured ×1 | 62.9 sparkDash (2026-08-28) | 78.0 | **80.0** (accept 7.0/7) |
| Hash-map prose ×1 | 32.1 (2026-09-08) | 34.8 | **37.4** (+16% vs 32.1) |

Structured 62.9 is an older sparkDash serve (no adaptive-k / dense FP8).
Overnight 2026-09-08 counting with those flags was ~63–70, not 62.9. Prose
32.1 **is** the matching baseline (same flags). Geo 1 vs geo 2 is a paired
measurement on this overlay.

Prefill is not a decode metric; E3 is unchanged. sparkDash prose ×2 (41.2 agg)
was not rerun.

## Confirm it is still this profile

```bash
curl -fsS --max-time 8 http://127.0.0.1:8888/health
docker logs glm53-exl3-head 2>&1 | grep -F 'Fixed-shape cooperative MoE wrappers installed'
ssh -o BatchMode=yes zurih@10.0.0.2 \
  "docker logs glm53-exl3-worker 2>&1 | grep -F 'Fixed-shape cooperative MoE wrappers installed'"
sha256sum ~/.cache/vllm-glm53-flash/cooperative_moe/runtime.py \
          ~/.cache/vllm-glm53-flash/cooperative_moe/cooperative_moe.so \
          ~/.cache/vllm-glm53-flash/cooperative_moe/build-manifest.json
```

Wanted: `geometry=1 (A-wide/B-wide)` on **both** ranks, a
`build-manifest.json` accepted by the current generator, and identical staged
files between head and worker. The historical C1 digests above identify the
old measured deployment; a new source-built ELF may have a different digest.

Re-bench:

```bash
python3 tests/bench_decode.py --phase structured --structured --runs 2 \
  --max-tokens 200 --out logs/coop-decode-structured.json
python3 tests/bench_decode.py --phase prose --runs 2 --max-tokens 400 \
  --skip-coherence --out logs/coop-decode-prose.json
```

Host checks (no GPU): `python3 extensions/cooperative_moe/test_dispatch.py`
and `test_profile.py`. The packaged GPU gate is
`extensions/cooperative_moe/test_cuda_integration.py` with
`GLM53_COOP_MAINTENANCE_TEST=1` in a maintenance container — **stop GLM
first**. Head gate passed 18/18 on geometry 2. Serving now uses geometry 1
(same kernels, wide B tile). Worker-node gate was not repeated.

## Roll back (GLM only)

```bash
logs/astra-goal-20260916/rollback-c1.sh     # C1 winner
logs/astra-goal-20260916/rollback-geo1.sh   # original geo1 adapter
logs/astra-goal-20260916/rollback-stock.sh  # stock overlay .env
```

Earlier copy: `logs/cooperative-moe.eFKgLt/original.env`. Staged
`cooperative_moe/` files do nothing unless `EXL3_OVERLAY_HOST` selects them.
Do not commit `.env` or weights. Do not stop DeepSeek.

## Change geometry or rebuild

`GEOMETRY` in `extensions/cooperative_moe/runtime.py`: `0` both-narrow, `1`
both-wide (current), `2` A-wide/B-narrow (Blackwell auto). After an edit:

1. `sha256sum` the new `runtime.py` and update `ADAPTER_SHA` in
   `prepare_profile.py`.
2. `install` it to both nodes’ `~/.cache/vllm-glm53-flash/cooperative_moe/`.
3. `./start.sh restart` (`SKIP_BUILD=1 SKIP_PULL=1` already in `.env`).

Rebuild the `.so` only if native source changes. Recipe image has nvcc but
not git: archive upstream `02aef45cd681b960a00afcd0749a4ab99e6c1bfe` on the
host, then `extensions/cooperative_moe/build.sh` in the container. A new
digest is `UNVALIDATED` until the GPU gate is rerun and the pin updated.
Kernel origin is exllamav3 `58d4d732` (MIT).

NFS: `.env` has `NFS_SHARE=1`. A start with `SKIP_SYNC=1` while the worker
volume is empty will fail (`config.json` missing). Recreate the NFS volume
(`addr=10.0.22.1` on this kit) and start **without** `SKIP_SYNC`.

## Do not

- Load the DS4.1 `cooperative_moe.so` or copy its eligibility checks.
- Replace E3 / disable `EXL3_FAT_GROUPED` to “make coop faster.” Decode never
  takes that branch (`tokens <= 32`).
- Treat prefill tok/s as a decode result.
- Retry stock after a partial CUDA launch (the adapter raises instead).
- Allocate coop scratch during CUDA-graph capture (it is created in
  `process_weights_after_loading`).
- Commit `.env` or checkpoints.
- Touch a live DS4.1 EXL3 serve for this work.
- Assume TP3/EP (local I=2048, 96 experts) will hit this kernel; those
  shapes stay stock.

## Remaining

- Real-weight activation dumps vs stock are still a numerical gap from the
  original Astra review; assistant-output quality and GPU synthetic gates
  passed for C1.
- Compile-time slot-ceiling `static_assert` is in source until the next
  native rebuild (binary kept hash-stable).
- Default recipe serving is still stock unless `EXL3_OVERLAY_HOST` points at
  the generated overlay (live C1 does).
- Do not expect another large decode jump from this kernel on K4 fused; the
  leftover is attention / draft / NCCL. Geometry investigation is complete:
  geo1 was not beaten by later kernels.
