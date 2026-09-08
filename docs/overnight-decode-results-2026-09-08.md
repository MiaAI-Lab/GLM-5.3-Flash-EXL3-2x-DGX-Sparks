# Overnight decode run — results (2026-09-08)

Full report with every table and receipt: `logs/overnight-decode-20260907T224521Z/REPORT.md` (timeline in `PROGRESS.md`). Code on branch `overnight-decode-20260908` (worktree `.claude/worktrees/overnight-decode`, not pushed, not merged); every knob defaults OFF. The cluster was restored to the user's own `.env` (byte-identical) at the end.

## What was kept

**CORRECTION (09:10Z, resolved 10:40Z):** the garbage seen after switching the live server came from a broken ABLIT in every worktree boot (invalid `ablit/transplant` symlink inside the container → direction-orthogonalization fallback), not from adaptive-k or FP8. With the transplant restored, both are clean under the served sampling defaults, and re-measured gains match the overnight numbers: FP8 alone Silk Road +11.7 %, adaptive-k + FP8 Silk Road +37.1 %, sky +21.3 %, hash-map +22.2 %, counting +9.9 % (vs the k=7 baseline). KL proxy vs stock 0.002–0.013 nats/position. The live server now runs adaptive-k + FP8 at 850k with the KV pool capped at 15 GiB. Original text of the correction: every candidate boot ran with a broken ABLIT (the worktree's `ablit/transplant` symlink was invalid inside the container, so the direction-orthogonalization fallback edited o_proj instead of the transplant). That, not adaptive-k or FP8, produced the garbage under sampling, and it also means the candidate arms below ran with different o_proj weights than the baselines; the tok/s deltas are being re-measured with the fix. See the REPORT's post-run correction.

**1. Adaptive verification length — (see correction above; original text follows).** With the served defaults (temperature 1.0, thinking on) a multi-turn chat degenerates into garbage from the second turn, with or without FP8, while the stock server is clean on the same conversation (`LIVE-adaptk/`, `LIVE-p2a/bug/`, `RESTORE2/` in the run dir). Every overnight measurement was at temperature 0 and is valid as such. Do not ship it for chat until the sampling path is fixed.

Original write-up: `GLM53_ADAPTIVE_K=ema` (`overlay/patch_adaptive_k.py`). The drafter keeps its 8-token block; the scheduler verifies a per-step prefix chosen from a per-request EMA of accepted draft tokens (set 2,4,7, α 0.25, margin 1.0), batch-uniform, so every decode step still runs a FULL CUDA graph (extra uniform graphs for query lengths 3 and 5). Matched A/B/A at 131k, 8 runs/prompt, bootstrap 95 % CI:

| prompt | baseline (A2) tok/s | adaptive (C1) tok/s | Δ | CI |
|---|---|---|---|---|
| Silk Road (hard prose) | 17.9 | 21.7 | +21 % | [+19, +27] |
| sky/sunset (prose) | 22.1 | 24.9 | +13 % | [+7, +21] |
| hash-map (easy prose) | 24.9 | 27.3 | +10 % | [+2, +15] |
| counting (structured) | 62.9 | 63.7 | +1 % | [+1, +2] |
| code (3 prompts) | 30.9–37.4 | 35.7–39.3 | +5 … +15 % | code-2 includes 0 |

Cycle time drops from ≈ 123 to ≈ 100 ms on prose (k/step 3.1–3.9), counting stays at k=7, acceptance of the retained positions is unchanged, no NaN. Confirmed at the shipped 850k context with the shippable configuration (set 2,4,7, 12 extra FULL graphs): Silk Road +20.5 % [+16, +28], sky +12 %, hash-map +12 %, counting +1.7 % vs the k=7 baseline, no memwatch trip.

To ship it: `GLM53_ADAPTIVE_K=ema`, `GLM53_ADAPTIVE_K_SET=2,4,7`, `EXTRA_ARGS="--cudagraph-capture-sizes 1 2 3 4 5 6 8 9 10 12 15 16 20 24 32"` in `.env` (the branch's `start.sh` forwards the knobs and runs the patch on both nodes).

**2. FP8 weight-only dense projections (Marlin) — KEEP-PROVISIONAL, off by default.** `GLM53_DENSE_FP8=dense,kda` (`overlay/exl3.py`, `overlay/patch_dense_fp8.py`). Per-output-channel FP8 e4m3 of the KDA projections and dense MLPs at load time, Marlin kernel at decode (the sm_120 cubins run on this sm_121 device). Cycle time −11 to −12 ms at every k. At k=7 vs baseline: prose +12 … +19 %, counting +11 %, code +13 … +31 %; stacked on adaptive-k vs baseline: Silk Road +38 %, sky +30 %, hash-map +27 %, counting +12 %. Acceptance vectors unchanged (the drafter sees no drift). It changes target numerics (≈ 2.6 % FP8 rounding per weight), so it stays off until a KLD panel is run; note also that vLLM hands the freed GPU memory to the KV pool, which lowers host headroom by ≈ 1 GiB unless the pool is capped.

## What was rejected or skipped

- **Decode MoE kernel (E3 grouped at M=8):** the grouped path is 6–13 % *slower* than the fused `exl3_moe` kernel at decode shapes (microbench, parity OK); the fused kernel already streams at ≈ 210–240 GB/s, i.e. near the 231 GB/s device copy ceiling. No live A/B, no kernel written.
- **Confidence-driven per-step length (v2):** not possible in this scheduler: the async scheduler fixes the next step's draft slots before the current drafts exist, and the worker-side alternative is the in-tree adaptive-verification path that the uniform-batch attention backends cannot run.
- **NCCL LL128:** ceiling ≈ 1.5 % of the cycle (all-reduces are 6.8 ms/step), not resolvable with 4 runs; the repo also has no `NCCL_PROTO` injection. Skipped.
- **ABLIT=0 acceptance A/B (informational):** without the ablation the drafter is accepted more on hash-map (+13 % tok/s) and code-0 (+12 %), within noise elsewhere; cycle time identical. The ablation costs 0–13 % of tok/s in draft acceptance. `ABLIT=1` stays the default.

## Findings worth knowing

- **Where the 121 ms cycle goes (profiled, both ranks):** dense BF16 GEMMs 52.7 ms (45 %, ≈ 170 GB/s, the sm80 CUTLASS WMMA small-M kernel), fused MoE 50.3 ms (43 %), NCCL 6.8 ms, everything else ≈ 8 ms; the GPU is busy 99 % of the time — there is no meaningful CPU/launch gap. The report's "24 ms residual" was GPU time inside the two big buckets.
- **The served stack is not run-to-run deterministic at temperature 0:** identical prose/code prompts diverge after a few words between runs of the same configuration (counting is deterministic). Any "greedy token agreement" gate is meaningless on this kit until that is fixed (probable cause: fp32 split-K/atomic reductions).
- The first candidate boot showed that this kit runs vLLM's `AsyncScheduler`, which never calls `update_draft_token_ids`; the hook had to move to `schedule()`.
