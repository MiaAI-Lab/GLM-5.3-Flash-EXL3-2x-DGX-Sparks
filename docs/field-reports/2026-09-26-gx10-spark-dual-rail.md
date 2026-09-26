# Field report: #251 on real 2× GB10 hardware + a hidden prefill cost of `GLM53_DENSE_FP8=kda`

2026-09-26 · 2× GB10 (ASUS GX10 head + DGX Spark worker), TP=2, dual-rail CX7 (MTU 9000,
`NCCL_IB_GID_INDEX` empty), driver 580.173.02. Checkpoint `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
@ `25a44fdb`, DFlash2 k=7 (`dc77ff1c`, draft TP=1), adaptive-k `ema` set `3,5,7` margin 0.5,
`GLM53_DENSE_FP8=dense,kda`, 850k context, 14 GiB KV, `MAX_NUM_SEQS=4`.

#251's description notes that no hardware boot or GPU qualification had been run. This report
adds hardware numbers for it, plus two findings from tuning the same kit.

Benchmarks: unique-salt cold prompts (0 prefix hits), thinking off; prefill = prompt tokens / TTFT
(3 warm runs per size, after one warm-up pass); decode = `tests/bench_decode.py`
(temp 0, 5 × 400 tokens, median).

## 1. #251 on hardware (main `70b2f33`, image built from the repo Dockerfile, `LOAD_FORMAT=`)

A/B against the same kit on `775a58b` (pre-#251), same `.env` apart from `IMAGE` and `LOAD_FORMAT`
(`safetensors` → empty):

| | pre-#251, `LOAD_FORMAT=safetensors` | main + #251, auto loader |
|---|---:|---:|
| `./start.sh restart` → `/v1/models` ready | 554 s | **300 s** (210 s on a warm restart) |
| swap written during the load (`pswpout` delta), head / worker | 3.66 GB / 2.75 GB | **2.8 MB / 0** |
| KV pool, concurrency per 850k request | 1.49× | **1.85×** (same 552 blocks; compact draft pages) |
| prefill 8k / 16k / 32k / 128k tok/s | 1603 / 1673 / 1700 / 1697 | 1582 / 1635 / 1662 / 1655 (−1.5…−2.5 %) |
| decode prose / code / structured tok/s | 34.6 / 58.2 / 79.2 | 33.9 / 54.0 / 78.7 |
| follow-up turn TTFT, 34k-token context (n=1) | 3.79 s | **1.48 s** |
| follow-up turn TTFT, 72k-token context (n=1) | 0.97 s | **0.69 s** |
| head `MemAvailable` min during 128k prefill | 3.4 GiB | 3.1 GiB |
| generation check (17×19), coherence probe | pass | pass |

Notes:
- The code-decode gap in the table partly reflects acceptance (4.68 vs 5.06 accepted per step).
  The prose and structured rows are within ~2 %.
- Right after boot the head showed 1.9 GiB `MemAvailable`, which recovered to ~3.3 GiB within about 2 minutes
  with no requests. Worth knowing before firing a long prompt at a fresh boot.
- Follow-up TTFT: a turn-1 prompt of N tokens, then the same conversation plus one assistant and one
  user message. One run each, so the direction is solid but the magnitudes aren't.

## 2. `GLM53_DENSE_FP8=kda` silently costs ~12 % prefill unless `GLM53_KDA_BF16_LARGE_M=1`

The README recommends `GLM53_DENSE_FP8=dense,kda` for decode. On this kit, enabling it after a
prefill-parity run dropped warm prefill by 12–13 %. Adding `GLM53_KDA_BF16_LARGE_M=1`
(#233) recovered it, and decode was unchanged (+3.3 GiB/rank):

| same image, same boot recipe | 8k | 16k | 32k | decode (short) |
|---|---:|---:|---:|---:|
| `dense,kda`, `KDA_BF16_LARGE_M=0` | 1339 | 1399 | 1426 | 44.9 |
| `dense,kda`, `KDA_BF16_LARGE_M=1` | 1505 | 1580 | 1619 | 44.7 |

Suggestion: mention the pairing next to the `GLM53_DENSE_FP8` README section, or have the
launcher warn when `kda` is in `GLM53_DENSE_FP8` and `GLM53_KDA_BF16_LARGE_M=0`.

Extending the same BF16 copy to every remaining FP8-Marlin projection (KDA `o_proj`/`f_b`/`g_b`,
dense MLP) gained only +1–3 % prefill for +1.6 GiB/rank. The head's `MemAvailable` low-water during
128k fell to 2.45 GiB, so we reverted it. Reporting it as a negative result.

## 3. The published GHCR image predates the FAST thin-decode kernel

`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` (digest `447114ee…`) has no
`glm53_fast_moe_version` symbol, so `GLM53_EXL3_MOE_FAST=1` needs a local build. Rebuilding only
the exllamav3_ext layer on top of the GHCR image (about 2 min compile), then `FAST=1`:

| bench_decode median | stock | FAST=1 | verify-step rate |
|---|---:|---:|---:|
| prose | 32.0 | 34.1 | +9.6 % |
| code | 49.1 | 57.9 | +8.9 % |
| structured | 73.2 | 79.3 | +8.1 % |

Step rate = tok/s ÷ (accepted/step + 1), which isolates the kernel from acceptance noise. That
matches the +8–15 % in `docs/sm121-perf-paths.md`. A refreshed GHCR image (or a README note that
FAST needs `BUILD=1`) would save the next person a debugging session.

## 4. MNBT 2048 → 7168 after the dual-rail fix

This kit had run MNBT 2048 since a single-rail period, when NCCL was ~35 % of the prefill trace. After the
dual-rail fix, 7168 (the maintainer default) was +4–5 % prefill at 8k–128k (1697 tok/s at 128k),
with decode unchanged. The profile of a warm 8k prefill (head rank, 5.05 s busy): routed MoE 1.61 s
(~55 TFLOPS, ~60 % of measured BF16 peak), other GEMMs 0.77, NCCL 0.53 (≈45 GB/s, wire speed),
mHC 0.51, KDA 0.45, sparse MLA 0.32, elementwise ~0.3.

---
Raw logs and JSONs are available on request.
