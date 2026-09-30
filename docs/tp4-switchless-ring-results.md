# TP=4 switchless-ring measurements (GLM-5.3 Flash EXL3)

Hardware validation of the opt-in ring path in `./start-tp4.sh`.
These numbers are **not** the switched CRS812 TP=4 table in the README, and
they are **not** `tests/bench_decode.py` medians (that harness hard-codes a
model id this serve does not use). Protocol: idle server, short prompts,
streaming `/v1/chat/completions`, thinking off unless noted, temperature 0.

Date: 2026-09-16 (Europe/Madrid evening). Image pin
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`
digest `sha256:447114ee77d14c9b4732ee23978ada2a0ee9027868a231d6fd42700a8b25be1d`
(id `sha256:ef9f5013c41adf93…`). Weights
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` revision `25a44fdbf16862a46b7cc9921142c6c81350af2f`
(120 shards / 163.65 GiB) plus `incoai/GLM-5.3-Flash-DFlash2` snapshot `bf582e4e…`.
Served id `GLM-5.3-Flash-EXL3`. Fabric: 4× GB10, 4× 200 GbE DAC ring, MTU 9000,
Gloo on the 1G management NIC. Driver `580.173.02`. Host NCCL 2.30.7 with
`SWITCHLESS_RING_ONLY`.

JSON copy: [tp4-switchless-ring-20260916.json](tp4-switchless-ring-20260916.json).

## Correctness

| Check | Result |
|---|---|
| `/v1/models` | `GLM-5.3-Flash-EXL3`, `max_model_len=262144` |
| structured short decode | HTTP 200, coherent |
| native tool call | `get_weather(Vigo)` parsed |
| thinking | `17*19=323` with reasoning |
| C8 concurrent short | 8/8 completed after DFlash k=3 + 8 NCCL channels |
| KV after short tests | ~0 %, 0 preemptions |
| host RAM free during tests | 17–21 GiB |

## Idle short-prompt decode (production profile)

Profile: 256k context, 8 seqs, GMU 0.75, MNBT 2048, DFlash2 k=3, **draft TP=4**,
`NCCL_NCHANNELS=8`, mixed-prefill `fair`.

| Workload | C1 | C4 agg | C8 agg |
|---|---:|---:|---:|
| structured | ~65 tok/s, TTFT 0.33 s | ~145 tok/s | ~179–195 tok/s; TTFT p50 1.3 s / p95 2.7 s |
| prose | ~35 tok/s | (not separately tabulated) | prose gains less than structured; DFlash accepts worse on free text |

Same short prompt, DFlash off / 4 NCCL channels / 4 seqs (commissioning):

| Workload | C1 | C4 agg | C8 |
|---|---:|---:|---:|
| structured | ~24 tok/s | ~72 tok/s | ~73 tok/s, TTFT 8.8 s (did not scale past 4) |

KV pool at the 256k / GMU 0.75 profile: ~3.40M tokens (~13× at 256k). Decode
is fabric-bound on the four-hop ring; extra KV does not buy tok/s.

## Ablations on this ring (do not copy blindly)

| Change | Result | Kept? |
|---|---|---|
| `DFLASH_DRAFT_TP=1` (rank 0 only) | C1 ~64–65 (flat); C4 145→130; C8 ~179→176 | **No** — reverted to draft TP=4 |
| `GPU_MEM_UTIL=0.85` | `NV_ERR_NO_MEMORY` | **No** |
| 512k + DFlash | hung on ~32k cold prefill (GPU ~96 %, shm 60 s) | **No** — do not combine |
| `MAX_NUM_SEQS=16` / MNBT 8192 | switched TP=4 already found these do not help; not re-run as a win here | left at 8 / 2048 |

Cold long prefills are not accelerated by DFlash. Prefix cache (fixed system /
tool prefix) is the cheap latency win without a restart.

## What is not claimed

- No 1M needle, no 512k production profile, no 30–150 min soak on this ring.
- No `tests/bench_decode.py` median-of-5 table (harness model id mismatch).
- No claim that this path fixes #128 / #159 / #113 (switched-kit long-prefill
  stalls). Those remain open; this PR only makes the **ring transport** boot.
- README switched-kit numbers (CRS812, mixed 200G/100G, 1M) are a different
  fabric and stay in the README.
