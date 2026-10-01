# Serving-tuning measurements on a second 2×Spark kit (2026-09-28)

All numbers below come from one kit (`spark-d931` head / `spark-ea54` worker, 2×GB10 SM121a,
CX7 link, recipe `943912c`, image built from this repo) and were taken with the engine warm and
**idle** (`vllm:num_requests_running + num_requests_waiting == 0`) after the boot-shape warmup
banner. Method notes at the end — two earlier "regressions" turned out to be measurement
artifacts, which is worth knowing before trusting any number from a busy box.

## 1. Concurrency sweep (prose, hash-map, 400 tokens, temp 0, thinking off)

| concurrency | stream tok/s | aggregate tok/s | TTFT |
|---:|---:|---:|---:|
| 1 | 37.7 | 37.7 | 0.26 s |
| 2 | 29.8 | 57.2 | 0.40 s |
| 4 | 21.4 | 79.0 | 0.36 s |
| 6 | 19.2 | 108.5 | 0.41 s |
| 8 | 16.3 | **124.5** | 0.42 s |

Config: `MAX_NUM_SEQS=8`, `EXL3_TEMP_ROWS_FUSED=64`, DFlash2 k=7, adaptive-k `ema`,
dense-FP8 `shared,dense,kda,mla`, coop overlay geometry 1, `MOE_FAST=1`, KDA large-M on,
KV pool 14 GiB, `MAX_MODEL_LEN=1000000`. Structured (count 1→200) ×1: **83.5 tok/s**.

For reference, the same kit with `MAX_NUM_SEQS=6` and dense-FP8 restricted to `dense,kda`
reached 92.8 aggregate at ×6 and could not serve ×8 without queueing — i.e. the ceiling, not the
per-step cost, was the limit for real multi-session traffic.

## 2. Two knobs that interact (`MAX_NUM_SEQS` and `EXL3_TEMP_ROWS_FUSED`)

README already states the rule ("keep ≥ `MAX_NUM_SEQS × (DFLASH_TOKENS+1)` so decode stays one
graph-safe launch"). Raising `MAX_NUM_SEQS` to 8 therefore implies 64, and the launcher's
auto-generated cudagraph list grows accordingly (24 entries, including 56 and 64). Measured at
×8 aggregate: 32 → 111.1, 64 → 111.1 (equal within noise); the value matters for correctness of
the intent rather than for the number.

## 3. Dense FP8 groups (`GLM53_DENSE_FP8`)

| groups | structured ×1 | prose ×1 | prose ×8 aggregate |
|---|---:|---:|---:|
| `dense,kda` | 80.20 | ~37-39 | 110.6 |
| `shared,dense,kda,mla` | **83.79** (+4.5 %) | 39.35 | **127.8** (+15.5 %) |

**Caveat, same as the README's**: this path is PROVISIONAL numerically ("KL proxy vs stock
0.002–0.013 nats/position; no full KLD panel yet"). The repo's own `bench_decode.py` coherence
check is only three sanity prompts (capital of France, 9.9 > 9.11, one sentence about sky blue)
plus NaN detection — a smoke test, **not** a quality gate. Treat the table as a speed datum and
run a scored A/B before adopting it on a production kit.

## 4. Adaptive verification length

| `GLM53_ADAPTIVE_K` | structured ×1 | prose ×1 |
|---|---:|---:|
| off (fixed k=7) | **82.0** | 33.2 |
| `ema` | 76.1 | **37.0** |

Consistent with the README's "prose +13–21 %": the trade is prose-positive, structured-negative
on this kit, so the right choice depends on the dominant load.

## 5. Cudagraph capture list: more entries are not better

Trimming 20 → 15 entries (dropping 9, 15, 18, 25, 30; the mandatory 3/8/16/24/32 kept) measured
**equal or better** than the 20-entry list at ×1/×2/×4 (e.g. ×2 aggregate 59.8 vs the README's
51.1 with 20 entries), while cutting graph memory. Adding entries therefore has no measurable
upside at this concurrency; the launcher's derived list is already in the flat part of the
curve. Note the padding is bounded by the gaps in the list, and the large gaps (49…7168) are
prefill batches, which are not graph-captured at all.

## 6. SM clock cap: a boot-time `-lgc` silently costs ~6 %

This kit shipped a one-shot unit that pinned the head to `nvidia-smi -lgc 0,2200` for "thermal
and power headroom" while the worker ran the full range. Because it is a boot unit, a manual
`-rgc` does not survive a reboot, and TP2 throughput follows the slower rank:

| | capped (2190 MHz) | unlocked (2480–2535 MHz) |
|---|---:|---:|
| structured ×1 | 76.1 | **80.3** |
| prose ×1 | 37.0 | **39.2** |

Temperatures at 96 % utilisation: 71 °C / 30 W capped vs 85 °C / 68 W unlocked (worker 84 °C),
i.e. the cap buys ~14 °C on the head. Whatever policy a kit wants, put it in the unit file and
keep both ranks consistent, or a per-rank mismatch becomes the cluster's ceiling.

## 7. Memory: why the head always looks ~2 GB tighter than the worker

cgroup accounting for the two engine containers (same image, same engine, TP2):

| MB | head | worker | delta |
|---|---:|---:|---:|
| container total | 12 063 | 11 068 | +995 |
| anon (processes) | **6 519** | 4 681 | **+1 838** |
| file (page cache, in container) | 5 194 | 6 053 | −859 |
| `Cached` / non-reclaimable `Shmem` | 6 119 / 3 762 | 7 824 / 3 716 | |
| `MemAvailable` | 2 994 | 4 342 | −1 348 |

The head runs `APIServer` (~2.0 GB) plus `EngineCore` (~1.2 GB) that the headless worker does not.
With a fixed 124.6 GiB UMA pool that shows up as ~1.8 GB less room for everything else, and
since `MemAvailable` counts reclaimable cache, the head's smaller cache widens the gap further.
Page cache is the *symptom*; the extra API-side processes are the cause.

Related field note (same kit, before the work in this report): with the head at `MemFree ≈ 1.6
GiB` and a page cache inflated by a 21 GB image build, a serving boot produced
`NVRM: NV_ERR_NO_MEMORY (0x00000051)` … `_memdescAllocInternal`, four
`systemd-journald: Under memory pressure` lines, and then a **hard reset with no shutdown
sequence**. Since then: drop clean page cache immediately before launch (the repo's
`GLM53_HOST_MEM_HYGIENE`, which on this kit needs a NOPASSWD helper because the launcher's own
`sudo -n sh -s` would require granting a generic shell), plus a 2-minute guard that drops the
cache again if `MemFree < 1 GiB`. Every boot since has 2–8 *transient* NVRM OOM lines that
recover, and `Xid`/pressure stay at 0.

Memory cost of the features used here, for planning: KDA large-M BF16 copies
`+98.2 MiB/rank x 34 layers = 3.26 GiB`, coop graph capture 1.8–2.9 GiB, KV pool as configured
(14 GiB = 1 621 700 tokens at `MAX_MODEL_LEN=1000000`, usable cached-conversation capacity
247 296 tokens).

## 8. Method notes (why some earlier numbers were wrong)

* **Warmup contamination**: the post-`/health` shape sweep keeps the engine busy for minutes; a
  benchmark started right after `/health` measured 44.9 instead of 83.5 structured. Gate on the
  start script's completion banner, not on `/health`.
* **Concurrent clients**: a second client polling a non-existent model (`404` flood) and four
  live sessions were enough to halve single-stream numbers. Gate on
  `num_requests_running + num_requests_waiting == 0` before and after each run.
* **Cross-kit comparisons are not a substitute for same-machine A/B**: an initial "dual rail
  does not help" conclusion came from comparing against the numbers in this repo, which were
  produced on a different kit under different flags. Re-testing on one machine in both
  directions gave +5 % prefill / neutral decode (see the dual-rail note).
* Receipts are the bench's own JSON (`tests/bench_decode.py --out …`) plus engine logs; the
  raw `.env` snapshots for every arm are kept alongside them.
