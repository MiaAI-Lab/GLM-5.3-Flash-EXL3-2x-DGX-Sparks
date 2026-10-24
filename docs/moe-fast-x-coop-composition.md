# `GLM53_EXL3_MOE_FAST` x cooperative MoE: the composition IS load-bearing (2026-09-28)

`docs/sm121-perf-paths.md` currently says: *"Composition with the cooperative-MoE path is
untested."* Measured it on a second 2×Spark kit (`spark-d931`/`spark-ea54`, recipe `943912c`,
image built from this repo), same machine, same boot configuration except the flag, empty
queue (`num_requests_running + waiting == 0`), warmup banner reached first.

## Result

| workload | `MOE_FAST=1` | `MOE_FAST=0` | delta |
|---|---:|---:|---:|
| structured (count 1→200) ×1, 3 runs | **80.20 tok/s** | **44.88 tok/s** | **−44 %** |
| prose ×8 aggregate, 3 runs | **110.64 tok/s** | 98.09 tok/s | −11 % |

Both arms: cooperative overlay selected (`EXL3_OVERLAY_HOST`, geometry 1), `EXL3_FUSED_MOE=1`,
`EXL3_FAT_GROUPED=1`, `EXL3_FAT_KERNEL=1`, `EXL3_TEMP_ROWS_FUSED=64`, DFlash2 k=7,
adaptive-k `ema`, `MAX_NUM_SEQS=8`, KV pool 14 GiB.

## Why this is not obvious from the code

The cooperative adapter replaces `apply_exl3_fused_moe` for **1–32-row** calls and leaves
everything else to the original path. The thin-decode kernels enabled by `MOE_FAST` live inside
that original path. So one could reasonably expect `MOE_FAST=0` to cost little when coop is
active and decode batches are small — the measurement says otherwise, by a wide margin:

* `MOE_FAST=1` also gates the native pointer tables built in
  `build_exl3_fused_state()` (the fail-closed check that raises when
  `exl3_ext.glm53_fast_moe_version` is missing). Turning it off changes which kernel the fused
  path runs for calls the adapter does **not** take, and at ×8 concurrency a large share of
  decode steps exceed the 32-row coop window (6 seqs x 8 rows = 48, 8 seqs x 8 rows = 64), so
  those steps go down that path on every layer.

## Practical guidance

* Keep `GLM53_EXL3_MOE_FAST=1` together with the cooperative overlay on this kit; the two are
  complementary, not alternatives.
* If a kit must run with `MOE_FAST=0` (e.g. an image without
  `overlay/patch_exl3_decode_pipeline.py`), expect roughly 40 % lower single-stream decode than
  the tables in `docs/cooperative-moe.md` suggest, and do not attribute that to coop.
* Worth a line in `docs/sm121-perf-paths.md` replacing "untested": the composition is tested and
  `MOE_FAST=1` is required for the documented coop numbers.

Raw receipts: `logs/mf_A_*.json` (`MOE_FAST=1`) and `logs/mf_B_*.json` (`MOE_FAST=0`) on the
reporting kit; each file is the bench's own JSON with `tok_s_median`, `ttft_median_s`,
`accept_ratio_median`.
