# Changelog

## 2026-09-07 — E3 grouped fat-expert MoE prefill (`EXL3_FAT_GROUPED`, now the default)

Cold prefill **+37–45%** on this 2× GB10 kit (16k: 1,155 → 1,578 tok/s; 128k: ~1,150 → 1,629; 256k: 1,087 → 1,576),
decode unchanged. Commit `1a0feb0` (merge `dfd8e0f`); made the launcher default later the same day together with `MAX_MODEL_LEN` 1M → 900k, `GPU_MEM_UTIL` 0.87 → 0.85, `GLM53_INDEXER_WORKSPACE` stock → rightsize, and `EXL3_TEMP_ROWS_FUSED` 128 → 32 (E3) / 256 (E2), in `start.sh` and `.env.example`. Kernel design from the Fable prototype
(`.claude/worktrees/fable-perf`), qualified and measured in `logs/overnight-20260906T164059Z/`.

### What was slow before (E2, `EXL3_FAT_KERNEL=1`, cap 256)

Every prefill chunk (7,168 tokens × top-8 = ~57k token→expert routes per layer, 42 MoE layers) split experts
into *thin* (≤ cap rows, one fused `exl3_moe` launch for all of them) and *fat* (> cap rows). Fat experts went
through a **host-driven loop, one expert at a time**:

1. one D2H copy + sync of the routing counts to learn which experts were fat;
2. per fat expert: `index_select` the rows, input Hadamard, **copy the gate and up trellises into a stacked
   scratch (4 MB)** plus their output scales, launch the direct GEMM, then five separate elementwise kernels
   (two clamps, sigmoid, two multiplies), an fp16 copy, the down-input Hadamard, the down GEMM + scatter —
   about **14 launches and ~0.3–0.5 ms of host work per expert**.

With real routing most of the 288 experts in a layer are fat in a 7,168-token chunk, so a layer spent most
of its ~80–90 ms waiting for the CPU to feed the next expert; lowering the cap made it *worse* (cap 32:
113 ms), because more experts fell into that loop. MoE was roughly half of chunk time.

### What E3 does instead (`overlay/exl3_fat_moe.cu`, `overlay/exl3.py`)

- **Device-side segment tables.** From the sorted routing counts, ~20 small torch ops build, on the GPU,
  a row table (fat row → token, expert, route weight) and a segment table (64-row tile → expert, first row,
  rows). The kernels read the live `num_rows` / `num_segs`, so **no host synchronization** on routing and
  the layer stays CUDA-graph capturable.
- **Three launches cover every fat expert of the layer:**
  1. `gather`: fat rows → contiguous buffer, input scale + Hadamard applied;
  2. `gateup`: 64-row × 128-column tiles, 4-stage `cp.async` pipeline, trellis tiles **dequantized once per
     16 K per warp and reused across all M blocks**, gate and up streams in the same tile; the epilogue fuses
     both output Hadamards, the SwiGLU clamp/activation, and the down-input Hadamard, writing fp16;
  3. `down`: same mainloop over the intermediate, output Hadamard + route weight, **16-byte vector
     `atomicAdd(float4)` scatter** into the fp32 output, so tiles of different experts run concurrently.
- Net effect per layer: hundreds of launches and hundreds of 4 MB weight copies → 3 launches + table build.
  Isolated 7,168-token layer: **77–91 ms (E2 cap 256) → 31 ms (E3 cap 32)**, 2.1–3.0× across routing skews,
  at 46 TFLOPS vs 16–19. That is where the +38% end-to-end comes from (MoE ≈ half of chunk time).

### What changed relative to the prototype so it could ship

- **E2 rounding boundaries restored**: input scale multiplied in fp16 before the fp32 Hadamard; SiLU with
  precise `expf`/division (module built without `--use_fast_math`, unlike exllamav3); activation rounded to
  fp16 and multiplied by `down.suh` in fp16 before the fp32 down-input Hadamard. E3's error vs the LinearEXL3
  reference is now identical to E2's on every metric (incl. real checkpoint experts); the remaining E3/E2
  difference is the atomic accumulation order (max 0.125 on outputs ~4,600).
- **Load-time eligibility** (K4/MCG, no `mul1`, shared gate/up SUH, hidden % 256, intermediate % 128,
  sm_90+, single device) with a visible fallback to the E2 tier; grouped requested without the kernels
  **fails closed** at boot; diag schema 2; scratch growth refused during graph capture; the fused cap is
  never changed implicitly (set `EXL3_TEMP_ROWS_FUSED=32` explicitly, keep it ≥ `MAX_NUM_SEQS × (DFLASH_TOKENS+1)`).
- Tests: table builder vs host reference, parity vs loop and E2 under frozen tolerances, value regimes, real
  checkpoint experts, graph replay with changed data, scratch growth, invalid routes, fallbacks
  (`tests/test_exl3_overlay.py`); layer bench `tests/bench_e3_microbench.py`.
- Build: `Dockerfile.e3-layer` + `overlay/build_exl3_fat_moe_ext.py` compile only the new translation unit
  onto the existing image (tested); the full `Dockerfile` path now also installs the sources (not yet exercised).

### Known limitation

E3 keeps a persistent fat-row scratch (`h13` 448 MiB + `h2` 112 MiB for 57,344 rows) that is allocated during
vLLM's profile run and therefore charged to the KV budget (−0.56 GiB, −1…4% of the pool depending on util).
At 1M context that removes the single-request capacity on this kit, so the shipped defaults are
`MAX_MODEL_LEN=900000`, `GPU_MEM_UTIL=0.85`, `GLM53_INDEXER_WORKSPACE=rightsize` (measured recipe: 500k / 0.84;
900k / 0.87 served a 256k prefill with driver retries). Fix path: fuse the gather
into the gate/up A-tile load (drops `h13`) or size scratch from actual fat rows. Prompts ≥ ~100k tokens
remain close to the head's host-memory limit at any util; a 256k prefill at util 0.87 with zero MemAvailable
crashed the head on 2026-09-06.

Also in the same change: `MAX_MODEL_LEN` caller override in `start.sh`, effective-EXL3-knobs boot line,
`.env.example` docs. Not included: the DFlash2 vocab-parallel top-k experiment (inconclusive, worktree only).
