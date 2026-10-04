# Acceptance criteria — GLM cooperative MoE decode work

Recorded 2026-09-16 before inspecting later candidate kernels. These gates
apply to every subsequent experiment. They do not redefine success around a
narrower subset of the original objective.

## Frozen serving capabilities

Must remain enabled and at least as capable as the goal-start serve:

| Knob | Required value |
|---|---|
| Checkpoint / quant | GLM-5.3-Flash EXL3 K4 MCG |
| Topology | TP2, two Sparks, not EP |
| Prefill | E3 grouped (`EXL3_FAT_GROUPED=1`), fused cap 32 |
| Speculative decode | DFlash2 k=7, adaptive-k `ema` set `2,4,7` |
| Dense path | `GLM53_DENSE_FP8=dense,kda` |
| CUDA graphs | on; capture sizes `1 2 3 4 5 6 8 9 10 12 15 16 20 24 32` |
| Context | `MAX_MODEL_LEN=850000` |
| KV | `--kv-cache-memory-bytes 15032385536` (~883,552 tokens, ~1.04×) |
| Concurrency | `MAX_NUM_SEQS=4` |
| Mixed prefill | `fair` |

Speedups obtained by shrinking any of the above are invalid.

## Numerical (GPU maintenance gate)

Reference is stock fused EXL3 on the same tensors. Cooperative is not
bit-exact. The historical global peak screen is retained and is **not**
sufficient by itself.

| Check | Pass |
|---|---|
| All outputs finite | required |
| Eager vs stock, unit-scale inputs (`x ~ 0.1`): `max(abs(err)) / max(abs(ref))` | `<= 0.003` |
| Relative L2 `‖err‖₂ / ‖ref‖₂` | `<= 0.05` |
| Per-row `max(abs(err_row)) / max(abs(ref_row))` | `<= 0.05` |
| High-activation mutation (`x ~ 0.6`) peak vs stock | recorded; fail if `> 0.01` |
| Invalid routes and zero weights | exact zeros |
| Rows 1–32 (including live 3 and 5) | cooperative selected |
| Rows 33 and 40 | stock; concentrated 33/40 must execute E3 grouped |
| 288-expert table | same selection rules |
| Graph replay | matches eager cooperative on the same tensors |
| `limit=0` and clipped `limit>0` | within the screens above |

The historical 0.003 peak screen is the primary eager unit-scale gate. Geometry-1
live capture size 10 at input scale 0.6 measured ~0.0036 peak / 0.0017 rel L2 on
both nodes; that is recorded, not used to silently drop the 0.003 unit-scale
rule. Real GLM weights/activations and assistant-output quality remain
additional required evidence.

## Quality (live assistant output)

Temperature 0, thinking off, matched prompts. Score generated assistant
text, not user-turn logprobs.

| Probe | Pass |
|---|---|
| Structured count | requested integers present in order, no NaNs |
| Prose (hash-map) | on-topic explanation, no NaNs, no empty content |
| Coding | syntactically plausible code, no NaNs |
| Coherence (`tests/bench_decode.py`) | Paris / 9.9>9.11 / sky probes as in stock |

Text need not be bitwise identical to stock. A candidate fails if it
introduces NaNs, empty assistant content, broken structured counting, or
clearly garbled coding/prose relative to the stock and geometry-1 baselines
on the same prompts.

## Performance claims

A claimed decode gain must:

1. Use matched prompts, output lengths, sampling, warmup, and cache isolation.
2. Report throughput, TTFT, elapsed decode ms per recorded draft step,
   acceptance, and adaptive-k verification-size distribution separately.
3. Include prose, structured, and coding at concurrency 1 and 2 (aggregate
   throughput at ×2).
4. Use `>= 5` measured requests per workload per arm after warmup.
5. Repeat baseline/candidate/baseline serving restarts; confirm promising
   comparisons on an independent boot.
6. Exceed measured run-to-run variation on the same protocol.
7. Leave E3 prefill and KV/memory capacity within the frozen capabilities.

Per-step ratios are serving-cycle measurements, not kernel timings. Kernel
attribution requires a fixed-input CUDA-event microbenchmark with separately
captured stock and candidate graphs.

## Rollback

- Stock overlay: `logs/astra-goal-20260916/rollback-stock.sh`
- Initial geometry-1 cooperative: `logs/astra-goal-20260916/rollback-geo1.sh`
- Earlier opt-in copy: `logs/cooperative-moe.eFKgLt/original.env`

Provenance at goal start: `logs/astra-goal-20260916/provenance.json`.
