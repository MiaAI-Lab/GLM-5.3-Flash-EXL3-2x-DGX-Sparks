# SENS8 opt-in decode routing (TP2 / SM121)

`GLM53_SENS8_ROUTER=0|1` (default `0`). Flag off leaves the stock router
path completely untouched (the integration is not installed). Flag on
installs the qualified graph-safe general router.

## What

On speculative verify steps, each request block (≤16 rows, ≤4 blocks) is
restricted to a request-local 28-expert coreset instead of all 288
experts. Coreset = top-28 by rank-weighted utility
(`sigmoid(logit) / (1 + rank)`, index-ascending ties) over the block's own
rows; per-row top-8 + `s/Σs × 2.5` weights use the exact stock pick
sequence and tie-break. No sensitive-layer mask. Prefill, no-spec steps,
and unsupported shapes fall back to stock selection inside the same kernel.

## Why

Decode MoE is expert-weight-bandwidth bound: typical M=8 verify touches
~38 unique experts/layer; the coreset cuts that to ~27 (−28%), saving
~21% routed-MoE time per layer for ~+30µs/layer router cost.

## Graph safety

One fixed launch topology (grid from M, never from the partition);
request boundaries ride as dynamic `int32[6]` device metadata
`{mode, nb, len0..len3}`, fully overwritten every prepare with
read-before-overwrite ordering. No partition-specific graph keys; the
vLLM graph subsystem is not forked. Warmup capture cannot bake stock
behavior (metadata is read per replay).

## Scope and limits

- TP=2, 2× GB10 / SM121, `MAX_NUM_SEQS` ≤ 4, E=288, top-8. TP3 unsupported
  (fail-closed to stock outside the qualified geometry).
- Approximate routing, NOT numerical equivalence: prospective
  stock-calibrated studies bound the deviation (boundary-rank swaps,
  top-1 routing untouched in aggregate) but do not prove zero impact.
- C=28 is a frozen implementation constant, not a knob.

## Files

- `kernel_lab/sens8/sens8g_router.cu` — kernel (E=288/K=8/C=28/×2.5).
- `kernel_lab/sens8/sens8g_loader.py` — SM12x-gated `load_inline` build.
- `overlay/exl3.py` — install-on-opt-in integration (prepare/metadata).
- `tests/test_sens8_router.py` — reference, parity, replay, gating tests.

## Provenance

Routing math informed by `rodman80/glm-5.3-flash-w4a16-2x-DGX-Sparks`
`perf/sens8-integration` (read-only donor). Graph-safe integration,
SM121 validation, and prospective quality qualification were done in
this repository against the current stack.
