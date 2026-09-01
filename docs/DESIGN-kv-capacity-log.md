# DESIGN — honest KV-capacity boot log (`patch_kv_capacity_log.py`)

Overlay: `overlay/patch_kv_capacity_log.py`. Test: `tests/test_kv_capacity_log.py`. Knob: `GLM53_KV_CAPACITY_LOG`.
`file:line` references are to the live fork inside `glm53-exl3-head` (vLLM `0.1.dev20051+g487ecf187`,
`/usr/local/lib/python3.12/dist-packages/vllm/`), read read-only on 2026-09-01. `U:` = `v1/core/kv_cache_utils.py`,
`I:` = `v1/kv_cache_interface.py`, `S:` = `v1/core/single_type_kv_cache_manager.py`, `C:` = `v1/core/kv_cache_coordinator.py`,
`B:` = `v1/core/block_pool.py`. Upstream feature request: [vllm-project/vllm#54662](https://github.com/vllm-project/vllm/issues/54662).

## 1. Problem

`update_kv_cache_capacity` (`U:2289-2303`, called once from `v1/engine/core.py:334` with the scheduler's
`KVCacheConfig`) logs:

```
GPU KV cache size: 1,553,140 tokens, Maximum concurrency for 1,000,000 tokens per request: 1.55x
```

`num_tokens = int(max_concurrency * max_model_len)` (`U:2276-2286`) with
`max_concurrency = num_blocks / num_blocks_per_request` and

```python
num_blocks_per_request = sum(                      # U:948-954
    cdiv(group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
         group.kv_cache_spec.page_size_bytes)
    for group in kv_cache_config.kv_cache_groups)
```

For one group that is the pool's token capacity up to rounding. For this model it is a *sum over seven groups*
drawing ids from **one** `BlockPool`, so the figure is "how many 1M-token requests fit" in token units. Neither
`num_blocks` nor `num_blocks_per_request` is logged; `max_concurrency` is printed at `%.2f`. From the line alone a
reader can pin the ratio to an interval and nothing more (the runbook recovered 643/414 by hand from the physical
bound). Two independent reviews read the line as a ~1.5M-token prefix cache before the arithmetic was redone; the
coexisting-conversation capacity was ~50–100K.

## 2. Deployment arithmetic (the numbers the test pins)

Group layout at boot (`C:1331` log line, plus the kpool tail group the coordinator lists separately):

| gid | spec (`I:`) | block | `max_memory_usage_bytes` | blocks / 1M-token request |
|---|---|---:|---|---:|
| 0 | `MLAAttentionSpec` (`I:295-300`) | 3584 | `cdiv(L, bs) * page` | 280 |
| 1 | `KpoolTailSpec` (`I:760-766`, opts out `I:779-786`) | 4 | `1 * page` | 1 |
| 2–5 | `MambaSpec`, `mamba_cache_mode=align` (`I:818-819`) | 3584 | `(2 + num_speculative_blocks) * page`, `num_speculative_blocks = num_speculative_tokens = 7` (`layers/mamba/abstract.py:74`) | 9 each = 36 |
| 6 | `SlidingWindowSpec` drafter, window 2048 (`I:616-647`) | 64 | `(cdiv(min(2047 + max_in_flight_tokens, L), 64) + 1) * page`, `max_in_flight_tokens = max_concurrent_batches(2, async) * 2048 = 4096` (`config/vllm.py:565-586`) | 97 |
| | | | **sum** | **414** |

`643 / 414 = 1.553140…` → `1,553,140 tokens`, `1.55x` — the logged line, reproduced. The current rightsized boot:
`820 / 414 = 1.98x`, `1,980,676 tokens` (boot log 2026-09-01 01:09:54).

`BlockPool.__init__` pops one block as the permanent null block (`B:188-191`), so **usable ids = num_blocks − 1**
= 642 (819). `num_gpu_blocks_override` sets the actual pool size, so the same subtraction applies there.

Ids one cached 3584-token segment costs, dense retention (what each manager's `reachable_block_mask` hashes with
no retention interval, hits ending on scheduler-block boundaries):

| gid | rule | ids |
|---|---|---:|
| 0 | `FullAttentionManager` keeps the base mask (`None` = every block, `S:482-500`) → `3584 / 3584` | 1 |
| 1 | opted out; `KpoolTailManager.cache_blocks` is a hard return | 0 |
| 2–5 | `MambaManager.reachable_block_mask`, `retention_interval is None` → dense (`S:1509-1511`) → one state per block position | 1 each = 4 |
| 6 | `need = _contiguous_blocks_for_hit(2048, 64, use_eagle=True) = cdiv(2047, 64) + 1 = 33` (`S:886-895`); `per_segment = 3584 / 64 = 56`; mask hashes `min(need, per_segment)` blocks per segment (`S:1032-1046`) | 33 |
| | **total** | **38** |

Capacity at this alignment = `642 // 38 * 3584 = 16 segments = 57,344 tokens` (819 → 21 segments = 75,264).
The design train measured the knee at 14 segments OK / 17 fail with a running request holding its own blocks
(`research-2026-08-31/DESIGN-drafter-retention.md` §4), consistent with a 16-segment idle bound.

## 3. What the overlay logs, and from where

Two sites in `kv_cache_utils.py`, both preflighted before either is written (`prepare()`), atomic replace,
idempotent, one `MARK` per site, `verified_state` exact post-state, drift ⇒ non-zero exit:

1. helpers inserted above `def get_max_concurrency_for_kv_cache_config(` (signature line is identical in-image and
   on upstream `22df3a3`; `patch_glm5_drafter_group.py` edits this file elsewhere and must run first);
2. one call appended after the stock `logger.info_once("GPU KV cache size: …")` block, which stays byte-identical
   (the test asserts the whole of `update_kv_cache_capacity` up to and including that block is unchanged and that
   `kv_cache_size_tokens` / `kv_cache_max_concurrency` are stored exactly as stock).

Per group: `index, spec type (UniformTypeKVCacheSpecs unwrapped), layers, block_size, page_size_bytes,
blocks/request@max_model_len, prefix_caching (fork participates_in_prefix_caching → upstream prefix_cacheable →
True), window + eagle for sliding-window kinds, mamba_cache_mode for mamba`. `blocks/request` is the same
`cdiv` expression as `U:948-954`; its column sum is the stock denominator.

Summary: `usable block ids: X (num_blocks=N incl. the null block; D ids per L-token request => R x); ids per
A-token cached segment across groups: Y (per group: […]); cached-conversation capacity at this alignment ≈ Z
tokens = S segments (aligned dense-retention prefix-cache upper bound: nothing running, every reachable block
hashed, block-aligned hits). The 'GPU KV cache size' line above is max_concurrency x max_model_len, not this
figure.` with `A = lcm(block sizes of all groups)` — what the coordinator asserts its scheduler block size to be
(`C:621-626`).

Per-spec policy (`_glm53_ids_per_segment`), by **exact** class name of the unwrapped spec — a subclass may come
with its own manager and reservation rule (`SinkFullAttentionSpec` keeps permanent sink blocks, `KpoolTailSpec` is
scratch, `TQFullAttentionSpec` / `RSWASpec` / `HiddenStateCacheSpec` carry other semantics), so nothing is costed
by its base class: opted-out → 0; `MambaSpec` in `align`/`all` → `A / bs`, the mode read from the spec the manager
acts on (`I:795`; a uniform group whose members disagree is "mixed"); `SlidingWindowSpec` / `SlidingWindowMLASpec`
→ `min(cdiv(window − 1, bs) + eagle, A / bs)`; `FullAttentionSpec` / `MLAAttentionSpec` → `A / bs`; everything
else (chunked-local, cross/encoder, sink, mamba mode `none`/mixed, a window-less SWA spec, unknown kinds) →
**unmodelled**: the summary names the group and withholds the capacity figure. The figure is also withheld
under `decode_context_parallel_size > 1` or `prefill_context_parallel_size > 1`, where the resolver rescales
block sizes per rank and the raw lcm is no longer the scheduler alignment (not this kit: both are 1, and the
SWA spec asserts DCP == 1).

EAGLE detection mirrors `C:637-651` with `patch_hybrid_prefix_hit.py` applied: `group.is_eagle_group` when any
group carries it; else, when `speculative_config.use_eagle()`, the groups whose unwrapped spec is an exact
`SlidingWindowSpec` (`_glm53_is_draft_swa_spec`); else every group (the upstream fallback).

## 4. Failure modes and the knob

`GLM53_KV_CAPACITY_LOG`: unset or `1` → log; `0` → one line `[glm53-kv-capacity-log] disabled
(GLM53_KV_CAPACITY_LOG=0)`; anything else → `ValueError` at the log site (outside the derivation's `try`: a typo'd
knob must not silently pick a mode). The launcher rejects the same set first (`_glm53_validate_bool_flag`,
inside the numeric guard, before `restart` stops anything), so the container only ever sees `0`/`1`.
Any exception inside the derivation → one `warning_once` naming it (`… could not derive the block-level KV capacity
(log-only; serving unaffected): <Type>: <msg>`), no info lines, boot proceeds. All log arguments are preformatted
strings (`info_once` caches on its arguments and needs them hashable).

## 5. Scope

Log-only: no control flow, allocation or config mutation; runs once per engine boot. Not modelled, on purpose:
- per-group retention (#83, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL[_SWA]`) lowers the drafter's cost to boundary
  tails only; fine-grained hits (#84) let a hit end on a 64-token boundary. Both are resolved later, in the
  coordinator, from env and manager capabilities; the summary states the policy it assumes (dense, block-aligned).
- a running request's own blocks (the figure is an idle upper bound; `414` ids per 1M-token request is the
  per-request peak, a different quantity).
- per-step accounting / gauges (the research train's Part B, do-not-build).
- KV connectors / CPU offload (none on this kit).
