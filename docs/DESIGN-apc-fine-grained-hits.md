# Opt-in fine-grained prefix-cache hits

`GLM53_FINEGRAINED_APC` controls the lookup gate installed by
[`patch_apc_fine_grained_hits.py`](../overlay/patch_apc_fine_grained_hits.py).
It defaults to **`0`** in `.env.example`, the TP=2 launcher, and the runtime.
Set it to `1` and restart to opt in; set it to `0` and restart to disable.
Only exact `0` and `1` are accepted, including when overriding `.env` from the
caller environment. An explicit empty value is rejected before restart.

The feature remains opt-in because the current measurements do not resolve
prior draft-acceptance concerns or qualify default enablement.

## Lookup and scratch-state contract

The pinned hybrid coordinator includes `KpoolTailManager` in its
`supports_fine_grained_hash_lookup` veto, although that scratch group declares
`participates_in_prefix_caching=False` and never answers cache lookups. On this
kit, that veto reduces lookup granularity from 64 to 3,584 tokens.

The overlay scopes the veto to participating groups. MLA and Mamba remain in
the hybrid hit minimum; a participating manager that requires coarser lookups
still disables fine hits. The overlay does not change hash indexing or assign
a different `_cache_hit_alignment_tokens`.

Non-participating scratch groups still constrain where prefill can resume.
The kpool tail contains an unfinished pool's raw state, which is not restored
by a prefix hit. A hit at `H` must therefore satisfy `H % index_kpool == 0`.
The runtime checks that `hash_block_size` is divisible by every scratch
alignment. The observed layout has `hash_block_size=64` and `index_kpool=4`;
these values are checked from manager/spec attributes, not hard-coded.
Available alignment attributes must be exact integers and agree with each
other; a missing, contradictory, or incompatible requirement is rejected when
fine hits would otherwise be enabled.

| Participating manager blocks fine lookup | Scratch alignment | Result with `=1` |
|---|---|---|
| No | Verified and compatible | Enable fine hits |
| No | Unsafe or unverifiable | Refuse initialization |
| Yes | Either | Keep the coarse fallback |

`=0` disables fine hits without requiring scratch qualification. The
coordinator logs the effective alignment, reason, and checked scratch groups.
This receipt belongs to the scheduler-owning process; matching code and
configuration on a headless worker do not imply a second scheduler receipt.

## Integration and source validation

The launcher validates the flag and overlay artifact before stopping either
rank, forwards the same flag to both containers, and mounts the same overlay
source on both. The Dockerfile includes and applies the overlay.

Patch application is atomic and idempotent. An already-patched file must
contain the canonical helper/gate regions in the expected AST scopes, without
duplicate or rebound owned names. Unsupported source or anchor drift fails
without modifying the target. There is no speculative upstream-fixed no-op
or connector qualification detector.

The overlay composes with `patch_hybrid_prefix_hit.py` and
`patch_apc_per_group_retention.py`. Their retention and DFlash replay checks
remain in force. In particular, a proposed fine boundary can be backed up by
the 2,048-token replay requirement and reconciled to an earlier checkpoint.
An enabled 64-token alignment or a logged clamp target is not an actual hit
receipt. Retention tradeoffs are documented [separately](apc-retention-qualification.md).

## Verification and limits

[`test_apc_fine_grained_hits.py`](../tests/test_apc_fine_grained_hits.py) covers
source drift, transactional refusal, idempotence, the runtime gate and strict
flag handling, and all six fine/hybrid/retention application orders. Supply
coordinator and companion source copies from the target image to run the
complete suite without importing vLLM:

```bash
GLM53_KV_COORDINATOR_PY_SRC=/path/to/kv_cache_coordinator.py \
GLM53_KV_COORDINATOR_PY_PRISTINE=/path/to/pristine/kv_cache_coordinator.py \
GLM53_BLOCK_POOL_PY_SRC=/path/to/block_pool.py \
GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY_SRC=/path/to/single_type_kv_cache_manager.py \
python3 tests/test_apc_fine_grained_hits.py
python3 tests/test_start_overrides.py
python3 tests/test_launcher_rank_parity.py
```

The launcher suites exercise caller overrides (including empties), actual
rank command construction, and rejection before stop. CPU checks do not
establish live cache reuse or internal numerical equivalence.

The [2026-09-15 verification record](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/171#issuecomment-5669908368)
contains the fixed configuration, BAAB protocol, metrics, and deviations.
The core 60-request comparison found no additional cache reuse. A separate
six-request AAB diagnostic demonstrated 4,736 additional reused tokens in
one enabled long continuation, with TTFT 2.77 s versus 6.10 s in two controls.
All 66 output checks passed; the acceptance differences were not repeatable.
This is a conditional benefit, not a default-on qualification.

The [#84 A/B evidence](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/84#issuecomment-5507694711)
and [acceptance assessment](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/84#issuecomment-5603758872)
remain relevant. Internal KV/KDA/CoW equivalence, eviction pressure, KV-transfer
connectors, long agentic-loop acceptance, and default enablement remain
unqualified by these observations.
