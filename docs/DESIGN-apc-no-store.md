# DESIGN — per-request GPU prefix-cache no-store (`skip_writing_prefix_cache`)

Overlay: `overlay/patch_apc_no_store.py`. Test: `tests/test_apc_no_store.py`. Knob: `GLM53_APC_NO_STORE`.
`file:line` references are to the live fork inside `glm53-exl3-head` (vLLM `0.1.dev20051+g487ecf187`,
`/usr/local/lib/python3.12/dist-packages/vllm/`), read read-only. **VERIFIED** = read in that source or a live
log line; **INFERRED** = reasoned from verified code. Host tests ran against upstream vLLM `22df3a3`, whose
`block_pool.py` carries the same anchors at the same lines.

## 1. Problem

`SamplingParams.skip_reading_prefix_cache` (`sampling_params.py:364`) is a **read**-side skip only: resolved at
`v1/request.py:213`, consumed at `v1/core/kv_cache_manager.py:217`, auto-set for `prompt_logprobs`. There is no
write-side counterpart (VERIFIED: 17 hits for `skip_reading_prefix_cache` in the fork, none on a store path;
`skip_writing_prefix_cache` does not exist). `cache_salt` is a *namespace* (`kv_cache_utils.py:560-568`), the
block is stored as usual. `delay_cache_blocks` (`kv_cache_manager.py:353`, `:552`) is a one-step P/D defer.

Why the write matters (VERIFIED mechanism, `block_pool.py:719-743`):

```python
if block.block_hash is None or not self.enable_caching:
    blocks_to_evict_first.append(block)     # LIFO, prepended
else:
    blocks_to_evict_last.append(block)      # FIFO / LRU, appended
```

`get_new_blocks` pops from the **front** (`:661`). A batch request's blocks are hashed, land at the back, and
the owner's idle 80K conversation — freed earlier, so nearer the front — is what gets evicted next. The batch
job *re-orders the eviction queue in its own favour*. With a no-store flag its blocks carry no hash, go to the
front, and are the very next ids recycled. On a hybrid model one logical segment costs one block id per group
out of one shared pool (MLA + 4 mamba + drafter SWA here), which amplifies the effect.

## 2. Design

### 2.1 Surface

- `SamplingParams.skip_writing_prefix_cache: bool | None = None` (sibling of the read-side field; typed route
  for a future upstream field).
- `vllm_xargs: {"skip_writing_prefix_cache": 1}` on `/v1/chat/completions`, `/v1/completions`, `/v1/responses`.
  Both already copy `vllm_xargs` into `SamplingParams.extra_args` (`chat_completion/protocol.py:487`/`:700`,
  `completion/protocol.py:219`/`:357`, `responses/protocol.py`), so the overlay needs **zero entrypoint
  edits** — which matters because this fork's `entrypoints/openai/` tree is restructured and an overlay
  anchored there would be brittle. `skip_reading_prefix_cache` itself is *not* reachable from the OpenAI API
  (not a `from_optional` parameter) and `cache_salt` travels the prompt/inputs path, so "expose it like
  `cache_salt`" cannot be taken literally.
- **Strict values**: bool, int `0`/`1`, str `"0"`/`"1"`. Everything else (floats incl. `1.0`, `"true"`,
  `"yes"`, `2`, lists) is rejected. `bool("0")` is `True`, so a lenient parser would let string-valued clients
  silently *enable* the flag (Codex finding). JSON `true`/`false` in `vllm_xargs` (typed
  `dict[str, str | int | float | list[...]]`) are **coerced by pydantic v2 to `1`/`0`** before vLLM sees them
  (VERIFIED on pydantic 2.13; test B1) — the intended meaning, so booleans are accepted; an earlier draft
  claimed they were rejected, which Codex's final review corrected.
- **Where rejection happens and why**: `Request` is materialised in the engine-core input thread
  (`EngineCore.preprocess_add_request` → `Request.from_engine_core_request`), not in the API server, so a
  `ValueError` raised from `Request.__init__` is **not** a request-scoped 400. Validation therefore runs in
  `SamplingParams.__post_init__` (anchor: the read-side auto-set at `sampling_params.py:529-533`), which
  executes in the API-server process for every OpenAI request (`to_sampling_params` → `from_optional` →
  constructor) and in the caller's process for the offline `LLM` API. A `ValueError` there is mapped to
  `BadRequestError` / HTTP 400 by the server's registered `ValueError` handler
  (`entrypoints/serve/exception_handling/register.py`, `error_response.py`). The engine-side resolver on
  `Request` re-parses strictly but **never raises**: an unparseable value there is unreachable through the
  API (it was already rejected); if it happens anyway (a params object mutated after construction) it logs a
  warning and stores normally.
- **Kill switch** `GLM53_APC_NO_STORE` (launcher knob, both ranks): exactly `0` or `1`, unset = `1`. Rule:
  malformed values are rejected *before* the switch is consulted; the switch only decides whether a valid `1`
  is honoured (`0` → ignored, logged once). Any other value raises at import of `sampling_params.py` (boot
  failure); the launcher's `_glm53_validate_bool_flag` refuses it before `restart` stops the pair. Default
  `1` is behaviour-neutral: requests never opt in on their own.
- **Receipts** (`logger.info_once`, no request id in the args so they really are one line per process):
  `[glm53-apc-no-store] first request resolved skip_writing_prefix_cache=1` at resolution — emitted even for
  a warm request that has nothing new to store — and `[glm53-apc-no-store] suppressing prefix-cache store
  (full site)` / `(partial site)` when a store is actually cut. Per-request ids go to `debug`. The partial
  site is reached only where the fork's fine-grained partial-hit producer is enabled (#84,
  `patch_apc_fine_grained_hits.py`; upstream's coordinator vetoes it for this model otherwise) and, for the
  mamba path, only when the prompt length is a `hash_block_size` (64) multiple that is not a 3584 multiple
  (`_cache_partial_tail_block:1866-1874`); the full-attention path fires for any prompt whose 64-token
  boundary is not a 3584 multiple.

### 2.2 Where the guard goes — and why *not* `kv_cache_manager.py:552`

The obvious one-liner extends `if not self.enable_caching or delay_cache_blocks:` to skip
`coordinator.cache_blocks`. **This is wrong and the bug is invisible locally.** Skipping the call skips
`SingleTypeKVCacheManager.cache_blocks`, whose last statement is
`self.num_cached_block[request.request_id] = num_full_blocks` (`single_type_kv_cache_manager.py:479`).
Membership in that dict is the **"is this a running request?" sentinel**:

- `get_num_blocks_to_allocate:196-202` — `if request_id in self.num_cached_block:` selects the fast path
  (asserts `len(new_computed_blocks) == 0`, returns `max(num_required - num_req_blocks, 0)`);
- `KVCacheCoordinator.allocate_new_computed_blocks:239-246` — skips `add_local_computed_blocks`, which
  asserts `len(req_blocks) == 0`.

A no-store request under chunked prefill would take the slow path on every later step, where
`get_num_skipped_tokens` is `0` for `FullAttentionManager` but the out-of-window prefix for
`SlidingWindowManager` — the drafter group's block accounting computed under the false premise "never
allocated". `KpoolTailManager` (`:1097-1199`), the in-tree never-stores manager, has to override
`get_num_blocks_to_allocate`, `get_num_skipped_tokens`, `remove_skipped_blocks` and
`add_local_computed_blocks` precisely so that none consult `num_cached_block`; a per-request flag cannot buy
those overrides.

**Adopted: suppress the hash insertion, keep every piece of bookkeeping.** There are exactly two request-driven
`_insert_block_hash` sites (VERIFIED: `grep -n _insert_block_hash block_pool.py` → `:293, :508, :607(def), :645`):

| site | inserts | caller |
|---|---|---|
| `cache_full_blocks` → `:293` | full-block hashes | `SingleTypeKVCacheManager.cache_blocks:469-477` |
| `cache_partial_block` → `:508` | fine-grained partial-tail entry | `FullAttentionManager._cache_partial_tail_block:815`, `MambaManager._cache_partial_tail_block:1858` |

The guards return early at the top of each. Downstream self-corrects because the fork's **sparse retention**
(`reachable_block_mask` → `block_mask`, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` live on this deployment)
already taught every consumer to tolerate "an allocated full block with no hash":

- `num_cached_block` still advances at `:479` → both sentinels correct, every manager, every step
  (test C2/C3L: the per-group `num_cached_block` traces of a no-store request and a normal twin are identical).
- `MambaManager._cache_partial_tail_block:1858` gets `None` → the request is never registered as a
  partial-tail **producer** (`_partial_hit_reqs`, `_producer_partial_tail_reqs`; registration at `:1865` is
  gated on `partial_hash is not None`), so `get_num_blocks_to_allocate`'s `has_partial_hit` term and
  `allocate_new_blocks`' CoW branch stop *over*-reserving — they under-reserve nothing (test C3a).
- `MambaManager.cache_blocks:1817-1827` already `continue`s on `block.block_hash is None` (`:1825`) →
  `cached_blocks_this_step` stays empty for a no-store request (test C3L).
- `free_blocks:733` sees `block_hash is None` → **front** of the free queue (tests C1/C2/C3b/C3L).
- Returning early also skips the `BlockStored` event emission (`:301-341`): nothing was stored.

**`move_block_hashes` (`:629-645`) is deliberately not guarded.** It has no request argument, it re-points
*existing* hashes, and normal requests need it (a running partial-tail owner keeps writing into its block).
Codex asked whether the Mamba `blocks_allocated` CoW branch (`:1751`) could land a hash on a no-store
request's CoW block. It cannot: that branch requires `_partial_hit_reqs[request_id]` for a *running* request,
and the only two writers of `_partial_hit_reqs` are (a) `add_local_computed_blocks:292` for a **new** request
(consumed by the same allocation's `allocate_new_blocks` with `blocks_allocated=False` → `_apply_cow`, and
unreachable later because `allocate_new_computed_blocks` skips requests already in `num_cached_block`), and
(b) the producer registration at `:1892`, which the partial guard suppresses. `pop_blocks_for_free:1794`
discards `_allocated_block_reqs` and `_partial_hit_reqs` on free, so a preempted request resumes as (a).
Tests C3a (running no-store producer: no registration, no CoW copy sourced from its blocks after two decode
steps) and C3b (no-store reader: source keeps its hash, private CoW block unhashed) are the executable form of
this argument.

### 2.3 Deliberately unchanged

- **Lookups stay enabled.** A no-store lane sharing the system prompt gets the free prefix and pays nothing
  back. Reading *touches* blocks (`add_local_computed_blocks:281` → `block_pool.touch`), i.e. refreshes their
  LRU position — write suppression alone does not make a request invisible to residency. Combine with
  `skip_reading_prefix_cache` for zero interaction (test C5); that field is not exposed on the OpenAI API by
  upstream and this overlay does not add it.
- **Reader-side partial-hit CoW is untouched** (test C3b): correctness there is about not writing into another
  request's block; it has nothing to do with storing.
- **KV connectors / CPU offload** are not touched: no `--kv-transfer-config` on this deployment (VERIFIED via
  `docker inspect`). The flag is documented as *GPU prefix-cache* no-store; a connector-side suppression would
  be a one-line addition in the offloading connector's offload decision, symmetric to its existing
  `skip_reading_prefix_cache` check, and is an upstream RFC item, not an untestable overlay edit.
- **PoolingParams** are out of scope (chat/completions only); the resolver reads `SamplingParams` only.

## 3. Risks

- **R1 preemption.** `Scheduler._preempt_request` frees all blocks and sets `num_computed_tokens = 0`. A no-store
  request resumes from whatever it can *read*: nothing if its prefix was cold (full recompute, the
  `--no-enable-prefix-caching` resume path, structurally supported), or the cached prefix if it read-hit one
  (test C4 covers both). Residual risk is thrash, not corruption: mark short lanes, run them at low `priority`
  (`--scheduling-policy priority` picks the preemption victim by `(priority, arrival_time)`), watch
  `num_preemptions` in the live test.
- **R2 metrics.** A no-store request still reads, so it is counted in `vllm:prefix_cache_queries/hits`. Measure the
  owner conversation's own `usage.prompt_tokens_details.cached_tokens` (per response, straight from
  `prefill_stats.num_cached_tokens`), never the aggregates, and not `kv_cache_usage_perc` (live-referenced
  blocks only).
- **R3 silent no-op.** Mis-typed key, JSON `true`, kill switch at `0`: the system behaves exactly as today. Hence the
  resolution receipt line; do not trust an A/B without it in `docker logs`.
- **R4 structured output.** No interaction: grammar state lives on `StructuredOutputRequest` and the per-step
  bitmask, never in a KV block. Structured-output batch lanes are exactly the lanes to mark no-store.
- **R5 cascade attention.** `get_num_common_prefix_blocks` counts `ref_cnt == len(req_to_blocks)`; a no-store
  request raises the denominator and can only *disable* an optimisation, as any divergent-prefix request does.

## 4. Receipts protocol (live; server must be idle)

Identical owner turn in every arm (Codex: A0 turn 3 vs A1/A2 turn 4 was not a clean comparison).

| arm | batch | steps |
|---|---|---|
| A0 | none | owner `C` (~80K, one salt) turns 1–3 → **turn 4** |
| A1 | 16 lanes, ~30K each, distinct prefixes, own salts, **no flag** | `C` 1–3 → batch to completion → **turn 4** |
| A2 | same lanes with `vllm_xargs {"skip_writing_prefix_cache": 1}` | as A1 |
| A3 (informational) | lanes that **share** `C`'s system prompt, flag on | as A1 — shows the residual read-touch effect |

`/reset_prefix_cache` between arms; each arm twice; deterministic generation. Primary metric: turn-4
`cached_tokens / prompt_tokens` of `C`. Secondary: turn-4 TTFT, `C`'s `num_preemptions` (log), the batch's
own `cached_tokens` in A2 (must stay > 0 on a second flagged lane that shares a prefix: reads still work).
**Pass criterion fixed in advance: A2 ≥ 0.9 × A0 and A2 − A1 ≥ 0.3 × A0.** If A1 ≈ A0 the premise fails on
this pool and the PR is closed as a negative result. Also: both-rank `GLM53_APC_NO_STORE` in `docker exec …
env`; the resolution + suppression receipt lines after the first flagged request; `"skip_writing_prefix_cache":
"yes"` → HTTP 400 naming the field; JSON `true` → pydantic 400; `tests/probe_prefix_equivalence.py` PASS with
the flag on the warm runs (reads unaffected) and a no-store→cold pair (same prompt re-sent after a flagged run
reports `cached_tokens` 0 with an identical position-0 token / ≤ 3× floor).

## 5. Upstream

The mechanism is generic (any hybrid/Mamba/SWA model in vLLM has the LRU re-ordering problem). RFC outline:
typed `skip_writing_prefix_cache` on `SamplingParams`/`PoolingParams`, `from_optional`, the OpenAI request
models; guards at the two `BlockPool` sites with the `num_cached_block` sentinel argument above; connector
offload-decision symmetry; tests mirroring C1–C6. Not filed yet — gated on the live receipts.

## 6. Audit log

- Codex design advisory (research train, 12 findings): kept the `BlockPool` placement; adopted strict value
  parsing, explicit read semantics, connector scope statement, same-turn A/B, hybrid/CoW/preemption tests.
- Codex final-diff review (`CODEX-PR4-REVIEW.md`, ship-with-changes, 4 findings): JSON boolean claim corrected
  (coerced, not rejected — accepted with the intended meaning; the proposed pre-validation in the three API
  models is not adopted: it would add entrypoint anchors for a case that is already semantically exact);
  equivalence probe origin/invocation stated (it ships on #79, not on this branch); partial-site receipt
  stimulus corrected; patcher now verifies every generated snippet verbatim on an already-marked file and
  writes atomically (temp file + `os.replace`). `move_block_hashes` rebuttal accepted as sound.
- Codex build-plan advisory (`CODEX-PR4-PLAN.md`, build-with-changes, 9 findings): adopted — isolated C1 legs,
  live 6/7-group fixture, both preemption cases, API-boundary validation (finding 5 matched an independent
  read of `EngineCore.preprocess_add_request`), typed PoolingParams dropped, resolution-time receipt,
  parse-before-kill-switch rule, placement-discriminating multi-step tests. Rebutted with evidence (§2.2):
  `move_block_hashes` reachability for a no-store request.
