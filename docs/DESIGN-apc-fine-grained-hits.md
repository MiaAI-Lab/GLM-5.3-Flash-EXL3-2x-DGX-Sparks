# DESIGN — fine-grained prefix-cache hits at a kpool/mamba-safe alignment

Train: `finegrained`. Date: 2026-08-31. Status: **design + patch + host test done, NOT applied, not live-tested.**
Method: read-only inspection of the running fork (`docker exec glm53-exl3-head`, no restarts, no inference requests)
plus the recipe checkout mirror at `/Users/qualitycontrol/Documents/local-models/glm-exl3-recipe-fork`.

### Citation provenance — read this before trusting a line number

Two files in the container are **already overlay-patched**, so their line numbers are *not* pristine upstream:

| file | overlay marks | note |
|---|---|---|
| `vllm/v1/core/kv_cache_coordinator.py` | 9 × `glm53` | `overlay/patch_hybrid_prefix_hit.py` applied. All `kv_cache_coordinator.py:N` below are **live-container** numbers; pristine upstream is ≈ `N − 25` for `N > 131`. |
| `vllm/v1/core/sched/scheduler.py` | 15 × `glm53` | `overlay/patch_scheduler_decode_floor.py` applied. |
| `single_type_kv_cache_manager.py`, `kv_cache_utils.py`, `block_pool.py`, `kv_cache_manager.py`, `kv_cache_interface.py`, `config/cache.py` | 0 | pristine fork source; line numbers are stable. |

(counts: `docker exec glm53-exl3-head grep -c glm53 <file>`, run 2026-08-31.)
Anchor **text** is quoted for every load-bearing claim so the citation survives drift.

---

## 0. Verdict

**The premise in `FIX-PLAN-CACHE-2026-08-31.md` §Fix 3 and `RUNBOOK-CACHE-FIX-2026-08-31.md` L21–22 is wrong in both
of its inputs, and the fix is smaller and better than planned.**

The plan assumed `kpool = 128` and "mamba state materialises only at 896-token chunk ends", hence
`_cache_hit_alignment_tokens = lcm(896, 128) = 896`. In fact:

1. **`index_kpool = 4`**, not 128 — from the served model's own `config.json` (§1.4). `KpoolTailSpec.block_size ==
   index_kpool` (`v1/kv_cache_interface.py:743`, built at `models/glm5next/nvidia/attention.py:191-198`).
2. **There is no 896-token mamba chunk.** In `mamba_cache_mode == "align"` the mamba block size is set equal to
   the attention block size — `platforms/interface.py:932-933`:
   `if cache_config.mamba_cache_mode == "align": cache_config.mamba_block_size = cache_config.block_size` — so
   **mamba `block_size` = 3584**, same as MLA. The 896 in the overlay notes is the *indexer storage block*,
   `storage_block_size = spec.block_size // index_kpool = 3584 // 4 = 896`
   (`models/glm5next/nvidia/attention.py:142`); it is a DeepGEMM paged-MQA tiling quantity and **never a
   prefix-cache hit boundary**. (This also re-reads the FIX-PLAN's "1 MLA id + 4 mamba-snapshot ids per 3584-token
   segment" correctly: 4 = the number of mamba *groups* (gids 2–5), not 3584/896.)
3. **Mamba align-mode state does not only materialise at block ends.** The fork contains a complete
   producer/consumer/scheduler triple that deliberately snapshots the running state at an arbitrary
   `hash_block_size` boundary (§3). That machinery is *dormant* precisely because the veto this train is about
   disables it.
4. Therefore the required alignment is **`lcm(hash_block_size=64, index_kpool=4, drafter_block=64) = 64`** — i.e.
   `_cache_hit_alignment_tokens` needs **no change at all**. The value it already computes when
   `enable_partial_hash_hits` is true (`hash_block_size`, `kv_cache_coordinator.py:643-650`) is exactly the
   state-safe value.
5. **Setting the alignment to 896 as the plan proposed would have been silently incorrect**, not merely
   conservative: the fine-grained lookup paths index the raw hash list *positionally* and are only sound when
   `alignment_tokens == hash_block_size` (§2.3). At `alignment_tokens = 896` with `hash_block_size = 64` the code
   would report `hit_length = (fine_idx+1) × 896` for a hash whose true prefix length is `(fine_idx+1) × 64` — a
   14× overstatement of the hit. **Do not ship the reviewer's `896` one-liner.**

So the whole change is: **stop a manager that never participates in prefix caching from vetoing everyone else, and
replace that accidental veto with the invariant it was standing in for** — verified at runtime from the actual
specs, and refusing to start rather than degrading silently if it does not hold (§4.3). Patch:
`overlay/patch_apc_fine_grained_hits.py` (one anchor for the gate, one for the helper block, transactional and
fail-closed in both directions, and a no-op on a vLLM that already carries the fix). Host test:
`tests/test_apc_fine_grained_hits.py`, 195 checks, all green against the live source. Opt-out:
`GLM53_FINEGRAINED_APC=0`, restart-only, validated as exactly `0`/`1` by both the launcher and the coordinator.

**Status: not deployed and not measured.** The five blocking live receipts — exact hit length at the 64 grid,
CoW partial-tail durability, kpool tail at zero/nonzero remainders, temp-0 equivalence off-grid vs a
block-aligned control, and the drafter eagle-peek at the 64 drop unit — are specified with pass criteria in
**§6.5** and none has been collected.

Expected gain and its honest bound are in §6 — it is **not** a flat "recompute ≤ 63 tokens"; the producer registers
only one fine boundary per request, which caps the win in a way the train brief did not anticipate. §6.3 states
what actually improves and by how much.

---

## 1. Ground truth for this deployment

### 1.1 Serving command (`cat /proc/1/cmdline` in the head container)

```
vllm serve .../GLM-5.3-Flash-tr3-4bpw --tensor-parallel-size 2 --nnodes 2 --enable-prefix-caching
  --quantization exl3 --max-model-len 1000000 --max-num-seqs 16 --max-num-batched-tokens 2048
  --kv-cache-dtype fp8 --no-scheduler-reserve-full-isl
  --speculative-config {"method":"dflash","num_speculative_tokens":7,"draft_tensor_parallel_size":2}
```

Relevant negatives: **no `--block-size`, no `--prefix-match-unit`, no `--mamba-block-size`, no
`--kv-transfer-config`** (all block sizes are auto-derived; there is no KV connector / no PD).

### 1.2 KV cache group layout (boot log, `docker logs glm53-exl3-head`)

```
[interface.py:635]  Setting kv cache block size to 64 for DEEPSEEK_V32_INDEXER backend.
[interface.py:926]  Setting attention block size to 3584 tokens to ensure that attention page size
                    is >= mamba page size.
[config.py:605]     Mamba cache mode is set to 'align' for Glm5NextForConditionalGeneration by default
                    when prefix caching is enabled
[kv_cache_coordinator.py:635] WARNING Disabling fine-grained prefix-cache hits because these KV cache
                    managers require block-aligned lookups: KpoolTailManager.
[kv_cache_coordinator.py:709] hybrid APC groups: [('MLAAttentionSpec', [0], 'FullAttentionManager', False),
                    ('MambaSpec', [2, 3, 4, 5], 'MambaManager', False),
                    ('SlidingWindowSpec', [6], 'SlidingWindowManager', True)]; eagle_group_ids=[6]
[kv_cache_utils.py:2297] GPU KV cache size: 1,463,768 tokens
```

| group id | spec | manager | `block_size` | participates in APC | in `attention_groups`? |
|---|---|---|---|---|---|
| 0 | `MLAAttentionSpec` | `FullAttentionManager` | **3584** | yes | yes |
| **1** | **`KpoolTailSpec`** | **`KpoolTailManager`** | **4 (= `index_kpool`)** | **no** | **no** |
| 2–5 | `MambaSpec` (KDA linear-attn, 34 layers → 4 groups), `mamba_cache_mode="align"` | `MambaManager` | **3584** (= `cache_config.block_size`, `platforms/interface.py:932-933`) | yes | yes |
| 6 | `SlidingWindowSpec` (DFlash2 drafter) | `SlidingWindowManager` | **64** | yes | yes (eagle) |

`scheduler_block_size = lcm(3584, 4, 3584, 64) = 3584`; `hash_block_size = gcd(3584, 3584, 64) = 64` — the
kpool tail's 4 is excluded from the GCD (`kv_cache_utils.py:605-687`, `resolve_kv_cache_block_sizes`, exclusion at
`:668-680`). Neither value is printed by this build; both are derived. The only *directly* evidenced fact is the
inequality `hash_block_size < 3584`, which follows from the `kv_cache_coordinator.py:635` warning firing at all
(it sits inside `if self.enable_partial_hash_hits:`, which requires a mamba group with
`block_size > hash_block_size`).

**There is no 896-token boundary anywhere in the KV cache geometry.** 896 = `3584 // index_kpool` is the indexer
*storage* block (`models/glm5next/nvidia/attention.py:142`), which exists so DeepGEMM paged-MQA pool pages of
32/64 entries tile it. It is not a cache block size, not a mamba chunk, and not a reachable hit boundary.

Group **1 is absent from the `hybrid APC groups:` log line** — that log enumerates `self.attention_groups`
(`kv_cache_coordinator.py:709-721`, inserted by `patch_hybrid_prefix_hit.py`), and
`verify_and_split_kv_cache_groups` skips non-participating groups (`kv_cache_coordinator.py:664-665`:
`if not g.kv_cache_spec.participates_in_prefix_caching: continue`). **This is the primary receipt that
`KpoolTailManager` is never consulted for a cache hit — and it is vetoing one anyway.**

Provenance of the group table: group ids 0/2-5/6 and their manager classes are read directly from the
`kv_cache_coordinator.py:709` boot line. **Group 1's identity is [INFERENCE]** — it is the gid gap in that line
(non-participating groups are skipped, `kv_cache_coordinator.py:664-665`) combined with the group assembly order
at `kv_cache_utils.py:1570-1575` and the fact that `kv_cache_coordinator.py:635` names `KpoolTailManager`. Group
0's and 2-5's `block_size` follow from `cache_config.block_size = 3584` (boot line `interface.py:926`) via
`platforms/interface.py:924-933`; group 6's 64 is logged directly (`kv_cache_utils.py:1549`: "DFlash2 drafter KV:
padded slot-share block=64 mla_page=2351104 (was block=16)").

### 1.3 `hash_block_size` is a first-class knob

`config/cache.py:57-68`:

```python
prefix_match_unit: int | None = Field(default=None, gt=0)
"""The finest token boundary (in tokens) a prefix-cache hit can land on. ...
This equals to the `hash_block_size` used throughout the KV cache code."""
```

CLI `--prefix-match-unit` (`engine/arg_utils.py:1236`). Not set by the recipe, so the GCD default (64) applies.
This matters for §4: if a future GLM config had a `index_kpool` that did **not** divide 64, the correct remedy is
to *raise* `--prefix-match-unit` to a kpool-safe divisor of every participating block size — **not** to decouple
the alignment from the hash size.

### 1.4 `index_kpool = 4` (decisive)

Served model `config.json` → `text_config`:

```
index_kpool = 4
index_kpool_always_select_tail = True
index_kpool_compress = True
index_topk = 2048
index_head_dim = 128
linear_attn_config.kda_layers = [0,1,2,4,...,44]   # 34 KDA layers; 11 full-attn layers
```

Independently corroborated inside vLLM: `kv_cache_utils.py:670-673` — *"Including them would drag the GCD down to
the scratch size (**kpool=4**)"*. And `v1/kv_cache_interface.py:743`: *"One block of `block_size` (==
`index_kpool`) slots per request"*.

**`64 % 4 == 0`.** The current `hash_block_size` is already kpool-safe.

---

## 2. Q1 — what `_cache_hit_alignment_tokens` must be

### 2.1 Where the veto happens

`kv_cache_coordinator.py:617-639` (live text, anchor for the patch):

```python
        # Fine-grained hash hits require Mamba "align", no context
        # parallelism, and compatible cache managers in every group.
        has_partial_mamba_group = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            and g.kv_cache_spec.mamba_cache_mode == "align"
            and g.kv_cache_spec.block_size > hash_block_size
            for g in kv_cache_config.kv_cache_groups
        )
        self.enable_partial_hash_hits = dcp_world_size == 1 and has_partial_mamba_group
        if self.enable_partial_hash_hits:
            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager in self.single_type_managers      # <-- ALL groups
                if not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }
            if unsupported_partial_hit_managers:
                self.enable_partial_hash_hits = False
                logger.warning_once(...)
```

For this deployment the set evaluates to `{"KpoolTailManager"}` and nothing else:
`FullAttentionManager.supports_fine_grained_hash_lookup = True`
(`single_type_kv_cache_manager.py:681`), `MambaManager` = `True` (`:1382`), `SlidingWindowManager` inherits
`False` from the base ClassVar (`:47`) **but its `block_size == 64 == hash_block_size`**, so it is not a blocker;
`KpoolTailManager` overrides to `False` (`:1116`) **and** has `block_size = 4 ≠ 64`. Hence the boot warning.

Consequence: `kv_cache_coordinator.py:642-650`

```python
    @property
    def _cache_hit_alignment_tokens(self) -> int:
        return (self.hash_block_size if self.enable_partial_hash_hits
                else self.scheduler_block_size)
```

→ **3584**, and every hit is rounded down to a 3584 multiple.

### 2.2 The state-safety constraints on a hit length `H`

| # | constraint | source | value here |
|---|---|---|---|
| C1 | `H % hash_block_size == 0` — hashes exist only at hash boundaries | `kv_cache_utils.py:691+` `get_request_block_hasher`, "Hashes are computed at `hash_block_size` granularity and chained over the prefix" | 64 |
| C2 | `H % index_kpool == 0` — the kpool indexer tail must hold **no** in-progress pool at the resume point | §2.4 | 4 |
| C3 | a mamba state snapshot must be *registered* at `H` | §3 — guaranteed by construction (lookup only returns `H` where a cached block was found); imposes no extra modulus beyond C1 | — |
| C4 | drafter SWA window must be re-derivable at `H` | `SlidingWindowManager.find_longest_cache_hit` asserts `alignment_tokens % block_size == 0` (`single_type_kv_cache_manager.py:917-919`) → needs `H % 64 == 0`; and `patch_hybrid_prefix_hit.py` already removes this group from the hybrid `min` (`kv_cache_coordinator.py:856-866`), leaving its blocks empty so a fresh window is allocated | 64 |
| C5 | `alignment_tokens == hash_block_size` — a **hard precondition of the lookup code**, not a policy choice | §2.3 | 64 |

**`lcm(64, 4, 64, 64) = 64`.** Required alignment = **64 = `hash_block_size`** = what the code already computes.
No change to `_cache_hit_alignment_tokens` is needed or safe.

### 2.3 Why the alignment cannot be raised above `hash_block_size` (the plan's 896 is unsound)

`kv_cache_utils.py:2765-2795`, `resolve_block_hashes`, documents the precondition explicitly:

```python
    # Fine-grained partial hits keep the raw hashes. The caller passes
    # alignment_tokens = hash_block_size to enable them, else >= block_size.
    if (supports_fine_grained_hash_lookup and alignment_tokens is not None
            and alignment_tokens < block_size and block_size % alignment_tokens == 0):
        return block_hashes      # <-- RAW list, stride == hash_block_size
```

The raw list has stride `hash_block_size`. Both fine-grained consumers then treat `alignment_tokens` as that
stride:

* `FullAttentionManager.find_longest_cache_hit`, `single_type_kv_cache_manager.py:748-763`
  ```python
  scale_factor = block_size // alignment_tokens
  first_partial_idx = len(computed_blocks[0]) * scale_factor
  ...
      cached_tail = block_pool.get_cached_block(block_hashes[fine_idx], ...)
      ...
      hit_length = (fine_idx + 1) * alignment_tokens
  ```
  and `:724-726` builds `BlockHashListWithBlockSize(block_hashes, alignment_tokens, block_size)` over the raw
  list.
* `MambaManager.find_longest_cache_hit`, `single_type_kv_cache_manager.py:1438-1458`
  ```python
  if alignment_tokens < block_size and block_size % alignment_tokens == 0:
      hash_block_size = alignment_tokens          # <-- literal rebind
      scale_factor = block_size // hash_block_size
      ...
          num_tokens = (fine_idx + 1) * hash_block_size
          block_hash = block_hashes[fine_idx]
  ```

Index `fine_idx` in the raw list denotes prefix length `(fine_idx+1) × hash_block_size`. The code assigns
`hit_length = (fine_idx+1) × alignment_tokens`. **Sound iff `alignment_tokens == hash_block_size`.** With the
plan's `alignment_tokens = 896` and `hash_block_size = 64` every fine hit would be overstated 14×: the engine
would skip 896 tokens of prefill per matched 64-token boundary and feed the model a KV/state prefix that does not
correspond to its token ids. That is a silent-wrong-output class of bug, not a slowdown.

Corollary: **"raise the alignment" is never the right lever. The lever is `--prefix-match-unit`**, which moves
`hash_block_size` and `alignment_tokens` together and is validated against every participating group's block size
(`kv_cache_utils.py:680-687`).

### 2.4 Why C2 (`H % kpool == 0`) is the real content of "wrong indexer tail state is fatal"

`KpoolTailManager` docstring, `single_type_kv_cache_manager.py:1098-1114`:

> The GLM5Next kpool indexer tail cache holds the in-progress (incomplete) pool's raw K + gate score: exactly one
> block of `kpool` slots per request, overwritten in place by `pos % kpool` as decode/spec-decode advances.
> Prefill seeds it; … decode reads it to compress the boundary pool correctly.

and `v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py:317`:

> pool (valid == `index_topk` + `context % index_kpool`)

The number of *valid* indexer entries is `index_topk` **plus `context % index_kpool` raw tail tokens**. On a warm
hit ending at `H`, `KpoolTailManager.allocate_new_blocks` hands the request a **fresh** block
(`single_type_kv_cache_manager.py:1174-1187`) and only `[H, N)` is prefilled. If `H % kpool ≠ 0`, the
`H % kpool` raw K/gate entries of the current in-progress pool were never written, and
`index_kpool_always_select_tail = true` makes the indexer select them anyway → the boundary pool compresses
garbage. That is the fatal case, and it is a **modulus condition on the hit length**, not a statement about
lookup capability.

Here it is satisfied for free: **any** 64-multiple is a 4-multiple.

---

## 3. Q2 — does the mamba producer/consumer already work at 64, and what `hash_block_size` do they assume?

**Yes, fully, and they assume `block_pool.hash_block_size` exactly.** There are three cooperating pieces, all
present in the running image and all currently dormant because `enable_partial_hash_hits` is `False`.

### 3.1 Producer — `MambaManager._cache_partial_tail_block` (`single_type_kv_cache_manager.py:1832-1873`)

```python
        hash_block_size = self.block_pool.hash_block_size        # 64, NOT alignment_tokens
        if self.block_size == hash_block_size:   return None
        if num_tokens % self.block_size == 0:    return None     # dense block already covers it
        if num_tokens % hash_block_size != 0:    return None
        latest_prompt_hash_boundary = (request.num_prompt_tokens // hash_block_size) * hash_block_size
        if num_tokens != latest_prompt_hash_boundary:  return None   # EXACT equality
        block_idx = num_tokens // self.block_size
        source_block = blocks[block_idx]
        ...
        partial_hash = self.block_pool.cache_partial_block(
            request=request, block=source_block, num_tokens=num_tokens,
            kv_cache_group_id=self.kv_cache_group_id, block_size=self.block_size)
        ...
        self._partial_hit_reqs[request.request_id] = (block_idx, source_block)
        self._producer_partial_tail_reqs[request.request_id] = num_tokens
```

So a mamba state snapshot **is** registered at an arbitrary 64-multiple `B = ⌊P/64⌋·64` (P = prompt length) — it
does *not* have to be a mamba block end (3584 here). `_cache_partial_tail_block` is reached from
`MambaManager.cache_blocks:1813-1816` only when `mamba_cache_mode == "align"`.

The exact-equality guard is why the state is trustworthy: the running-state block is overwritten in place every
step, so the snapshot is only valid if the step *ended* at `B`. It is then made durable by the CoW in
`allocate_new_blocks:1740-1775` (`move_block_hashes` + `_pending_cow_copies`, or `_apply_cow`).

`FullAttentionManager._cache_partial_tail_block` (`:793-821`) does the same for MLA, also keyed on
`block_pool.hash_block_size`, but with `boundary_tokens > num_tokens` rather than equality — correct, because
attention KV is append-only.

`block_pool.cache_partial_block` (`block_pool.py:445-512`) asserts `block_size > self.hash_block_size` and
`block_size % self.hash_block_size == 0` and stores the entry under the prefix-chain hash at `num_tokens`
(`_get_partial_block_hash`), i.e. **raw hash index `num_tokens/hash_block_size − 1`** — the exact index the
consumers in §2.3 probe. Producer and consumer agree, at `hash_block_size`.

### 3.2 Scheduler — the chunk stop that makes the exact-equality guard reachable

`sched/scheduler.py:411-420`:

```python
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align")
        # A finer prefix_match_unit is configured: a mamba partial tail entry
        # can only be registered by a step ending exactly at the prompt's last
        # hash boundary, so the split adds that stop.
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and self.hash_block_size < self.block_size
            and self.kv_cache_manager.coordinator.enable_partial_hash_hits)   # <-- reads the veto
```

and `sched/scheduler.py:495-531` (`_mamba_block_aligned_split`):

```python
        # Invariant: slot p holds the state after exactly (p + 1) * block_size
        # tokens. State is written at chunk ends, so chunk ends must be block
        # aligned. Exempt: the prompt's last chunk, ...
        ...
        tail_boundary = (request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
                         if self.mamba_partial_cache_hit else 0)
        stops = (
            next_block_boundary if start % block_size != 0 else 0,
            last_cache_position,
            # Fine-grained hits: the prompt's partial-tail entry can only be
            # registered by a chunk ending exactly at its last hash boundary.
            tail_boundary if last_cache_position < tail_boundary < request.num_prompt_tokens else 0,
            ... shared_prefix_boundary ...)
        end = min((s for s in stops if start < s < end), default=end)
```

**This is the decisive evidence that "mamba align-mode state only materialises at block ends" is false.**
The scheduler deliberately breaks its own "chunk ends must be block aligned" invariant for exactly one chunk so
the state lands on a 64-boundary, and `_cache_partial_tail_block` then lifts it out via CoW. The whole path is gated on
`coordinator.enable_partial_hash_hits` — which the KpoolTail veto pins to `False`. **One veto disables three
cooperating subsystems.**

Note the scheduler reads `coordinator.enable_partial_hash_hits` in its own `__init__`, *after* the coordinator is
constructed (`sched/scheduler.py:374` builds `KVCacheManager`, `:417` reads the flag). **Patching only the
coordinator is sufficient; the scheduler picks the change up.**

### 3.3 Consumer

Covered in §2.3. At `alignment_tokens = hash_block_size = 64` all three participating managers are correct:

| manager | `block_size` | path taken at alignment 64 | result |
|---|---|---|---|
| `FullAttentionManager` (MLA) | 3584 | `fine_grained = 64 < 3584 ∧ 3584%64==0` → True; phase 1 coarse over 3584-blocks, phase 2 probes the 55 interior 64-boundaries of the first non-full block (`:745-764`); the round-down at `:775` becomes a no-op | hit at a 64-multiple |
| `MambaManager` | 3584 | partial branch `:1438-1458` (`64 < 3584 ∧ 3584%64==0`); `scale_factor = 56` | hit at a 64-multiple |
| `SlidingWindowManager` (drafter) | 64 | `resolve_block_hashes` early-returns (`block_size == hash_block_size`, `kv_cache_utils.py:2779-2780`); the assert at `:917-919` passes (`64 % 64 == 0`); all `block_size != alignment_tokens` re-align loops become no-ops | hit at a 64-multiple; excluded from the `min` by `patch_hybrid_prefix_hit.py` anyway |

---

## 4. Q3 — is excluding non-participating managers *sufficient*?

**Necessary but not sufficient on its own. Two things must hold, and only one of them is "exclusion".**

### 4.1 Exclusion is right, and the fork already does it in 2 of 3 places

Same fork, same release, three sites that ask "which groups matter for prefix caching":

| site | behaviour |
|---|---|
| `kv_cache_utils.py:668-680` (`resolve_kv_cache_block_sizes`) | **excludes** non-participating groups, with a comment naming KpoolTailSpec and kpool=4 explicitly |
| `kv_cache_coordinator.py:664-665` (`verify_and_split_kv_cache_groups`) | **excludes** them (`continue`) |
| `kv_cache_coordinator.py:589-598` (the `block_size % hash_block_size` assert) | **excludes** them |
| **`kv_cache_coordinator.py:627-632` (the partial-hit veto)** | **does not** — scans `self.single_type_managers` |

Four sites, three exclude, one does not — 13 lines apart in the same constructor. That is an oversight, not a
deliberate conservatism. (See §8.)

### 4.2 `KpoolTailManager` does **not** need alignment-aware lookup logic

It never runs a lookup: it is absent from `attention_groups` (§1.2 receipt), its
`find_longest_cache_hit` returns `0` unconditionally (`single_type_kv_cache_manager.py:1119-1133`), its
`cache_blocks` is a no-op (`:1135-1142`), `get_num_common_prefix_blocks` returns 0 (`:1144-1145`),
`add_local_computed_blocks` is a no-op (`:1189-1199`), and it allocates exactly one fresh block per request
(`:1161-1187`). There is no cached tail state that could be *stale*. Adding alignment logic to that class would be
dead code.

### 4.3 What must be added instead: an explicit scratch-alignment invariant

The veto, while wrong in its scope, was accidentally standing in for a real constraint (C2, §2.4). Nothing else in
the codebase enforces it — on the contrary, `resolve_kv_cache_block_sizes` **deliberately keeps kpool out of the
GCD** (`kv_cache_utils.py:668-680`), so `hash_block_size` is free to be finer than, or coprime with, `index_kpool`
in some future config. Today `64 % 4 == 0` by luck of the numbers, not by construction.

So the patch replaces the accidental veto with the explicit one:

```
for every group with participates_in_prefix_caching == False:
    alignment = verify_from_the_actual_spec(manager, group.kv_cache_spec)
    require hash_block_size % alignment == 0
```

`alignment` is **read at runtime from the real objects, never assumed**, and **every** source is cross-checked
against every other: `fine_grained_hit_alignment` and `block_size` on the manager *and* on the spec, plus
`spec.index_kpool` / `spec.kpool` — which for `KpoolTailSpec` *is* the quantity the invariant is about. Two
sources that disagree, a missing `block_size`, or a non-integer are all treated as *unverifiable*, not as
*probably fine*.

An explicit `fine_grained_hit_alignment` capability is **not authoritative on its own.** A capability of `16` on a
spec whose `block_size` is `128` is a contradiction, and a contradiction is *unverifiable* → refuse. Taking the
capability at its word there would enable 64-token hits (16 divides 64) on a group that actually needs 128 —
precisely the unsafe mid-pool resume this gate exists to prevent. The capability's real job is narrower: it lets a
future scratch manager that exposes no `block_size` at all state its requirement, and when it is the only source
present it stands alone. (Host test B13/B13b/B13c/B13d.)

Values are parsed with a **strict integer check, never `int()`**. `int("4.5")` raises but `int(4.5)` is `4`, and
`int(" 4")` is `4`: a value that is not an integer at all would come back wearing a verified badge, and `4`
divides 64. Only a real `int` (not `bool`) or an unpadded ASCII decimal string is accepted; `4.5`, `" 4"`, `""`,
`"0x40"`, `"4_0"` and non-ASCII digits are unverifiable. (Host test B23/B23b/B26.)

The `GLM53_FINEGRAINED_APC` kill switch is likewise **exactly `0` or `1`**, at both ends. Any other value —
`true`, `01`, `" 0"`, `""` — refuses at coordinator init with a `[glm53-apc-finegrained]` message rather than
being read as "on", because guessing it wrong silently changes the KV-cache hit alignment of the whole engine.
`start.sh` validates the same knob the same way inside its `# GLM53 numeric config guard` block, so the typo is
caught by the launcher (on `start`/`restart`, before `stop`) instead of by a container that will not boot.
(Host test B27, C1–C3.)

**A violation, or an unverifiable scratch group, refuses to start — but only when it is the thing that matters.**
`HybridKVCacheCoordinator.__init__` raises `Glm53FineGrainedAPCError` with a `[glm53-apc-finegrained]` message
naming the group, the alignment, where the alignment was read from, and the remedy. This is deliberate and it is
the one place this patch is *less* forgiving than upstream:

* silently falling back to block-aligned hits would leave a box serving at 3584 alignment while the operator
  believes fine hits are on — the exact failure mode this whole change exists to remove, and one that shows up
  only as unexplained TTFT;
* the safety argument in §2.4 is what licenses excluding the scratch group from the veto at all. On a layout
  where that argument does not hold, the right answer is to stop, not to guess.

The escape hatch is documented and restart-only: **`GLM53_FINEGRAINED_APC=0`** restores the upstream
(all-managers) veto verbatim and never raises.

The *participating* half of the gate keeps upstream's behaviour exactly — a coarse manager that cannot answer a
fine lookup **disables** fine hits with upstream's warning, because block-aligned hits are the correct and safe
fallback for that case.

#### The mixed-layout matrix

Two populations, so four combinations, and only one of them may raise:

| participating blocker | scratch group unsafe / unverifiable | outcome | why |
|---|---|---|---|
| no | no | **ENABLE** | the verified case; alignment `hash_block_size` = 64 |
| no | **yes** | **RAISE** | nothing else stops fine-grained hits, so this refusal is the only thing between the layout and an unsafe mid-pool resume |
| **yes** | no | **DISABLE** | upstream's own condition and upstream's own fallback; alignment stays `scheduler_block_size` |
| **yes** | **yes** | **DISABLE** | the blocker already pins alignment to `scheduler_block_size`, so **no fine-grained hit is ever taken** and the scratch invariant is unreachable — the layout behaves exactly as upstream's does, and refusing to boot on it would convert an upstream-equivalent configuration into an outage |

The last row is why the gate *collects* scratch faults rather than raising on sight: the refusal is load-bearing
only where fine-grained hits would otherwise be **enabled**. The tolerated fault is not swallowed — it is named
in the receipt line (§5.1) as `KpoolTailManager: UNSAFE (requires hit alignment 128, …)`, so a later change that
removes the participating blocker cannot quietly promote row 4 into row 2 unnoticed.

Structural failures — manager/group cardinality mismatch, a group with no `kv_cache_spec`, a nonsense
`hash_block_size` — are *not* subject to this matrix and raise immediately: they mean the layout could not be
classified at all, so "a participating blocker makes it moot" cannot be established.

The host test pins every cell (B3/B7/B10–B14/B23 raise, B5/B8/B15 disable, B24/B25 mixed → disable, B28 drives
all four cells through the shipped gate block itself).

---

## 5. Q4 — the code change and the host test

### 5.1 `overlay/patch_apc_fine_grained_hits.py`

Recipe-overlay style, MARK `# [glm53-finegrained-apc]`, fail-closed, idempotent. Target:
`vllm/v1/core/kv_cache_coordinator.py`. Env override `GLM53_KV_COORDINATOR_PY` (same as
`patch_hybrid_prefix_hit.py`).

Three edits:
1. insert `import os` after `from abc import ABC, abstractmethod` (the live file does **not** import `os`);
2. insert a module-level helper block — `Glm53FineGrainedAPCError`, `_glm53_strict_int`,
   `_glm53_scratch_alignment`, `_glm53_finegrained_enabled`, `_glm53_connector_receipt`,
   `_glm53_finegrained_hit_gate` — bracketed by
   `# [glm53-finegrained-apc] helper-begin` / `helper-end` sentinels. Insert point: **before any sibling
   `_glm53_*` helper if one is already present, else before `def _validate_prefix_cache_retention_interval(`**.
   `patch_hybrid_prefix_hit.py` inserts its own helper before that same needle, so "always insert before the
   needle" makes the composed file depend on which patch ran first; anchoring ahead of the sibling makes the two
   byte-commutative in either order. (This was a real defect found by the corrected host test — see §5.2.)
3. replace the veto block (`kv_cache_coordinator.py:626-639`, anchor text quoted in §2.1) with the kill-switch
   check plus a call to the helper, plus the **effective-value receipt** below.

**Patcher fail-closed properties** (all pinned by host test Part A):

* **anchors counted before anything is mutated** — drift aborts with no write at all (A3, A3b);
* **transactional apply** — the result is compiled and fully validated *before* it is written, the write is a
  temp file in the same directory plus `os.replace`, and the bytes on disk are re-validated afterwards. An
  interrupted or failing run can never leave a half-patched coordinator, and never leaves temp litter;
* **a pre-existing `MARK` is not trusted** — the patcher does not simply skip. It validates the *complete*
  patched state (helper block sentinels, all four helper defs unique, the runtime tag, the kill switch, the
  enable-path log line, no surviving upstream veto, and that the file compiles) and fails closed if any of it is
  missing. A gutted or partially reverted overlay is a hard error, not a silent "already present" (A5);
* **cardinality asserted before iterating** — upstream pairs managers with groups using `zip()`, which silently
  truncates if the two lists ever diverge and would therefore *skip real blockers*. The helper refuses instead
  (B16, B16b);
* **"upstream already fixed it" is a no-op, not a failure** — vLLM main `e126687a` scopes this very veto to
  `KVCacheSpec.prefix_cacheable` groups (its scratch invariant being the `tokens_per_state` check in
  `resolve_kv_cache_block_sizes`). A recipe image rebased onto that vLLM must not fail to boot because our anchor
  is gone. The patcher extracts the actual `if self.enable_partial_hash_hits:` suite **by indentation** and, if
  that suite already carries both the `supports_fine_grained_hash_lookup` veto and a scoping attribute
  (`prefix_cacheable` / `participates_in_prefix_caching`), prints
  `[glm53-apc-finegrained] upstream already scopes the veto; nothing to do` and exits 0 without touching the
  file. The extraction is structural precisely so a neighbour cannot be mistaken for the fix: `cache_blocks`
  opens an identical `if` 35 lines below and `verify_and_split_kv_cache_groups` mentions
  `participates_in_prefix_caching` ~25 lines below, both of which a fixed-size text window would swallow. A veto
  that is merely *edited* or *deleted* is drift, not an upstream fix, and still fails closed (A6, A6b, A6c).

Kill switch: `GLM53_FINEGRAINED_APC=0` in the engine environment restores upstream behaviour without unpatching —
important because the launcher exports it to both ranks in `nccl_common` the way it already does for
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`, giving a restart-only rollback. It is accepted as **exactly `0` or `1`**
at both ends (§4.3): `start.sh` rejects anything else in `validate_numeric_config` (on `start`/`restart`, before
`stop`), and the coordinator rejects anything else at init.

Recipe wiring: mounted on both ranks and run in both in-container patch stanzas, alongside
`patch_hybrid_prefix_hit.py`. Order is irrelevant (test A4 proves the two patches commute byte-for-byte).

**The effective-value receipt.** One line, emitted on every boot on **both** ranks, stating what is actually in
force — not what was intended. `enabled/disabled` + `reason` + the alignment hits will really land on + every
scratch group that was checked + the connector receipt:

```
INFO [kv_cache_coordinator.py:NNN] [glm53-apc-finegrained] Fine-grained prefix-cache hits ENABLED:
     reason=every participating manager can answer a hash_block_size-granular lookup, and every
     non-participating scratch alignment divides it; effective alignment=64 tokens
     (hash_block_size; scheduler_block_size=3584); scratch groups checked:
     {'KpoolTailManager': 4}; KV-transfer connector: absent (no kv_transfer_config) --
     truncate_computed_blocks unreachable.
```

The disabled form is the same line with `DISABLED`, `effective alignment=3584 tokens
(scheduler_block_size; hash_block_size=64)`, and a `reason=` that is either the kill switch or the participating
blockers by name; upstream's own `Disabling fine-grained prefix-cache hits …` warning is still emitted beside it.
On the mixed layout of §4.3 row 4 the tolerated scratch fault appears in `scratch groups checked` as
`UNSAFE (…)` / `UNVERIFIABLE (…)`, so "we disabled for reason A while B was also wrong" is on the record. Host
test B29 asserts every field of all three forms against the shipped gate block.

The connector clause is the boot receipt Codex asked for against R1: it turns "this deployment has no
`--kv-transfer-config`, so the `truncate_computed_blocks` assert is unreachable" from an assumption into a logged
fact. It is best-effort and never a gate — if the config cannot be read it logs `unknown (<ExceptionType>)` and
init continues.

**That log line is the go/no-go gate for the whole change.** If it does not appear, nothing downstream is worth
measuring.

### 5.2 `tests/test_apc_fine_grained_hits.py` — host test, no vLLM import needed

Run: `GLM53_KV_COORDINATOR_PY_SRC=<coordinator copy> python3 tests/test_apc_fine_grained_hits.py`
(pull the source with `docker exec glm53-exl3-head cat .../kv_cache_coordinator.py > /tmp/…`). Optionally set
`GLM53_KV_COORDINATOR_PY_PRISTINE=<unpatched copy>` to get the second composition leg described below; it
defaults to `/tmp/kv_cache_coordinator_pristine.py` and is skipped with a printed note if absent.

Part A (mechanics): MARK / helper sentinels / all helper defs / `import os` present; patched file compiles;
second apply is a byte-identical no-op; **fails closed and leaves the file byte-identical when the gate anchor or
the helper insert point drifts**, with no temp litter; **fails closed when `MARK` is present but the patch is
incomplete** (helper block gutted, kill switch stripped, upstream veto reintroduced); composes with
`patch_hybrid_prefix_hit.py` in both orders, over both the live source and a pristine one, producing identical
bytes, and re-applying both is a no-op; and **A6**: a synthesized coordinator whose veto is already scoped to
`group.kv_cache_spec.prefix_cacheable` (vLLM main `e126687a`) is a clean exit-0 no-op that leaves the file
byte-identical and stays a no-op on re-run, while a *deleted* veto (A6b) and a *drifted but unscoped* veto (A6c —
with `participates_in_prefix_caching` present 25 lines below it, the window-vs-suite trap) both still fail closed.

> The both-orders check is what caught a real defect. Run only against the *live* coordinator — where
> `patch_hybrid_prefix_hit.py` is already applied, so re-applying it is a no-op — it was vacuous and passed.
> Against a **pristine** source the two orders differed: both patches inserted their helper before the same
> needle, so whichever ran second sat lower in the file. Same lines, different order, so semantically harmless,
> but it means "the overlay is commutative" was untested. Fixed by the anchor rule in §5.1(2); the test now runs
> A4 over every available source and takes `GLM53_KV_COORDINATOR_PY_PRISTINE` for the second one.

Part B (semantics): `exec`s the whole sentinel-bracketed helper block in a bare namespace, then drives it with
fakes. Note the policy of §4.3: participating blockers **disable**, scratch violations **raise** — unless a
participating blocker is also present, in which case the safe fallback wins and nothing raises.

| case | layout | expected |
|---|---|---|
| B1 | live layout (MLA 3584 / KpoolTail 4 / 4×Mamba 3584 / SWA 64), hash 64 | **enable**, scratch `{KpoolTailManager: 4}` |
| B2 | the upstream rule applied to the same layout | vetoes `{KpoolTailManager}` — reproduces the boot warning |
| B3 | hypothetical `index_kpool = 128` | **raise**, names the alignment and `GLM53_FINEGRAINED_APC=0` |
| B4 | `index_kpool = 32` | enable (any divisor of 64 is safe) |
| B5 | a *participating* coarse manager without fine lookup (SWA raised to 3584) | **disable**, blocker says `participating` |
| B6 | SWA `block_size == hash_block_size` despite `supports_fine_grained_hash_lookup=False` | enable |
| B7 | scratch `block_size = 0` | **raise**, no ZeroDivisionError |
| B8 | spec lacking `participates_in_prefix_caching` | treated as participating (upstream-safe default) → disable |
| B10 | scratch spec exposes no `block_size` | **raise** — unverifiable, not "probably fine" |
| B11 | scratch `spec.block_size` disagrees with `manager.block_size` | **raise** |
| B12 | scratch `spec.index_kpool` disagrees with `spec.block_size` | **raise** |
| B13 | explicit `fine_grained_hit_alignment = 16` agreeing with `block_size` | enable (16 divides 64) |
| B13b | the capability is the **only** source (no `block_size` anywhere) | enable — the forward-compat hook still works |
| B13c | capability `16` **contradicts** `block_size 128` | **raise** — 16 divides 64, so trusting it would have enabled unsafe hits |
| B13d | capability `4` contradicts `index_kpool 8` | **raise** |
| B14 | explicit `fine_grained_hit_alignment = 96` | **raise** |
| B23 / B23b | capability `"4.5"`; `spec.block_size = " 4"` | **raise** — `int()` would have truncated/trimmed both to `4`, which divides 64 |
| B24 / B25 | **mixed**: a participating blocker *and* a violating (B24) or unverifiable (B25) scratch group | **disable**, not raise — §4.3 row 4 |
| B15 | manager missing `supports_fine_grained_hash_lookup` entirely | treated as `False` → disable |
| B16 / B16b | manager/group cardinality mismatch | **raise**; B16b shows the `zip()` truncation would have hidden a real blocker |
| B17 | group with no `kv_cache_spec` | **raise**, not `AttributeError` |
| B18 | `hash_block_size` of `0` / `-64` / `None` / `64.0` | **raise** |
| B19 | alignment is genuinely read from the spec | source names every agreeing attribute, e.g. `manager.block_size == spec.block_size == spec.index_kpool` |
| B20 | connector boot receipt with no vLLM importable | returns a string, never raises |
| B22 | the **patched gate block itself**, lifted out of `__init__` and executed against fakes | `GLM53_FINEGRAINED_APC=0` disables and does **not** raise even on a layout that would otherwise refuse; unset/`1` enables |
| B26 | `_glm53_strict_int` accept/reject table: `4`, `"0004"`, `"+4"` accepted; `4.5`, `4.0`, `"4.5"`, `" 4"`, `"4 "`, `""`, `"0x40"`, `"4_0"`, `True`, Arabic-Indic/superscript digits rejected | every reject is a value `int()` would have taken |
| B27 | kill-switch parse: `"1"`/`"0"` only; `""`, `" 0"`, `"01"`, `"true"`, `"1.0"`, … **raise** at init, both directly and through the gate block | |
| B28 | the §4.3 **2×2 matrix**, all four cells, driven through the shipped gate block | enable / raise / disable / disable — and the tolerated fault is still named in the receipt |
| B29 | the **effective-value receipt**: enabled, kill-switched, and mixed-layout forms | each states enabled/disabled, `reason=`, the effective alignment, the scratch groups checked, and the connector; the disable path still emits upstream's own warning |
| B9 | arithmetic: `64 % 4 == 0`; `3584 % 64 == 0`; `896 = 3584/index_kpool` is the indexer *storage* block, not a hit boundary; `lcm(64,4,64,64) = 64` | — |
| B21 | the §6.5 L1 receipt arithmetic: `P = 31672` → fine hit `31616`, coarse control `28672`, `31616 % 3584 ≠ 0`, `31616 % 4 == 0` | — |

Every `raise` case additionally asserts the message carries the `[glm53-apc-finegrained]` tag and names
`GLM53_FINEGRAINED_APC=0` as the remedy.

Part C (launcher knob): sources start.sh's `# GLM53 numeric config guard` block in `bash` — the shipped shell,
not a paraphrase — and asserts `GLM53_FINEGRAINED_APC` is accepted only as exactly `0`, `1`, or unset (C2, 20
values); that it is validated inside that guard, which `main()` runs only on `start|restart` and, for `restart`,
before `stop` (C1); and that the validated value is the one both ranks receive — exported by value inside
`nccl_common`, with a caller-supplied value captured before `.env` and restored after (C3).

**Result on the live source, 2026-08-31: all 195 checks pass** (86 before the fix-first round).

The test is now **invoked** by the recipe, not merely copied: `Dockerfile` runs
`python3 /opt/glm53/test_apc_fine_grained_hits.py` immediately *before*
`patch_apc_fine_grained_hits.py`, because the drift and fail-closed legs need an unpatched partial-hit gate in
the target file. At that point `patch_hybrid_prefix_hit.py` is already applied — exactly the composition A4
asserts.

---

## 6. Q5 — live test plan and the honest expected gain

### 6.1 Correction to the brief's framing

The brief asks for "temp-0 logprob comparison … at hits ending at 896-multiples vs 3584-multiples". After §0.2
and §2 the right contrast is **64-multiples vs 3584-multiples**. 896 is not a boundary of anything the prefix
cache can land on: MLA and all four mamba groups have `block_size = 3584`, the drafter has 64, and 896 is only
`3584 // index_kpool`, the indexer's internal storage tiling.

### 6.2 Equivalence gate (blocking — reuse `tests/validate_apc_retention.py`'s method verbatim)

Use the numerical, sampling-free gate that `validate_apc_retention.py` (docstring L1-19 of
`glm-exl3-recipe-fork/tests/validate_apc_retention.py`) already settled on after the Codex re-review — do **not**
regress to first-divergence-of-text, which failed its own receipt:

* temp 0, `enable_thinking: false`, `logprobs: true`, `top_logprobs: 5`, 64 tokens, code + prose tasks;
* **cold-vs-cold noise floor**: same tokens, two disjoint `cache_salt` namespaces (measured 0.15–0.41 nats on this
  kit — TP=2 reduction order, EXL3, fp8 KV);
* **a true cold arm, not just a fresh salt.** The cold run must come from a reset cache —
  `POST /reset_prefix_cache`, or a separate engine process — for the *identical* turn-2 prompt. A different
  `cache_salt` only guarantees a different hash namespace; it does not guarantee the cold path was taken through
  the same code;
* **gate**: cold-vs-warm chosen tokens **identical at every returned position** (not merely over "leading
  agreeing positions" — a corruption that diverges at position 40, or that only reorders lower-ranked candidates,
  must not pass), and `max |Δlogprob|` over the **full top-5 at every returned position**
  ≤ `3 × max(cold-vs-cold max, 0.02)`. Position 0 remains the sharpest single probe of the kpool tail and the
  mamba snapshot, but it is no longer the only one that counts;
* **DFlash acceptance** over a 400-token generation: warm ≥ 0.85 × cold, from `vllm:spec_decode_*` counters;
  missing counters = fail. Acceptance is a **performance floor, not a correctness proof** — the target-logprob
  equivalence above is what proves correctness.

**Prompt construction is the part that must change.** Choose the turn-1 prompt length `P` so that the resulting
hit boundary is provably *not* on the old grid:

```
B = floor(P / 64) * 64
require:  B % 3584 != 0        # would have been reachable before -> proves nothing
          P % 64   != 0        # forces the scheduler tail_boundary stop (scheduler.py:523-524)
e.g. P = 3584*8 + 3000 = 31672  ->  B = 31616;  31616 % 3584 = 2944 ≠ 0;  31672 % 64 = 56 ≠ 0
```

Then: turn 1 cold (× 2 salts, noise floor) → short reply (`max_tokens = 32`) → turn 2 warm, same salt.

**`P` must be the real tokenized length, not the intended one.** Tokenize the fully rendered chat template
(`POST /tokenize` with the identical `messages` and `enable_thinking: false`) and assert `P == 31672` exactly
before running anything; adjust the filler text until it is exact. A `P` that is off by even one token moves `B`
and invalidates every number below. Likewise capture the **full assistant reply** and its exact
`usage.completion_tokens`, and tokenize the assembled turn-2 messages to get `N₂` exactly — the earlier "≈94
recomputed tokens" was raw arithmetic (`P mod 64 = 56`, plus a 32-token reply) that ignored per-message template
and serialization overhead, so the pass criterion is the **identity** in §6.5 L1, never the constant.

Also run the gate **at both retention settings**: dense (`GLM53_APC_RETENTION_INTERVAL=` empty) and the shipped
`14336`. See risk R2.

### 6.3 Expected performance gain — and its real ceiling

The train brief predicts "tail recompute ≤ alignment−1 = 63 tokens". **That is not achievable with the current
producer, and the design doc should say so.** Both `_cache_partial_tail_block` implementations register
**exactly one** fine boundary per request, at that request's own prompt tail:

* MLA: `boundary_tokens = request.num_prompt_tokens // hash_block_size * hash_block_size`
  (`single_type_kv_cache_manager.py:805`)
* mamba: `latest_prompt_hash_boundary`, same formula (`:1844-1846`)

No fine boundary is ever registered inside the *generated* region. So for a conversation where turn N had prompt
`P_N` and generated `G_N`, turn N+1's reconciled hit is

```
H = max( floor(C / 3584) * 3584 ,  B_N )      where C = common prefix length ≈ P_N + G_N
                                                    B_N = floor(P_N / 64) * 64
```

and `B_N` is only reachable when it lies inside the consumer's first non-full MLA block, i.e. when
`floor(C/3584)*3584 ≤ B_N` — which holds exactly when `P_N` and `C` fall in the same 3584-token MLA block, i.e.
**when the previous reply was short relative to 3584 tokens.** [INFERENCE from the phase-2 window
`single_type_kv_cache_manager.py:749-755`; to be confirmed by the live hit counts in §6.2.]

| scenario | hit today | hit after | wasted (already-computed) recompute |
|---|---|---|---|
| agent loop, short replies / tool calls (`G_N` ≲ 3584) — **the dominant case for OpenCode/subagents** | `⌊C/3584⌋·3584` | `⌊P_N/64⌋·64` | **`C − ⌊C/3584⌋·3584` (0–3583, mean ≈1792, ~1.5–3 s) → `G_N + (P_N mod 64)` (< 64 + reply)** |
| long single reply (`G_N` > 3584) | `⌊C/3584⌋·3584` | same | unchanged — never worse (phase 2 only *extends* phase 1) |
| subagent forking a root prefix at the root's prompt tail | `⌊C/3584⌋·3584`, or the 14336 retention grid | the root's exact prompt tail | directly addresses the FIX-PLAN §Fix 1 limitation "a *diverging* prefix hits only at the interval grid — measured 45 % of a ~29K prompt" |
| same prompt re-sent (retry, resample) | `⌊C/3584⌋·3584` | `⌊P/64⌋·64` | ≤ 63 — this is the only case that reaches the brief's bound |

Concrete falsifiable prediction for the §6.2 rig (`P = 31672`, `max_tokens = 32`, turn-2 prompt ≈ 31710):
**before: hit 28672 (90.4 %), ≈3038 recomputed tokens; after: hit 31616 (99.7 %), ≈94 recomputed tokens.**
If the observed post-patch hit is 28672 (i.e. unchanged) rather than 31616, the phase-2 window inference above
is wrong and §6.3 must be revised before shipping. There is no intermediate value to land on: every participating
group's dense boundary is 3584 (drafter 64, but it is excluded from the `min`), so the hit is either the old
coarse boundary or the fine one.

**Follow-up worth a separate train (out of scope here):** registering a partial tail at
`floor(num_computed_tokens/64)*64` during decode — not only at the prompt tail — would make the ≤63 bound real in
every scenario. It costs extra hash-map entries per request and therefore interacts directly with the block-id
budget that `FIX-PLAN` §Fix 1 is rationing, so it needs its own capacity ladder.

### 6.4 Non-equivalence live checks

* `bench_decode` solo throughput ±3 % (no decode-path change is expected; this catches the extra prefill chunk
  stop costing a step).
* 16-lane ladder unchanged.
* Pair ladder 45K/56K/66K/80K from `validate_apc_retention.py` check 3 — fine hits register **more** hash entries
  (see R2); this is the capacity regression detector.
* Cold TTFT unchanged (the `tail_boundary` stop adds at most one short prefill chunk per request,
  `sched/scheduler.py:523-524`).

---

### 6.5 Receipt ledger — the five blocking live receipts

Nothing below has been run (§R6). These are the receipts the adversarial review requires before this ships;
each is **blocking**, each names what to capture and what counts as a pass. Record the observed value next to
every expected one, including the ones that pass — a receipt with no recorded number is not a receipt.

**Boot receipt B0 (go/no-go for everything else).** The effective-value receipt of §5.1 — the line that states
what is actually in force, not what was intended. Grep it from **both ranks**, from `docker logs`, and paste both
outputs into the ledger:

```bash
# head (rank 0, this box) — the API rank
docker logs glm53-exl3-head   2>&1 | grep -n '\[glm53-apc-finegrained\]'
# worker (rank 1) — headless: `docker logs` is the ONLY place this line exists,
# there is no API and no console to read it from
ssh "$WORKER_SSH" "docker logs glm53-exl3-worker 2>&1 | grep -n '\[glm53-apc-finegrained\]'"
# and the negative control, on both:
docker logs glm53-exl3-head   2>&1 | grep -c 'Disabling fine-grained prefix-cache hits'
ssh "$WORKER_SSH" "docker logs glm53-exl3-worker 2>&1 | grep -c 'Disabling fine-grained prefix-cache hits'"
```

**Pass — all four, and the head is what matters most (it owns the scheduler that takes the hit), but a worker
that disagrees with it is a hard stop, since the two ranks would be running different KV alignments:**

1. exactly one `[glm53-apc-finegrained]` line per rank, reading `Fine-grained prefix-cache hits ENABLED`;
2. `effective alignment=64 tokens (hash_block_size; scheduler_block_size=3584)` — this is the receipt that the
   knob actually reached the engine, and it is the *only* one: `-e GLM53_FINEGRAINED_APC=1` in `docker inspect`
   proves the variable was passed, not that the gate honoured it;
3. `scratch groups checked: {'KpoolTailManager': 4}` — the verified alignment, from the real spec. Any
   `UNSAFE (…)` / `UNVERIFIABLE (…)` value here means §4.3 row 4 tolerated a fault because something else
   disabled fine hits: record it, and do not treat the run as a test of this change;
4. `KV-transfer connector: absent (no kv_transfer_config)` (R1), and the old
   `Disabling fine-grained prefix-cache hits … KpoolTailManager` warning count is `0` on both ranks.

If B0 fails, nothing downstream is worth measuring. If the engine instead refuses to start with a
`[glm53-apc-finegrained]` message, the running layout violates the §2.4 invariant — do **not** work around it
with `GLM53_FINEGRAINED_APC=0` and carry on measuring; that combination is the pre-patch baseline, not this
change. If a rank prints `upstream already scopes the veto; nothing to do` at build time, the image was rebased
onto a vLLM that carries the fix natively (§5.1) and this overlay is inert on it — the measurement is then about
upstream's implementation, not ours.

---

**L1 — exact hit-length receipt at the 64 grid.** *(Codex #7)*

Rig: §6.2, `P = 31672` verified by `/tokenize`, `max_tokens = 32`, turn 2 = turn-1 messages + the captured reply
+ a fixed short user turn, tokenized to `N₂`.

Capture, **per request** (aggregate ratios are corroborating only, never the gate):

* `usage.prompt_tokens_details.cached_tokens` for the turn-2 request;
* `usage.prompt_tokens` for the same request;
* the scheduler's per-request `local_cache_hit` / `local_compute` deltas;
* `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` deltas across the turn-2 request alone.

**Pass:**

| | expected | why |
|---|---|---|
| `cached_tokens` (patched) | **exactly 31616** | `floor(31672/64)*64`; not `≥`, not a percentage |
| `cached_tokens` (pre-patch control, same rig) | **exactly 28672** | `floor(31672/3584)*3584` — proves the rig can tell the two apart |
| recompute | `prompt_tokens − cached_tokens == N₂ − 31616` | the identity, not the ≈94 estimate |
| `local_cache_hit` delta | `== cached_tokens` | the two accountings must agree |

**`usage.prompt_tokens_details` is mandatory.** If the build does not expose it, the gate **fails** — do not fall
back to the aggregate counters, which cannot attribute a hit to a request. There is no intermediate value to land
on: every participating group's dense boundary is 3584 and the drafter is excluded from the `min`, so an observed
28672 means the §6.3 phase-2 window inference is **wrong** and §6.3 must be rewritten before shipping.

---

**L2 — CoW partial-tail durability under mutation and eviction pressure.** *(Codex #4, the critical untested path)*

The partial snapshot at `H = 31616` is initially attached to a **live 3584-token source block that keeps
mutating** as the producer generates. It is safe only if copy-on-write happens before that mutation or reuse, the
copy completes before any later request reads it, and the partial hash stays attached to the *copied* block.

Procedure:

1. request A: the L1 turn-1 prompt; snapshot registers at `H = 31616`;
2. **keep mutating A** — continue generating well past the snapshot (≥ 512 tokens) and issue a further turn on the
   same prefix, so the source block is written after the snapshot was taken;
3. **force pressure** — drive concurrent traffic until `vllm:gpu_cache_usage_perc` is at the working ceiling and
   `vllm:prefix_cache_evictions` (or the block-pool free count) shows real eviction;
4. request B: the identical 31616-token prefix, read the partial hit;
5. compare B against a **cold** B (cache reset, §6.2).

**Pass:** B's `cached_tokens == 31616`; B vs cold-B passes the full §6.2 logprob gate (identical chosen tokens at
every position, top-5 Δ within the noise band); zero `AssertionError` / refcount / `KVCacheBlock` errors in either
rank's log for the whole run.

**Cost receipt (record, do not gate):** the expected extra physical footprint is **one partial block for MLA plus
one for each of the four Mamba groups — 5 per snapshotting request** — plus the hash metadata and the CoW copy
bandwidth. Capture `vllm:gpu_cache_usage_perc` at steady state and re-run the `validate_apc_retention.py` check-3
pair ladder at 45K / 56K / 66K / 80K. **Pass:** the 80K pair still admits, i.e. the capacity floor is unchanged;
a lost rung is a blocking regression even if L1 and L4 are green.

---

**L3 — kpool tail validation at zero and nonzero uncached remainders.** *(Codex #2)*

The reasoning is that after a hit at `H` with `H % 4 == 0`, prefill writes `[H, N₂)` and the indexer tail holds
`r = (N₂ − H) mod 4` valid entries. The untested half is the claim that
`index_kpool_always_select_tail` does **not** mean "always read four entries even when the valid tail length is
zero".

Procedure: run the L1 rig four times with the turn-2 user suffix padded so that `(N₂ − 31616) mod 4` takes each of
`0, 1, 2, 3` — verify each `N₂` by `/tokenize`, do not assume the padding lands where intended. `r = 0` is the
decisive case.

**Pass:** all four pass the full §6.2 logprob gate against their own cold controls. Record the boot value of
`index_kpool_always_select_tail` alongside the results; if it is not `true` on this build, L3 proves nothing about
the configuration that ships and must be re-run against one where it is.

---

**L4 — temp-0 equivalence, off-grid vs a matched block-aligned control.** *(Codex #8)*

Two arms, identical method, run at **both** retention settings (dense and `14336`, R2):

* **off-grid arm** — the L1 rig, hit `31616`, `31616 % 3584 = 2944 ≠ 0`;
* **block-aligned control** — a matched rig with `P` chosen so `floor(P/64)*64 % 3584 == 0` (e.g. `P = 28700`
  → `B = 28672`), which the pre-patch build could also reach.

**Pass:** both arms pass the §6.2 gate — chosen tokens identical at every returned position, top-5
`max |Δlogprob| ≤ 3 × max(cold-vs-cold max, 0.02)` — against true cold runs from a reset cache. The control
failing is a rig bug; the off-grid arm failing while the control passes is the fine-grained path corrupting state
and is a hard stop.

---

**L5 — drafter eagle-peek at the hash-block drop unit.** *(Codex #3)*

With alignment 64 the drafter SWA group reports hits at 64 granularity, so its drop/verify unit becomes 64.
`patch_hybrid_prefix_hit.py` keeps it out of the `min`, so a drafter miss must allocate and zero a fresh window
rather than shorten the target's MLA/Mamba hit. That is the claim to prove, and acceptance alone does not prove it.

Procedure: exercise `drop_eagle_block` and `eagle_verified` explicitly at a 64-token boundary — one turn-2 prompt
whose hit lands exactly on a drafter-window-aligned 64 multiple, one at `64k + 32` off it — with DFlash2 enabled,
400-token generations. Capture per-arm: `vllm:spec_decode_num_accepted_tokens_total`,
`vllm:spec_decode_num_draft_tokens_total`, and the target logprobs.

**Pass:** (a) target logprob equivalence per L4 in **both** arms — this is the correctness half and it is
non-negotiable; (b) acceptance warm ≥ **0.85 ×** cold in both arms, missing counters = fail. A drop in (b) with
(a) green is a performance regression to record and weigh, not a correctness failure — the window is refilling.

---

**Rollback.** `GLM53_FINEGRAINED_APC=0` on both ranks restores the upstream all-managers veto and the 3584
alignment without unpatching or rebuilding; it is a restart, not a redeploy. The recipe plumbs it to both ranks in
`nccl_common`, and the value must be exactly `0` — the launcher rejects anything else before it stops the running
engine, and the coordinator rejects anything else at init (§4.3). Confirm the rollback the same way as B0: the
receipt must read `DISABLED`, `reason=GLM53_FINEGRAINED_APC=0 (kill switch)`, `effective alignment=3584 tokens`
on **both** ranks. Setting it also suppresses the `[glm53-apc-finegrained]` refusal described in §4.3 — which is
the point: it is the documented way to run a layout this patch will not vouch for.

---

## 7. Risks

* **R1 — `truncate_computed_blocks` asserts block alignment.** `kv_cache_manager.py:794`:
  `assert num_computed_tokens % manager.block_size == 0`. A 64-aligned hit violates it for the MLA manager
  (`64 % 3584 ≠ 0`). **Not reachable on this deployment**: the only caller is `sched/scheduler.py:917`, inside the
  branch guarded by `connector is not None and connector.supports_divergent_local_hybrid_hits`
  (`sched/scheduler.py:541`), and the serving command has no `--kv-transfer-config`. **Blocker for any future PD
  / NIXL deployment** — this is a genuine latent upstream bug that fine-grained hits expose. Flag it in the
  upstream PR. The precondition is now a **boot receipt**, not an assumption: the enable-path INFO line prints
  `KV-transfer connector: absent (no kv_transfer_config)` (§5.1). If that clause ever reads `PRESENT`, this
  deployment has entered the unvalidated path and §6.5 L1–L5 do not cover it.
* **R2 — interaction with `GLM53_APC_RETENTION_INTERVAL=14336` (FIX-PLAN Fix 1).**
  (a) `HybridKVCacheCoordinator.cache_blocks:724-725` stops flooring `num_computed_tokens` to 3584 once partial
  hits are on, so managers cache at their own granularity every step — more block-ids consumed, which is exactly
  the pool pressure Fix 1 is rationing. (b) `SingleTypeKVCacheManager.cache_blocks:463` passes
  `alignment_tokens=self.scheduler_block_size` (3584) to `reachable_block_mask`, **not**
  `_cache_hit_alignment_tokens`, so `MambaManager.reachable_block_mask:1537`
  (`aligned = boundary_tokens // alignment_tokens * alignment_tokens`) floors the pinned replay boundary to the
  3584 grid while hits can now land at 64. **This is a reuse-efficiency inconsistency, not a correctness one** —
  `reachable_block_mask` only decides which blocks get *hashed*; a missing hash is a missed hit, never wrong
  state. Left unpatched deliberately (it would mean editing a second, currently-pristine file). Run §6.2 at both
  dense and 14336 retention and record the delta.
* **R3 — the `_cache_partial_tail_block` exact-equality guard depends on the scheduler stop.** If
  `_mamba_block_aligned_split` ever fails to land a chunk end exactly on `tail_boundary` (e.g. a budget clip in
  `patch_scheduler_decode_floor.py`'s late-cap path interacting with `stops`), mamba registers no partial entry
  and the reconciled hit silently falls back to the dense `⌊C/3584⌋·3584`. Degradation, not corruption. Detect via the hit counts in §6.2.
* **R4 — drafter acceptance.** With alignment 64 the drafter SWA group now reports hits at 64 granularity instead
  of 3584. `patch_hybrid_prefix_hit.py` (`kv_cache_coordinator.py:856-866`) keeps it out of the `min` and leaves
  its blocks empty when the window does not cover the hit, so the *target* distribution is unaffected by
  construction; **acceptance is what can move**. That is why the DFlash acceptance ≥ 0.85 × cold check in §6.2 is
  mandatory, not optional.
* **R5 — patch composition.** Covered by host test A4 (commutes byte-for-byte with `patch_hybrid_prefix_hit.py`).
  Both patches insert a helper before the same needle; each is guarded by a `def _glm53_…(` name check.
* **R6 — this train ran read-only.** Nothing here has been applied or measured live. Every number in §6.3 is a
  prediction derived from source reading.

---

## 8. Upstreamability

**Yes — the veto is a generic bug, and the cleanest evidence is internal inconsistency inside one constructor.**

The fork/upstream already encodes the rule "a group with `participates_in_prefix_caching == False` is invisible to
the prefix-caching machinery" in three places, including one 26 lines *above* the veto and one 25 lines *below*
it:

* `kv_cache_coordinator.py:589-598` — excluded from the `block_size % hash_block_size` assert, with the comment
  *"Only groups that participate in prefix caching must satisfy the divisibility constraint; groups that opt out
  (e.g. GLM5Next's kpool tail, block_size=kpool) are scratch buffers and excluded."*
* `kv_cache_coordinator.py:664-665` — excluded from `attention_groups`, with the comment *"their blocks are
  per-request scratch, never shareable, so they must not participate in hit lookup (their manager-level hooks
  already no-op)"*.
* `kv_cache_utils.py:668-680` — excluded from the hash-granularity GCD, naming KpoolTailSpec and kpool=4.

Only `kv_cache_coordinator.py:627-632` scans all managers. A manager that `verify_and_split_kv_cache_groups`
refuses to put in `attention_groups` cannot have its `find_longest_cache_hit` called by
`HybridKVCacheCoordinator.find_longest_cache_hit` (`:801-850` iterates `self.attention_groups`), so its
`supports_fine_grained_hash_lookup` value is unobservable — yet it silently degrades every other group's hit
granularity by `scheduler_block_size / hash_block_size` (here 56×).

**Upstream has since landed it.** vLLM main `e126687a` scopes this veto to `KVCacheSpec.prefix_cacheable` groups
and expresses the scratch invariant as the `tokens_per_state` check in `resolve_kv_cache_block_sizes` — the same
two commits sketched below, by upstream's own route. The overlay therefore detects an already-scoped coordinator
and no-ops on it (§5.1, host test A6) rather than failing on a missing anchor, and should be retired when the
recipe image rebases onto a vLLM that carries that commit (§9.4). What follows is the PR that was going to be
proposed, kept because it is the argument for why upstream's change is right.

**Proposed upstream PR** (two commits, both small):

1. *fix(v1/core): scope the fine-grained-hit veto to prefix-caching participants.* Body = §4.1's four-site table.
   Repro without GLM: any hybrid config with a spec whose `participates_in_prefix_caching` is `False` and whose
   `block_size != hash_block_size`.
2. *feat(v1/core): require scratch-group alignment before enabling partial hash hits.* Adds the
   `hash_block_size % block_size == 0` requirement with the kpool rationale (§2.4). This is the piece that is
   arguably *not* purely a bug fix: it hardens an invariant that today holds only numerically.

Two further items to raise in the PR discussion, both discovered here and both out of scope for the patch:

* `kv_cache_manager.truncate_computed_blocks:794` asserts `num_computed_tokens % manager.block_size == 0`, which
  fine-grained hits violate on the connector path (R1). Either relax it to the hash alignment or gate the
  connector path on `not enable_partial_hash_hits`.
* `SingleTypeKVCacheManager.cache_blocks:463` passes `scheduler_block_size` where the coordinator's
  `_cache_hit_alignment_tokens` is the semantically correct value (R2b).

The `896` alignment proposed in `FIX-PLAN-CACHE-2026-08-31.md` §Fix 3 should be **withdrawn**, not upstreamed:
§2.3 shows it would break the fine-grained lookup's positional indexing.

---

## 9. Open items before this can ship

1. **Confirm `scheduler_block_size` / `hash_block_size` directly.** Neither is printed by this build (§1.2);
   both are derived from the logged per-group block sizes. Cheap and worth doing while patching: extend the
   `patch_hybrid_prefix_hit.py` group log, or the new `overlay/patch_apc_fine_grained_hits.py` INFO line, to print both. Everything
   in §2 and §6 hinges on `hash_block_size == 64`.
2. **Confirm the §6.3 phase-2 window inference** with the live hit counts from the §6.2 rig before quoting any
   speed-up to anyone.
3. ~~Wire the patch into the recipe.~~ **Done.** `overlay/patch_apc_fine_grained_hits.py` and
   `tests/test_apc_fine_grained_hits.py` are `COPY`'d and the patch is `RUN` in the `Dockerfile`; `start.sh`
   preflights it, `scp`s it to the worker, mounts it on both ranks, and runs it in both in-container patch
   stanzas. The host test is **invoked** at build time (immediately before the patch it validates, §5.2), not
   merely copied. The wiring is now asserted statically too (Part C): the knob is validated on `start|restart`,
   exported by value inside `nccl_common` so **both** ranks receive it, and a caller-supplied value is captured
   before `.env` and restored after. Still outstanding, and only a live boot can settle it: that the value the
   engine actually honours matches — i.e. the §6.5 B0 receipt read from **both** ranks' `docker logs`. The kill
   switch is only a rollback if the worker sees it too.
4. **Watch upstream.** vLLM main `e126687a` scopes this veto to `KVCacheSpec.prefix_cacheable` groups and puts
   the scratch invariant in `resolve_kv_cache_block_sizes`'s `tokens_per_state` check — i.e. upstream has landed
   the same fix, by its own route. The patcher already treats such a coordinator as a no-op (§5.1, host test A6),
   so a rebased image boots rather than failing on a missing anchor, but the overlay is then **inert**: the
   `[glm53-apc-finegrained]` receipt will not appear, `GLM53_FINEGRAINED_APC` will have no effect, and the
   scratch invariant in force will be upstream's `tokens_per_state`, not ours. When the recipe rebases onto a
   vLLM that carries `e126687a`, re-read §4.3 against upstream's rule, re-run the §6.5 receipts on that build,
   and retire this overlay rather than carrying a dead one — §8 (upstreamability) is settled by that commit.
5. Sequence against the FIX-PLAN order: this is Fix 3, currently scheduled after Fix 1 (shipped), Fix 2, Fix 4,
   Fix 5. It should be A/B'd on the Fix 1 baseline with retention both dense and 14336 (R2).
6. Nothing in this train has been applied to the live boxes. The overlay is in the recipe directory and the host
   test is green (195 checks against the live coordinator source), but **no receipt in §6.5 has been collected** —
   L1 through L5 are all outstanding, and B0 has never been observed on a real boot.
7. Decide the §4.3 refusal policy is what you want *before* the first deploy, not during one. On a layout that
   violates the kpool invariant **and would otherwise have fine-grained hits enabled**, the engine now **fails to
   start** rather than quietly serving at 3584 alignment. That is the intended fail-closed behaviour and
   `GLM53_FINEGRAINED_APC=0` is the escape hatch, but it converts a silent performance regression into a boot
   failure — make sure whoever is on call knows the message `[glm53-apc-finegrained]` and the one-env-var remedy.
   Note the deliberate carve-out (§4.3 row 4): if a *participating* manager already disables fine hits, a bad
   scratch group does **not** raise, because that configuration is exactly what upstream would run. The fault is
   reported in the receipt instead — which means a later change that removes the participating blocker can turn a
   logged warning into a boot failure. That transition is intentional, and the receipt is how it is seen coming.
