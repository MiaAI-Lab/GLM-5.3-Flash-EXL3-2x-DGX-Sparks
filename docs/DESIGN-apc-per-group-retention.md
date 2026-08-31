# DESIGN — per-group prefix-cache retention: stop the DFlash2 drafter from evicting the KV cache

Train: root-cause code fix for prefix-cache eviction by the DFlash2 drafter's cache blocks, and per-group retention.
Status: **design + unapplied overlay patch + host test + live test plan.** Nothing was written into the container;
all source facts below are read-only reads of the live fork.
Revision 2 (2026-08-31): incorporates the adversarial review in `research-2026-08-31/CODEX-drafter-retention.md`
— one explicit capacity formula (§4.1), unconditional SWA env validation and a min-exemption-derived,
fail-closed override (§5.2), the resolved-vector boot log as the deployment acceptance criterion (§7, L1),
overlay-composition coverage (§8.1.7), and the divergence-suffix / acceptance-rate gate L5a (§8.2).

Source paths (in the container, `glm53-exl3-head`, `/usr/local/lib/python3.12/dist-packages/vllm/`):

| tag | file |
|---|---|
| `C` | `v1/core/kv_cache_coordinator.py` |
| `S` | `v1/core/single_type_kv_cache_manager.py` |
| `B` | `v1/core/block_pool.py` |
| `M` | `v1/core/kv_cache_manager.py` |
| `U` | `v1/core/kv_cache_utils.py` |
| `I` | `v1/kv_cache_interface.py` |

Every claim below is either a `file:line` citation or explicitly marked **[inference]**.

---

## 1. The live group layout

Boot log (`deb8:/home/blockbrain/glm-restart-sweep-28672.log`, 08-31 13:16:09, emitted by `C:709`):

```
hybrid APC groups: [('MLAAttentionSpec', [0], 'FullAttentionManager', False),
                    ('MambaSpec', [2, 3, 4, 5], 'MambaManager', False),
                    ('SlidingWindowSpec', [6], 'SlidingWindowManager', True)];
eagle_group_ids=[6]
```

Seven KV-cache groups, one `BlockPool` with **globally unique block ids** (`U:941-946`, `B:174-181`):

| gid | spec | manager | block_size | in `attention_groups`? | eagle |
|---|---|---|---|---|---|
| 0 | `UniformTypeKVCacheSpecs` of 11 `MLAAttentionSpec` (`compress_ratio==1`) + 11 indexer (`compress_ratio>1`) | `FullAttentionManager` | 3584 | yes | no |
| 1 | `KpoolTailSpec` | `KpoolTailManager` | kpool | **no** — `participates_in_prefix_caching=False` (`I:775-786`), skipped at `C:664-665` | – |
| 2–5 | `MambaSpec`, `mamba_cache_mode="align"`, padded to `mla_page` | `MambaManager` | 3584 | yes | no |
| 6 | `SlidingWindowSpec` **exact type**, window 2048, `block_size=64`, `page_size_padded=mla_page` (`U:1533-1552`) | `SlidingWindowManager` | 64 | yes | **yes** |

The drafter group is created by `_get_kv_cache_groups_glm5_next` on the **PADDED SLOT-SHARE** branch
(`U:1519-1552`, boot line `U:1549`: `padded slot-share block=64 mla_page=2351104 … draft_bytes/token=2048`).
Because `page_size_bytes` returns the *padded* page, `draft_shared = draft_page == mla_page` is true
(`U:1777-1780`), so the drafter adds **zero bytes** per pool block but **one full-MLA-page-priced block id**
per 64 drafter tokens. That asymmetry is the whole bug.

`eagle_group_ids=[6]` and the drafter's exemption from the hybrid `min()` both come from Mia's overlay
(`C:121-133`, `C:856-868` — `overlay/patch_hybrid_prefix_hit.py`).

---

## 2. Exact per-group id cost of one cached 3584-token segment

`scheduler_block_size = 3584`; fine-grained hash hits are **off** (`C:635` boot warning: "Disabling
fine-grained prefix-cache hits because these KV cache managers require block-aligned lookups:
KpoolTailManager"), so `_cache_hit_alignment_tokens == scheduler_block_size == 3584` (`C:643-651`).

Which blocks get hashed is decided by `reachable_block_mask` → `block_mask` → `BlockPool.cache_full_blocks`,
which skips masked-out and null blocks (`S:429-476`, `B:275-277`).

### 2.1 MLA group (gid 0)

`FullAttentionManager` does not override `reachable_block_mask`; the base returns `None` = "cache every
non-null block" (`S:482-500`). Block size 3584 ⇒ **1 id per 3584-token segment, always, retention has no
effect** (this is deliberate: full attention must keep the fine hit granularity, `C:52-56`).

### 2.2 Mamba groups (gid 2–5)

`MambaManager.reachable_block_mask` (`S:1487-1542`):
* `retention_interval is None` → early `return None` (`S:1509-1511`) ⇒ dense ⇒ **1 id per group per segment = 4 ids**.
* `retention_interval == R > 0` → `per_segment = R // 3584`, one state kept per segment (`S:1520-1533`)
  ⇒ **4 · 3584/R ids per 3584 tokens**.
* Plus one state per group at each `reachable_boundary` (`S:1534-1541`).

### 2.3 Drafter SWA group (gid 6) — the flood

`SlidingWindowManager.reachable_block_mask` (`S:998-1057`):

```
need  = _contiguous_blocks_for_hit(2048, 64, use_eagle=True)      # S:886-895
      = cdiv(2048-1, 64) + 1 = 32 + 1 = 33
shift = 1                                                          # S:1024 (use_eagle)
segment_tokens = alignment_tokens = 3584   when retention_interval is None   # S:1032-1037
per_segment    = 3584 // 64 = 56                                   # S:1039
mask[i] = (i >= 1) and ((i - 1) % 56) >= 56 - 33 == 23             # S:1044-1046
```

⇒ **33 of every 56 drafter blocks per 3584-token segment are hashed.** With `retention_interval = R`:
`per_segment = R // 64`, still `need = 33` ⇒ **33 ids per R tokens = 33 · 3584/R per segment**, plus a
33-block tail at every `reachable_boundary` (`S:1048-1056`).

All 33 masked blocks really are live (not yet nulled) when `cache_blocks` runs. Order inside
`KVCacheManager.allocate_slots` is `remove_skipped_blocks` → allocate → `cache_blocks` (`M:495-508`, `M:564`);
`remove_skipped_blocks` frees below `processed - window + 1` (`S:663-675`, `S:624-661`) while the mask covers
tokens `[B-2048, B+64)` for boundary `B` — i.e. exactly the surviving window. **[inference, from the two
formulas; not separately instrumented.]**

### 2.4 Kpool tail (gid 1)

`KpoolTailManager.cache_blocks` is a hard `return` (`S:1135-1142`) and `remove_skipped_blocks`/`find_longest_cache_hit`
are no-ops (`S:1147-1160`, `S:1119-1133`) ⇒ **0 ids cached, 1 live block per request** (`I:770-777`).

### 2.5 Summary table

| group | ids per cached 3584-token segment (dense) | at retention `R` |
|---|---:|---|
| 0 MLA (+indexer) | **1** | 1 (unchanged — `reachable_block_mask` returns `None`) |
| 1 kpool tail | 0 | 0 |
| 2–5 mamba ×4 | **4** | 4·3584/R |
| 6 drafter SWA | **33** | 33·3584/R |
| **total** | **38** | 1 + 37·3584/R |

**Dense: one block id burned per 94 tokens of conversation, 87 % of them by a drafter group that is
exempted from the hybrid hit and therefore contributes nothing to hit length.**

At the current default `R = 14336`: 1 + 1 + 8.25 = **10.25 ids / 3584 tokens**.
At `R = 7168`: 1 + 2 + 16.5 = **19.5**. At `R = 28672`: 1 + 0.5 + 4.125 = **5.625**.

Per *turn* (per `reachable_boundary`, `S:1048-1056` / `S:1534-1541`) a group with a
**non-`None`** retention interval also pins a tail: 33 drafter + 4 mamba = **37 ids**.
A dense group (`retention is None`) has no separate boundary tail — its mask is `None`,
so those blocks are already counted in the per-segment number. In the proposed mode
mamba/MLA are dense and only the drafter has a boundary tail, so the per-turn cost is
**33**, not 37.

**Do boundary tails overlap the periodically retained blocks?** They can: when a
`reachable_boundary` happens to land on a multiple of `R`, the pinned tail *is* the
periodic tail and the two counts are the same blocks. The model below adds them, so it
over-counts `C` in that case (a *conservative* error) and under-counts nowhere. It is
not modelled per-boundary because the boundary positions depend on
`request.num_prompt_tokens`, which varies per turn. **[inference — not instrumented.]**

---

## 3. Pool size

`num_blocks = available_memory // per_block` where `per_block = 11·mla_page + 11·idx_page`
(`U:1789-1794`, mirrored by `U:982-1004`); the drafter is slot-shared so it adds no bytes
(`U:1777-1780`, `U:996-1002`). `mla_page = 2,351,104 = 3584 × 656` — the `fp8_ds_mla` V3.2 layout
(`I:443-445`). Boot: `Available KV cache memory: 16.75 GiB` (this evening's boots range 16.4–17.09 GiB).

The pool size is not logged. Derive it from the logged capacity line
(`GPU KV cache size: 1,553,140 tokens, Maximum concurrency for 1,000,000 tokens per request: 1.55x`,
emitted at `U:2297`, value = `int(max_concurrency × max_model_len)`, `U:2260-2266`), with
`max_concurrency = num_blocks / num_blocks_per_request` and both terms integers (`U:937-956`):

```
max_concurrency ∈ [1.553140, 1.553141)
physical bound:  num_blocks ≤ 17.99e9 / (11 × 2,351,104) = 695   (idx_page > 0)
unique integer solution in range:  num_blocks = 643,  num_blocks_per_request = 414
```

**Pool = 643 block ids; 642 usable** (id 0 is the null block, `B:190-191`; `get_usage` divides by
`num_gpu_blocks - 1`, `B:808-819`). Implied `idx_page ≈ 191 kB ≈ 53 B/token/layer. **[inference —
the derivation is exact given 11 MLA layers, but `idx_page` was not read from the model config.]**

`num_blocks_per_request = 414` decomposes as 280 (`cdiv(1e6, 3584)` MLA) + 1 (kpool tail, `I:770-777`)
+ 4 mamba groups (`I:812-821`, align ⇒ `2 + num_speculative_blocks` pages each) + drafter
(`I:617-647`, `cdiv(min(2047 + max_in_flight_tokens, max_model_len), 64) + 1`). **[inference — the exact
split depends on `max_in_flight_tokens`, which is not logged; the total 414 is the derived quantity.]**

---

## 4. Reconciling the measured thresholds (14 segments OK / 17 fail, dense)

The failure is not "too many ids" alone — it is **LRU position**. `BlockPool.free_blocks` (`B:719-743`):

```python
if block.block_hash is None or not self.enable_caching:
    blocks_to_evict_first.append(block)   # -> prepend_n : LIFO front, reused immediately
else:
    blocks_to_evict_last.append(block)    # -> append_n  : LRU tail, evicted last
```

and `get_new_blocks` pops from the **front** (`B:647-677`), evicting whatever hash the popped block holds
(`B:679-700`).

Timeline for a conversation A of `S` segments:

* Its drafter blocks are freed **progressively during prefill** by
  `SlidingWindowManager.remove_skipped_blocks` → `_remove_blocks_in_range` → `free_blocks`
  (`S:624-661`, `S:597-622`). The 33 masked ones are **hashed** ⇒ appended to the LRU tail; the other 23
  per segment are unhashed ⇒ prepended to the front and immediately recycled.
* Its MLA and mamba blocks are freed only when the request completes, `KVCacheManager.free` →
  `coordinator.free` → per-manager `free` in group order (`S:521-530`), i.e. **behind** all of A's
  mid-prefill drafter frees in the queue.

So the queue after A finishes is, front → back:
`[A drafter, hashed, oldest first] … [A MLA] [A mamba] [A drafter window]`.

Conversation B then prefills. Per segment B makes 38 "deep" pops (its own 23 unhashed drafter blocks
recycle from the front for free). A's MLA/mamba survive iff

```
free_at_B_start  +  A's_drafter_cache   ≥   B's_deep_pops
(642 − 38·S)     +      33·S            ≥       38·S
                       642 ≥ 43·S   ⇒   S ≤ 14.9 segments per conversation
```

**Measured (runbook A6, clean server, dense): 14 segments OK, 17 fail; 45K (12 seg) 99 %, 56K (15 seg) 0 %,
66K (18 seg) 0 %.** The predicted knee sits between 14 and 15.

The measured knee is **consistent with** `num_blocks = 643`, not a proof of it. The two were computed
from disjoint evidence, which is worth something — but `642 ≥ 43·S` only pins the pool to the range
that puts the knee in `[14, 15)`, i.e. `602 ≤ pool_usable < 645`. Any pool in that band reproduces the
observation. The `num_blocks = 643` value comes from §3's `max_concurrency` derivation alone.

### 4.1 The capacity formula (one formula, all rows)

Let a conversation of `S` cached 3584-token segments and `t` turn boundaries cache

```
C = c·S + b·t          ids in total
D = d·S + b_d·t        of which the drafter's share
```

`c`/`d` are the per-segment costs of §2.5, `b`/`b_d` the per-turn boundary tails (0 for a dense group).
The drafter's ids are freed **cached, mid-prefill**, so by the time the next conversation B prefills they
are already parked on the protected LRU tail; A's MLA/mamba blocks are freed later and sit ahead of them
(the timeline above). A survives B iff the free pool plus A's drafter cache covers B's deep pops:

```
(P − C) + D  ≥  C            ⇔        P ≥ 2C − D

                    P − (2b − b_d)·t
        ⇒   S  ≤   ───────────────────           P = 642 usable ids
                        2c − d
```

With `t = 3` turns (`floor` of the bound; the table's `2b − b_d` is 0 dense, 41 for any `R > 0`
(`2·37 − 33`), 33 for the proposed mode (`2·33 − 33`)):

| config | `c` | `d` | `b` | `b_d` | bound (t=3) | tokens | measured |
|---|---:|---:|---:|---:|---|---:|---|
| dense | 38 | 33 | 0 | 0 | `642 / 43` ⇒ **S ≤ 14** | 50K | 14 OK / 17 fail ✓ |
| R=7168 | 19.5 | 16.5 | 37 | 33 | `(642−123) / 22.5` ⇒ **S ≤ 23** | 82K | 56K 98 %, **80K (23 seg) 0 %** ✗ |
| R=14336 | 10.25 | 8.25 | 37 | 33 | `(642−123) / 12.25` ⇒ **S ≤ 42** | 150K | 45/56/66/80K = 99/98/97/96 % ✓ |
| R=28672 | 5.625 | 4.125 | 37 | 33 | `(642−123) / 7.125` ⇒ **S ≤ 72** | 258K | 56K 98 %, 80K 96 % ✓ (no gain over 14336, as predicted) |
| **proposed**: MLA+mamba dense, drafter boundary-only | **5** | 0 | 33 | 33 | `(642−99) / 10` ⇒ **S ≤ 54** | **193.5K** | to be measured |

(These five numbers are asserted by `tests/test_apc_per_group_retention.py::test_id_cost`, so the table
and the test cannot drift apart.)

**The `R=7168` row is a miss, and a material one.** An 80K pair needs `cdiv(80000, 3584) = 23` segments
per conversation — exactly the bound, i.e. zero headroom — and it measured **0 %**, not "marginal". The
model is therefore optimistic by at least one segment at that operating point. The terms it omits all
push the same way and are not quantified here:

* the admission watermark and `scheduler_reserve_full_isl` headroom (blocks B must reserve up front, not
  merely pop as it goes);
* the **in-flight** request's live, unfreeable blocks — at 80K both conversations are large enough that
  A's own live window overlaps B's prefill;
* partial-block tails at the end of each segment;
* `t` is a floor: an 80K conversation over three turns has boundary tails at each turn *plus* the replay
  boundary.

Against that, the boundary/periodic overlap noted in §2.5 pushes the other way. The net is measured, not
predicted. Treat the formula as an ordering device — it predicts the sign and the rank of every measured
row, and it is the reason the proposed mode is expected to beat `R=14336` — **not** as a capacity
guarantee. The live ladder (§8.2 L2) is what decides.

### The one-line root cause

> A sliding-window KV group whose window (2048) is smaller than the prefix-cache hit granularity (3584)
> hashes `cdiv(window−1, block)+1 = 33` block ids at **every** hit boundary, and those ids are freed
> **cached** — onto the *most protected* end of a global LRU that is shared with the groups that actually
> carry the hit. 87 % of the prefix cache is spent on a group whose hit is discarded by `min()` anyway.

---

## 5. Design A (recommended): per-group retention interval

### 5.1 What changes

`retention_interval` is currently a **single scalar** on the coordinator, read once from the env
(`C:151-157`) and handed identically to every manager (`C:302-317` base, `C:723-755` hybrid). Make it a
per-group tuple.

```
C:__init__          self.retention_interval          (kept: global default, validation, logging)
              +     self.retention_interval_by_group : tuple[int|None, ...]
C:cache_blocks (base)     manager.cache_blocks(..., retention_interval=self.retention_interval_by_group[i])
C:cache_blocks (hybrid)   same, inside the existing per-manager loop
```

Nothing below the coordinator changes: `SingleTypeKVCacheManager.cache_blocks` already takes
`retention_interval` per call (`S:429-434`) and every `reachable_block_mask` already honours
`None` / `0` / `>0` (`S:482-500`, `S:998-1057`, `S:1487-1542`).

### 5.2 The rule

Two tiers.

**Tier 1 — the GLM-5.3 knob that ships now.** New env `VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA`
(launcher knob `GLM53_APC_RETENTION_INTERVAL_SWA`). Default **`0` = "keep only the reachable
boundaries"**, which under `S:1048-1056` means the 33-block tail ending at the request's replay boundary
(`request.num_prompt_tokens - 1`) and at any `shared_prefix_boundary` — and nothing else. Every other
group keeps the global value, whose **new recommended default is unset (dense)**.

Two guards, both fail-closed at boot:

1. **The raw value is validated unconditionally**, before any group is inspected:
   integer, non-negative, a multiple of `scheduler_block_size` (so a retained tail lands on a real hit
   boundary), and `≤ 1,000,000`. A typo is a boot failure even on a model with no sliding-window group —
   it is never silently ignored.
2. **It is applied only to a group the coordinator itself has singled out as the EAGLE-exempt drafter**,
   never to "anything whose class is called `SlidingWindowSpec`". `_glm53_min_exempt_group_ids` derives
   that set from coordinator state: group `i` qualifies iff its inner spec is an *exact*
   `SlidingWindowSpec` (never `KpoolTailSpec`, which subclasses it) **and** `eagle_group_ids` is
   *exactly* the set of such groups. The second clause is the state `patch_hybrid_prefix_hit.py`
   establishes (`C:121-133`): it narrows `eagle_group_ids` from the upstream all-groups fallback to the
   drafter SWA groups, and — off the same discriminator — makes those groups skip the hybrid hit `min()`
   (`C:856-868`). Under the undiscriminating all-groups fallback the equality fails, the exempt set is
   empty, and setting the variable **raises at boot** rather than sparsifying a group that still shortens
   every hit. The automatic rule (Tier 2) stays safe and inert in that case.

   *Limit of the inference:* the equality is a proxy for "this group is out of the hit `min()`", not a
   reading of `find_longest_cache_hit`. If some future upstream annotates `is_eagle_group` on a genuine
   SWA group **and** keeps it inside the `min()`, the proxy would let the override through. That is the
   one configuration where the knob must not be set; the automatic rule's extra `window < alignment`
   condition still covers it, which is why the *default* is auto and not `0`.

**Tier 2 — the generic default (also the upstream proposal).** When no explicit SWA value is given,
derive one per group:

> A sliding-window group is *hit-inert* when `sliding_window < alignment_tokens` (its cached window can
> only ever be consulted at an alignment boundary) **and** it is EAGLE/min-exempt (its hit length is
> discarded by the coordinator). Such a group pays `need = cdiv(window−1, block) + 1` ids at every
> boundary — always ≥ 2 for an EAGLE group — and buys nothing in hit length, so cache it at the
> reachable boundaries only (`retention_interval = 0`). Every other group keeps the global value.

For this kit: `2048 < 3584` and gid 6 is EAGLE-flagged ⇒ retention 0 (`need = 33`). For a genuine SWA
model whose window spans the alignment, or an SWA group that is still inside the hit `min()`, the rule
is inert and behaviour is unchanged.

### 5.3 Why the drafter can lose its window at a boundary

The hit is already reconciled without the drafter today; this design only makes it the common case.

1. `find_longest_cache_hit` (`C:856-868`, Mia's patch): if the drafter's own hit is shorter than the
   candidate, the branch `continue`s **before** `curr_hit_length = _new_hit_length` and before
   `longest_hit_length = max(...)`, so the drafter neither shortens the hybrid hit nor inflates
   `num_uncached_common_prefix_tokens` (`C:886-893`). Its `hit_blocks_by_group[6]` stays `None` and is
   emitted as `[]` (`C:894-896`).
2. Admission: `get_num_blocks_to_allocate` with `new_computed_blocks=[]` computes
   `num_skipped_blocks` from `get_num_skipped_tokens` (`S:196-224`, `S:1059-1085`) so the drafter reserves
   only a fresh ~33-block window, not the whole prefix.
3. Allocation: `add_local_computed_blocks` pads `req_to_blocks` with `num_skipped_blocks` null blocks and
   sets `num_cached_block = len(req_blocks)` (`S:270-286`); `allocate_new_blocks` then pulls exactly the
   window (`S:332-364`). Newly allocated drafter blocks are recorded for worker-side zeroing because the
   spec is an `AttentionSpec` (`S:90-93`), so the fresh window is zeroed, not stale.
4. Cost: **correctness is unaffected** (draft tokens are verified by the target). The cost is a draft
   **acceptance dip** while the 2048-token window refills. That refill happens during the prefill of the
   *uncached remainder*, so it is free whenever the remainder ≥ 2048 tokens — which is exactly the
   diverging-prefix / new-turn case. The one case where the remainder is short is a conversation's own
   follow-up — and that is precisely the case the pinned replay-boundary tail (retention `0`, not "never
   cache") keeps warm. The design is self-consistent on this point; the live plan measures it anyway (§8).

### 5.4 Predicted effect

| | dense | R=14336 everywhere (today's default) | **per-group (MLA+mamba dense, drafter boundary-only)** |
|---|---|---|---|
| ids per 3584 tokens | 38 | 10.25 | **5** (+33 per turn) |
| pair capacity (S ≤, §4.1, t=3) | 14 seg (50K) | 42 seg (150K) | **54 seg (193.5K)** |
| hit grid for a *diverging* prefix | 3584 | **14336** | **3584** |
| own-conversation re-turn | 99 % | 99 % | 99 % (replay boundary pinned) |
| subagent divergence hit | n/a (evicts) | 45 % | **≈ 65–70 %** (the 7168 run already showed 67.6 % on a 21504 boundary) |

The point: today's fix buys capacity by coarsening *everybody's* grid. Per-group retention buys the same
capacity by coarsening only the group whose hit is thrown away — so capacity **and** the fine grid.

### 5.5 Risks

| risk | assessment |
|---|---|
| Draft acceptance dip on a diverging warm hit | Bounded, self-limiting (§5.3.4). Already happens today for any hit off the 14336 grid. Measured by the decode-rate control in §8. |
| `SlidingWindowManager.find_longest_cache_hit` scan cost rises | It scans right→left over `max_length // 64` blocks and early-stops on a run of 33 (`S:944-970`; the `TODO` at `S:938-943` acknowledges the O(n) scan). With few cached drafter blocks a miss walks the full range: ~1250 dict lookups at an 80K candidate, ~3100 at 200K, once per admission. Sub-millisecond; note it, do not block on it. |
| Mamba dense again ⇒ 4 ids/segment instead of 1 | Included in the §5.4 arithmetic (the 5). Still 7.6× cheaper than dense-with-drafter. |
| A group getting `0` when it *is* in the `min()` | Guarded on **both** paths (Codex #4): explicit and automatic alike require min-exemption derived from `eagle_group_ids` (§5.2), and the automatic path additionally requires `window < alignment`. Mamba and MLA must never receive `0` — a missing mamba state is a correctness hole (vLLM #47491/#43090, quoted in `overlay/patch_hybrid_prefix_hit.py`). |
| Env/validation drift | `_validate_prefix_cache_retention_interval` (`C:46-73`) must run per group; the patch adds a per-group validator over the *resolved* vector **and** validates the raw SWA value unconditionally (non-negative, scheduler-block multiple, ≤ 1,000,000). Both fail closed at boot. |
| Stale global knob (Codex #6) | The launcher may still be exporting `GLM53_APC_RETENTION_INTERVAL=14336`, in which case the fine hit grid is *not* restored and the win is only capacity. This is why the acceptance criterion is the **resolved vector**, logged at init (§7), not the env var: the head must show `retention_by_group=[None,None,None,None,None,None,0]`. |
| Drafter reads zeroed KV after a sparse miss (Codex #2) | Freshly allocated drafter blocks are zeroed worker-side (`S:90-93`), so they never carry another request's data — but "zeroed" is not *valid* KV for the preceding prompt tokens either. Target verification hides this in the output; it does not make it safe. Gated by the divergence-suffix + acceptance-rate probes in §8.2 (L5a/L5b), not by the equivalence gate. |
| Worker rank | The knob must reach both ranks like the existing one (`start.sh:1136-1140`). |

---

## 6. Design B (alternative, **not recommended**): free out-of-window drafter blocks *uncached*

Mechanically: in `SlidingWindowManager._remove_blocks_in_range` (`S:597-622`), call
`block_pool._maybe_evict_cached_block(blk)` (`B:679-700`) before `free_blocks`, so `block_hash is None`
and `B:734-736` prepends them to the LIFO front instead of the LRU tail — except blocks in a
boundary tail.

It would work on capacity (a recycled block costs zero deep pops), but:

1. **It is destructive on shared hashes.** `_maybe_evict_cached_block` removes the hash from
   `cached_block_hash_to_block` *globally* (`B:571-590`). Blocks are shared objects: conversation A caches
   a block, conversation B hits it and `touch`es it (`S:296`, `B:702-717`). When B's window later slides
   past that block, B would delete **A's** cache entry. Design A cannot do this — a block that was never
   hashed is freed unhashed by the existing code path, with no global side effect.
2. **No simplicity win.** It still has to exempt the boundary tails, which means recomputing
   `reachable_boundaries` inside `remove_skipped_blocks` — the same information `reachable_block_mask`
   already has at cache time, one call site earlier.
3. It still pays the hash insert + remove churn on 33 blocks per segment.
4. **Not upstreamable as a default.** For a real SWA model, an out-of-window block *is* legitimately
   reusable by another request whose prefix ends there.

Keep it documented as a fallback only if the per-group threading turns out to be infeasible.

**Codex #3:** reason (1) is argued from `B:571-590` / `B:702-717`, it is **not tested** — no shared-hash
regression exists, because Design B is not implemented. If Design B is ever revived, that test is a
prerequisite, not a follow-up: construct two requests sharing a hashed block, let one slide its window
past it, and assert the other's `cached_block_hash_to_block` entry survives. Until then, uncached frees
stay off the table without ownership/reference tracking.

---

## 7. The overlay patch

`overlay/patch_apc_per_group_retention.py`, recipe-overlay style: fail-closed, idempotent,
MARK/anchor, matching `overlay/patch_hybrid_prefix_hit.py`.

* Target: `C` (`GLM53_KV_COORDINATOR_PY`, same env var as the hybrid patch).
* MARK: `# [glm53-apc-per-group]`; re-run is a no-op.
* Inserts `_glm53_swa_retention_env()`, `_glm53_min_exempt_group_ids()`, `_glm53_retention_for_group()`,
  `_glm53_resolve_retention_by_group()`, `_glm53_validate_retention_intervals()` and
  `_glm53_format_retention_vector()` (self-contained, no new imports beyond `os`, which the module does
  not yet import — the patch adds it under the MARK).
* Logs the **resolved** vector at coordinator init, one greppable line:
  `[glm53-apc-per-group] retention_by_group=[None,None,None,None,None,None,0] (global=None swa_env=0
  eagle_min_exempt=[6])`. This is the deployment acceptance criterion (§8.2 L1) and is checkable from
  `docker logs <container> 2>&1 | grep retention_by_group` on **both** ranks.
* Anchors (each must appear **exactly once**, else `SystemExit`):
  1. `self.retention_interval = envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL` + its validator call (`C:154-157`)
  2. the base `cache_blocks` loop (`C:313-317`)
  3. the hybrid `cache_blocks` loop header (`C:737-738`) and its `manager.cache_blocks(...)` call (`C:750-754`)
* Depends on `_glm53_is_draft_swa_spec` / `_glm53_inner_kv_spec` (`C:33-43`); inserts them if the hybrid
  patch has not run, so ordering between the two overlays does not matter. Asserted, not assumed: the
  host test applies both overlays to a **pristine** copy in both orders and checks that they succeed, are
  idempotent under re-application in any order, and produce **byte-identical** files (§8.1).
* Launcher wiring (`start.sh:1080-1085`, already on this branch):
  `GLM53_APC_RETENTION_INTERVAL_SWA` → `VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA`, accepted
  values `""` (auto/Tier-2 rule), `0`, or a positive multiple of 3584 ≤ 1,000,000 (largest legal value
  `999936 = 279 × 3584`); exported to **both** ranks. The launcher need not re-validate — the coordinator
  validates the raw value unconditionally at boot and refuses to start on anything else — and the
  resolved per-group vector is logged by the overlay itself.

Recommended production setting after validation:
```
GLM53_APC_RETENTION_INTERVAL=            # unset -> dense MLA + mamba, 3584 hit grid
GLM53_APC_RETENTION_INTERVAL_SWA=0       # drafter: replay/junction boundaries only
```

## 8. Test plan

### 8.1 Host test — `tests/test_apc_per_group_retention.py`

Runs anywhere with a copy of `kv_cache_coordinator.py`; no GPU, no vLLM import.

1. Apply the patch to a copy of the fork's file; assert MARK present, anchors consumed, `py_compile`
   clean. Re-apply; assert byte-identical (idempotence).
2. **Call sites and boot log.** Zero remaining `retention_interval=self.retention_interval,`, exactly two
   `retention_interval_by_group[i]`, only the two `cache_blocks` loops enumerated, and an init
   `logger.info` carrying `retention_by_group=%s` fed from `_glm53_format_retention_vector`.
3. **Min-exemption derivation** (Codex #4). `_glm53_min_exempt_group_ids` over the live seven-group
   layout: `{6}` under `eagle_group_ids={6}`; **empty** under the upstream all-groups fallback, under
   `{0}`, under the superset `{0,6}`, under `∅`, and when the only SWA-derived spec is `KpoolTailSpec`.
4. **Routing matrix.** `exec` the injected helpers in an isolated namespace with stub spec objects:
   exact-`SlidingWindowSpec` **and min-exempt** → SWA value; the same spec **not** min-exempt → global
   (the explicit path honours min-exemption, not just the class name); `KpoolTailSpec` (subclass) →
   global; `MambaSpec`/`MLAAttentionSpec`/`UniformTypeKVCacheSpecs`-wrapped → global even when
   eagle-flagged; auto mode (`swa=None`) → `0` for a hit-inert min-exempt SWA group, global for a wide
   window / `window == align` / non-exempt group / missing alignment / bare spec.
5. **Resolved vector + fail-closed override.** `_glm53_resolve_retention_by_group` over the live layout
   returns `(None,)*6 + (0,)` and renders as `[None,None,None,None,None,None,0]` — the exact string the
   deployment acceptance criterion greps for. A leftover global `14336` shows up in the vector rather
   than being hidden (Codex #6). Setting the SWA value raises `ValueError` for every non-exempt
   `eagle_group_ids` and for a model with no SWA group at all, while the automatic rule stays inert
   there.
6. **Unconditional env validation** (Codex #5). `_glm53_swa_retention_env` accepts unset/empty/blank →
   auto, `0`, `14336`, and `999936` (= 279·3584, the largest legal value); rejects junk (`"nope"`,
   `"3584.0"`), negatives (`-1`, `-3584`), non-multiples (`1`, `3000`) and over-cap (`1003520`), and its
   documented cap is asserted to be exactly `1,000,000`. The per-group validator over the resolved
   vector is asserted separately (no cap there — inherited *global* values are upstream's to bound).
7. **Overlay composition** (Codex #8). Against a **pristine** copy of the fork's file, apply
   `patch_hybrid_prefix_hit.py` then this one, and this one then `patch_hybrid_prefix_hit.py`. Both
   orders must apply, `py_compile`, carry both MARKs, define the shared `_glm53_inner_kv_spec` /
   `_glm53_is_draft_swa_spec` exactly once, add `import os` exactly once, survive re-application of
   either patch in either order byte-identically, retain Mia's hybrid-`min()` skip and
   `eagle_group_ids` narrowing, resolve the live layout to `[None,…,None,0]` — and the two orders must
   produce **byte-identical** output.
8. **Id-cost + capacity arithmetic** (§2, §4.1) against a pure-Python replica of the two
   `reachable_block_mask` formulas (dense 33/56; `R=14336` 33/224; boundary tail 33), and the §4.1
   formula evaluated for all five rows — `14 / 23 / 42 / 72 / 54` — so the doc's table and the test
   cannot drift apart.

Exit non-zero on any failure (Codex's "validators must fail closed" rule).

**Status: green.** Run:

```
GLM53_KV_COORDINATOR_PY_SRC=<copy of the fork's kv_cache_coordinator.py> \
GLM53_KV_COORDINATOR_PY_PRISTINE=<pristine copy of the same file> \
    python3 tests/test_apc_per_group_retention.py
```

`..._PRISTINE` may be omitted when `..._SRC` is itself unpatched, or when a pristine copy sits at
`/tmp/kv_cache_coordinator_pristine.py`; the composition case **fails closed** rather than skipping if
neither is available. Verified 2026-08-31 against a copy of the live container file (which already
carries `patch_hybrid_prefix_hit`) plus a pinned pristine copy of the same fork source.

### 8.2 Live plan (after the current sweep finishes; the server must be idle and past the boot warm-up burst)

Config under test: `GLM53_APC_RETENTION_INTERVAL=` (dense) + `GLM53_APC_RETENTION_INTERVAL_SWA=0`.
Control: today's default `GLM53_APC_RETENTION_INTERVAL=14336`.

| # | probe | pass criterion |
|---|---|---|
| L1 | boot receipt | `docker logs … \| grep retention_by_group` shows **`retention_by_group=[None,None,None,None,None,None,0]`** on **both** ranks (this, not the env var, is the acceptance criterion — Codex #6); `docker exec … env` on both ranks shows the SWA var |
| L2 | pair ladder 45K / 56K / 66K / 80K (`tests/validate_apc_retention.py`) | **all ≥ 96 %**; predicted headroom to ~193.5K/conv (§4.1 — an *ordering* prediction, already one segment optimistic at `R=7168`) |
| L3 | 3-turn multiturn | ≥ 99 % per turn; re-turn TTFT 0.9–4.0 s |
| L4 | subagent divergence (the ~20K shared-prefix probe) | hit lands on the **3584** grid ⇒ **65–70 %** (vs 45 % at 14336, 67.6 % at 7168) |
| **L5a** | **divergence-suffix sweep** (see below) | at every suffix length the run completes, and the drafter **acceptance rate** is within the stated band |
| **L5b** | **drafter cold-window cost, repeated** | decode tok/s over the first 256 output tokens of an L4 warm hit ≥ 0.90 × the same prompt's cold-run rate, on the **median of 3 repetitions**, and the **worst of the 3** ≥ 0.85 × cold. If it fails, fall back to `GLM53_APC_RETENTION_INTERVAL_SWA=14336` (still 4× cheaper than dense, still a 3584 grid for MLA/mamba) |
| L6 | equivalence gate v3 (logprob, A13 thresholds: cold-vs-warm max\|Δlogprob\| ≤ 3× the cold-vs-cold floor, position-0 token identical) | pass — **necessary, not sufficient**, see below |
| L7 | pool pressure | `vllm:kv_cache_usage_perc` after L2 ≤ the 14336 control's value |

#### L5a — divergence-suffix sweep (Codex #2)

A sparse drafter miss allocates a **fresh, zeroed** window (`S:90-93`). Zeroed is not another request's
data — but it is not valid KV for the preceding 2048 prompt tokens either, and the target's verification
will hide that in the output. So probe it directly.

Fix one warm 3584-aligned prefix, then issue a follow-up whose **divergence suffix** (tokens after the
last cached boundary) is **0, 1, 64, 2047, 2048** tokens. 0 and 2048 are the interesting ends: at 0 the
drafter must read the pinned replay-boundary tail and at 2048 the whole window is rewritten by the
prefill of the uncached remainder, so it is free (§5.3.4). 1 / 64 / 2047 are the cases the design claims
are bounded and self-limiting.

For each suffix, measure the **draft acceptance rate** from the Prometheus deltas across the request:

```
acceptance = Δ vllm:spec_decode_num_accepted_tokens_total
             / Δ vllm:spec_decode_num_draft_tokens_total
```

Record it against a **cold** run of the identical prompt (fresh `cache_salt`, same seed) and against the
`R=14336` control. Pass: the warm acceptance rate at every suffix ≥ **0.90 ×** the cold rate for the same
prompt, and the 0-token suffix (the pinned replay boundary — the case retention `0` exists to keep warm)
≥ **0.97 ×**. Require both counters to be present; a missing metric fails, it does not default to pass.
Failing this at 1/64/2047 but passing at 0/2048 is the predicted acceptance dip and argues for
`..._SWA=14336`, not for the design being wrong. Failing at 0 falsifies the "pinned replay boundary keeps
the own-conversation follow-up warm" claim in §5.3.4.

#### Why L6 is not enough

**Target-output equivalence does not prove the drafter's KV is valid.** The target model verifies and
corrects every draft token, so a drafter reading zeroed KV still yields byte-identical output — it just
yields it more slowly, with a collapsed acceptance rate. L6 therefore gates *correctness of the answer*;
L5a gates *validity of the drafter state*. Neither substitutes for the other, and only L5a can fail in a
way that says "the speculative path is now doing no work".

#### Reproducibility (Codex #7)

Every validator asserts and exits non-zero; refuse a busy server; require each metric to be present
rather than defaulting to pass. Beyond that:

* **Identical seeds and cache namespaces** across the cold/warm pair and across the proposed/control
  arms: fixed `seed`, `temperature=0`, and a unique `cache_salt` per *cold* run (the warm run reuses the
  preceding run's salt — that is what makes it warm).
* **3 repetitions** of the L5b cold/warm pair; report **median and the lower tail** (min of 3), not a
  single noisy comparison. One run is not evidence at this effect size.
* **"first 256 decode tokens" is defined as**: output tokens 1…256 of the warm request, timed from the
  first token's arrival (so **TTFT and prefill are excluded**), excluding the final chunk if the request
  stops before 256, and excluding any request that hit a preemption (`vllm:num_preempted_requests`
  delta > 0) — such a run is discarded and repeated, not averaged in.

Extension probe (cheap, worth doing once the above is green): push the ladder to 120K and 160K pairs — the
model predicts coexistence to ~193.5K per conversation, which no configuration has reached yet — with
the §4.1 caveat that the model was already one segment optimistic at `R=7168`.

## 9. Upstreamability

Two separable vLLM PRs, in increasing order of ambition:

**PR-U1 — "prefix-cache retention interval per KV-cache group".**
Purely mechanical: `KVCacheCoordinator.retention_interval` becomes `retention_interval_by_group`, the
manager API is untouched (it is already per-call, `S:429-434`), and the default resolution keeps the
current global value for every group ⇒ **byte-identical behaviour** unless a group opts out. This is a
clean, reviewable change that unblocks any hybrid model whose groups have wildly different id costs per
token (this kit: 33 vs 1). It is also the enabling refactor for anything else in this space.

**PR-U2 — "do not LRU-cache sliding-window blocks outside the reachable set".**
The general statement of the bug, model-independent:

> When a sliding-window group's window is smaller than the coordinator's cache-hit alignment, a
> `need`-block tail is hashed at every alignment boundary. Those blocks are later freed *cached*
> (`BlockPool.free_blocks`, `B:719-743`), landing at the most-protected end of a global LRU shared with
> full-attention and Mamba groups whose per-token id cost is `need` times lower. The result is a priority
> inversion: the group with the least reusable state gets the most eviction protection.

Proposed upstream default: derive the SWA group's retention from `need` so its id cost per alignment
segment is ≤ 1 — i.e. `retention_swa = alignment_tokens × need`, or `0` when the group is additionally
excluded from the hit `min()`. Gate it behind the existing `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`
machinery so the change is opt-out.

A third, smaller upstream note worth filing separately: `SlidingWindowManager.find_longest_cache_hit`'s
`TODO` at `S:938-943` (skip `sliding_window_contiguous_blocks` on a miss) becomes materially more
valuable once SWA groups cache sparsely — it turns the miss scan from O(n) into O(n/need + need).


## 9.1 Initialized-KV evidence (Codex final pass, finding 2)
What can be proven from code today: a drafter (SWA) miss at a boundary is served by `add_local_computed_blocks` padding
nulls and `allocate_new_blocks` pulling a fresh window whose block ids are recorded via `_record_new_block_ids`
(single_type_kv_cache_manager.py:270-286, :332-364), i.e. the drafter never reads another request's data — the blocks are
either this request's own newly allocated blocks (written by the current prefill/decode) or null blocks. What is NOT proven by
target-output equivalence is the *quality* of the drafter's proposals right after such a miss (the window is refilled by the
uncached remainder before decode; a very short remainder leaves part of the window empty) — that is exactly what L5a's
divergence-suffix sweep (0/1/64/2047/2048) with draft-acceptance deltas measures, with `GLM53_APC_RETENTION_INTERVAL_SWA=14336`
as the documented fallback if acceptance drops below the 0.90× gate at any suffix. A runtime assertion that every drafter block
read belongs to the current request is out of scope for this overlay (it would live in the attention backend's block-table
validation); the block-table plumbing above is the guarantee.
