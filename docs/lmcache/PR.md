# PR: make LMCache external KV actually work for GLM-5.3-Flash on 2× DGX Spark

**Target:** `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` (LMCache integration + recipe Dockerfile)
**Upstream lib under test:** LMCache `v0.5.4 (g3e11b8ed)`, vLLM `v0.1.dev20051+g487ecf187`
**Status:** ✅ **end-to-end external KV reuse PROVEN on a live TP=2 serve** (receipts below)
**Receipts:** `~/ai-lab/experiments/glm53-omp-ux-20260920/lmcache-patches/` (fixes + proof scripts),
`~/glm53-exl3/EVIDENCE-external-kv-proof.txt` (raw run output on the box)

---

## Result

```
=== RESULT ===
  EXT_HITS=132605.0      (delta across the decisive resend: +32,256 = 9 chunks x 3584)
  srv lookup_req=154112.0 hit=143360.0 l1_read=43.0

=== LMCTRACE (fold, server side) ===
  KEY0 hash=b6d08efb... kvrank=33554944 group=0     <- key now matches the stored shape
  RESULT found_count=15 num_chunks=18 stride=1
  RESULT found_count=9  num_chunks=9  stride=1      <- full 9/9 chunk fold hit
  ==> EXTERNAL KV REUSE PROVEN
  ==> connector consulted LMCache: YES
```

Metric used is vLLM's own connector counter (help text: *"External prefix cache hits from KV
connector cross-instance cache sharing"*), distinct from the in-process APC counter
`vllm:prefix_cache_hits_total`. Server-side `lookup_hit_tokens_total` moved 0 → 143,360.

---

## Defects found (9), with status

| # | defect | file(s) | status |
|---|---|---|---|
| 1 | Per-layer leaf-spec resolution hands every layer the group's wrapper spec (GLM group 0 = one `UniformTypeKVCacheSpecs` wrapping 24 heterogeneous MLA leaves) | `integration/vllm/kv_cache_groups.py` | ✅ fixed, proven |
| 2 | `build_engine_group_infos` assigns an engine group to every vLLM group incl. non-prefix-cachable scratch/ring groups (GLM group 1 = 12 `KpoolTailSpec`, dim-0-padded) | `integration/vllm/kv_cache_groups.py` | ✅ fixed, proven |
| 3 | Kernel page count not always a whole multiple of the logical/kernel block ratio | transfer kernel page math | ✅ fixed (upstream #4889 backport) |
| 4 | **Store staging poisoned by the excluded scratch group** — `GetStoreMetadata` `min()` over *all* groups caps `allocated_tokens` at `1 block x 4 tokens = 4` → `num_chunks = 0` → STORE never fires | `integration/vllm/lmcache_mp_metadata.py` | ✅ fixed, proven |
| 5 | Lookup result lost to a cleanup race (`update_state_after_alloc` discards an unresolved async lookup) | `integration/vllm/*.py`, server `lookup.py` | ✅ fixed (necessary, not sufficient) |
| 6 | *(retracted)* "stores stop reaching a recreated server" — fresh-counter artifact | — | ❌ retracted |
| 7 | *(superseded by 8)* "L2 lookup key mismatch" | — | ❌ superseded |
| 8 | **Fold requires every kv_rank shard per chunk and intersects all object groups; node-partitioned stores can never satisfy it** | `v1/distributed/bitmap_ops/fold.py`, `v1/multiprocess/modules/lookup.py` | ✅ fixed, proven |
| 8b | **My own expansion unpacked `(kv_rank, object_group_id)` in the wrong order** | `v1/multiprocess/modules/lookup.py` | ✅ fixed, proven |
| 9 | `POST /reset_prefix_cache` fails (`Failed to reset KV cache even when all the running requests are done`) while LMCache holds L1 read locks — blocks any in-place cold-cache A/B | `v1/.../lookup.py` lock lifetime | ⚠️ workaround documented |

---

## Defect 4 — the STORE-side blocker

`GetStoreMetadata` bounds the storable prefix with

```python
allocated_tokens = min(allocated_lengths.get(i, 0) * group_tokens_per_block[i]
                       for i in range(num_engine_groups))
```

GLM-5.3-Flash has `group_tokens_per_block = [3584, 4, 3584, 3584, 3584]` — **group 1 is a
`KpoolTailSpec` scratch ring with `block_size=4` and a fixed single-block allocation that never
grows with the sequence**. So the `min()` evaluates `1 × 4 = 4` for every request →
`allocated_tokens = 4` → `num_chunks = 4 // 3584 = 0` → **STORE never fires for any request**.

Captured live:

```
LMCDBG meta groups=5 tpn=[3584,4,3584,3584,3584] alloc={0:1, 1:1, 2:3, 3:3, 4:3}
```

**Fix:** skip `participates_in_prefix_caching == False` groups in that bound *only*. Block-id
tuples must stay in vLLM group order, because `group_layers_by_identity()` does **not** renumber:
the server's protocol group ids remain vLLM's `{0,2,3,4}` (hole at 1), `num_engine_groups =
max(id)+1 = 5`, and the hole carries no tensors.

After the fix: **95 staging steps with `chunks>=1`, 1,217 chunks stored to L1+L2.**

## Defect 8 — the LOOKUP-side blocker

`_chunk_major_object_keys` expands `chunk × object_group(0..4) × kv_rank(0..1)`, and
`fold()` (docstring: *"A chunk is present for a group only when every kv_rank shard is present"*)
intersects across **all** object groups. But stores are **node-local** (IPC requires same-node
tensors) and write **one merged object per chunk**:

```
disk:  24-79 L2 files, ALL with (kv_rank=0x02000200, object_group_id=0)
lookup: expands the full 5-group × 2-rank grid
=> intersection empty by construction => found_count = 0 on every lookup
```

Verified the hashes themselves were fine — **9/9 lookup chunk hashes present on disk**.

**Fix (per-server held-set fold):** expand and fold only over the `(kv_rank, object_group_id)`
combinations the server actually holds (L1 index ∪ L2 filenames), falling back to the full grid
when empty; reduce `AttnWindowDesc`/`group_layout_descs` to the held set; fold with
`num_ranks = len(held_ranks)`. The scheduler's existing `min()` across servers then performs the
correct global fold.

## Defect 8b — the bug that made fix 8 look broken

`held_object_key_shapes()` returns `(kv_rank, object_group_id)` tuples. My expansion loop wrote
`for group_id, kv_rank in held` — **swapping the fields**:

```
lookup built:  ObjectKey(kv_rank=0, object_group_id=33554944)
disk holds:    kv_rank=33554944 (rank 0 of ws=2), object_group_id=0
```

Instrumentation caught it directly (`LMCTRACE2 KEY0 ... kvrank=0 group=33554944`, and
`l1_present=0 absent=0` — keys present in the map but never matching). Fix: unpack in the
declared order. Two-line change; it was the difference between "structurally impossible" and
"working".

---

## Deployment note (this is what made the earlier attempts fail silently)

**The fixes must be applied to the LMCache *server* image, not the vLLM image.**
`_chunk_major_object_keys`, `fold()`, and `query_prefetch_status` all execute in the
multiprocess **server** process (`v1/multiprocess/server.py` instantiates `LookupModule`), inside
the `lmcache` server container — while defects 1–4 live in the vLLM-side integration. Applying
everything to the vLLM image only (`glm53-sm121-lmc-dbg:test`) leaves the running server
**completely unpatched**: confirmed by marker audit —

```
glm53-lmc-scratch:test      : lookup.py 643 lines, 0 glm53 markers
glm53-sm121-lmc-dbg:test    : lookup.py 707 lines, fold=3 race=1
```

→ the server kept executing the unpatched fold for every lookup. This is why "the fold fix
doesn't produce hits" was wrong: **it was never running.**

---

## Operational findings worth documenting

- **`Conflicts=` must be symmetric.** A stale renamed unit (`ds4-serve.service` vs the live
  `dsv41-serve.service`) silently disabled eviction and let two TP=2 tenants co-load — the exact
  memory-starve condition that wedges a Spark.
- **Restart ordering.** `start.sh`'s `stop()` uses `docker rm -f` (SIGKILL), so the connector
  never runs `UNREGISTER_KV_CACHE` and each node's LMCache server keeps a **CUDA-IPC mapping of
  the dead worker's KV (~17.4 GiB)**. The worker preflight then sees `MemAvailable < 99.35 GiB`
  and **refuses to boot**. Fixes: recreate the servers between restarts
  (`lmc-prep-boot.sh`), or wait >120 s for the reaper. Real fix: `docker stop` in `start.sh`.
- `nvidia-smi --query-compute-apps` **double-counts** IPC-mapped KV — never size a co-tenant
  from it; measure `MemAvailable` deltas.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is incompatible with
  `LMCacheMPConnector` (hard-validated); `start.sh` defaults it when unset → set it **empty**.
- Honored connector keys are `lmcache.mp.host=tcp://localhost` + `lmcache.mp.port=5555`;
  `lmcache.server_url` is silently ignored. `tcp://localhost` is correct for per-node servers
  (each TP rank runs `--network host` on its own node).
- The server container has **no volume mounts** — the fs L2 adapter writes into the container's
  writable layer. Recreating the server deletes L2 unless you `docker commit` first.
- `VLLM_SERVER_DEV_MODE=1` (or the recipe's narrower `GLM53_EXPOSE_CACHE_RESET=1`) exposes
  `POST /reset_prefix_cache` — the only way to clear the in-process APC without a full reboot.

---

## Latency

Matched-prompt measurements on the live serve:

```
cold prefill (never-stored prompt, cold APC) : 1,035-1,068 tok/s   (60.8k tok -> 56.9s; 92.0k tok -> 88.9s)
external-served (LMCache hit, ext +32,256)   : 34,411 tok in 3.7s  => ~9,300 tok/s
```

~9× on the prefill phase for a stored prompt. Note the benefit is **cross-restart persistence**:
vLLM's in-process APC already serves repeat turns at 4-6 s, so LMCache only pays off when the
APC is cold (fresh boot, eviction, or a different replica).

---

## Suggested PR scope

1. `lmc-fix-perlayer.py`, `lmc-exclude-scratch.py`, `lmc-patch4889.py` — already in the recipe
   Dockerfile (`Dockerfile.lmcache-stage`).
2. `debug_store_staging.py` (defect 4) — add to that stage.
3. `lmc-fix-lookup-race.py` (defect 5) + `lmc-fix-fold-topology.py` (defect 8) +
   `lmc-fix-held-unpack.py` (defect 8b) — **apply to the LMCache server image**, and document
   that split in the Dockerfile, since mis-targeting is silent.
4. `lmc-prep-boot.sh` — the boot-preparation step required by the SIGKILL restart ordering.
5. Docs: restart ordering, image-split rule, `reset_prefix_cache` for cold-cache testing.
