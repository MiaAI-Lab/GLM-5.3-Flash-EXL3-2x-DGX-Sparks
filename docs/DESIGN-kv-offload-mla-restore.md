# Stage-2 g0 (MLA + indexer) disk restore (`GLM53_KV_OFFLOAD_RESTORE`)

Design record for `overlay/patch_kv_offload_restore_g0.py` + the `start.sh`
wiring + the stage-3 mamba differential harness under `tests/`. Stage 2 of
the disk KV-offload research plan (PLAN-KV-OFFLOAD.md §11), stacked on the
stage-1 store-only tier (`docs/DESIGN-kv-offload-store-only.md`).

## The honest headline

**A group-0-only restore yields ZERO hit uplift on GLM-5.3-Flash itself, by
design.** Two independent facts force this:

1. **The hybrid min.** The connector's reconciled external hit is the min
   across all KV-cache groups. The four mamba (KDA) groups have no restore
   source in this stage; reporting external tokens their recurrent state
   cannot back would serve garbage as a hit.
2. **The core invalid-block path is single-group.** Pinned receipt
   `tests/fixtures/image_487ecf187_core_sched_scheduler.py:2954-2955`:
   `(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)` under
   `TODO (davidb): add support for hybrid memory allocator` — a load-failure
   report on the 7-group live layout would crash the scheduler core.

The eligibility predicate therefore requires **exactly one TOTAL KV-cache
group** (not one *eligible* group — an excluded scratch/drafter group is
still a core group the restored tokens cannot back; Codex OFFLOAD2
finding 2), full attention, `blocks_per_chunk=1`, prefix caching on, both
knobs on. On the live hybrid the restore machinery is **INERT with one boot
log line naming the reason** and serving is byte-identical to stage 1.

**What stage-2 receipts CAN prove** (on a probe that isolates g0 — a
single-full-attention-group deployment, exactly the shape of g0's mixed
MLA+indexer group):
- wire-correct restore of g0 bytes through the plan-§5 chain: on-disk
  manifests answer the lookup → `allocate_slots(delay_cache_blocks=True)` →
  `WAITING_FOR_REMOTE_KVS` → worker pread with header/CRC/segment-table
  verification → DMA into the allocated blocks → `cache_blocks` re-entry →
  the restored prefix serves as an ordinary GPU prefix hit (the fork-trace
  receipt);
- correct failure degradation (the stage-2 T4 contract): per-(chunk, rank)
  failure → zero-fill + `invalid_block_ids` → core truncation at the failed
  chunk → recompute; fresh-session control green.

**What they CANNOT prove**: any TTFT uplift on this model (needs stage-3
mamba restore); mamba state semantic correctness across boots (the
differential harness built here is the stage-3 instrument for exactly that —
and it is a fixture comparator, not evidence; see below).

## How the restore path works

### Lookup: on-disk manifests, keys/hashes only (plan §4 C2)

The scheduler NEVER touches the filesystem (plan R0: "the scheduler handles
only keys/hashes, never paths"). Each rank's store writer emits **manifest
availability events** over the existing worker→scheduler metadata channel:

    ("+" | "-" | "F", rank, namespace_hash, key, aux, writer_boot_id)

- `+` on a validated manifest publication (including the pre-existing-file
  dedup path, which now VALIDATES the manifest and every referenced chunk
  header before registering or announcing — Codex OFFLOAD2 finding 9);
- `-` on a keep-K retention supersession;
- `F` on a chunk write failure — **the per-job store failure bit is on the
  wire and load-bearing**: the lookup denies any boundary whose g0 key
  failed on any rank this boot (this closes the recorded stage-1 limit).

The scheduler ANDs `+` events across ranks (`min across ranks`, plan §4 C2):
a boundary is offer-able IFF every rank reported it under the pinned
namespace and none retracted it. Contiguity needs no chain walk: prefix-cache
block hashes are prefix-CHAINED (sha256, receipt R15), so boundary-hash
equality implies an identical cumulative chunk chain. Each WORKER re-validates
its own on-disk manifest (namespace, boundary, chain tail) before its preads;
a manifest missing at load time (retention race, crash) degrades to
recompute, never to a guess.

v1 restores at 3584 boundaries only (plan F6): a fine-grained local hit tail
is never extended from disk (non-chunk-aligned local hit ⇒ miss).

Restore concurrency = 1: while a disk load job is in flight the lookup
returns None (the stock deferred-lookup machinery re-queries). The gate
stores the job id and **self-heals** if the job vanished (reset), clears on
all-rank completion and on `reset_cache` (Codex OFFLOAD2 finding 11).

### Load job: a versioned plain dict, worker-local paths

`update_state_after_alloc` skips `manager.prepare_load` entirely for disk
hits (the CPU staging tier is a bounce buffer, never a restore source) and
ships `{"glm53_disk_load": 1, "v": 1, namespace, boundary_token_index,
entries}` with entries aligned 1:1 with the dst block order. A scheduler-side
shape mismatch ships an empty-entry spec, which the worker turns into an
all-chunks-failed job (T4 degradation).

The worker's `Glm53DiskLoader` runs SYNCHRONOUSLY inside
`start_kv_transfers` — one chunk at a time, pread → verify → scatter into the
GPU canonical tensors. Verification per chunk (Codex OFFLOAD2 finding 13):
full-payload CRC, header identity (namespace, hash, group, spec kind, block
size, token count, `tp_rank == this rank`, `tp_world`), and **exact
segment-table equality** against the expectation derived from this rank's own
specs (the writer's `_layer_segments` derivation) — a byte-count match alone
cannot detect swapped layers. Preconditions fail closed (writer disabled,
unknown spec version, GPU/staging ref layout mismatch, canonical-mapped
layout, missing GPU caches ⇒ every chunk failed, never a guess).

### T4: failure → zero-fill → invalid ids → recompute

ANY failure stops the read loop, zero-fills the failed chunk **and every
later chunk's** dst blocks (their contents are unwritten or garbage; DS4F
over-report pattern), records their physical block ids (never null block 0)
and STILL completes the job — every terminal path acks. The facade override
`get_block_ids_with_load_errors` drains the ids; the model-runner mixin reads
them immediately after `get_finished`
(`kv_connector_model_runner_mixin.py:105`, captured receipt), so failures
ride the same step's `KVConnectorOutput.invalid_block_ids`; the executor
unions across ranks; the core (`recompute_kv_load_failures=True` default,
core fixture :241) truncates `num_computed_tokens` at the first invalid block
and recomputes. Core truncation and core zeroing (`needs_kv_cache_zeroing`)
are separate, conditional mechanisms — which is why the loader zero-fills
itself and never relies on them.

**Why no cross-step fencing machinery** (Codex OFFLOAD2 finding 4,
dispositioned): the loader is synchronous within the engine step — there is
no cross-step disk DMA to fence; restore concurrency is 1; and a late ack
after `reset_cache` is discarded by the scheduler's existing
`_stale_job_threshold`. Preemption of a request holding the disk load job no
longer trips the stock `assert is_store` (the flush loop skips load jobs and
clears the gate — finding 3); a store job for the same request cannot race
the load job because both ranks ack in the same step the job started, and the
scheduler processes those acks before the request can be scheduled again.

### T2: restored boundaries stay reachable

`update_state_after_alloc` sets `request.glm53_restored_boundary` — the
**end-exclusive** restored token count (a multiple of the chunk size; 7168
keeps the mamba state block ending at token 7168 via the fixture's
`aligned // block_size - 1` arithmetic). The single_type manager's
`cache_blocks` appends it to `reachable_boundaries`, so sparse retention
masks (SWA/Mamba subclasses) can never silently drop a restored boundary's
re-entry; dense (full-attention) masks return None and are unaffected.
Lifecycle: set at allocation, persists for the request's lifetime
(deliberately not cleared — the restored anchor must stay reachable in every
later `cache_blocks` of this request), overwritten by a later re-restore,
dies with the Request. The anchor also applies verbatim to the upstream tree
(core scalar retention exists at [U]) — verified by
`tests/test_kv_offload_restore_venv.py`.

### S1: manifests on zero-cow layouts

Stage-1's manifest-candidate builder required recurrent (cow) groups, so a
pure full-attention layout — the g0 probe — would never publish a manifest
and the lookup would answer from nothing. The restore overlay extends the
rule: candidates = mamba chunk idxs when cow groups exist (live hybrid:
byte-identical to stage 1), else the job's full-group chunk idxs.

## Stage-3 mamba differential harness (designed here, NOT evidence)

`tests/_mamba_differential.py` + `tests/test_mamba_differential_harness.py`
build park-at-boundary fixtures with STUBBED KDA state tensors at the
receipted ABI (conv (10, 12288) bf16 245,760 B + temporal (32, 128, 128)
fp32 2,097,152 B per layer; real page 2,342,912 B; 9/9/8/8 layers; num_spec 7
in every header; the +8,192 B/layer slot-share padding never on disk) and the
exact bit-comparator (`compare_state_bits`) stage 3 will run against real
captures — a single flipped bit is localized to its (group, layer, tensor).
`tests/stage3_mamba_differential_protocol.sh` pins the live protocol
(step 0 = receipt the still-open mamba state-index convention, then boot-A
park → restart → boot-B restore → bit differential → logprob equivalence →
needle → fresh-session control → the full T4 fault matrix) and refuses to run
in this branch. None of this proves cross-boot mamba restore correctness —
that is precisely what stage 3 must measure (Codex OFFLOAD2 finding 14).

## Knobs

| knob | default | meaning |
|---|---|---|
| `GLM53_KV_OFFLOAD_RESTORE` | 0 | 1 = enable the g0 disk-restore lookup/load path. Refused by the launcher unless `GLM53_KV_OFFLOAD=1`. 0 = stage-1 behavior (store-only short-circuit) |

All stage-1 knobs unchanged; the launcher ships the restore overlay to both
ranks, pins the order scope → store_local → restore_g0, self-checks its
injected sources pre-stop, and keeps `offload_prompt_only: false` untouched.

## Codex gates (stage 2)

Advisory on the build plan BEFORE coding: CODEX-OFFLOAD2-PLAN.md
(gpt-5.6-luna, DO-NOT-PROCEED, 5 BLOCKER + 10 MAJOR → every finding adopted
or rebutted in writing; the build implements the revised design). Adversarial
review of the final diff: CODEX-OFFLOAD2-REVIEW.md (DO-NOT-SHIP, 1 BLOCKER +
8 MAJOR + 2 MINOR → all adopted or rebutted; fixes in the review-fixes
commit; confirm pass appended there). Both docs live in
cache-scheduling-2026-08-31/.

## Receipts

Host-side coverage: `tests/test_kv_offload_restore_g0.py` (patcher hygiene,
patched-runtime scheduler/loader/event-channel/T2 drives),
`tests/test_mamba_differential_harness.py`, `tests/test_kv_offload_restore_venv.py`,
extensions to `tests/test_launcher_kv_offload.py`. Live receipts (probe
window) are enumerated in the PR body and pending: inert-on-hybrid boot line,
probe fork-trace, fault-matrix cells, fresh-session control.
