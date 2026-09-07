# Stage-1 store-only disk KV tier (`GLM53_KV_OFFLOAD`)

Design record for `overlay/patch_kv_offload_scope.py` +
`overlay/patch_kv_offload_store_local.py` + the `start.sh` wiring. Stage 1 of
the disk KV-offload research plan: the tier **writes**; nothing reads it back
(`GLM53_KV_OFFLOAD_RESTORE` is refused at 1 by the launcher). Restore plumbing
is stage 2.

## Why

A session whose KV has left the ~455K-token resident GPU envelope reprefills
from zero (174 s measured at 120K). The plan's endgame is
park-to-disk/restore-on-touch; stage 1 lands the write side with zero blast
radius: knob off ⇒ no connector, byte-identical serving.

## The two boot blockers this image has

1. **Scratch-group divisibility crash.** `build_offloading_config` builds an
   offload group config for every KV-cache group and asserts
   `tokens_per_block % tokens_per_hash == 0` — the KpoolTailSpec scratch group
   (block 4, hash grid 64) crashes the connector at boot. Our upstream PR
   vllm-project/vllm#54743 scopes the group list to prefix-cacheable groups;
   the scope overlay ports that onto this image's older tree.
2. **Wrapped group specs.** This image hands the connector
   `UniformTypeKVCacheSpecs` wrappers (subclass of `KVCacheSpec` only), which
   the connector's isinstance dispatch cannot classify: the full-attention
   assert would crash on group 0, and mamba groups would silently lose their
   alignment/CoW semantics. The scope overlay adds a fail-closed single-type
   unwrap (`_glm53_kvo_inner_spec`), and reads cacheability through the
   image's `participates_in_prefix_caching` name.

Eligible set on this deployment: **{0, 2, 3, 4, 5}** — g1 (kpool scratch) out
by the upstream predicate, g6 (DFlash2 drafter) out by fork policy
(min-exempt, retention 0: its window is rebuilt by the remainder's prefill;
`GLM53_KV_OFFLOAD_DRAFTER=1` re-includes it for experiments). All eligible
groups sit on the 3584 grid, so `blocks_per_chunk=1` gives one 3584-token
chunk per group per boundary.

## Why the store is worker-local (risk R0)

`TieringOffloadingSpec.get_manager()` constructs the CPU primary tier AND all
secondary tiers on the scheduler and hands the fs tier the *scheduler node's*
staging memoryview. On this 2-node TP=2 kit, rank 1's staging mmap lives on
the other Spark: the scheduler-side view's rank-1 slots are never written,
and the stock fs tier would persist rank-0 bytes + uninitialised garbage as
rank-1 halves — corruption that only surfaces at restore. So:

- the connector JSON configures **CPU staging only** (no `secondary_tiers`)
  and sets `offload_prompt_only: false` — the upstream default (true) would
  silently skip decoded tokens, and a parked session must cover its
  responses too;
- store jobs carry a metadata side-channel (`TransferJob.glm53_store_meta`):
  ordered `(hash, group_idx, chunk_idx)` aligned with the job's block order,
  plus boundary-manifest candidates (cumulative chunk-hash chains);
- each worker SNAPSHOTS the job's bytes out of its own staging mmap
  synchronously at DMA completion (capture-then-write; adversarial-review
  finding 1: `reset_cache()` frees staging rows regardless of pending disk
  work, so async reads from the mmap could tear) — at `get_finished` before
  its ack, or right after `worker.wait()` on the flush/reset path — then a
  2-thread pool writes the private buffers (512 MB budget, over-budget
  captures dropped as lost stores) to **its own NVMe** under
  `GLM53_KV_OFFLOAD_DIR/glm53kv_<model>_<ns>_r<rank>/…`; acks are never
  deferred;
- rank is the initialized TP rank cross-checked against the connector
  config's rank — disagreement disables the writer rather than risking
  cross-rank mixing.

## On-disk format (v1)

`GLM53KV1` magic | u32 header_len | canonical-JSON header | u32 header-crc32
| payload. Header: format_version, namespace_hash, ORIGINAL group_idx,
spec_kind, layers, block_size_tokens, n_tokens_valid, boundary_token_index,
tp_rank/tp_world, per-layer segment table (KDA layers: conv/temporal split
with shape/dtype/derived-contiguous stride), state-ABI tag +
num_speculative_tokens for mamba groups, payload_len, payload_crc32
(zlib crc32; the header names the algorithm). Payload = **real (unpadded)
bytes only** — the CanonicalKVCacheRef layout the DMA uses already carries
unpadded page sizes, so the mamba slot-share padding never reaches disk.

Namespace hash: sha256[:12] over model, vllm version, build id, TP/world, KV
dtype, prefix-cache hash algo, tokens_per_hash, blocks_per_chunk,
num_speculative_tokens (baked into the KDA conv width), per-eligible-group
(original idx, block size, layers, real page sizes), the KDA state-ABI
descriptor, format_version. Any change forks the directory. Retention K is
recorded in the namespace file but does NOT fence the hash.

**Boundary manifests**: per (chain, boundary), keyed by the boundary hash,
published only after all four mamba payloads AND every cumulative full-attn
chunk are durable AND header-verified on that rank (namespace + hash + group
identity; per-chunk `[hash, len, crc]` recorded for full groups too; dedup of
an existing file likewise requires a verifying header, else it is rewritten).
fsync file+dir, tmp+rename. No manifest ⇒
the boundary does not exist for stage-2 lookup. Cross-rank rule (stage 2): a
boundary is restore-eligible iff BOTH ranks hold a valid manifest.

**Retention** (`GLM53_KV_OFFLOAD_KEEP_BOUNDARIES`, default 2): inline, per
publish — a manifest is superseded (manifest deleted, its mamba payloads
unlinked) once it is beyond the K most recent boundaries of every chain
containing it; full-attention chunks stay dense (cheap, and cumulative
references from any live manifest keep shallower boundaries restorable).
Cross-boot leftovers: `overlay/kv_offload_store_gc.py` (dry-run default).

## Failure semantics (everything degrades to reprefill)

| failure | behaviour |
|---|---|
| write fails | tmp unlinked, one log line, key enters the failed ledger (never manifested this boot) — lost store only |
| capture over budget | write dropped (lost store), key stays manifest-eligible for a later re-store |
| ENOSPC | writer pauses (all further writes dropped, one log); serving unaffected |
| crash between group writes | orphan payloads, no live manifest; GC reports/sweeps |
| torn/truncated file | header+CRC verification rejects at read (stage 2 / GC) |
| rank mismatch, wrong layout, blocks_per_chunk≠1 | writer disabled at init with a named reason; jobs flow normally |

Known stage-1 limit (recorded): the worker→scheduler ack protocol has no
per-job failure bit, so the CPU tier believes a failed key stored and will
not retry it this boot. Safe while restore is off (an incomplete boundary is
simply invisible); the failure bit lands with stage 2's T4 invalidation work.

## Knobs

| knob | default | meaning |
|---|---|---|
| `GLM53_KV_OFFLOAD` | 0 | 1 = enable the connector + store tier. 0 = byte-identical serving (no argv, no mounts) |
| `GLM53_KV_OFFLOAD_DIR` | `~/glm53-kv-offload` | host dir on EACH Spark's local NVMe; mounted rw at `/data/glm53-kv-offload` on both ranks |
| `GLM53_KV_OFFLOAD_CPU_GB` | 4 | CPU staging bounce (`cpu_bytes_to_use`); never a capacity tier |
| `GLM53_KV_OFFLOAD_RESTORE` | 0 | stage-1 hard rule: 1 is refused by the launcher |
| `GLM53_KV_OFFLOAD_DRAFTER` | 0 | 1 = include the drafter group (stage-4 experiment) |
| `GLM53_KV_OFFLOAD_KEEP_BOUNDARIES` | 2 | inline manifest retention (0 = dense) |

## Deviations from the research plan (recorded)

- `blocks_per_chunk=1` uniformly (plan wanted 16 for MLA): the image has one
  global chunk size and all eligible groups share block 3584.
- CRC is zlib crc32, not crc32c (stdlib; algorithm named in the header).
- The stock fs secondary tier is not configured (Codex advisory finding 3 /
  Q1): scheduler-side byte I/O is exactly the R0 hazard; the worker-local
  writer is the disk tier.

## Receipts

Live receipts (knob=0 byte-parity, knob=1 boot + store-cascade evidence on
both Sparks, store overhead within noise, fail-closed negatives) are captured
in the PR body before the PR opens; host-side coverage is
`tests/test_kv_offload_scope.py`, `tests/test_kv_offload_store_local.py`,
`tests/test_launcher_kv_offload.py` against the pinned image fixtures
(`tests/fixtures/`, sha256s in its README).
