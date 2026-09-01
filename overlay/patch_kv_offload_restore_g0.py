#!/usr/bin/env python3
"""Group-0 (full-attention) disk restore for the kv-offload tier
(GLM53_KV_OFFLOAD_RESTORE).

Stage 2 of the disk KV-offload plan (PLAN-KV-OFFLOAD.md §11): the tier can now
READ BACK group-0 payloads (MLA + indexer KV) at 3584-token boundaries — on
layouts where that is provably safe. Mamba (g2-5) and drafter (g6) groups are
explicitly NOT restored in this stage.

The honest headline (design doc + PR body carry it verbatim)
------------------------------------------------------------
**A group-0-only restore yields ZERO hit uplift on GLM-5.3-Flash itself, by
design.** The connector's reconciled external hit is the min across all
groups; the four mamba groups have no restore source in this stage, and
reporting external tokens the mamba state cannot back would be served garbage.
Additionally the core scheduler's invalid-block path is single-group
(pinned fixture image_487ecf187_core_sched_scheduler.py:2954-2955:
``(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)`` under a
``TODO (davidb): add support for hybrid memory allocator``) — a load-failure
report on a multi-group layout would crash the core. The eligibility
predicate therefore requires EXACTLY ONE TOTAL KV-cache group (Codex OFFLOAD2
findings 1/2/6): on the live 7-group hybrid the restore machinery is INERT
with one boot log line naming the reason, and serving is unchanged. What
stage 2 proves, on a single-full-attention-group probe (exactly the shape of
g0): wire-correct restore of g0 bytes through the §5 chain, and correct
failure degradation (T4). What it cannot prove: any TTFT uplift on this
model, or mamba state semantics (stage 3; harness under tests/).

What this overlay does (five files; runs AFTER patch_kv_offload_scope.py AND
patch_kv_offload_store_local.py — several anchors pin their patched output)
---------------------------------------------------------------------------
1. ``offloading/common.py`` — ``OffloadingWorkerMetadata`` gains
   ``glm53_manifest_events`` (worker→scheduler manifest availability channel,
   plan §4 C2: "workers report it; the scheduler takes the min across
   ranks"); ``aggregate()`` concatenates it via ``getattr`` (legacy-meta
   tolerant; Codex OFFLOAD2 finding 7).
2. ``offloading/scheduler.py`` —
   - ``Glm53RestoreState``: the eligibility predicate (one TOTAL group,
     full-attention, blocks_per_chunk=1, prefix caching on, both knobs on),
     the AND-across-ranks boundary registry fed by worker events ("+"
     publish / "-" retention supersede / "F" store-write failure — the
     per-job store-failure bit is ON THE WIRE and load-bearing at lookup:
     a boundary whose g0 key failed on any rank is never offered), the
     single-flight restore gate (concurrency=1; the gate stores the job id
     and SELF-HEALS if the job vanished — finding 11), and the disk lookup
     (keys/hashes only, NO filesystem access on the scheduler — finding 8;
     contiguity needs no chain walk because prefix-cache block hashes are
     prefix-CHAINED: boundary-hash equality implies an identical cumulative
     chain).
   - the stage-1 restore-off short-circuit now routes to the disk lookup
     when GLM53_KV_OFFLOAD_RESTORE=1 (exactly one line replaced — every
     piece of stage-1 bookkeeping around it is preserved; finding 12).
   - ``update_state_after_alloc``: a disk hit skips ``manager.prepare_load``
     entirely; the load job's ``src_spec`` is a versioned plain dict
     (``{"glm53_disk_load": 1, "v": 1, namespace, boundary, entries}``,
     entries aligned 1:1 with the dst blocks); the request gains
     ``glm53_restored_boundary`` (END-EXCLUSIVE restored token count,
     multiple of the chunk size; consumed by the T2 patch below; persists
     for the request's lifetime so the restored anchor stays reachable in
     every later cache_blocks — finding 10).
   - preemption flush: the stock path asserts every in-flight job of a
     preempted request is a store; a disk LOAD job is now skipped with the
     gate cleared instead of asserting (finding 3).
   - store-meta builder: boundary-manifest candidates now also exist on
     layouts with NO recurrent groups (candidates = the job's full-group
     chunk idxs); hybrid layouts are byte-identical to stage 1
     (disposition S1 — without this a pure full-attention layout would
     never publish a manifest and the lookup would answer from nothing).
3. ``offloading/worker.py`` —
   - ``_init_worker`` stashes the GPU-side CanonicalKVCaches;
   - disk load jobs NEVER enter ``worker.submit_load``/``get_finished``'s
     success asserts: ``_glm53_run_disk_load`` is a SYNCHRONOUS
     pread → verify → scatter loop (one chunk at a time; restore
     concurrency 1; no cross-step disk DMA exists, which IS the
     preemption/reset fence — finding 4 disposition; stale completions are
     discarded by the scheduler's ``_stale_job_threshold``);
   - verification per chunk (finding 13): full-payload CRC, header identity
     (namespace/hash/group/spec/block-size/tokens/rank/world/format), and
     EXACT segment-table equality against the expectation derived from this
     rank's own specs (the writer's ``_layer_segments`` derivation);
     manifest re-validation on THIS rank before any pread;
   - T4 (stage-2 contract): ANY failure zero-fills the failed chunk and
     every later chunk's dst blocks, records their physical ids (never null
     block 0) for ``get_block_ids_with_load_errors``, and the job STILL
     completes — every terminal path acks + reports before the model
     runner builds its output (finding 5; the mixin drains load errors
     immediately after ``get_finished`` — receipt in the design doc);
   - the stage-1 writer (store overlay's injected class) gains manifest
     EVENTS: "+" on validated publish, "-" on retention supersede, "F" on a
     write failure; the pre-existing-manifest dedup branch now VALIDATES
     the manifest and every referenced chunk header before registering or
     announcing it, and applies retention (finding 9);
   - ``build_connector_worker_meta`` returns a meta when events are pending
     even with zero completed jobs (finding 7).
4. ``offloading_connector.py`` (facade) — ``get_block_ids_with_load_errors``
   override draining the worker-side error set (the base default returns
   empty; the model-runner mixin consumes it into
   ``KVConnectorOutput.invalid_block_ids``).
5. ``v1/core/single_type_kv_cache_manager.py`` — T2: ``cache_blocks`` appends
   ``request.glm53_restored_boundary`` (when set) to ``reachable_boundaries``
   so sparse retention masks (SWA/Mamba subclasses) can never silently drop
   a restored boundary's re-entry; dense (full-attention) masks are
   unaffected (mask None). Live-relevant from stage 3; host-tested now.

Kill switch: GLM53_KV_OFFLOAD_RESTORE=0 (default) keeps stage-1 behavior —
the lookup short-circuit still returns 0 external hits, no load jobs, no
WAITING_FOR_REMOTE_KVS. The launcher refuses RESTORE=1 unless
GLM53_KV_OFFLOAD=1.

Conventions follow the stage-1 patchers: pinned ANCHORs, MARK sentinels,
``verified_state``, ``prepare``, idempotent, atomic replace, pyc clear,
drift => nonzero. ALL anchors in ALL five files preflight before ANY write.
MUST run AFTER patch_kv_offload_scope.py + patch_kv_offload_store_local.py.

Usage::

    python3 patch_kv_offload_restore_g0.py              # apply
    python3 patch_kv_offload_restore_g0.py --preflight  # validate anchors only
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

VLLM_ROOT = os.environ.get(
    "GLM53_VLLM_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"
)
_CONN = "distributed/kv_transfer/kv_connector/v1/offloading"

TARGET_COMMON = Path(
    os.environ.get("GLM53_KVO_COMMON_PY", f"{VLLM_ROOT}/{_CONN}/common.py")
)
TARGET_SCHED = Path(
    os.environ.get("GLM53_KVO_SCHED_PY", f"{VLLM_ROOT}/{_CONN}/scheduler.py")
)
TARGET_WORKER = Path(
    os.environ.get("GLM53_KVO_WORKER_PY", f"{VLLM_ROOT}/{_CONN}/worker.py")
)
TARGET_FACADE = Path(
    os.environ.get(
        "GLM53_KVO_FACADE_PY",
        f"{VLLM_ROOT}/distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
    )
)
TARGET_SINGLE_TYPE = Path(
    os.environ.get(
        "GLM53_SINGLE_TYPE_PY",
        f"{VLLM_ROOT}/v1/core/single_type_kv_cache_manager.py",
    )
)

TAG = "[glm53-kv-offload-restore]"
ENV_RESTORE = "GLM53_KV_OFFLOAD_RESTORE"


# ---------------------------------------------------------------------------
# Scheduler-side injected source: Glm53RestoreState.
# Uses _glm53_kvo_restore_enabled/_glm53_kvo_store_local_enabled from the
# store overlay's mode helpers (injected earlier in scheduler.py).
# ---------------------------------------------------------------------------
RESTORE_SCHED_SRC = '''# [glm53-kv-offload-restore] scheduler-side disk-restore state
_GLM53_KVR_TAG = "[glm53-kv-offload-restore]"
_GLM53_KVR_SPEC_V = 1


class Glm53RestoreState:
    """Disk-restore lookup state: eligibility, the AND-across-ranks manifest
    registry (worker events), the single-flight gate, and the boundary
    lookup. Keys/hashes only -- the scheduler NEVER touches the filesystem
    (plan R0; Codex OFFLOAD2 finding 8)."""

    # Registry bounds (Codex OFFLOAD2-REVIEW finding 8): fail CLOSED on
    # overflow -- new boundaries are dropped (missed hits only), never wrong
    # hits; one log line when first hit.
    _MAX_BOUNDARIES = 1 << 16
    _MAX_FAILED_KEYS = 1 << 16

    def __init__(self, config, vllm_config, log):
        self._log = log
        self._config = config
        self._inflight_job = None
        self._disk_jobs = set()
        self._namespace = None
        # boundary_hash -> {"ranks": set[int], "tokens": int}
        self._boundaries = {}
        # rank -> set[(hash_hex, group_idx)] whose store WRITE failed on
        # that rank under its CURRENT writer generation -- the per-job store
        # failure bit, on the wire and load-bearing at lookup (stage-1 known
        # limit closed per the task contract). Sticky WITHIN a generation
        # (mirrors the stage-1 writer's failed-key ledger); a writer restart
        # (boot-id change) retracts that rank's failures along with its
        # availability, so a recovered writer is not permanently suppressed
        # (Codex confirm pass, finding 9).
        self._failed_by_rank = {}
        # rank -> writer boot id: a changed boot id means that rank's writer
        # restarted; its previous availability is void (finding 8 fencing).
        self._rank_boot = {}
        # req_id -> pending disk-hit boundary (end-exclusive tokens), set by
        # lookup, consumed by update_state_after_alloc, dropped on miss/reset.
        self._pending_hits = {}
        self._ns_mismatch_logged = False
        self._overflow_logged = False
        self._failed_overflow = False

        reason = None
        try:
            restore_on = _glm53_kvo_restore_enabled()
            store_on = _glm53_kvo_store_local_enabled()
        except ValueError as exc:
            restore_on = store_on = False
            reason = str(exc)
        if reason is None and not restore_on:
            reason = "GLM53_KV_OFFLOAD_RESTORE=0"
        elif reason is None and not store_on:
            reason = "GLM53_KV_OFFLOAD=0 (restore reads the store tier)"
        elif reason is None and config.blocks_per_chunk != 1:
            reason = f"blocks_per_chunk={config.blocks_per_chunk} (need 1)"
        elif reason is None and (
            not config.kv_group_configs
            or config.kv_group_configs[0].tokens_per_chunk <= 0
            or config.kv_group_configs[0].hashes_per_chunk <= 0
            or config.tokens_per_hash <= 0
            or config.kv_group_configs[0].tokens_per_chunk
            != config.kv_group_configs[0].hashes_per_chunk
            * config.tokens_per_hash
        ):
            reason = "degenerate chunk/hash geometry"
        elif reason is None and (
            config.num_kv_cache_groups != 1 or len(config.kv_group_configs) != 1
        ):
            # ONE TOTAL group required, not one eligible group (Codex
            # OFFLOAD2 finding 2): an excluded scratch/drafter group would
            # still be a core KV-cache group the restored tokens cannot
            # back; and the core invalid-block path is single-group (pinned
            # fixture image_487ecf187_core_sched_scheduler.py:2954-2955).
            reason = (
                f"layout has {config.num_kv_cache_groups} KV-cache groups "
                f"({len(config.kv_group_configs)} eligible): g0-only restore "
                "yields no external hits on a hybrid layout by design"
            )
        elif reason is None:
            g = config.kv_group_configs[0]
            if g.sliding_window_size_in_chunks is not None:
                reason = f"group {g.group_idx} is not full-attention"
            elif g.requires_cow_source:
                reason = f"group {g.group_idx} has recurrent (mamba) semantics"
            elif g.is_eagle_group:
                reason = f"group {g.group_idx} is a draft-model group"
            elif not bool(
                getattr(
                    getattr(vllm_config, "cache_config", None),
                    "enable_prefix_caching",
                    False,
                )
            ):
                reason = "prefix caching disabled"
        self.disabled_reason = reason
        if reason is not None and restore_on:
            self._log.info(
                "%s restore INERT on this layout: %s", _GLM53_KVR_TAG, reason
            )
        elif reason is None:
            self._log.info(
                "%s g0 disk restore ACTIVE (single full-attention group %d, "
                "chunk %d tokens, %d workers)",
                _GLM53_KVR_TAG,
                config.kv_group_configs[0].group_idx,
                config.kv_group_configs[0].tokens_per_chunk,
                config.num_workers,
            )

    # ---- worker event registry (plan C2: min across ranks) --------------
    def _retract_rank(self, rank: int) -> None:
        for key in list(self._boundaries):
            entry = self._boundaries[key]
            entry["ranks"].discard(rank)
            if not entry["ranks"]:
                self._boundaries.pop(key, None)
        # Failures are generation-scoped too: a restarted writer starts a
        # fresh ledger, so its old failures must not suppress it forever.
        self._failed_by_rank.pop(rank, None)

    def consume_events(self, events) -> None:
        for ev in events:
            try:
                code = ev[0]
                rank = int(ev[1])
                ns = str(ev[2])
                key = str(ev[3])
                aux = int(ev[4])
                boot = str(ev[5]) if len(ev) > 5 else ""
            except (TypeError, ValueError, IndexError):
                continue
            # Rank must name a real worker (Codex OFFLOAD2-REVIEW finding 7):
            # garbage ranks must never help satisfy the quorum.
            if not 0 <= rank < self._config.num_workers:
                continue
            if self._namespace is None:
                self._namespace = ns
            if ns != self._namespace:
                if not self._ns_mismatch_logged:
                    self._ns_mismatch_logged = True
                    self._log.warning(
                        "%s dropping events from foreign namespace %s "
                        "(registry pinned to %s)",
                        _GLM53_KVR_TAG,
                        ns[:12],
                        self._namespace[:12],
                    )
                continue
            # Writer-generation fencing (finding 8): a changed boot id means
            # that rank's writer restarted -- everything it previously
            # announced is void.
            pinned_boot = self._rank_boot.get(rank)
            if pinned_boot is None:
                self._rank_boot[rank] = boot
            elif pinned_boot != boot:
                self._log.warning(
                    "%s rank %d writer generation changed (%s -> %s): "
                    "retracting its availability",
                    _GLM53_KVR_TAG,
                    rank,
                    pinned_boot[:16],
                    boot[:16],
                )
                self._retract_rank(rank)
                self._rank_boot[rank] = boot
            if code == "+":
                if (
                    key not in self._boundaries
                    and len(self._boundaries) >= self._MAX_BOUNDARIES
                ):
                    if not self._overflow_logged:
                        self._overflow_logged = True
                        self._log.warning(
                            "%s boundary registry full (%d): new boundaries "
                            "are dropped (missed hits only, fail closed)",
                            _GLM53_KVR_TAG,
                            self._MAX_BOUNDARIES,
                        )
                    continue
                entry = self._boundaries.setdefault(
                    key, {"ranks": set(), "tokens": aux}
                )
                if entry["tokens"] != aux:
                    # Same hash, different boundary tokens: corrupt or a
                    # collision -- drop the boundary entirely, fail closed.
                    self._boundaries.pop(key, None)
                    continue
                entry["ranks"].add(rank)
            elif code == "-":
                entry = self._boundaries.get(key)
                if entry is not None:
                    entry["ranks"].discard(rank)
                    if not entry["ranks"]:
                        self._boundaries.pop(key, None)
            elif code == "F":
                failed = self._failed_by_rank.setdefault(rank, set())
                if len(failed) < self._MAX_FAILED_KEYS:
                    failed.add((key, aux))
                elif not self._failed_overflow:
                    # Losing failure knowledge would be fail-OPEN; poison the
                    # lookup instead (restore off for the rest of the boot).
                    self._failed_overflow = True
                    self._log.error(
                        "%s failed-key ledger full (%d) for rank %d: disk "
                        "lookup DISABLED for the rest of this boot "
                        "(fail closed)",
                        _GLM53_KVR_TAG,
                        self._MAX_FAILED_KEYS,
                        rank,
                    )
                elif not self._failed_overflow:
                    # Losing failure knowledge would be fail-OPEN; poison the
                    # lookup instead (restore off for the rest of the boot).
                    self._failed_overflow = True
                    self._log.error(
                        "%s failed-key ledger full (%d): disk lookup "
                        "DISABLED for the rest of this boot (fail closed)",
                        _GLM53_KVR_TAG,
                        self._MAX_FAILED_KEYS,
                    )

    # ---- single-flight gate (finding 11: self-healing invariant) --------
    def job_started(self, job_id: int) -> None:
        self._inflight_job = job_id
        # DISK job ids, so the preemption/store-jobs guards act only on OUR
        # loads and preserve stock semantics (the assert) for any other
        # load-job kind (Codex confirm pass, new finding 1).
        self._disk_jobs.add(job_id)

    def is_disk_job(self, job_id: int) -> bool:
        return job_id in self._disk_jobs

    def job_done(self, job_id: int) -> None:
        if self._inflight_job == job_id:
            self._inflight_job = None
        self._disk_jobs.discard(job_id)

    def reset(self) -> None:
        self._inflight_job = None
        self._disk_jobs.clear()
        self._pending_hits.clear()

    # ---- the lookup -----------------------------------------------------
    def take_pending_hit(self, req_id):
        return self._pending_hits.pop(req_id, None)

    def lookup(self, sched, req_status):
        """Disk-manifest boundary lookup. Returns hit tokens (0 = miss,
        None = defer while another restore is in flight)."""
        req_id = req_status.req.request_id
        self._pending_hits.pop(req_id, None)
        if self.disabled_reason is not None or self._failed_overflow:
            return 0
        cfg = self._config.kv_group_configs[0]
        tpc = cfg.tokens_per_chunk
        hpc = cfg.hashes_per_chunk
        req = req_status.req
        local = req_status.num_locally_computed_tokens
        if local % tpc != 0:
            # v1 restores at chunk boundaries only (plan F6): a fine-grained
            # local hit tail is not extended from disk.
            return 0
        if self._inflight_job is not None:
            if self._inflight_job in sched._jobs:
                # Restore concurrency = 1: defer; the stock deferred-lookup
                # machinery re-queries this request later.
                return None
            # The job vanished (reset/terminal path missed): self-heal --
            # and drop its id from the disk-job set so stale ids never
            # accumulate (final-confirm minor).
            self._disk_jobs.discard(self._inflight_job)
            self._inflight_job = None
        max_chunk = min(req.num_tokens // tpc, len(req.block_hashes) // hpc)
        for k in range(max_chunk - 1, local // tpc - 1, -1):
            bhash = bytes(req.block_hashes[(k + 1) * hpc - 1]).hex()
            entry = self._boundaries.get(bhash)
            if entry is None:
                continue
            if entry["tokens"] != (k + 1) * tpc:
                continue
            # Exact rank-set equality (finding 7): every worker, no strays.
            if entry["ranks"] != set(range(self._config.num_workers)):
                continue
            if any(
                (bhash, cfg.group_idx) in failed
                for failed in self._failed_by_rank.values()
            ):
                continue
            hit = (k + 1) * tpc - local
            if hit <= 0:
                return 0
            self._pending_hits[req_id] = (k + 1) * tpc
            return hit
        return 0

    def build_load_spec(self, request, req_status, boundary_tokens, num_cached):
        """The versioned plain-dict src_spec for a disk load job. Entries are
        aligned 1:1 with the job's dst block order; a scheduler-side
        inconsistency ships an empty-entry spec, which the worker turns into
        an all-chunks-failed job (T4 degradation, never a guess)."""
        cfg = self._config.kv_group_configs[0]
        tpc = cfg.tokens_per_chunk
        hpc = cfg.hashes_per_chunk
        local = req_status.num_locally_computed_tokens
        entries = []
        if (
            boundary_tokens == num_cached
            and tpc > 0
            and hpc > 0
            and local % tpc == 0
            and boundary_tokens % tpc == 0
            # Enough HASHES for every entry (Codex OFFLOAD2-REVIEW finding 2:
            # the draft compared the CHUNK count against the hash list).
            and (boundary_tokens // tpc) * hpc <= len(request.block_hashes)
        ):
            for j in range(local // tpc, boundary_tokens // tpc):
                entries.append(
                    (
                        bytes(request.block_hashes[(j + 1) * hpc - 1]).hex(),
                        cfg.group_idx,
                        j,
                        tpc,
                    )
                )
        else:
            self._log.error(
                "%s disk-hit shape mismatch for %s (boundary %d, cached %d): "
                "shipping an all-fail load spec (recompute)",
                _GLM53_KVR_TAG,
                request.request_id,
                boundary_tokens,
                num_cached,
            )
        return {
            "glm53_disk_load": 1,
            "v": _GLM53_KVR_SPEC_V,
            "namespace_hash": self._namespace,
            "boundary_token_index": boundary_tokens,
            "entries": entries,
        }


'''

# ---------------------------------------------------------------------------
# Worker-side injected source: the synchronous disk loader.
# Uses the store overlay's codec + writer accessors (injected earlier).
# ---------------------------------------------------------------------------
RESTORE_WORKER_SRC = '''# [glm53-kv-offload-restore] worker-side synchronous disk loader
_GLM53_KVR_TAG = "[glm53-kv-offload-restore]"
_GLM53_KVR_SPEC_V = 1


def _glm53_attach_manifest_events(cw) -> None:
    """Drain the store writer's manifest events into the worker meta."""
    writer = cw._glm53_store_writer
    if writer is None:
        return
    drain = getattr(writer, "drain_manifest_events", None)
    if drain is None:
        return
    events = drain()
    if not events:
        return
    meta_events = getattr(
        cw._connector_worker_meta, "glm53_manifest_events", None
    )
    if meta_events is None:
        logger.warning(
            "%s worker meta lacks glm53_manifest_events (common.py not "
            "patched?) -- dropping %d events",
            _GLM53_KVR_TAG,
            len(events),
        )
        return
    meta_events.extend(events)


def _glm53_drain_disk_loads(cw):
    done = cw._glm53_disk_done
    cw._glm53_disk_done = []
    return done


def _glm53_expected_segments(writer, gidx, gpu_refs):
    """The exact segment table this rank expects for one chunk of group
    ``gidx`` -- derived from the same spec walk the writer's gather uses, so
    header equality proves layer identity/shape/dtype/offset/length, not
    just a total byte count (Codex OFFLOAD2 finding 13)."""
    group = writer._kv_groups[gidx]
    layer_names = list(group.layer_names)
    segments = []
    offset = 0
    for i, ref in enumerate(gpu_refs):
        layer = layer_names[i] if i < len(layer_names) else f"ref{i}"
        segments.extend(
            writer._layer_segments(group, layer, offset, ref.page_size_bytes)
        )
        offset += ref.page_size_bytes
    return segments, offset


def _glm53_hex64(value) -> bool:
    """Exact 64-char lowercase hex -- required before ANY path construction
    from wire/manifest data (Codex OFFLOAD2-REVIEW finding 5: a corrupt hash
    must never contribute separators or traversal components)."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


# Hard per-chunk payload cap (finding 6): the live g0 chunk is 27,163,136 B;
# anything past 256 MiB is garbage regardless of layout.
_GLM53_KVR_MAX_PAYLOAD = 256 * 1024 * 1024


def _glm53_read_chunk_payload(path, header):
    """Bounded read of one chunk's payload (the codec's header reader already
    validated envelope structure and total file length; this seeks past the
    envelope, reads exactly the expected bytes, and CRC-checks them)."""
    import zlib as _zlib

    expected = header["payload_len"]
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) != 12:
            raise ValueError(f"truncated envelope in {path!r}")
        hlen = int.from_bytes(head[8:12], "little")
        f.seek(12 + hlen + 4)
        payload = f.read(expected + 1)
    if len(payload) != expected:
        raise ValueError(f"payload length changed under us in {path!r}")
    if (_glm53_zlib_crc32(payload)) != header["payload_crc32"]:
        raise ValueError(f"payload crc mismatch in {path!r}")
    return payload


def _glm53_zlib_crc32(data) -> int:
    import zlib as _zlib

    return _zlib.crc32(data) & 0xFFFFFFFF


def _glm53_run_disk_load(cw, job_id, src_spec, dst_spec) -> None:
    """Synchronous disk restore of one load job (restore concurrency = 1).

    Terminal-result contract (Codex OFFLOAD2 findings 5 + REVIEW finding 3):
    the done-list append lives in a ``finally`` -- EVERY exit (happy path,
    precondition failure, manifest failure, per-chunk pread/header/CRC/
    segment/scatter failure, even an exception inside the failure
    bookkeeping) marks the job done (drained by get_finished, which acks +
    reports finished_recving) and, for failures, records the failed suffix's
    physical ids for get_block_ids_with_load_errors BEFORE zero-filling
    best-effort. The ONE deliberate exception: an unparseable dst_spec
    (destinations not identifiable) PROPAGATES -- acking without being able
    to report or zero the destinations could serve stale pool bytes as a
    hit, and fail-stop is the stage-1-adopted rule for exactly that class.
    The copy runs entirely inside this engine step: no cross-step disk DMA
    exists (the preemption/reset fence), and a late ack after reset_cache is
    discarded by the scheduler's _stale_job_threshold."""
    import time as _time

    # Destinations must be identifiable BEFORE the catch-all: see docstring.
    dst_ids = [int(b) for b in dst_spec.block_ids]

    failed_from = None
    reason = None
    tensors = None
    gpu_refs = None
    n_bytes = 0
    t0 = _time.monotonic()
    entries = []
    try:
        try:
            writer = _glm53_get_store_writer(cw)
            if writer._disabled_reason is not None:
                raise ValueError(
                    f"store writer disabled: {writer._disabled_reason}"
                )
            if (
                not isinstance(src_spec, dict)
                or src_spec.get("v") != _GLM53_KVR_SPEC_V
            ):
                raise ValueError(
                    f"unknown disk-load spec version {src_spec.get('v')!r}"
                )
            if src_spec.get("namespace_hash") != writer._namespace_hash:
                raise ValueError("disk-load namespace mismatch")
            entries = list(src_spec.get("entries") or [])
            if not entries or len(entries) != len(dst_ids):
                raise ValueError(
                    f"{len(entries)} entries vs {len(dst_ids)} dst blocks"
                )
            gidx = int(entries[0][1])
            if gidx not in writer._eligible_groups:
                raise ValueError(f"group {gidx} not eligible on this rank")
            caches = cw._glm53_gpu_caches
            if caches is None:
                raise ValueError("GPU canonical caches not registered")
            tensors = caches.tensors
            gpu_refs = list(caches.group_data_refs[gidx])
            staging_refs = writer._refs_per_group[gidx]
            if [r.page_size_bytes for r in gpu_refs] != [
                r.page_size_bytes for r in staging_refs
            ]:
                raise ValueError("GPU vs staging ref layout mismatch")
            for ref in gpu_refs:
                if getattr(ref, "mapping", None) is not None:
                    # A canonical mapping means a raw row copy is NOT the
                    # inverse of the store-side gather: fail closed.
                    raise ValueError(
                        "canonical-mapped layout not restorable (v1)"
                    )
            expected_segments, expected_len = _glm53_expected_segments(
                writer, gidx, gpu_refs
            )
            expected_segments = _glm53_json_norm(expected_segments)
            if not 0 < expected_len <= _GLM53_KVR_MAX_PAYLOAD:
                raise ValueError(f"implausible payload size {expected_len}")
            group = writer._kv_groups[gidx]
            inner = _glm53_unwrap_kv_spec(group.kv_cache_spec)
            expected_spec_kind = type(inner).__name__
            expected_block_size = getattr(inner, "block_size", None)

            # Manifest re-validation on THIS rank before any pread
            # (finding 8 of the advisory + REVIEW finding 4): identity,
            # FULL cumulative chain arithmetic, hex-validated hashes, and
            # the per-chunk [len, crc] ledger used against every header.
            boundary_tokens = src_spec.get("boundary_token_index")
            boundary_hash = str(entries[-1][0])
            if not _glm53_hex64(boundary_hash):
                raise ValueError("boundary hash is not 64-char lowercase hex")
            mpath = writer._manifest_path(boundary_hash)
            with open(mpath, encoding="utf-8") as f:
                man = _glm53_json.load(f)
            chain = man.get("chunk_hashes") or []
            tpc = int(entries[0][3])
            if (
                man.get("namespace_hash") != writer._namespace_hash
                or man.get("boundary_token_index") != boundary_tokens
                or not isinstance(boundary_tokens, int)
                or tpc <= 0
                or boundary_tokens != len(chain) * tpc
                or chain[-1:] != [boundary_hash]
                or not all(_glm53_hex64(h) for h in chain)
            ):
                raise ValueError("manifest identity mismatch at load time")
            # The job's entries must BE the chain's tail (content addressing
            # makes deeper equality redundant, but the manifest is the
            # commit record -- trust nothing shallower).
            tail = chain[len(chain) - len(entries):]
            if [str(e[0]) for e in entries] != tail:
                raise ValueError("job entries diverge from the manifest chain")
            ledger = (man.get("full_groups") or {}).get(str(gidx))
            if (
                not isinstance(ledger, list)
                or len(ledger) != len(chain)
                or any(
                    not (isinstance(entry_l, list) and len(entry_l) == 3)
                    for entry_l in ledger
                )
            ):
                raise ValueError("manifest chunk ledger missing/malformed")
            ledger_by_hash = {row[0]: (row[1], row[2]) for row in ledger}

            import torch as _torch

            for i, ent in enumerate(entries):
                try:
                    hash_hex = str(ent[0])
                    egidx = int(ent[1])
                    n_tokens = int(ent[3])
                    if not _glm53_hex64(hash_hex):
                        raise ValueError("entry hash is not 64-char hex")
                    if egidx != gidx:
                        raise ValueError("mixed groups in one disk load job")
                    row = dst_ids[i]
                    if row == 0:
                        raise ValueError("null dst block in a disk load job")
                    path = writer._file_path(hash_hex, gidx)
                    header = glm53_read_chunk_header(path)
                    led = ledger_by_hash.get(hash_hex)
                    if (
                        header.get("namespace_hash") != writer._namespace_hash
                        or header.get("hash") != hash_hex
                        or header.get("group_idx") != gidx
                        or header.get("spec_kind") != expected_spec_kind
                        or header.get("block_size_tokens")
                        != expected_block_size
                        or header.get("tp_rank") != writer._rank
                        or header.get("tp_world") != writer._world
                        or header.get("n_tokens_valid") != n_tokens
                        or header.get("payload_len") != expected_len
                        or led is None
                        or (header["payload_len"], header["payload_crc32"])
                        != (led[0], led[1])
                    ):
                        raise ValueError(
                            f"chunk header/ledger identity mismatch: {path}"
                        )
                    if header.get("segment_table") != expected_segments:
                        raise ValueError(f"segment table mismatch: {path}")
                    payload = _glm53_read_chunk_payload(path, header)
                    offset = 0
                    for ref in gpu_refs:
                        seg = payload[offset : offset + ref.page_size_bytes]
                        buf = _torch.frombuffer(
                            bytearray(seg), dtype=_torch.int8
                        )
                        tensors[ref.tensor_idx].tensor[
                            row, : ref.page_size_bytes
                        ].copy_(buf)
                        offset += ref.page_size_bytes
                    n_bytes += len(payload)
                except Exception as exc:  # noqa: BLE001 - per-chunk T4 cell
                    failed_from = i
                    reason = f"{type(exc).__name__}: {exc}"
                    break
        except Exception as exc:  # noqa: BLE001 - whole-job T4 cell
            failed_from = 0
            reason = f"{type(exc).__name__}: {exc}"

        if failed_from is not None:
            # Report FIRST (one atomic update; the ids are the T4 fence),
            # then zero-fill best-effort.
            cw._glm53_load_error_ids.update(
                int(r) for r in dst_ids[failed_from:] if r != 0
            )
            logger.warning(
                "%s disk load job %d FAILED at chunk %d/%d (%s): reported "
                "%d invalid block(s), zero-filling the suffix (recompute)",
                _GLM53_KVR_TAG,
                job_id,
                failed_from,
                len(entries) or len(dst_ids),
                reason,
                len(dst_ids[failed_from:]),
            )
            if tensors is not None and gpu_refs is not None:
                for row in dst_ids[failed_from:]:
                    if row == 0:
                        continue
                    try:
                        for ref in gpu_refs:
                            tensors[ref.tensor_idx].tensor[
                                row, : ref.page_size_bytes
                            ].zero_()
                    except Exception:  # noqa: BLE001 - ids already reported
                        pass
        else:
            stats = getattr(cw._connector_worker_meta, "transfer_stats", None)
            if stats is not None:
                try:
                    stats.load.record(n_bytes, _time.monotonic() - t0)
                except Exception:  # noqa: BLE001 - stats are best-effort
                    pass
            logger.info(
                "%s disk load job %d restored %d chunk(s), %d bytes, %.3f s",
                _GLM53_KVR_TAG,
                job_id,
                len(entries),
                n_bytes,
                _time.monotonic() - t0,
            )
    finally:
        if job_id not in cw._glm53_disk_done:
            cw._glm53_disk_done.append(job_id)


def _glm53_json_norm(segments):
    """JSON round-trip normalization for the header comparison."""
    return _glm53_json.loads(_glm53_json.dumps(segments))


'''


# ===========================================================================
# File 1: offloading/common.py — worker-meta manifest-event channel
# ===========================================================================
MARK_M1 = "    glm53_manifest_events: list = field(default_factory=list)  # [glm53-kv-offload-restore]\n"

ANCHOR_M1 = """    completed_jobs: dict[int, int] = field(default_factory=dict)
    transfer_stats: TransferStats = field(default_factory=TransferStats)

    def mark_completed(self, job_id: int) -> None:
"""

PATCHED_M1 = """    completed_jobs: dict[int, int] = field(default_factory=dict)
    transfer_stats: TransferStats = field(default_factory=TransferStats)
    # Manifest availability events from this rank's store writer:
    # ("+"|"-"|"F", rank, namespace_hash, key, aux, writer_boot_id) where
    # key/aux = (boundary_hash, boundary_token_index) for +/- and
    # (chunk_hash, group_idx) for F. Plain tuples: picklable on the wire.
    glm53_manifest_events: list = field(default_factory=list)  # [glm53-kv-offload-restore]

    def mark_completed(self, job_id: int) -> None:
"""

MARK_M2 = "            glm53_manifest_events=(  # [glm53-kv-offload-restore]\n"

ANCHOR_M2 = """        return OffloadingWorkerMetadata(
            completed_jobs=merged,
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
        )
"""

PATCHED_M2 = """        return OffloadingWorkerMetadata(
            completed_jobs=merged,
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
            glm53_manifest_events=(  # [glm53-kv-offload-restore]
                # getattr: tolerate a legacy meta without the field (Codex
                # OFFLOAD2 finding 7).
                list(getattr(self, "glm53_manifest_events", ()))
                + list(getattr(other, "glm53_manifest_events", ()))
            ),
        )
"""

SITES_COMMON = (
    ("M1 worker-meta event field", MARK_M1, ANCHOR_M1, PATCHED_M1),
    ("M2 aggregate concatenates events", MARK_M2, ANCHOR_M2, PATCHED_M2),
)


# ===========================================================================
# File 2: offloading/scheduler.py
# ===========================================================================
MARK_R1 = "# [glm53-kv-offload-restore] scheduler-side disk-restore state\n"

ANCHOR_R1 = """class OffloadingConnectorScheduler:
"""

PATCHED_R1 = RESTORE_SCHED_SRC + ANCHOR_R1

# R2: route the enabled-restore lookup to the disk lookup (one line; every
# piece of stage-1 bookkeeping around it is preserved -- finding 12).
MARK_R2 = "            num_hit_tokens = self._glm53_restore.lookup(self, req_status)  # [glm53-kv-offload-restore]\n"

ANCHOR_R2 = """            num_hit_tokens = self._lookup(req_status)
"""

PATCHED_R2 = """            # Restore reads the DISK tier only: the CPU staging tier is a
            # bounce buffer, never a restore source; self._lookup (the
            # manager lookup) stays unused under this fork.
            num_hit_tokens = self._glm53_restore.lookup(self, req_status)  # [glm53-kv-offload-restore]
"""

MARK_R3 = "        self._glm53_restore = Glm53RestoreState(  # [glm53-kv-offload-restore]\n"

ANCHOR_R3 = """        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)
"""

PATCHED_R3 = """        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)
        self._glm53_restore = Glm53RestoreState(  # [glm53-kv-offload-restore]
            self.config, vllm_config, logger
        )
"""

# R4: update_state_after_alloc -- disk hits skip the manager entirely.
MARK_R4 = "        _glm53_boundary = self._glm53_restore.take_pending_hit(  # [glm53-kv-offload-restore]\n"

ANCHOR_R4 = """        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )
"""

PATCHED_R4 = """        _glm53_boundary = self._glm53_restore.take_pending_hit(  # [glm53-kv-offload-restore]
            request.request_id
        )
        if _glm53_boundary is not None:
            # Disk-restore load job: keys/hashes only (the worker derives
            # its own paths); no manager staging involvement, so
            # keys_to_load empties -- TransferJobStatus.keys becomes empty
            # and complete_load(set()) is a no-op by construction.
            keys_to_load = []
            src_spec = self._glm53_restore.build_load_spec(
                request, req_status, _glm53_boundary, num_cached_tokens
            )
            # T2 (finding 10): END-EXCLUSIVE restored token count; persists
            # for the request's lifetime so the restored boundary stays
            # reachable in every later cache_blocks of this request.
            request.glm53_restored_boundary = _glm53_boundary
        else:
            src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )
"""

# R5: register the single-flight gate on the created load job.
MARK_R5 = "        if _glm53_boundary is not None:  # [glm53-kv-offload-restore] gate\n"

ANCHOR_R5 = """        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )
"""

PATCHED_R5 = """        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )
        if _glm53_boundary is not None:  # [glm53-kv-offload-restore] gate
            self._glm53_restore.job_started(load_job_id)
"""

# R6: consume worker manifest events.
MARK_R6 = "        self._glm53_restore.consume_events(  # [glm53-kv-offload-restore]\n"

ANCHOR_R6 = """        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
"""

PATCHED_R6 = """        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        self._glm53_restore.consume_events(  # [glm53-kv-offload-restore]
            getattr(meta, "glm53_manifest_events", None) or ()
        )
"""

# R7: completion clears the gate.
MARK_R7 = "                self._glm53_restore.job_done(job_id)  # [glm53-kv-offload-restore]\n"

ANCHOR_R7 = """            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._chunks_being_loaded:
                    self._chunks_being_loaded.difference_update(job_status.keys)
"""

PATCHED_R7 = """            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._chunks_being_loaded:
                    self._chunks_being_loaded.difference_update(job_status.keys)
                self._glm53_restore.job_done(job_id)  # [glm53-kv-offload-restore]
"""

# R8: reset clears the gate + pending hits.
MARK_R8 = "        self._glm53_restore.reset()  # [glm53-kv-offload-restore]\n"

ANCHOR_R8 = """        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
"""

PATCHED_R8 = """        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        self._glm53_restore.reset()  # [glm53-kv-offload-restore]
"""

# R9: preemption flush must not assert on a disk LOAD job (finding 3).
MARK_R9 = "            if not self._jobs[any_jid].is_store:  # [glm53-kv-offload-restore]\n"

ANCHOR_R9 = """            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)
"""

PATCHED_R9 = """            any_jid = next(iter(req_status.transfer_jobs))
            if not self._jobs[any_jid].is_store:  # [glm53-kv-offload-restore]
                if self._glm53_restore.is_disk_job(any_jid):
                    # OUR disk-restore load job: not flushable through the
                    # store path. The synchronous loader already ran (or
                    # acks next step) and a late ack after reset is fenced
                    # by _stale_job_threshold; clear the single-flight gate
                    # so the re-admitted request re-runs the lookup (T5).
                    self._glm53_restore.job_done(any_jid)
                    continue
                # Any OTHER load-job kind keeps the stock invariant crash
                # semantics (Codex confirm pass, new finding 1: never
                # misclassify a CPU-tier load as ours).
                raise AssertionError(
                    f"non-store, non-disk job {any_jid} on a preempted request"
                )
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)
"""

# R10: manifest candidates on zero-cow layouts (disposition S1). Anchors the
# store overlay's injected meta-builder text.
MARK_R10 = "        if cow_groups:  # [glm53-kv-offload-restore] S1\n"

ANCHOR_R10 = """    manifests = []
    if (
        cow_groups
        and full_groups
        and len(set(tokens_per_chunk.values())) == 1
        and len(set(hashes_per_chunk.values())) == 1
    ):
        stride = next(iter(hashes_per_chunk.values()))
        tokens = next(iter(tokens_per_chunk.values()))
        for k in sorted(mamba_chunk_idxs):
"""

PATCHED_R10 = """    manifests = []
    if (
        full_groups
        and len(set(tokens_per_chunk.values())) == 1
        and len(set(hashes_per_chunk.values())) == 1
    ):
        stride = next(iter(hashes_per_chunk.values()))
        tokens = next(iter(tokens_per_chunk.values()))
        # Boundary candidates: mamba chunk idxs on hybrid layouts (byte-
        # identical to stage 1); on layouts with NO recurrent groups (the
        # g0-isolating probe) every full-group chunk in this job is a
        # candidate -- without this a pure full-attention layout would never
        # publish a manifest and the disk lookup would answer from nothing.
        if cow_groups:  # [glm53-kv-offload-restore] S1
            _glm53_cand_idxs = sorted(mamba_chunk_idxs)
        else:
            _glm53_cand_idxs = sorted(
                {c for _h, g, c, _t in keys if g in full_groups}
            )
        for k in _glm53_cand_idxs:
"""

# R11: _build_store_jobs asserts every in-flight job of a request is a store
# (fixture scheduler :1401-1403). A request whose disk LOAD job is still
# registered (aborted mid-load, or worker output delayed a step under async
# scheduling) can reach store-job construction and trip the assert (Codex
# OFFLOAD2-REVIEW finding 1, BLOCKER). Defer store creation for that request
# instead: the stock invariant IS "stores only when no load is pending", and
# next_stored_chunk_idx is untouched so nothing is lost -- the next step
# reconsiders after the load completes.
MARK_R11 = "            if req_status.transfer_jobs and not self._jobs[  # [glm53-kv-offload-restore] R11\n"

ANCHOR_R11 = """            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req = req_status.req

            if req.status is RequestStatus.FINISHED_ABORTED:
"""

PATCHED_R11 = """            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            if req_status.transfer_jobs and not self._jobs[  # [glm53-kv-offload-restore] R11
                next(iter(req_status.transfer_jobs))
            ].is_store and self._glm53_restore.is_disk_job(
                next(iter(req_status.transfer_jobs))
            ):
                # OUR disk-restore LOAD job is still registered for this
                # request (per the connector invariant, the only job then):
                # defer store-job creation to a later step -- reaching the
                # is_store assert below would crash the scheduler, and the
                # stock invariant is exactly "stores only when no load is
                # pending". next_stored_chunk_idx is untouched: nothing is
                # lost. Any OTHER load-job kind falls through to the stock
                # assert (confirm pass, new finding 1).
                continue
            req = req_status.req

            if req.status is RequestStatus.FINISHED_ABORTED:
"""

SITES_SCHED = (
    ("R11 store-jobs load-defer guard", MARK_R11, ANCHOR_R11, PATCHED_R11),
    ("R1 restore state class", MARK_R1, ANCHOR_R1, PATCHED_R1),
    ("R2 disk lookup route", MARK_R2, ANCHOR_R2, PATCHED_R2),
    ("R3 restore state init", MARK_R3, ANCHOR_R3, PATCHED_R3),
    ("R4 disk load spec", MARK_R4, ANCHOR_R4, PATCHED_R4),
    ("R5 single-flight gate", MARK_R5, ANCHOR_R5, PATCHED_R5),
    ("R6 consume events", MARK_R6, ANCHOR_R6, PATCHED_R6),
    ("R7 completion clears gate", MARK_R7, ANCHOR_R7, PATCHED_R7),
    ("R8 reset clears gate", MARK_R8, ANCHOR_R8, PATCHED_R8),
    ("R9 preemption load-job guard", MARK_R9, ANCHOR_R9, PATCHED_R9),
    ("R10 zero-cow manifest candidates", MARK_R10, ANCHOR_R10, PATCHED_R10),
)


# ===========================================================================
# File 3: offloading/worker.py
# ===========================================================================
MARK_W1 = "# [glm53-kv-offload-restore] worker-side synchronous disk loader\n"

ANCHOR_W1 = """class OffloadingConnectorWorker:
"""

PATCHED_W1 = RESTORE_WORKER_SRC + ANCHOR_W1

# W2: per-worker restore state (anchors the store overlay's patched text).
MARK_W2 = "        self._glm53_gpu_caches = None  # [glm53-kv-offload-restore]\n"

ANCHOR_W2 = """        self._glm53_store_writer = None  # [glm53-kv-offload-store] lazy
        self._glm53_job_meta: dict[int, tuple] = {}
"""

PATCHED_W2 = """        self._glm53_store_writer = None  # [glm53-kv-offload-store] lazy
        self._glm53_job_meta: dict[int, tuple] = {}
        self._glm53_gpu_caches = None  # [glm53-kv-offload-restore]
        self._glm53_disk_done: list[int] = []
        self._glm53_load_error_ids: set[int] = set()
"""

# W3: stash the GPU-side canonical caches (single chokepoint).
MARK_W3 = "        self._glm53_gpu_caches = kv_caches  # [glm53-kv-offload-restore]\n"

ANCHOR_W3 = """    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self.worker = self.spec.get_worker(kv_caches)
"""

PATCHED_W3 = """    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self._glm53_gpu_caches = kv_caches  # [glm53-kv-offload-restore]
        self.worker = self.spec.get_worker(kv_caches)
"""

# W4: disk load jobs bypass worker.submit_load.
MARK_W4 = "            if (  # [glm53-kv-offload-restore] disk jobs bypass submit_load\n"

ANCHOR_W4 = """        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            success = self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec)
            assert success
"""

PATCHED_W4 = """        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            if (  # [glm53-kv-offload-restore] disk jobs bypass submit_load
                # and its success asserts: synchronous pread+verify+scatter
                # with an explicit terminal result either way (T4).
                isinstance(entry.src_spec, dict)
                and entry.src_spec.get("glm53_disk_load")
            ):
                _glm53_run_disk_load(self, job_id, entry.src_spec, entry.dst_spec)
                continue
            success = self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec)
            assert success
"""

# W5: drain finished disk loads in get_finished.
MARK_W5 = "        for job_id in _glm53_drain_disk_loads(self):  # [glm53-kv-offload-restore]\n"

ANCHOR_W5 = """        return set(), finished_recving
"""

PATCHED_W5 = """        for job_id in _glm53_drain_disk_loads(self):  # [glm53-kv-offload-restore]
            self._connector_worker_meta.mark_completed(job_id)
            _glm53_req_id = self._load_jobs.pop(job_id, None)
            if _glm53_req_id is not None:
                finished_recving.add(_glm53_req_id)
        return set(), finished_recving
"""

# W6: worker meta flows when events are pending, completions or not.
MARK_W6 = "        _glm53_attach_manifest_events(self)  # [glm53-kv-offload-restore]\n"

ANCHOR_W6 = """    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        \"\"\"Return completed transfer job IDs since the last call.\"\"\"
        if not self._connector_worker_meta.completed_jobs:
            return None
"""

PATCHED_W6 = """    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        \"\"\"Return completed transfer job IDs since the last call.\"\"\"
        _glm53_attach_manifest_events(self)  # [glm53-kv-offload-restore]
        if not self._connector_worker_meta.completed_jobs and not getattr(
            self._connector_worker_meta, "glm53_manifest_events", None
        ):
            return None
"""

# W7-W11: store-writer event emission + validated dedup (anchors the store
# overlay's injected writer text).
MARK_W7 = "        self._glm53_manifest_events: list = []  # [glm53-kv-offload-restore]\n"

ANCHOR_W7 = """        # In-memory manifest index for inline K-boundary retention:
        # boundary_hash -> {"parent": hash|None, "boundary": int,
        #                   "mamba_keys": [(hash, gidx)...]}
        self._manifest_index: dict = {}
"""

PATCHED_W7 = """        # In-memory manifest index for inline K-boundary retention:
        # boundary_hash -> {"parent": hash|None, "boundary": int,
        #                   "mamba_keys": [(hash, gidx)...]}
        self._manifest_index: dict = {}
        # Manifest availability events for the worker->scheduler channel
        # (drained by _glm53_attach_manifest_events under the lock).
        self._glm53_manifest_events: list = []  # [glm53-kv-offload-restore]
        self._glm53_boot_id = f"{_glm53_time.time():.0f}-{id(self):x}"
"""

MARK_W8 = "                self._glm53_manifest_events.append(  # [glm53-kv-offload-restore] publish\n"

ANCHOR_W8 = """        with self._lock:
            self._manifest_index[boundary_hash] = {
                "parent": chunk_hashes[-2] if len(chunk_hashes) > 1 else None,
                "boundary": cand["boundary_token_index"],
                "mamba_keys": [(boundary_hash, g) for g in cow_groups],
            }
"""

PATCHED_W8 = """        with self._lock:
            self._manifest_index[boundary_hash] = {
                "parent": chunk_hashes[-2] if len(chunk_hashes) > 1 else None,
                "boundary": cand["boundary_token_index"],
                "mamba_keys": [(boundary_hash, g) for g in cow_groups],
            }
            # Cap applies to "+" ONLY: a dropped "+" is a missed hit
            # (fail closed); dropping "-"/"F" would be fail-OPEN (stale
            # availability / lost failure knowledge), so those always ride.
            if len(self._glm53_manifest_events) < 100000:  # "+" cap
                self._glm53_manifest_events.append(  # [glm53-kv-offload-restore] publish
                    (
                        "+",
                        self._rank,
                        self._namespace_hash,
                        boundary_hash,
                        cand["boundary_token_index"],
                        self._glm53_boot_id,
                    )
                )
"""

MARK_W9 = "                    self._glm53_manifest_events.append(  # [glm53-kv-offload-restore] supersede\n"

ANCHOR_W9 = """            with self._lock:
                self._manifest_index.pop(boundary_hash, None)
"""

PATCHED_W9 = """            with self._lock:
                self._manifest_index.pop(boundary_hash, None)
                if True:  # "-" always rides (fail-closed direction)
                    self._glm53_manifest_events.append(  # [glm53-kv-offload-restore] supersede
                        (
                            "-",
                            self._rank,
                            self._namespace_hash,
                            boundary_hash,
                            entry["boundary"],
                            self._glm53_boot_id,
                        )
                    )
"""

MARK_W10 = "                self._glm53_manifest_events.append(  # [glm53-kv-offload-restore] write failure\n"

ANCHOR_W10 = """        self._files_dropped += 1
        with self._lock:
            self._failed_keys.add((hash_hex, group_idx))
"""

PATCHED_W10 = """        self._files_dropped += 1
        with self._lock:
            self._failed_keys.add((hash_hex, group_idx))
            # The per-job store failure bit, on the wire: the scheduler's
            # disk lookup denies boundaries whose keys failed on ANY rank.
            if True:  # "F" always rides (fail-closed direction)
                self._glm53_manifest_events.append(  # [glm53-kv-offload-restore] write failure
                    (
                        "F",
                        getattr(self, "_rank", -1),
                        getattr(self, "_namespace_hash", ""),
                        hash_hex,
                        group_idx,
                        self._glm53_boot_id,
                    )
                )
"""

# W11: validated dedup branch + the validator/drain methods (finding 9).
MARK_W11 = "                if self._glm53_validate_manifest_file(  # [glm53-kv-offload-restore]\n"

ANCHOR_W11 = """            mpath = self._manifest_path(boundary_hash)
            if _os.path.exists(mpath):
                self._register_manifest(cand, cow_groups)
                continue
"""

PATCHED_W11 = """            mpath = self._manifest_path(boundary_hash)
            if _os.path.exists(mpath):
                # Validate before registering or ANNOUNCING (Codex OFFLOAD2
                # finding 9): a pre-existing manifest must parse, match
                # namespace + boundary + chain, and every referenced chunk
                # header must verify; retention applies on this branch too.
                # Invalid => unlink and fall through to a fresh publish.
                if self._glm53_validate_manifest_file(  # [glm53-kv-offload-restore]
                    mpath, boundary_hash, cand, cow_groups, full_groups
                ):
                    self._register_manifest(cand, cow_groups)
                    self._apply_retention()
                    continue
                try:
                    _os.unlink(mpath)
                except OSError:
                    pass
"""

MARK_W12 = "    # [glm53-kv-offload-restore] manifest validation + event drain\n"

ANCHOR_W12 = """    # ---- manifests ------------------------------------------------------
    def _finish_job(self, job) -> None:
"""

PATCHED_W12 = """    # [glm53-kv-offload-restore] manifest validation + event drain
    def drain_manifest_events(self):
        with self._lock:
            events = self._glm53_manifest_events
            self._glm53_manifest_events = []
        return events

    def _glm53_expected_chunk_shape(self, gidx):
        \"\"\"(spec_kind, block_size, segment_table, payload_len) this rank
        expects for one chunk of group ``gidx`` -- the writer's own gather
        derivation, JSON-normalized for header comparison.\"\"\"
        refs = self._refs_per_group[gidx]
        group = self._kv_groups[gidx]
        names = list(group.layer_names)
        segs = []
        off = 0
        for i, ref in enumerate(refs):
            layer = names[i] if i < len(names) else f"ref{i}"
            segs.extend(
                self._layer_segments(group, layer, off, ref.page_size_bytes)
            )
            off += ref.page_size_bytes
        inner = _glm53_unwrap_kv_spec(group.kv_cache_spec)
        return (
            type(inner).__name__,
            getattr(inner, "block_size", None),
            _glm53_json.loads(_glm53_json.dumps(segs)),
            off,
        )

    def _glm53_chunk_verifies(self, hash_hex, gidx, expected) -> bool:
        \"\"\"Full chunk verification: identity + spec/rank/world + segment
        table + payload length (checked BEFORE the payload read, which
        bounds it) + full payload CRC.\"\"\"
        kind, bs, segs, plen = expected
        fp = self._file_path(hash_hex, gidx)
        try:
            fh = glm53_read_chunk_header(fp)
        except (OSError, ValueError):
            return False
        if (
            fh.get("namespace_hash") != self._namespace_hash
            or fh.get("hash") != hash_hex
            or fh.get("group_idx") != gidx
            or fh.get("spec_kind") != kind
            or fh.get("block_size_tokens") != bs
            or fh.get("tp_rank") != self._rank
            or fh.get("tp_world") != self._world
            or fh.get("payload_len") != plen
            or fh.get("segment_table") != segs
        ):
            return False
        try:
            glm53_read_chunk_header(fp, verify_payload=True)
        except (OSError, ValueError):
            return False
        return True

    def _glm53_validate_manifest_file(
        self, mpath, boundary_hash, cand, cow_groups, full_groups
    ) -> bool:
        \"\"\"Full validation of a pre-existing manifest before it is
        registered/announced (Codex OFFLOAD2-REVIEW f4 + confirm pass):
        parse, identity, chain equality, and full chunk verification
        (identity + spec/rank/world + segment table + bounded payload CRC)
        for EVERY referenced chunk on this rank.\"\"\"
        try:
            with open(mpath, encoding="utf-8") as f:
                man = _glm53_json.load(f)
        except (OSError, ValueError):
            return False
        if (
            man.get("format_version") != _GLM53_KVO_FORMAT_VERSION
            or man.get("namespace_hash") != self._namespace_hash
            or man.get("boundary_token_index") != cand["boundary_token_index"]
            or man.get("chunk_hashes") != cand["chunk_hashes"]
        ):
            return False
        with self._lock:
            failed = set(self._failed_keys)
        shapes = {}
        for gidx in cow_groups:
            if (boundary_hash, gidx) in failed:
                return False
            if gidx not in shapes:
                shapes[gidx] = self._glm53_expected_chunk_shape(gidx)
            if not self._glm53_chunk_verifies(
                boundary_hash, gidx, shapes[gidx]
            ):
                return False
        for gidx in full_groups:
            if gidx not in shapes:
                shapes[gidx] = self._glm53_expected_chunk_shape(gidx)
            for h in cand["chunk_hashes"]:
                if (h, gidx) in failed:
                    return False
                if not self._glm53_chunk_verifies(h, gidx, shapes[gidx]):
                    return False
        return True

    # ---- manifests ------------------------------------------------------
    def _finish_job(self, job) -> None:
"""

# W13: the store writer's existing-chunk dedup must verify the PAYLOAD, not
# only the header (Codex OFFLOAD2-REVIEW finding 4: a stale file with a valid
# unchanged header but torn payload must be rewritten, not deduped).
MARK_W13 = "                # [glm53-kv-offload-restore] dedup: bounded payload verification\n"

ANCHOR_W13 = """            try:
                h = glm53_read_chunk_header(path)
                if (
                    h.get("namespace_hash") == self._namespace_hash
                    and h.get("hash") == hash_hex
                    and h.get("group_idx") == group_idx
                ):
"""

PATCHED_W13 = """            try:
                # [glm53-kv-offload-restore] dedup: bounded payload verification
                # (review f4 + confirm N3, ordering per the final confirm):
                # identity and length equality FIRST -- length equality with
                # THIS job's payload caps the verifying read -- then a full
                # CRC pass; a stale file with a valid header but torn payload
                # is rewritten, never deduped.
                h = glm53_read_chunk_header(path)
                if (
                    h.get("payload_len") == len(payload)
                    and h.get("namespace_hash") == self._namespace_hash
                    and h.get("hash") == hash_hex
                    and h.get("group_idx") == group_idx
                    and glm53_read_chunk_header(path, verify_payload=True)
                ):
"""

SITES_WORKER = (
    ("W1 disk loader", MARK_W1, ANCHOR_W1, PATCHED_W1),
    ("W2 worker restore state", MARK_W2, ANCHOR_W2, PATCHED_W2),
    ("W3 GPU cache stash", MARK_W3, ANCHOR_W3, PATCHED_W3),
    ("W4 disk load dispatch", MARK_W4, ANCHOR_W4, PATCHED_W4),
    ("W5 disk load drain", MARK_W5, ANCHOR_W5, PATCHED_W5),
    ("W6 event-only meta", MARK_W6, ANCHOR_W6, PATCHED_W6),
    ("W7 writer event state", MARK_W7, ANCHOR_W7, PATCHED_W7),
    ("W8 publish event", MARK_W8, ANCHOR_W8, PATCHED_W8),
    ("W9 supersede event", MARK_W9, ANCHOR_W9, PATCHED_W9),
    ("W10 write-failure event", MARK_W10, ANCHOR_W10, PATCHED_W10),
    ("W11 validated dedup", MARK_W11, ANCHOR_W11, PATCHED_W11),
    ("W12 validator + drain methods", MARK_W12, ANCHOR_W12, PATCHED_W12),
    ("W13 payload-verifying chunk dedup", MARK_W13, ANCHOR_W13, PATCHED_W13),
)


# ===========================================================================
# File 4: offloading_connector.py (facade)
# ===========================================================================
MARK_F1 = "    def get_block_ids_with_load_errors(self) -> set[int]:  # [glm53-kv-offload-restore]\n"

ANCHOR_F1 = """    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        if self.connector_worker is not None:
            return self.connector_worker.build_connector_worker_meta()
        return None
"""

PATCHED_F1 = """    def get_block_ids_with_load_errors(self) -> set[int]:  # [glm53-kv-offload-restore]
        # Drained per step by the model-runner mixin right after
        # get_finished(); the executor unions across ranks into
        # KVConnectorOutput.invalid_block_ids (T4). Empty without a worker
        # or without any disk-restore failure.
        worker = self.connector_worker
        if worker is None:
            return set()
        errors = getattr(worker, "_glm53_load_error_ids", None)
        if not errors:
            return set()
        drained = set(errors)
        errors.clear()
        return drained

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        if self.connector_worker is not None:
            return self.connector_worker.build_connector_worker_meta()
        return None
"""

SITES_FACADE = (("F1 load-error drain", MARK_F1, ANCHOR_F1, PATCHED_F1),)


# ===========================================================================
# File 5: v1/core/single_type_kv_cache_manager.py (T2)
# ===========================================================================
MARK_T1 = "        _glm53_rb = getattr(request, \"glm53_restored_boundary\", 0)  # [glm53-kv-offload-restore]\n"

ANCHOR_T1 = """        reachable_boundaries = [request.num_prompt_tokens - 1]
        if request.shared_prefix_boundary:
            reachable_boundaries.append(request.shared_prefix_boundary)
"""

PATCHED_T1 = """        reachable_boundaries = [request.num_prompt_tokens - 1]
        if request.shared_prefix_boundary:
            reachable_boundaries.append(request.shared_prefix_boundary)
        # T2 (kv-offload restore): a disk-restored boundary must stay
        # reachable in every later cache_blocks of this request, or a sparse
        # retention mask could silently drop its re-entry. The value is the
        # END-EXCLUSIVE restored token count (a multiple of the restore
        # chunk size, e.g. 7168 keeps the mamba state block ending at token
        # 7168). Dense (full-attention) masks return None and ignore it.
        _glm53_rb = getattr(request, "glm53_restored_boundary", 0)  # [glm53-kv-offload-restore]
        if _glm53_rb:
            reachable_boundaries.append(_glm53_rb)
"""

SITES_SINGLE_TYPE = (("T1 restored reachable boundary", MARK_T1, ANCHOR_T1, PATCHED_T1),)


TARGETS = (
    ("offloading/common.py", TARGET_COMMON, SITES_COMMON),
    ("offloading/scheduler.py", TARGET_SCHED, SITES_SCHED),
    ("offloading/worker.py", TARGET_WORKER, SITES_WORKER),
    ("offloading_connector.py", TARGET_FACADE, SITES_FACADE),
    ("single_type_kv_cache_manager.py", TARGET_SINGLE_TYPE, SITES_SINGLE_TYPE),
)

# The scheduler/worker anchors pin the scope + store overlays' output; refuse
# to run against an unpatched tree with a clear message instead of drift.
SCOPE_MARK = "[glm53-kv-offload-scope]"
STORE_MARK = "[glm53-kv-offload-store]"


def verified_state(text: str, sites) -> bool:
    return all(
        text.count(mark) == 1
        and text.count(patched) == 1
        and text.count(anchor) == patched.count(anchor)
        for _name, mark, anchor, patched in sites
    )


def prepare(source: str, sites, label: str) -> tuple[str, str]:
    marks = sum(source.count(mark) for _n, mark, _a, _p in sites)
    if marks:
        # This overlay is the LAST of the kv-offload trio -- nothing edits
        # inside its patched regions afterwards, so the already-present path
        # can and MUST verify the full patched blocks, not just the marks
        # (Codex OFFLOAD2-REVIEW finding 10: a tampered file that kept its
        # marker lines would otherwise pass silently).
        if marks != len(sites) or any(
            source.count(mark) != 1 for _n, mark, _a, _p in sites
        ) or not verified_state(source, sites):
            raise ValueError(
                f"partial/inconsistent/tampered kv-offload-restore patch in "
                f"{label} (marks={marks}, expected {len(sites)}) -- refusing "
                "to touch the file"
            )
        return source, "already present"
    for name, _mark, anchor, _patched in sites:
        n = source.count(anchor)
        if n != 1:
            raise ValueError(
                f"pinned kv-offload-restore anchor '{name}' drifted in {label} "
                f"(found {n}, expected 1)"
            )
    out = source
    for _name, _mark, anchor, patched in sites:
        out = out.replace(anchor, patched, 1)
    if not verified_state(out, sites):
        raise ValueError(
            f"kv-offload-restore post-patch verification failed in {label}"
        )
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kv-offload-restore.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(f"{target.stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]

    if "--check-injected" in argv[1:]:
        # Host-side gate mode: compile every injected source standalone (no
        # target access) so a truncated/corrupted string literal inside this
        # file fails BEFORE the launcher stops a healthy pair.
        for name, src in (
            ("scheduler restore state", RESTORE_SCHED_SRC),
            ("worker disk loader", RESTORE_WORKER_SRC),
        ):
            compile(src, f"<glm53-kv-offload-restore {name}>", "exec")
        print("patch_kv_offload_restore_g0.py: injected sources compile OK")
        return 0

    sched_source = TARGET_SCHED.read_text() if TARGET_SCHED.is_file() else ""
    for required_mark, producer in (
        (SCOPE_MARK, "patch_kv_offload_scope.py"),
        (STORE_MARK, "patch_kv_offload_store_local.py"),
    ):
        if required_mark not in sched_source:
            raise SystemExit(
                f"kv-offload-restore preflight failed: scheduler.py is not "
                f"{required_mark}-patched -- run {producer} first "
                "(GLM53_OVERLAY_ORDER pins this)"
            )

    prepared: list[tuple[Path, str, str, str]] = []
    for label, target, sites in TARGETS:
        if not target.is_file():
            raise SystemExit(f"missing {target}")
        source = target.read_text()
        try:
            patched, action = prepare(source, sites, label)
        except ValueError as exc:
            raise SystemExit(
                f"kv-offload-restore preflight failed: {exc}"
            ) from exc
        compile(patched, str(target), "exec")
        prepared.append((target, source, patched, action))
        if preflight_only:
            print(f"{target.name}: kv-offload-restore preflight OK ({action})")

    if preflight_only:
        return 0

    for target, source, patched, action in prepared:
        if patched != source:
            replace_file(target, patched)
            clear_pyc(target)
    print(
        f"kv-offload-restore {prepared[0][3]} "
        f"({ENV_RESTORE}={os.environ.get(ENV_RESTORE, '0 (unset)')!r})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
