#!/usr/bin/env python3
"""Store-only worker-local disk KV tier with chunk headers (GLM53_KV_OFFLOAD).

Stage 1 of the disk KV-offload plan (PLAN-KV-OFFLOAD.md §11): the tier WRITES;
nothing reads it back (restore stays off behind GLM53_KV_OFFLOAD_RESTORE=0).

Why worker-local (plan risk R0)
-------------------------------
``TieringOffloadingSpec.get_manager()`` builds the CPU primary tier AND every
secondary tier on the SCHEDULER side and hands ``FileSystemTierManager`` the
scheduler node's staging memoryview. On this 2-node TP=2 deployment
(``--nnodes 2``) rank 1's staging mmap lives on the other Spark: the
scheduler-side view's rank-1 page slots are never written, so the stock fs
tier would persist rank-0 bytes plus uninitialised garbage as the rank-1 half
of every file — corruption that only surfaces at restore time. The CPU
primary tier itself is multi-node-correct (GPU<->CPU DMA is worker-local
against each node's own mmap), so this overlay keeps the CPU staging path
stock and moves the DISK write to the workers: each rank writes its own bytes
from its own staging mmap to its own local NVMe under a per-rank root.

What this overlay does (three files)
------------------------------------
The stock fs secondary tier is NOT configured at all (Codex advisory
CODEX-OFFLOAD1-PLAN.md, finding 3 / Q1): the launcher's connector JSON is
``TieringOffloadingSpec`` with CPU staging only, and THIS overlay's
worker-local writer is the disk tier. The in-image
``FileSystemTierManager`` stays untouched and unused; stage 2's per-rank
manifest-aggregating lookup replaces it on the read side.

1. ``offloading/common.py`` — ``TransferJob`` gains an optional
   ``glm53_store_meta`` field (versioned picklable dict; rides the existing
   scheduler->worker metadata channel; a plain dict is deliberate — pickled
   custom classes would couple scheduler/worker code versions on the wire).
2. ``offloading/scheduler.py`` (AFTER patch_kv_offload_scope.py — two of the
   anchors below pin that overlay's output) —
   - store jobs collect an ordered (key, group_idx, chunk_idx) list aligned
     with the job's src/dst block order and attach it, plus boundary-manifest
     candidates (chunk-hash chains from ``req.block_hashes``), as
     ``glm53_store_meta``;
   - under ``GLM53_KV_OFFLOAD_RESTORE=0`` (default) the external lookup in
     ``get_num_new_matched_tokens`` is short-circuited to 0 hits: no loads,
     no promotions, no WAITING_FOR_REMOTE_KVS — store-only, zero restore
     machinery active.
3. ``offloading/worker.py`` — the worker-local writer:
   - store-job metadata is captured per job (both the ``prepare_store_kv``
     and the ``handle_preemptions`` flush path);
   - when a store job's GPU->CPU DMA completes, the job is NOT acked to the
     scheduler yet; a small writer pool (2 threads) writes one file per
     (block-hash, group): real (unpadded) bytes only, gathered from the
     worker's own CPU staging tensors via the same CanonicalKVCacheRef layout
     the DMA used, with a §4 chunk header (format v1: magic, versioned JSON
     header with namespace hash, ORIGINAL group index, per-layer segment
     table incl. the KDA conv/temporal split, payload CRC, header CRC);
   - the job is acked only after its writes finish — staging blocks stay
     pinned by the store protocol until every worker acks, so the bytes
     cannot be evicted or rewritten under the disk writer;
   - after a job's payloads are durable (write+fsync+rename+parent-dir
     fsync, unique pid-suffixed temp names), boundary manifests are
     published (existence-verified cumulative references, tmp+rename with
     file+dir fsync) — a crash between group writes leaves orphan payloads,
     never a live half-boundary; a key whose write failed this boot can
     never enter a manifest (failed-key ledger + on-disk header check);
   - K-boundary retention INLINE (plan §7, ``GLM53_KV_OFFLOAD_KEEP_BOUNDARIES``,
     default 2): after each publish, a manifest is superseded (deleted, its
     mamba payloads unlinked) once it is beyond the K most recent boundaries
     of EVERY chain containing it — chains are prefix-related manifest
     chains, so a shared divergence-point boundary survives as long as any
     chain still keeps it; full-attention chunks stay dense (cheap; plan §7).
     Cross-boot leftovers are the GC tool's job (kv_offload_store_gc.py);
   - failure semantics (plan §8): a failed write unlinks its tmp and logs
     (lost store = future reprefill; restore is off, nothing can serve wrong
     bytes); ENOSPC pauses the writer (all further writes dropped, one log);
     the job is acked either way so serving is never blocked on the tier.
     KNOWN stage-1 limit (Codex finding 2, partially rebutted): the ack
     protocol has no per-job failure bit, so the CPU tier will believe a
     failed key stored and not retry it this boot — safe while restore is
     off (a boundary with a missing payload simply never gets a manifest and
     stays invisible); the protocol failure bit lands with stage 2's T4
     invalidation work, where it is load-bearing.

File format v1 (see also overlay/kv_offload_store_gc.py, which reads it)
------------------------------------------------------------------------
``GLM53KV1`` magic (8 B) | u32 header_len | canonical-JSON header |
u32 crc32(header) | payload. The header carries format_version,
namespace_hash, group_idx, spec_kind, layers, block_size_tokens,
n_tokens_valid, tp_rank/tp_world, an explicit per-layer segment table
(``(layer, kind, offset, length)``; two segments (conv/temporal, with shapes
and dtypes) per KDA layer when the spec exposes them), the state-ABI tag +
``num_speculative_tokens`` for mamba groups, boundary_token_index,
payload_len and payload_crc32 (zlib crc32 — stdlib has no crc32c; the header
names its algorithm so a crc32c bump is a format-version change, not silent
corruption). Any field mismatch at read => reject.

The namespace hash (§6) covers model path, vllm version, TP/world, KV dtype,
tokens_per_hash, blocks_per_chunk, per-eligible-group (group_idx, block size,
layer count, real per-layer page sizes), the KDA state-ABI descriptor and
``num_speculative_tokens`` (stage-0 receipt A2: num_spec is baked into the
conv width), and format_version. Any change forks the directory.

Preconditions enforced by the writer (fail-closed: refuse to write, never
write garbage; serving unaffected): ``blocks_per_chunk == 1``, a sane
``parallel.rank``, and the direct (non-canonical) staging layout.

Conventions follow ``overlay/patch_kv_capacity_log.py``: pinned ANCHORs, MARK
sentinels, ``verified_state``, ``prepare``, idempotent, atomic replace, pyc
clear, drift => nonzero. ALL anchors in ALL three files preflight before ANY
write. MUST run AFTER ``patch_kv_offload_scope.py``.

Usage::

    python3 patch_kv_offload_store_local.py              # apply
    python3 patch_kv_offload_store_local.py --preflight  # validate anchors only
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

TAG = "[glm53-kv-offload-store]"
ENV_ENABLE = "GLM53_KV_OFFLOAD"
ENV_RESTORE = "GLM53_KV_OFFLOAD_RESTORE"
ENV_DIR = "GLM53_KV_OFFLOAD_DIR"

MAGIC = b"GLM53KV1"
FORMAT_VERSION = 1
STATE_ABI_VERSION = "v0-kda-r13"


# ---------------------------------------------------------------------------
# Shared mode helpers (exec'd for host tests AND injected into three files).
# ---------------------------------------------------------------------------
MODE_HELPERS_SRC = '''# [glm53-kv-offload-store] mode helpers -- see overlay/patch_kv_offload_store_local.py
_GLM53_KVO_STORE_TAG = "[glm53-kv-offload-store]"


def _glm53_kvo_bool_env(name: str, default: str) -> bool:
    """Exactly "0" or "1"; unset means the given default. "" is a value and
    an error: a typo'd knob must not silently pick a storage mode."""
    import os as _os

    raw = _os.environ.get(name)
    if raw is None:
        raw = default
    if raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"{name} must be exactly 0 or 1 (got: {raw!r})")


def _glm53_kvo_store_local_enabled() -> bool:
    return _glm53_kvo_bool_env("GLM53_KV_OFFLOAD", "0")


def _glm53_kvo_restore_enabled() -> bool:
    return _glm53_kvo_bool_env("GLM53_KV_OFFLOAD_RESTORE", "0")


'''

# ---------------------------------------------------------------------------
# Chunk-header codec + store-meta builder + worker-local writer.
#
# One source string, exec'd below into module scope (tests + the GC tool
# import the codec from THIS overlay module) and injected verbatim into
# offloading/worker.py; the meta-builder part is also injected into
# scheduler.py. Stdlib-only imports: the payload gathering uses only tensor
# methods (`.numpy().tobytes()`), so no torch/numpy import is needed.
# ---------------------------------------------------------------------------
CODEC_SRC = '''# [glm53-kv-offload-store] chunk-header codec (format v1)
import json as _glm53_json
import struct as _glm53_struct
import zlib as _glm53_zlib

_GLM53_KVO_MAGIC = b"GLM53KV1"
_GLM53_KVO_FORMAT_VERSION = 1
_GLM53_KVO_STATE_ABI = "v0-kda-r13"
_GLM53_KVO_MAX_HEADER = 1 << 20


def glm53_encode_chunk(header: dict, payload: bytes) -> bytes:
    """Serialize header+payload into the v1 on-disk chunk format."""
    h = dict(header)
    h["format_version"] = _GLM53_KVO_FORMAT_VERSION
    h["crc_algo"] = "crc32-zlib"
    h["payload_len"] = len(payload)
    h["payload_crc32"] = _glm53_zlib.crc32(payload) & 0xFFFFFFFF
    hjson = _glm53_json.dumps(h, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(hjson) > _GLM53_KVO_MAX_HEADER:
        raise ValueError("chunk header too large")
    hcrc = _glm53_zlib.crc32(hjson) & 0xFFFFFFFF
    return (
        _GLM53_KVO_MAGIC
        + _glm53_struct.pack("<I", len(hjson))
        + hjson
        + _glm53_struct.pack("<I", hcrc)
        + payload
    )


def glm53_read_chunk_header(path: str, verify_payload: bool = False) -> dict:
    """Read+validate a v1 chunk header; raises ValueError on ANY mismatch.

    With verify_payload=True the payload is read fully and CRC-checked.
    """
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != _GLM53_KVO_MAGIC:
            raise ValueError(f"bad magic in {path!r}")
        raw_len = f.read(4)
        if len(raw_len) != 4:
            raise ValueError(f"truncated header length in {path!r}")
        (hlen,) = _glm53_struct.unpack("<I", raw_len)
        if not 0 < hlen <= _GLM53_KVO_MAX_HEADER:
            raise ValueError(f"implausible header length {hlen} in {path!r}")
        hjson = f.read(hlen)
        if len(hjson) != hlen:
            raise ValueError(f"truncated header in {path!r}")
        raw_crc = f.read(4)
        if len(raw_crc) != 4:
            raise ValueError(f"truncated header crc in {path!r}")
        (hcrc,) = _glm53_struct.unpack("<I", raw_crc)
        if (_glm53_zlib.crc32(hjson) & 0xFFFFFFFF) != hcrc:
            raise ValueError(f"header crc mismatch in {path!r}")
        try:
            header = _glm53_json.loads(hjson.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"header not JSON in {path!r}: {exc}") from exc
        if header.get("format_version") != _GLM53_KVO_FORMAT_VERSION:
            raise ValueError(
                f"format_version {header.get('format_version')!r} in {path!r}"
            )
        expected = header.get("payload_len")
        if not isinstance(expected, int) or expected < 0:
            raise ValueError(f"bad payload_len in {path!r}")
        if verify_payload:
            payload = f.read(expected + 1)
            if len(payload) != expected:
                raise ValueError(
                    f"payload length {len(payload)} != {expected} in {path!r}"
                )
            if (_glm53_zlib.crc32(payload) & 0xFFFFFFFF) != header.get(
                "payload_crc32"
            ):
                raise ValueError(f"payload crc mismatch in {path!r}")
        else:
            import os as _os

            actual = _os.fstat(f.fileno()).st_size - 8 - 4 - hlen - 4
            if actual != expected:
                raise ValueError(
                    f"payload length {actual} != {expected} in {path!r}"
                )
    return header


'''

META_BUILDER_SRC = '''# [glm53-kv-offload-store] store-meta builder (scheduler side)
def _glm53_build_store_meta(config, req, entries):
    """Build the picklable per-job store metadata.

    ``entries`` is the ordered list of (offload_key, group_idx, chunk_idx)
    aligned 1:1 with the job's src GPU / dst CPU block order. Returns None
    when store-local mode is off or the job carries nothing.
    """
    if not entries or not _glm53_kvo_store_local_enabled():
        return None
    from vllm.v1.kv_offload.base import get_offload_block_hash

    cow_groups = [
        g.group_idx for g in config.kv_group_configs if g.requires_cow_source
    ]
    full_groups = [
        g.group_idx
        for g in config.kv_group_configs
        if g.sliding_window_size_in_chunks is None and not g.requires_cow_source
    ]
    tokens_per_chunk = {
        g.group_idx: g.tokens_per_chunk for g in config.kv_group_configs
    }
    hashes_per_chunk = {
        g.group_idx: g.hashes_per_chunk for g in config.kv_group_configs
    }

    keys = []
    mamba_chunk_idxs = set()
    for key, group_idx, chunk_idx in entries:
        keys.append(
            (
                bytes(get_offload_block_hash(key)).hex(),
                group_idx,
                chunk_idx,
                tokens_per_chunk[group_idx],
            )
        )
        if group_idx in cow_groups:
            mamba_chunk_idxs.add(chunk_idx)

    # Boundary-manifest candidates need one uniform chunk grid across the
    # eligible groups (true here: every eligible group is block 3584 with
    # blocks_per_chunk 1). On a non-uniform layout payloads still land;
    # manifests are skipped (stage-2 lookup then sees no boundaries).
    manifests = []
    if (
        cow_groups
        and full_groups
        and len(set(tokens_per_chunk.values())) == 1
        and len(set(hashes_per_chunk.values())) == 1
    ):
        stride = next(iter(hashes_per_chunk.values()))
        tokens = next(iter(tokens_per_chunk.values()))
        for k in sorted(mamba_chunk_idxs):
            if (k + 1) * stride > len(req.block_hashes):
                continue
            chunk_hashes = [
                bytes(req.block_hashes[(j + 1) * stride - 1]).hex()
                for j in range(k + 1)
            ]
            manifests.append(
                {
                    "boundary_token_index": (k + 1) * tokens,
                    "chunk_hashes": chunk_hashes,
                }
            )
    return {
        "v": 1,
        "keys": keys,
        "cow_groups": cow_groups,
        "full_groups": full_groups,
        "manifests": manifests,
    }


'''

WRITER_SRC = '''# [glm53-kv-offload-store] worker-local store writer
import hashlib as _glm53_hashlib
import queue as _glm53_queue
import threading as _glm53_threading
import time as _glm53_time


def _glm53_unwrap_kv_spec(spec):
    specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(specs, dict) and specs:
        return next(iter(specs.values()))
    return spec


def _glm53_dtype_itemsize(dtype) -> int | None:
    itemsize = getattr(dtype, "itemsize", None)
    if isinstance(itemsize, int):
        return itemsize
    name = str(dtype)
    sizes = {
        "torch.float32": 4, "torch.float": 4, "torch.bfloat16": 2,
        "torch.float16": 2, "torch.half": 2, "torch.uint8": 1,
        "torch.int8": 1, "torch.float8_e4m3fn": 1,
    }
    return sizes.get(name)


class Glm53LocalStoreWriter:
    """Per-worker disk writer: real bytes from the local staging mmap to the
    local NVMe, one headered file per (block hash, group), plus boundary
    manifests. Fail-closed: any precondition violation disables the writer
    (jobs still ack; nothing is ever written half-right)."""

    def __init__(self, connector_worker, logger):
        self._log = logger
        self._paused_reason = None
        self._disabled_reason = None
        self._pending_jobs = {}
        self._done_queue = _glm53_queue.Queue()
        self._task_queue = _glm53_queue.Queue()
        self._threads = []
        self._lock = _glm53_threading.Lock()
        self._files_written = 0
        self._files_dropped = 0
        # Keys whose write failed THIS boot: a manifest must never reference
        # them even if a stale file appears on disk later (plan §8).
        self._failed_keys: set = set()
        # In-memory manifest index for inline K-boundary retention:
        # boundary_hash -> {"parent": hash|None, "boundary": int,
        #                   "mamba_keys": [(hash, gidx)...]}
        self._manifest_index: dict = {}
        try:
            self._init_layout(connector_worker)
        except Exception as exc:  # noqa: BLE001 - fail closed, keep serving
            self._disabled_reason = f"{type(exc).__name__}: {exc}"
            self._log.error(
                "%s writer DISABLED (store tier inactive, serving unaffected): %s",
                _GLM53_KVO_STORE_TAG,
                self._disabled_reason,
            )

    # ---- layout / namespace -------------------------------------------
    def _init_layout(self, cw) -> None:
        import os as _os

        spec = cw.spec
        worker = cw.worker
        assert worker is not None, "writer initialised before register_kv_caches"
        if spec.blocks_per_chunk != 1:
            raise ValueError(
                f"store-local requires blocks_per_chunk == 1 "
                f"(got {spec.blocks_per_chunk})"
            )
        handler = worker._store_handler
        if getattr(handler, "_canonical_copy_plans", None) is not None:
            raise ValueError("store-local requires the direct staging layout")
        parallel = spec.config.parallel
        world = parallel.world_size
        # Rank routing (plan R0; Codex OFFLOAD1 finding 1): the initialized
        # TP rank of THIS worker process is authoritative -- NEVER a local
        # device index (both one-GPU nodes report device 0). The connector
        # config's parallel.rank must agree; disagreement disables the
        # writer rather than risking cross-rank mixing.
        rank = None
        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank as _glm53_tp_rank,
            )

            rank = int(_glm53_tp_rank())
        except Exception:  # noqa: BLE001 - group not initialized: fall back
            rank = None
        cfg_rank = parallel.rank
        if rank is None:
            rank = cfg_rank
        elif isinstance(cfg_rank, int) and cfg_rank != rank:
            raise ValueError(
                f"TP rank {rank} disagrees with OffloadingParallelConfig.rank "
                f"{cfg_rank} -- refusing to pick a per-rank directory"
            )
        if not (isinstance(rank, int) and 0 <= rank < world):
            raise ValueError(f"implausible parallel rank {rank!r} of {world!r}")
        root = _os.environ.get("GLM53_KV_OFFLOAD_DIR")
        if not root:
            raise ValueError("GLM53_KV_OFFLOAD_DIR is not set")
        keep_raw = _os.environ.get("GLM53_KV_OFFLOAD_KEEP_BOUNDARIES", "2")
        if not keep_raw.isdigit():
            raise ValueError(
                "GLM53_KV_OFFLOAD_KEEP_BOUNDARIES must be a non-negative "
                f"base-10 integer (got: {keep_raw!r})"
            )
        self._keep_boundaries = int(keep_raw)

        self._rank = rank
        self._world = world
        self._cpu_tensors = handler.dst_tensors
        self._refs_per_group = handler.layer_refs_per_group
        kv_groups = cw.kv_cache_config.kv_cache_groups
        if len(self._refs_per_group) != len(kv_groups):
            raise ValueError(
                f"group refs ({len(self._refs_per_group)}) != kv cache groups "
                f"({len(kv_groups)})"
            )
        self._kv_groups = kv_groups
        vllm_config = cw.vllm_config
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        self._num_spec = int(
            getattr(spec_cfg, "num_speculative_tokens", 0) or 0
        )
        import vllm as _vllm

        eligible = []
        for g in spec.config.groups:
            refs = self._refs_per_group[g.group_idx]
            eligible.append(
                {
                    "group_idx": g.group_idx,
                    "tokens_per_block": g.tokens_per_block,
                    "layers": len(g.layer_names),
                    "real_page_sizes": [r.page_size_bytes for r in refs],
                }
            )
        self._eligible_groups = {e["group_idx"] for e in eligible}
        cache_cfg = getattr(vllm_config, "cache_config", None)
        namespace_fields = {
            "format_version": _GLM53_KVO_FORMAT_VERSION,
            "state_abi_version": _GLM53_KVO_STATE_ABI,
            "model": spec.config.model.name,
            "kv_dtype": spec.config.model.dtype,
            "vllm_version": getattr(_vllm, "__version__", "unknown"),
            # Fork/overlay build identity (operator-pinned; empty = unpinned).
            "build_id": _os.environ.get("GLM53_KV_OFFLOAD_BUILD_ID", ""),
            # Prefix-cache hash algo + seed policy fence (plan §6): sha256 is
            # content-stable; anything else (process-seeded builtin) must fork
            # the namespace so cross-boot keys can never collide.
            "prefix_caching_hash_algo": str(
                getattr(cache_cfg, "prefix_caching_hash_algo", "unknown")
            ),
            "tp_size": parallel.tp_size,
            "world_size": world,
            "tokens_per_hash": spec.tokens_per_hash,
            "blocks_per_chunk": spec.blocks_per_chunk,
            "num_speculative_tokens": self._num_spec,
            "groups": eligible,
            "state_abi": self._state_abi_descriptor(),
        }
        canonical = _glm53_json.dumps(
            namespace_fields, sort_keys=True, separators=(",", ":")
        )
        self._namespace_hash = _glm53_hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()[:12]
        # keep_boundaries is retention POLICY, not byte semantics: it rides
        # the on-disk namespace record for operators but must NOT fence the
        # namespace hash (changing K must not orphan the whole store).
        namespace_fields["keep_boundaries"] = self._keep_boundaries
        canonical = _glm53_json.dumps(
            namespace_fields, sort_keys=True, separators=(",", ":")
        )
        safe_model = spec.config.model.name.replace("/", "_")
        self._base = (
            f"{root}/glm53kv_{safe_model}_{self._namespace_hash}_r{rank}"
        )
        _os.makedirs(self._base, exist_ok=True)
        ns_path = f"{root}/glm53kv_{safe_model}_{self._namespace_hash}.json"
        if not _os.path.exists(ns_path):
            tmp = ns_path + f".tmp.r{rank}.{_os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(canonical)
            _os.replace(tmp, ns_path)
        for _ in range(2):
            t = _glm53_threading.Thread(
                target=self._writer_loop,
                name="glm53-kv-offload-store",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        self._log.info(
            "%s writer up: rank %d/%d base=%s eligible=%s",
            _GLM53_KVO_STORE_TAG,
            rank,
            world,
            self._base,
            sorted(self._eligible_groups),
        )

    def _state_abi_descriptor(self):
        abi = {}
        for gidx, group in enumerate(self._kv_groups):
            inner = _glm53_unwrap_kv_spec(group.kv_cache_spec)
            if type(inner).__name__ != "MambaSpec":
                continue
            shapes = getattr(inner, "shapes", None)
            dtypes = getattr(inner, "dtypes", None)
            abi[str(gidx)] = {
                "shapes": [list(s) for s in shapes] if shapes else None,
                "dtypes": [str(d) for d in dtypes] if dtypes else None,
                "num_spec": self._num_spec,
            }
        return abi

    # ---- job intake ----------------------------------------------------
    def submit_job(self, job_id: int, meta: dict, dst_block_ids) -> bool:
        """Queue a completed-DMA store job for disk writing. Returns True when
        the ack must be deferred until the writes finish."""
        if self._disabled_reason is not None:
            return False
        keys = meta.get("keys") or []
        rows = [int(b) for b in dst_block_ids]
        if len(keys) != len(rows):
            self._log.error(
                "%s job %d: %d keys vs %d dst blocks -- dropping write "
                "(lost store only)",
                _GLM53_KVO_STORE_TAG,
                job_id,
                len(keys),
                len(rows),
            )
            return False
        job = {
            "job_id": job_id,
            "meta": meta,
            "remaining": len(keys),
            "written": [],
        }
        with self._lock:
            self._pending_jobs[job_id] = job
        for (hash_hex, group_idx, chunk_idx, n_tokens), row in zip(keys, rows):
            self._task_queue.put(
                (job, hash_hex, int(group_idx), int(chunk_idx), int(n_tokens), row)
            )
        if not keys:
            self._finish_job(job)
        return True

    def drain_done(self):
        done = []
        while True:
            try:
                done.append(self._done_queue.get_nowait())
            except _glm53_queue.Empty:
                return done

    def shutdown(self, timeout: float = 30.0) -> None:
        deadline = _glm53_time.monotonic() + timeout
        while _glm53_time.monotonic() < deadline:
            with self._lock:
                if not self._pending_jobs:
                    return
            _glm53_time.sleep(0.05)

    # ---- write path ----------------------------------------------------
    def _file_path(self, hash_hex: str, group_idx: int) -> str:
        return (
            f"{self._base}/{hash_hex[:3]}/{hash_hex[3:5]}_g{group_idx}"
            f"/{hash_hex}.bin"
        )

    def _manifest_path(self, boundary_hash: str) -> str:
        return f"{self._base}/manifests/{boundary_hash[:3]}/{boundary_hash}.json"

    def _writer_loop(self) -> None:
        while True:
            job, hash_hex, group_idx, chunk_idx, n_tokens, row = (
                self._task_queue.get()
            )
            try:
                self._write_one(job, hash_hex, group_idx, chunk_idx, n_tokens, row)
            except Exception as exc:  # noqa: BLE001 - lost store only
                self._on_write_error(hash_hex, group_idx, exc)
            finally:
                with self._lock:
                    job["remaining"] -= 1
                    finished = job["remaining"] <= 0
                if finished:
                    self._finish_job(job)

    def _on_write_error(self, hash_hex, group_idx, exc) -> None:
        import errno as _errno

        self._files_dropped += 1
        with self._lock:
            self._failed_keys.add((hash_hex, group_idx))
        if isinstance(exc, OSError) and exc.errno == _errno.ENOSPC:
            if self._paused_reason is None:
                self._paused_reason = "ENOSPC"
                self._log.error(
                    "%s store tier PAUSED (ENOSPC): all further writes are "
                    "dropped; serving unaffected",
                    _GLM53_KVO_STORE_TAG,
                )
            return
        self._log.warning(
            "%s write failed for %s g%d (lost store only): %s: %s",
            _GLM53_KVO_STORE_TAG,
            hash_hex[:12],
            group_idx,
            type(exc).__name__,
            exc,
        )

    def _write_one(self, job, hash_hex, group_idx, chunk_idx, n_tokens, row):
        import os as _os

        if self._paused_reason is not None or self._disabled_reason is not None:
            self._files_dropped += 1
            return
        if group_idx not in self._eligible_groups:
            raise ValueError(f"ineligible group {group_idx} reached the writer")
        path = self._file_path(hash_hex, group_idx)
        if _os.path.exists(path):
            _os.utime(path)  # dedup refresh, mtime stays meaningful
            with self._lock:
                job["written"].append((hash_hex, group_idx))
            return

        refs = self._refs_per_group[group_idx]
        group = self._kv_groups[group_idx]
        layer_names = list(group.layer_names)
        segments = []
        parts = []
        offset = 0
        for i, ref in enumerate(refs):
            tensor = self._cpu_tensors[ref.tensor_idx]
            chunk = tensor[row, : ref.page_size_bytes]
            parts.append(chunk.numpy().tobytes())
            layer = layer_names[i] if i < len(layer_names) else f"ref{i}"
            segments.extend(
                self._layer_segments(group, layer, offset, ref.page_size_bytes)
            )
            offset += ref.page_size_bytes
        payload = b"".join(parts)

        inner = _glm53_unwrap_kv_spec(group.kv_cache_spec)
        header = {
            "namespace_hash": self._namespace_hash,
            "group_idx": group_idx,
            "spec_kind": type(inner).__name__,
            "layers": len(refs),
            "block_size_tokens": getattr(inner, "block_size", None),
            "n_tokens_valid": n_tokens,
            "boundary_token_index": (chunk_idx + 1) * n_tokens,
            "tp_rank": self._rank,
            "tp_world": self._world,
            "segment_table": segments,
            "hash": hash_hex,
        }
        if type(inner).__name__ == "MambaSpec":
            header["state_abi_version"] = _GLM53_KVO_STATE_ABI
            header["num_speculative_tokens"] = self._num_spec
        blob = glm53_encode_chunk(header, payload)

        _os.makedirs(_os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.r{self._rank}.{_os.getpid()}"
        try:
            with open(tmp, "wb") as f:
                f.write(blob)
                f.flush()
                _os.fsync(f.fileno())
            _os.replace(tmp, path)
            dfd = _os.open(_os.path.dirname(path), _os.O_RDONLY)
            try:
                _os.fsync(dfd)
            finally:
                _os.close(dfd)
        except BaseException:
            try:
                _os.unlink(tmp)
            except OSError:
                pass
            raise
        self._files_written += 1
        with self._lock:
            job["written"].append((hash_hex, group_idx))

    def _layer_segments(self, group, layer, offset, length):
        inner = _glm53_unwrap_kv_spec(group.kv_cache_spec)
        if type(inner).__name__ == "MambaSpec":
            shapes = getattr(inner, "shapes", None)
            dtypes = getattr(inner, "dtypes", None)
            if shapes and dtypes and len(shapes) == len(dtypes):
                sizes = []
                for shape, dtype in zip(shapes, dtypes):
                    n = 1
                    for d in shape:
                        n *= int(d)
                    itemsize = _glm53_dtype_itemsize(dtype)
                    if itemsize is None:
                        sizes = None
                        break
                    sizes.append((list(shape), str(dtype), n * itemsize))
                if sizes is not None and sum(s[2] for s in sizes) == length:
                    segs = []
                    kinds = ("conv_state", "temporal_state")
                    run = offset
                    for i, (shape, dtype, size) in enumerate(sizes):
                        # Row-major contiguous element strides, DERIVED (not
                        # observed live -- stage-0 receipt R13 caveat rides
                        # in "stride_provenance" so a reader cannot mistake
                        # them for a measured layout).
                        strides = []
                        acc = 1
                        for d in reversed(shape):
                            strides.append(acc)
                            acc *= int(d)
                        strides.reverse()
                        segs.append(
                            {
                                "layer": layer,
                                "kind": kinds[i] if i < 2 else f"state{i}",
                                "shape": shape,
                                "dtype": dtype,
                                "stride": strides,
                                "stride_provenance": "derived-contiguous",
                                "offset": run,
                                "length": size,
                            }
                        )
                        run += size
                    return segs
        return [
            {
                "layer": layer,
                "kind": "raw",
                "shape": [length],
                "dtype": "uint8",
                "stride": [1],
                "stride_provenance": "derived-contiguous",
                "offset": offset,
                "length": length,
            }
        ]

    # ---- manifests ------------------------------------------------------
    def _finish_job(self, job) -> None:
        try:
            if self._paused_reason is None and self._disabled_reason is None:
                self._publish_manifests(job["meta"])
        except Exception as exc:  # noqa: BLE001 - manifests are best-effort
            self._log.warning(
                "%s manifest publish failed (boundary stays invisible): %s: %s",
                _GLM53_KVO_STORE_TAG,
                type(exc).__name__,
                exc,
            )
        finally:
            with self._lock:
                self._pending_jobs.pop(job["job_id"], None)
            self._done_queue.put(job["job_id"])

    def _publish_manifests(self, meta) -> None:
        import os as _os

        cow_groups = meta.get("cow_groups") or []
        full_groups = meta.get("full_groups") or []
        for cand in meta.get("manifests") or []:
            chunk_hashes = cand["chunk_hashes"]
            boundary_hash = chunk_hashes[-1]
            mpath = self._manifest_path(boundary_hash)
            if _os.path.exists(mpath):
                self._register_manifest(cand, cow_groups)
                continue
            # A key that failed to write THIS boot must never be referenced,
            # even if a stale on-disk file would pass the header check.
            with self._lock:
                failed = set(self._failed_keys)
            groups_entry = {}
            complete = True
            for gidx in cow_groups:
                if (boundary_hash, gidx) in failed:
                    complete = False
                    break
                fpath = self._file_path(boundary_hash, gidx)
                try:
                    fheader = glm53_read_chunk_header(fpath)
                except (OSError, ValueError):
                    complete = False
                    break
                groups_entry[str(gidx)] = {
                    "hash": boundary_hash,
                    "payload_len": fheader["payload_len"],
                    "payload_crc32": fheader["payload_crc32"],
                }
            if not complete:
                continue
            for gidx in full_groups:
                for h in chunk_hashes:
                    if (h, gidx) in failed or not _os.path.exists(
                        self._file_path(h, gidx)
                    ):
                        complete = False
                        break
                if not complete:
                    break
            if not complete:
                continue
            manifest = {
                "format_version": _GLM53_KVO_FORMAT_VERSION,
                "namespace_hash": self._namespace_hash,
                "boundary_token_index": cand["boundary_token_index"],
                "chunk_hashes": chunk_hashes,
                "cow_groups": groups_entry,
                "full_groups": {str(g): len(chunk_hashes) for g in full_groups},
                "rank": self._rank,
                "created_at": _glm53_time.time(),
            }
            _os.makedirs(_os.path.dirname(mpath), exist_ok=True)
            tmp = f"{mpath}.tmp.r{self._rank}.{_os.getpid()}"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    _glm53_json.dump(manifest, f, sort_keys=True)
                    f.flush()
                    _os.fsync(f.fileno())
                _os.replace(tmp, mpath)
                dfd = _os.open(_os.path.dirname(mpath), _os.O_RDONLY)
                try:
                    _os.fsync(dfd)
                finally:
                    _os.close(dfd)
            except BaseException:
                try:
                    _os.unlink(tmp)
                except OSError:
                    pass
                raise
            self._register_manifest(cand, cow_groups)
            self._apply_retention()

    # ---- inline K-boundary retention (plan §7; Codex OFFLOAD1 finding 4) --
    def _register_manifest(self, cand, cow_groups) -> None:
        chunk_hashes = cand["chunk_hashes"]
        boundary_hash = chunk_hashes[-1]
        with self._lock:
            self._manifest_index[boundary_hash] = {
                "parent": chunk_hashes[-2] if len(chunk_hashes) > 1 else None,
                "boundary": cand["boundary_token_index"],
                "mamba_keys": [(boundary_hash, g) for g in cow_groups],
            }

    def _apply_retention(self) -> None:
        """Supersede manifests beyond the K most recent boundaries of EVERY
        chain containing them (chains = parent-linked manifest chains; a
        shared divergence-point boundary survives while any chain keeps it).
        Only manifests published/observed this boot are considered; cross-boot
        leftovers belong to the GC tool. Mamba payloads of a superseded
        boundary are unlinked; full-attention chunks stay dense (plan §7)."""
        import os as _os

        if self._keep_boundaries <= 0:
            return
        with self._lock:
            index = {k: dict(v) for k, v in self._manifest_index.items()}
        if not index:
            return
        parents = {h: v["parent"] for h, v in index.items()}
        child_count = {h: 0 for h in index}
        for h, p in parents.items():
            if p in child_count:
                child_count[p] += 1
        leaves = [h for h, c in child_count.items() if c == 0]
        keep: set = set()
        for leaf in leaves:
            node, kept = leaf, 0
            while node is not None and kept < self._keep_boundaries:
                if node in index:
                    keep.add(node)
                    kept += 1
                node = parents.get(node)
        drop = [h for h in index if h not in keep]
        for boundary_hash in drop:
            entry = index[boundary_hash]
            mpath = self._manifest_path(boundary_hash)
            try:
                _os.unlink(mpath)
            except OSError:
                pass
            for hash_hex, gidx in entry["mamba_keys"]:
                try:
                    _os.unlink(self._file_path(hash_hex, gidx))
                except OSError:
                    pass
            with self._lock:
                self._manifest_index.pop(boundary_hash, None)
            self._log.info(
                "%s superseded boundary %d (%s) under keep_boundaries=%d",
                _GLM53_KVO_STORE_TAG,
                entry["boundary"],
                boundary_hash[:12],
                self._keep_boundaries,
            )


def _glm53_get_store_writer(connector_worker):
    writer = connector_worker._glm53_store_writer
    if writer is None:
        writer = Glm53LocalStoreWriter(connector_worker, logger)
        connector_worker._glm53_store_writer = writer
    return writer


def _glm53_drain_finished_disk_writes(connector_worker) -> None:
    writer = connector_worker._glm53_store_writer
    if writer is None:
        return
    for job_id in writer.drain_done():
        connector_worker._connector_worker_meta.mark_completed(job_id)


def _glm53_intercept_store_completion(connector_worker, job_id: int) -> bool:
    """Route a DMA-complete store job into the local disk writer. Returns True
    when the ack is deferred until the disk write finishes."""
    captured = connector_worker._glm53_job_meta.pop(job_id, None)
    if captured is None:
        return False
    meta, dst_spec = captured
    try:
        writer = _glm53_get_store_writer(connector_worker)
        return writer.submit_job(job_id, meta, dst_spec.block_ids)
    except Exception as exc:  # noqa: BLE001 - never block serving on the tier
        logger.error(
            "%s store interception failed (job acked, store lost): %s: %s",
            _GLM53_KVO_STORE_TAG,
            type(exc).__name__,
            exc,
        )
        return False


'''


def load_helpers() -> dict:
    """Exec the shared sources into a namespace (host tests + GC tool)."""
    ns: dict = {}
    src = MODE_HELPERS_SRC + CODEC_SRC + META_BUILDER_SRC
    exec(compile(src, "<glm53-kv-offload-store helpers>", "exec"), ns)
    return ns


_HELPERS = load_helpers()
kvo_store_local_enabled = _HELPERS["_glm53_kvo_store_local_enabled"]
kvo_restore_enabled = _HELPERS["_glm53_kvo_restore_enabled"]
encode_chunk = _HELPERS["glm53_encode_chunk"]
read_chunk_header = _HELPERS["glm53_read_chunk_header"]
build_store_meta = _HELPERS["_glm53_build_store_meta"]


def load_writer_helpers(logger) -> dict:
    """Exec the FULL injected worker source with a host-supplied logger."""
    ns: dict = {"logger": logger}
    src = MODE_HELPERS_SRC + CODEC_SRC + WRITER_SRC
    exec(compile(src, "<glm53-kv-offload-store writer>", "exec"), ns)
    return ns


# ===========================================================================
# File 1: offloading/common.py — TransferJob.glm53_store_meta
# ===========================================================================
MARK_C1 = "    # [glm53-kv-offload-store] optional store-side metadata\n"

ANCHOR_C1 = """    req_id: ReqId
    src_spec: LoadStoreSpec
    dst_spec: LoadStoreSpec
"""

PATCHED_C1 = """    req_id: ReqId
    src_spec: LoadStoreSpec
    dst_spec: LoadStoreSpec
    # [glm53-kv-offload-store] optional store-side metadata
    # (keys/manifests for the worker-local disk writer; None for loads and
    # when GLM53_KV_OFFLOAD store-local mode is off). Plain dict: picklable
    # across the scheduler->worker metadata channel.
    glm53_store_meta: object | None = None
"""

SITES_COMMON = (("TransferJob.glm53_store_meta", MARK_C1, ANCHOR_C1, PATCHED_C1),)


# ===========================================================================
# File 2: offloading/scheduler.py (anchored on the SCOPE-patched text)
# ===========================================================================
MARK_L1 = "# [glm53-kv-offload-store] mode helpers -- see overlay/patch_kv_offload_store_local.py\n"

ANCHOR_L1 = """def get_sliding_window_size_in_chunks(
"""

PATCHED_L1 = MODE_HELPERS_SRC + META_BUILDER_SRC + ANCHOR_L1

MARK_L2 = "        if request.skip_reading_prefix_cache or not _glm53_kvo_restore_enabled():  # [glm53-kv-offload-store]\n"

ANCHOR_L2 = """        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0
        else:
"""

PATCHED_L2 = """        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache or not _glm53_kvo_restore_enabled():  # [glm53-kv-offload-store]
            # Store-only stage 1: never report external hits, so no loads,
            # no promotions and no WAITING_FOR_REMOTE_KVS can occur. Stores
            # (below in build_connector_meta) are unaffected.
            num_hit_tokens = 0
        else:
"""

# L3 anchors on scope-patched S17 output.
MARK_L3 = "            _glm53_store_meta_entries: list = []  # [glm53-kv-offload-store]\n"

ANCHOR_L3 = """            group_sizes = [0] * self.config.num_kv_cache_groups  # [glm53-kv-offload-scope]
            block_indices = [0] * self.config.num_kv_cache_groups
"""

PATCHED_L3 = """            group_sizes = [0] * self.config.num_kv_cache_groups  # [glm53-kv-offload-scope]
            block_indices = [0] * self.config.num_kv_cache_groups
            _glm53_store_meta_entries: list = []  # [glm53-kv-offload-store]
"""

MARK_L4 = "                        _glm53_store_meta_entries.append(  # [glm53-kv-offload-store]\n"

ANCHOR_L4 = """                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
"""

PATCHED_L4 = """                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        _glm53_store_meta_entries.append(  # [glm53-kv-offload-store]
                            (offload_key, group_config.group_idx, chunk_idx)
                        )
                        num_group_blocks += 1
"""

MARK_L5 = "                glm53_store_meta=_glm53_build_store_meta(  # [glm53-kv-offload-store]\n"

ANCHOR_L5 = """            store_jobs[job_id] = TransferJob(
                req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
            )
"""

PATCHED_L5 = """            store_jobs[job_id] = TransferJob(
                req_id=req_id,
                src_spec=src_spec,
                dst_spec=dst_spec,
                glm53_store_meta=_glm53_build_store_meta(  # [glm53-kv-offload-store]
                    self.config, req, _glm53_store_meta_entries
                ),
            )
"""

SITES_SCHED = (
    ("L1 scheduler helpers", MARK_L1, ANCHOR_L1, PATCHED_L1),
    ("L2 restore-off short-circuit", MARK_L2, ANCHOR_L2, PATCHED_L2),
    ("L3 store-meta list init", MARK_L3, ANCHOR_L3, PATCHED_L3),
    ("L4 store-meta entry append", MARK_L4, ANCHOR_L4, PATCHED_L4),
    ("L5 TransferJob store meta", MARK_L5, ANCHOR_L5, PATCHED_L5),
)


# ===========================================================================
# File 3: offloading/worker.py — capture, delay acks, write locally
# ===========================================================================
MARK_W1 = "# [glm53-kv-offload-store] worker-local store writer\n"

ANCHOR_W1 = """class OffloadingConnectorWorker:
"""

PATCHED_W1 = MODE_HELPERS_SRC + CODEC_SRC + WRITER_SRC + ANCHOR_W1

MARK_W2 = "        self._glm53_store_writer = None  # [glm53-kv-offload-store] lazy\n"

ANCHOR_W2 = """        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()
"""

PATCHED_W2 = """        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()
        self._glm53_store_writer = None  # [glm53-kv-offload-store] lazy
        self._glm53_job_meta: dict[int, tuple] = {}
"""

MARK_W3 = "                    _glm53_meta = getattr(entry, \"glm53_store_meta\", None)  # [glm53-kv-offload-store] flush path\n"

ANCHOR_W3 = """                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    if not self._is_store_writer:
                        self._connector_worker_meta.mark_completed(job_id)
                        continue
                    assert isinstance(entry.src_spec, GPULoadStoreSpec)
"""

PATCHED_W3 = """                entry = kv_connector_metadata.store_jobs.pop(job_id, None)
                if entry is not None:
                    if not self._is_store_writer:
                        self._connector_worker_meta.mark_completed(job_id)
                        continue
                    _glm53_meta = getattr(entry, \"glm53_store_meta\", None)  # [glm53-kv-offload-store] flush path
                    if _glm53_meta is not None:
                        self._glm53_job_meta[job_id] = (_glm53_meta, entry.dst_spec)
                    assert isinstance(entry.src_spec, GPULoadStoreSpec)
"""

MARK_W4 = "            _glm53_meta = getattr(entry, \"glm53_store_meta\", None)  # [glm53-kv-offload-store]\n"

ANCHOR_W4 = """        for job_id, entry in metadata.store_jobs.items():
            if not self._is_store_writer:
                # Gate before queueing: no _unsubmitted_store_jobs entry.
                self._connector_worker_meta.mark_completed(job_id)
                continue
"""

PATCHED_W4 = """        for job_id, entry in metadata.store_jobs.items():
            if not self._is_store_writer:
                # Gate before queueing: no _unsubmitted_store_jobs entry.
                self._connector_worker_meta.mark_completed(job_id)
                continue
            _glm53_meta = getattr(entry, \"glm53_store_meta\", None)  # [glm53-kv-offload-store]
            if _glm53_meta is not None:
                self._glm53_job_meta[job_id] = (_glm53_meta, entry.dst_spec)
"""

MARK_W5 = "        _glm53_drain_finished_disk_writes(self)  # [glm53-kv-offload-store]\n"

ANCHOR_W5 = """        assert self.worker is not None
        finished_recving: set[str] = set()
        for transfer_result in self.worker.get_finished():
"""

PATCHED_W5 = """        assert self.worker is not None
        finished_recving: set[str] = set()
        _glm53_drain_finished_disk_writes(self)  # [glm53-kv-offload-store]
        for transfer_result in self.worker.get_finished():
"""

MARK_W5B = "            if _glm53_intercept_store_completion(self, job_id):  # [glm53-kv-offload-store]\n"

ANCHOR_W5B = """            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
"""

PATCHED_W5B = """            if _glm53_intercept_store_completion(self, job_id):  # [glm53-kv-offload-store]
                # DMA done; the ack is deferred until this rank's disk write
                # finishes (staging stays pinned by the store protocol, so
                # the bytes cannot be evicted under the writer).
                continue
            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
"""

MARK_W6 = "        if self._glm53_store_writer is not None:  # [glm53-kv-offload-store]\n"

ANCHOR_W6 = """    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
"""

PATCHED_W6 = """    def shutdown(self) -> None:
        if self._glm53_store_writer is not None:  # [glm53-kv-offload-store]
            self._glm53_store_writer.shutdown()
        self._unsubmitted_store_jobs.clear()
"""

SITES_WORKER = (
    ("W1 writer classes", MARK_W1, ANCHOR_W1, PATCHED_W1),
    ("W2 worker state", MARK_W2, ANCHOR_W2, PATCHED_W2),
    ("W3 flush-path meta capture", MARK_W3, ANCHOR_W3, PATCHED_W3),
    ("W4 store meta capture", MARK_W4, ANCHOR_W4, PATCHED_W4),
    ("W5 drain finished writes", MARK_W5, ANCHOR_W5, PATCHED_W5),
    ("W5B intercept completion", MARK_W5B, ANCHOR_W5B, PATCHED_W5B),
    ("W6 shutdown drain", MARK_W6, ANCHOR_W6, PATCHED_W6),
)


TARGETS = (
    ("offloading/common.py", TARGET_COMMON, SITES_COMMON),
    ("offloading/scheduler.py", TARGET_SCHED, SITES_SCHED),
    ("offloading/worker.py", TARGET_WORKER, SITES_WORKER),
)

# The scheduler anchors L3/L5 pin patch_kv_offload_scope.py output; refuse to
# run against an unscoped tree with a clear message instead of anchor drift.
SCOPE_MARK = "[glm53-kv-offload-scope]"


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
        # Marks-only already-present check (same rationale as the scope
        # patcher: a later overlay may edit inside patched regions; full
        # verification is enforced on this patcher's own output below).
        if marks != len(sites) or any(
            source.count(mark) != 1 for _n, mark, _a, _p in sites
        ):
            raise ValueError(
                f"partial/inconsistent kv-offload-store patch in {label} "
                f"(marks={marks}, expected {len(sites)}) -- refusing to touch "
                "a half-patched file"
            )
        return source, "already present"
    for name, _mark, anchor, _patched in sites:
        n = source.count(anchor)
        if n != 1:
            raise ValueError(
                f"pinned kv-offload-store anchor '{name}' drifted in {label} "
                f"(found {n}, expected 1)"
            )
    out = source
    for _name, _mark, anchor, patched in sites:
        out = out.replace(anchor, patched, 1)
    if not verified_state(out, sites):
        raise ValueError(f"kv-offload-store post-patch verification failed in {label}")
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kv-offload-store.tmp")
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

    sched_source = TARGET_SCHED.read_text() if TARGET_SCHED.is_file() else ""
    if SCOPE_MARK not in sched_source:
        raise SystemExit(
            "kv-offload-store preflight failed: scheduler.py is not "
            "scope-patched -- run patch_kv_offload_scope.py first "
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
            raise SystemExit(f"kv-offload-store preflight failed: {exc}") from exc
        compile(patched, str(target), "exec")
        prepared.append((target, source, patched, action))
        if preflight_only:
            print(f"{target.name}: kv-offload-store preflight OK ({action})")

    if preflight_only:
        return 0

    for target, source, patched, action in prepared:
        if patched != source:
            replace_file(target, patched)
            clear_pyc(target)
    print(
        f"kv-offload-store {prepared[0][3]} "
        f"({ENV_ENABLE}={os.environ.get(ENV_ENABLE, '0 (unset)')!r})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
