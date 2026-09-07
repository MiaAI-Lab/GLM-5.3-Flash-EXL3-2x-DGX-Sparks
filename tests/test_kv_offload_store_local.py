#!/usr/bin/env python3
"""Regression tests for overlay/patch_kv_offload_store_local.py.

A  Chunk-header codec (format v1): round-trip, truncation at every region,
   CRC corruption, wrong magic, oversized header — all rejected.
B  Patched-runtime store-job drive (stubbed vllm, PATCHED modules): a request
   walking three 3584 boundaries produces store jobs whose GPULoadStoreSpec
   is full-length with zeros for scratch/drafter groups, whose
   glm53_store_meta keys align 1:1 with the src/dst block order, whose
   manifest candidates carry CUMULATIVE chunk-hash chains, and whose
   TransferJob pickles round-trip. Knob off => meta is None.
C  Worker-local writer on a miniature 7-group layout: per-(hash,group) files
   with REAL (unpadded) bytes gathered ref-by-ref from the staging tensors,
   headers with KDA conv/temporal segment split (shape/dtype/stride),
   manifest publish ordering (missing any mamba payload or any cumulative
   full-attn chunk => no manifest), inline K-boundary retention,
   header-verified dedup, failed-key ledger, ENOSPC pause,
   rank-disagreement disable, capture-then-write snapshot isolation (the
   reset_cache()/row-reuse regression: staging mutated after capture must
   not change disk bytes), flush-path capture, namespace forking on
   num_spec change.
D  GC tool (overlay/kv_offload_store_gc.py): dry-run reports orphans (found
   from headers even with zero manifests), corrupt files and stale temps
   without deleting; --sweep removes them; --keep-boundaries applies the
   writer's leaf-walk keep-set (forked chains: shared divergence ancestors
   survive) and preserves cumulative full-attn references (plan §4 C1).

Run:  python3 tests/test_kv_offload_store_local.py   (or pytest)
"""
from __future__ import annotations

import errno
import json
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "overlay"))
sys.path.insert(0, str(HERE.parent))

import patch_kv_offload_store_local as store  # noqa: E402

FIXTURES = HERE / "fixtures"
IN_IMAGE = not FIXTURES.is_dir()

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


# ------------------------------------------------------------------ part A --
def test_codec() -> None:
    print("Part A: chunk-header codec")
    payload = bytes(range(256)) * 4
    header = {"namespace_hash": "abc", "group_idx": 2, "hash": "ff" * 16}
    blob = store.encode_chunk(header, payload)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.bin"
        p.write_bytes(blob)
        h = store.read_chunk_header(str(p), verify_payload=True)
        check(
            h["group_idx"] == 2 and h["payload_len"] == len(payload),
            "A1 round-trip keeps fields and payload length",
        )
        check(h["crc_algo"] == "crc32-zlib", "A2 header names its CRC algorithm")

        cuts = {
            "magic": 4,
            "header-length": 10,
            "header": 14,
            "header-crc": len(blob) - len(payload) - 2,
            "payload": len(blob) - 100,
        }
        for label, cut in cuts.items():
            p.write_bytes(blob[:cut])
            try:
                store.read_chunk_header(str(p), verify_payload=True)
                check(False, f"A3 truncated at {label} rejected")
            except ValueError:
                check(True, f"A3 truncated at {label} rejected")

        flipped = bytearray(blob)
        flipped[-1] ^= 0xFF
        p.write_bytes(bytes(flipped))
        try:
            store.read_chunk_header(str(p), verify_payload=True)
            check(False, "A4 payload bit-flip rejected (CRC)")
        except ValueError:
            check(True, "A4 payload bit-flip rejected (CRC)")

        p.write_bytes(b"NOTMAGIC" + blob[8:])
        try:
            store.read_chunk_header(str(p))
            check(False, "A5 wrong magic rejected")
        except ValueError:
            check(True, "A5 wrong magic rejected")

        # Extra trailing bytes (torn rename of a longer file) rejected too.
        p.write_bytes(blob + b"x")
        try:
            store.read_chunk_header(str(p))
            check(False, "A6 trailing garbage rejected")
        except ValueError:
            check(True, "A6 trailing garbage rejected")


# ------------------------------------------------------------------ part B --
def test_store_job_meta() -> None:
    if IN_IMAGE:
        print("Part B: skipped in-image (fixtures not shipped)")
        return
    print("Part B: patched-runtime store-job drive")
    os.environ["GLM53_KV_OFFLOAD"] = "1"
    os.environ["GLM53_KV_OFFLOAD_RESTORE"] = "0"
    os.environ.pop("GLM53_KV_OFFLOAD_DRAFTER", None)
    from _kv_offload_stub_env import (
        FakeRequest,
        FakeSchedSpec,
        FakeVllmConfig,
        FakeKVTransferConfig,
        boot_layout,
        build_offloading_config,
        install_fake_vllm,
    )

    mods = install_fake_vllm()
    kv_cache_config = boot_layout()
    vllm_config = FakeVllmConfig(kv_transfer_config=FakeKVTransferConfig())
    cfg = build_offloading_config(mods, vllm_config, kv_cache_config)
    sched_mod = mods["scheduler"]
    spec = FakeSchedSpec(mods, cfg)
    scheduler = sched_mod.OffloadingConnectorScheduler(
        spec, vllm_config, kv_cache_config
    )

    n_chunks = 3
    req = FakeRequest(num_tokens=n_chunks * 3584 + 5)
    scheduler.on_new_request(req)
    scheduler.get_num_new_matched_tokens(req, 0)

    per_group_blocks = {0: 1, 1: 896, 2: 1, 3: 1, 4: 1, 5: 1, 6: 56}
    block_id_groups = tuple(
        [1000 * g + j + 1 for j in range(per_group_blocks[g] * n_chunks)]
        for g in range(7)
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={req.request_id: req.num_tokens},
        finished_req_ids=set(),
        req_data={req.request_id: (block_id_groups, False)},
        partial_tail_offloads=None,
    )
    scheduler._update_req_states(scheduler_output)
    store_jobs = scheduler._build_store_jobs(scheduler_output)
    check(len(store_jobs) == 1, "B1 one store job for the request")
    job = next(iter(store_jobs.values()))
    gs = job.src_spec.group_sizes
    check(
        len(gs) == 7 and gs[1] == 0 and gs[6] == 0 and gs[0] == n_chunks,
        f"B2 store group_sizes full-length, zeros at 1/6 (got {gs})",
    )
    meta = job.glm53_store_meta
    check(meta is not None and meta["v"] == 1, "B3 store meta attached, versioned")
    check(
        len(meta["keys"]) == len(job.src_spec.block_ids) == 5 * n_chunks,
        "B4 meta keys align 1:1 with src blocks (5 eligible groups x 3 chunks)",
    )
    check(
        meta["cow_groups"] == [2, 3, 4, 5] and meta["full_groups"] == [0],
        "B5 cow/full group sets carry original indices",
    )
    bounds = {m["boundary_token_index"]: len(m["chunk_hashes"]) for m in meta["manifests"]}
    check(
        bounds == {3584: 1, 7168: 2, 10752: 3},
        f"B6 manifest candidates cumulative per boundary (got {bounds})",
    )
    # Pickle round-trip: the stub spec classes are function-local (unpicklable
    # by construction), so exercise the WIRE payload — the patched TransferJob
    # dataclass with its meta field — with picklable stand-in specs.
    wire = mods["common"].TransferJob(
        req_id=job.req_id, src_spec=None, dst_spec=None, glm53_store_meta=meta
    )
    job2 = pickle.loads(pickle.dumps(wire))
    check(
        job2.glm53_store_meta == meta,
        "B7 TransferJob.glm53_store_meta pickles round-trip unchanged",
    )
    default_job = mods["common"].TransferJob(req_id="r", src_spec=None, dst_spec=None)
    check(
        default_job.glm53_store_meta is None,
        "B7b meta defaults to None (stock constructors unchanged)",
    )

    # Knob off => no meta (and a fresh request to avoid job-state carryover).
    os.environ["GLM53_KV_OFFLOAD"] = "0"
    req_off = FakeRequest(num_tokens=3584 + 5, req_id="req-off", salt=b"t")
    scheduler.on_new_request(req_off)
    scheduler.get_num_new_matched_tokens(req_off, 0)
    bid_off = tuple(
        [5000 * (g + 1) + j + 1 for j in range(per_group_blocks[g])] for g in range(7)
    )
    so_off = SimpleNamespace(
        num_scheduled_tokens={req_off.request_id: req_off.num_tokens},
        finished_req_ids=set(),
        req_data={req_off.request_id: (bid_off, False)},
        partial_tail_offloads=None,
    )
    scheduler._update_req_states(so_off)
    jobs_off = scheduler._build_store_jobs(so_off)
    check(
        all(j.glm53_store_meta is None for j in jobs_off.values()),
        "B8 GLM53_KV_OFFLOAD=0: no store meta is attached",
    )
    _cleanup_env()


# ------------------------------------------------------------------ part C --
class FakeSlice:
    def __init__(self, data: bytes):
        self._data = data

    def numpy(self):
        return self

    def tobytes(self) -> bytes:
        return self._data


class FakeCpuTensor:
    """(num_rows, page) staging tensor; row content is deterministic."""

    def __init__(self, tensor_idx: int, num_rows: int, page: int):
        self.page = page
        self.rows = [
            bytes(((tensor_idx * 31 + r * 7 + i) % 256) for i in range(page))
            for r in range(num_rows)
        ]

    def __getitem__(self, key):
        row, sl = key
        return FakeSlice(self.rows[row][sl])


@dataclass
class FakeRef:
    tensor_idx: int
    page_size_bytes: int


def _mini_env(rank=0, cfg_rank=None, num_spec=7):
    """Miniature 7-group layout for the writer (tiny pages, real semantics)."""
    from _kv_offload_stub_env import (
        FakeKVCacheGroup,
        FakeParallelConfig,
        FakeVllmConfig,
        MLAAttentionSpec,
        KpoolTailSpec,
        MambaSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
        FakeSpeculativeConfig,
    )

    # conv (2,3) bf16 = 12 B + temporal (2,2) f32 = 16 B => real page 28.
    mamba_spec = lambda: MambaSpec(  # noqa: E731
        3584, shapes=((2, 3), (2, 2)), dtypes=("torch.bfloat16", "torch.float32")
    )
    groups = [
        FakeKVCacheGroup(
            UniformTypeKVCacheSpecs(
                3584,
                {"mla.0": MLAAttentionSpec(3584), "mla.1": MLAAttentionSpec(3584), "idx.0": MLAAttentionSpec(3584)},
            ),
            ("mla.0", "mla.1", "idx.0"),
        ),
        FakeKVCacheGroup(KpoolTailSpec(4, prefix_cacheable=False), ("kpool.0",)),
        FakeKVCacheGroup(UniformTypeKVCacheSpecs(3584, {"kda2.0": mamba_spec(), "kda2.1": mamba_spec()}), ("kda2.0", "kda2.1")),
        FakeKVCacheGroup(UniformTypeKVCacheSpecs(3584, {"kda3.0": mamba_spec()}), ("kda3.0",)),
        FakeKVCacheGroup(UniformTypeKVCacheSpecs(3584, {"kda4.0": mamba_spec()}), ("kda4.0",)),
        FakeKVCacheGroup(UniformTypeKVCacheSpecs(3584, {"kda5.0": mamba_spec()}), ("kda5.0",)),
        FakeKVCacheGroup(SlidingWindowSpec(64, 2048), ("draft.0",)),
    ]
    kv_cache_config = SimpleNamespace(kv_cache_groups=groups)

    # tensors: t0/t1 page 64 (MLA slots, mamba parasitizes), t2 page 8 (idx)
    tensors = [FakeCpuTensor(0, 64, 64), FakeCpuTensor(1, 64, 64), FakeCpuTensor(2, 64, 8)]
    refs_per_group = [
        [FakeRef(0, 64), FakeRef(1, 64), FakeRef(2, 8)],  # g0 (3 layers)
        [FakeRef(2, 2)],                                   # g1 (never written)
        [FakeRef(0, 28), FakeRef(1, 28)],                  # g2 (2 KDA layers)
        [FakeRef(0, 28)],                                  # g3
        [FakeRef(1, 28)],                                  # g4
        [FakeRef(0, 28)],                                  # g5
        [FakeRef(1, 16)],                                  # g6 (never written)
    ]
    handler = SimpleNamespace(
        dst_tensors=tensors,
        layer_refs_per_group=refs_per_group,
        _canonical_copy_plans=None,
    )
    worker = SimpleNamespace(_store_handler=handler)
    # Shaped like OffloadingParallelConfig (rank/world_size/tp_size), NOT the
    # vllm ParallelConfig: that is what the writer reads via spec.config.
    parallel = SimpleNamespace(
        rank=cfg_rank if cfg_rank is not None else rank,
        world_size=2,
        tp_size=2,
    )
    del FakeParallelConfig  # imported for parity; unused on purpose
    eligible = []
    for gidx in (0, 2, 3, 4, 5):
        eligible.append(
            SimpleNamespace(
                group_idx=gidx,
                tokens_per_block=3584,
                layer_names=groups[gidx].layer_names,
            )
        )
    spec = SimpleNamespace(
        blocks_per_chunk=1,
        tokens_per_hash=64,
        config=SimpleNamespace(
            parallel=parallel,
            model=SimpleNamespace(name="test/model", dtype="fp8"),
            groups=eligible,
        ),
    )
    vllm_config = FakeVllmConfig()
    vllm_config.speculative_config = FakeSpeculativeConfig(
        num_speculative_tokens=num_spec
    )
    cw = SimpleNamespace(
        spec=spec,
        worker=worker,
        kv_cache_config=kv_cache_config,
        vllm_config=vllm_config,
        _glm53_store_writer=None,
        _glm53_job_meta={},
        _connector_worker_meta=SimpleNamespace(
            completed=[], mark_completed=lambda job_id: None
        ),
    )
    cw._connector_worker_meta.mark_completed = cw._connector_worker_meta.completed.append
    return cw, tensors, refs_per_group


class _TestLogger:
    def __init__(self):
        self.lines = []

    def _fmt(self, msg, *args):
        try:
            self.lines.append(msg % args if args else str(msg))
        except TypeError:
            self.lines.append(str(msg))

    info = debug = warning = error = exception = _fmt


def _hash(i: int) -> str:
    import hashlib

    return hashlib.sha256(b"chunk%d" % i).hexdigest()[:64]


def _job_meta(n_boundaries: int):
    """Meta for one job storing all 5 eligible groups at n boundaries."""
    keys = []
    for k in range(n_boundaries):
        for g in (0, 2, 3, 4, 5):
            keys.append((_hash(k), g, k, 3584))
    manifests = [
        {
            "boundary_token_index": (k + 1) * 3584,
            "chunk_hashes": [_hash(j) for j in range(k + 1)],
        }
        for k in range(n_boundaries)
    ]
    return {
        "v": 1,
        "keys": keys,
        "cow_groups": [2, 3, 4, 5],
        "full_groups": [0],
        "manifests": manifests,
    }


def test_writer() -> None:
    print("Part C: worker-local writer")
    import _kv_offload_stub_env as stub

    if "vllm" not in sys.modules:
        stub.install_fake_vllm()

    logger = _TestLogger()
    with tempfile.TemporaryDirectory() as td:
        os.environ["GLM53_KV_OFFLOAD_DIR"] = td
        os.environ["GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"] = "0"
        ns = store.load_writer_helpers(logger)
        cw, tensors, refs = _mini_env()
        writer = ns["Glm53LocalStoreWriter"](cw, logger)
        check(writer._disabled_reason is None, "C1 writer initialises on the layout")
        check(writer._rank == 0 and writer._base.endswith("_r0"), "C2 per-rank base dir")

        meta = _job_meta(2)
        rows = list(range(len(meta["keys"])))
        # Capture-then-write (review finding 1): the capture snapshots bytes
        # synchronously; mutating the staging rows AFTERWARDS must not change
        # what lands on disk (this is the reset_cache()/row-reuse regression).
        g2_row = rows[meta["keys"].index((_hash(0), 2, 0, 3584))]
        pre0 = tensors[0].rows[g2_row][:28]
        pre1 = tensors[1].rows[g2_row][:28]
        writer.capture_job(7, meta, rows)
        for t in tensors:
            t.rows = [bytes(len(r)) for r in t.rows]  # zero every staging row
        writer.shutdown(timeout=20)
        done = writer.drain_done()
        check(done == [7], "C3 job completion surfaces via the done queue")
        check(
            writer.capture_job(7, meta, rows) is None,
            "C4 capture is idempotent per job id and never defers acks",
        )

        base = Path(writer._base)
        g2_path = base / _hash(0)[:3] / f"{_hash(0)[3:5]}_g2" / f"{_hash(0)}.bin"
        check(g2_path.is_file(), "C5 mamba chunk file lands under its group dir")
        h = store.read_chunk_header(str(g2_path), verify_payload=True)
        check(
            h["payload_len"] == 56 and h["spec_kind"] == "MambaSpec",
            "C6 mamba payload = REAL bytes (2 layers x 28 B, padding never written)",
        )
        segs = h["segment_table"]
        check(
            len(segs) == 4
            and segs[0]["kind"] == "conv_state"
            and segs[0]["shape"] == [2, 3]
            and segs[0]["dtype"] == "torch.bfloat16"
            and segs[0]["stride"] == [3, 1]
            and segs[1]["kind"] == "temporal_state"
            and segs[1]["length"] == 16
            and segs[2]["offset"] == 28,
            "C7 KDA segment table: conv/temporal split with shape/dtype/stride",
        )
        # Payload bytes must equal the PRE-MUTATION staging snapshot, in
        # group layer order — proof the writer worked from private buffers.
        expected = pre0 + pre1
        blob = g2_path.read_bytes()
        check(
            blob.endswith(expected) and any(expected),
            "C8 payload == pre-mutation staging snapshot (private buffers)",
        )

        g0_path = base / _hash(0)[:3] / f"{_hash(0)[3:5]}_g0" / f"{_hash(0)}.bin"
        h0 = store.read_chunk_header(str(g0_path))
        check(h0["payload_len"] == 64 + 64 + 8, "C9 g0 payload spans all 3 layer pages")

        m0 = base / "manifests" / _hash(0)[:3] / f"{_hash(0)}.json"
        m1 = base / "manifests" / _hash(1)[:3] / f"{_hash(1)}.json"
        check(m0.is_file() and m1.is_file(), "C10 manifests published for both boundaries")
        man = json.loads(m1.read_text())
        g0_chunks = man["full_groups"]["0"]
        check(
            man["chunk_hashes"] == [_hash(0), _hash(1)]
            and set(man["cow_groups"]) == {"2", "3", "4", "5"}
            and [c[0] for c in g0_chunks] == [_hash(0), _hash(1)]
            and all(c[1] == 136 and isinstance(c[2], int) for c in g0_chunks),
            "C11 manifest: cumulative chain + per-chunk [hash, len, crc] per group",
        )
        n_files = len(list(base.rglob("*.bin")))
        check(n_files == 10, f"C12 5 groups x 2 boundaries = 10 chunk files (got {n_files})")

        # Incomplete boundary: mamba g3 write missing => no manifest.
        meta3 = _job_meta(3)
        del_idx = meta3["keys"].index((_hash(2), 3, 2, 3584))
        meta3["keys"] = meta3["keys"][:del_idx] + meta3["keys"][del_idx + 1 :]
        meta3["manifests"] = meta3["manifests"][2:]  # only the new boundary
        writer.capture_job(8, meta3, list(range(len(meta3["keys"]))))
        writer.shutdown(timeout=20)
        m2 = base / "manifests" / _hash(2)[:3] / f"{_hash(2)}.json"
        check(
            not m2.is_file(),
            "C13 a boundary missing one mamba payload publishes NO manifest",
        )

        # Failed-key ledger: even with the file later present, a key that
        # failed this boot cannot enter a manifest.
        writer._failed_keys.add((_hash(2), 3))
        meta_fk = {
            "v": 1,
            "keys": [(_hash(2), 3, 2, 3584)],
            "cow_groups": [2, 3, 4, 5],
            "full_groups": [0],
            "manifests": [
                {
                    "boundary_token_index": 3 * 3584,
                    "chunk_hashes": [_hash(0), _hash(1), _hash(2)],
                }
            ],
        }
        writer.capture_job(9, meta_fk, [40])
        writer.shutdown(timeout=20)
        check(
            not m2.is_file(),
            "C14 failed-key ledger blocks the manifest even after a later write",
        )

    # ENOSPC pause.
    logger2 = _TestLogger()
    with tempfile.TemporaryDirectory() as td:
        os.environ["GLM53_KV_OFFLOAD_DIR"] = td
        ns = store.load_writer_helpers(logger2)
        cw, _, _ = _mini_env()
        writer = ns["Glm53LocalStoreWriter"](cw, logger2)
        real_open = open

        def enospc_open(path, mode="r", *a, **k):
            if "b" in mode and "w" in mode:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_open(path, mode, *a, **k)

        ns["open"] = enospc_open
        writer.capture_job(1, _job_meta(1), list(range(5)))
        writer.shutdown(timeout=20)
        check(
            writer._paused_reason == "ENOSPC" and writer.drain_done() == [1],
            "C15 ENOSPC pauses the writer; the job still completes (lost store)",
        )
        check(
            not list(Path(writer._base).rglob("*.bin"))
            and not list(Path(writer._base).rglob("*.tmp.*")),
            "C16 no partial files or stale temps left behind",
        )
        check(
            not list(Path(writer._base).rglob("manifests/**/*.json")),
            "C17 paused writer publishes no manifests",
        )

    # Retention K=1: publishing boundary 2 supersedes boundary 1.
    logger3 = _TestLogger()
    with tempfile.TemporaryDirectory() as td:
        os.environ["GLM53_KV_OFFLOAD_DIR"] = td
        os.environ["GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"] = "1"
        ns = store.load_writer_helpers(logger3)
        cw, _, _ = _mini_env()
        writer = ns["Glm53LocalStoreWriter"](cw, logger3)
        writer.capture_job(1, _job_meta(2), list(range(10)))
        writer.shutdown(timeout=20)
        base = Path(writer._base)
        m0 = base / "manifests" / _hash(0)[:3] / f"{_hash(0)}.json"
        m1 = base / "manifests" / _hash(1)[:3] / f"{_hash(1)}.json"
        g2_b0 = base / _hash(0)[:3] / f"{_hash(0)[3:5]}_g2" / f"{_hash(0)}.bin"
        g0_b0 = base / _hash(0)[:3] / f"{_hash(0)[3:5]}_g0" / f"{_hash(0)}.bin"
        check(
            m1.is_file() and not m0.is_file(),
            "C18 keep_boundaries=1: older boundary manifest superseded",
        )
        check(
            not g2_b0.is_file() and g0_b0.is_file(),
            "C19 superseded boundary: mamba payload unlinked, full-attn chunk kept (C1)",
        )

    # Rank disagreement: TP rank 0 vs config rank 1 => disabled, no writes.
    ps = sys.modules["vllm.distributed.parallel_state"]

    old = ps.get_tensor_model_parallel_rank
    ps.get_tensor_model_parallel_rank = lambda: 0
    try:
        logger4 = _TestLogger()
        with tempfile.TemporaryDirectory() as td:
            os.environ["GLM53_KV_OFFLOAD_DIR"] = td
            ns = store.load_writer_helpers(logger4)
            cw, _, _ = _mini_env(cfg_rank=1)
            writer = ns["Glm53LocalStoreWriter"](cw, logger4)
            check(
                writer._disabled_reason is not None
                and "disagrees" in writer._disabled_reason,
                "C20 TP-rank/config-rank disagreement DISABLES the writer",
            )
            writer.capture_job(1, _job_meta(1), list(range(5)))
            check(
                not list(Path(td).rglob("*.bin")),
                "C21 disabled writer captures nothing (jobs flow normally)",
            )
    finally:
        ps.get_tensor_model_parallel_rank = old

    # Namespace forks on num_spec (stage-0 A2: num_spec is state-ABI).
    logger5 = _TestLogger()
    with tempfile.TemporaryDirectory() as td:
        os.environ["GLM53_KV_OFFLOAD_DIR"] = td
        os.environ["GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"] = "2"
        ns = store.load_writer_helpers(logger5)
        cw7, _, _ = _mini_env(num_spec=7)
        cw8, _, _ = _mini_env(num_spec=8)
        w7 = ns["Glm53LocalStoreWriter"](cw7, logger5)
        w8 = ns["Glm53LocalStoreWriter"](cw8, logger5)
        check(
            w7._namespace_hash != w8._namespace_hash,
            "C22 num_speculative_tokens forks the namespace hash",
        )
        # Policy/diagnostic fields must NOT disable a later boot (fenced-only
        # comparison): same config, different keep_boundaries reuses the store.
        os.environ["GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"] = "5"
        cw7b, _, _ = _mini_env(num_spec=7)
        w7b = ns["Glm53LocalStoreWriter"](cw7b, logger5)
        check(
            w7b._disabled_reason is None
            and w7b._namespace_hash == w7._namespace_hash,
            "C22b retention-K change neither forks nor disables the namespace",
        )
        os.environ["GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"] = "2"
        # A corrupted namespace record fails closed.
        ns_files = [
            f
            for f in Path(os.environ["GLM53_KV_OFFLOAD_DIR"]).glob("glm53kv_*.json")
            if w7._namespace_hash in f.name
        ]
        check(len(ns_files) == 1, "C22c one namespace record per namespace")
        ns_files[0].write_text("{ truncated")
        cw7c, _, _ = _mini_env(num_spec=7)
        w7c = ns["Glm53LocalStoreWriter"](cw7c, logger5)
        check(
            w7c._disabled_reason is not None
            and "unreadable" in w7c._disabled_reason,
            "C22d corrupted namespace record disables the writer (fail closed)",
        )

    # Capture helpers through the connector-worker seam (both call sites).
    logger6 = _TestLogger()
    with tempfile.TemporaryDirectory() as td:
        os.environ["GLM53_KV_OFFLOAD_DIR"] = td
        ns = store.load_writer_helpers(logger6)
        cw, _, _ = _mini_env()
        dst = SimpleNamespace(block_ids=list(range(5)))
        cw._glm53_job_meta[42] = (_job_meta(1), dst)
        ns["_glm53_capture_store_job"](cw, 42)
        check(
            42 not in cw._glm53_job_meta
            and cw._glm53_store_writer is not None,
            "C23 get_finished capture consumes the job meta",
        )
        cw._glm53_store_writer.shutdown(timeout=20)
        base = Path(cw._glm53_store_writer._base)
        check(
            len(list(base.rglob("*.bin"))) == 5,
            "C24 captured job's files land on disk",
        )
        ns["_glm53_capture_store_job"](cw, 43)
        check(
            43 not in cw._glm53_job_meta,
            "C25 jobs without meta (loads) are a no-op capture",
        )
        # Flush/reset path: capture fires for exactly the flushed store jobs.
        cw._glm53_job_meta[44] = (_job_meta(1), dst)
        ns["_glm53_capture_flushed_store_jobs"](cw, {44, 45})
        cw._glm53_store_writer.shutdown(timeout=20)
        check(
            44 not in cw._glm53_job_meta,
            "C26 flush-path capture consumes flushed store jobs (reset-safe)",
        )
    _cleanup_env()


# ------------------------------------------------------------------ part D --
def test_gc_tool() -> None:
    print("Part D: GC/verify tool")
    import _kv_offload_stub_env as stub

    if "vllm" not in sys.modules:
        stub.install_fake_vllm()
    import kv_offload_store_gc as gc_tool

    logger = _TestLogger()
    with tempfile.TemporaryDirectory() as td:
        os.environ["GLM53_KV_OFFLOAD_DIR"] = td
        os.environ["GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"] = "0"
        ns = store.load_writer_helpers(logger)
        cw, _, _ = _mini_env()
        writer = ns["Glm53LocalStoreWriter"](cw, logger)
        writer.capture_job(1, _job_meta(2), list(range(10)))
        writer.shutdown(timeout=20)
        base = Path(writer._base)

        rc = gc_tool.main([str(base)])
        check(rc == 0, "D1 clean store passes the dry-run")

        # Orphan: a mamba payload whose manifest never published.
        meta3 = {
            "v": 1,
            "keys": [(_hash(9), 2, 9, 3584)],
            "cow_groups": [2, 3, 4, 5],
            "full_groups": [0],
            "manifests": [],
        }
        writer2 = ns["Glm53LocalStoreWriter"](cw, logger)
        writer2.capture_job(2, meta3, [50])
        writer2.shutdown(timeout=20)
        orphan = base / _hash(9)[:3] / f"{_hash(9)[3:5]}_g2" / f"{_hash(9)}.bin"
        check(orphan.is_file(), "D2 orphan payload staged")
        rc = gc_tool.main([str(base)])
        check(rc == 1 and orphan.is_file(), "D3 dry-run reports the orphan, deletes nothing")

        # Corrupt file: truncate one chunk.
        victim = base / _hash(1)[:3] / f"{_hash(1)[3:5]}_g3" / f"{_hash(1)}.bin"
        victim.write_bytes(victim.read_bytes()[:20])
        rc = gc_tool.main([str(base)])
        check(rc == 1, "D4 dry-run flags the truncated chunk")

        rc = gc_tool.main([str(base), "--sweep"])
        check(
            not orphan.is_file() and not victim.is_file(),
            "D5 --sweep removes orphan and corrupt files",
        )

        # keep-boundaries: keep 1 => boundary-0 manifest superseded, its mamba
        # chunk swept, but the cumulative g0 chunk of boundary 0 survives (C1).
        rc = gc_tool.main([str(base), "--sweep", "--keep-boundaries", "1"])
        m0 = base / "manifests" / _hash(0)[:3] / f"{_hash(0)}.json"
        g0_b0 = base / _hash(0)[:3] / f"{_hash(0)[3:5]}_g0" / f"{_hash(0)}.bin"
        g2_b0 = base / _hash(0)[:3] / f"{_hash(0)[3:5]}_g2" / f"{_hash(0)}.bin"
        check(not m0.is_file(), "D6 --keep-boundaries supersedes the older manifest")
        check(
            g0_b0.is_file() and not g2_b0.is_file(),
            "D7 cumulative full-attn refs protected; superseded mamba swept",
        )

    # Crash-before-first-manifest: orphans must be found from HEADERS alone.
    logger7 = _TestLogger()
    with tempfile.TemporaryDirectory() as td:
        os.environ["GLM53_KV_OFFLOAD_DIR"] = td
        os.environ["GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"] = "0"
        ns = store.load_writer_helpers(logger7)
        cw, _, _ = _mini_env()
        w = ns["Glm53LocalStoreWriter"](cw, logger7)
        meta = {
            "v": 1,
            "keys": [(_hash(3), 4, 3, 3584)],
            "cow_groups": [2, 3, 4, 5],
            "full_groups": [0],
            "manifests": [],
        }
        w.capture_job(1, meta, [7])
        w.shutdown(timeout=20)
        base = Path(w._base)
        out = gc_tool.main([str(base)])
        check(
            out == 1,
            "D8 orphan reported with ZERO manifests present (header-based)",
        )
        # Stale temps: payload temps under the base AND namespace-record
        # temps beside it (the store root) are reported and swept.
        t = base / _hash(3)[:3] / f"{_hash(3)[3:5]}_g4" / "x.bin.tmp.r0.1.2"
        t.write_bytes(b"partial")
        ns_t = base.parent / "glm53kv_test_model_deadbeef.json.tmp.r0.1"
        ns_t.write_text("{}")
        check(gc_tool.main([str(base)]) == 1, "D9 stale temps flagged by dry-run")
        gc_tool.main([str(base), "--sweep"])
        check(
            not t.exists() and not ns_t.exists(),
            "D10 --sweep removes payload AND namespace temps",
        )

    # Forked chains: a shared divergence-point boundary survives while ANY
    # branch still keeps it (writer rule mirrored in the GC tool).
    logger8 = _TestLogger()
    with tempfile.TemporaryDirectory() as td:
        os.environ["GLM53_KV_OFFLOAD_DIR"] = td
        os.environ["GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"] = "0"
        ns = store.load_writer_helpers(logger8)
        cw, _, _ = _mini_env()
        w = ns["Glm53LocalStoreWriter"](cw, logger8)
        keys = []
        for k, h in ((0, _hash(0)), (1, _hash(10)), (1, _hash(11))):
            for g in (0, 2, 3, 4, 5):
                keys.append((h, g, k, 3584))
        meta = {
            "v": 1,
            "keys": keys,
            "cow_groups": [2, 3, 4, 5],
            "full_groups": [0],
            "manifests": [
                {"boundary_token_index": 3584, "chunk_hashes": [_hash(0)]},
                {"boundary_token_index": 7168, "chunk_hashes": [_hash(0), _hash(10)]},
                {"boundary_token_index": 7168, "chunk_hashes": [_hash(0), _hash(11)]},
            ],
        }
        w.capture_job(1, meta, list(range(len(keys))))
        w.shutdown(timeout=20)
        base = Path(w._base)
        m_shared = base / "manifests" / _hash(0)[:3] / f"{_hash(0)}.json"
        check(m_shared.is_file(), "D11 forked store staged (shared ancestor manifested)")
        gc_tool.main([str(base), "--sweep", "--keep-boundaries", "2"])
        check(
            m_shared.is_file(),
            "D12 K=2: shared divergence-point ancestor kept (both branches need it)",
        )
        gc_tool.main([str(base), "--sweep", "--keep-boundaries", "1"])
        m_a = base / "manifests" / _hash(10)[:3] / f"{_hash(10)}.json"
        m_b = base / "manifests" / _hash(11)[:3] / f"{_hash(11)}.json"
        check(
            m_a.is_file() and m_b.is_file() and not m_shared.is_file(),
            "D13 K=1: both leaves kept, only the shared ancestor superseded",
        )
    _cleanup_env()




def _cleanup_env() -> None:
    """These tests mutate GLM53_KV_OFFLOAD* in os.environ; scrub them so the
    rest of the suite (and any subprocess-driving test that inherits the
    process env) sees a clean slate."""
    for name in ("GLM53_KV_OFFLOAD","GLM53_KV_OFFLOAD_DIR","GLM53_KV_OFFLOAD_CPU_GB","GLM53_KV_OFFLOAD_RESTORE","GLM53_KV_OFFLOAD_DRAFTER","GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"):
        os.environ.pop(name, None)


def main() -> int:
    test_codec()
    test_store_job_meta()
    test_writer()
    test_gc_tool()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
