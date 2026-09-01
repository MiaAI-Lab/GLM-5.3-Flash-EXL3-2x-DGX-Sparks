#!/usr/bin/env python3
"""Regression tests for overlay/patch_kv_offload_restore_g0.py.

A  Patcher hygiene on the pinned image fixtures (incl. the two NEW stage-2
   captures): preflight/apply/idempotence/tamper on all five targets;
   refusal against a tree missing the scope/store marks; --check-injected.
B  Patched-runtime scheduler drive (stubbed vllm, PATCHED modules):
   eligibility matrix (7-group hybrid INERT with the named reason;
   single-TOTAL-group probe ACTIVE; one-eligible-plus-excluded-group INERT —
   Codex OFFLOAD2 finding 2), AND-across-ranks manifest registry (one rank
   => miss, both => hit, "-" retraction => miss, "F" store failure => miss),
   deepest-boundary lookup + alignment guard, single-flight gate (defer +
   self-heal + completion/reset clear), disk load-job shape (dict src_spec,
   1:1 entries, ZERO manager lookup/prepare_load calls, stage-1 bookkeeping
   preserved — finding 12), request.glm53_restored_boundary lifecycle,
   preemption load-job guard (finding 3), zero-cow manifest candidates (S1),
   RESTORE=0 short-circuit parity.
C  Worker-side synchronous disk loader against REAL writer output: byte-
   exact happy-path restore into stubbed GPU tensors, full segment-table
   verification (finding 13), T4 failure matrix (missing file / truncated /
   CRC flip / wrong rank / null dst / manifest missing / precondition) =>
   zero-fill + invalid ids + the job STILL completes (finding 5), earlier
   chunks keep their restored bytes, later chunks abandoned.
D  Manifest-event channel: writer "+"/"-"/"F" emission, validated dedup
   (finding 9), worker-meta attach + event-only meta flows (finding 7),
   aggregate() concatenation incl. a legacy meta without the field, pickle
   round-trip; facade get_block_ids_with_load_errors drain (patched facade
   text driven directly).
E  T2 (patched single_type_kv_cache_manager text): cache_blocks appends
   glm53_restored_boundary to reachable_boundaries (absent/0 => unchanged);
   MambaManager.reachable_block_mask keeps exactly the restored boundary's
   state block under retention_interval=0 (end-exclusive convention,
   finding 10: 3584/7168 select blocks 0/1; 3583 selects none).

Run:  python3 tests/test_kv_offload_restore_g0.py   (or pytest)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "overlay"))

import patch_kv_offload_restore_g0 as restore  # noqa: E402
import patch_kv_offload_store_local as store  # noqa: E402

FIXTURES = HERE / "fixtures"

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


# ------------------------------------------------------------------ part A --
def test_patcher_hygiene() -> None:
    print("Part A: patcher hygiene")
    from _kv_offload_stub_env import patched_texts

    texts = patched_texts(with_store=True, with_restore=True)
    for name in ("common", "scheduler", "worker"):
        compile(texts[name], name, "exec")
    check(True, "A1 restore overlay applies onto scope+store output and compiles")

    fac = (FIXTURES / "image_487ecf187_offloading_connector.py").read_text()
    st = (FIXTURES / "image_487ecf187_single_type_kv_cache_manager.py").read_text()
    fac2, _ = restore.prepare(fac, restore.SITES_FACADE, "facade")
    st2, _ = restore.prepare(st, restore.SITES_SINGLE_TYPE, "single_type")
    compile(fac2, "facade", "exec")
    compile(st2, "single_type", "exec")
    check(True, "A2 facade + single_type fixtures patch and compile")

    check(
        restore.prepare(fac2, restore.SITES_FACADE, "f")[1] == "already present"
        and restore.prepare(st2, restore.SITES_SINGLE_TYPE, "s")[1]
        == "already present"
        and restore.prepare(
            texts["scheduler"], restore.SITES_SCHED, "sched"
        )[1]
        == "already present",
        "A3 idempotence: patched output detected as already present",
    )

    tampered = texts["scheduler"].replace(restore.MARK_R2, "", 1)
    try:
        restore.prepare(tampered, restore.SITES_SCHED, "t")
        check(False, "A4 half-patched file refused")
    except ValueError as exc:
        check("tampered" in str(exc), "A4 half-patched file refused")

    # F10 (review): tampering INSIDE a patched block while keeping every
    # marker line must be refused on the already-present path.
    tampered2 = texts["scheduler"].replace(
        "self._glm53_restore.job_started(load_job_id)",
        "pass  # neutered gate",
        1,
    )
    assert all(
        tampered2.count(m) == 1 for _n, m, _a, _p in restore.SITES_SCHED
    )
    try:
        restore.prepare(tampered2, restore.SITES_SCHED, "t")
        check(False, "A4b marked-but-tampered file refused (review f10)")
    except ValueError:
        check(True, "A4b marked-but-tampered file refused (review f10)")

    # A store-less scheduler must fail anchors (R10 anchors store output).
    store_less = patched_texts(with_store=False)["scheduler"]
    try:
        restore.prepare(store_less, restore.SITES_SCHED, "t")
        check(False, "A5 store-less scheduler refused (anchor drift)")
    except ValueError:
        check(True, "A5 store-less scheduler refused (anchor drift)")

    import subprocess

    r = subprocess.run(
        [sys.executable, str(ROOT / "overlay" / "patch_kv_offload_restore_g0.py"),
         "--check-injected"],
        capture_output=True,
        text=True,
    )
    check(r.returncode == 0, "A6 --check-injected compiles all injected sources")


# ------------------------------------------------------------------ part B --
class FakeBlock:
    def __init__(self, block_id, is_null=False, block_hash=None):
        self.block_id = block_id
        self.is_null = is_null
        self.block_hash = block_hash


def _single_group_layout(stub):
    """One TOTAL KV-cache group: full attention, block 3584 (the g0 probe)."""
    layers = ("mla.0", "mla.1")
    g0 = stub.FakeKVCacheGroup(
        stub.UniformTypeKVCacheSpecs(
            3584, {n: stub.MLAAttentionSpec(3584) for n in layers}
        ),
        layers,
    )
    return stub.FakeKVCacheConfig([g0])


def _one_eligible_plus_excluded_layout(stub):
    """One eligible full-attn group + one EXCLUDED scratch group (finding 2)."""
    cfg = _single_group_layout(stub)
    g1 = stub.FakeKVCacheGroup(
        stub.KpoolTailSpec(4, prefix_cacheable=False), ("kpool.0",)
    )
    cfg.kv_cache_groups.append(g1)
    return cfg


def _mk_scheduler(stub, mods, kv_cache_config, spec_off=False):
    vllm_config = stub.FakeVllmConfig(
        kv_transfer_config=stub.FakeKVTransferConfig()
    )
    if spec_off:
        vllm_config.speculative_config = None
    cfg = stub.build_offloading_config(mods, vllm_config, kv_cache_config)
    spec = stub.FakeSchedSpec(mods, cfg)
    sched = mods["scheduler"].OffloadingConnectorScheduler(
        spec, vllm_config, kv_cache_config
    )
    return sched, spec


def _events(ranks, bhash, tokens, ns="ns1", code="+"):
    return [(code, r, ns, bhash, tokens, "boot0") for r in ranks]


def _boundary_hash(req, k, hashes_per_chunk=56):
    return bytes(req.block_hashes[(k + 1) * hashes_per_chunk - 1]).hex()


def test_scheduler_restore() -> None:
    print("Part B: patched-runtime scheduler drive")
    os.environ["GLM53_KV_OFFLOAD"] = "1"
    os.environ["GLM53_KV_OFFLOAD_RESTORE"] = "1"
    os.environ.pop("GLM53_KV_OFFLOAD_DRAFTER", None)
    import _kv_offload_stub_env as stub

    mods = stub.install_fake_vllm(with_restore=True)
    sched_mod = mods["scheduler"]

    # B1: the live 7-group hybrid is INERT with a named reason.
    sched7, _ = _mk_scheduler(stub, mods, stub.boot_layout())
    reason7 = sched7._glm53_restore.disabled_reason
    check(
        reason7 is not None and "7 KV-cache groups" in reason7,
        f"B1 hybrid layout INERT with named reason (got {reason7!r})",
    )
    check(
        any("restore INERT" in ln for ln in mods["logger"].lines),
        "B1b inert reason logged at boot",
    )

    # B2: single-TOTAL-group probe is ACTIVE (speculative off: the image
    # flags every group eagle under an eagle spec config with no marks).
    sched, spec = _mk_scheduler(
        stub, mods, _single_group_layout(stub), spec_off=True
    )
    rst = sched._glm53_restore
    check(
        rst.disabled_reason is None,
        f"B2 single-group probe ACTIVE (got {rst.disabled_reason!r})",
    )

    # B2b: one eligible + one excluded group must stay INERT (finding 2).
    sched_x, _ = _mk_scheduler(
        stub, mods, _one_eligible_plus_excluded_layout(stub), spec_off=True
    )
    rx = sched_x._glm53_restore.disabled_reason
    check(
        rx is not None and "2 KV-cache groups" in rx and "1 eligible" in rx,
        f"B2b eligible+excluded layout INERT (got {rx!r})",
    )

    # B3: AND-across-ranks registry (num_workers=2 in the stub config).
    req = stub.FakeRequest(num_tokens=3 * 3584 + 5, req_id="rq1")
    sched.on_new_request(req)
    b1 = _boundary_hash(req, 1)
    rst.consume_events(_events([0], b1, 7168))
    hit, _async = sched.get_num_new_matched_tokens(req, 0)
    check(hit == 0, "B3 one-rank manifest => miss (AND across ranks)")
    rst.consume_events(_events([1], b1, 7168))
    hit, is_async = sched.get_num_new_matched_tokens(req, 0)
    check(
        hit == 7168 and is_async,
        f"B3b both-rank manifest => hit at the boundary (got {hit})",
    )
    rst.consume_events(_events([1], b1, 7168, code="-"))
    hit, _ = sched.get_num_new_matched_tokens(req, 0)
    check(hit == 0, "B3c retraction on one rank => miss again")
    rst.consume_events(_events([1], b1, 7168))
    rst.consume_events([("F", 1, "ns1", b1, 0, "boot0")])
    hit, _ = sched.get_num_new_matched_tokens(req, 0)
    check(hit == 0, "B3d store-write failure event denies the boundary")

    # B4: deepest boundary wins; foreign namespace dropped; alignment guard.
    rst._failed_by_rank.clear()
    b2 = _boundary_hash(req, 2)
    rst.consume_events(_events([0, 1], b2, 10752))
    hit, _ = sched.get_num_new_matched_tokens(req, 0)
    check(hit == 10752, f"B4 deepest manifested boundary wins (got {hit})")
    rst.consume_events(_events([0, 1], b2, 10752, ns="other"))
    check(
        rst._namespace == "ns1", "B4b foreign-namespace events are dropped"
    )
    req_off = stub.FakeRequest(num_tokens=3 * 3584, req_id="rq-unaligned")
    sched.on_new_request(req_off)
    req_off.block_hashes = req.block_hashes[: len(req_off.block_hashes)]
    sched._req_status[req_off.request_id].num_locally_computed_tokens = 0
    check(
        rst.lookup(sched, sched._req_status[req_off.request_id]) in (10752,),
        "B4c same-chain request hits via content addressing",
    )
    st_un = sched._req_status[req_off.request_id]
    st_un.num_locally_computed_tokens = 100  # not chunk-aligned
    check(
        rst.lookup(sched, st_un) == 0,
        "B4d non-chunk-aligned local hit => no disk extension (plan F6)",
    )

    # B5/B6: allocation builds a DISK load job; manager untouched.
    manager = spec._manager
    manager.prepare_load_calls.clear()
    lookup_calls_before = len(manager.lookup_calls)
    touched_before = len(manager.touched)
    hit, _ = sched.get_num_new_matched_tokens(req, 0)
    check(hit == 10752, "B5 lookup ready for allocation")
    n_blocks = 3
    blocks = mods["kv_cache_manager"].KVCacheBlocks(
        blocks=([FakeBlock(100 + i) for i in range(n_blocks)],)
    )
    sched.update_state_after_alloc(req, blocks, hit)
    check(
        len(manager.prepare_load_calls) == 0
        and len(manager.lookup_calls) == lookup_calls_before,
        "B6 ZERO manager lookup/prepare_load calls on the disk path",
    )
    check(
        len(manager.touched) > touched_before
        and sched._req_status[req.request_id].group_states[0].offload_keys,
        "B6b stage-1 bookkeeping preserved (touch ran, offload keys built)",
    )
    check(
        len(sched._current_batch_load_jobs) == 1,
        "B6c one disk load job created",
    )
    job_id, job = next(iter(sched._current_batch_load_jobs.items()))
    ss = job.src_spec
    check(
        isinstance(ss, dict)
        and ss["glm53_disk_load"] == 1
        and ss["v"] == 1
        and ss["namespace_hash"] == "ns1"
        and ss["boundary_token_index"] == 10752
        and len(ss["entries"]) == n_blocks
        and [e[2] for e in ss["entries"]] == [0, 1, 2]
        and ss["entries"][-1][0] == b2
        and list(job.dst_spec.block_ids) == [100, 101, 102],
        f"B6d disk src_spec shape + 1:1 entry alignment (got {ss})",
    )
    check(
        sched._jobs[job_id].keys == set()
        and not sched._jobs[job_id].is_store,
        "B6e TransferJobStatus: empty keys, is_store=False",
    )
    check(
        getattr(req, "glm53_restored_boundary", None) == 10752,
        "B6f request.glm53_restored_boundary set END-EXCLUSIVE (T2)",
    )

    # B7: single-flight gate defers, self-heals, and clears on completion.
    req_b = stub.FakeRequest(num_tokens=2 * 3584, req_id="rq2", salt=b"s")
    req_b.block_hashes = req.block_hashes[: len(req_b.block_hashes)]
    sched.on_new_request(req_b)
    hit_b, _ = sched.get_num_new_matched_tokens(req_b, 0)
    check(hit_b is None, "B7 second restore while in flight => defer (None)")
    rst._inflight_job = 424242  # vanished job
    hit_b, _ = sched.get_num_new_matched_tokens(req_b, 0)
    check(
        hit_b == 7168 and rst._inflight_job is None,
        "B7b gate self-heals when its job vanished (finding 11)",
    )
    rst.job_started(job_id)
    out_meta = mods["common"].OffloadingWorkerMetadata(
        completed_jobs={job_id: 2}
    )
    sched.update_connector_output(
        SimpleNamespace(kv_connector_worker_meta=out_meta)
    )
    check(
        rst._inflight_job is None and job_id not in sched._jobs,
        "B7c all-rank completion clears the gate (complete_load(set()) no-op)",
    )

    # B8: preemption with a disk LOAD job must not assert (finding 3).
    req_c = stub.FakeRequest(num_tokens=2 * 3584, req_id="rq3", salt=b"s")
    req_c.block_hashes = req.block_hashes[: len(req_c.block_hashes)]
    sched.on_new_request(req_c)
    hit_c, _ = sched.get_num_new_matched_tokens(req_c, 0)
    check(hit_c == 7168, "B8 setup hit")
    blocks_c = mods["kv_cache_manager"].KVCacheBlocks(
        blocks=([FakeBlock(200), FakeBlock(201)],)
    )
    sched.update_state_after_alloc(req_c, blocks_c, hit_c)
    load_c = next(iter(sched._current_batch_load_jobs))
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={},
        finished_req_ids=set(),
        req_data={},
        partial_tail_offloads=None,
        preempted_req_ids={req_c.request_id},
        scheduled_new_reqs=[],
    )
    meta_out = sched.build_connector_meta(scheduler_output)
    check(
        load_c not in (meta_out.jobs_to_flush or set())
        and rst._inflight_job is None,
        "B8b preempted load job: no assert, not flushed, gate cleared",
    )

    # B9: reset clears gate + pending hits; lookup self-heals after.
    rst.job_started(9999)
    rst._pending_hits["zombie"] = 3584
    sched.reset_cache()
    check(
        rst._inflight_job is None and not rst._pending_hits,
        "B9 reset_cache clears the gate and pending hits",
    )

    # B10 (S1): a zero-cow layout publishes manifest candidates.
    req_s = stub.FakeRequest(num_tokens=2 * 3584 + 5, req_id="rq-s1", salt=b"x")
    sched.on_new_request(req_s)
    sched.get_num_new_matched_tokens(req_s, 0)
    so = SimpleNamespace(
        num_scheduled_tokens={req_s.request_id: req_s.num_tokens},
        finished_req_ids=set(),
        req_data={req_s.request_id: (([300 + j for j in range(2)],), False)},
        partial_tail_offloads=None,
    )
    sched._update_req_states(so)
    jobs_s = sched._build_store_jobs(so)
    metas = [j.glm53_store_meta for j in jobs_s.values() if j.glm53_store_meta]
    cands = [m["manifests"] for m in metas]
    check(
        len(jobs_s) == 1
        and metas
        and metas[0]["cow_groups"] == []
        and {m["boundary_token_index"] for m in cands[0]} == {3584, 7168}
        and [len(m["chunk_hashes"]) for m in sorted(
            cands[0], key=lambda m: m["boundary_token_index"]
        )] == [1, 2],
        f"B10 zero-cow manifest candidates cumulative per boundary (got {cands})",
    )

    # B12 (review f11): distinct num_tokens vs num_prompt_tokens — the
    # lookup caps at num_tokens (a re-admitted request's decoded tokens are
    # legitimately restorable; offload_prompt_only=false stores them).
    req_d = stub.FakeRequest(
        num_tokens=3 * 3584,
        req_id="rq-decode",
        num_prompt_tokens=2 * 3584,
    )
    req_d.block_hashes = req.block_hashes[: len(req_d.block_hashes)]
    sched.on_new_request(req_d)
    hit_d, _ = sched.get_num_new_matched_tokens(req_d, 0)
    check(
        hit_d == 10752,
        f"B12 boundary past num_prompt but within num_tokens hits (got {hit_d})",
    )

    # B13: local prefix already AT the deepest boundary => no restore hit.
    st_d = sched._req_status[req_d.request_id]
    st_d.num_locally_computed_tokens = 10752
    check(
        rst.lookup(sched, st_d) == 0,
        "B13 local == deepest boundary => 0 (nothing beyond to restore)",
    )

    # B14 (review f7/f8): garbage ranks never satisfy the quorum; a changed
    # writer boot id retracts that rank's availability.
    req_e = stub.FakeRequest(num_tokens=2 * 3584, req_id="rq-e", salt=b"e14")
    sched.on_new_request(req_e)
    b1e = _boundary_hash(req_e, 1)
    rst.consume_events([("+", 0, "ns1", b1e, 7168, "boot0")])
    rst.consume_events([("+", 7, "ns1", b1e, 7168, "boot0")])  # bogus rank
    hit_e, _ = sched.get_num_new_matched_tokens(req_e, 0)
    check(hit_e == 0, "B14 out-of-range rank never completes the quorum")
    rst.consume_events([("+", 1, "ns1", b1e, 7168, "boot0")])
    hit_e, _ = sched.get_num_new_matched_tokens(req_e, 0)
    check(hit_e == 7168, "B14b real second rank completes it")
    rst.consume_events([("+", 1, "ns1", "ff" * 32, 3584, "boot1")])  # restart
    hit_e, _ = sched.get_num_new_matched_tokens(req_e, 0)
    check(
        hit_e == 0,
        "B14c changed writer boot id retracts that rank's availability",
    )
    # Repair rank 1's availability under its new generation for B16 (the
    # boot-id retraction dropped rank 1 from EVERY boundary, including the
    # main chain's b1 that B16 relies on).
    rst.consume_events([("+", 1, "ns1", b1, 7168, "boot1")])

    # B15 (review f2): a short hash list ships an all-fail spec, never an
    # IndexError out of update_state_after_alloc.
    spec_bad = rst.build_load_spec(
        SimpleNamespace(request_id="rq-short", block_hashes=[b"x" * 32]),
        SimpleNamespace(num_locally_computed_tokens=0),
        10752,
        10752,
    )
    check(
        spec_bad["entries"] == [] and spec_bad["glm53_disk_load"] == 1,
        "B15 short hash list => empty-entry (all-fail) spec, no IndexError",
    )

    # B16 (review f1, BLOCKER): a request whose disk LOAD job is still
    # registered must not reach _build_store_jobs' is_store assert — store
    # creation is deferred instead.
    req_f = stub.FakeRequest(num_tokens=2 * 3584 + 5, req_id="rq-f")
    req_f.block_hashes = req.block_hashes[: len(req_f.block_hashes)]
    sched.on_new_request(req_f)
    hit_f, _ = sched.get_num_new_matched_tokens(req_f, 0)
    check(hit_f == 7168, "B16 setup hit")
    blocks_f = mods["kv_cache_manager"].KVCacheBlocks(
        blocks=([FakeBlock(400), FakeBlock(401)],)
    )
    sched.update_state_after_alloc(req_f, blocks_f, hit_f)
    sched._current_batch_load_jobs.clear()  # job registered in _jobs
    # The request is ABORTED while its disk load is still registered — the
    # exact route the review's BLOCKER named (finished_req_ids processing in
    # _build_store_jobs while a non-store job is in transfer_jobs).
    req_f.status = sys.modules["vllm.v1.request"].RequestStatus.FINISHED_ABORTED
    req_f.num_computed_tokens = 0
    req_f.is_finished = lambda: True
    so_f = SimpleNamespace(
        num_scheduled_tokens={},
        finished_req_ids={req_f.request_id},
        req_data={},
        partial_tail_offloads=None,
    )
    sched._update_req_states(so_f)
    jobs_f = sched._build_store_jobs(so_f)  # would assert without R11
    check(
        jobs_f == {},
        "B16b aborted-during-load: store creation deferred, no assert",
    )
    # Complete B16's load job (both ranks ack) so the single-flight gate is
    # free for the following subtests.
    jid_f = rst._inflight_job
    sched.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=mods["common"].OffloadingWorkerMetadata(
                completed_jobs={jid_f: 2}
            )
        )
    )
    check(rst._inflight_job is None, "B16c load-job completion frees the gate")

    # B17 (confirm f9): a writer restart retracts that rank's FAILURES too —
    # a recovered writer is not permanently suppressed.
    req_g = stub.FakeRequest(num_tokens=2 * 3584, req_id="rq-g", salt=b"g17")
    sched.on_new_request(req_g)
    b1g = _boundary_hash(req_g, 1)
    rst.consume_events(
        [("+", 0, "ns1", b1g, 7168, "boot0"), ("+", 1, "ns1", b1g, 7168, "boot1")]
    )
    rst.consume_events([("F", 0, "ns1", b1g, 0, "boot0")])
    hit_g, _ = sched.get_num_new_matched_tokens(req_g, 0)
    check(hit_g == 0, "B17 rank-0 write failure denies the boundary")
    # rank 0's writer restarts (new generation) and republishes cleanly:
    rst.consume_events([("+", 0, "ns1", b1g, 7168, "boot0b")])
    hit_g, _ = sched.get_num_new_matched_tokens(req_g, 0)
    check(
        hit_g == 7168,
        "B17b generation change clears the old generation's failures",
    )

    # B18 (confirm new f1): a NON-disk load job keeps the stock crash
    # semantics — never misclassified as ours.
    req_h = stub.FakeRequest(num_tokens=3584, req_id="rq-h", salt=b"h18")
    sched.on_new_request(req_h)
    st_h = sched._req_status[req_h.request_id]
    foreign_jid = sched._generate_job_id()
    sched._jobs[foreign_jid] = sched_mod.TransferJobStatus(
        req_id=req_h.request_id, pending_count=2, keys=set(), is_store=False
    )
    st_h.transfer_jobs.add(foreign_jid)
    so_h = SimpleNamespace(
        num_scheduled_tokens={},
        finished_req_ids=set(),
        req_data={},
        partial_tail_offloads=None,
        preempted_req_ids={req_h.request_id},
        scheduled_new_reqs=[],
    )
    try:
        sched.build_connector_meta(so_h)
        check(False, "B18 foreign (non-disk) load keeps stock assert semantics")
    except AssertionError:
        check(True, "B18 foreign (non-disk) load keeps stock assert semantics")
    st_h.transfer_jobs.discard(foreign_jid)
    sched._jobs.pop(foreign_jid, None)

    # B11: RESTORE=0 keeps the stage-1 short-circuit (no disk lookup).
    os.environ["GLM53_KV_OFFLOAD_RESTORE"] = "0"
    mods0 = stub.install_fake_vllm(with_restore=True)
    sched0, _ = _mk_scheduler(
        stub, mods0, _single_group_layout(stub), spec_off=True
    )
    check(
        sched0._glm53_restore.disabled_reason == "GLM53_KV_OFFLOAD_RESTORE=0",
        "B11 RESTORE=0: restore state disabled with the knob reason",
    )
    req0 = stub.FakeRequest(num_tokens=7168, req_id="rq0")
    sched0.on_new_request(req0)
    hit0, async0 = sched0.get_num_new_matched_tokens(req0, 0)
    check(hit0 == 0 and not async0, "B11b RESTORE=0: zero external hits")
    _cleanup_env()


# ------------------------------------------------------------------ part C --
class FakeTorchBuf:
    def __init__(self, data: bytearray):
        self.data = data


class FakeGpuRow:
    def __init__(self, backing: bytearray, start: int, length: int):
        self.backing = backing
        self.start = start
        self.length = length

    def copy_(self, buf: FakeTorchBuf):
        assert len(buf.data) == self.length
        self.backing[self.start : self.start + self.length] = buf.data

    def zero_(self):
        self.backing[self.start : self.start + self.length] = bytes(self.length)


class FakeGpuTensor:
    """(num_rows, page) int8 GPU tensor over a flat bytearray."""

    def __init__(self, num_rows: int, page: int):
        self.page = page
        self.backing = bytearray(os.urandom(num_rows * page))

    def __getitem__(self, key):
        row, sl = key
        start, stop, _ = sl.indices(self.page)
        return FakeGpuRow(self.backing, row * self.page + start, stop - start)

    def row_bytes(self, row: int, n: int) -> bytes:
        return bytes(self.backing[row * self.page : row * self.page + n])


def _install_fake_torch():
    t = types.ModuleType("torch")
    t.int8 = "int8"

    def frombuffer(data, dtype):
        assert dtype == t.int8
        return FakeTorchBuf(data)

    t.frombuffer = frombuffer
    t.Tensor = object
    t.Event = object
    sys.modules["torch"] = t


def _g0_env(store_mod, tmpdir, logger):
    """Single-full-attn-group writer env (2 layers: pages 64 + 8)."""
    from test_kv_offload_store_local import FakeCpuTensor, FakeRef

    import _kv_offload_stub_env as stub

    layers = ("mla.0", "idx.0")
    g0 = stub.FakeKVCacheGroup(
        stub.UniformTypeKVCacheSpecs(
            3584, {n: stub.MLAAttentionSpec(3584) for n in layers}
        ),
        layers,
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=[g0])
    tensors = [FakeCpuTensor(0, 64, 64), FakeCpuTensor(1, 64, 8)]
    refs = [[FakeRef(0, 64), FakeRef(1, 8)]]
    handler = SimpleNamespace(
        dst_tensors=tensors,
        layer_refs_per_group=refs,
        _canonical_copy_plans=None,
    )
    worker = SimpleNamespace(_store_handler=handler)
    parallel = SimpleNamespace(rank=0, world_size=2, tp_size=2)
    spec = SimpleNamespace(
        blocks_per_chunk=1,
        tokens_per_hash=64,
        config=SimpleNamespace(
            parallel=parallel,
            model=SimpleNamespace(name="test/model", dtype="fp8"),
            groups=[
                SimpleNamespace(
                    group_idx=0, tokens_per_block=3584, layer_names=layers
                )
            ],
        ),
    )
    vllm_config = stub.FakeVllmConfig()
    cw = SimpleNamespace(
        spec=spec,
        worker=worker,
        kv_cache_config=kv_cache_config,
        vllm_config=vllm_config,
        _glm53_store_writer=None,
        _glm53_job_meta={},
        _glm53_gpu_caches=None,
        _glm53_disk_done=[],
        _glm53_load_error_ids=set(),
        _connector_worker_meta=SimpleNamespace(
            completed_jobs={},
            glm53_manifest_events=[],
            transfer_stats=SimpleNamespace(
                load=SimpleNamespace(record=lambda n, t: None)
            ),
            mark_completed=lambda job_id: None,
        ),
    )
    os.environ["GLM53_KV_OFFLOAD_DIR"] = str(tmpdir)
    ns = _load_restore_helpers(store_mod, logger)
    writer = ns["Glm53LocalStoreWriter"](cw, logger)
    cw._glm53_store_writer = writer
    # GPU-side canonical caches mirroring the staging layout (mapping=None).
    gpu_tensors = [
        SimpleNamespace(tensor=FakeGpuTensor(64, 64), page_size_bytes=64),
        SimpleNamespace(tensor=FakeGpuTensor(64, 8), page_size_bytes=8),
    ]
    gpu_refs = [[FakeRef(0, 64), FakeRef(1, 8)]]
    cw._glm53_gpu_caches = SimpleNamespace(
        tensors=gpu_tensors, group_data_refs=gpu_refs
    )
    return cw, writer, ns, tensors, gpu_tensors


def _load_restore_helpers(store_mod, logger) -> dict:
    """Exec the store writer + restore worker sources with one logger, the
    same composition the patched worker.py carries."""
    ns: dict = {"logger": logger}
    src = (
        store_mod.MODE_HELPERS_SRC
        + store_mod.CODEC_SRC
        + _restore_patched_writer_src()
        + restore.RESTORE_WORKER_SRC
    )
    exec(compile(src, "<restore-test helpers>", "exec"), ns)
    return ns


def _restore_patched_writer_src() -> str:
    """The store WRITER_SRC with the restore overlay's writer sites applied
    (W7-W12 anchor inside the writer text), so tests drive the same composed
    class the container runs."""
    src = store.WRITER_SRC
    for _n, _m, anchor, patched in restore.SITES_WORKER:
        if anchor in src:
            src = src.replace(anchor, patched, 1)
    return src


def _hash(i: int) -> str:
    import hashlib

    return hashlib.sha256(b"g0chunk%d" % i).hexdigest()


def _g0_job_meta(n_boundaries: int):
    keys = [(_hash(k), 0, k, 3584) for k in range(n_boundaries)]
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
        "cow_groups": [],
        "full_groups": [0],
        "manifests": manifests,
    }


def _load_spec(writer, n: int, start: int = 0):
    return {
        "glm53_disk_load": 1,
        "v": 1,
        "namespace_hash": writer._namespace_hash,
        "boundary_token_index": n * 3584,
        "entries": [(_hash(k), 0, k, 3584) for k in range(start, n)],
    }


class _TestLogger:
    def __init__(self):
        self.lines = []

    def _fmt(self, msg, *args):
        try:
            self.lines.append(msg % args if args else str(msg))
        except TypeError:
            self.lines.append(str(msg))

    info = debug = warning = error = exception = _fmt


def test_disk_loader() -> None:
    print("Part C: worker-side synchronous disk loader")
    import _kv_offload_stub_env as stub

    if "vllm" not in sys.modules:
        stub.install_fake_vllm(with_restore=True)
    _install_fake_torch()

    logger = _TestLogger()
    with tempfile.TemporaryDirectory() as td:
        os.environ["GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"] = "0"
        cw, writer, ns, staging, gpu_tensors = _g0_env(store, Path(td), logger)
        check(writer._disabled_reason is None, "C1 writer up on the g0 layout")

        # Store three boundaries (real writer path => real files+manifests).
        meta = _g0_job_meta(3)
        rows = list(range(3))
        expected = [
            staging[0].rows[r][:64] + staging[1].rows[r][:8] for r in rows
        ]
        writer.capture_job(1, meta, rows)
        writer.shutdown(timeout=20)
        base = Path(writer._base)
        check(
            len(list(base.rglob("*.bin"))) == 3
            and len(list(base.rglob("manifests/**/*.json"))) == 3,
            "C2 zero-cow store: 3 chunk files + 3 boundary manifests",
        )

        # C3 happy path: byte-exact restore into the GPU tensors.
        dst = SimpleNamespace(block_ids=[10, 11, 12])
        ns["_glm53_run_disk_load"](cw, 77, _load_spec(writer, 3), dst)
        check(cw._glm53_disk_done == [77], "C3 job completes (done queue)")
        check(not cw._glm53_load_error_ids, "C3b no load errors on happy path")
        got = [
            gpu_tensors[0].tensor.row_bytes(10 + i, 64)
            + gpu_tensors[1].tensor.row_bytes(10 + i, 8)
            for i in range(3)
        ]
        check(
            got == expected and any(any(b) for b in expected),
            "C3c GPU bytes == original staging bytes (byte-exact restore)",
        )
        cw._glm53_disk_done.clear()

        # C4 failure matrix: corrupt chunk 1 => chunk 0 restored, 1+2 failed
        # and zero-filled, job still completes.
        victim = Path(writer._file_path(_hash(1), 0))
        blob = victim.read_bytes()
        victim.write_bytes(blob[:-3] + b"\x00\x00\x00")
        for t in gpu_tensors:
            t.tensor.backing[:] = os.urandom(len(t.tensor.backing))
        pre_fail_row2 = gpu_tensors[0].tensor.row_bytes(22, 64)
        ns["_glm53_run_disk_load"](
            cw, 78, _load_spec(writer, 3), SimpleNamespace(block_ids=[20, 21, 22])
        )
        check(
            cw._glm53_disk_done == [78]
            and cw._glm53_load_error_ids == {21, 22},
            f"C4 CRC failure at chunk 1: ids 21+22 invalid, job done "
            f"(got {cw._glm53_load_error_ids})",
        )
        check(
            gpu_tensors[0].tensor.row_bytes(20, 64) == expected[0][:64]
            and gpu_tensors[0].tensor.row_bytes(21, 64) == bytes(64)
            and gpu_tensors[1].tensor.row_bytes(22, 8) == bytes(8)
            and gpu_tensors[0].tensor.row_bytes(22, 64) != pre_fail_row2,
            "C4b earlier chunk restored; failed suffix zero-filled",
        )
        victim.write_bytes(blob)  # repair
        cw._glm53_disk_done.clear()
        cw._glm53_load_error_ids.clear()

        # C5 missing chunk file.
        gone = Path(writer._file_path(_hash(2), 0))
        gone_blob = gone.read_bytes()
        gone.unlink()
        ns["_glm53_run_disk_load"](
            cw, 79, _load_spec(writer, 3), SimpleNamespace(block_ids=[30, 31, 32])
        )
        check(
            cw._glm53_disk_done == [79] and cw._glm53_load_error_ids == {32},
            "C5 missing chunk file: only its suffix invalid, job done",
        )
        gone.write_bytes(gone_blob)
        cw._glm53_disk_done.clear()
        cw._glm53_load_error_ids.clear()

        # C6 wrong namespace => whole job failed (all ids), still done.
        bad = _load_spec(writer, 3)
        bad["namespace_hash"] = "deadbeef"
        ns["_glm53_run_disk_load"](
            cw, 80, bad, SimpleNamespace(block_ids=[40, 41, 42])
        )
        check(
            cw._glm53_disk_done == [80]
            and cw._glm53_load_error_ids == {40, 41, 42},
            "C6 namespace mismatch: every chunk failed, job done",
        )
        cw._glm53_disk_done.clear()
        cw._glm53_load_error_ids.clear()

        # C7 manifest missing at load time (retention race) => all failed.
        mpath = Path(writer._manifest_path(_hash(2)))
        mblob = mpath.read_bytes()
        mpath.unlink()
        ns["_glm53_run_disk_load"](
            cw, 81, _load_spec(writer, 3), SimpleNamespace(block_ids=[50, 51, 52])
        )
        check(
            cw._glm53_disk_done == [81]
            and cw._glm53_load_error_ids == {50, 51, 52},
            "C7 manifest gone at load: all chunks failed (T4 degradation)",
        )
        mpath.write_bytes(mblob)
        cw._glm53_disk_done.clear()
        cw._glm53_load_error_ids.clear()

        # C8 wrong-rank chunk header rejected.
        h0path = Path(writer._file_path(_hash(0), 0))
        h0_orig_blob = h0path.read_bytes()
        h0 = store.read_chunk_header(str(h0path), verify_payload=True)
        payload = h0path.read_bytes()[16 + len(json.dumps(h0, sort_keys=True,
                                                          separators=(",", ":")).encode()):]
        # (re-encode with tp_rank flipped; payload re-derived via the codec)
        raw = h0path.read_bytes()
        hlen = int.from_bytes(raw[8:12], "little")
        payload = raw[16 + hlen:]
        h_bad = {k: v for k, v in h0.items()
                 if k not in ("payload_len", "payload_crc32", "crc_algo",
                              "format_version")}
        h_bad["tp_rank"] = 1
        h0path.write_bytes(store.encode_chunk(h_bad, payload))
        ns["_glm53_run_disk_load"](
            cw, 82, _load_spec(writer, 3), SimpleNamespace(block_ids=[60, 61, 62])
        )
        check(
            cw._glm53_disk_done == [82]
            and cw._glm53_load_error_ids == {60, 61, 62},
            "C8 wrong-rank header: failed from chunk 0 (never cross-rank bytes)",
        )
        h0path.write_bytes(h0_orig_blob)
        cw._glm53_disk_done.clear()
        cw._glm53_load_error_ids.clear()

        # C9 null dst block refused.
        ns["_glm53_run_disk_load"](
            cw, 83, _load_spec(writer, 1), SimpleNamespace(block_ids=[0])
        )
        check(
            cw._glm53_disk_done == [83] and 0 not in cw._glm53_load_error_ids,
            "C9 null block 0 never enters the invalid set (DS4F rule)",
        )
        cw._glm53_disk_done.clear()
        cw._glm53_load_error_ids.clear()

        # C10 (review f11): truncated chunk file.
        v0 = Path(writer._file_path(_hash(0), 0))
        v0_blob = v0.read_bytes()
        v0.write_bytes(v0_blob[:-5])
        ns["_glm53_run_disk_load"](
            cw, 84, _load_spec(writer, 3), SimpleNamespace(block_ids=[70, 71, 72])
        )
        check(
            cw._glm53_disk_done == [84]
            and cw._glm53_load_error_ids == {70, 71, 72},
            "C10 truncated chunk: failed from chunk 0, job done",
        )
        v0.write_bytes(v0_blob)
        cw._glm53_disk_done.clear()
        cw._glm53_load_error_ids.clear()

        # C11: wrong spec_kind header rejected.
        h0 = store.read_chunk_header(str(v0))
        raw = v0.read_bytes()
        hlen = int.from_bytes(raw[8:12], "little")
        payload0 = raw[16 + hlen :]
        h_bad = {
            k: v
            for k, v in h0.items()
            if k not in ("payload_len", "payload_crc32", "crc_algo", "format_version")
        }
        h_bad["spec_kind"] = "MambaSpec"
        v0.write_bytes(store.encode_chunk(h_bad, payload0))
        ns["_glm53_run_disk_load"](
            cw, 85, _load_spec(writer, 1), SimpleNamespace(block_ids=[75])
        )
        check(
            cw._glm53_disk_done == [85] and cw._glm53_load_error_ids == {75},
            "C11 wrong spec_kind rejected",
        )
        v0.write_bytes(v0_blob)
        cw._glm53_disk_done.clear()
        cw._glm53_load_error_ids.clear()

        # C12: segment-table mismatch (tampered dtype) rejected.
        h_bad2 = {
            k: v
            for k, v in h0.items()
            if k not in ("payload_len", "payload_crc32", "crc_algo", "format_version")
        }
        import copy

        h_bad2["segment_table"] = copy.deepcopy(h0["segment_table"])
        h_bad2["segment_table"][0]["dtype"] = "torch.float16"
        v0.write_bytes(store.encode_chunk(h_bad2, payload0))
        ns["_glm53_run_disk_load"](
            cw, 86, _load_spec(writer, 1), SimpleNamespace(block_ids=[76])
        )
        check(
            cw._glm53_disk_done == [86] and cw._glm53_load_error_ids == {76},
            "C12 segment-table mismatch rejected (byte count alone not enough)",
        )
        v0.write_bytes(v0_blob)
        cw._glm53_disk_done.clear()
        cw._glm53_load_error_ids.clear()

        # C13: manifest chain tampered => all chunks failed.
        m2p = Path(writer._manifest_path(_hash(2)))
        m2_raw = json.loads(m2p.read_text())
        m2_bad = dict(m2_raw)
        m2_bad["chunk_hashes"] = [_hash(9), _hash(1), _hash(2)]
        m2p.write_text(json.dumps(m2_bad, sort_keys=True))
        ns["_glm53_run_disk_load"](
            cw, 87, _load_spec(writer, 3), SimpleNamespace(block_ids=[80, 81, 82])
        )
        check(
            cw._glm53_disk_done == [87]
            and cw._glm53_load_error_ids == {80, 81, 82},
            "C13 manifest chain divergence: all chunks failed",
        )
        m2p.write_text(json.dumps(m2_raw, sort_keys=True))
        cw._glm53_disk_done.clear()
        cw._glm53_load_error_ids.clear()

        # C14: manifest per-chunk CRC ledger mismatch => that chunk fails.
        m2_bad2 = json.loads(json.dumps(m2_raw))
        m2_bad2["full_groups"]["0"][1][2] ^= 0xFF
        m2p.write_text(json.dumps(m2_bad2, sort_keys=True))
        ns["_glm53_run_disk_load"](
            cw, 88, _load_spec(writer, 3), SimpleNamespace(block_ids=[90, 91, 92])
        )
        check(
            cw._glm53_disk_done == [88]
            and cw._glm53_load_error_ids == {91, 92},
            "C14 ledger CRC mismatch: chunk 1 suffix failed",
        )
        m2p.write_text(json.dumps(m2_raw, sort_keys=True))
        cw._glm53_disk_done.clear()
        cw._glm53_load_error_ids.clear()

        # C15 (review f3): unidentifiable destinations PROPAGATE (fail-stop,
        # never an ack that could serve stale bytes).
        try:
            ns["_glm53_run_disk_load"](
                cw, 89, _load_spec(writer, 1),
                SimpleNamespace(block_ids=["not-an-int"]),
            )
            check(False, "C15 malformed dst_spec propagates (fail-stop)")
        except (TypeError, ValueError):
            check(
                89 not in cw._glm53_disk_done,
                "C15 malformed dst_spec propagates (fail-stop)",
            )
    _cleanup_env()


# ------------------------------------------------------------------ part D --
def test_event_channel() -> None:
    print("Part D: manifest-event channel + facade drain")
    import _kv_offload_stub_env as stub

    if "vllm" not in sys.modules:
        stub.install_fake_vllm(with_restore=True)
    _install_fake_torch()
    logger = _TestLogger()

    with tempfile.TemporaryDirectory() as td:
        os.environ["GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"] = "1"
        cw, writer, ns, _staging, _gpu = _g0_env(store, Path(td), logger)
        writer._keep_boundaries = 1
        writer.capture_job(1, _g0_job_meta(2), [0, 1])
        writer.shutdown(timeout=20)
        events = writer.drain_manifest_events()
        codes = [e[0] for e in events]
        check(
            codes.count("+") == 2
            and codes.count("-") == 1
            and events[0][1] == 0
            and events[0][2] == writer._namespace_hash,
            f"D1 publish x2 + supersede x1 events with rank+namespace (got {codes})",
        )
        check(
            writer.drain_manifest_events() == [],
            "D1b drain empties the queue",
        )

        # D2 write-failure event ("F") from an ineligible-group key.
        bad_meta = {
            "v": 1,
            "keys": [(_hash(9), 5, 0, 3584)],
            "cow_groups": [],
            "full_groups": [0],
            "manifests": [],
        }
        writer.capture_job(2, bad_meta, [7])
        writer.shutdown(timeout=20)
        evs = writer.drain_manifest_events()
        check(
            any(e[0] == "F" and e[3] == _hash(9) and e[4] == 5 for e in evs),
            f"D2 write failure emits an F event (got {evs})",
        )

        # D3 validated dedup: republish over an existing valid manifest
        # re-emits '+'; a corrupted manifest is rewritten instead.
        writer2 = ns["Glm53LocalStoreWriter"](cw, logger)
        writer2._keep_boundaries = 0
        writer2.capture_job(3, _g0_job_meta(2), [0, 1])
        writer2.shutdown(timeout=20)
        evs2 = writer2.drain_manifest_events()
        check(
            [e[0] for e in evs2].count("+") == 2,
            "D3 dedup path validates then re-announces existing manifests",
        )
        mpath = Path(writer2._manifest_path(_hash(0)))
        mpath.write_text("{ corrupt")
        writer3 = ns["Glm53LocalStoreWriter"](cw, logger)
        writer3._keep_boundaries = 0
        writer3.capture_job(4, _g0_job_meta(1), [0])
        writer3.shutdown(timeout=20)
        man = json.loads(mpath.read_text())
        check(
            man.get("chunk_hashes") == [_hash(0)]
            and [e[0] for e in writer3.drain_manifest_events()] == ["+"],
            "D3b corrupt existing manifest is rewritten, then announced",
        )

    # D4 worker-meta plumbing: event-only meta flows; aggregate concatenates;
    # legacy meta without the field tolerated; pickle round-trip.
    common = sys.modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common"
    ]
    m1 = common.OffloadingWorkerMetadata()
    m1.glm53_manifest_events.append(("+", 0, "ns", "h", 3584, "b"))
    check(
        bool(m1.glm53_manifest_events) and not m1.completed_jobs,
        "D4 event-only meta constructible",
    )
    m2 = common.OffloadingWorkerMetadata(completed_jobs={5: 1})
    m2.glm53_manifest_events.append(("+", 1, "ns", "h", 3584, "b"))
    agg = m1.aggregate(m2)
    check(
        len(agg.glm53_manifest_events) == 2 and agg.completed_jobs == {5: 1},
        "D4b aggregate() concatenates events and keeps completions",
    )
    legacy = common.OffloadingWorkerMetadata(completed_jobs={6: 1})
    del legacy.__dict__["glm53_manifest_events"]
    agg2 = m1.aggregate(legacy)
    check(
        len(agg2.glm53_manifest_events) == 1,
        "D4c legacy meta without the field aggregates via getattr",
    )
    rt = pickle.loads(pickle.dumps(agg))
    check(
        rt.glm53_manifest_events == agg.glm53_manifest_events,
        "D4d events pickle round-trip on the wire",
    )

    # D5 patched connector-worker: build_connector_worker_meta returns an
    # event-only meta; get_finished drains disk loads into finished_recving.
    worker_mod = sys.modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker"
    ]
    cw2 = object.__new__(worker_mod.OffloadingConnectorWorker)
    cw2._connector_worker_meta = common.OffloadingWorkerMetadata()
    cw2._glm53_store_writer = None
    cw2._glm53_disk_done = []
    cw2._glm53_load_error_ids = set()
    cw2._load_jobs = {}
    cw2._unsubmitted_store_jobs = []
    check(
        cw2.build_connector_worker_meta() is None,
        "D5 no completions + no events => meta None (stock behavior kept)",
    )
    cw2._connector_worker_meta.glm53_manifest_events.append(
        ("+", 0, "ns", "h", 3584, "b")
    )
    meta = cw2.build_connector_worker_meta()
    check(
        meta is not None and len(meta.glm53_manifest_events) == 1,
        "D5b event-only meta FLOWS (finding 7)",
    )
    cw2.worker = SimpleNamespace(get_finished=lambda: [])
    cw2._glm53_disk_done = [11]
    cw2._load_jobs = {11: "req-z"}
    sent, recv = cw2.get_finished(set())
    check(
        recv == {"req-z"}
        and cw2._connector_worker_meta.completed_jobs.get(11) == 1,
        "D5c get_finished drains disk loads: ack + finished_recving",
    )

    # D6 facade drain (patched facade text, constructor bypassed).
    fac_text, _ = restore.prepare(
        (FIXTURES / "image_487ecf187_offloading_connector.py").read_text(),
        restore.SITES_FACADE,
        "facade",
    )
    import re as _re

    m = _re.search(
        r"    def get_block_ids_with_load_errors.*?\n(?=    def )",
        fac_text,
        _re.S,
    )
    fac_ns: dict = {}
    exec(  # drive the patched method text verbatim
        compile(
            "class _F:\n" + m.group(0) + "    pass\n", "facade-method", "exec"
        ),
        fac_ns,
    )
    f = fac_ns["_F"]()
    f.connector_worker = None
    check(f.get_block_ids_with_load_errors() == set(), "D6 no worker => empty")
    f.connector_worker = SimpleNamespace(_glm53_load_error_ids={3, 4})
    got = f.get_block_ids_with_load_errors()
    check(
        got == {3, 4}
        and f.connector_worker._glm53_load_error_ids == set()
        and f.get_block_ids_with_load_errors() == set(),
        "D6b facade drains-and-clears the error set",
    )
    _cleanup_env()


# ------------------------------------------------------------------ part E --
def _install_single_type_stub():
    """Extra stub modules the patched single_type manager imports."""
    import _kv_offload_stub_env as stub

    def _mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    m = _mod("vllm.v1.core.block_pool")
    m.BlockPool = object
    m = _mod("vllm.v1.core.kv_cache_utils")
    for n in (
        "BlockHashList",
        "BlockHashListWithBlockSize",
        "BlockHashWithGroupId",
        "KVCacheBlock",
        "resolve_block_hashes",
    ):
        setattr(m, n, object)
    kci = sys.modules["vllm.v1.kv_cache_interface"]
    for n in (
        "CrossAttentionSpec",
        "HiddenStateCacheSpec",
        "KpoolTailSpec",
        "RSWASpec",
        "SinkFullAttentionSpec",
        "SlidingWindowMLASpec",
        "TQFullAttentionSpec",
    ):
        if not hasattr(kci, n):
            setattr(m if False else kci, n, type(n, (stub.KVCacheSpec,), {}))
    m = _mod("vllm.v1.kv_cache_spec_registry")

    class KVCacheSpecRegistry:
        @staticmethod
        def register(*a, **k):
            pass

    m.KVCacheSpecRegistry = KVCacheSpecRegistry


def test_single_type_t2() -> None:
    print("Part E: T2 reachable_boundaries (patched single_type text)")
    import _kv_offload_stub_env as stub

    if "vllm" not in sys.modules:
        stub.install_fake_vllm(with_restore=True)
    _install_single_type_stub()
    st_text, _ = restore.prepare(
        (FIXTURES / "image_487ecf187_single_type_kv_cache_manager.py").read_text(),
        restore.SITES_SINGLE_TYPE,
        "st",
    )
    st_mod = types.ModuleType("patched_single_type")
    exec(compile(st_text, "single_type_kv_cache_manager.py", "exec"),
         st_mod.__dict__)

    recorded = {}

    class Recorder:
        def cache_full_blocks(self, **kw):
            recorded["block_mask"] = kw.get("block_mask")

    def _mask_recorder(**kw):
        recorded["reachable_boundaries"] = list(kw["reachable_boundaries"])
        return None

    def _drive(request, retention_interval=0):
        recorded.clear()
        self_stub = SimpleNamespace(
            num_cached_block={},
            block_size=3584,
            scheduler_block_size=3584,
            kv_cache_spec=None,
            use_eagle=False,
            kv_cache_group_id=0,
            block_pool=Recorder(),
            req_to_blocks={request.request_id: [None, None]},
            reachable_block_mask=_mask_recorder,
        )
        st_mod.SingleTypeKVCacheManager.cache_blocks(
            self_stub, request, 7168, retention_interval=retention_interval
        )
        return recorded.get("reachable_boundaries", [])

    req = SimpleNamespace(
        request_id="r1", num_prompt_tokens=9000, shared_prefix_boundary=0
    )
    rb = _drive(req)
    check(
        rb == [8999],
        f"E1 absent attribute: reachable_boundaries unchanged (got {rb})",
    )
    req.glm53_restored_boundary = 7168
    rb = _drive(req)
    check(
        rb == [8999, 7168],
        f"E2 restored boundary appended END-EXCLUSIVE (got {rb})",
    )
    req.glm53_restored_boundary = 0
    rb = _drive(req)
    check(rb == [8999], "E3 zero value is ignored (falsy guard)")
    req.shared_prefix_boundary = 3584
    req.glm53_restored_boundary = 7168
    rb = _drive(req)
    check(
        rb == [8999, 3584, 7168],
        "E4 composes with shared_prefix_boundary (both appended)",
    )

    # E5 MambaManager.reachable_block_mask keeps exactly the restored
    # boundary's state block under retention 0 (end-exclusive convention).
    spec = stub.MambaSpec(3584)
    mask = st_mod.MambaManager.reachable_block_mask(
        start_block=0,
        end_block=3,
        alignment_tokens=3584,
        kv_cache_spec=spec,
        use_eagle=False,
        retention_interval=0,
        reachable_boundaries=[7168],
    )
    check(mask == [False, True, False], f"E5 7168 keeps mamba block 1 (got {mask})")
    mask = st_mod.MambaManager.reachable_block_mask(
        start_block=0,
        end_block=3,
        alignment_tokens=3584,
        kv_cache_spec=spec,
        use_eagle=False,
        retention_interval=0,
        reachable_boundaries=[3584],
    )
    check(mask == [True, False, False], f"E5b 3584 keeps block 0 (got {mask})")
    mask = st_mod.MambaManager.reachable_block_mask(
        start_block=0,
        end_block=3,
        alignment_tokens=3584,
        kv_cache_spec=spec,
        use_eagle=False,
        retention_interval=0,
        reachable_boundaries=[3583],
    )
    check(
        mask == [False, False, False],
        f"E5c 3583 (not end-exclusive-aligned) keeps nothing (got {mask})",
    )


def _cleanup_env() -> None:
    for name in (
        "GLM53_KV_OFFLOAD",
        "GLM53_KV_OFFLOAD_DIR",
        "GLM53_KV_OFFLOAD_CPU_GB",
        "GLM53_KV_OFFLOAD_RESTORE",
        "GLM53_KV_OFFLOAD_DRAFTER",
        "GLM53_KV_OFFLOAD_KEEP_BOUNDARIES",
    ):
        os.environ.pop(name, None)


def main() -> int:
    test_patcher_hygiene()
    test_scheduler_restore()
    test_disk_loader()
    test_event_channel()
    test_single_type_t2()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
