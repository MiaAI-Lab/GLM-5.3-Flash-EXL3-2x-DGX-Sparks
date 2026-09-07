#!/usr/bin/env python3
"""Regression tests for overlay/patch_kv_offload_scope.py (port of vllm#54743).

A  Patcher mechanics on the PINNED image fixtures (byte-exact captures of the
   deployed tree, tests/fixtures/README.md): preflight, apply, idempotence,
   tamper/drift refusal, half-patched refusal, compile of the patched text.
B  Eligibility predicate (the exec'd shipped helper, not a replica): kpool
   scratch never eligible; drafter excluded only under EAGLE context and only
   while GLM53_KV_OFFLOAD_DRAFTER=0; strict 0/1 env parsing.
C  Patched-runtime drive (Codex OFFLOAD1 finding 6): the PATCHED
   build_offloading_config / scheduler module executed under a stubbed vllm
   namespace on the 7-group boot layout — eligible set {0,2,3,4,5}, original
   indices in keys, full-length group_sizes with zeros for ineligible groups,
   num_kv_cache_groups=7, mamba alignment 3584, supports_partial_tail False,
   no-scratch uniform model unchanged (upstream-equivalence guard).
D  In-image preflight when GLM53_REQUIRE_TARGET=1 (Docker build): the real
   tree must accept the anchors before the patcher bakes them in.

Run:  python3 tests/test_kv_offload_scope.py   (or pytest)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str((ROOT / "overlay")))
# In-image the test rides next to the overlay under /opt/glm53.
sys.path.insert(0, str(HERE.parent))

import patch_kv_offload_scope as scope  # noqa: E402

FIXTURES = HERE / "fixtures"
IN_IMAGE = not FIXTURES.is_dir()

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


# ------------------------------------------------------------------ part A --
def test_patcher_mechanics() -> None:
    if IN_IMAGE:
        print("Part A: skipped (fixtures not shipped in-image; D covers the real tree)")
        return
    print("Part A: patcher mechanics on pinned image fixtures")
    cases = [
        ("image_487ecf187_kv_offload_config.py", scope.SITES_KVO_CONFIG),
        ("image_487ecf187_offloading_config.py", scope.SITES_CONN_CONFIG),
        ("image_487ecf187_offloading_scheduler.py", scope.SITES_SCHED),
    ]
    for name, sites in cases:
        src = _fixture(name)
        patched, action = scope.prepare(src, sites, name)
        check(action == "patched" and patched != src, f"A1 {name}: applies")
        try:
            compile(patched, name, "exec")
            check(True, f"A2 {name}: patched text compiles")
        except SyntaxError as exc:
            check(False, f"A2 {name}: patched text compiles ({exc})")
        again, action2 = scope.prepare(patched, sites, name)
        check(
            action2 == "already present" and again == patched,
            f"A3 {name}: idempotent second run",
        )
        # Drift: tamper the first anchor (drop its first character — always a
        # real mutation regardless of the anchor's shape).
        _n, _m, anchor, _p = sites[0]
        broken = src.replace(anchor, anchor[1:], 1)
        assert broken != src
        try:
            scope.prepare(broken, sites, name)
            check(False, f"A4 {name}: tampered anchor refused")
        except ValueError:
            check(True, f"A4 {name}: tampered anchor refused")
        # Half-patched: one mark present, others missing.
        if len(sites) > 1:
            half = src
            n0, m0, a0, p0 = sites[0]
            half = half.replace(a0, p0, 1)
            try:
                scope.prepare(half, sites, name)
                check(False, f"A5 {name}: half-patched file refused")
            except ValueError:
                check(True, f"A5 {name}: half-patched file refused")


# ------------------------------------------------------------------ part B --
def test_eligibility_predicate() -> None:
    print("Part B: eligibility predicate (exec'd shipped helper)")
    from _kv_offload_stub_env import boot_layout

    layout = boot_layout().kv_cache_groups
    layout[6].is_eagle_group = False  # GLM sets no flag; EAGLE context decides

    def eligible(drafter: bool, use_eagle: bool = True) -> list[int]:
        return [
            i
            for i, g in enumerate(layout)
            if scope.kvo_group_eligible(g, drafter, use_eagle)
        ]

    check(eligible(False) == [0, 2, 3, 4, 5], "B1 default eligible set {0,2,3,4,5}")
    check(
        eligible(True) == [0, 2, 3, 4, 5, 6],
        "B2 GLM53_KV_OFFLOAD_DRAFTER=1 adds only the drafter group",
    )
    check(
        eligible(False, use_eagle=False) == [0, 2, 3, 4, 5, 6],
        "B3 without EAGLE context a SlidingWindowSpec group is genuine SWA and stays",
    )
    layout[6].is_eagle_group = True
    check(
        eligible(False, use_eagle=False) == [0, 2, 3, 4, 5],
        "B4 is_eagle_group excludes even without speculative context",
    )
    layout[6].is_eagle_group = False
    check(1 not in eligible(True), "B5 kpool scratch is NEVER eligible")

    for raw, expect in ((None, False), ("0", False), ("1", True)):
        if raw is None:
            os.environ.pop("GLM53_KV_OFFLOAD_DRAFTER", None)
        else:
            os.environ["GLM53_KV_OFFLOAD_DRAFTER"] = raw
        check(
            scope.kvo_drafter_included() is expect,
            f"B6 GLM53_KV_OFFLOAD_DRAFTER={raw!r} -> {expect}",
        )
    for junk in ("", "2", "yes", " 1", "01"):
        os.environ["GLM53_KV_OFFLOAD_DRAFTER"] = junk
        try:
            scope.kvo_drafter_included()
            check(False, f"B7 junk {junk!r} raises")
        except ValueError:
            check(True, f"B7 junk {junk!r} raises")
    os.environ.pop("GLM53_KV_OFFLOAD_DRAFTER", None)


# ------------------------------------------------------------------ part C --
def test_patched_runtime() -> None:
    if IN_IMAGE:
        print("Part C: skipped in-image (fixtures not shipped)")
        return
    print("Part C: patched-runtime drive (stubbed vllm, PATCHED module text)")
    os.environ.pop("GLM53_KV_OFFLOAD_DRAFTER", None)
    os.environ["GLM53_KV_OFFLOAD"] = "0"
    os.environ["GLM53_KV_OFFLOAD_RESTORE"] = "0"
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
    check(
        [g.group_idx for g in cfg.groups] == [0, 2, 3, 4, 5],
        "C1 build_offloading_config scopes to {0,2,3,4,5} and boots (no g1 assert crash)",
    )
    check(
        all(g.tokens_per_block == 3584 for g in cfg.groups),
        "C2 every eligible group is on the 3584 grid",
    )
    log_lines = "\n".join(mods["logger"].lines)
    check(
        "eligible groups" in log_lines and "[0, 2, 3, 4, 5]" in log_lines,
        "C3 boot log names the eligible set",
    )

    sched_mod = mods["scheduler"]
    spec = FakeSchedSpec(mods, cfg)
    sched_cfg = sched_mod.SchedulerOffloadConfig.from_spec(
        spec, vllm_config, kv_cache_config
    )
    check(sched_cfg.num_kv_cache_groups == 7, "C4 num_kv_cache_groups is the FULL count")
    check(
        tuple(g.group_idx for g in sched_cfg.kv_group_configs) == (0, 2, 3, 4, 5),
        "C5 kv_group_configs keep original indices",
    )
    check(
        sched_mod.resolve_mamba_align_size(spec, kv_cache_config) == 3584,
        "C6 mamba alignment resolves to 3584 over eligible groups",
    )
    check(not sched_cfg.supports_partial_tail, "C7 partial tails off (scratch groups)")

    scheduler = sched_mod.OffloadingConnectorScheduler(
        spec, vllm_config, kv_cache_config
    )
    check(
        sorted(scheduler._group_config_by_idx) == [0, 2, 3, 4, 5],
        "C8 _group_config_by_idx keyed by original indices",
    )
    check(
        scheduler._sliding_window_groups == (2, 3, 4, 5)
        and scheduler._lookup_groups[0] == 0,
        "C9 mamba groups looked up as window groups after the full-attn group",
    )

    req = FakeRequest(num_tokens=7 * 3584 + 100)
    scheduler.on_new_request(req)
    st = scheduler._req_status[req.request_id]
    check(len(st.group_states) == 7, "C10 group_states sized by the FULL group count")

    num_hit, is_async = scheduler.get_num_new_matched_tokens(req, 0)
    check(
        (num_hit, is_async) == (0, False),
        "C11 store-only mode: external hits are always (0, False)",
    )
    check(
        spec._manager.lookup_calls == [],
        "C12 store-only mode: the manager lookup is NEVER consulted",
    )

    base = mods["base"]
    gidxs = set()
    for g_idx, gs in enumerate(st.group_states):
        for key in gs.offload_keys:
            gidxs.add(base.get_offload_group_idx(key))
            check_hash = base.get_offload_block_hash(key)
            assert check_hash in req.block_hashes
    check(
        gidxs == {0, 2, 3, 4, 5},
        f"C13 offload keys carry ONLY eligible original indices (got {sorted(gidxs)})",
    )
    n_expected = (req.num_tokens // 3584)
    check(
        all(
            len(st.group_states[g].offload_keys) == n_expected
            for g in (0, 2, 3, 4, 5)
        )
        and all(len(st.group_states[g].offload_keys) == 0 for g in (1, 6)),
        "C14 per-group key counts: full chunks for eligible, zero for scratch/drafter",
    )

    # update_state_after_alloc: full-length layout, zeros at 1 and 6.
    class FakeBlock:
        def __init__(self, block_id):
            self.block_id = block_id
            self.is_null = False
            self.block_hash = None

    kvm = mods["kv_cache_manager"]
    n_cached = 3584
    per_group_needed = [1, 896, 1, 1, 1, 1, 56]
    blocks = kvm.KVCacheBlocks(
        tuple(
            [FakeBlock(100 * g + j + 1) for j in range(per_group_needed[g])]
            for g in range(7)
        )
    )
    st.num_locally_computed_tokens = 0
    scheduler.update_state_after_alloc(req, blocks, n_cached)
    job = next(iter(scheduler._current_batch_load_jobs.values()))
    gs = job.dst_spec.group_sizes
    check(
        len(gs) == 7 and gs[1] == 0 and gs[6] == 0 and gs[0] == 1,
        f"C15 load-job group_sizes full-length with zeros at 1/6 (got {gs})",
    )
    check(
        len(job.dst_spec.block_indices) == 7,
        "C16 block_indices full-length",
    )

    # Upstream-equivalence guard: a uniform no-scratch model is unchanged.
    from _kv_offload_stub_env import (
        FakeKVCacheConfig,
        FakeKVCacheGroup,
        MLAAttentionSpec,
    )

    uniform = FakeKVCacheConfig(
        [
            FakeKVCacheGroup(MLAAttentionSpec(64), ("l0", "l1")),
        ]
    )
    vllm_plain = FakeVllmConfig(
        kv_transfer_config=FakeKVTransferConfig(), speculative_config=None
    )
    cfg_u = build_offloading_config(mods, vllm_plain, uniform)
    check(
        [g.group_idx for g in cfg_u.groups] == [0],
        "C17 no-scratch uniform model keeps every group (behaviour unchanged)",
    )

    # Divisibility regression: a prefix-cacheable misaligned group still asserts.
    bad = boot_layout()
    bad.kv_cache_groups[2].kv_cache_spec.block_size = 96  # 96 % 64 != 0
    try:
        build_offloading_config(mods, vllm_config, bad)
        check(False, "C18 misaligned CACHEABLE group still asserts at boot")
    except AssertionError:
        check(True, "C18 misaligned CACHEABLE group still asserts at boot")
    _cleanup_env()


# ------------------------------------------------------------------ part D --
def test_in_image_preflight() -> None:
    if os.environ.get("GLM53_REQUIRE_TARGET") != "1":
        print("Part D: skipped (GLM53_REQUIRE_TARGET!=1)")
        return
    print("Part D: real-tree preflight (in-image)")
    rc = 0
    try:
        scope.main(["patch_kv_offload_scope.py", "--preflight"])
    except SystemExit as exc:
        rc = int(exc.code or 0)
    check(rc == 0, "D1 patcher preflight accepts the real in-image tree")
    _cleanup_env()




def _cleanup_env() -> None:
    """These tests mutate GLM53_KV_OFFLOAD* in os.environ; scrub them so the
    rest of the suite (and any subprocess-driving test that inherits the
    process env) sees a clean slate."""
    for name in ("GLM53_KV_OFFLOAD","GLM53_KV_OFFLOAD_DIR","GLM53_KV_OFFLOAD_CPU_GB","GLM53_KV_OFFLOAD_RESTORE","GLM53_KV_OFFLOAD_DRAFTER","GLM53_KV_OFFLOAD_KEEP_BOUNDARIES"):
        os.environ.pop(name, None)


def main() -> int:
    test_patcher_mechanics()
    test_eligibility_predicate()
    test_patched_runtime()
    test_in_image_preflight()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
