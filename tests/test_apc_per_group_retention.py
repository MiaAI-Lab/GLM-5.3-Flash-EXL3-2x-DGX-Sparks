#!/usr/bin/env python3
"""Host test for patch_apc_per_group_retention.py (no GPU, no vLLM import).

Applies the overlay to *copies* of kv_cache_coordinator.py and block_pool.py,
then exercises the injected helpers and eviction policy directly. Fails closed:
any problem exits non-zero.

    GLM53_KV_COORDINATOR_PY_SRC=/path/to/fork/kv_cache_coordinator.py \\
    GLM53_BLOCK_POOL_PY_SRC=/path/to/fork/block_pool.py \\
        python3 test_apc_per_group_retention.py

`..._SRC` may already carry overlay/patch_hybrid_prefix_hit.py. The overlay
*composition* case additionally needs a pristine (unpatched) copy of the same
file; it is taken from `GLM53_KV_COORDINATOR_PY_PRISTINE`, else from `..._SRC`
itself when that is unpatched, else from /tmp/kv_cache_coordinator_pristine.py.
Default source is the in-container path.
"""

from __future__ import annotations

import ast
import copy
from collections import namedtuple
from collections.abc import Iterable
import difflib
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find(*names: str) -> Path | None:
    for name in names:
        for cand in (HERE / name, HERE.parent / "overlay" / name):
            if cand.is_file():
                return cand
    return None


PATCH = _find("patch_apc_per_group_retention.py")
MIA_PATCH = _find("patch_hybrid_prefix_hit.py")
DEFAULT_SRC = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py"
)
DEFAULT_BP_SRC = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py"
)
DEFAULT_STM_SRC = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/"
    "single_type_kv_cache_manager.py"
)
FALLBACK_PRISTINE = Path("/tmp/kv_cache_coordinator_pristine.py")

MARKER = "# [glm53-apc-per-group]"
MIA_MARKER = "# [glm53-hybrid-apc]"
PRIORITY_MARKER = "# [glm53-apc-drafter-priority]"
PRIOR_HELPER_MARKER = "# [glm53-dflash-prior-helper-v1]"
PRIOR_HELPER_V2_MARKER = "# [glm53-dflash-prior-helper-v2]"
PRIOR_POLICY_MARKER = "# [glm53-dflash-prior-policy-v1]"
PRIOR_MANAGER_MARKER = "# [glm53-dflash-prior-manager-v1]"
FREE_ORDER_MARKER = "# [glm53-apc-free-order-v1]"

HELPERS = (
    "_glm53_inner_kv_spec",
    "_glm53_is_draft_swa_spec",
    "_glm53_swa_retention_env",
    "_glm53_min_exempt_group_ids",
    "_glm53_retention_for_group",
    "_glm53_resolve_retention_by_group",
    "_glm53_validate_retention_intervals",
    "_glm53_format_retention_vector",
    "_glm53_dflash_prior_mamba_group_ids",
)
COMPOSED_HELPERS = HELPERS + (
    "_glm53_dflash_swa_replay_tokens",
    "_glm53_dflash_replay_safe_hit",
)

ALIGN = 3584  # scheduler_block_size on this kit
DRAFT_WINDOW = 2048
DRAFT_BLOCK = 64
KPOOL_TAIL_TOKENS = 4
SWA_ENV = "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA"


# --------------------------------------------------------------- stub specs --


class SlidingWindowSpec:  # exact type name is the discriminator
    def __init__(self, sliding_window=DRAFT_WINDOW, block_size=DRAFT_BLOCK):
        self.sliding_window = sliding_window
        self.block_size = block_size
        self.participates_in_prefix_caching = True


class KpoolTailSpec(SlidingWindowSpec):
    """Subclasses SlidingWindowSpec upstream; must NOT be treated as the drafter."""

    def __init__(self):
        super().__init__(
            sliding_window=KPOOL_TAIL_TOKENS, block_size=KPOOL_TAIL_TOKENS
        )
        self.participates_in_prefix_caching = False


class MambaSpec:
    def __init__(self, block_size=ALIGN):
        self.block_size = block_size
        self.mamba_cache_mode = "align"
        self.participates_in_prefix_caching = True


class MLAAttentionSpec:
    def __init__(self, block_size=ALIGN):
        self.block_size = block_size
        self.participates_in_prefix_caching = True


class UniformTypeKVCacheSpecs:
    def __init__(self, inner):
        self.kv_cache_specs = {"layer.0": inner}
        self.block_size = inner.block_size
        self.participates_in_prefix_caching = inner.participates_in_prefix_caching


class Group:
    """Stand-in for KVCacheGroupSpec (only .kv_cache_spec is read)."""

    def __init__(self, spec):
        self.kv_cache_spec = spec
        self.is_eagle_group = False


def uniform(inner):
    return UniformTypeKVCacheSpecs(inner)


def live_layout():
    """The seven groups of the live GLM-5.3 kit (DESIGN §1)."""
    return [
        Group(uniform(MLAAttentionSpec())),  # 0 MLA + indexer
        Group(uniform(KpoolTailSpec())),  # 1 kpool tail
        Group(MambaSpec()),  # 2
        Group(MambaSpec()),  # 3
        Group(MambaSpec()),  # 4
        Group(MambaSpec()),  # 5
        Group(uniform(SlidingWindowSpec())),  # 6 DFlash2 drafter
    ]


LIVE_EAGLE = {6}  # what patch_hybrid_prefix_hit.py narrows eagle_group_ids to


# ------------------------------------------------------- reference formulas --


def contiguous_blocks_for_hit(window, block, use_eagle):
    """Replica of SlidingWindowManager._contiguous_blocks_for_hit (S:886-895)."""
    blocks = -(-(window - 1) // block)
    return blocks + 1 if use_eagle else blocks


def swa_ids_per_segment(window, block, align, retention, use_eagle=True):
    """Replica of SlidingWindowManager.reachable_block_mask (S:998-1057),
    counted over one `align`-token segment far from any reachable boundary."""
    need = contiguous_blocks_for_hit(window, block, use_eagle)
    shift = 1 if use_eagle else 0
    segment_tokens = align if retention is None else (None if retention == 0 else retention)
    if segment_tokens is None:
        return 0
    per_segment = segment_tokens // block
    if need >= per_segment:
        return per_segment  # mask None -> every block cached
    # cache_blocks caches [num_cached_blocks, num_full_blocks); for an EAGLE
    # group num_full_blocks == aligned/block + 1, so each steady-state range is
    # [k*per_segment + shift, (k+1)*per_segment + shift). Count over that.
    total = 0
    for i in range(shift, shift + align // block):
        if i >= shift and (i - shift) % per_segment >= per_segment - need:
            total += 1
    return total


def mamba_ids_per_segment(block, align, retention):
    """Replica of MambaManager.reachable_block_mask (S:1487-1542)."""
    if retention is None:
        return align // block  # dense
    if retention == 0:
        return 0
    per_segment = retention // block
    if per_segment <= 1:
        return align // block
    return (align // block) / per_segment


# The ONE capacity formula of DESIGN §4. A conversation of S segments and t turn
# boundaries caches C = c*S + b*t ids, of which D = d*S + b_d*t are the drafter's
# (freed cached mid-prefill, so already parked on the protected LRU tail before
# the next conversation prefills). A's MLA/mamba survive B iff P >= 2C - D.
def max_segments(c, d, b, b_d, turns=3, pool=642):
    denom = 2 * c - d
    return int((pool - (2 * b - b_d) * turns) // denom)


# ----------------------------------------------------------------- helpers --


def load_helpers(patched_text: str, helpers=HELPERS) -> dict:
    """Exec only the injected helper functions in an isolated namespace."""
    tree = ast.parse(patched_text)
    wanted = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helpers
    }
    missing = set(helpers) - set(wanted)
    if missing:
        raise AssertionError(f"patched file is missing helpers: {sorted(missing)}")
    module = ast.Module(body=[wanted[name] for name in helpers], type_ignores=[])
    ast.fix_missing_locations(module)
    ns: dict = {"os": os}
    exec(compile(module, "<glm53-helpers>", "exec"), ns)  # noqa: S102
    return ns


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def apply_patch(
    patch: Path,
    target: Path,
    block_pool: Path,
    single_type_manager: Path,
) -> None:
    env = os.environ.copy()
    env["GLM53_KV_COORDINATOR_PY"] = str(target)
    env["GLM53_BLOCK_POOL_PY"] = str(block_pool)
    env["GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY"] = str(single_type_manager)
    subprocess.check_call([sys.executable, str(patch)], env=env)


# ------------------------------------------------------------------- cases --


def test_min_exemption(ns):
    """Min-exemption is derived from coordinator state, not from a class name."""
    fn = ns["_glm53_min_exempt_group_ids"]
    groups = live_layout()

    check(fn(groups, LIVE_EAGLE, True) == frozenset({6}), "live layout: gid 6 is min-exempt")
    check(fn(groups, LIVE_EAGLE, False) == frozenset(), "base coordinator -> empty")
    # Upstream all-groups EAGLE fallback distinguishes nothing -> nothing exempt.
    check(fn(groups, set(range(7)), True) == frozenset(), "all-groups fallback -> empty")
    # EAGLE flagged somewhere else -> the drafter is not the exempted group.
    check(fn(groups, {0}, True) == frozenset(), "eagle on MLA -> empty")
    check(fn(groups, {0, 6}, True) == frozenset(), "eagle superset -> empty")
    check(fn(groups, set(), True) == frozenset(), "no eagle group -> empty")
    # No exact SlidingWindowSpec at all (kpool tail subclass does not count).
    check(fn(groups[:6], {1}, True) == frozenset(), "kpool tail is not a drafter group")
    unitary = [Group(uniform(SlidingWindowSpec()))]
    check(
        fn(unitary, {0}, False) == frozenset(),
        "unitary SWA/EAGLE base coordinator -> empty",
    )
    print("  min-exemption derivation OK")


def test_routing(ns):
    """The per-group routing matrix."""
    fn = ns["_glm53_retention_for_group"]
    drafter = uniform(SlidingWindowSpec())
    tail = uniform(KpoolTailSpec())
    mamba = MambaSpec()
    mla = uniform(MLAAttentionSpec())

    # Explicit SWA value: only the min-exempt drafter diverges.
    for global_v in (None, 0, 3584, 14336):
        for swa_v in (0, 14336, 118272):
            check(
                fn(drafter, global_v, swa_v, True) == swa_v,
                f"min-exempt drafter should take swa={swa_v} (global={global_v})",
            )
            # Codex #4: the explicit path must honour min-exemption too.
            check(
                fn(drafter, global_v, swa_v, False) == global_v,
                "a SWA group that is NOT min-exempt must keep the global value "
                "even under an explicit override",
            )
            for other, name in ((tail, "kpool"), (mamba, "mamba"), (mla, "mla")):
                check(
                    fn(other, global_v, swa_v, False) == global_v,
                    f"{name} must keep global={global_v}",
                )
            # even if a non-drafter group were eagle-flagged
            check(fn(mamba, global_v, swa_v, True) == global_v, "mamba/eagle")
            check(fn(mla, global_v, swa_v, True) == global_v, "mla/eagle")

    # Unset is deliberately inert, including for a min-exempt drafter.
    for global_v in (None, 14336):
        check(fn(drafter, global_v, None, True) == global_v, "unset: drafter")
        check(fn(drafter, global_v, None, False) == global_v, "unset: non-exempt SWA")
        check(fn(mamba, global_v, None, False) == global_v, "unset: mamba")
        check(fn(tail, global_v, None, False) == global_v, "unset: kpool tail")
    print("  routing matrix OK")


def test_resolve(ns):
    """The resolved vector, and the fail-closed override guard (Codex #4/#6)."""
    fn = ns["_glm53_resolve_retention_by_group"]
    fmt = ns["_glm53_format_retention_vector"]
    groups = live_layout()

    # The deployment acceptance criterion: [None,...,None,0] on the head.
    vec = fn(groups, LIVE_EAGLE, True, None, 0)
    check(vec == (None,) * 6 + (0,), f"proposed config vector wrong: {vec}")
    check(
        fmt(vec) == "[None,None,None,None,None,None,0]",
        f"log rendering must be greppable, got {fmt(vec)}",
    )
    # Unset keeps the global policy; sparse retention requires explicit opt-in.
    check(
        fn(groups, LIVE_EAGLE, True, None, None) == (None,) * 7,
        "unset SWA interval must remain dense",
    )
    # Codex #6: the global knob still being set must not silently win/lose.
    check(
        fn(groups, LIVE_EAGLE, True, 14336, 0) == (14336,) * 6 + (0,),
        "a leftover global 14336 must show up in the vector, not be hidden",
    )
    check(
        fmt(fn(groups, LIVE_EAGLE, True, 14336, None))
        == "[14336,14336,14336,14336,14336,14336,14336]",
        "unset SWA interval must inherit global 14336",
    )

    # Fail closed: an explicit override with no EAGLE-exempt drafter group.
    for eagle in (set(range(7)), {0}, {0, 6}, set()):
        check(
            raises(fn, groups, eagle, True, None, 0),
            f"explicit SWA override must fail closed for eagle_group_ids={eagle}",
        )
        inherited = fn(groups, eagle, True, None, None)
        check(
            inherited == (None,) * 7,
            f"unset must inherit global for eagle={eagle}, got {inherited}",
        )

    # A model with no sliding-window group at all: override refused, unset inert.
    plain = [Group(uniform(MLAAttentionSpec())), Group(MambaSpec())]
    check(raises(fn, plain, {0, 1}, True, None, 3584), "no SWA group -> refuse")
    check(fn(plain, {0, 1}, True, 3584, None) == (3584, 3584), "no SWA group -> inherit")
    unitary = [Group(uniform(SlidingWindowSpec()))]
    check(
        raises(fn, unitary, {0}, False, None, 0),
        "unitary SWA/EAGLE base coordinator -> refuse explicit override",
    )
    check(
        fn(unitary, {0}, False, None, None) == (None,),
        "unitary SWA/EAGLE base coordinator -> unset is inert",
    )
    print("  resolved vector + fail-closed override OK")


def test_dflash_prior_groups(ns):
    """Only sparse Mamba groups in a hybrid DFlash layout get the extra state."""
    fn = ns["_glm53_dflash_prior_mamba_group_ids"]
    groups = live_layout()
    check(
        fn(groups, (0,) * 7, True) == frozenset({2, 3, 4, 5}),
        "live all-zero layout must retain one prior checkpoint in four Mamba groups",
    )
    check(
        fn(groups, (None, None, 0, 14336, 3584, None, 0), True)
        == frozenset({2, 3}),
        "boundary-only and positive-sparse Mamba groups qualify; dense groups do not",
    )
    check(fn(groups, (0,) * 7, False) == frozenset(), "base coordinator -> none")
    without_dflash = groups[:-1]
    check(
        fn(without_dflash, (0,) * len(without_dflash), True) == frozenset(),
        "hybrid layout without DFlash SWA -> none",
    )
    print("  DFlash prior-Mamba group selection OK")


def test_env(ns):
    """Codex #5: the raw env value is validated unconditionally."""
    fn = ns["_glm53_swa_retention_env"]
    saved = os.environ.pop(SWA_ENV, None)
    try:
        check(fn(ALIGN) is None, "unset -> None (inherit global)")
        for blank in ("", "  "):
            os.environ[SWA_ENV] = blank
            check(fn(ALIGN) is None, f"{blank!r} -> None (inherit global)")
        os.environ[SWA_ENV] = "0"
        check(fn(ALIGN) == 0, "'0' -> 0 (boundary-only)")
        os.environ[SWA_ENV] = " 14336 "
        check(fn(ALIGN) == 14336, "'14336' -> 14336")

        for bad, why in (
            ("nope", "junk"),
            ("3584.0", "float-ish junk"),
            ("-3584", "negative"),
            ("-1", "negative non-multiple"),
            ("3000", "non-multiple"),
            ("1", "non-multiple"),
            ("1003520", "over cap (280 * 3584 > 1_000_000)"),
        ):
            os.environ[SWA_ENV] = bad
            check(raises(fn, ALIGN), f"{why}: {bad!r} must be rejected")

        # The cap is the documented 1,000,000 and is enforced independently of
        # whether the model even has a sliding-window group.
        cap = fn.__defaults__[0]
        check(cap == 1_000_000, f"documented cap is 1,000,000, got {cap}")
        os.environ[SWA_ENV] = "999936"  # 279 * 3584, the largest legal value
        check(fn(ALIGN) == 999936, "largest legal value accepted")
    finally:
        os.environ.pop(SWA_ENV, None)
        if saved is not None:
            os.environ[SWA_ENV] = saved
    print("  env parsing + unconditional validation OK")


def test_validator(ns):
    fn = ns["_glm53_validate_retention_intervals"]
    fn((None, 0, 3584, 14336, None), ALIGN)  # all legal
    for bad in ((3000,), (-3584,), (None, 5000), (1,)):
        check(raises(fn, bad, ALIGN), f"validator accepted illegal intervals {bad}")
    print("  per-group validator OK")


def test_id_cost():
    """The arithmetic the whole design rests on (DESIGN §2 / §4)."""
    need = contiguous_blocks_for_hit(DRAFT_WINDOW, DRAFT_BLOCK, use_eagle=True)
    check(need == 33, f"need should be 33, got {need}")

    dense = swa_ids_per_segment(DRAFT_WINDOW, DRAFT_BLOCK, ALIGN, None)
    check(dense == 33, f"dense drafter should hash 33 of 56 per 3584, got {dense}")

    boundary_only = swa_ids_per_segment(DRAFT_WINDOW, DRAFT_BLOCK, ALIGN, 0)
    check(boundary_only == 0, f"retention 0 must hash no segment tails, got {boundary_only}")

    r14336 = swa_ids_per_segment(DRAFT_WINDOW, DRAFT_BLOCK, ALIGN, 14336)
    check(0 <= r14336 <= 33, "retention 14336 keeps at most one tail per 4 segments")
    # 33 ids per 14336 tokens == 8.25 per 3584 on average
    per = 14336 // DRAFT_BLOCK  # 224 drafter blocks per retention interval
    total = sum(1 for i in range(1, 1 + per) if (i - 1) % per >= per - 33)
    check(total == 33, f"retention 14336 should hash 33 per interval, got {total}")
    check(abs(total * ALIGN / 14336 - 8.25) < 1e-9, "== 8.25 ids per 3584 tokens")

    check(mamba_ids_per_segment(ALIGN, ALIGN, None) == 1, "mamba dense = 1/segment/group")
    check(mamba_ids_per_segment(ALIGN, ALIGN, 0) == 0, "mamba 0 = boundaries only")
    check(mamba_ids_per_segment(ALIGN, ALIGN, 14336) == 0.25, "mamba 14336 = 1 per 4")

    # Totals quoted in docs/DESIGN-apc-per-group-retention.md §2.5 / §4.
    dense_total = 1 + 4 * 1 + 33
    check(dense_total == 38, "dense segment cost")
    r14336_total = 1 + 4 * 0.25 + 33 / 4
    check(abs(r14336_total - 10.25) < 1e-9, "retention 14336 segment cost")
    proposed_total = 1 + 4 * 1 + 0
    check(proposed_total == 5, "proposed (dense mamba/MLA, boundary-only drafter)")

    # DESIGN §4, one formula, t = 3 turns, P = 642 usable ids. Rows are
    # (label, c, d, b, b_d, expected S). Boundary tails b exist only when the
    # group's retention is not None: dense caches every block already, and in
    # the proposed mode only the drafter has a boundary tail (33).
    rows = (
        ("dense", 38, 33, 0, 0, 14),
        ("R=7168", 19.5, 16.5, 37, 33, 23),
        ("R=14336", 10.25, 8.25, 37, 33, 42),
        ("R=28672", 5.625, 4.125, 37, 33, 72),
        ("proposed", 5, 0, 33, 33, 54),
    )
    for label, c, d, b, b_d, expected in rows:
        got = max_segments(c, d, b, b_d)
        check(got == expected, f"capacity model {label}: expected {expected}, got {got}")
    # The dense row is the one with a measured knee (14 OK / 17 fail).
    check(max_segments(38, 33, 0, 0) == 14, "dense knee must land on the measured 14/17")
    # 80K needs cdiv(80000, 3584) = 23 segments: R=7168 sits exactly on its knee.
    check(-(-80000 // ALIGN) == 23, "80K = 23 segments")
    check(54 * ALIGN == 193536, "proposed mode ~193.5K tokens per conversation")
    print("  id-cost + capacity arithmetic OK")


def test_call_sites(pristine: str, text: str):
    loop = "for i, manager in enumerate(self.single_type_managers):"
    check(pristine.count("retention_interval=self.retention_interval,") == 2,
          "expected exactly two global-interval cache_blocks call sites before the patch")
    check(text.count("retention_interval=self.retention_interval,") == 0,
          "no cache_blocks call may still pass the single global interval")
    check(text.count("retention_interval=self.retention_interval_by_group[i]") == 2,
          "both cache_blocks call sites must pass the per-group interval")
    check(text.count(loop) == pristine.count(loop) + 2,
          "both cache_blocks loops must be enumerated (and no other loop touched)")
    check("self.retention_interval_by_group = _glm53_resolve_retention_by_group(" in text,
          "per-group resolution missing")
    # Codex #6: the resolved vector must be greppable in `docker logs`.
    check("retention_by_group=%s" in text, "init must log retention_by_group=<vector>")
    check("_glm53_format_retention_vector(self.retention_interval_by_group)" in text,
          "the logged vector must be the resolved one")
    check(text.count(MARKER) >= 6, "MARK must annotate every edit")
    print("  call sites + boot log line OK")


def test_drafter_priority(block_pool_text: str, coordinator_text: str):
    """Exercise one-batch policy and real per-manager free call ordering."""
    tree = ast.parse(block_pool_text)
    block_pool_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BlockPool"
    )
    free_blocks = next(
        node
        for node in block_pool_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "free_blocks"
    )
    module = ast.Module(body=[free_blocks], type_ignores=[])
    ast.fix_missing_locations(module)

    class FakeBlock:
        def __init__(self, block_id, block_hash, ref_cnt=1, is_null=False):
            self.block_id = block_id
            self.block_hash = block_hash
            self.ref_cnt = ref_cnt
            self.is_null = is_null

    class FakeQueue:
        def __init__(self):
            self.items = ["existing"]

        def prepend_n(self, blocks):
            self.items[:0] = blocks

        def append_n(self, blocks):
            self.items.extend(blocks)

    ns = {
        "Iterable": Iterable,
        "KVCacheBlock": FakeBlock,
        "get_group_id": lambda key: key[0],
    }
    exec(compile(module, "<glm53-block-pool>", "exec"), ns)  # noqa: S102

    pool = types.SimpleNamespace(
        enable_caching=True,
        low_priority_cache_group_ids=frozenset({6}),
        cached_block_hashes_by_block={4: {(2, "mamba")}},
        free_block_queue=FakeQueue(),
    )
    unhashed = FakeBlock(1, None)
    drafter = FakeBlock(2, (6, "draft"))
    target = FakeBlock(3, (0, "mla"))
    mixed = FakeBlock(4, (6, "draft-shared"))
    still_used = FakeBlock(5, (6, "active"), ref_cnt=2)
    ns["free_blocks"](pool, [unhashed, drafter, target, mixed, still_used])
    check(
        pool.free_block_queue.items == [unhashed, drafter, "existing", target, mixed],
        "eviction order must be unhashed, drafter-only, existing, then target/mixed",
    )
    check(still_used.ref_cnt == 1, "referenced block must not enter the free queue")

    inherited = types.SimpleNamespace(
        enable_caching=True,
        low_priority_cache_group_ids=frozenset(),
        cached_block_hashes_by_block={},
        free_block_queue=FakeQueue(),
    )
    ordinary_draft = FakeBlock(6, (6, "ordinary"))
    ns["free_blocks"](inherited, [ordinary_draft])
    check(
        inherited.free_block_queue.items == ["existing", ordinary_draft],
        "without explicit SWA=0, drafter blocks must retain ordinary LRU priority",
    )

    # Coordinator.free invokes BlockPool.free_blocks once per manager. The
    # low-priority drafter must be called first: otherwise its later prepend
    # overtakes target unhashed blocks that were already placed at the front.
    multi_pool = types.SimpleNamespace(
        enable_caching=True,
        low_priority_cache_group_ids=frozenset({6}),
        cached_block_hashes_by_block={},
        free_block_queue=FakeQueue(),
        blocks=[],
    )
    multi_pool.get_num_free_blocks = lambda: len(multi_pool.free_block_queue.items)
    multi_pool.free_blocks = types.MethodType(ns["free_blocks"], multi_pool)
    target_unhashed = FakeBlock(10, None)
    draft_cached = FakeBlock(11, (6, "draft"))
    multi_pool.blocks = [target_unhashed, draft_cached]

    class ReleaseManager:
        def __init__(self, block=None):
            self.block = block
            self.req_to_blocks = {"r": [] if block is None else [block]}

        def free(self, request_id):
            blocks = self.req_to_blocks.pop(request_id)
            if blocks:
                multi_pool.free_blocks(blocks)

    managers = [ReleaseManager(target_unhashed)]
    managers.extend(ReleaseManager() for _ in range(5))
    managers.append(ReleaseManager(draft_cached))
    free_name, free_module = _method_from_source(
        coordinator_text, "KVCacheCoordinator", "free"
    )

    class FakeLogger:
        def info(self, *args):
            pass

    free_ns = {
        "KVCacheBlock": FakeBlock,
        "logger": FakeLogger(),
    }
    exec(compile(free_module, "<glm53-coordinator-free>", "exec"), free_ns)  # noqa: S102
    coordinator = types.SimpleNamespace(
        block_pool=multi_pool,
        single_type_managers=managers,
        _glm53_free_manager_order=(6, 0, 1, 2, 3, 4, 5),
    )
    free_ns[free_name](coordinator, "r")
    check(
        multi_pool.free_block_queue.items
        == [target_unhashed, draft_cached, "existing"],
        "multi-call order must keep target unhashed globally ahead of cached drafter",
    )
    check(
        all("r" not in manager.req_to_blocks for manager in managers),
        "coordinator free must release every manager exactly once",
    )
    check(
        "[glm53-apc-retention-diag]" not in coordinator_text
        and "[glm53-apc-hit-diag]" not in coordinator_text
        and "get_group_id," not in coordinator_text,
        "temporary qualification diagnostics must be absent from composed runtime",
    )
    print("  drafter priority + global per-manager free ordering OK")


def test_prior_boundary_cache_call(single_type_text: str):
    """Execute cache_blocks and inspect the boundaries passed to Mamba masking."""
    tree = ast.parse(single_type_text)
    manager_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SingleTypeKVCacheManager"
    )
    cache_blocks = next(
        node
        for node in manager_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "cache_blocks"
    )
    cache_blocks.decorator_list = []
    module = ast.Module(body=[cache_blocks], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"Request": object}
    exec(compile(module, "<glm53-single-type>", "exec"), ns)  # noqa: S102

    seen = []

    class Pool:
        def cache_full_blocks(self, **kwargs):
            seen.append(kwargs["block_mask"])

    def make_manager(enabled):
        manager = types.SimpleNamespace(
            num_cached_block={"r": 0},
            block_size=ALIGN,
            scheduler_block_size=ALIGN,
            kv_cache_spec=MambaSpec(),
            use_eagle=False,
            block_pool=Pool(),
            req_to_blocks={"r": [object()] * 6},
            kv_cache_group_id=2,
        )
        if enabled:
            manager._glm53_retain_previous_dflash_boundary = True

        def reachable_block_mask(**kwargs):
            return tuple(kwargs["reachable_boundaries"])

        manager.reachable_block_mask = reachable_block_mask
        return manager

    request = types.SimpleNamespace(
        request_id="r", num_prompt_tokens=21505, shared_prefix_boundary=14336
    )
    ns["cache_blocks"](make_manager(True), request, 21504, retention_interval=0)
    check(
        seen.pop() == (21504, 14336, 17920),
        "non-aligned prompt must retain final, junction, and one prior replay boundary",
    )
    ns["cache_blocks"](make_manager(False), request, 21504, retention_interval=0)
    check(
        seen.pop() == (21504, 14336),
        "groups without the scoped policy must retain upstream boundaries only",
    )
    print("  prior-boundary cache_blocks append/edit/junction inputs OK")


def _method_from_source(text: str, class_name: str, method_name: str):
    tree = ast.parse(text)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = copy.deepcopy(
        next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
    )
    method.decorator_list = []
    method.name = f"_{class_name}_{method_name}"
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    return method.name, module


def test_composed_runtime_paths(
    coordinator_text: str,
    block_pool_text: str,
    single_type_text: str,
):
    """Execute the injected init/cache/hit/free paths on the live seven groups.

    This is deliberately stubbed below GPU/model loading but uses the methods
    extracted from the fully composed runtime files. It catches unresolved
    injected names and marker-only migration mistakes before a production boot.
    """
    for marker in (
        "# [glm53-hybrid-apc]",
        "# [glm53-dflash-swa-replay-v1]",
        "# [glm53-dflash-swa-replay-v2]",
        PRIOR_HELPER_MARKER,
        PRIOR_HELPER_V2_MARKER,
        PRIOR_POLICY_MARKER,
        FREE_ORDER_MARKER,
    ):
        check(marker in coordinator_text, f"composed coordinator missing {marker}")
    check(
        PRIOR_MANAGER_MARKER in single_type_text,
        "composed single-type manager missing prior-boundary migration",
    )

    helper_ns = load_helpers(coordinator_text, COMPOSED_HELPERS)

    class FakeBlockPool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.low_priority_cache_group_ids = frozenset()

    class FakeManager:
        supports_fine_grained_hash_lookup = False

        def __init__(self, spec, group_id):
            self.block_size = spec.block_size
            self.group_id = group_id
            self.use_eagle = False
            self.cache_calls = []

        def cache_blocks(self, request, num_tokens, retention_interval=None):
            self.cache_calls.append((request, num_tokens, retention_interval))

    def get_manager_for_kv_cache_spec(kv_cache_spec, kv_cache_group_id, **kwargs):
        return FakeManager(kv_cache_spec, kv_cache_group_id)

    class HybridKVCacheCoordinator:
        """Temporary discriminator while the exact base initializer is compiled."""

    class FakeLogger:
        def __init__(self):
            self.records = []

        def info(self, *args):
            self.records.append(args)

        def warning_once(self, *args):
            self.records.append(args)

    init_name, init_module = _method_from_source(
        coordinator_text, "KVCacheCoordinator", "__init__"
    )
    logger = FakeLogger()
    ns = {
        **helper_ns,
        "KVCacheConfig": object,
        "KVCacheMetricsCollector": object,
        "BlockPool": FakeBlockPool,
        "get_manager_for_kv_cache_spec": get_manager_for_kv_cache_spec,
        "_validate_prefix_cache_retention_interval": lambda *args: None,
        "envs": types.SimpleNamespace(VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0),
        "HybridKVCacheCoordinator": HybridKVCacheCoordinator,
        "logger": logger,
    }
    exec(compile(init_module, "<exact-image-coordinator-init>", "exec"), ns)  # noqa: S102

    base_init = ns[init_name]

    class KVCacheCoordinator:
        def __init__(self, *args, **kwargs):
            base_init(self, *args, **kwargs)

        def verify_and_split_kv_cache_groups(self):
            pass

    tree = ast.parse(coordinator_text)
    exact_hybrid = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "HybridKVCacheCoordinator"
        )
    )
    exact_hybrid.bases = [ast.Name(id="KVCacheCoordinator", ctx=ast.Load())]
    exact_hybrid.keywords = []
    exact_hybrid.body = [
        node
        for node in exact_hybrid.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"__init__", "_cache_hit_alignment_tokens"}
    ]
    hybrid_module = ast.Module(body=[exact_hybrid], type_ignores=[])
    ast.fix_missing_locations(hybrid_module)
    hybrid_ns = {
        "KVCacheCoordinator": KVCacheCoordinator,
        "KVCacheConfig": object,
        "KVCacheMetricsCollector": object,
        "FullAttentionSpec": MLAAttentionSpec,
        "MambaSpec": MambaSpec,
        "logger": logger,
        **helper_ns,
    }
    exec(compile(hybrid_module, "<exact-image-hybrid-init>", "exec"), hybrid_ns)  # noqa: S102
    HybridKVCacheCoordinator = hybrid_ns["HybridKVCacheCoordinator"]
    base_init.__globals__["HybridKVCacheCoordinator"] = HybridKVCacheCoordinator

    config = types.SimpleNamespace(
        kv_cache_groups=live_layout(),
        num_blocks=390,
        needs_kv_cache_zeroing=False,
    )
    saved = os.environ.get(SWA_ENV)
    os.environ[SWA_ENV] = "0"
    try:
        coordinator = HybridKVCacheCoordinator(
            config,
            max_model_len=262144,
            max_in_flight_tokens=2048,
            use_eagle=True,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            scheduler_block_size=ALIGN,
            hash_block_size=64,
        )
    finally:
        if saved is None:
            os.environ.pop(SWA_ENV, None)
        else:
            os.environ[SWA_ENV] = saved

    check(coordinator.eagle_group_ids == {6}, "runtime init must isolate drafter gid 6")
    check(
        coordinator.retention_interval_by_group == (0,) * 7,
        "runtime init must resolve the experimental all-zero vector",
    )
    check(
        coordinator.dflash_replay_prior_group_ids == frozenset({2, 3, 4, 5}),
        "runtime init must select all four sparse Mamba groups",
    )
    check(
        coordinator.dflash_swa_replay_tokens == DRAFT_WINDOW,
        "runtime init must derive DFlash's 2048-token SWA replay window",
    )
    check(
        coordinator._glm53_free_manager_order == (6, 0, 1, 2, 3, 4, 5),
        "runtime init must release low-priority drafter manager before target managers",
    )
    check(
        {
            i
            for i, manager in enumerate(coordinator.single_type_managers)
            if getattr(manager, "_glm53_retain_previous_dflash_boundary", False)
        }
        == {2, 3, 4, 5},
        "runtime init must arm only the four Mamba managers",
    )

    cache_name, cache_module = _method_from_source(
        coordinator_text, "HybridKVCacheCoordinator", "cache_blocks"
    )
    cache_ns = {"Request": object}
    exec(compile(cache_module, "<exact-image-coordinator-cache>", "exec"), cache_ns)  # noqa: S102
    coordinator.enable_partial_hash_hits = False
    coordinator.single_type_managers[6].use_eagle = True
    request = types.SimpleNamespace(request_id="r")
    cache_ns[cache_name](coordinator, request, 21569)
    check(
        [m.cache_calls[-1][2] for m in coordinator.single_type_managers] == [0] * 7,
        "cache path must forward the resolved interval to every group",
    )
    check(
        coordinator.single_type_managers[6].cache_calls[-1][1] == 21568,
        "drafter cache path must preserve the one-block EAGLE lookahead",
    )

    class FullAttentionSpec:
        pass

    class HitMambaSpec:
        pass

    HitMambaSpec.__name__ = "MambaSpec"

    class HitSlidingWindowSpec:
        sliding_window = DRAFT_WINDOW

    HitSlidingWindowSpec.__name__ = "SlidingWindowSpec"

    class HitManager:
        supports_fine_grained_hash_lookup = False

        @classmethod
        def find_longest_cache_hit(cls, max_length, kv_cache_group_ids, **kwargs):
            length = max_length // ALIGN * ALIGN
            blocks = [[object()] * (length // ALIGN) for _ in kv_cache_group_ids]
            return tuple(blocks), length

    class HitBlock:
        def __init__(self, is_null=False):
            self.is_null = is_null

    class DraftInvalidAtCurrentManager(HitManager):
        @classmethod
        def find_longest_cache_hit(cls, max_length, kv_cache_group_ids, **kwargs):
            # Length alone claims the latest boundary, but the visible tail is
            # incomplete. The coordinator must clamp and discard these blocks.
            if max_length < 21504:
                return tuple([] for _ in kv_cache_group_ids), 0
            blocks = [HitBlock() for _ in range(21504 // DRAFT_BLOCK)]
            blocks[-1] = HitBlock(is_null=True)
            return tuple(
                list(blocks) for _ in kv_cache_group_ids
            ), 21504

    class DraftValidAtCurrentManager(HitManager):
        @classmethod
        def find_longest_cache_hit(cls, max_length, kv_cache_group_ids, **kwargs):
            length = max_length // ALIGN * ALIGN
            blocks = [HitBlock() for _ in range(length // DRAFT_BLOCK)]
            return tuple(
                list(blocks) for _ in kv_cache_group_ids
            ), length

    SpecGroup = namedtuple("SpecGroup", "spec group_ids manager_cls use_eagle")
    coordinator.attention_groups = [
        SpecGroup(FullAttentionSpec(), [0], HitManager, False),
        SpecGroup(HitMambaSpec(), [2, 3, 4, 5], HitManager, False),
        SpecGroup(HitSlidingWindowSpec(), [6], DraftInvalidAtCurrentManager, True),
    ]
    coordinator.dflash_swa_replay_tokens = DRAFT_WINDOW
    coordinator.hash_block_size = 64
    coordinator.dcp_world_size = 1
    for i in (0, 2, 3, 4, 5):
        coordinator.single_type_managers[i].block_size = ALIGN
    coordinator.single_type_managers[6].block_size = 64

    hit_name, hit_module = _method_from_source(
        coordinator_text, "HybridKVCacheCoordinator", "find_longest_cache_hit"
    )
    hit_ns = {
        "BlockHash": object,
        "KVCacheBlock": object,
        "FullAttentionSpec": FullAttentionSpec,
        "MambaSpec": HitMambaSpec,
        "cdiv": lambda a, b: (a + b - 1) // b,
        "logger": logger,
        **helper_ns,
    }
    exec(compile(hit_module, "<exact-image-coordinator-hit>", "exec"), hit_ns)  # noqa: S102
    blocks, hit, uncached = hit_ns[hit_name](coordinator, [object()] * 400, 21568)
    check(hit == 17920, f"hit path must clamp 21504 to prior state, got {hit}")
    check(uncached == 3584, f"hit path uncached delta must be 3584, got {uncached}")
    check(len(blocks) == 7 and len(blocks[0]) == 5, "hit path must return 7 groups")
    check(
        blocks[6] == [],
        "backed-up target hit must discard incomplete later-boundary DFlash blocks, "
        f"got {len(blocks[6])}",
    )

    coordinator.attention_groups[-1] = SpecGroup(
        HitSlidingWindowSpec(), [6], DraftValidAtCurrentManager, False
    )
    blocks, hit, uncached = hit_ns[hit_name](coordinator, [object()] * 400, 21568)
    check(
        hit == 17920 and uncached == 3584,
        "a complete draft tail without EAGLE-pop semantics must still clamp",
    )

    coordinator.attention_groups[-1] = SpecGroup(
        HitSlidingWindowSpec(), [6], DraftValidAtCurrentManager, True
    )
    blocks, hit, uncached = hit_ns[hit_name](
        coordinator, [object()] * 400, 21568
    )
    check(
        hit == 21504,
        f"complete reconciled DFlash boundary must avoid 3584-token clamp, got {hit}",
    )
    check(uncached == 0, f"complete current-boundary hit must not report gap, got {uncached}")
    check(
        len(blocks[6]) == 21504 // DRAFT_BLOCK
        and all(not block.is_null for block in blocks[6][-32:]),
        "reused DFlash hit must carry the complete 2048-token visible tail",
    )

    test_prior_boundary_cache_call(single_type_text)
    test_drafter_priority(block_pool_text, coordinator_text)
    print("  exact composed init/cache/hit/free execution OK")


def test_composition(
    pristine_src: Path,
    pristine_bp_src: Path,
    pristine_stm_src: Path,
    tmp: Path,
):
    """Codex #8: both overlays, both application orders, on a pristine source."""
    if MIA_PATCH is None:
        raise SystemExit("missing patch_hybrid_prefix_hit.py (overlay composition)")
    results = {}
    for label, order in (
        ("mia-then-ours", (MIA_PATCH, PATCH)),
        ("ours-then-mia", (PATCH, MIA_PATCH)),
    ):
        dst = tmp / f"compose_{label}.py"
        bp_dst = tmp / f"block_pool_{label}.py"
        stm_dst = tmp / f"single_type_{label}.py"
        shutil.copyfile(pristine_src, dst)
        shutil.copyfile(pristine_bp_src, bp_dst)
        shutil.copyfile(pristine_stm_src, stm_dst)
        for patch in order:
            apply_patch(patch, dst, bp_dst, stm_dst)
        text = dst.read_text()
        bp_text = bp_dst.read_text()
        stm_text = stm_dst.read_text()
        py_compile.compile(str(dst), cfile=str(tmp / f"{label}.pyc"), doraise=True)
        py_compile.compile(
            str(bp_dst), cfile=str(tmp / f"block-pool-{label}.pyc"), doraise=True
        )
        py_compile.compile(
            str(stm_dst), cfile=str(tmp / f"single-type-{label}.pyc"), doraise=True
        )
        check(MARKER in text and MIA_MARKER in text, f"{label}: a MARK is missing")
        check(PRIORITY_MARKER in bp_text, f"{label}: block-pool priority patch missing")
        check(PRIOR_HELPER_MARKER in text, f"{label}: coordinator prior helper missing")
        check(PRIOR_POLICY_MARKER in text, f"{label}: coordinator prior policy missing")
        check(PRIOR_MANAGER_MARKER in stm_text, f"{label}: manager prior policy missing")
        check(text.count("def _glm53_inner_kv_spec(") == 1,
              f"{label}: shared helper duplicated")
        check(text.count("def _glm53_is_draft_swa_spec(") == 1,
              f"{label}: shared discriminator duplicated")
        check(text.count("import os  # [glm53-apc-per-group]") == 1,
              f"{label}: os import missing or duplicated")
        # Re-applying either patch in either order must be a no-op.
        for patch in order + tuple(reversed(order)):
            apply_patch(patch, dst, bp_dst, stm_dst)
        check(dst.read_text() == text, f"{label}: composition is not idempotent")
        check(bp_dst.read_text() == bp_text, f"{label}: block-pool patch is not idempotent")
        check(stm_dst.read_text() == stm_text, f"{label}: manager patch is not idempotent")
        # Both overlays' behaviour survives composition.
        ns = load_helpers(text)
        check(
            ns["_glm53_resolve_retention_by_group"](
                live_layout(), LIVE_EAGLE, True, None, 0
            )
            == (None,) * 6 + (0,),
            f"{label}: resolved vector wrong after composition",
        )
        check("if _glm53_is_draft_swa_spec(spec):  # [glm53-hybrid-apc]" in text,
              f"{label}: Mia's hybrid-min skip is missing")
        check("swa_ids or set(" in text,
              f"{label}: Mia's eagle_group_ids narrowing is missing")
        test_prior_boundary_cache_call(stm_text)
        test_composed_runtime_paths(text, bp_text, stm_text)
        results[label] = text
    if results["mia-then-ours"] != results["ours-then-mia"]:
        print(
            "".join(
                difflib.unified_diff(
                    results["mia-then-ours"].splitlines(keepends=True),
                    results["ours-then-mia"].splitlines(keepends=True),
                    fromfile="mia-then-ours",
                    tofile="ours-then-mia",
                )
            )
        )
        raise AssertionError(
            "the two overlays must compose to the same file in either order"
        )
    print("  overlay composition (both orders, idempotent) OK")


def resolve_pristine(src: Path) -> Path:
    env = os.environ.get("GLM53_KV_COORDINATOR_PY_PRISTINE", "").strip()
    if env:
        p = Path(env)
        if not p.is_file():
            raise SystemExit(f"GLM53_KV_COORDINATOR_PY_PRISTINE points at nothing: {p}")
        return p
    if MIA_MARKER not in src.read_text() and MARKER not in src.read_text():
        return src
    if FALLBACK_PRISTINE.is_file():
        return FALLBACK_PRISTINE
    raise SystemExit(
        f"{src} already carries an overlay; the composition test needs a pristine "
        "copy of the same file. Set GLM53_KV_COORDINATOR_PY_PRISTINE (or drop one "
        f"at {FALLBACK_PRISTINE})."
    )


def main() -> int:
    if PATCH is None:
        raise SystemExit("missing patch_apc_per_group_retention.py")
    src = Path(os.environ.get("GLM53_KV_COORDINATOR_PY_SRC", DEFAULT_SRC))
    bp_src = Path(os.environ.get("GLM53_BLOCK_POOL_PY_SRC", DEFAULT_BP_SRC))
    stm_src = Path(
        os.environ.get("GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY_SRC", DEFAULT_STM_SRC)
    )
    if not src.is_file():
        raise SystemExit(
            f"missing kv_cache_coordinator.py at {src}; "
            "set GLM53_KV_COORDINATOR_PY_SRC to a copy of the fork's file"
        )
    if not bp_src.is_file():
        raise SystemExit(
            f"missing block_pool.py at {bp_src}; "
            "set GLM53_BLOCK_POOL_PY_SRC to a copy of the fork's file"
        )
    if not stm_src.is_file():
        raise SystemExit(
            f"missing single_type_kv_cache_manager.py at {stm_src}; "
            "set GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY_SRC to the fork source copy"
        )
    if PRIORITY_MARKER in bp_src.read_text():
        raise SystemExit(f"{bp_src} already carries {PRIORITY_MARKER}; use a pristine copy")
    pristine_src = resolve_pristine(src)
    if MIA_MARKER in pristine_src.read_text() or MARKER in pristine_src.read_text():
        raise SystemExit(f"{pristine_src} is not pristine (it carries an overlay MARK)")
    print(
        f"  source: {src}\n  pristine: {pristine_src}\n"
        f"  block pool: {bp_src}\n  single-type manager: {stm_src}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        dst = tmp / "kv_cache_coordinator.py"
        bp_dst = tmp / "block_pool.py"
        stm_dst = tmp / "single_type_kv_cache_manager.py"
        shutil.copyfile(src, dst)
        shutil.copyfile(bp_src, bp_dst)
        shutil.copyfile(stm_src, stm_dst)

        pristine = dst.read_text()
        apply_patch(PATCH, dst, bp_dst, stm_dst)
        text = dst.read_text()
        bp_text = bp_dst.read_text()
        stm_text = stm_dst.read_text()
        check(MARKER in text, "MARK missing after apply")
        check(PRIORITY_MARKER in bp_text, "block-pool priority MARK missing after apply")
        check(PRIOR_HELPER_MARKER in text, "coordinator prior-helper MARK missing")
        check(PRIOR_POLICY_MARKER in text, "coordinator prior-policy MARK missing")
        check(PRIOR_MANAGER_MARKER in stm_text, "manager prior-policy MARK missing")

        py_compile.compile(str(dst), cfile=str(tmp / "out.pyc"), doraise=True)
        py_compile.compile(str(bp_dst), cfile=str(tmp / "block-pool.pyc"), doraise=True)
        py_compile.compile(str(stm_dst), cfile=str(tmp / "single-type.pyc"), doraise=True)
        print("  applies and compiles OK")

        apply_patch(PATCH, dst, bp_dst, stm_dst)
        check(dst.read_text() == text, "patch is not idempotent")
        check(bp_dst.read_text() == bp_text, "block-pool patch is not idempotent")
        check(stm_dst.read_text() == stm_text, "single-type patch is not idempotent")
        print("  idempotent OK")

        test_call_sites(pristine, text)
        ns = load_helpers(text)
        test_min_exemption(ns)
        test_routing(ns)
        test_resolve(ns)
        test_dflash_prior_groups(ns)
        test_env(ns)
        test_validator(ns)
        test_drafter_priority(bp_text, text)
        test_prior_boundary_cache_call(stm_text)
        test_composition(pristine_src, bp_src, stm_src, tmp)

    test_id_cost()
    print("drafter-retention patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
