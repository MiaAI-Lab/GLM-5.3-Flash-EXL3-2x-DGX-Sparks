#!/usr/bin/env python3
"""Host test for patch_apc_per_group_retention.py (no GPU, no vLLM import).

Applies the overlay to a *copy* of kv_cache_coordinator.py and then exercises the
injected helpers directly, so the test asserts behaviour rather than the presence
of strings. Fails closed: any problem exits non-zero.

    GLM53_KV_COORDINATOR_PY_SRC=/path/to/fork/kv_cache_coordinator.py \\
        python3 test_apc_per_group_retention.py

`..._SRC` may already carry overlay/patch_hybrid_prefix_hit.py. The overlay
*composition* case additionally needs a pristine (unpatched) copy of the same
file; it is taken from `GLM53_KV_COORDINATOR_PY_PRISTINE`, else from `..._SRC`
itself when that is unpatched, else from /tmp/kv_cache_coordinator_pristine.py.
Default source is the in-container path.
"""

from __future__ import annotations

import ast
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
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
FALLBACK_PRISTINE = Path("/tmp/kv_cache_coordinator_pristine.py")

MARKER = "# [glm53-apc-per-group]"
MIA_MARKER = "# [glm53-hybrid-apc]"

HELPERS = (
    "_glm53_inner_kv_spec",
    "_glm53_is_draft_swa_spec",
    "_glm53_swa_retention_env",
    "_glm53_min_exempt_group_ids",
    "_glm53_retention_for_group",
    "_glm53_resolve_retention_by_group",
    "_glm53_validate_retention_intervals",
    "_glm53_format_retention_vector",
)

ALIGN = 3584  # scheduler_block_size on this kit
DRAFT_WINDOW = 2048
DRAFT_BLOCK = 64
SWA_ENV = "VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA"


# --------------------------------------------------------------- stub specs --


class SlidingWindowSpec:  # exact type name is the discriminator
    def __init__(self, sliding_window=DRAFT_WINDOW, block_size=DRAFT_BLOCK):
        self.sliding_window = sliding_window
        self.block_size = block_size


class KpoolTailSpec(SlidingWindowSpec):
    """Subclasses SlidingWindowSpec upstream; must NOT be treated as the drafter."""


class MambaSpec:
    def __init__(self, block_size=ALIGN):
        self.block_size = block_size


class MLAAttentionSpec:
    def __init__(self, block_size=ALIGN):
        self.block_size = block_size


class UniformTypeKVCacheSpecs:
    def __init__(self, inner):
        self.kv_cache_specs = {"layer.0": inner}


class Group:
    """Stand-in for KVCacheGroupSpec (only .kv_cache_spec is read)."""

    def __init__(self, spec):
        self.kv_cache_spec = spec


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


def load_helpers(patched_text: str) -> dict:
    """Exec only the injected helper functions in an isolated namespace."""
    tree = ast.parse(patched_text)
    wanted = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS
    }
    missing = set(HELPERS) - set(wanted)
    if missing:
        raise AssertionError(f"patched file is missing helpers: {sorted(missing)}")
    module = ast.Module(body=[wanted[name] for name in HELPERS], type_ignores=[])
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


def apply_patch(patch: Path, target: Path) -> None:
    env = os.environ.copy()
    env["GLM53_KV_COORDINATOR_PY"] = str(target)
    subprocess.check_call([sys.executable, str(patch)], env=env)


# ------------------------------------------------------------------- cases --


def test_min_exemption(ns):
    """Min-exemption is derived from coordinator state, not from a class name."""
    fn = ns["_glm53_min_exempt_group_ids"]
    groups = live_layout()

    check(fn(groups, LIVE_EAGLE) == frozenset({6}), "live layout: gid 6 is min-exempt")
    # Upstream all-groups EAGLE fallback distinguishes nothing -> nothing exempt.
    check(fn(groups, set(range(7))) == frozenset(), "all-groups fallback -> empty")
    # EAGLE flagged somewhere else -> the drafter is not the exempted group.
    check(fn(groups, {0}) == frozenset(), "eagle on MLA -> empty")
    check(fn(groups, {0, 6}) == frozenset(), "eagle superset -> empty")
    check(fn(groups, set()) == frozenset(), "no eagle group -> empty")
    # No exact SlidingWindowSpec at all (kpool tail subclass does not count).
    check(fn(groups[:6], {1}) == frozenset(), "kpool tail is not a drafter group")
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
                fn(drafter, global_v, swa_v, ALIGN, True) == swa_v,
                f"min-exempt drafter should take swa={swa_v} (global={global_v})",
            )
            # Codex #4: the explicit path must honour min-exemption too.
            check(
                fn(drafter, global_v, swa_v, ALIGN, False) == global_v,
                "a SWA group that is NOT min-exempt must keep the global value "
                "even under an explicit override",
            )
            for other, name in ((tail, "kpool"), (mamba, "mamba"), (mla, "mla")):
                check(
                    fn(other, global_v, swa_v, ALIGN, False) == global_v,
                    f"{name} must keep global={global_v}",
                )
            # even if a non-drafter group were eagle-flagged
            check(fn(mamba, global_v, swa_v, ALIGN, True) == global_v, "mamba/eagle")
            check(fn(mla, global_v, swa_v, ALIGN, True) == global_v, "mla/eagle")

    # Auto rule (swa=None): hit-inert min-exempt drafter -> 0; everything else global.
    for global_v in (None, 14336):
        check(fn(drafter, global_v, None, ALIGN, True) == 0, "auto: drafter -> 0")
        check(
            fn(drafter, global_v, None, ALIGN, False) == global_v,
            "auto: non-exempt SWA keeps global (it is inside the hit min)",
        )
        # window >= alignment: a real SWA model, rule is inert
        wide = uniform(SlidingWindowSpec(sliding_window=8192, block_size=64))
        check(fn(wide, global_v, None, ALIGN, True) == global_v, "auto: wide window")
        # window exactly at the alignment: not hit-inert, rule is inert
        edge = uniform(SlidingWindowSpec(sliding_window=ALIGN, block_size=64))
        check(fn(edge, global_v, None, ALIGN, True) == global_v, "auto: window == align")
        # bare (non-uniform-wrapped) drafter spec is recognised too
        bare = SlidingWindowSpec()
        check(fn(bare, global_v, None, ALIGN, True) == 0, "auto: bare drafter spec")
        # missing alignment -> conservative fall-through
        check(fn(drafter, global_v, None, None, True) == global_v, "auto: no alignment")
        check(fn(mamba, global_v, None, ALIGN, False) == global_v, "auto: mamba")
        check(fn(tail, global_v, None, ALIGN, False) == global_v, "auto: kpool tail")
    print("  routing matrix OK")


def test_resolve(ns):
    """The resolved vector, and the fail-closed override guard (Codex #4/#6)."""
    fn = ns["_glm53_resolve_retention_by_group"]
    fmt = ns["_glm53_format_retention_vector"]
    groups = live_layout()

    # The deployment acceptance criterion: [None,...,None,0] on the head.
    vec = fn(groups, LIVE_EAGLE, None, 0, ALIGN)
    check(vec == (None,) * 6 + (0,), f"proposed config vector wrong: {vec}")
    check(
        fmt(vec) == "[None,None,None,None,None,None,0]",
        f"log rendering must be greppable, got {fmt(vec)}",
    )
    # Auto mode reaches the same vector without the env var.
    check(fn(groups, LIVE_EAGLE, None, None, ALIGN) == vec, "auto == explicit 0 here")
    # Codex #6: the global knob still being set must not silently win/lose.
    check(
        fn(groups, LIVE_EAGLE, 14336, 0, ALIGN) == (14336,) * 6 + (0,),
        "a leftover global 14336 must show up in the vector, not be hidden",
    )
    check(
        fmt(fn(groups, LIVE_EAGLE, 14336, None, ALIGN))
        == "[14336,14336,14336,14336,14336,14336,0]",
        "auto rule under a global 14336",
    )

    # Fail closed: an explicit override with no EAGLE-exempt drafter group.
    for eagle in (set(range(7)), {0}, {0, 6}, set()):
        check(
            raises(fn, groups, eagle, None, 0, ALIGN),
            f"explicit SWA override must fail closed for eagle_group_ids={eagle}",
        )
        # ... but the automatic rule stays safe/inert there.
        auto = fn(groups, eagle, None, None, ALIGN)
        check(auto == (None,) * 7, f"auto must be inert for eagle={eagle}, got {auto}")

    # A model with no sliding-window group at all: override refused, auto inert.
    plain = [Group(uniform(MLAAttentionSpec())), Group(MambaSpec())]
    check(raises(fn, plain, {0, 1}, None, 3584, ALIGN), "no SWA group -> refuse")
    check(fn(plain, {0, 1}, 3584, None, ALIGN) == (3584, 3584), "no SWA group -> auto")
    print("  resolved vector + fail-closed override OK")


def test_env(ns):
    """Codex #5: the raw env value is validated unconditionally."""
    fn = ns["_glm53_swa_retention_env"]
    saved = os.environ.pop(SWA_ENV, None)
    try:
        check(fn(ALIGN) is None, "unset -> None (auto)")
        for blank in ("", "  "):
            os.environ[SWA_ENV] = blank
            check(fn(ALIGN) is None, f"{blank!r} -> None (auto)")
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


def test_composition(pristine_src: Path, tmp: Path):
    """Codex #8: both overlays, both application orders, on a pristine source."""
    if MIA_PATCH is None:
        raise SystemExit("missing patch_hybrid_prefix_hit.py (overlay composition)")
    results = {}
    for label, order in (
        ("mia-then-ours", (MIA_PATCH, PATCH)),
        ("ours-then-mia", (PATCH, MIA_PATCH)),
    ):
        dst = tmp / f"compose_{label}.py"
        shutil.copyfile(pristine_src, dst)
        for patch in order:
            apply_patch(patch, dst)
        text = dst.read_text()
        py_compile.compile(str(dst), cfile=str(tmp / f"{label}.pyc"), doraise=True)
        check(MARKER in text and MIA_MARKER in text, f"{label}: a MARK is missing")
        check(text.count("def _glm53_inner_kv_spec(") == 1,
              f"{label}: shared helper duplicated")
        check(text.count("def _glm53_is_draft_swa_spec(") == 1,
              f"{label}: shared discriminator duplicated")
        check(text.count("import os  # [glm53-apc-per-group]") == 1,
              f"{label}: os import missing or duplicated")
        # Re-applying either patch in either order must be a no-op.
        for patch in order + tuple(reversed(order)):
            apply_patch(patch, dst)
        check(dst.read_text() == text, f"{label}: composition is not idempotent")
        # Both overlays' behaviour survives composition.
        ns = load_helpers(text)
        check(
            ns["_glm53_resolve_retention_by_group"](
                live_layout(), LIVE_EAGLE, None, 0, ALIGN
            )
            == (None,) * 6 + (0,),
            f"{label}: resolved vector wrong after composition",
        )
        check("if _glm53_is_draft_swa_spec(spec):  # [glm53-hybrid-apc]" in text,
              f"{label}: Mia's hybrid-min skip is missing")
        check("swa_ids or set(" in text,
              f"{label}: Mia's eagle_group_ids narrowing is missing")
        results[label] = text
    check(results["mia-then-ours"] == results["ours-then-mia"],
          "the two overlays must compose to the same file in either order")
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
    if not src.is_file():
        raise SystemExit(
            f"missing kv_cache_coordinator.py at {src}; "
            "set GLM53_KV_COORDINATOR_PY_SRC to a copy of the fork's file"
        )
    pristine_src = resolve_pristine(src)
    if MIA_MARKER in pristine_src.read_text() or MARKER in pristine_src.read_text():
        raise SystemExit(f"{pristine_src} is not pristine (it carries an overlay MARK)")
    print(f"  source: {src}\n  pristine: {pristine_src}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        dst = tmp / "kv_cache_coordinator.py"
        shutil.copyfile(src, dst)

        pristine = dst.read_text()
        apply_patch(PATCH, dst)
        text = dst.read_text()
        check(MARKER in text, "MARK missing after apply")

        py_compile.compile(str(dst), cfile=str(tmp / "out.pyc"), doraise=True)
        print("  applies and compiles OK")

        apply_patch(PATCH, dst)
        check(dst.read_text() == text, "patch is not idempotent")
        print("  idempotent OK")

        test_call_sites(pristine, text)
        ns = load_helpers(text)
        test_min_exemption(ns)
        test_routing(ns)
        test_resolve(ns)
        test_env(ns)
        test_validator(ns)
        test_composition(pristine_src, tmp)

    test_id_cost()
    print("drafter-retention patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
