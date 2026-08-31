#!/usr/bin/env python3
"""Host test for overlay/patch_apc_fine_grained_hits.py.

Runs anywhere Python 3.10+ is available -- no vLLM import required.

Part A (patch mechanics, mirrors tests/test_hybrid_prefix_hit.py):
  apply to a COPY of kv_cache_coordinator.py, assert the MARK and anchors
  landed, assert idempotence, assert fail-closed on a drifted anchor, assert
  fail-closed on a pre-existing-but-incomplete marker, assert the apply is
  transactional (no partial writes, no temp litter), assert composability
  with overlay/patch_hybrid_prefix_hit.py in both orders and under re-apply,
  and assert the "upstream already fixed it" path (A6): a coordinator whose
  veto is already scoped to prefix_cacheable groups (vLLM main e126687a) is a
  clean no-op exit 0, while a merely drifted or deleted veto still fails closed.

Part B (gate semantics):
  exec the injected helper block in a bare namespace, then drive it with fakes
  that mirror the live KV cache layout and several hostile variants.  This is
  the part that actually encodes the correctness argument.

  Note the fail-closed policy under test, as a 2x2 (DESIGN 4.3) -- rows are
  "does a PARTICIPATING manager already block fine lookups", columns are "is a
  NON-PARTICIPATING scratch group unsafe or unverifiable":

      blocker vs scratch |   ok    |  bad
      ------------------+---------+---------
      no                | ENABLE  | RAISE
      yes               | DISABLE | DISABLE

    * a PARTICIPATING manager that cannot do fine lookups -> DISABLE
      (block-aligned hits are the correct, safe fallback; this is upstream's
      own condition) -- and that stays DISABLE even when a scratch group is
      also bad, because no fine-grained hit is taken on that layout, so the
      scratch invariant is unreachable and a boot failure would be gratuitous;
    * a NON-PARTICIPATING scratch group whose alignment requirement is violated
      or unverifiable, on a layout that would otherwise ENABLE -> RAISE
      Glm53FineGrainedAPCError at coordinator init, tagged
      "[glm53-apc-finegrained]".  Silently degrading there would hide that this
      patch's safety argument does not hold on that layout.
      GLM53_FINEGRAINED_APC=0 is the documented escape hatch.

Part C (launcher knob):
  drives start.sh's "GLM53 numeric config guard" block in bash and asserts
  GLM53_FINEGRAINED_APC is accepted only as exactly 0 or 1, the same rule the
  coordinator enforces at init.

Source of truth for the live layout (docker logs glm53-exl3-head):
  kv_cache_coordinator.py:709  hybrid APC groups:
    [('MLAAttentionSpec', [0], 'FullAttentionManager', False),
     ('MambaSpec', [2, 3, 4, 5], 'MambaManager', False),
     ('SlidingWindowSpec', [6], 'SlidingWindowManager', True)]
  -> group 1 (KpoolTailSpec) is absent: participates_in_prefix_caching=False.
  interface.py:635  kv cache block size 64 (DEEPSEEK_V32_INDEXER backend)
  interface.py:926  attention block size 3584 (>= mamba page size)
  platforms/interface.py:932-933  mamba_cache_mode=="align" -> mamba_block_size = block_size = 3584
  config.json       text_config.index_kpool = 4  -> KpoolTailSpec.block_size
                    (models/glm5next/nvidia/attention.py:191-198)
  NOTE 896 is NOT a cache boundary: it is block_size // index_kpool, the indexer
       storage block (models/glm5next/nvidia/attention.py:142).

Usage:
  python3 test_apc_fine_grained_hits.py
  GLM53_KV_COORDINATOR_PY_SRC=/path/to/kv_cache_coordinator.py \
      python3 test_apc_fine_grained_hits.py
  # optional second source for a true both-orders composition test:
  GLM53_KV_COORDINATOR_PY_PRISTINE=/tmp/kv_cache_coordinator_pristine.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent

PATCH = next(
    (
        p
        for p in (
            HERE / "patch_apc_fine_grained_hits.py",
            HERE.parent / "overlay" / "patch_apc_fine_grained_hits.py",
        )
        if p.is_file()
    ),
    None,
)

HYBRID_PATCH = next(
    (
        p
        for p in (
            HERE / "patch_hybrid_prefix_hit.py",
            # the overlay dir of THIS checkout comes first on purpose: a
            # sibling clone must never be what this test validates against.
            HERE.parent / "overlay" / "patch_hybrid_prefix_hit.py",
            HERE.parent.parent / "glm-exl3-recipe-fork" / "overlay"
            / "patch_hybrid_prefix_hit.py",
        )
        if p.is_file()
    ),
    None,
)

DEFAULT_SRC = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py"
)
DEFAULT_PRISTINE = Path("/tmp/kv_cache_coordinator_pristine.py")
MARK = "# [glm53-finegrained-apc]"
RUNTIME_TAG = "[glm53-apc-finegrained]"
HELPER_BEGIN = "# [glm53-finegrained-apc] helper-begin"
HELPER_END = "# [glm53-finegrained-apc] helper-end"

# The unscoped upstream veto this overlay replaces (kv_cache_coordinator.py
# :626-639) and the shape vLLM main e126687a gave it once upstream scoped the
# check to prefix-cacheable groups. Used by A6 to synthesize an
# already-fixed-upstream coordinator.
UPSTREAM_VETO = """        if self.enable_partial_hash_hits:
            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager in self.single_type_managers
                if not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }
"""

UPSTREAM_FIXED_VETO = """        if self.enable_partial_hash_hits:
            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager, group in zip(
                    self.single_type_managers, kv_cache_config.kv_cache_groups
                )
                if group.kv_cache_spec.prefix_cacheable
                and not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }
"""

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILURES.append(label)


def apply_patch(patch: Path, target: Path, expect_fail: bool = False) -> str:
    env = os.environ.copy()
    env["GLM53_KV_COORDINATOR_PY"] = str(target)
    env.pop("GLM53_FINEGRAINED_APC", None)
    proc = subprocess.run(
        [sys.executable, str(patch)],
        env=env,
        capture_output=True,
        text=True,
    )
    if expect_fail:
        if proc.returncode == 0:
            raise AssertionError(
                f"{patch.name} succeeded where it had to fail closed"
            )
        return proc.stderr + proc.stdout
    if proc.returncode != 0:
        raise AssertionError(
            f"{patch.name} failed: rc={proc.returncode}\n{proc.stderr}{proc.stdout}"
        )
    return proc.stdout


def no_temp_litter(directory: Path) -> bool:
    return not any(p.name.endswith(".tmp") for p in directory.iterdir())


# --------------------------------------------------------------------------
# Part A -- patch mechanics
# --------------------------------------------------------------------------


def part_a(src: Path) -> str:
    print("Part A: patch mechanics")
    patched_text = ""
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "kv_cache_coordinator.py"

        # A1 apply once
        shutil.copyfile(src, dst)
        apply_patch(PATCH, dst)
        text = dst.read_text()
        patched_text = text
        check(MARK in text, "A1 MARK present")
        check("def _glm53_finegrained_hit_gate(" in text, "A1 helper inserted")
        check("\nimport os\n" in text, "A1 'import os' ensured")
        check(
            HELPER_BEGIN in text and HELPER_END in text,
            "A1 helper block sentinels present",
        )
        check(
            "participates_in_prefix_caching"
            in text.split("def _glm53_finegrained_hit_gate")[1][:4000],
            "A1 helper checks participates_in_prefix_caching",
        )
        check("GLM53_FINEGRAINED_APC" in text, "A1 kill switch present")
        check(
            "class Glm53FineGrainedAPCError(RuntimeError):" in text,
            "A1 refusal exception type defined",
        )
        check(RUNTIME_TAG in text, f"A1 runtime tag {RUNTIME_TAG} present")
        check(
            "def _glm53_scratch_alignment(" in text,
            "A1 runtime spec-derived alignment helper inserted",
        )
        check(
            "def _glm53_connector_receipt(" in text,
            "A1 connector boot receipt helper inserted",
        )
        check(
            "_glm53_connector_receipt()" in text.split("Fine-grained prefix-cache hits ENABLED")[1][:800],
            "A1 connector receipt emitted on the enable path",
        )
        # The old veto built this set over ALL managers. It must be gone --
        # note "for manager in self.single_type_managers" alone is NOT a valid
        # probe: cache_blocks() legitimately iterates all managers too.
        check(
            "unsupported_partial_hit_managers = {" not in text,
            "A1 upstream all-managers veto comprehension removed",
        )
        check(
            "for manager in self.single_type_managers" in text,
            "A1 cache_blocks' own all-managers loop left intact",
        )
        check(
            "Fine-grained prefix-cache hits ENABLED" in text,
            "A1 enable-path log line present",
        )
        check(
            compile(text, str(dst), "exec") is not None,
            "A1 patched file compiles",
        )
        check(no_temp_litter(Path(tmp)), "A1 no temp file left behind")

        # A2 idempotence
        out = apply_patch(PATCH, dst)
        check("already present" in out, "A2 second apply is a no-op (announced)")
        check(dst.read_text() == text, "A2 second apply byte-identical")

        # A3 fail-closed on drift, transactionally (nothing written at all)
        shutil.copyfile(src, dst)
        drifted = dst.read_text().replace(
            "if not manager.supports_fine_grained_hash_lookup",
            "if not manager.supports_fine_grained_hash_lookup  # drifted",
            1,
        )
        dst.write_text(drifted)
        err = apply_patch(PATCH, dst, expect_fail=True)
        check(
            "expected one partial-hit-gate target" in err,
            "A3 fails closed when the gate anchor drifts",
        )
        check(MARK not in dst.read_text(), "A3 drifted file left unpatched")
        check(
            dst.read_text() == drifted,
            "A3 transactional: drifted file byte-identical after failure",
        )
        check(no_temp_litter(Path(tmp)), "A3 no temp file left behind on failure")

        # A3b fail-closed when the helper insert point drifts
        shutil.copyfile(src, dst)
        t = dst.read_text().replace(
            "def _validate_prefix_cache_retention_interval(\n",
            "def _validate_prefix_cache_retention_interval_RENAMED(\n",
            1,
        )
        dst.write_text(t)
        err = apply_patch(PATCH, dst, expect_fail=True)
        check(
            "helper insert point" in err,
            "A3b fails closed when the helper insert point drifts",
        )
        check(dst.read_text() == t, "A3b transactional: file untouched")

        # A5 a pre-existing MARK is not trusted on its own: an incomplete
        # patched state must fail closed, not be skipped as "already done".
        for label, mutate in (
            (
                "helper block deleted",
                lambda s: re.sub(
                    re.escape(HELPER_BEGIN) + r".*?" + re.escape(HELPER_END),
                    "# gutted",
                    s,
                    flags=re.S,
                ),
            ),
            (
                "kill switch stripped",
                lambda s: s.replace("GLM53_FINEGRAINED_APC", "GLM53_DISABLED_KNOB"),
            ),
            (
                "upstream veto reintroduced",
                lambda s: s.replace(
                    "        self.enable_partial_hash_hits = _glm53_ok\n",
                    "        unsupported_partial_hit_managers = {}\n"
                    "        self.enable_partial_hash_hits = _glm53_ok\n",
                    1,
                ),
            ),
        ):
            shutil.copyfile(src, dst)
            apply_patch(PATCH, dst)
            broken = mutate(dst.read_text())
            dst.write_text(broken)
            err = apply_patch(PATCH, dst, expect_fail=True)
            check(
                "INCOMPLETE" in err,
                f"A5 pre-existing MARK + {label} -> fails closed",
            )
            check(
                dst.read_text() == broken,
                f"A5 pre-existing MARK + {label} -> file untouched",
            )

        # A6 upstream landed the same fix (vLLM main e126687a scopes the veto
        # to KVCacheSpec.prefix_cacheable groups; its scratch invariant is the
        # tokens_per_state check in resolve_kv_cache_block_sizes). A recipe
        # image rebased onto that vLLM must be a clean no-op, not a boot
        # failure from a missing anchor.
        shutil.copyfile(src, dst)
        upstream_fixed = dst.read_text().replace(
            UPSTREAM_VETO, UPSTREAM_FIXED_VETO, 1
        )
        check(
            upstream_fixed != dst.read_text(),
            "A6 fixture: the upstream veto anchor was found and replaced",
        )
        dst.write_text(upstream_fixed)
        out = apply_patch(PATCH, dst)
        check(
            "upstream already scopes the veto; nothing to do" in out,
            "A6 upstream-fixed coordinator -> no-op, announced",
        )
        check(RUNTIME_TAG in out, f"A6 the no-op message carries {RUNTIME_TAG}")
        check(
            dst.read_text() == upstream_fixed,
            "A6 upstream-fixed coordinator left byte-identical",
        )
        check(no_temp_litter(Path(tmp)), "A6 no temp file left behind")
        out = apply_patch(PATCH, dst)
        check(
            "nothing to do" in out and dst.read_text() == upstream_fixed,
            "A6 re-running on an upstream-fixed coordinator is still a no-op",
        )

        # A6b the veto is GONE entirely -- that is drift, not an upstream fix.
        shutil.copyfile(src, dst)
        gutted = dst.read_text().replace(UPSTREAM_VETO, "", 1)
        dst.write_text(gutted)
        err = apply_patch(PATCH, dst, expect_fail=True)
        check(
            "expected one partial-hit-gate target" in err,
            "A6b a deleted veto still fails closed (not read as 'upstream fixed')",
        )
        check(dst.read_text() == gutted, "A6b transactional: file untouched")

        # A6c an unscoped veto that merely drifted must still fail closed, even
        # though `participates_in_prefix_caching` appears ~25 lines below it in
        # verify_and_split_kv_cache_groups. A text window would read that
        # neighbour as proof of a fix; the indentation-scoped extractor does not.
        shutil.copyfile(src, dst)
        drifted2 = dst.read_text().replace(
            "                if not manager.supports_fine_grained_hash_lookup\n",
            "                if not manager.supports_fine_grained_hash_lookup  # x\n",
            1,
        )
        check(drifted2 != dst.read_text(), "A6c fixture: veto line drifted")
        dst.write_text(drifted2)
        err = apply_patch(PATCH, dst, expect_fail=True)
        check(
            "expected one partial-hit-gate target" in err,
            "A6c drifted-but-unscoped veto still fails closed",
        )
        check(
            "participates_in_prefix_caching" in drifted2,
            "A6c the neighbour attribute really is present in the file",
        )

    # A4 composability with patch_hybrid_prefix_hit.py, both orders.
    # Run over every available source: the caller's src (which on this kit is
    # the LIVE file, i.e. hybrid already applied -> also an idempotence probe)
    # and, when available, a pristine upstream copy where both patches really
    # do apply.
    if HYBRID_PATCH is None:
        print("  skip A4 (patch_hybrid_prefix_hit.py not found)")
    else:
        pristine = Path(
            os.environ.get("GLM53_KV_COORDINATOR_PY_PRISTINE", DEFAULT_PRISTINE)
        )
        sources = [("src", src)]
        if pristine.is_file() and pristine.resolve() != src.resolve():
            sources.append(("pristine", pristine))
        else:
            print("  note A4: no separate pristine source; running over src only")
        for src_label, src_path in sources:
            results = {}
            for order in (("fine", "hybrid"), ("hybrid", "fine")):
                with tempfile.TemporaryDirectory() as tmp:
                    dst = Path(tmp) / "kv_cache_coordinator.py"
                    shutil.copyfile(src_path, dst)
                    for which in order:
                        apply_patch(
                            PATCH if which == "fine" else HYBRID_PATCH, dst
                        )
                    t = dst.read_text()
                    results[order] = t
                    tag = f"A4[{src_label}] {order[0]}->{order[1]}"
                    check(
                        MARK in t and "[glm53-hybrid-apc]" in t,
                        f"{tag}: both MARKs present",
                    )
                    check(
                        t.count("def _glm53_finegrained_hit_gate(") == 1
                        and t.count("def _glm53_inner_kv_spec(") == 1,
                        f"{tag}: each helper inserted exactly once",
                    )
                    try:
                        compile(t, str(dst), "exec")
                        ok = True
                    except SyntaxError as exc:  # pragma: no cover
                        ok = False
                        print(f"       syntax error: {exc}")
                    check(ok, f"{tag}: compiles")
                    # re-apply both, in the same order: must be a no-op
                    for which in order:
                        apply_patch(
                            PATCH if which == "fine" else HYBRID_PATCH, dst
                        )
                    check(
                        dst.read_text() == t,
                        f"{tag}: re-applying both is byte-identical (idempotent)",
                    )
            check(
                results[("fine", "hybrid")] == results[("hybrid", "fine")],
                f"A4[{src_label}] patch order is commutative (identical bytes)",
            )

    return patched_text


# --------------------------------------------------------------------------
# Part B -- gate semantics
# --------------------------------------------------------------------------


class FakeSpec:
    def __init__(self, name: str, participates: bool = True, **attrs) -> None:
        self._name = name
        self.participates_in_prefix_caching = participates
        for key, value in attrs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:  # pragma: no cover
        return self._name


class FakeGroup:
    def __init__(self, spec) -> None:
        self.kv_cache_spec = spec


def make_manager(cls_name: str, block_size, fine: bool, **attrs):
    body = {}
    if fine is not None:
        body["supports_fine_grained_hash_lookup"] = fine
    cls = type(cls_name, (), body)
    obj = cls()
    if block_size is not None:
        obj.block_size = block_size
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


GATE_BLOCK_START = "        if self.enable_partial_hash_hits:\n            # [glm53-finegrained-apc]"
GATE_BLOCK_END = "        self.verify_and_split_kv_cache_groups()"


def extract_gate_block(patched_text: str) -> str:
    """The patched gate, lifted out of HybridKVCacheCoordinator.__init__.

    Kept at its original 8-space indent so it can be re-hosted verbatim under a
    `def` header -- what runs here is the shipped text, not a paraphrase.
    """
    start = patched_text.find(GATE_BLOCK_START)
    end = patched_text.find(GATE_BLOCK_END, start)
    if start < 0 or end < 0:
        raise AssertionError("could not extract the patched gate block")
    return patched_text[start:end]


def extract_helpers(patched_text: str) -> dict:
    """exec the whole injected helper block in a bare namespace (no vllm)."""
    m = re.search(
        re.escape(HELPER_BEGIN) + r"(.*?)" + re.escape(HELPER_END),
        patched_text,
        re.S,
    )
    if m is None:
        raise AssertionError("could not extract the glm53 helper block")
    ns: dict = {}
    exec(compile(m.group(1), "<helper>", "exec"), ns)  # noqa: S102
    return ns


def spec_for(name: str, participates: bool, block_size, extra: dict | None):
    attrs = dict(extra or {})
    if block_size is not None and "block_size" not in attrs:
        attrs["block_size"] = block_size
    return FakeSpec(name, participates, **attrs)


LIVE_LAYOUT = [
    # (manager cls, block_size, supports_fine, participates, spec extras)
    ("FullAttentionManager", 3584, True, True, None),
    ("KpoolTailManager", 4, False, False, {"index_kpool": 4}),
    ("MambaManager", 3584, True, True, None),
    ("MambaManager", 3584, True, True, None),
    ("MambaManager", 3584, True, True, None),
    ("MambaManager", 3584, True, True, None),
    ("SlidingWindowManager", 64, False, True, None),
]

# label, layout, hash_bs, expect ("enable"/"disable"/"raise"), must_mention
CASES = [
    (
        "B1 live layout (MLA 3584 / KpoolTail 4 / 4x Mamba 3584 / SWA 64), hash=64",
        LIVE_LAYOUT,
        64,
        "enable",
        None,
    ),
    (
        "B3 kpool=128 (hypothetical): 64 % 128 != 0 -> must REFUSE (raise)",
        [
            ("FullAttentionManager", 3584, True, True, None),
            ("KpoolTailManager", 128, False, False, {"index_kpool": 128}),
            ("MambaManager", 3584, True, True, None),
            ("SlidingWindowManager", 64, False, True, None),
        ],
        64,
        "raise",
        "is not a multiple of it",
    ),
    (
        "B4 kpool=32 divides 64 -> ENABLE (any divisor of hash_block_size is safe)",
        [
            ("FullAttentionManager", 3584, True, True, None),
            ("KpoolTailManager", 32, False, False, {"index_kpool": 32}),
            ("MambaManager", 3584, True, True, None),
            ("SlidingWindowManager", 64, False, True, None),
        ],
        64,
        "enable",
        None,
    ),
    (
        "B5 participating coarse manager without fine lookup -> DISABLE (not raise)",
        [
            ("FullAttentionManager", 3584, True, True, None),
            ("KpoolTailManager", 4, False, False, {"index_kpool": 4}),
            ("MambaManager", 3584, True, True, None),
            ("SlidingWindowManager", 3584, False, True, None),  # SWA raised
        ],
        64,
        "disable",
        "participating",
    ),
    (
        "B6 SWA at block_size == hash_block_size is fine despite flag=False",
        [
            ("FullAttentionManager", 3584, True, True, None),
            ("SlidingWindowManager", 64, False, True, None),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "enable",
        None,
    ),
    (
        "B7 zero scratch block_size -> REFUSE (no division by zero)",
        [
            ("FullAttentionManager", 3584, True, True, None),
            ("KpoolTailManager", 0, False, False, None),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "raise",
        "requires hit alignment 0",
    ),
    (
        "B8 spec without participates_in_prefix_caching defaults to participating",
        [
            ("FullAttentionManager", 3584, False, None, None),
            ("MambaManager", 3584, True, None, None),
        ],
        64,
        "disable",
        "participating",
    ),
    (
        "B10 scratch spec exposes no block_size -> unverifiable -> REFUSE",
        [
            ("FullAttentionManager", 3584, True, True, None),
            ("KpoolTailManager", 4, False, False, {"block_size": None}),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "raise",
        "cannot verify",
    ),
    (
        "B11 scratch spec.block_size disagrees with manager.block_size -> REFUSE",
        [
            ("FullAttentionManager", 3584, True, True, None),
            ("KpoolTailManager", 4, False, False, {"block_size": 8}),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "raise",
        "disagrees with",
    ),
    (
        "B12 scratch spec.index_kpool disagrees with block_size -> REFUSE",
        [
            ("FullAttentionManager", 3584, True, True, None),
            ("KpoolTailManager", 4, False, False, {"index_kpool": 6}),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "raise",
        "index_kpool=6 disagrees",
    ),
    (
        "B13 explicit fine_grained_hit_alignment agreeing with block_size -> ENABLE",
        [
            ("FullAttentionManager", 3584, True, True, None),
            (
                "FutureScratchManager",
                16,
                False,
                False,
                {"fine_grained_hit_alignment": 16},
            ),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "enable",
        None,
    ),
    (
        "B13b explicit capability is the ONLY source (no block_size anywhere) "
        "-> ENABLE (the forward-compat hook still works)",
        [
            ("FullAttentionManager", 3584, True, True, None),
            (
                "FutureScratchManager",
                None,
                False,
                False,
                {"fine_grained_hit_alignment": 16},
            ),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "enable",
        None,
    ),
    (
        "B13c capability 16 CONTRADICTS block_size 128 -> REFUSE. 16 divides 64, "
        "so trusting the capability would have ENABLED unsafe hits on a group "
        "that needs 128",
        [
            ("FullAttentionManager", 3584, True, True, None),
            (
                "FutureScratchManager",
                128,
                False,
                False,
                {"fine_grained_hit_alignment": 16},
            ),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "raise",
        "disagrees with",
    ),
    (
        "B13d capability contradicts index_kpool -> REFUSE",
        [
            ("FullAttentionManager", 3584, True, True, None),
            (
                "KpoolTailManager",
                4,
                False,
                False,
                {"fine_grained_hit_alignment": 4, "index_kpool": 8},
            ),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "raise",
        "index_kpool=8 disagrees",
    ),
    (
        "B14 explicit fine_grained_hit_alignment that does NOT divide -> REFUSE",
        [
            ("FullAttentionManager", 3584, True, True, None),
            (
                "FutureScratchManager",
                96,
                False,
                False,
                {"fine_grained_hit_alignment": 96},
            ),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "raise",
        "requires hit alignment 96",
    ),
    (
        "B23 capability '4.5' -> REFUSE (int() would have truncated it to 4, "
        "which divides 64)",
        [
            ("FullAttentionManager", 3584, True, True, None),
            (
                "FutureScratchManager",
                None,
                False,
                False,
                {"fine_grained_hit_alignment": "4.5"},
            ),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "raise",
        "is not an integer",
    ),
    (
        "B23b scratch spec.block_size ' 4' (padded string) -> REFUSE",
        [
            ("FullAttentionManager", 3584, True, True, None),
            ("KpoolTailManager", 4, False, False, {"block_size": " 4"}),
            ("MambaManager", 3584, True, True, None),
        ],
        64,
        "raise",
        "is not an integer",
    ),
    (
        "B24 MIXED: participating blocker AND a violating scratch group "
        "-> DISABLE safely (upstream behaviour), do NOT raise",
        [
            ("FullAttentionManager", 3584, True, True, None),
            ("KpoolTailManager", 128, False, False, {"index_kpool": 128}),
            ("MambaManager", 3584, True, True, None),
            ("SlidingWindowManager", 3584, False, True, None),  # SWA raised
        ],
        64,
        "disable",
        "participating",
    ),
    (
        "B25 MIXED: participating blocker AND an UNVERIFIABLE scratch group "
        "-> DISABLE safely, do NOT raise",
        [
            ("FullAttentionManager", 3584, True, True, None),
            ("KpoolTailManager", 4, False, False, {"block_size": None}),
            ("MambaManager", 3584, True, True, None),
            ("SlidingWindowManager", 3584, False, True, None),  # SWA raised
        ],
        64,
        "disable",
        "participating",
    ),
    (
        "B15 manager missing supports_fine_grained_hash_lookup -> treated as False",
        [
            ("FullAttentionManager", 3584, None, True, None),
            ("KpoolTailManager", 4, False, False, {"index_kpool": 4}),
        ],
        64,
        "disable",
        "participating",
    ),
]


def build(layout):
    managers = []
    groups = []
    for cls_name, block_size, fine, participates, extra in layout:
        managers.append(make_manager(cls_name, block_size, fine))
        if participates is None:
            class LegacySpec:  # no participates_in_prefix_caching attribute
                pass

            spec = LegacySpec()
            if block_size is not None:
                spec.block_size = block_size
            groups.append(FakeGroup(spec))
        else:
            groups.append(
                FakeGroup(spec_for(cls_name, participates, block_size, extra))
            )
    return managers, groups


def part_b(patched_text: str) -> None:
    print("Part B: gate semantics")
    ns = extract_helpers(patched_text)
    gate = ns["_glm53_finegrained_hit_gate"]
    err_cls = ns["Glm53FineGrainedAPCError"]

    for label, layout, hash_bs, expect, must_mention in CASES:
        managers, groups = build(layout)
        try:
            enable, blockers, scratch = gate(managers, groups, hash_bs)
            raised = None
        except err_cls as exc:
            enable, blockers, scratch, raised = None, [], {}, exc

        if expect == "raise":
            ok = raised is not None
            detail = f"raised={type(raised).__name__ if raised else None}"
            if ok:
                msg = str(raised)
                ok = RUNTIME_TAG in msg
                detail = f"{msg[:150]}..."
                if must_mention is not None:
                    ok = ok and must_mention in msg
                ok = ok and "GLM53_FINEGRAINED_APC=0" in msg
        else:
            ok = raised is None and enable is (expect == "enable")
            detail = f"enable={enable} blockers={blockers} scratch={scratch}"
            if must_mention is not None:
                ok = ok and any(must_mention in b for b in blockers)
        check(ok, f"{label} -> {detail}")

    # B2: the upstream rule vetoes the live layout (what this patch fixes).
    managers, _groups = build(LIVE_LAYOUT)
    upstream_blockers = {
        type(m).__name__
        for m in managers
        if not getattr(m, "supports_fine_grained_hash_lookup", False)
        and m.block_size != 64
    }
    check(
        upstream_blockers == {"KpoolTailManager"},
        f"B2 upstream rule would veto the live layout -> {sorted(upstream_blockers)}",
    )

    # B16: cardinality mismatch must fail closed, not silently truncate.
    managers, groups = build(LIVE_LAYOUT)
    try:
        gate(managers, groups[:-1], 64)
        ok, detail = False, "no raise"
    except err_cls as exc:
        ok = (
            RUNTIME_TAG in str(exc)
            and "cardinality mismatch" in str(exc)
        )
        detail = str(exc)[:120]
    check(ok, f"B16 manager/group cardinality mismatch REFUSES -> {detail}")

    # B16b: the truncation zip() would have caused is a real miss, not cosmetic
    # -- drop the trailing coarse SWA blocker and upstream's zip would pass it.
    bad_layout = LIVE_LAYOUT[:-1] + [
        ("SlidingWindowManager", 3584, False, True, None)
    ]
    managers, groups = build(bad_layout)
    try:
        gate(managers, groups[:-1], 64)
        ok, detail = False, "no raise (a real blocker was silently truncated)"
    except err_cls as exc:
        ok = "cardinality mismatch" in str(exc)
        detail = "refused"
    check(ok, f"B16b truncation would have hidden a real blocker -> {detail}")

    # B17: a group with no kv_cache_spec at all.
    managers, groups = build(LIVE_LAYOUT)
    groups[1] = FakeGroup.__new__(FakeGroup)
    try:
        gate(managers, groups, 64)
        ok, detail = False, "no raise"
    except err_cls as exc:
        ok = RUNTIME_TAG in str(exc) and "no kv_cache_spec" in str(exc)
        detail = str(exc)[:120]
    except AttributeError:
        ok, detail = False, "AttributeError (not fail-closed)"
    check(ok, f"B17 group without kv_cache_spec REFUSES -> {detail}")

    # B18: a nonsense hash_block_size.
    for bad in (0, -64, None, 64.0):
        managers, groups = build(LIVE_LAYOUT)
        try:
            gate(managers, groups, bad)
            ok = False
        except err_cls as exc:
            ok = RUNTIME_TAG in str(exc) and "hash_block_size" in str(exc)
        check(ok, f"B18 hash_block_size={bad!r} REFUSES")

    # B19: the scratch alignment really is read from the spec, not assumed.
    scratch_align = ns["_glm53_scratch_alignment"]
    mgr = make_manager("KpoolTailManager", 4, False)
    spec = FakeSpec("KpoolTailSpec", False, block_size=4, index_kpool=4)
    align, source = scratch_align(mgr, spec)
    check(
        align == 4 and "spec.block_size" in source and "index_kpool" in source,
        f"B19 alignment 4 verified from the actual spec -> ({align}, {source})",
    )
    check(
        scratch_align(mgr, FakeSpec("X", False, block_size=4))[0] == 4,
        "B19 spec without index_kpool still verified via block_size",
    )

    # B20: the connector receipt never raises, even with no vLLM present.
    receipt = ns["_glm53_connector_receipt"]
    try:
        value = receipt()
        ok = isinstance(value, str) and value != ""
    except Exception:  # pragma: no cover
        ok, value = False, "raised"
    check(ok, f"B20 connector boot receipt is non-fatal -> {value!r}")

    # B22: the kill switch, executed rather than grepped. Extracts the patched
    # gate block out of __init__ and runs it against fakes, so
    # GLM53_FINEGRAINED_APC=0 is proven to (a) disable and (b) NOT raise even on
    # a layout the gate would otherwise refuse to start on.
    block = extract_gate_block(patched_text)
    runner = compile(
        "def _glm53_run(self, kv_cache_config, hash_block_size, os, logger):\n"
        + block,
        "<gate-block>",
        "exec",
    )
    run_ns = dict(ns)
    exec(runner, run_ns)  # noqa: S102
    run = run_ns["_glm53_run"]

    class RecordingLogger:
        """Renders every call the way logging would, so the receipt is testable."""

        def __init__(self) -> None:
            self.lines: list[str] = []

        def _record(self, msg, *args) -> None:
            self.lines.append(msg % args if args else msg)

        def info(self, msg, *args, **kwargs):
            self._record(msg, *args)

        def warning(self, msg, *args, **kwargs):
            self._record(msg, *args)

        def warning_once(self, msg, *args, **kwargs):
            self._record(msg, *args)

    def drive(layout, kill_switch):
        """Run the shipped gate block; return (enable_partial_hash_hits, log)."""
        managers, groups = build(layout)
        obj = types.SimpleNamespace(
            enable_partial_hash_hits=True,
            single_type_managers=managers,
            scheduler_block_size=3584,
        )
        cfg = types.SimpleNamespace(kv_cache_groups=groups)
        env = dict(os.environ)
        env.pop("GLM53_FINEGRAINED_APC", None)
        if kill_switch is not None:
            env["GLM53_FINEGRAINED_APC"] = kill_switch
        fake_os = types.SimpleNamespace(environ=env)
        log = RecordingLogger()
        run(obj, cfg, 64, fake_os, log)
        return obj.enable_partial_hash_hits, log.lines

    BAD = [
        ("FullAttentionManager", 3584, True, True, None),
        ("KpoolTailManager", 128, False, False, {"index_kpool": 128}),
        ("MambaManager", 3584, True, True, None),
    ]
    # BAD plus a participating blocker: the mixed cell, through the real block.
    BAD_MIXED = BAD + [("SlidingWindowManager", 3584, False, True, None)]

    enabled, log = drive(LIVE_LAYOUT, None)
    check(enabled is True, "B22 gate block enables on the live layout")
    check(
        drive(LIVE_LAYOUT, "0")[0] is False,
        "B22 GLM53_FINEGRAINED_APC=0 disables fine hits",
    )
    check(
        drive(LIVE_LAYOUT, "1")[0] is True,
        "B22 GLM53_FINEGRAINED_APC=1 is the default (enabled)",
    )
    try:
        drive(BAD, None)
        ok = False
    except err_cls:
        ok = True
    check(ok, "B22 a violating layout REFUSES through the real gate block")
    try:
        ok = drive(BAD, "0")[0] is False
    except err_cls:
        ok = False
    check(ok, "B22 GLM53_FINEGRAINED_APC=0 escapes the refusal (documented opt-out)")

    # ---------------------------------------------------------------- B26
    # Strict integer parsing. int() would accept every value in the reject
    # list, and truncating "4.5" to 4 (which divides 64) is exactly how an
    # unverified alignment gets a verified badge.
    strict_int = ns["_glm53_strict_int"]
    for value, expected in (
        (4, 4),
        (0, 0),
        (-4, -4),
        ("4", 4),
        ("+4", 4),
        ("0004", 4),
        ("-4", -4),
    ):
        check(strict_int(value) == expected, f"B26 strict int accepts {value!r}")
    for value in (
        4.5,
        4.0,
        "4.5",
        " 4",
        "4 ",
        "",
        "0x40",
        "4_0",
        "4\n",
        "\u0664",     # ARABIC-INDIC DIGIT FOUR: str.isdigit() is True
        "\u00b2",     # SUPERSCRIPT TWO: str.isdigit() is True
        True,
        False,
        None,
        object(),
    ):
        check(strict_int(value) is None, f"B26 strict int REJECTS {value!r}")
    check(
        int("4.5".replace(".5", "")) == 4 and strict_int("4.5") is None,
        "B26 the value int() would have silently truncated is refused instead",
    )

    # ---------------------------------------------------------------- B27
    # The kill switch is exactly '0' or '1'; anything else refuses at init.
    enabled_fn = ns["_glm53_finegrained_enabled"]
    check(enabled_fn("1") is True, "B27 kill switch '1' -> on")
    check(enabled_fn("0") is False, "B27 kill switch '0' -> off")
    for bad in ("", " ", "0 ", " 0", "01", "00", "2", "-0", "true", "false",
                "yes", "no", "on", "off", "True", "1.0"):
        try:
            enabled_fn(bad)
            ok, detail = False, "accepted"
        except err_cls as exc:
            ok = RUNTIME_TAG in str(exc) and "GLM53_FINEGRAINED_APC" in str(exc)
            detail = "refused"
        check(ok, f"B27 kill switch {bad!r} REFUSES at init -> {detail}")
    for bad in ("true", "", "01"):
        try:
            drive(LIVE_LAYOUT, bad)
            ok = False
        except err_cls:
            ok = True
        check(ok, f"B27 GLM53_FINEGRAINED_APC={bad!r} refuses through the gate block")

    # ---------------------------------------------------------------- B28
    # The 2x2 mixed-layout matrix (DESIGN 4.3), all four cells, driven through
    # the shipped gate block rather than the helper alone.
    SAFE_SCRATCH = ("KpoolTailManager", 4, False, False, {"index_kpool": 4})
    BAD_SCRATCH = ("KpoolTailManager", 128, False, False, {"index_kpool": 128})
    COARSE_BLOCKER = ("SlidingWindowManager", 3584, False, True, None)
    BASE = [("FullAttentionManager", 3584, True, True, None),
            ("MambaManager", 3584, True, True, None)]
    matrix = [
        ("no blocker  + scratch ok  -> ENABLE", BASE + [SAFE_SCRATCH], "enable"),
        ("no blocker  + scratch bad -> RAISE", BASE + [BAD_SCRATCH], "raise"),
        ("blocker     + scratch ok  -> DISABLE",
         BASE + [SAFE_SCRATCH, COARSE_BLOCKER], "disable"),
        ("blocker     + scratch bad -> DISABLE (safe, not a boot failure)",
         BASE + [BAD_SCRATCH, COARSE_BLOCKER], "disable"),
    ]
    for label, layout, expected in matrix:
        try:
            enabled, lines = drive(layout, None)
            got = "enable" if enabled else "disable"
        except err_cls:
            got = "raise"
            lines = []
        check(got == expected, f"B28 matrix {label} -> {got}")
    # and the cell the reviewer flagged keeps the diagnosis in the receipt
    _enabled, lines = drive(BAD_MIXED, None)
    check(
        any("UNSAFE" in line and "128" in line for line in lines),
        "B28 the tolerated scratch fault is still named in the receipt",
    )

    # ---------------------------------------------------------------- B29
    # Effective-value receipt: one line stating enabled/disabled, the reason,
    # the alignment actually in force, and the scratch groups checked. This is
    # the line DESIGN 6.5 B0 greps out of `docker logs` on BOTH ranks.
    _enabled, lines = drive(LIVE_LAYOUT, None)
    receipt = next((l for l in lines if RUNTIME_TAG in l), "")
    for token in (
        "Fine-grained prefix-cache hits ENABLED",
        "reason=",
        "effective alignment=64 tokens",
        "scheduler_block_size=3584",
        "scratch groups checked: {'KpoolTailManager': 4}",
        "KV-transfer connector:",
    ):
        check(token in receipt, f"B29 enabled receipt states {token!r}")

    _enabled, lines = drive(LIVE_LAYOUT, "0")
    receipt = next((l for l in lines if RUNTIME_TAG in l), "")
    for token in (
        "Fine-grained prefix-cache hits DISABLED",
        "reason=GLM53_FINEGRAINED_APC=0 (kill switch)",
        "effective alignment=3584 tokens",
        "hash_block_size=64",
        "scratch groups checked:",
    ):
        check(token in receipt, f"B29 kill-switch receipt states {token!r}")

    _enabled, lines = drive(BAD_MIXED, None)
    receipt = next((l for l in lines if RUNTIME_TAG in l), "")
    for token in (
        "Fine-grained prefix-cache hits DISABLED",
        "reason=participating managers require block-aligned lookups",
        "effective alignment=3584 tokens",
        "scratch groups checked:",
    ):
        check(token in receipt, f"B29 mixed-layout receipt states {token!r}")
    check(
        any(
            "Disabling fine-grained prefix-cache hits because these KV cache "
            "managers require block-aligned lookups" in line
            for line in lines
        ),
        "B29 upstream's own warning is still emitted on the disable path",
    )

    # B9: the arithmetic claim the whole design rests on.
    check(64 % 4 == 0, "B9 hash_block_size 64 is a multiple of index_kpool 4")
    check(3584 % 64 == 0, "B9 3584 (MLA and mamba block) is a multiple of hash 64")
    check(
        3584 // 4 == 896,
        "B9 896 is block_size//index_kpool (indexer storage block), NOT a hit boundary",
    )
    check(
        __import__("math").lcm(64, 4, 64, 64) == 64,
        "B9 lcm(hash 64, kpool 4, drafter 64, mamba-snapshot granularity 64) == 64",
    )
    # B21: the exact hit receipts the live plan asserts (DESIGN 6.2/6.3).
    P_TURN1 = 3584 * 8 + 3000  # 31672
    check(P_TURN1 == 31672, "B21 turn-1 prompt P = 3584*8 + 3000 = 31672")
    check(
        P_TURN1 // 64 * 64 == 31616,
        "B21 fine hit floor(P/64)*64 == 31616 (the receipt to assert live)",
    )
    check(
        P_TURN1 // 3584 * 3584 == 28672,
        "B21 coarse hit floor(P/3584)*3584 == 28672 (the pre-patch control)",
    )
    check(
        31616 % 3584 != 0 and P_TURN1 % 64 != 0,
        "B21 31616 is off the old grid and P forces a tail_boundary stop",
    )
    check(31616 % 4 == 0, "B21 the fine hit lands on an empty kpool (31616 % 4 == 0)")


# --------------------------------------------------------------------------
# Part C -- the launcher knob
# --------------------------------------------------------------------------

START_SH = HERE.parent / "start.sh"


def guard_source() -> str:
    """start.sh's numeric-config guard block, lifted out by its sentinels.

    Same technique as tests/test_numeric_config.py: the block is sourced on its
    own so the assertion is about the shipped shell, not a paraphrase of it.
    """
    source = START_SH.read_text()
    begin = source.index("# GLM53 numeric config guard (begin)")
    end_marker = "# GLM53 numeric config guard (end)"
    end = source.index(end_marker, begin) + len(end_marker)
    return source[begin:end]


def run_guard(flag: str | None) -> int:
    script = (
        guard_source()
        + '\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4\n'
        + 'MAX_NUM_BATCHED_TOKENS=1024\n'
        + 'validate_numeric_config || exit $?\n'
        + 'printf "ok %s\\n" "${GLM53_FINEGRAINED_APC-unset}"\n'
    )
    env = {**os.environ, "LC_ALL": "C"}
    env.pop("GLM53_FINEGRAINED_APC", None)
    if flag is not None:
        env["GLM53_FINEGRAINED_APC"] = flag
    return subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    ).returncode


def part_c() -> None:
    print("Part C: launcher knob (start.sh)")
    if not START_SH.is_file():
        print("  skip C (start.sh not found)")
        return

    source = START_SH.read_text()

    # C1 the knob is validated by the same guard as the numeric config, which
    # main() runs only on start|restart -- and, for restart, before `stop`.
    check(
        "_glm53_validate_bool_flag GLM53_FINEGRAINED_APC" in guard_source(),
        "C1 GLM53_FINEGRAINED_APC is validated inside the numeric config guard",
    )
    main_at = source.index("main() {")
    check(
        source.index("start|restart) validate_numeric_config", main_at)
        < source.index("restart)  stop; start", main_at),
        "C1 validation still runs on start|restart, before stop",
    )

    # C2 exactly 0 or 1, matching _glm53_finegrained_enabled in the overlay.
    for value in (None, "0", "1"):
        check(run_guard(value) == 0, f"C2 GLM53_FINEGRAINED_APC={value!r} accepted")
    for value in ("", " ", "0 ", " 0", "01", "00", "2", "-1", "true", "false",
                  "yes", "no", "on", "off", "True", "1.0", "0\r"):
        check(
            run_guard(value) == 2,
            f"C2 GLM53_FINEGRAINED_APC={value!r} rejected with rc=2",
        )

    # C3 the validated value is what both rank containers actually receive.
    check(
        '-e "GLM53_FINEGRAINED_APC=$GLM53_FINEGRAINED_APC"' in source,
        "C3 the knob is exported to the containers by value",
    )
    nccl = source.index("local -a nccl_common=(")
    check(
        nccl < source.index('-e "GLM53_FINEGRAINED_APC=', nccl)
        < source.index(")\n", nccl),
        "C3 it rides nccl_common, so head AND worker get the same value",
    )
    caller = source.index('_cli_finegrained="${GLM53_FINEGRAINED_APC-}"')
    check(
        caller < source.index('source "$SCRIPT_DIR/.env"')
        < source.index('[ -n "${_cli_finegrained}" ]'),
        "C3 a caller-supplied value is captured before .env and restored after",
    )


def main() -> int:
    if PATCH is None:
        raise SystemExit("missing patch_apc_fine_grained_hits.py")
    src = Path(os.environ.get("GLM53_KV_COORDINATOR_PY_SRC", DEFAULT_SRC))
    if not src.is_file():
        raise SystemExit(
            f"missing kv_cache_coordinator.py at {src}\n"
            "Set GLM53_KV_COORDINATOR_PY_SRC to a copy pulled from the image, e.g.\n"
            "  ssh ... 'docker exec glm53-exl3-head cat "
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/"
            "kv_cache_coordinator.py' > /tmp/kv_cache_coordinator.py"
        )
    print(f"source: {src}")
    if HYBRID_PATCH is not None:
        print(f"hybrid overlay: {HYBRID_PATCH}")
    patched_text = part_a(src)
    part_b(patched_text)
    part_c()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    print("fine-grained APC patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
