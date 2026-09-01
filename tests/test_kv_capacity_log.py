#!/usr/bin/env python3
"""Tests for the honest KV-capacity boot log overlay (host-only, no GPU, no torch).

Part A  the derivation, driven through the SAME helper text the overlay writes
        into kv_cache_utils.py (exec'd, not re-implemented): the deployment's
        hybrid layout reproduces the logged stock line (643 ids / 414 per
        request -> 1,553,140 tokens, 1.55x) and the runbook numbers (642 usable
        ids, 38 ids per 3584-token segment, 57,344 tokens); the 820-id boot;
        single-group / uniform-type / EAGLE / unmodelled / null-block edges;
        the knob matrix; failure modes (malformed knob raises, derivation
        error -> warning, no config mutation, hashable log arguments).
Part B  patch mechanics on a fixture built from verbatim excerpts of the live
        container's kv_cache_utils.py (both pinned anchors in file order, in a
        module that compiles): stock line byte-identical after patching,
        end-to-end call through the patched update_kv_cache_capacity with
        vllm stubbed, idempotent re-apply, per-anchor drift -> non-zero exit
        with NOTHING written, partial / altered markers refused, missing
        binding refused, --preflight, pyc clear; the real installed file when
        present (mandatory with GLM53_REQUIRE_TARGET=1, i.e. in the image).
Part C  launcher wiring: the numeric-config guard accepts exactly 0/1 for
        GLM53_KV_CAPACITY_LOG (unset -> 1), caller capture is set-ness aware,
        artifact guard entry, scp + both mounts + `-e` forward, overlay order
        after patch_glm5_drafter_group.py, Dockerfile test-before-patch,
        .env.example and README rows.
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (HERE / "patch_kv_capacity_log.py", ROOT / "overlay" / "patch_kv_capacity_log.py")
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
import patch_kv_capacity_log as P  # noqa: E402

START = ROOT / "start.sh"
DOCKERFILE = ROOT / "Dockerfile"
ENV_EXAMPLE = ROOT / ".env.example"
README = ROOT / "README.md"

CHECKS = 0
FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    global CHECKS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


# --------------------------------------------------------------------------
# synthetic specs, duck-typed exactly as the fork's kv_cache_interface.py
# (class NAMES and MRO matter: the helpers classify by them)
# --------------------------------------------------------------------------
class _Ns:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class KVCacheSpec:
    def __init__(self, block_size: int, page: int):
        self.block_size = block_size
        self._page = page

    @property
    def page_size_bytes(self) -> int:
        return self._page


class AttentionSpec(KVCacheSpec):
    pass


class FullAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, cfg) -> int:  # I:295-300
        return P.cdiv(cfg.model_config.max_model_len, self.block_size) * self.page_size_bytes


class MLAAttentionSpec(FullAttentionSpec):
    pass


class SlidingWindowSpec(AttentionSpec):
    def __init__(self, block_size: int, page: int, window: int):
        super().__init__(block_size, page)
        self.sliding_window = window

    def max_admission_blocks_per_request(self, max_in_flight_tokens: int, max_model_len: int) -> int:  # I:616-637
        num_tokens = min(self.sliding_window - 1 + max_in_flight_tokens, max_model_len)
        return P.cdiv(num_tokens, self.block_size) + 1

    def max_memory_usage_bytes(self, cfg) -> int:  # I:639-647
        return self.max_admission_blocks_per_request(cfg.max_in_flight_tokens, cfg.model_config.max_model_len) * self.page_size_bytes


class SlidingWindowMLASpec(SlidingWindowSpec):
    pass


class KpoolTailSpec(SlidingWindowSpec):  # I:739-786
    def max_admission_blocks_per_request(self, max_in_flight_tokens: int, max_model_len: int) -> int:
        return 1

    @property
    def participates_in_prefix_caching(self) -> bool:
        return False


class MambaSpec(KVCacheSpec):  # I:790-821
    def __init__(self, block_size: int, page: int, num_speculative_blocks: int, mamba_cache_mode: str = "align"):
        super().__init__(block_size, page)
        self.num_speculative_blocks = num_speculative_blocks
        self.mamba_cache_mode = mamba_cache_mode  # set from cache_config at spec creation (abstract.py:74)

    def max_memory_usage_bytes(self, cfg) -> int:
        mode = cfg.cache_config.mamba_cache_mode  # the fork reads cache_config here (I:813-818)
        if mode == "all":
            return (P.cdiv(cfg.model_config.max_model_len, self.block_size) + self.num_speculative_blocks) * self.page_size_bytes
        if mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        return self.page_size_bytes * (1 + self.num_speculative_blocks)


class ChunkedLocalAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, cfg) -> int:
        return 3 * self.page_size_bytes


class CrossAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, cfg) -> int:
        return 2 * self.page_size_bytes


class SinkFullAttentionSpec(FullAttentionSpec):  # I:866: its manager keeps permanent sink blocks
    pass


class TQFullAttentionSpec(FullAttentionSpec):  # I:392
    pass


class RSWASpec(FullAttentionSpec):  # I:508
    pass


class UniformTypeKVCacheSpecs(KVCacheSpec):  # I:920-948
    def __init__(self, kv_cache_specs: dict):
        self.kv_cache_specs = kv_cache_specs
        self.block_size = next(iter(kv_cache_specs.values())).block_size

    @property
    def participates_in_prefix_caching(self) -> bool:
        return all(getattr(s, "participates_in_prefix_caching", True) for s in self.kv_cache_specs.values())

    @property
    def page_size_bytes(self) -> int:
        return sum(s.page_size_bytes for s in self.kv_cache_specs.values())

    def max_memory_usage_bytes(self, cfg) -> int:
        pages = max(P.cdiv(s.max_memory_usage_bytes(cfg), s.page_size_bytes) for s in self.kv_cache_specs.values())
        return pages * self.page_size_bytes


def group(spec, layers: int, eagle: bool = False):
    return _Ns(layer_names=[f"layer.{i}" for i in range(layers)], kv_cache_spec=spec, is_eagle_group=eagle)


# Live deployment (head Spark): MAX_MODEL_LEN 1e6, MNBT 2048, async scheduling
# (max_concurrent_batches 2 -> max_in_flight_tokens 4096), k=7 DFlash2 tokens,
# mamba block 3584 in align mode, drafter SWA window 2048 block 64, kpool 4.
LIVE_MAX_MODEL_LEN = 1_000_000
LIVE_IN_FLIGHT = 2 * 2048
LIVE_SPEC_TOKENS = 7
MLA_PAGE = 2_351_104 * 11  # 3584 x 656 B x 11 layers (fp8_ds_mla), uniform group


def live_cfg(max_model_len=LIVE_MAX_MODEL_LEN, mamba_mode="align", use_eagle=True, in_flight=LIVE_IN_FLIGHT, dcp=1, pcp=1):
    spec_cfg = _Ns(use_eagle=(lambda: use_eagle)) if use_eagle is not None else None
    return _Ns(
        model_config=_Ns(max_model_len=max_model_len),
        cache_config=_Ns(mamba_cache_mode=mamba_mode),
        parallel_config=_Ns(decode_context_parallel_size=dcp, prefill_context_parallel_size=pcp),
        speculative_config=spec_cfg,
        max_in_flight_tokens=in_flight,
    )


def live_groups(mamba_mode="align"):
    return (
        [group(MLAAttentionSpec(3584, MLA_PAGE), 11), group(KpoolTailSpec(4, 1024, 4), 11)]
        + [group(MambaSpec(3584, 4096, LIVE_SPEC_TOKENS, mamba_mode), 9) for _ in range(4)]
        + [group(SlidingWindowSpec(64, 8192, 2048), 1)]
    )


def kv_config(num_blocks: int, groups=None, mamba_mode="align"):
    return _Ns(num_blocks=num_blocks, kv_cache_groups=live_groups(mamba_mode) if groups is None else groups)


def stock_line(num_blocks: int, bpr: int, max_model_len: int) -> tuple[str, str]:
    """What update_kv_cache_capacity prints: int(mc * L) and %.2fx."""
    mc = num_blocks / bpr
    return f"{int(mc * max_model_len):,}", f"{mc:.2f}x"


def drive(cfg, kc, max_model_len=None, env=None):
    """Run the shipped _glm53_log_kv_capacity with a recording logger."""
    log = P._RecordingLogger()
    ns = P.load_helpers(log)
    saved = os.environ.get(P.ENV_NAME)
    try:
        if env is None:
            os.environ.pop(P.ENV_NAME, None)
        else:
            os.environ[P.ENV_NAME] = env
        ns["_glm53_log_kv_capacity"](cfg, kc, cfg.model_config.max_model_len if max_model_len is None else max_model_len)
    finally:
        if saved is None:
            os.environ.pop(P.ENV_NAME, None)
        else:
            os.environ[P.ENV_NAME] = saved
    return log


# --------------------------------------------------------------------------
# Part A -- derivation
# --------------------------------------------------------------------------
def part_a() -> None:
    print("Part A: derivation (shipped helper text, exec'd)")
    check(P.HELPERS_SRC in P.PATCHED_HELPERS and P.PATCHED_HELPERS.endswith(P.ANCHOR_HELPERS), "A0 the exec'd helper text is the text written above the anchor, verbatim")

    cfg, kc = live_cfg(), kv_config(643)
    rows = P.kv_capacity_rows(cfg, kc)
    bpr = [r["blocks_per_request"] for r in rows]
    check(bpr == [280, 1, 9, 9, 9, 9, 97], f"A1 blocks/request per group = {bpr} (MLA cdiv(1e6,3584); kpool 1; mamba align 2+7; drafter cdiv(min(2047+4096,1e6),64)+1)")
    check(sum(bpr) == 414, "A1 sum = 414 = the stock denominator (DESIGN-drafter-retention §3)")
    tokens, ratio = stock_line(643, 414, LIVE_MAX_MODEL_LEN)
    check(tokens == "1,553,140" and ratio == "1.55x", f"A1 stock line reproduced from 643 ids: {tokens} tokens, {ratio} (runbook 08-31)")
    check([r["spec"] for r in rows] == ["MLAAttentionSpec", "KpoolTailSpec"] + ["MambaSpec"] * 4 + ["SlidingWindowSpec"], "A1 spec names by group")
    check([r["prefix_cacheable"] for r in rows] == [True, False, True, True, True, True, True], "A1 only the kpool tail opts out of prefix caching")
    check([r["eagle"] for r in rows] == [False] * 6 + [True], "A1 EAGLE flag: the exact-SlidingWindowSpec drafter group only (coordinator fallback mirrored)")
    check(rows[6]["sliding_window"] == 2048 and rows[2]["mamba_cache_mode"] == "align", "A1 window / mamba mode carried on the rows")

    s = P.kv_capacity_summary(rows, 643)
    check(s["usable"] == 642 and s["num_blocks"] == 643, "A2 usable block ids = 642 (643 minus the null block)")
    check(s["alignment"] == 3584, "A2 alignment = lcm(3584, 4, 64) = 3584 (the coordinator's scheduler block)")
    check(s["per_group"] == [1, 0, 1, 1, 1, 1, 33], f"A2 ids per 3584-token segment per group = {s['per_group']}")
    check(s["total"] == 38, "A2 ids per segment across groups = 38 (1 MLA + 4 mamba + 33 drafter)")
    check(s["segments"] == 16 and s["capacity_tokens"] == 57_344, "A2 capacity = 642 // 38 * 3584 = 57,344 tokens (partial final segment floored: 642 = 16*38 + 34)")
    check(s["unmodelled"] == [] and s["blocks_per_request"] == 414, "A2 nothing unmodelled; denominator carried")

    lines = P.kv_capacity_lines(rows, s, LIVE_MAX_MODEL_LEN)
    check(len(lines) == 8 and all(isinstance(x, str) and "%" not in x for x in lines), "A3 8 preformatted lines (7 groups + summary), hashable strings, no printf directives (info_once caches on its args)")
    check(lines[0] == "[glm53-kv-capacity-log] group 0: MLAAttentionSpec layers=11 block_size=3584 page_size=25,862,144 B blocks/request@1,000,000=280 prefix_caching=yes", "A3 group 0 line verbatim")
    check(lines[1].endswith("blocks/request@1,000,000=1 prefix_caching=no (scratch) window=4 eagle=no") and "KpoolTailSpec" in lines[1], "A3 kpool tail line: scratch, 1 block/request")
    check(lines[6] == "[glm53-kv-capacity-log] group 6: SlidingWindowSpec layers=1 block_size=64 page_size=8,192 B blocks/request@1,000,000=97 prefix_caching=yes window=2048 eagle=yes", "A3 drafter line verbatim")
    summary = lines[7]
    for frag in (
        "usable block ids: 642 (num_blocks=643 incl. the null block; 414 ids per 1,000,000-token request => 1.55x)",
        "ids per 3584-token cached segment across groups: 38 (per group: [1, 0, 1, 1, 1, 1, 33])",
        "cached-conversation capacity at this alignment ≈ 57,344 tokens = 16 segments",
        "dense-retention",
        "'GPU KV cache size' line above is max_concurrency x max_model_len, not this figure",
    ):
        check(frag in summary, f"A3 summary carries {frag[:60]!r}")

    log = drive(cfg, kc)
    check(log.info == lines and not log.warnings, "A4 _glm53_log_kv_capacity emits exactly those lines through logger.info_once, no warning")

    # The current rightsized boot (deploy/dogfood-next 2026-09-01 08:0x): 820 ids.
    rows820 = P.kv_capacity_rows(cfg, kv_config(820))
    s820 = P.kv_capacity_summary(rows820, 820)
    tokens, ratio = stock_line(820, 414, LIVE_MAX_MODEL_LEN)
    check(tokens == "1,980,676" and ratio == "1.98x", f"A5 820 ids -> stock line {tokens} / {ratio} (boot log 09-01 01:09:54)")
    check(s820["usable"] == 819 and s820["segments"] == 21 and s820["capacity_tokens"] == 75_264, "A5 820 ids -> 819 usable, 21 segments, 75,264 tokens")
    line = P.kv_capacity_lines(rows820, s820, LIVE_MAX_MODEL_LEN)[-1]
    check("usable block ids: 819 (num_blocks=820" in line and "≈ 75,264 tokens = 21 segments" in line, "A5 820-id summary line")

    # Single-group full-attention model: the stock figure and the summary agree
    # up to the null block, which is the whole point of the label upstream.
    one = [group(FullAttentionSpec(16, 4096), 32)]
    c1 = live_cfg(max_model_len=4096, use_eagle=None)
    r1 = P.kv_capacity_rows(c1, kv_config(1000, one))
    s1 = P.kv_capacity_summary(r1, 1000)
    check(r1[0]["blocks_per_request"] == 256 and s1["alignment"] == 16 and s1["per_group"] == [1] and s1["capacity_tokens"] == 999 * 16, "A6 single full-attention group: capacity = usable * block_size = 15,984")
    check(stock_line(1000, 256, 4096)[0] == "16,000" and 16_000 - s1["capacity_tokens"] == 16, "A6 ... one block (the null block) below the stock 16,000 figure")

    # UniformTypeKVCacheSpecs wrapper: name the inner class, page = sum, opt-out propagates.
    uni = UniformTypeKVCacheSpecs({"a": MLAAttentionSpec(3584, 100), "b": MLAAttentionSpec(3584, 200)})
    ru = P.kv_capacity_rows(cfg, kv_config(10, [group(uni, 2)]))
    check(ru[0]["spec"] == "MLAAttentionSpec" and ru[0]["page_size_bytes"] == 300 and ru[0]["blocks_per_request"] == 280 and ru[0]["prefix_cacheable"], "A7 uniform-type wrapper unwrapped for the name; page/blocks from the wrapper")
    uni_scratch = UniformTypeKVCacheSpecs({"a": KpoolTailSpec(4, 8, 4)})
    check(P.spec_prefix_cacheable(uni_scratch) is False and P.spec_prefix_cacheable(uni) is True, "A7 wrapper participation aggregates its members")

    # Participation flags: fork's first, upstream's second, default True.
    class _Up(FullAttentionSpec):
        @property
        def prefix_cacheable(self):
            return False

    class _Both(FullAttentionSpec):
        @property
        def participates_in_prefix_caching(self):
            return True

        @property
        def prefix_cacheable(self):
            return False

    check(P.spec_prefix_cacheable(_Up(16, 1)) is False, "A8 upstream `prefix_cacheable` False is honoured when the fork flag is absent")
    check(P.spec_prefix_cacheable(_Both(16, 1)) is True, "A8 the fork flag wins when both exist")
    check(P.spec_prefix_cacheable(FullAttentionSpec(16, 1)) is True, "A8 neither flag -> participates (legacy specs cache)")
    ru2 = P.kv_capacity_rows(cfg, kv_config(10, [group(_Up(16, 1), 1)]))
    check(P.kv_capacity_summary(ru2, 10)["per_group"] == [0], "A8 an opted-out group costs 0 ids per segment")

    # EAGLE detection.
    ids = P.eagle_group_ids(live_cfg(use_eagle=None), kv_config(10))
    check(ids == set(), "A9 no speculative config -> no EAGLE group")
    ids = P.eagle_group_ids(live_cfg(use_eagle=False), kv_config(10))
    check(ids == set(), "A9 use_eagle() False -> no EAGLE group")
    gs = live_groups()
    gs[0].is_eagle_group = True
    check(P.eagle_group_ids(cfg, kv_config(10, gs)) == {0}, "A9 is_eagle_group is authoritative when any group carries it (fallback not consulted)")
    gs = [group(MLAAttentionSpec(3584, 1), 1), group(SlidingWindowMLASpec(64, 1, 2048), 1)]
    check(P.eagle_group_ids(cfg, kv_config(10, gs)) == {0, 1}, "A9 use_eagle() with no exact SlidingWindowSpec group -> every group (upstream fallback; SlidingWindowMLASpec subclass is NOT the drafter predicate, same as the coordinator overlay)")
    rn = P.kv_capacity_rows(live_cfg(use_eagle=None), kv_config(10))
    check(P.kv_capacity_summary(rn, 643)["per_group"][6] == 32, "A9 drafter without EAGLE: cdiv(2047, 64) = 32 ids per segment")
    wide = [group(MLAAttentionSpec(3584, 1), 1), group(SlidingWindowSpec(64, 1, 4096), 1)]
    sw = P.kv_capacity_summary(P.kv_capacity_rows(cfg, kv_config(10, wide)), 10)
    check(sw["per_group"] == [1, 56], "A9 window wider than the segment: need 65 >= 56 -> every block of the segment (reachable_block_mask returns None)")

    # Unmodelled kinds withhold the capacity instead of guessing.
    rr = P.kv_capacity_rows(live_cfg(mamba_mode="none"), kv_config(643, mamba_mode="none"))
    ss = P.kv_capacity_summary(rr, 643)
    check(ss["unmodelled"] == [2, 3, 4, 5] and ss["capacity_tokens"] is None and ss["segments"] is None, "A10 mamba_cache_mode=none: mamba groups unmodelled, capacity withheld")
    ll = P.kv_capacity_lines(rr, ss, LIVE_MAX_MODEL_LEN)[-1]
    check("not derived (unmodelled: group 2 MambaSpec" in ll and "cached-conversation capacity at this alignment: not derived" in ll and "≈" not in ll, "A10 ... and the summary line says so")
    rr = P.kv_capacity_rows(live_cfg(mamba_mode="align"), kv_config(643, mamba_mode="none"))
    ss = P.kv_capacity_summary(rr, 643)
    check(rr[2]["mamba_cache_mode"] == "none" and ss["unmodelled"] == [2, 3, 4, 5], "A10 the mode is read from the MambaSpec the manager acts on, not cache_config (spec none / config align -> unmodelled)")
    mixed_mamba = UniformTypeKVCacheSpecs({"a": MambaSpec(3584, 8, 7, "align"), "b": MambaSpec(3584, 8, 7, "none")})
    rr = P.kv_capacity_rows(cfg, kv_config(10, [group(mixed_mamba, 2)]))
    check(rr[0]["mamba_cache_mode"] == "mixed" and P.kv_capacity_summary(rr, 10)["per_group"] == [None], "A10 a uniform mamba group with disagreeing member modes is 'mixed' and unmodelled")
    rr = P.kv_capacity_rows(live_cfg(mamba_mode="all"), kv_config(643, mamba_mode="all"))
    ss = P.kv_capacity_summary(rr, 643)
    check([r["blocks_per_request"] for r in rr][2] == 287 and ss["per_group"][2] == 1 and ss["total"] == 38, "A10 mamba all-mode: 280+7 blocks/request, still one state per block position")
    for cls in (SinkFullAttentionSpec, TQFullAttentionSpec, RSWASpec):
        rr = P.kv_capacity_rows(c1, kv_config(100, [group(cls(16, 1), 1)]))
        check(P.kv_capacity_summary(rr, 100)["per_group"] == [None], f"A10 {cls.__name__} (a FullAttentionSpec subclass with its own manager rule) is unmodelled, never costed as dense")
    rr = P.kv_capacity_rows(live_cfg(use_eagle=None), kv_config(10, [group(MLAAttentionSpec(3584, 1), 1), group(SlidingWindowMLASpec(64, 1, 2048), 1)]))
    check(P.kv_capacity_summary(rr, 10)["per_group"] == [1, 32], "A10 SlidingWindowMLASpec is costed by the window rule (exact-name allowlist), no EAGLE without a speculative config")
    for label, kw in (("dcp=2", {"dcp": 2}), ("pcp=2", {"pcp": 2})):
        rr = P.kv_capacity_rows(live_cfg(**kw), kv_config(643))
        ss = P.kv_capacity_summary(rr, 643, live_cfg(**kw))
        ll = P.kv_capacity_lines(rr, ss, LIVE_MAX_MODEL_LEN)[-1]
        check(ss["capacity_tokens"] is None and ss["withheld"] and "context parallelism" in ll and "not derived" in ll, f"A10 {label}: the figure is withheld (block sizes are rescaled per rank; alignment not derived here)")
    ss = P.kv_capacity_summary(rows, 643, cfg)
    check(ss["withheld"] is None and ss["capacity_tokens"] == 57_344, "A10 dcp=pcp=1 (this kit): nothing withheld")
    mixed = [group(FullAttentionSpec(16, 1), 1), group(ChunkedLocalAttentionSpec(16, 1), 1), group(CrossAttentionSpec(16, 1), 1)]
    sm = P.kv_capacity_summary(P.kv_capacity_rows(c1, kv_config(100, mixed)), 100)
    check(sm["per_group"] == [1, None, None] and sm["unmodelled"] == [1, 2] and sm["capacity_tokens"] is None, "A10 chunked-local / cross attention are unmodelled")
    check(P.ids_per_segment({"prefix_cacheable": True, "block_size": 16, "spec": "WeirdSpec", "kinds": ("WeirdSpec", "KVCacheSpec", "object"), "eagle": False, "sliding_window": None, "mamba_cache_mode": None}, 16) is None, "A10 an unknown spec kind is unmodelled (never silently costed)")
    check(P.ids_per_segment({"prefix_cacheable": True, "block_size": 64, "spec": "SlidingWindowSpec", "kinds": ("SlidingWindowSpec", "AttentionSpec", "KVCacheSpec", "object"), "eagle": True, "sliding_window": None, "mamba_cache_mode": None}, 3584) is None, "A10 a sliding-window spec without a window is unmodelled")

    # Null block / tiny pools / max_model_len below the alignment.
    for nb, usable, cap in ((1, 0, 0), (0, 0, 0), (38, 37, 0), (39, 38, 3584)):
        st = P.kv_capacity_summary(rows, nb)
        check(st["usable"] == usable and st["capacity_tokens"] == cap, f"A11 num_blocks={nb}: usable={usable}, capacity={cap} (never negative)")
    small = live_cfg(max_model_len=1000)
    rs = P.kv_capacity_rows(small, kv_config(643))
    ss = P.kv_capacity_summary(rs, 643)
    check([r["blocks_per_request"] for r in rs][0] == 1 and ss["capacity_tokens"] == 57_344, "A11 max_model_len 1000 < alignment: 1 MLA block/request; the segment arithmetic is independent of max_model_len")
    s_empty = P.kv_capacity_summary([], 643)
    l_empty = P.kv_capacity_lines([], s_empty, 10)
    check(s_empty["capacity_tokens"] is None and len(l_empty) == 1 and "not derived (no prefix-cacheable group costs ids" in l_empty[0] and "=> n/a" in l_empty[0], "A11 empty group list: one summary line, capacity not derived, no division by zero")
    s_scratch = P.kv_capacity_summary(P.kv_capacity_rows(cfg, kv_config(10, [group(KpoolTailSpec(4, 8, 4), 1)])), 10)
    check(s_scratch["total"] == 0 and s_scratch["capacity_tokens"] is None and "not derived (no prefix-cacheable group costs ids" in P.kv_capacity_lines([], s_scratch, 10)[-1], "A11 only opted-out groups: capacity not derived rather than infinite")

    # Knob matrix (same contract as the launcher's _glm53_validate_bool_flag).
    def knob(value):
        saved = os.environ.get(P.ENV_NAME)
        try:
            if value is None:
                os.environ.pop(P.ENV_NAME, None)
            else:
                os.environ[P.ENV_NAME] = value
            return P.kv_capacity_log_enabled()
        finally:
            if saved is None:
                os.environ.pop(P.ENV_NAME, None)
            else:
                os.environ[P.ENV_NAME] = saved

    check(knob(None) is True and knob("1") is True and knob("0") is False, "A12 unset -> on, 1 -> on, 0 -> off")
    for bad in ("", " ", "01", "2", "yes", "true", "1 ", " 1", "1\r", "0x1", "-1", "on"):
        try:
            knob(bad)
            check(False, f"A12 {bad!r} raises")
        except ValueError as exc:
            check(P.ENV_NAME in str(exc), f"A12 {bad!r} raises ValueError naming the knob")
    log = drive(cfg, kc, env="0")
    check(log.info == ["[glm53-kv-capacity-log] disabled (GLM53_KV_CAPACITY_LOG=0)"] and not log.warnings, "A12 disabled: one line saying so, nothing derived")
    try:
        drive(cfg, kc, env="yes")
        check(False, "A12 malformed knob raises out of the log call (not swallowed by the derivation guard)")
    except ValueError:
        check(True, "A12 malformed knob raises out of the log call (not swallowed by the derivation guard)")

    # Derivation failure -> warning, boot proceeds, nothing else logged.
    class _Broken(FullAttentionSpec):
        def max_memory_usage_bytes(self, cfg):
            raise RuntimeError("synthetic spec failure")

    log = drive(cfg, kv_config(10, [group(_Broken(16, 1), 1)]))
    check(log.info == [] and len(log.warnings) == 1 and "RuntimeError: synthetic spec failure" in log.warnings[0] and "serving unaffected" in log.warnings[0], "A13 derivation exception -> one warning naming it, no info lines, no raise")

    # No config mutation.
    cfg2, kc2 = live_cfg(), kv_config(643)
    before = (copy.deepcopy(cfg2.__dict__["model_config"].__dict__), copy.deepcopy(cfg2.cache_config.__dict__), kc2.num_blocks, [(g.kv_cache_spec.block_size, g.is_eagle_group, list(g.layer_names)) for g in kc2.kv_cache_groups])
    drive(cfg2, kc2)
    after = (cfg2.model_config.__dict__, cfg2.cache_config.__dict__, kc2.num_blocks, [(g.kv_cache_spec.block_size, g.is_eagle_group, list(g.layer_names)) for g in kc2.kv_cache_groups])
    check(before == after, "A14 the log call mutates neither vllm_config nor kv_cache_config")


# --------------------------------------------------------------------------
# Part B -- patch mechanics on a fixture of the live file
# --------------------------------------------------------------------------
# Verbatim excerpts of glm53-exl3-head's kv_cache_utils.py (vLLM
# 0.1.dev20051+g487ecf187, read read-only 2026-09-01), trimmed to the lines the
# overlay depends on, in file order, in a module that compiles. The in-image
# docstring of get_max_concurrency_for_kv_cache_config differs from upstream
# 22df3a3; the signature line (the anchor) and the log block are identical in
# both.
FIXTURE = (
    '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-Cache Utilities."""

import copy
import hashlib
import math
import os
from collections import defaultdict

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
)


logger = init_logger(__name__)

# The hash seed for the first block of any prefix block sequence.
NONE_HASH = 0


def is_kv_cache_type_uniform(kv_cache_spec: dict) -> bool:
    try:
        kv_cache_spec_values = list(kv_cache_spec.values())
        _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
    except AssertionError:
        return False
    return True


'''
    + P.ANCHOR_HELPERS
    + '''    """
    Get the maximum concurrency for the given KV cache configuration.

    Block ids are globally unique across groups (one BlockPool), so a
    max-length request costs the sum over groups of that group's block
    demand — layout-independent, each group divides by its own page size.
    """
    num_blocks_per_request = sum(
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in kv_cache_config.kv_cache_groups
    )
    max_concurrency = kv_cache_config.num_blocks / num_blocks_per_request
    return max_concurrency


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """
    Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    The override is logged once, at the call site in `get_kv_cache_configs`.
    """
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
    return num_blocks


def get_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> tuple[int, float]:
    """
    Get the group-aware KV cache token capacity and max concurrency.
    """
    max_model_len = vllm_config.model_config.max_model_len
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    return int(max_concurrency * max_model_len), max_concurrency


def update_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """Store and log the resolved KV cache capacity."""
    num_tokens, max_concurrency = get_kv_cache_capacity(vllm_config, kv_cache_config)
    vllm_config.cache_config.kv_cache_size_tokens = num_tokens
    vllm_config.cache_config.kv_cache_max_concurrency = max_concurrency
    max_model_len = vllm_config.model_config.max_model_len
'''
    + P.ANCHOR_CALL
    + '''

def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    return 0
'''
)


def _stub_vllm(logger: P._RecordingLogger) -> dict:
    """sys.modules stubs so the (patched) fixture can be exec'd end to end."""
    mods = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        mods[name] = m
        return m

    mod("vllm", envs=types.SimpleNamespace())
    mod("vllm.config", VllmConfig=object)
    mod("vllm.logger", init_logger=lambda name: logger)
    mod("vllm.utils")
    mod("vllm.utils.math_utils", cdiv=P.cdiv, round_up=lambda x, y: -(-x // y) * y)
    mod("vllm.v1")
    mod("vllm.v1.kv_cache_interface", KVCacheConfig=object, KVCacheGroupSpec=object)
    return mods


def exec_module(text: str, logger: P._RecordingLogger) -> dict:
    saved = {k: sys.modules.get(k) for k in _stub_vllm(logger)}
    try:
        sys.modules.update(_stub_vllm(logger))
        ns: dict = {"__name__": "kv_cache_utils_fixture"}
        exec(compile(text, "kv_cache_utils_fixture.py", "exec"), ns)
        return ns
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def run_overlay(target: Path, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GLM53_KV_CACHE_UTILS_PY": str(target)}
    env.pop(P.ENV_NAME, None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, str(PATCH), *args], text=True, capture_output=True, env=env)


def part_b() -> None:
    print("Part B: patch mechanics (fixture of the live kv_cache_utils.py)")
    compile(FIXTURE, "fixture", "exec")
    check(FIXTURE.count(P.ANCHOR_HELPERS) == 1 and FIXTURE.count(P.ANCHOR_CALL) == 1, "B0 fixture carries both pinned anchors exactly once")
    check(not P.verified_state(FIXTURE) and all(m not in FIXTURE for _n, m, _a, _p in P.SITES), "B0 pristine fixture is not in the verified state and carries no mark")

    patched, action = P.prepare(FIXTURE)
    compile(patched, "patched", "exec")
    check(action == "patched" and P.verified_state(patched), "B1 prepare -> patched, verified state")
    check(patched.count(P.ANCHOR_CALL) == 1 and patched.count(P.ANCHOR_HELPERS) == 1, "B1 the stock 'GPU KV cache size' info_once and the function signature survive byte-identical")
    check(P.ANCHOR_CALL + P.MARK_CALL in patched, "B1 the overlay call is the very next line after the stock line (same function, same indentation)")
    check(patched.index(P.MARK_HELPERS) < patched.index(P.ANCHOR_HELPERS) < patched.index(P.MARK_CALL), "B1 helpers precede the function they mirror, the call comes last")
    check(P.HELPERS_SRC in patched, "B1 the injected helper text is HELPERS_SRC verbatim")
    stock_before = FIXTURE[FIXTURE.index("def update_kv_cache_capacity") : FIXTURE.index(P.ANCHOR_CALL) + len(P.ANCHOR_CALL)]
    stock_after = patched[patched.index("def update_kv_cache_capacity") : patched.index(P.ANCHOR_CALL) + len(P.ANCHOR_CALL)]
    check(stock_before == stock_after, "B1 update_kv_cache_capacity up to and including the stock line is unchanged")

    # End to end through the patched text: the stock line first, then ours.
    log = P._RecordingLogger()
    ns = exec_module(patched, log)
    cfg, kc = live_cfg(), kv_config(643)
    cfg.cache_config.num_gpu_blocks_override = None
    saved = os.environ.pop(P.ENV_NAME, None)
    try:
        ns["update_kv_cache_capacity"](cfg, kc)
    finally:
        if saved is not None:
            os.environ[P.ENV_NAME] = saved
    check(log.info[0] == "GPU KV cache size: 1,553,140 tokens, Maximum concurrency for 1,000,000 tokens per request: 1.55x", "B2 patched update_kv_cache_capacity still logs the stock line first, verbatim")
    check(len(log.info) == 9 and log.info[1].startswith("[glm53-kv-capacity-log] group 0: MLAAttentionSpec") and "usable block ids: 642" in log.info[8] and "38 (per group: [1, 0, 1, 1, 1, 1, 33])" in log.info[8] and not log.warnings, "B2 ... followed by the 7 group lines and the summary (642 / 38 / 57,344)")
    check(cfg.cache_config.kv_cache_size_tokens == 1_553_140 and abs(cfg.cache_config.kv_cache_max_concurrency - 643 / 414) < 1e-12, "B2 kv_cache_size_tokens / kv_cache_max_concurrency stored exactly as stock")
    log0 = P._RecordingLogger()
    ns0 = exec_module(FIXTURE, log0)
    cfg0 = live_cfg()
    cfg0.cache_config.num_gpu_blocks_override = None
    ns0["update_kv_cache_capacity"](cfg0, kv_config(643))
    check(log0.info == log.info[:1], "B2 pristine fixture logs exactly the first of those lines (the overlay adds, never alters)")

    again, action2 = P.prepare(patched)
    check(again == patched and action2 == "already present", "B3 idempotent: prepare on a patched file is a byte-identical no-op")

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        target = tmp / "kv_cache_utils.py"
        target.write_text(FIXTURE)
        os.chmod(target, 0o644)
        pyc_dir = tmp / "__pycache__"
        pyc_dir.mkdir()
        stale = pyc_dir / "kv_cache_utils.cpython-312.pyc"
        stale.write_bytes(b"stale")

        r = run_overlay(target, "--preflight")
        check(r.returncode == 0 and "preflight OK (patched)" in r.stdout and target.read_text() == FIXTURE and stale.exists(), f"B4 --preflight on a pristine file: rc=0, reports 'patched', writes nothing (stdout={r.stdout.strip()!r})")
        r = run_overlay(target)
        check(r.returncode == 0 and "kv-capacity-log patched" in r.stdout and target.read_text() == patched, f"B4 apply: rc=0, file == prepare() output (stdout={r.stdout.strip()!r})")
        check(not stale.exists(), "B4 stale bytecode for the target was cleared")
        check(stat_mode(target) == 0o644 and not list(tmp.glob(".*.tmp")), "B4 mode preserved, no temp litter")
        r = run_overlay(target)
        check(r.returncode == 0 and "already present" in r.stdout and target.read_text() == patched, "B4 second apply: 'already present', byte-identical")
        r = run_overlay(target, "--preflight", env_extra={P.ENV_NAME: ""})
        check(r.returncode == 0 and "already present" in r.stdout, "B4 --preflight does not consult the knob (an empty knob is the runtime's error, not the installer's)")
        r = run_overlay(target, env_extra={P.ENV_NAME: ""})
        check(r.returncode == 0 and "GLM53_KV_CAPACITY_LOG=''" in r.stdout, "B4 apply reports the knob as set (empty shown as '', not normalised to 1)")

        # Drift: each anchor altered by one character, one at a time -> refused, nothing written.
        for name, _mark, anchor, _patched in P.SITES:
            drifted = FIXTURE.replace(anchor, anchor.replace("_", "-", 1), 1)  # one character
            assert drifted != FIXTURE
            target.write_text(drifted)
            r = run_overlay(target)
            check(r.returncode != 0 and "drifted" in r.stderr and name in r.stderr and target.read_text() == drifted and not list(tmp.glob(".*.tmp")), f"B5 anchor '{name}' drifted -> non-zero exit naming it, file untouched")
            try:
                P.prepare(drifted)
                check(False, f"B5 prepare() raises on drifted '{name}'")
            except ValueError as exc:
                check(name in str(exc), f"B5 prepare() raises on drifted '{name}'")

        # Partial / altered markers refused.
        half = FIXTURE.replace(P.ANCHOR_HELPERS, P.PATCHED_HELPERS, 1)
        target.write_text(half)
        r = run_overlay(target)
        check(r.returncode != 0 and "partial" in r.stderr and target.read_text() == half, "B6 only the helpers present (marks=1 of 2) -> refused, untouched")
        altered = patched.replace("_glm53_log_kv_capacity(vllm_config, kv_cache_config, max_model_len)  # [glm53-kv-capacity-log]", "_glm53_log_kv_capacity(vllm_config, kv_cache_config, 0)  # [glm53-kv-capacity-log]", 1)
        target.write_text(altered)
        r = run_overlay(target)
        check(r.returncode != 0 and "partial" in r.stderr and target.read_text() == altered, "B6 marked but altered call -> refused, untouched")
        dup = patched + "\n" + P.MARK_CALL
        target.write_text(dup)
        r = run_overlay(target)
        check(r.returncode != 0 and target.read_text() == dup, "B6 duplicated mark -> refused, untouched")

        # A file that lacks a binding the helpers need is refused before any write.
        nobind = FIXTURE.replace("import math\n", "", 1)
        target.write_text(nobind)
        r = run_overlay(target)
        check(r.returncode != 0 and "does not bind 'import math'" in r.stderr and target.read_text() == nobind, "B7 missing `import math` above the insert point -> refused (helpers would NameError)")
        nolog = FIXTURE.replace("logger = init_logger(__name__)\n", "", 1)
        target.write_text(nolog)
        r = run_overlay(target)
        check(r.returncode != 0 and "logger = init_logger" in r.stderr and target.read_text() == nolog, "B7 missing module logger -> refused")

        r = run_overlay(tmp / "absent.py")
        check(r.returncode != 0 and "missing" in r.stderr, "B7 missing target -> non-zero exit")

    # The real installed file, when present (mandatory in the image).
    installed = P.TARGET
    if installed.is_file():
        text = installed.read_text()
        try:
            out, act = P.prepare(text)
            compile(out, str(installed), "exec")
            check(act in ("patched", "already present") and P.verified_state(out), f"B8 installed {installed}: preflight OK ({act})")
            check(text.count(P.ANCHOR_CALL) == 1, "B8 installed file carries the stock 'GPU KV cache size' block exactly once")
        except ValueError as exc:
            check(False, f"B8 installed {installed}: preflight failed: {exc}")
    elif os.environ.get("GLM53_REQUIRE_TARGET") == "1":
        check(False, f"B8 GLM53_REQUIRE_TARGET=1 but {installed} is missing")
    else:
        print(f"  skip B8 installed file not present ({installed})")


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


# --------------------------------------------------------------------------
# Part C -- launcher / image wiring
# --------------------------------------------------------------------------
def guard_source() -> str:
    text = START.read_text()
    begin = text.index("# GLM53 numeric config guard (begin)")
    end_marker = "# GLM53 numeric config guard (end)"
    return text[begin : text.index(end_marker, begin) + len(end_marker)]


def part_c() -> None:
    print("Part C: launcher / image wiring")
    if not START.is_file():
        check(False, "C0 start.sh missing")
        return
    src = START.read_text()
    guard = guard_source()
    check('_glm53_validate_bool_flag GLM53_KV_CAPACITY_LOG "${GLM53_KV_CAPACITY_LOG-1}"' in guard, "C1 the numeric guard validates GLM53_KV_CAPACITY_LOG with the 0/1 validator (unset -> 1)")
    check('_cli_kvcap_set="${GLM53_KV_CAPACITY_LOG+1}"' in src and '_cli_kvcap="${GLM53_KV_CAPACITY_LOG-}"' in src and '[ -n "${_cli_kvcap_set}" ] && GLM53_KV_CAPACITY_LOG="$_cli_kvcap"' in src, "C1 caller export wins over .env, set-ness aware (an explicitly empty export is captured, then rejected)")
    check(src.index('_cli_kvcap_set="${GLM53_KV_CAPACITY_LOG+1}"') < src.index('source "$SCRIPT_DIR/.env"') < src.index('[ -n "${_cli_kvcap_set}" ] && GLM53_KV_CAPACITY_LOG="$_cli_kvcap"'), "C1 capture before .env is sourced, replay after")
    check('GLM53_KV_CAPACITY_LOG="${GLM53_KV_CAPACITY_LOG-1}"' in src, "C1 default 1 applies only when UNSET")
    check('KVCAP_PATCH_HOST="${KVCAP_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_kv_capacity_log.py}"' in src, "C2 KVCAP_PATCH_HOST defaults to the shipped overlay")
    check('"$KVCAP_PATCH_HOST|[glm53-kv-capacity-log]|$main_guard"' in src and "validate_overlay_artifacts" in src, "C2 the overlay is in the fail-closed artifact guard (identity string + EOF sentinel)")
    check("start|restart) validate_numeric_config; validate_overlay_artifacts ;;" in src, "C2 the guard runs on start|restart before anything else in main()")
    check('[ -f "$KVCAP_PATCH_HOST" ] || die "$KVCAP_PATCH_HOST missing"' in src and 'scp -q -o BatchMode=yes "$KVCAP_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_kv_capacity_log.py"' in src, "C2 preflight existence check + scp to the worker")
    check("-v '/tmp/patch_kv_capacity_log.py:/opt/glm53/patch_kv_capacity_log.py:ro'" in src and '-v "$KVCAP_PATCH_HOST:/opt/glm53/patch_kv_capacity_log.py:ro"' in src, "C2 read-only mount on both ranks")
    check('-e "GLM53_KV_CAPACITY_LOG=$GLM53_KV_CAPACITY_LOG"' in src, "C2 the knob is forwarded in nccl_common (both ranks)")
    check('log "boot KV-capacity breakdown log: GLM53_KV_CAPACITY_LOG=${GLM53_KV_CAPACITY_LOG} (both ranks)"' in src, "C2 the launcher logs the effective value")
    order = src[src.index("GLM53_OVERLAY_ORDER=(") : src.index(")", src.index("GLM53_OVERLAY_ORDER=("))]
    names = order.split()[1:]
    check("patch_kv_capacity_log.py" in names and names.index("patch_kv_capacity_log.py") > names.index("patch_glm5_drafter_group.py") and names.index("patch_kv_capacity_log.py") > names.index("patch_hybrid_prefix_hit.py") and names.index("patch_kv_capacity_log.py") < names.index("patch_xgrammar_termination.py"), f"C3 pinned order: drafter-group -> hybrid -> ... -> kv-capacity-log -> xgrammar ({names.index('patch_kv_capacity_log.py')})")
    check('emit_overlay_block >> "$HEAD_SCRIPT"' in src and 'emit_overlay_block >> "$WORKER_SCRIPT"' in src, "C3 both rank scripts take the order from emit_overlay_block")
    overlay_text = PATCH.read_text()
    last = [ln for ln in overlay_text.splitlines() if ln.strip()][-1]
    check(last == "    sys.exit(main())" and "[glm53-kv-capacity-log]" in overlay_text, "C3 overlay ends with the EOF sentinel and carries its identity string (artifact guard contract)")

    def run(value):
        script = guard + "\nGPU_MEM_UTIL=0.87; MAX_MODEL_LEN=1000000; MAX_NUM_SEQS=4; MAX_NUM_BATCHED_TOKENS=1024\n" + "validate_numeric_config || exit $?\n" + 'printf "%s\\n" "${GLM53_KV_CAPACITY_LOG-unset}"\n'
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C"}
        if value is not None:
            env["GLM53_KV_CAPACITY_LOG"] = value
        r = subprocess.run(["bash", "-c", script], text=True, capture_output=True, env=env)
        return r.returncode, r.stdout.strip(), r.stderr.strip()

    for value, out in ((None, "unset"), ("0", "0"), ("1", "1")):
        rc, o, e = run(value)
        check(rc == 0 and o == out, f"C4 GLM53_KV_CAPACITY_LOG={value!r} accepted (rc={rc} out={o!r} {e[:60]})")
    for value in ("", " ", "01", "2", "yes", "true", "1 ", " 1", "1\r", "0x1", "-1"):
        rc, o, e = run(value)
        check(rc == 2 and "GLM53_KV_CAPACITY_LOG" in e, f"C4 GLM53_KV_CAPACITY_LOG={value!r} rejected rc=2 with a named error (rc={rc} {e[:60]!r})")

    if DOCKERFILE.is_file():
        d = DOCKERFILE.read_text()
        i_hyb = d.index("RUN python3 /opt/glm53/patch_hybrid_prefix_hit.py")
        i_drf = d.index("RUN python3 /opt/glm53/patch_glm5_drafter_group.py")
        i_tst = d.index("python3 /opt/glm53/test_kv_capacity_log.py")
        i_pat = d.index("RUN python3 /opt/glm53/patch_kv_capacity_log.py")
        check("COPY overlay/patch_kv_capacity_log.py /opt/glm53/patch_kv_capacity_log.py" in d and "COPY tests/test_kv_capacity_log.py /opt/glm53/test_kv_capacity_log.py" in d, "C5 Dockerfile copies overlay + test")
        check(i_drf < i_hyb < i_tst < i_pat and "GLM53_REQUIRE_TARGET=1" in d[i_tst - 200 : i_tst], "C5 Dockerfile: drafter-group -> hybrid -> this test (installed file mandatory) -> this patch")
    if ENV_EXAMPLE.is_file():
        check("GLM53_KV_CAPACITY_LOG=1" in ENV_EXAMPLE.read_text(), "C6 .env.example documents the knob")
    if README.is_file():
        rd = README.read_text()
        check("| `GLM53_KV_CAPACITY_LOG` | `1` |" in rd and "## What the KV cache boot line means" in rd and "`overlay/patch_kv_capacity_log.py`" in rd, "C6 README: knob row, section, overlay row")


def main() -> int:
    print(f"overlay: {PATCH}  python: {sys.executable}")
    part_a()
    part_b()
    part_c()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}/{CHECKS}): " + "; ".join(FAILURES))
        return 1
    print(f"kv-capacity-log overlay OK ({CHECKS} checks)")
    return 0


def test_kv_capacity_log() -> None:
    """pytest entry point."""
    FAILURES.clear()
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
