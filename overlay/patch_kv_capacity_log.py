#!/usr/bin/env python3
"""Honest KV-capacity boot log for the hybrid model (log-only overlay).

The problem
-----------
``update_kv_cache_capacity`` (``v1/core/kv_cache_utils.py``) logs, once per
boot::

    GPU KV cache size: 1,553,140 tokens, Maximum concurrency for 1,000,000 tokens per request: 1.55x

The first number is ``int(max_concurrency * max_model_len)`` where
``max_concurrency = num_blocks / num_blocks_per_request`` and
``num_blocks_per_request`` is a SUM OVER KV-CACHE GROUPS of
``cdiv(spec.max_memory_usage_bytes(cfg), spec.page_size_bytes)``
(``get_max_concurrency_for_kv_cache_config``). For a single-group model that
is the pool's token capacity up to rounding, which is what the label suggests.
For this hybrid model (MLA + kpool tail + 4 mamba + DFlash2 drafter SWA, one
shared BlockPool with globally unique block ids) it is a concurrency figure in
token units: neither ``num_blocks`` nor ``num_blocks_per_request`` is logged,
``max_concurrency`` is printed to two decimals, and only their ratio can be
recovered. On this kit the line read 1.5M tokens while the pool was 643 block
ids (642 usable) and one cached 3584-token segment cost 38 of them
(1 MLA + 4 mamba + 33 drafter) -- about 57K tokens of cached conversation with
nothing running. Two reviewers independently read the line as a 1.5M-token
prefix cache. Upstream feature request: vllm-project/vllm#54662.

What this overlay does
----------------------
Immediately after the existing line (kept byte-identical) it logs, from the
SAME config objects the existing line is computed from:

* one line per KV-cache group: index, spec type, block_size, page size, blocks
  per ``max_model_len`` request (the same ``cdiv`` expression as
  ``get_max_concurrency_for_kv_cache_config``, so the two cannot drift), and
  whether the group takes part in prefix caching;
* one summary line: usable block ids (``num_blocks - 1``: ``BlockPool``
  permanently holds back block id 0 as the null block), the ids one aligned
  cached segment costs across groups (per group), and the resulting
  cached-conversation capacity at that alignment.

Log-only. No control flow, no allocation, no config mutation, runs once in the
engine core at boot. Knob ``GLM53_KV_CAPACITY_LOG``: unset or ``1`` = log,
``0`` = one line saying it is disabled, anything else = ``ValueError`` (a
typo'd knob must not silently pick a mode; the launcher rejects it first).
Any failure inside the derivation itself is a ``warning_once`` naming the
exception, and boot proceeds: the feature is diagnostic and must never take
the server down.

What the summary models (and what it does not)
----------------------------------------------
"Ids per cached segment" is the DENSE-retention cost with block-aligned hits,
i.e. what the managers' ``reachable_block_mask`` hashes when no retention
interval is set and hits end on scheduler-block boundaries:

* exactly ``FullAttentionSpec`` / ``MLAAttentionSpec``: every block is hashed
  -> ``alignment / block_size`` ids (1 here);
* exactly ``MambaSpec`` in ``align`` / ``all`` cache mode (read from the spec
  the manager acts on): one state per block -> ``alignment / block_size`` ids
  per group (1 each here);
* exactly ``SlidingWindowSpec`` / ``SlidingWindowMLASpec``:
  ``min(need, alignment / block_size)`` with
  ``need = cdiv(window - 1, block_size) + (1 if EAGLE group)`` -- the
  contiguous run a hit needs (``SlidingWindowManager._contiguous_blocks_for_hit``;
  33 of 56 here);
* groups that opt out of prefix caching (``KpoolTailSpec``): 0 -- a live block
  per request, never a cached one.

Alignment is the lcm of every group's block size, which is what the
coordinator asserts its scheduler block size to be (3584 here) when no
context parallelism rescales block sizes; under DCP/PCP > 1 the figure is
withheld. Any other spec class -- subclasses included, because a subclass
may come with its own manager and reservation rule (``SinkFullAttentionSpec``
keeps permanent sink blocks; ``KpoolTailSpec`` is scratch) -- and mamba cache
mode ``none`` / a mixed uniform group are reported as UNMODELLED and the
capacity figure is withheld rather than guessed. Not modelled on purpose: the
per-group retention overlay (#83) makes the drafter cost boundary tails only,
and fine-grained hits (#84) change where a segment may end; both are decided
later, in the coordinator, and the summary says which policy it assumes.

EAGLE detection mirrors the coordinator exactly: ``group.is_eagle_group`` when
any group carries it, else -- when ``speculative_config.use_eagle()`` -- the
groups whose unwrapped spec is an exact ``SlidingWindowSpec``
(``patch_hybrid_prefix_hit.py``'s ``_glm53_is_draft_swa_spec``), else every
group (the upstream fallback).

Conventions follow ``overlay/patch_indexer_workspace.py`` /
``patch_kpool_tail_slotmap.py``: pinned ANCHOR, MARK sentinel,
``verified_state``, ``prepare``, idempotent, atomic replace, pyc clear, drift
=> nonzero exit. Both sites are preflighted before either is written. Must be
applied AFTER ``patch_glm5_drafter_group.py`` (same file); it has no anchors
in common with any other overlay.

Usage::

    python3 patch_kv_capacity_log.py              # apply
    python3 patch_kv_capacity_log.py --preflight  # validate anchors only
"""
from __future__ import annotations

import math
import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_KV_CACHE_UTILS_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py",
    )
)

ENV_NAME = "GLM53_KV_CAPACITY_LOG"
TAG = "[glm53-kv-capacity-log]"


# ---------------------------------------------------------------------------
# The injected helpers, as source.
#
# This string is BOTH what gets written into kv_cache_utils.py AND what the
# host tests exercise: it is exec'd below to produce module-level callables.
# One implementation of the derivation, so a test can never pass against a
# replica that has drifted from the shipped code. Inside kv_cache_utils.py the
# names it relies on (``cdiv``, ``math``, ``os``, ``logger``) are all bound
# above the insert point; ``prepare`` checks that.
# ---------------------------------------------------------------------------
HELPERS_SRC = '''# [glm53-kv-capacity-log] helpers -- log-only; see overlay/patch_kv_capacity_log.py
_GLM53_KV_CAPACITY_ENV = "GLM53_KV_CAPACITY_LOG"
_GLM53_KV_CAPACITY_TAG = "[glm53-kv-capacity-log]"


def _glm53_kv_capacity_log_enabled() -> bool:
    """Exactly "0" or "1"; an UNSET var means "1" (log).

    Same contract as the launcher's ``_glm53_validate_bool_flag``: no strip,
    no lower, "" is a value and not an absence. A typo'd knob raises rather
    than silently choosing.
    """
    raw = os.environ.get(_GLM53_KV_CAPACITY_ENV)
    if raw is None or raw == "1":
        return True
    if raw == "0":
        return False
    raise ValueError(
        f"{_GLM53_KV_CAPACITY_ENV} must be exactly 0 or 1 (got: {raw!r})"
    )


def _glm53_inner_kv_spec(spec):
    """Unwrap a UniformTypeKVCacheSpecs so the group's real spec class is named."""
    specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(specs, dict) and specs:
        return next(iter(specs.values()))
    return spec


def _glm53_spec_kind_names(spec) -> tuple:
    return tuple(c.__name__ for c in type(spec).__mro__)


def _glm53_spec_prefix_cacheable(spec) -> bool:
    """Fork flag first, then upstream's, else True (legacy specs cache)."""
    for name in ("participates_in_prefix_caching", "prefix_cacheable"):
        value = getattr(spec, name, None)
        if callable(value):
            value = value()
        if isinstance(value, bool):
            return value
    return True


def _glm53_eagle_group_ids(vllm_config, kv_cache_config) -> set:
    """Mirror of KVCacheCoordinator.__init__ (with patch_hybrid_prefix_hit.py).

    ``is_eagle_group`` is authoritative when any group carries it. Otherwise,
    when the speculative config is EAGLE-like, the exact-``SlidingWindowSpec``
    groups are flagged (the DFlash2 drafter), else every group.
    """
    groups = list(kv_cache_config.kv_cache_groups)
    ids = {i for i, g in enumerate(groups) if bool(getattr(g, "is_eagle_group", False))}
    if ids:
        return ids
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    use_eagle = getattr(spec_cfg, "use_eagle", None) if spec_cfg is not None else None
    if callable(use_eagle):
        use_eagle = use_eagle()
    if not use_eagle:
        return set()
    swa = {
        i
        for i, g in enumerate(groups)
        if type(_glm53_inner_kv_spec(g.kv_cache_spec)).__name__ == "SlidingWindowSpec"
    }
    return swa or set(range(len(groups)))


def _glm53_kv_capacity_rows(vllm_config, kv_cache_config) -> list:
    """One dict per KV-cache group, from the same objects the stock line uses.

    ``blocks_per_request`` is deliberately the same expression as the
    comprehension in ``get_max_concurrency_for_kv_cache_config``: the sum of
    this column IS that function's denominator.
    """
    eagle = _glm53_eagle_group_ids(vllm_config, kv_cache_config)
    rows = []
    for index, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        inner = _glm53_inner_kv_spec(spec)
        kinds = _glm53_spec_kind_names(inner)
        row = {
            "index": index,
            "spec": kinds[0],
            "kinds": kinds,
            "layers": len(getattr(group, "layer_names", ()) or ()),
            "block_size": int(spec.block_size),
            "page_size_bytes": int(spec.page_size_bytes),
            "blocks_per_request": cdiv(
                spec.max_memory_usage_bytes(vllm_config),
                spec.page_size_bytes,
            ),
            "prefix_cacheable": _glm53_spec_prefix_cacheable(spec),
            "eagle": index in eagle,
            "sliding_window": None,
            "mamba_cache_mode": None,
        }
        if "SlidingWindowSpec" in kinds:
            window = getattr(inner, "sliding_window", None)
            row["sliding_window"] = int(window) if window is not None else None
        if "MambaSpec" in kinds:
            row["mamba_cache_mode"] = _glm53_mamba_cache_mode(spec)
        rows.append(row)
    return rows


def _glm53_mamba_cache_mode(spec):
    """The mode the managers act on: each resolved MambaSpec's own field
    (set from cache_config at spec creation, but the spec is what the manager
    reads). A uniform group whose members disagree is reported as "mixed" and
    left unmodelled.
    """
    members = getattr(spec, "kv_cache_specs", None)
    specs = list(members.values()) if isinstance(members, dict) and members else [spec]
    modes = {getattr(s, "mamba_cache_mode", None) for s in specs}
    if len(modes) != 1:
        return "mixed"
    return modes.pop()


def _glm53_ids_per_segment(row: dict, alignment: int):
    """Block ids one aligned cached segment costs in this group under DENSE
    retention with block-aligned hits (what ``reachable_block_mask`` hashes
    when no retention interval is set). ``None`` = not modelled: the summary
    then withholds the capacity figure instead of guessing.
    """
    if not row["prefix_cacheable"]:
        return 0
    block_size = row["block_size"]
    if block_size <= 0 or alignment % block_size:
        return None
    per_segment = alignment // block_size
    # EXACT class names only. A subclass may come with its own manager and its
    # own reservation rule (SinkFullAttentionSpec keeps permanent sink blocks,
    # KpoolTailSpec is a scratch buffer, TQ/RSWA/HiddenState carry other
    # semantics), so anything not listed is unmodelled, never costed by its
    # base class.
    kind = row["spec"]
    if kind == "MambaSpec":
        # align: one state snapshot per block position; all: every block.
        return per_segment if row["mamba_cache_mode"] in ("align", "all") else None
    if kind in ("SlidingWindowSpec", "SlidingWindowMLASpec"):
        window = row["sliding_window"]
        if not window:
            return None
        # SlidingWindowManager._contiguous_blocks_for_hit: the run a hit needs,
        # +1 when the EAGLE last-block drop applies. reachable_block_mask
        # caches every block once need >= per_segment.
        need = cdiv(window - 1, block_size) + (1 if row["eagle"] else 0)
        return min(need, per_segment)
    if kind in ("FullAttentionSpec", "MLAAttentionSpec"):
        # FullAttentionManager keeps the base reachable_block_mask (None):
        # every block is hashed.
        return per_segment
    return None


def _glm53_context_parallel_sizes(vllm_config) -> tuple:
    pc = getattr(vllm_config, "parallel_config", None)
    dcp = int(getattr(pc, "decode_context_parallel_size", 1) or 1)
    pcp = int(getattr(pc, "prefill_context_parallel_size", 1) or 1)
    return dcp, pcp


def _glm53_kv_capacity_summary(rows: list, num_blocks: int, vllm_config=None) -> dict:
    num_blocks = int(num_blocks)
    usable = max(num_blocks - 1, 0)  # block id 0 is BlockPool's null block
    blocks_per_request = sum(int(r["blocks_per_request"]) for r in rows)
    # lcm of the raw group block sizes == the coordinator's scheduler block
    # size when no context parallelism scales the per-rank block sizes.
    alignment = 1
    for r in rows:
        alignment = math.lcm(alignment, max(int(r["block_size"]), 1))
    per_group = [_glm53_ids_per_segment(r, alignment) for r in rows]
    unmodelled = [r["index"] for r, v in zip(rows, per_group) if v is None]
    withheld = None
    dcp, pcp = _glm53_context_parallel_sizes(vllm_config) if vllm_config is not None else (1, 1)
    if dcp > 1 or pcp > 1:
        withheld = f"context parallelism (dcp={dcp}, pcp={pcp}) rescales block sizes; alignment not derived here"
    total = sum(v for v in per_group if v)
    capacity = None
    segments = None
    if rows and not unmodelled and withheld is None and total > 0:
        segments = usable // total
        capacity = segments * alignment
    return {
        "num_blocks": num_blocks,
        "usable": usable,
        "blocks_per_request": blocks_per_request,
        "alignment": alignment,
        "per_group": per_group,
        "unmodelled": unmodelled,
        "withheld": withheld,
        "total": total,
        "segments": segments,
        "capacity_tokens": capacity,
    }


def _glm53_kv_capacity_lines(rows: list, summary: dict, max_model_len: int) -> list:
    """Preformatted lines (info_once caches on its arguments: pass strings)."""
    tag = _GLM53_KV_CAPACITY_TAG
    lines = []
    for r in rows:
        extra = ""
        if r["sliding_window"] is not None:
            extra += f" window={r['sliding_window']} eagle={'yes' if r['eagle'] else 'no'}"
        if r["mamba_cache_mode"] is not None:
            extra += f" mamba_cache_mode={r['mamba_cache_mode']}"
        cacheable = "yes" if r["prefix_cacheable"] else "no (scratch)"
        lines.append(
            f"{tag} group {r['index']}: {r['spec']} layers={r['layers']} "
            f"block_size={r['block_size']} page_size={r['page_size_bytes']:,} B "
            f"blocks/request@{int(max_model_len):,}={r['blocks_per_request']} "
            f"prefix_caching={cacheable}{extra}"
        )
    bpr = summary["blocks_per_request"]
    ratio = f"{summary['num_blocks'] / bpr:.2f}x" if bpr else "n/a"
    head = (
        f"{tag} usable block ids: {summary['usable']} "
        f"(num_blocks={summary['num_blocks']} incl. the null block; "
        f"{bpr} ids per {int(max_model_len):,}-token request => {ratio})"
    )
    if summary["capacity_tokens"] is None:
        if summary["unmodelled"]:
            why = "unmodelled: " + ", ".join(
                f"group {r['index']} {r['spec']}" for r in rows if r["index"] in summary["unmodelled"]
            )
        elif summary.get("withheld"):
            why = summary["withheld"]
        else:
            why = "no prefix-cacheable group costs ids"
        lines.append(
            f"{head}; ids per {summary['alignment']}-token cached segment across groups: "
            f"not derived ({why}; per group: {summary['per_group']}); "
            "cached-conversation capacity at this alignment: not derived"
        )
    else:
        lines.append(
            f"{head}; ids per {summary['alignment']}-token cached segment across groups: "
            f"{summary['total']} (per group: {summary['per_group']}); "
            f"cached-conversation capacity at this alignment ≈ {summary['capacity_tokens']:,} tokens "
            f"= {summary['segments']} segments (aligned dense-retention prefix-cache upper bound: "
            "nothing running, every reachable block hashed, block-aligned hits). "
            "The 'GPU KV cache size' line above is max_concurrency x max_model_len, not this figure."
        )
    return lines


def _glm53_log_kv_capacity(vllm_config, kv_cache_config, max_model_len) -> None:
    """Called right after the stock 'GPU KV cache size' line. Log-only."""
    # The knob is validated OUTSIDE the try: a malformed value must raise.
    if not _glm53_kv_capacity_log_enabled():
        logger.info_once(
            f"{_GLM53_KV_CAPACITY_TAG} disabled ({_GLM53_KV_CAPACITY_ENV}=0)"
        )
        return
    try:
        rows = _glm53_kv_capacity_rows(vllm_config, kv_cache_config)
        summary = _glm53_kv_capacity_summary(rows, kv_cache_config.num_blocks, vllm_config)
        lines = _glm53_kv_capacity_lines(rows, summary, max_model_len)
    except Exception as exc:  # diagnostic only: never take the server down
        logger.warning_once(
            f"{_GLM53_KV_CAPACITY_TAG} could not derive the block-level KV "
            f"capacity (log-only; serving unaffected): {type(exc).__name__}: {exc}"
        )
        return
    for line in lines:
        logger.info_once(line)


'''


class _RecordingLogger:
    """Host stand-in for vLLM's logger: keeps what would have been logged."""

    def __init__(self) -> None:
        self.info: list[str] = []
        self.warnings: list[str] = []

    def info_once(self, msg, *args, **kwargs) -> None:
        self.info.append(msg % args if args else msg)

    def warning_once(self, msg, *args, **kwargs) -> None:
        self.warnings.append(msg % args if args else msg)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def load_helpers(logger: _RecordingLogger | None = None) -> dict:
    """Exec HELPERS_SRC into a fresh namespace (what the tests drive)."""
    ns: dict = {"os": os, "math": math, "cdiv": cdiv, "logger": logger or _RecordingLogger()}
    exec(compile(HELPERS_SRC, "<glm53-kv-capacity-log helpers>", "exec"), ns)
    return ns


_HELPERS = load_helpers()
kv_capacity_log_enabled = _HELPERS["_glm53_kv_capacity_log_enabled"]
inner_kv_spec = _HELPERS["_glm53_inner_kv_spec"]
spec_prefix_cacheable = _HELPERS["_glm53_spec_prefix_cacheable"]
eagle_group_ids = _HELPERS["_glm53_eagle_group_ids"]
kv_capacity_rows = _HELPERS["_glm53_kv_capacity_rows"]
ids_per_segment = _HELPERS["_glm53_ids_per_segment"]
kv_capacity_summary = _HELPERS["_glm53_kv_capacity_summary"]
kv_capacity_lines = _HELPERS["_glm53_kv_capacity_lines"]


# ---------------------------------------------------------------------------
# Site 1 -- helpers, inserted above the function whose denominator they mirror
# ---------------------------------------------------------------------------
MARK_HELPERS = "# [glm53-kv-capacity-log] helpers -- log-only; see overlay/patch_kv_capacity_log.py\n"

ANCHOR_HELPERS = """def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
"""

PATCHED_HELPERS = HELPERS_SRC + ANCHOR_HELPERS


# ---------------------------------------------------------------------------
# Site 2 -- the call, appended after the stock line (which stays byte-identical)
# ---------------------------------------------------------------------------
MARK_CALL = "    _glm53_log_kv_capacity(vllm_config, kv_cache_config, max_model_len)  # [glm53-kv-capacity-log]\n"

ANCHOR_CALL = """    logger.info_once(
        "GPU KV cache size: %s tokens, "
        "Maximum concurrency for %s tokens per request: %.2fx",
        f"{num_tokens:,}",
        f"{max_model_len:,}",
        max_concurrency,
    )
"""

PATCHED_CALL = ANCHOR_CALL + MARK_CALL


SITES = (
    ("capacity helpers", MARK_HELPERS, ANCHOR_HELPERS, PATCHED_HELPERS),
    ("capacity log call", MARK_CALL, ANCHOR_CALL, PATCHED_CALL),
)

# Names the injected helpers rely on; each must be bound above the insert point.
REQUIRED_BINDINGS = (
    "from vllm.utils.math_utils import cdiv",
    "import math\n",
    "import os\n",
    "logger = init_logger(__name__)\n",
)


def verified_state(text: str) -> bool:
    """Exact post-state check.

    Both replacements keep their anchor (they are insertions), so expect exactly
    as many surviving anchors as the replacement itself contains, one mark per
    site, and the helper block above the function it mirrors.
    """
    ok = all(
        text.count(mark) == 1
        and text.count(patched) == 1
        and text.count(anchor) == patched.count(anchor)
        for _name, mark, anchor, patched in SITES
    )
    return ok and text.index(MARK_HELPERS) < text.index(ANCHOR_HELPERS) < text.index(MARK_CALL)


def prepare(source: str) -> tuple[str, str]:
    """Idempotent, fail-closed. Returns ``(text, action)``. Nothing is written here."""
    marks = sum(source.count(mark) for _n, mark, _a, _p in SITES)
    if marks:
        if marks != len(SITES) or not verified_state(source):
            raise ValueError(
                "partial/inconsistent kv-capacity-log patch "
                f"(marks={marks}, expected {len(SITES)}) -- refusing to touch "
                "a half-patched file"
            )
        return source, "already present"

    for binding in REQUIRED_BINDINGS:
        head = source[: source.find(ANCHOR_HELPERS)] if ANCHOR_HELPERS in source else source
        if binding not in head:
            raise ValueError(
                f"kv_cache_utils.py does not bind {binding.strip()!r} above the "
                "insert point; the injected helpers would NameError"
            )

    out = source
    for name, _mark, anchor, patched in SITES:
        n = out.count(anchor)
        if n != 1:
            raise ValueError(
                f"pinned kv-capacity-log anchor '{name}' drifted (found {n}, expected 1)"
            )
    for name, _mark, anchor, patched in SITES:
        out = out.replace(anchor, patched, 1)

    if not verified_state(out):
        raise ValueError("kv-capacity-log post-patch verification failed")
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kv-capacity-log.tmp")
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

    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"kv-capacity-log preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")

    if preflight_only:
        print(f"{TARGET.name}: kv-capacity-log preflight OK ({action})")
        return 0

    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    # Report the value as set, not as normalised: at image build time the knob
    # is usually unset, and printing "1" for an explicitly empty value would
    # hide the error the runtime will raise.
    mode = os.environ.get(ENV_NAME, "1 (unset)")
    print(f"{TARGET.name}: kv-capacity-log {action} ({ENV_NAME}={mode!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
