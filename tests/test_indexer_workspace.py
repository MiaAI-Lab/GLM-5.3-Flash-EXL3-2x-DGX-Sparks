#!/usr/bin/env python3
"""Tests for the opt-in sparse-indexer prefill workspace right-sizing.

Host-only, no GPU, no torch. Three groups:

* the sizing formula, against the edge cases raised in Codex's review of the
  research draft (MNBT-bounded chunk count, max_num_seqs, k=7 spec tokens, the
  true per-step maximum, and the stock clamp);
* a proof by exhaustion that the right-sized workspace produces the SAME chunk
  list as the stock ``max_model_len * 40`` one for every batch the scheduler can
  legally form -- the equivalence claim, checked rather than asserted;
* apply / idempotence / fail-closed drift against a fixture built from the live
  container's ``indexer.py`` (and, opt-in, against the live file itself).
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_indexer_workspace.py",
        ROOT / "overlay" / "patch_indexer_workspace.py",
    )
    if p.is_file()
)
sys.path.insert(0, str(PATCH.parent))
from patch_indexer_workspace import (  # noqa: E402
    ANCHOR_GUARD,
    ANCHOR_IMPORT,
    ANCHOR_SIZE,
    BYTES_PER_ENTRY,
    ENV_NAME,
    MARK_SIZE,
    STOCK_MULTIPLIER,
    indexer_compress_ratio,
    prepare,
    rightsized_workspace_entries,
    split_prefill_chunks,
    stock_workspace_entries,
    verified_state,
    workspace_mode,
)

INSTALLED = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"
)

# Live deployment (head Spark, glm53-exl3-head, build 0.1.dev20051+g487ecf187).
LIVE_MAX_MODEL_LEN = 1_000_000
LIVE_KPOOL = 4
LIVE_MAX_NUM_SEQS = 16
LIVE_MNBT = 2048
LIVE_SPEC = 7
RADIX_TOPK_WORKSPACE_BYTES = 1024 * 1024

# vLLM's own default for VLLM_SPARSE_INDEXER_MAX_LOGITS_MB.
MAX_LOGITS_BYTES = 512 * 1024 * 1024

# Fixture assembled from verbatim excerpts of the live container file, pulled
# read-only. All three pinned anchors, in file order, in a module that compiles.
PINNED_FIXTURE = (
    ANCHOR_IMPORT
    + '''
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import MLAAttentionSpec

logger = init_logger(__name__)


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    total_seq_lens: int


'''
    + ANCHOR_SIZE
    + '''

class DeepseekV32IndexerMetadataBuilder:
    def __init__(self, kv_cache_spec, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config
        self.device = device
        self.dcp_world_size = 1
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)

'''
    + ANCHOR_GUARD
)


# --------------------------------------------------------------------------
# config stubs (duck-typed exactly as the injected helpers traverse them)
# --------------------------------------------------------------------------
class _Ns:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def cfg(
    max_model_len=LIVE_MAX_MODEL_LEN,
    kpool=LIVE_KPOOL,
    max_num_seqs=LIVE_MAX_NUM_SEQS,
    mnbt=LIVE_MNBT,
    spec=LIVE_SPEC,
    with_kpool_attr=True,
):
    hf = _Ns(index_kpool=kpool) if with_kpool_attr else _Ns()
    return _Ns(
        model_config=_Ns(max_model_len=max_model_len, hf_text_config=hf),
        scheduler_config=_Ns(max_num_seqs=max_num_seqs, max_num_batched_tokens=mnbt),
        speculative_config=_Ns(num_speculative_tokens=spec) if spec is not None else None,
    )


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _set_mode(value):
    if value is None:
        os.environ.pop(ENV_NAME, None)
    else:
        os.environ[ENV_NAME] = value


# --------------------------------------------------------------------------
# 1. mode enum
# --------------------------------------------------------------------------
def test_mode_enum() -> None:
    """Literal match, exactly like start.sh's ``_glm53_validate_enum``.

    The launcher expands ``${GLM53_INDEXER_WORKSPACE-stock}`` -- default on
    UNSET only -- and then compares the value as-is against ``stock rightsize``.
    The python side must agree character for character, or a value the launcher
    rejects (or accepts) changes meaning across the container boundary.
    """
    saved = os.environ.get(ENV_NAME)
    try:
        # Only an absent var defaults.
        _set_mode(None)
        assert workspace_mode() == "stock"
        for good in ("stock", "rightsize"):
            _set_mode(good)
            assert workspace_mode() == good
        # A typo, a case variant, padding, or an explicitly empty value must
        # not select a serving mode -- all four are launcher errors too.
        for bad in (
            "", " ", "  ", "\t", "\n",
            "Stock", "STOCK", "RIGHTSIZE", "Rightsize",
            " rightsize ", "rightsize ", " stock", "stock\n",
            "1", "0", "on", "yes", "right-size", "rightsized", "true",
        ):
            _set_mode(bad)
            try:
                workspace_mode()
            except ValueError as exc:
                assert ENV_NAME in str(exc), bad
                assert repr(bad) in str(exc), bad
            else:
                raise AssertionError(f"{bad!r} must be rejected")
    finally:
        _set_mode(saved)


def test_mode_enum_matches_launcher_enum() -> None:
    """Cross-check the two implementations against one another, by value.

    Runs start.sh's own ``_glm53_validate_enum`` (lifted out of the guard
    block) over the same values and requires the accept/reject verdicts to
    match ``_glm53_workspace_mode`` exactly.
    """
    start = ROOT / "start.sh"
    if not start.is_file():
        return
    source = start.read_text()
    begin = source.index("# GLM53 numeric config guard (begin)")
    end = source.index("# GLM53 numeric config guard (end)")
    guard = source[begin:end]

    script = (
        guard
        + '\n_glm53_validate_enum GLM53_INDEXER_WORKSPACE '
        + '"${GLM53_INDEXER_WORKSPACE-stock}" stock rightsize\n'
    )
    saved = os.environ.get(ENV_NAME)
    try:
        for value in (
            None, "stock", "rightsize",
            "", " ", "Stock", "RIGHTSIZE", " rightsize ", "1", "on", "true",
        ):
            env = {k: v for k, v in os.environ.items() if k != ENV_NAME}
            if value is not None:
                env[ENV_NAME] = value
            shell_ok = subprocess.run(
                ["bash", "-c", script],
                check=False, capture_output=True, text=True, env=env,
            ).returncode == 0
            _set_mode(value)
            try:
                workspace_mode()
                py_ok = True
            except ValueError:
                py_ok = False
            assert py_ok == shell_ok, (value, py_ok, shell_ok)
    finally:
        _set_mode(saved)


# --------------------------------------------------------------------------
# 2. the sizing formula
# --------------------------------------------------------------------------
def test_stock_matches_live_workspace_receipt() -> None:
    """The stock sizing reproduces the measured lock, to the byte.

    Head Spark, Stage A boot 2026-09-01, VLLM_DEBUG_WORKSPACE=1:

        [WORKSPACE DEBUG] Resized workspace from
          'sparse_attn_indexer_kpool.py:284:sparse_attn_indexer_kpool':
          0.00 MB -> 5036.40 MB (ubatch 0)

    sparse_attn_indexer_kpool.py:284 is the profiling-run
    ``get_simultaneous(values_spec, scales_spec, radix)`` call, so the receipt
    covers the gather workspace plus the 1 MiB radix top-k scratch.
    """
    entries = stock_workspace_entries(LIVE_MAX_MODEL_LEN)
    assert entries == LIVE_MAX_MODEL_LEN * STOCK_MULTIPLIER == 40_000_000
    total = entries * BYTES_PER_ENTRY + RADIX_TOPK_WORKSPACE_BYTES
    assert total == 5_281_048_576
    assert f"{total / (1024 * 1024):.2f}" == "5036.40"


def test_no_op_when_uncompressed() -> None:
    stock = stock_workspace_entries(LIVE_MAX_MODEL_LEN)
    # compress_ratio 1 => the workspace really is token-granular; leave it.
    assert rightsized_workspace_entries(cfg(kpool=1)) == stock
    assert rightsized_workspace_entries(cfg(kpool=0)) == stock
    assert rightsized_workspace_entries(cfg(kpool=None)) == stock
    # A config that has no index_kpool at all (DeepSeek-V3.2 dense indexer).
    assert rightsized_workspace_entries(cfg(with_kpool_attr=False)) == stock
    assert indexer_compress_ratio(cfg(with_kpool_attr=False)) == 1
    # A junk value must not size a workspace.
    assert rightsized_workspace_entries(cfg(kpool="four")) == stock


def test_true_per_step_maximum() -> None:
    """The default is the legal maximum, not a tuned request count.

    Codex's review rejected the draft's ``REQS=8``: 264 MB is a throughput
    tradeoff, not the required workspace. This sizes the largest total
    compressed N a single step can legally present.
    """
    got = rightsized_workspace_entries(cfg())
    per_req = cdiv(LIVE_MAX_MODEL_LEN + LIVE_SPEC, LIVE_KPOOL)
    assert per_req == 250_002
    assert got == per_req * LIVE_MAX_NUM_SEQS == 4_000_032
    assert got * BYTES_PER_ENTRY == 528_004_224
    # Recovered against stock: ~4.43 GiB at max_num_seqs=16.
    reclaimed = (stock_workspace_entries(LIVE_MAX_MODEL_LEN) - got) * BYTES_PER_ENTRY
    assert 4.40 < reclaimed / 1024**3 < 4.45
    # The recipe default max_num_seqs=4 reclaims more.
    got4 = rightsized_workspace_entries(cfg(max_num_seqs=4))
    assert got4 == per_req * 4 == 1_000_008
    reclaimed4 = (stock_workspace_entries(LIVE_MAX_MODEL_LEN) - got4) * BYTES_PER_ENTRY
    assert 4.75 < reclaimed4 / 1024**3 < 4.85


def test_spec_tokens_are_headroom() -> None:
    """k=7 draft tokens can push seq_lens_cpu_upper_bound past max_model_len.

    The builder feeds the splitter ``seq_lens_cpu_upper_bound``, documented as
    "an upper bound for async-spec extend rows", so the per-request span must
    carry the draft length.
    """
    no_spec = rightsized_workspace_entries(cfg(spec=0))
    with_spec = rightsized_workspace_entries(cfg(spec=LIVE_SPEC))
    none_cfg = rightsized_workspace_entries(cfg(spec=None))
    assert no_spec == cdiv(LIVE_MAX_MODEL_LEN, LIVE_KPOOL) * LIVE_MAX_NUM_SEQS
    assert with_spec > no_spec
    assert with_spec - no_spec == 2 * LIVE_MAX_NUM_SEQS  # cdiv rolls over by 2
    assert none_cfg == no_spec
    # Monotone in the draft length.
    prev = 0
    for k in range(0, 32):
        cur = rightsized_workspace_entries(cfg(spec=k))
        assert cur >= prev
        prev = cur


def test_prefill_request_count_is_mnbt_bounded() -> None:
    """A prefill row costs >= 1 query token, so MNBT caps the request count."""
    per_req = cdiv(LIVE_MAX_MODEL_LEN + LIVE_SPEC, LIVE_KPOOL)
    # MNBT below max_num_seqs: MNBT binds.
    assert rightsized_workspace_entries(cfg(max_num_seqs=64, mnbt=8)) == per_req * 8
    # max_num_seqs below MNBT: max_num_seqs binds.
    assert rightsized_workspace_entries(cfg(max_num_seqs=8, mnbt=2048)) == per_req * 8
    # Equal: either.
    assert rightsized_workspace_entries(cfg(max_num_seqs=8, mnbt=8)) == per_req * 8
    # Degenerate configs still leave room for one whole request (the hard
    # floor: the splitter sub-chunks an oversized single request on the query
    # dimension only, never on N).
    for seqs, mnbt in ((1, 1), (0, 2048), (16, 0)):
        got = rightsized_workspace_entries(cfg(max_num_seqs=seqs, mnbt=mnbt))
        assert got >= per_req, (seqs, mnbt, got)


def test_never_exceeds_stock() -> None:
    """Narrowing only. A config whose legal max is above stock keeps stock."""
    stock = stock_workspace_entries(LIVE_MAX_MODEL_LEN)
    # 256 seqs x 250,002 = 64,000,512 > 40,000,000.
    assert rightsized_workspace_entries(cfg(max_num_seqs=256)) == stock
    for M in (8192, 65_536, 163_840, 262_144, 1_000_000):
        for kpool in (2, 4, 8, 16):
            for seqs in (1, 4, 16, 64, 256):
                for mnbt in (1, 2048, 8192):
                    c = cfg(max_model_len=M, kpool=kpool, max_num_seqs=seqs, mnbt=mnbt)
                    got = rightsized_workspace_entries(c)
                    assert got <= M * STOCK_MULTIPLIER
                    assert got >= min(
                        cdiv(M + LIVE_SPEC, kpool), M * STOCK_MULTIPLIER
                    )


# --------------------------------------------------------------------------
# 3. chunking equivalence -- the "no behaviour change" claim, checked
# --------------------------------------------------------------------------
def _legal_batches(rng, count=400):
    """Batches the scheduler can actually form at the live config."""
    for _ in range(count):
        n = rng.randint(1, LIVE_MAX_NUM_SEQS)
        # Query tokens: >= 1 each, total <= MNBT.
        qlens = [1] * n
        for _ in range(rng.randint(0, LIVE_MNBT - n)):
            qlens[rng.randrange(n)] += 1
        seqs = [rng.randint(q, LIVE_MAX_MODEL_LEN) for q in qlens]
        yield [s // LIVE_KPOOL for s in seqs], qlens
    # The extremes, deterministically.
    yield [LIVE_MAX_MODEL_LEN // LIVE_KPOOL] * LIVE_MAX_NUM_SEQS, [
        LIVE_MNBT // LIVE_MAX_NUM_SEQS
    ] * LIVE_MAX_NUM_SEQS
    yield [LIVE_MAX_MODEL_LEN // LIVE_KPOOL], [LIVE_MNBT]
    yield [1] * LIVE_MAX_NUM_SEQS, [1] * LIVE_MAX_NUM_SEQS


def test_chunking_is_identical_to_stock() -> None:
    stock_ws = stock_workspace_entries(LIVE_MAX_MODEL_LEN)
    right_ws = rightsized_workspace_entries(cfg())
    assert right_ws < stock_ws
    rng = random.Random(20260901)
    seen_multi_chunk = False
    for compressed, qlens in _legal_batches(rng):
        a = split_prefill_chunks(compressed, qlens, stock_ws, MAX_LOGITS_BYTES)
        b = split_prefill_chunks(compressed, qlens, right_ws, MAX_LOGITS_BYTES)
        assert a == b, (compressed[:4], qlens[:4], len(a), len(b))
        seen_multi_chunk = seen_multi_chunk or len(a) > 1
    # The comparison has to have exercised the splitter's other constraint,
    # or it proves nothing about chunking at all.
    assert seen_multi_chunk


def test_chunking_test_has_power() -> None:
    """An under-sized workspace must visibly change the chunk list.

    Without this the equivalence test above could pass on a splitter replica
    that ignores the workspace argument.
    """
    stock_ws = stock_workspace_entries(LIVE_MAX_MODEL_LEN)
    per_req = LIVE_MAX_MODEL_LEN // LIVE_KPOOL
    compressed = [per_req] * 4
    qlens = [8] * 4
    full = split_prefill_chunks(compressed, qlens, stock_ws, MAX_LOGITS_BYTES)
    starved = split_prefill_chunks(compressed, qlens, per_req, MAX_LOGITS_BYTES)
    assert len(starved) > len(full)
    # The draft's rejected REQS=8 sizing starves a 16-way 1M-context prefill:
    # at one query token per row the logits budget still admits all 16, but a
    # 2,000,000-entry workspace splits them into two chunks and re-runs
    # cp_gather_indexer_k_quant_cache. The legal-maximum sizing does not.
    wide, thin = [per_req] * 16, [1] * 16
    stock_chunks = split_prefill_chunks(wide, thin, stock_ws, MAX_LOGITS_BYTES)
    assert len(stock_chunks) == 1
    assert (
        split_prefill_chunks(wide, thin, per_req * 8, MAX_LOGITS_BYTES)
        != stock_chunks
    )
    assert (
        split_prefill_chunks(
            wide, thin, rightsized_workspace_entries(cfg()), MAX_LOGITS_BYTES
        )
        == stock_chunks
    )


# --------------------------------------------------------------------------
# 4. patch apply / idempotence / drift
# --------------------------------------------------------------------------
def _run_patch(target: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_INDEXER_BACKEND_PY"] = str(target)
    return subprocess.run(
        [sys.executable, str(PATCH), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_fixture_apply_and_idempotence() -> None:
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "indexer.py"
        target.write_text(PINNED_FIXTURE)

        pre = _run_patch(target, "--preflight")
        assert pre.returncode == 0, pre.stderr
        assert "preflight OK" in pre.stdout
        assert target.read_text() == PINNED_FIXTURE  # preflight writes nothing

        first = _run_patch(target)
        assert first.returncode == 0, first.stderr
        text = target.read_text()
        assert verified_state(text)
        assert MARK_SIZE in text
        assert "_glm53_rightsized_workspace_entries" in text
        assert "already present" not in first.stdout
        compile(text, str(target), "exec")

        second = _run_patch(target)
        assert second.returncode == 0, second.stderr
        assert "already present" in second.stdout
        assert target.read_text() == text

        again, action = prepare(text)
        assert action == "already present"
        assert again == text

        post = _run_patch(target, "--preflight")
        assert post.returncode == 0, post.stderr


class _MLAAttentionSpec:
    """Stand-in for vllm.v1.kv_cache_interface.MLAAttentionSpec.

    Only ``compress_ratio`` and the isinstance check matter to the guard.
    """

    def __init__(self, compress_ratio: int):
        self.compress_ratio = compress_ratio


class _NonMLASpec:
    """A kv-cache spec that is NOT an MLAAttentionSpec.

    Stock leaves ``self.compress_ratio = 1`` for these, which is the same
    under-sizing shape as an MLA spec that reports 1.
    """


def _patched_sizing_namespace() -> dict:
    """Exec the patched sizing AND the patched builder OUT OF THE FILE TEXT.

    HELPERS_SRC is shared between the overlay and the injected code, but the
    body of ``get_max_prefill_buffer_size`` itself -- the mode dispatch, the
    stock clamp and its warning -- and the whole builder cross-check exist only
    in the patched file. Pull them out by AST and run them, so the shipped
    dispatch and the shipped guard are covered and not just the formula.
    """
    import ast

    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "indexer.py"
        target.write_text(PINNED_FIXTURE)
        assert _run_patch(target).returncode == 0
        tree = ast.parse(target.read_text())

    def _wanted(node) -> bool:
        if isinstance(node, ast.FunctionDef):
            return (
                node.name.startswith("_glm53_")
                or node.name == "get_max_prefill_buffer_size"
            )
        if isinstance(node, ast.ClassDef):
            return node.name == "DeepseekV32IndexerMetadataBuilder"
        if isinstance(node, ast.Assign):
            return getattr(node.targets[0], "id", "") == "_GLM53_WORKSPACE_ENV"
        return False

    wanted = [node for node in tree.body if _wanted(node)]
    for name in ("get_max_prefill_buffer_size", "DeepseekV32IndexerMetadataBuilder"):
        assert any(getattr(n, "name", None) == name for n in wanted), name
    calls: list[tuple[str, tuple]] = []

    class _Logger:
        def info(self, *a):
            calls.append(("info", a))

        def warning(self, *a):
            calls.append(("warning", a))

    ns = {
        "os": os,
        "logger": _Logger(),
        "VllmConfig": object,
        "MLAAttentionSpec": _MLAAttentionSpec,
        "_calls": calls,
    }
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "<patched>", "exec"), ns)
    return ns


def test_patched_sizing_dispatch() -> None:
    ns = _patched_sizing_namespace()
    get_size = ns["get_max_prefill_buffer_size"]
    calls = ns["_calls"]
    stock = stock_workspace_entries(LIVE_MAX_MODEL_LEN)
    saved = os.environ.get(ENV_NAME)
    try:
        # Default and explicit stock are the stock expression, exactly.
        for value in (None, "stock"):
            _set_mode(value)
            calls.clear()
            assert get_size(cfg()) == stock == LIVE_MAX_MODEL_LEN * 40
            assert calls == []  # stock is silent: no new boot-log noise

        _set_mode("rightsize")
        calls.clear()
        assert get_size(cfg()) == 4_000_032
        assert [k for k, _ in calls] == ["info"]

        # Legal max above stock -> stock, loudly.
        calls.clear()
        assert get_size(cfg(max_num_seqs=256)) == stock
        assert [k for k, _ in calls] == ["warning"]

        # Non-kpool config is a no-op even when the knob is on.
        calls.clear()
        assert get_size(cfg(kpool=1)) == stock
        assert [k for k, _ in calls] == ["warning"]

        _set_mode("bogus")
        try:
            get_size(cfg())
        except ValueError as exc:
            assert ENV_NAME in str(exc)
        else:
            raise AssertionError("bad mode must raise from the patched module")
    finally:
        _set_mode(saved)


def _build(ns, spec, config):
    return ns["DeepseekV32IndexerMetadataBuilder"](spec, config, "cpu")


def test_builder_ratio_mismatch_raises_both_directions() -> None:
    """The cross-check is unconditional in rightsize mode, in both directions.

    Regression for the blocker Codex found in the first PR-8 draft: the guard
    read ``if mode == "rightsize" and self.compress_ratio > 1``, so the ONE
    combination that actually under-sizes -- ``hf_text_config.index_kpool=4``
    with ``kv_cache_spec.compress_ratio=1`` -- skipped the check entirely. The
    workspace is then sized for compressed pools while the runtime hands the
    splitter raw token counts, roughly 4x too many.
    """
    ns = _patched_sizing_namespace()
    calls = ns["_calls"]
    saved = os.environ.get(ENV_NAME)
    try:
        _set_mode("rightsize")

        # Direction A -- the under-sizing one, and the one the old guard skipped.
        # Sizing believes ratio 4; the runtime divides by 1.
        sized_for = ns["get_max_prefill_buffer_size"](cfg(kpool=4))
        assert sized_for == 4_000_032
        runtime_worst_case = LIVE_MAX_NUM_SEQS * (LIVE_MAX_MODEL_LEN + LIVE_SPEC)
        assert runtime_worst_case > 3.9 * sized_for  # ~4x under-sized, uncaught
        for spec in (_MLAAttentionSpec(1), _NonMLASpec()):
            calls.clear()
            try:
                _build(ns, spec, cfg(kpool=4))
            except ValueError as exc:
                msg = str(exc)
                assert "index_kpool=4" in msg, msg
                assert "compress_ratio=1" in msg, msg
                assert ENV_NAME in msg, msg
            else:
                raise AssertionError(f"index_kpool=4 vs compress_ratio=1 ({spec}) "
                                     "must raise")

        # Direction B -- config says no compression, the runtime compresses.
        # Sizing returns stock here, so it cannot under-run, but the two config
        # sources disagree about the model and rightsize must not serve on that.
        for kpool in (1, 0, None):
            calls.clear()
            try:
                _build(ns, _MLAAttentionSpec(4), cfg(kpool=kpool))
            except ValueError as exc:
                msg = str(exc)
                assert "index_kpool=1" in msg, msg
                assert "compress_ratio=4" in msg, msg
            else:
                raise AssertionError(f"index_kpool={kpool} vs compress_ratio=4 "
                                     "must raise")
        # ... including the config that has no index_kpool attribute at all.
        try:
            _build(ns, _MLAAttentionSpec(4), cfg(with_kpool_attr=False))
        except ValueError as exc:
            assert "compress_ratio=4" in str(exc)
        else:
            raise AssertionError("missing index_kpool vs compress_ratio=4 must raise")
    finally:
        _set_mode(saved)


def test_builder_agreeing_ratios_pass() -> None:
    """Agreement builds, at the compressed AND the uncompressed ratio."""
    ns = _patched_sizing_namespace()
    calls = ns["_calls"]
    stock = stock_workspace_entries(LIVE_MAX_MODEL_LEN)
    saved = os.environ.get(ENV_NAME)
    try:
        _set_mode("rightsize")

        calls.clear()
        builder = _build(ns, _MLAAttentionSpec(4), cfg(kpool=4))
        assert builder.max_prefill_buffer_size == 4_000_032
        # One sizing info + one builder info, no warning.
        assert [k for k, _ in calls] == ["info", "info"]

        # ratio 1 on both sides: a dense indexer. Sized at stock, and the guard
        # now runs (it used to be skipped) and finds agreement.
        calls.clear()
        builder = _build(ns, _MLAAttentionSpec(1), cfg(kpool=1))
        assert builder.max_prefill_buffer_size == stock
        assert [k for k, _ in calls] == ["warning", "info"]

        calls.clear()
        builder = _build(ns, _NonMLASpec(), cfg(with_kpool_attr=False))
        assert builder.max_prefill_buffer_size == stock
    finally:
        _set_mode(saved)


def test_builder_guard_is_inert_in_stock_mode() -> None:
    """Stock mode is untouched: no cross-check, no logs, stock sizing."""
    ns = _patched_sizing_namespace()
    calls = ns["_calls"]
    stock = stock_workspace_entries(LIVE_MAX_MODEL_LEN)
    saved = os.environ.get(ENV_NAME)
    try:
        for mode in (None, "stock"):
            _set_mode(mode)
            for spec, config in (
                (_MLAAttentionSpec(1), cfg(kpool=4)),   # direction A
                (_MLAAttentionSpec(4), cfg(kpool=1)),   # direction B
                (_MLAAttentionSpec(4), cfg(kpool=4)),   # agreement
                (_NonMLASpec(), cfg(kpool=4)),
            ):
                calls.clear()
                builder = _build(ns, spec, config)
                assert builder.max_prefill_buffer_size == stock, mode
                assert calls == [], (mode, calls)
    finally:
        _set_mode(saved)


def test_builder_guard_has_no_ratio_condition() -> None:
    """The shipped guard must not reintroduce a ``compress_ratio > 1`` gate.

    The behavioural tests above would still pass against a guard that was
    merely reordered; this pins the source form that made the blocker possible.
    """
    from patch_indexer_workspace import PATCHED_GUARD

    dispatch = PATCHED_GUARD.split("[glm53-indexer-workspace]", 1)[1]
    assert 'if _glm53_workspace_mode() == "rightsize":' in dispatch
    old_form = 'if _glm53_workspace_mode() == "rightsize" and self.compress_ratio'
    assert old_form not in dispatch


def test_fail_closed_on_drift() -> None:
    for old, new in (
        ("return max_model_len * 40", "return max_model_len * 48"),
        ("from dataclasses import dataclass", "from dataclasses import dataclass, field"),
        ("self.compress_ratio = self.kv_cache_spec.compress_ratio",
         "self.compress_ratio = self.kv_cache_spec.kv_compress_ratio"),
    ):
        drifted = PINNED_FIXTURE.replace(old, new, 1)
        assert drifted != PINNED_FIXTURE, old
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "indexer.py"
            target.write_text(drifted)
            result = _run_patch(target)
            assert result.returncode != 0, old
            assert "preflight failed" in result.stderr
            assert "drifted" in result.stderr
            # A drifted anchor leaves the file untouched.
            assert target.read_text() == drifted
            # ... and --preflight fails the same way, so CI catches drift while
            # the knob is still off.
            pre = _run_patch(target, "--preflight")
            assert pre.returncode != 0


def test_fail_closed_on_half_patched() -> None:
    partial = PINNED_FIXTURE.replace(ANCHOR_SIZE, MARK_SIZE + ANCHOR_SIZE, 1)
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "indexer.py"
        target.write_text(partial)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "partial/inconsistent" in result.stderr
        assert target.read_text() == partial


def test_live_copy_if_present() -> None:
    """Apply to a copy of the installed/live file when one is reachable."""
    src = Path(os.environ.get("GLM53_INDEXER_BACKEND_PY_SRC", INSTALLED))
    if not src.is_file():
        return
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "indexer.py"
        target.write_text(src.read_text())
        result = _run_patch(target)
        assert result.returncode == 0, result.stderr
        text = target.read_text()
        assert verified_state(text)
        compile(text, str(target), "exec")


def test_live_container_copy_if_enabled() -> None:
    """Opt-in: pull indexer.py read-only from the running head container.

    Off by default -- the image build and CI have no route to the Sparks.
    Enable with GLM53_INDEXER_LIVE_SSH=1.
    """
    if os.environ.get("GLM53_INDEXER_LIVE_SSH") != "1":
        return
    jump = os.environ.get("GLM53_LIVE_SSH_JUMP", "blockbrain@192.168.1.116")
    head = os.environ.get("GLM53_LIVE_SSH_HEAD", "10.0.22.1")
    container = os.environ.get("GLM53_LIVE_CONTAINER", "glm53-exl3-head")
    remote = (
        "/usr/local/lib/python3.12/dist-packages/vllm/"
        "v1/attention/backends/mla/indexer.py"
    )
    fetched = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", jump,
            f'ssh {head} "docker exec {container} cat {remote}"',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert fetched.returncode == 0, fetched.stderr
    assert "def get_max_prefill_buffer_size" in fetched.stdout
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "indexer.py"
        target.write_text(fetched.stdout)
        result = _run_patch(target)
        assert result.returncode == 0, result.stderr
        assert verified_state(target.read_text())


# --------------------------------------------------------------------------
# 5. recipe wiring
# --------------------------------------------------------------------------
def test_recipe_wiring_if_present() -> None:
    start = ROOT / "start.sh"
    dockerfile = ROOT / "Dockerfile"
    readme = ROOT / "README.md"
    if not start.is_file() or not dockerfile.is_file():
        return
    launcher = start.read_text()
    image = dockerfile.read_text()
    assert 'GLM53_INDEXER_WORKSPACE="${GLM53_INDEXER_WORKSPACE-stock}"' in launcher
    assert '_cli_indexer_workspace="${GLM53_INDEXER_WORKSPACE-}"' in launcher
    # Setness-aware capture: an explicitly empty caller value must survive the
    # .env source and reach the enum guard, not be swallowed by a .env value.
    assert '_cli_indexer_workspace_set="${GLM53_INDEXER_WORKSPACE+1}"' in launcher
    assert (
        '[ -n "${_cli_indexer_workspace_set}" ] '
        '&& GLM53_INDEXER_WORKSPACE="$_cli_indexer_workspace"'
    ) in launcher
    assert '_glm53_validate_enum GLM53_INDEXER_WORKSPACE' in launcher
    assert '-e "GLM53_INDEXER_WORKSPACE=$GLM53_INDEXER_WORKSPACE"' in launcher
    assert launcher.count("python3 /opt/glm53/patch_indexer_workspace.py") == 2
    assert "COPY overlay/patch_indexer_workspace.py" in image
    assert "COPY tests/test_indexer_workspace.py" in image
    assert "RUN python3 /opt/glm53/patch_indexer_workspace.py" in image
    assert "python3 /opt/glm53/test_indexer_workspace.py" in image
    if readme.is_file():
        text = readme.read_text()
        assert "`GLM53_INDEXER_WORKSPACE`" in text
        assert "tests/test_indexer_workspace.py" in text


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"indexer workspace right-sizing OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
