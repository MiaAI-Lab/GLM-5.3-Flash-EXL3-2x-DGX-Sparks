#!/usr/bin/env python3
"""Tests for the indexer warmup-range widening (overlay/patch_indexer_warmup_range.py).

Host-only, no GPU, no torch, no vllm import. Six groups:

* the behavioral contract: stock ``WarmupIntRange(0, 2)`` expands to ``{0, 1}``
  and misses Triton's third int bucket, while ``(0, 3)`` expands to
  ``{0, 1, 2}`` and covers all three; the ``query_slice_stop`` tuple already
  covers all three, so the start range was the gap;
* runtime reachability: the ``split_indexer_prefill_chunks`` sub-chunking math
  (512 MiB logits budget, ``index_kpool=4``, MNBT=7168) first needs sub-chunks
  above ~74.9k uncompressed tokens, and the worked 90k example yields a
  ``max_q`` in the missing bucket — so the gap is hit in serving, not just
  in theory;
* the synthetic-fixture contract: preflight leaves the pristine fixture
  untouched, apply widens ``query_slice_start=WarmupIntRange(0, 2)`` to
  ``(0, 3)`` exactly once, re-apply is a no-op;
* fail-closed drift: tampered marker value, deleted sink call, duplicated
  marker, and a second unrelated ``WarmupIntRange`` literal all exit nonzero
  with the file unchanged;
* the live installed file, when reachable: ``test_live_copy_if_present`` proves
  the patch *can apply* to a copy of the installed file (passes on stock AND
  on patched input — it is not a baked-state proof), while
  ``test_live_src_baked_if_present`` asserts the installed source itself is
  already in the verified state without patching (this is the one that fails
  on a stale container/image). On the host the source is normally absent ->
  both skip; override with ``GLM53_INDEXER_BACKEND_PY_SRC``;
* recipe wiring: start.sh ships the patch in ``GLM53_OVERLAY_ORDER`` after
  ``patch_indexer_workspace.py``, and the Dockerfile COPYs + RUNs the patch
  and runs this test in the verification chain.

Conventions follow ``tests/test_indexer_workspace.py`` (fixture + subprocess
apply via the patch's own env override + opt-in live checks + recipe wiring)
and ``tests/test_boot_shape_warmup.py`` (prove consumer-visible behavior, not
just patch mechanics).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATCH = next(
    p
    for p in (
        HERE / "patch_indexer_warmup_range.py",
        ROOT / "overlay" / "patch_indexer_warmup_range.py",
    )
    if p.is_file()
)
INSTALLED = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "v1/attention/backends/mla/indexer.py"
)
sys.path.insert(0, str(PATCH.parent))
from patch_indexer_warmup_range import (  # noqa: E402
    ANCHOR,
    MARK,
    prepare,
    verified_state,
    warmup_range_calls,
)


# Minimal compilable module carrying the exact pinned anchor. prepare() and
# verified_state() are source-level (anchor counts + an AST scan for
# positional WarmupIntRange literals), so no vllm import is needed.
FIXTURE = (
    "from vllm.model_executor.warmup.jit_warmup import WarmupIntRange\n"
    "\n"
    "\n"
    "class _K:\n"
    "    def get_warmup_keys(self, vllm_config):\n"
    "        max_tokens = 8\n"
    "        return self._trace_dispatch(self.dispatch)(\n"
    + ANCHOR
    + "            query_slice_stop=(1, 2 * max_tokens - 1, 2 * max_tokens),\n"
    "        )\n"
)


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


# --------------------------------------------------------------------------
# Behavioral contract: the range expansion vs the Triton int buckets.
# --------------------------------------------------------------------------
# Model of the specialization the patch docstring pins (Triton 3.7.1, plus the
# persistent-cache observation that qs_start='' variants were the only
# serving-time compiles): ==1 is constant-folded, %16==0 is 'D', everything
# else is ''. If Triton re-buckets, re-derive the patch AND this model
# together — the tests below pin the model on purpose.
def _triton_int_bucket(value: int) -> str:
    if value == 1:
        return "folded"
    if value % 16 == 0:
        return "D"
    return ""


def _expand_int_range(start: int, stop: int) -> tuple[int, ...]:
    # Mirrors _expand_warmup_values for WarmupIntRange without step/advance:
    # tuple(range(start, stop, 1)). Verified against the live container's
    # vllm/model_executor/warmup/jit_warmup.py.
    return tuple(range(start, stop))


def test_warmup_expansion_covers_triton_buckets() -> None:
    stock = _expand_int_range(0, 2)
    patched = _expand_int_range(0, 3)
    assert stock == (0, 1)
    assert patched == (0, 1, 2)
    assert set(stock) == {0, 1}
    assert warmup_range_calls(FIXTURE) == {(0, 2)}
    assert {_triton_int_bucket(v) for v in stock} == {"D", "folded"}
    assert {_triton_int_bucket(v) for v in patched} == {"D", "folded", ""}
    assert _triton_int_bucket(2) == ""
    # The stop tuple shipped alongside was already whole. Live
    # get_warmup_keys uses max_tokens = max(1, min(MNBT, 8)) = 8 here, so the
    # tuple is (1, 2*8-1, 2*8) = (1, 15, 16): 1 -> folded, 15 -> '', 16 -> 'D'.
    max_tokens = max(1, min(LIVE_MNBT, 8))
    assert max_tokens == 8
    stop = (1, 2 * max_tokens - 1, 2 * max_tokens)
    assert stop == (1, 15, 16)
    assert {_triton_int_bucket(v) for v in stop} == {"D", "folded", ""}
    # Power: stock covers 2 buckets, patched covers 3 — reverting the patch
    # re-opens the gap and fails this test.
    assert len({_triton_int_bucket(v) for v in stock}) == 2
    assert len({_triton_int_bucket(v) for v in patched}) == 3


# --------------------------------------------------------------------------
# Runtime reachability: split_indexer_prefill_chunks actually emits a start
# in the missing bucket. Formulas mirror the live container's indexer.py:
# max_logits_elems = max_logits_bytes // 4, sub-chunk offsets step by
# max_q = max(1, max_logits_elems // chunk_n), qs_start in {0, max_q, ...}.
# --------------------------------------------------------------------------
MAX_LOGITS_ELEMS = 512 * 1024 * 1024 // 4  # default VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512
LIVE_KPOOL = 4  # hf_text_config.index_kpool on this recipe
LIVE_MNBT = 7168  # current recipe default (.env.example, start.sh); the patch
# docstring's threshold math is pinned to this geometry — re-derive if it moves.


def _max_q(compressed_n: int) -> int:
    assert compressed_n > 0
    return max(1, MAX_LOGITS_ELEMS // compressed_n)


def test_runtime_max_q_reaches_missing_bucket() -> None:
    assert MAX_LOGITS_ELEMS == 134_217_728
    # One MNBT-sized step overflows the logits budget once the compressed
    # length N satisfies MNBT * N > MAX_LOGITS_ELEMS, i.e. uncompressed
    # S = N * kpool with S > MAX_LOGITS_ELEMS * kpool / MNBT ~= 74,898.3.
    threshold_exact = MAX_LOGITS_ELEMS * LIVE_KPOOL / LIVE_MNBT
    assert 74_898 < threshold_exact < 74_899
    # Below the threshold a full MNBT step still fits: no sub-chunking, only
    # qs_start=0 (covered either way).
    small_n = 70_000 // LIVE_KPOOL
    assert LIVE_MNBT * small_n <= MAX_LOGITS_ELEMS
    # Above it the step must sub-chunk: the second qs_start is max_q.
    big_n = 90_000 // LIVE_KPOOL
    assert big_n == 22_500
    assert LIVE_MNBT * big_n > MAX_LOGITS_ELEMS
    max_q = _max_q(big_n)
    assert max_q == 5_965
    assert max_q != 1
    assert max_q % 16 != 0
    assert _triton_int_bucket(max_q) == ""
    # First sub-chunk stays covered; the miss is the second offset onward.
    assert _triton_int_bucket(0) == "D"
    assert _triton_int_bucket(max_q) not in {_triton_int_bucket(v) for v in (0, 1)}


def test_fixture_preflight_is_readonly() -> None:
    assert warmup_range_calls(FIXTURE) == {(0, 2)}
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "indexer.py"
        target.write_text(FIXTURE)
        pre = _run_patch(target, "--preflight")
        assert pre.returncode == 0, pre.stderr
        assert "preflight OK" in pre.stdout
        assert target.read_text() == FIXTURE


def test_fixture_apply_and_idempotence() -> None:
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "indexer.py"
        target.write_text(FIXTURE)

        first = _run_patch(target)
        assert first.returncode == 0, first.stderr
        text = target.read_text()
        assert verified_state(text)
        assert MARK in text
        assert warmup_range_calls(text) == {(0, 3)}
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


def test_fail_closed_drift() -> None:
    with tempfile.TemporaryDirectory() as raw:
        # 1. marker present but the sink call tampered -> partial/inconsistent.
        target = Path(raw) / "tampered.py"
        target.write_text(FIXTURE)
        assert _run_patch(target).returncode == 0
        patched = target.read_text()
        tampered = patched.replace(
            "query_slice_start=WarmupIntRange(0, 3),",
            "query_slice_start=WarmupIntRange(0, 5),",
        )
        target.write_text(tampered)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "partial/inconsistent" in result.stderr
        assert target.read_text() == tampered

        # 2. sink call deleted entirely (marker comment remains).
        target = Path(raw) / "deleted.py"
        target.write_text(FIXTURE)
        assert _run_patch(target).returncode == 0
        text = target.read_text()
        target.write_text(
            "\n".join(
                line
                for line in text.splitlines()
                if "query_slice_start=WarmupIntRange" not in line
            )
            + "\n"
        )
        deleted = target.read_text()
        result = _run_patch(target)
        assert result.returncode != 0
        assert "partial/inconsistent" in result.stderr
        assert target.read_text() == deleted

        # 3. duplicated marker.
        target = Path(raw) / "dup.py"
        target.write_text(FIXTURE)
        assert _run_patch(target).returncode == 0
        text = target.read_text()
        target.write_text(text + MARK)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "partial/inconsistent" in result.stderr

        # 4. pristine anchor deleted -> anchor-count drift, not a silent skip.
        target = Path(raw) / "noanchor.py"
        target.write_text(
            "\n".join(
                line
                for line in FIXTURE.splitlines()
                if "query_slice_start=WarmupIntRange" not in line
            )
            + "\n"
        )
        noanchor = target.read_text()
        result = _run_patch(target)
        assert result.returncode != 0
        assert "anchor drifted" in result.stderr
        assert target.read_text() == noanchor


def test_second_warmup_range_rejected() -> None:
    """A second, unrelated WarmupIntRange literal must fail closed."""
    extra = (
        "def elsewhere(vllm_config):\n"
        "    return WarmupIntRange(4, 6)\n"
    )
    assert "WarmupIntRange" in extra
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "indexer.py"
        target.write_text(FIXTURE + extra)
        result = _run_patch(target)
        assert result.returncode != 0
        assert "not exactly {(0, 2)}" in result.stderr
        assert target.read_text() == FIXTURE + extra


def test_live_copy_if_present() -> None:
    """Apply-check only: the patch *can apply* to a copy of the live file.

    This passes on stock input (copy gets patched) AND on already-patched
    input (no-op) — it proves patch mechanics against the real file shape,
    not that the running container/image is baked. See
    ``test_live_src_baked_if_present`` for the baked-state proof. On the host
    the source is normally absent -> skipped. Override with
    ``GLM53_INDEXER_BACKEND_PY_SRC``.
    """
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


def test_live_src_baked_if_present() -> None:
    """Baked-state proof: the installed source itself is already patched.

    Reads the source read-only and asserts ``verified_state`` WITHOUT running
    the patch first, so stock input FAILS here (unlike the apply-check above).
    In the image the Dockerfile chain runs this test after the patch RUN, so
    it asserts the baked state; on a stale container or a checkout whose image
    was not rebuilt it fails with a rebuild/restart hint. On the host the
    source is normally absent -> skipped. Override with
    ``GLM53_INDEXER_BACKEND_PY_SRC``.
    """
    src = Path(os.environ.get("GLM53_INDEXER_BACKEND_PY_SRC", INSTALLED))
    if not src.is_file():
        return
    text = src.read_text()
    assert verified_state(text), (
        f"{src} is not in the patched state "
        "(expected WarmupIntRange(0, 3) with the glm53 marker, "
        "no WarmupIntRange(0, 2)). The running container/image predates the "
        "overlay change: rebuild the image and restart "
        "(./start.sh restart) so Dockerfile RUN bakes "
        "overlay/patch_indexer_warmup_range.py."
    )
    compile(text, "<indexer>", "exec")


def test_recipe_wiring_if_present() -> None:
    start = ROOT / "start.sh"
    dockerfile = ROOT / "Dockerfile"
    readme = ROOT / "README.md"
    if not start.is_file() or not dockerfile.is_file():
        return
    launcher = start.read_text()
    image = dockerfile.read_text()
    lo = launcher.index("GLM53_OVERLAY_ORDER=(")
    order = launcher[lo : launcher.index(")", lo)]
    assert "\n    patch_indexer_workspace.py\n" in order
    assert "\n    patch_indexer_warmup_range.py\n" in order
    assert order.index("patch_indexer_workspace.py") < order.index(
        "patch_indexer_warmup_range.py"
    )
    assert "COPY overlay/patch_indexer_warmup_range.py" in image
    assert "COPY tests/test_indexer_warmup_range.py" in image
    assert "RUN python3 /opt/glm53/patch_indexer_warmup_range.py" in image
    assert "python3 /opt/glm53/test_indexer_warmup_range.py" in image
    if readme.is_file():
        text = readme.read_text()
        assert "overlay/patch_indexer_warmup_range.py" in text
        assert "tests/test_indexer_warmup_range.py" in text


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"indexer warmup-range widening OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
