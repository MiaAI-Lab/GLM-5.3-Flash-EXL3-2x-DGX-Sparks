#!/usr/bin/env python3
"""Host test for patch_mamba_align_state_free.py (no GPU, no vLLM import).

Applies the overlay to copies of single_type_kv_cache_manager.py and
kv_cache_interface.py, then replays a cold prefill through the real
``MambaManager`` (align mode, speculative blocks) the way ``allocate_slots``
drives it: ``remove_skipped_blocks`` on the processed prefix (computed minus
in-flight), allocation, ``cache_blocks``. Fails closed: any problem exits
non-zero.

    GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY=/path/to/single_type_kv_cache_manager.py \\
    GLM53_KV_CACHE_INTERFACE_PY=/path/to/kv_cache_interface.py \\
        python3 test_mamba_align_state_free.py

The sources may already carry the overlay (idempotence is part of the test).
``python3 -m pytest tests/test_mamba_align_state_free.py`` runs the same
suite; absent sources are a skip naming the variables.
"""
from __future__ import annotations

import abc
import ast
import collections
import collections.abc
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import types
import typing
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(
    p for p in (HERE / "patch_mamba_align_state_free.py", HERE.parent / "overlay" / "patch_mamba_align_state_free.py")
    if p.is_file()
)
_VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
STM_SRC = Path(os.environ.get("GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY", f"{_VLLM}/v1/core/single_type_kv_cache_manager.py"))
KVI_SRC = Path(os.environ.get("GLM53_KV_CACHE_INTERFACE_PY", f"{_VLLM}/v1/kv_cache_interface.py"))
MARK = "# [glm53-mamba-align-state-free-v1]"
BLOCK = 3584
SPEC_BLOCKS = 7

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


def apply(stm: Path, kvi: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY": str(stm), "GLM53_KV_CACHE_INTERFACE_PY": str(kvi)}
    return subprocess.run([sys.executable, str(PATCH)], env=env, text=True, capture_output=True)


def definitions(path: Path, ns: dict, names: set[str]) -> None:
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)


class Block:
    def __init__(self, block_id: int) -> None:
        self.block_id, self.ref_cnt, self.block_hash, self.is_null = block_id, 0, None, False


class Pool:
    """Ref-counting stand-in for BlockPool: enough for align-mode lifetime."""

    def __init__(self, num_blocks: int = 4096) -> None:
        self.hash_block_size = BLOCK
        self.null_block = Block(0)
        self.null_block.is_null = True
        self.free = collections.deque(Block(i) for i in range(1, num_blocks))
        self.num_gpu_blocks = num_blocks
        self.enable_caching = True

    def get_new_blocks(self, n: int) -> list[Block]:
        out = []
        for _ in range(n):
            b = self.free.popleft()
            b.block_hash, b.ref_cnt = None, 1
            out.append(b)
        return out

    def free_blocks(self, blocks) -> None:
        for b in blocks:
            if b.is_null:
                continue
            b.ref_cnt -= 1
            assert b.ref_cnt >= 0 and not b.is_null
            if b.ref_cnt == 0:
                self.free.append(b)

    def touch(self, blocks) -> None:
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null:
                self.free.remove(b)
            b.ref_cnt += 1

    def get_cached_block(self, block_hash, group_ids):
        return None

    def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id, block_mask) -> None:
        for i in range(num_cached_blocks, num_full_blocks):
            if not blocks[i].is_null and (block_mask is None or block_mask[i - num_cached_blocks]):
                blocks[i].block_hash = (i, kv_cache_group_id)

    def cache_partial_block(self, **kw):
        return None

    def referenced(self) -> int:
        return self.num_gpu_blocks - 1 - len(self.free)


def load_manager(stm: Path):
    spec_cls = type("MambaSpec", (), {})
    ns = {
        "ABC": abc.ABC, "abstractmethod": abc.abstractmethod, "ClassVar": typing.ClassVar,
        "Sequence": collections.abc.Sequence, "defaultdict": collections.defaultdict,
        "itertools": itertools, "cdiv": lambda a, b: -(-a // b), "MambaSpec": spec_cls,
        "AttentionSpec": type("AttentionSpec", (), {}),
        "logger": types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
    }
    definitions(stm, ns, {"SingleTypeKVCacheManager", "MambaManager"})
    return ns["MambaManager"], spec_cls


def replay(manager_cls, spec_cls, chunk: int, in_flight: bool, tokens: int, reuse=None):
    """One request's cold prefill; returns (held per step, held at end, pool)."""
    if reuse is None:
        spec = spec_cls()
        spec.block_size, spec.mamba_cache_mode, spec.num_speculative_blocks = BLOCK, "align", SPEC_BLOCKS
        pool = Pool()
        mgr = manager_cls(spec, block_pool=pool, enable_caching=True, kv_cache_group_id=2, scheduler_block_size=BLOCK)
    else:
        mgr, pool = reuse
    req = types.SimpleNamespace(request_id="r", num_prompt_tokens=tokens, num_tokens=tokens, block_hashes=[], shared_prefix_boundary=0)
    computed, last, held = 0, 0, []
    while computed < tokens:
        new = min(chunk, tokens - computed)
        processed = max(0, computed - (last if in_flight else 0))
        mgr.new_step_starts()
        mgr.remove_skipped_blocks("r", processed, num_prompt_tokens=tokens)
        blocks = mgr.req_to_blocks["r"]
        # The block holding the state at the processed boundary feeds the step
        # in flight; it must still be resident after every release.
        needed = -(-processed // BLOCK) - 1
        check_needed = (processed == 0 or not blocks[needed].is_null) and (
            computed == 0 or not blocks[-(-computed // BLOCK) - 1].is_null
        )
        mgr.get_num_blocks_to_allocate("r", computed + new, [], computed, computed, computed + new)
        mgr.allocate_new_blocks("r", computed + new, computed + new)
        mgr.cache_blocks(req, computed + new)
        computed += new
        last = new
        held.append((sum(not b.is_null for b in mgr.req_to_blocks["r"]), check_needed))
    return mgr, pool, held


def part_a(stm: Path, kvi: Path) -> None:
    print("Part A: installer is fail-closed and idempotent")
    r1 = apply(stm, kvi)
    if r1.returncode:
        raise RuntimeError(r1.stderr)
    before = (stm.read_bytes(), kvi.read_bytes())
    r2 = apply(stm, kvi)
    check(r2.returncode == 0 and (stm.read_bytes(), kvi.read_bytes()) == before, "A2 second application is a byte-identical no-op")
    with tempfile.TemporaryDirectory() as tmp:
        drifted = Path(tmp) / "stm.py"
        drifted.write_text(before[0].decode().replace("stale_state_block_idxs", "stale_state_block_idxz", 1))
        kvi_copy = Path(tmp) / "kvi.py"
        kvi_copy.write_bytes(before[1])
        drifted_before = (drifted.read_bytes(), kvi_copy.read_bytes())
        r3 = apply(drifted, kvi_copy)
        check(r3.returncode != 0 and (drifted.read_bytes(), kvi_copy.read_bytes()) == drifted_before, "A4 an edited applied block is rejected without changing either file")


def part_b(stm: Path, pristine_stm: Path) -> None:
    print("Part B: align-mode state blocks are released; in-flight state is kept")
    patched, spec_cls = load_manager(stm)
    original, original_spec_cls = load_manager(pristine_stm)
    pristine_available = MARK not in pristine_stm.read_text()
    tokens = 60 * BLOCK
    for label, chunk, in_flight, batches in (
        ("stock chunk 6272, one batch in flight", 6272, True, 2),
        ("stock chunk 7104, one batch in flight", 7104, True, 2),
        ("stock chunk 6272, synchronous", 6272, False, 1),
        ("custom chunk 896, one batch in flight", 896, True, 2),
    ):
        bound = 1 + batches + SPEC_BLOCKS
        _, pool_old, held_old = replay(original, original_spec_cls, chunk, in_flight, tokens)
        mgr, pool, held = replay(patched, spec_cls, chunk, in_flight, tokens)
        peak = max(h for h, _ in held)
        check(peak <= bound and pool.referenced() == held[-1][0], f"B1 [{label}] resident blocks stay within 1 + batches + spec = {bound} (peak {peak}, old peak {max(h for h, _ in held_old)})")
        check(all(ok for _, ok in held), f"B2 [{label}] the state block feeding the in-flight step is never released")
        if chunk > BLOCK and in_flight and pristine_available:
            check(max(h for h, _ in held_old) > bound + 10, f"B3 [{label}] unpatched manager retains superseded blocks ({max(h for h, _ in held_old)})")
        mgr.free("r")
        check(pool.referenced() == 0, f"B4 [{label}] freeing a request returns every referenced block")
        _, _, reused = replay(patched, spec_cls, chunk, in_flight, 6 * chunk, reuse=(mgr, pool))
        check(max(h for h, _ in reused) <= bound and all(ok for _, ok in reused), f"B5 [{label}] reusing a freed request ID preserves bounded ownership and live state")
        mgr.free("r")
        check(pool.referenced() == 0, f"B6 [{label}] the reused request releases all ownership")


def part_c(kvi: Path) -> None:
    print("Part C: align reservation covers one superseded block per concurrent batch")
    ns = {"cdiv": lambda a, b: -(-a // b)}
    src = kvi.read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MambaSpec")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "max_memory_usage_bytes")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(kvi), "exec"), ns)
    page = 2_351_104
    spec = types.SimpleNamespace(page_size_bytes=page, block_size=BLOCK, num_speculative_blocks=SPEC_BLOCKS)

    def cfg(mode, batches=2, max_len=262_144):
        return types.SimpleNamespace(cache_config=types.SimpleNamespace(mamba_cache_mode=mode), max_concurrent_batches=batches, model_config=types.SimpleNamespace(max_model_len=max_len))

    usage = ns["max_memory_usage_bytes"]
    check(usage(spec, cfg("align", 2)) == page * 10 and usage(spec, cfg("align", 1)) == page * 9, "C1 align: (1 + concurrent batches + spec) pages, i.e. 10 async / 9 sync here")
    check(usage(spec, cfg("none")) == page * 8 and usage(spec, cfg("all")) == page * (74 + 7), "C2 modes none / all unchanged")


def main() -> int:
    for var, path in (("GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY", STM_SRC), ("GLM53_KV_CACHE_INTERFACE_PY", KVI_SRC)):
        if not path.is_file():
            raise SystemExit(f"missing {path} (set {var})")
    with tempfile.TemporaryDirectory() as tmp:
        stm, kvi, pristine = Path(tmp) / "stm.py", Path(tmp) / "kvi.py", Path(tmp) / "stm_pristine.py"
        shutil.copyfile(STM_SRC, stm)
        shutil.copyfile(KVI_SRC, kvi)
        shutil.copyfile(STM_SRC, pristine)
        if MARK in pristine.read_text():
            print("  note: source already carries the overlay; B3 (unpatched growth) is skipped")
        part_a(stm, kvi)
        part_b(stm, pristine)
        part_c(kvi)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:\n  " + "\n  ".join(FAILURES))
        return 1
    print("mamba align state-free patch OK")
    return 0


def test_mamba_align_state_free() -> None:
    import pytest

    for var, path in (("GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY", STM_SRC), ("GLM53_KV_CACHE_INTERFACE_PY", KVI_SRC)):
        if not path.is_file():
            pytest.skip(f"missing {path}; set {var}")
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
