#!/usr/bin/env python3
"""Host test for patch_mamba_align_chunking.py (no GPU, no vLLM import).

Applies the overlay to a copy of scheduler.py and drives the real
``_mamba_block_aligned_split`` with the state the patched ``__init__`` derives,
checking chunk ends and the positions of states passed to the block pool
for hashing. The native manager is read beside the scheduler source unless
GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY overrides its path.

    GLM53_SCHEDULER_PY=/path/to/vllm/v1/core/sched/scheduler.py \\
        python3 test_mamba_align_chunking.py

The source may be pristine or carry patch_scheduler_decode_floor.py v5 (and
this overlay). ``python3 -m pytest tests/test_mamba_align_chunking.py`` runs
the same suite; an absent source is a skip naming the variable.
"""
from __future__ import annotations

import ast
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(
    p for p in (HERE / "patch_mamba_align_chunking.py", HERE.parent / "overlay" / "patch_mamba_align_chunking.py")
    if p.is_file()
)
SRC = Path(os.environ.get("GLM53_SCHEDULER_PY", "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"))
STM_SRC = Path(os.environ.get("GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY", SRC.parent.parent / "single_type_kv_cache_manager.py"))
MARK = "# [glm53-mamba-align-chunking-v2]"
MAMBA = 3584
DRAFT = 896

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + label)
    if not cond:
        FAILURES.append(label)


def apply(path: Path, patch: Path = PATCH) -> subprocess.CompletedProcess:
    env = {**os.environ, "GLM53_SCHEDULER_PY": str(path)}
    return subprocess.run([sys.executable, str(patch)], env=env, text=True, capture_output=True)


def split_function(path: Path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_mamba_block_aligned_split")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["_mamba_block_aligned_split"]


def init_statements(path: Path):
    """Execute the checkpoint attributes derived by Scheduler.__init__."""
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    wanted = {"mamba_align_sub_block_sizes", "mamba_align_eagle_backoff"}
    stmts = [
        s for s in init.body
        if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Attribute) and s.targets[0].attr in wanted
    ]
    assert len(stmts) == 2, [ast.dump(s.targets[0]) for s in stmts]
    fn = ast.FunctionDef(
        name="derive", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self"), ast.arg(arg="kv_cache_config")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=stmts, decorator_list=[], returns=None,
    )
    ns: dict = {"MambaSpec": MambaSpec}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), ns)
    return ns["derive"]


class MambaSpec:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.participates_in_prefix_caching = True


class Sched:
    """The attributes _mamba_block_aligned_split reads."""

    def __init__(self, budget: int, block_size: int, backoff: bool, eagle: bool = True, align_cap: int | None = None) -> None:
        self.cache_config = types.SimpleNamespace(block_size=DRAFT)
        self.max_num_scheduled_tokens = budget
        self.scheduler_config = types.SimpleNamespace(long_prefill_token_threshold=0)
        self.hash_block_size = DRAFT
        self.mamba_partial_cache_hit = False
        self.use_eagle = eagle
        self.block_size = block_size
        self.mamba_align_sub_block_sizes = ()
        self.mamba_align_eagle_backoff = backoff
        if align_cap is not None:
            self._glm53_align_prefill_limit = align_cap


def request(num_prompt: int, computed: int = 0) -> types.SimpleNamespace:
    return types.SimpleNamespace(num_computed_tokens=computed, num_prompt_tokens=num_prompt, num_tokens=num_prompt, shared_prefix_boundary=0)


def chunk_ends(split, sched: Sched, num_prompt: int, allowance: int, start: int = 0) -> list[int]:
    ends, computed, guard = [], start, 0
    while computed < num_prompt:
        req = request(num_prompt, computed)
        new = split(sched, req, min(allowance, num_prompt - computed), 0, 0)
        if new <= 0:
            ends.append(None)
            break
        computed += new
        ends.append(computed)
        guard += 1
        assert guard < 10_000
    return ends


def part_a(src: Path) -> None:
    print("Part A: installer")
    original = src.read_text()
    r1 = apply(src)
    if r1.returncode:
        raise RuntimeError(r1.stderr)
    before = src.read_bytes()
    r2 = apply(src)
    check(r2.returncode == 0 and src.read_bytes() == before, "A2 second application is a byte-identical no-op")
    if MARK not in original:
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "scheduler.py"
            unsupported = original.replace("# [glm53-decode-floor:v5]", "# [glm53-decode-floor:v4]")
            old.write_text(unsupported)
            r3 = apply(old)
            check(r3.returncode != 0 and old.read_text() == unsupported, "A4 an older decode-floor version is rejected without writing")
    else:
        print("  note: source already carries this overlay; A4 requires an unpatched scheduler")
    patch = runpy.run_path(str(PATCH))
    current = src.read_text()
    clean = current
    for label, old, new in patch["EDITS"]:
        assert clean.count(new) == 1, label
        clean = clean.replace(new, old, 1)
    retained = clean
    for label, old, new in patch["LEGACY_EDITS"]:
        assert retained.count(old) == 1, label
        retained = retained.replace(old, new, 1)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "scheduler.py"
        target.write_text(retained)
        migrated = apply(target)
        check(migrated.returncode == 0 and target.read_text() == current,
              "A5 retained checkpoint overlay migrates to identical combined source")
        for label, old, new in patch["EDITS"]:
            drifted = current.replace(new, old, 1)
            target.write_text(drifted)
            rejected = apply(target)
            check(rejected.returncode != 0 and target.read_text() == drifted,
                  f"A6 partially reverted {label} fails without writing")
    restarted = apply(src, PATCH.with_name("patch_scheduler_decode_floor.py"))
    check(restarted.returncode == 0 and src.read_text() == current,
          "A7 decode-floor restart preserves dedicated checkpoint ownership")


def part_b(src: Path) -> None:
    print("Part B: chunk ends land on the Mamba block, sub-block progress kept")
    split = split_function(src)
    derive = init_statements(src)
    groups = [
        types.SimpleNamespace(kv_cache_spec=types.SimpleNamespace(
            block_size=MAMBA, participates_in_prefix_caching=True)),
        types.SimpleNamespace(kv_cache_spec=MambaSpec(MAMBA)),
        types.SimpleNamespace(kv_cache_spec=MambaSpec(MAMBA)),
        types.SimpleNamespace(kv_cache_spec=type("SlidingWindowSpec", (), {
            "block_size": DRAFT, "participates_in_prefix_caching": True})()),
    ]
    for eagle_ids, expect in (({3}, False), ({0, 1, 2, 3}, True), ({1}, True), (set(), False)):
        s = Sched(budget=7168, block_size=MAMBA, backoff=False)
        s.kv_cache_manager = types.SimpleNamespace(coordinator=types.SimpleNamespace(eagle_group_ids=eagle_ids))
        derive(s, types.SimpleNamespace(kv_cache_groups=groups))
        ends = chunk_ends(split, s, 30720, 7160)
        check(None not in ends and all(end % MAMBA == 0 for end in ends[:-1]) and ((28672 not in ends) is expect), f"B1 target EAGLE groups {sorted(eagle_ids)} determine the last reusable checkpoint")

    stock = Sched(budget=7168, block_size=MAMBA, backoff=False)
    ends = chunk_ends(split, stock, 30720, 7160)
    check(ends == [MAMBA * k for k in range(1, 9)] + [30720], "B3 stock prefill checkpoints each Mamba boundary before its partial tail")
    ends = chunk_ends(split, Sched(budget=7168, block_size=MAMBA, backoff=True), 30720, 7160)
    check(28672 not in ends and 25088 in ends, "B4 with the EAGLE back-off the last checkpoint is skipped (documented full-attention drop)")
    old = Sched(budget=7168, block_size=DRAFT, backoff=True)
    check(28672 not in chunk_ends(split, old, 30720, 7160), "B5 drafter-page alignment misses the 28672 checkpoint (the SD1 loss)")

    custom = Sched(budget=1024, block_size=MAMBA, backoff=False)
    ends = chunk_ends(split, custom, 30720, 1016)
    check(None not in ends and all(b in ends for b in range(MAMBA, 30720, MAMBA)) and max(e - p for p, e in zip([0, *ends], ends)) <= 1016, "B6 sub-block budget advances inside a block and stops at each boundary")
    resumed = chunk_ends(split, stock, 30720, 7160, start=25088)
    check(resumed == [28672, 30720], "B7 a resumed prefix hit at 25088 checkpoints 28672 before the last chunk")
    if "_glm53_align_prefill_limit" in src.read_text():
        capped = Sched(budget=7168, block_size=MAMBA, backoff=False, align_cap=3576)
        ends = chunk_ends(split, capped, 30720, 3576)
        check(None not in ends and ends[0] == 3576 and ends[1] == MAMBA, "B8 a per-request cap below the block (decode-floor v5) keeps nonzero sub-block progress")
    else:
        print("  note: no decode-floor v5 in the source; B8 (capped sub-block progress) not exercised")
    remaining = Sched(budget=7168, block_size=MAMBA, backoff=False)
    ends = chunk_ends(split, remaining, 30720, 128)
    check(None not in ends and ends[0] == 128 and MAMBA in ends,
          "B9 a remaining grant below the configured budget cannot starve prefill")
    mixed = Sched(budget=7168, block_size=MAMBA, backoff=False)
    mixed.kv_cache_manager = types.SimpleNamespace(
        coordinator=types.SimpleNamespace(eagle_group_ids={3}))
    mixed_groups = [groups[0], types.SimpleNamespace(kv_cache_spec=MambaSpec(896)),
                    types.SimpleNamespace(kv_cache_spec=MambaSpec(1792)), groups[3]]
    derive(mixed, types.SimpleNamespace(kv_cache_groups=mixed_groups))
    check(split(mixed, request(30720), 4000) == MAMBA,
          "B10 shared checkpoints use the scheduler LCM even when all Mamba pages are smaller")
    check(split(mixed, request(30720, 2048), 4000) == 640,
          "B11 a smaller private Mamba page finishes before crossing its hash boundary")
    check(split(mixed, request(30720, 2048), 1024) == 640,
          "B12 a sub-page grant still stops at its private-state boundary")
    check(split(stock, request(30720, 3500), 128) == 84,
          "B13 a sub-page grant crossing a shared checkpoint stops exactly there")
    resumed = request(3584, 7000)
    resumed.num_tokens = 7745
    check(split(stock, resumed, 700) == 168,
          "B14 replaying output tokens still visits the next shared checkpoint")
    resumed.num_computed_tokens = 7168
    check(split(stock, resumed, 577) == 577,
          "B15 resumed final prefill tail is not rounded down")
    tail_sched = Sched(budget=7168, block_size=MAMBA, backoff=False)
    tail_sched.hash_block_size = 64
    tail_sched.mamba_partial_cache_hit = True
    check(split(tail_sched, request(7745, 7680), 65) == 64,
          "B16 final prompt tail stops on its fine hash boundary before logits")


def part_c(src: Path) -> None:
    """A mixed-cap chunk must not label a short state as a complete prefix."""
    from test_mamba_align_state_free import Pool, load_manager

    manager_cls, spec_cls = load_manager(STM_SRC)
    spec = spec_cls()
    spec.block_size, spec.mamba_cache_mode, spec.num_speculative_blocks = MAMBA, "align", 7
    pool = Pool(num_blocks=128)
    manager = manager_cls(spec, block_pool=pool, enable_caching=True, kv_cache_group_id=2, scheduler_block_size=MAMBA)
    split = split_function(src)
    sched = Sched(budget=7168, block_size=MAMBA, backoff=False)
    req = request(30719)
    req.request_id, req.block_hashes = "mixed", []
    state_positions, checkpoints = {}, []
    original_cache = pool.cache_full_blocks

    def cache(request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id, block_mask):
        for index in range(num_cached_blocks, num_full_blocks):
            if not blocks[index].is_null and (block_mask is None or block_mask[index - num_cached_blocks]):
                checkpoints.append(((index + 1) * block_size, state_positions[blocks[index].block_id]))
        return original_cache(request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id, block_mask)

    pool.cache_full_blocks = cache
    for allowance in (896, 1792, 1792):
        sched._glm53_align_prefill_limit = allowance
        end = req.num_computed_tokens + split(sched, req, allowance)
        manager.new_step_starts()
        manager.remove_skipped_blocks(req.request_id, req.num_computed_tokens, num_prompt_tokens=req.num_prompt_tokens)
        manager.allocate_new_blocks(req.request_id, end, end)
        state_block = manager.req_to_blocks[req.request_id][(end - 1) // MAMBA]
        state_positions[state_block.block_id] = end
        manager.cache_blocks(req, end)
        req.num_computed_tokens = end
    check(checkpoints == [(3584, 3584)], "C1 mixed caps hash the state after 3584 tokens, not the prior state after 2688")


def main() -> int:
    for var, path in (("GLM53_SCHEDULER_PY", SRC), ("GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY", STM_SRC)):
        if not path.is_file():
            raise SystemExit(f"missing {path} (set {var})")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "scheduler.py"
        shutil.copyfile(SRC, src)
        decode_floor = apply(src, PATCH.with_name("patch_scheduler_decode_floor.py"))
        if decode_floor.returncode:
            raise RuntimeError(decode_floor.stderr)
        part_a(src)
        part_b(src)
        part_c(src)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:\n  " + "\n  ".join(FAILURES))
        return 1
    print("mamba align chunking patch OK")
    return 0


def test_mamba_align_chunking() -> None:
    import pytest

    for var, path in (("GLM53_SCHEDULER_PY", SRC), ("GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY", STM_SRC)):
        if not path.is_file():
            pytest.skip(f"missing {path}; set {var}")
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
