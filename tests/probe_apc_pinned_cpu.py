#!/usr/bin/env python3
"""CPU-only investigation of the Dockerfile-pinned APC implementation.

Explicit invocation only (not collected by pytest):
  python3 tests/probe_apc_pinned_cpu.py --source-root /path/to/pristine/vllm \
      --manifest /path/to/six-apc-source-manifest.json --output /tmp/apc.json

Uses complete real cache modules, real patchers, and the AST of the real
scheduler method. Only import-only GPU/config/metrics dependencies and Request
input data are stubbed. No scheduler/cache/hash/retention algorithm is copied.
Exit 0: contracts passed, 1: observed defect/contract failure, 2: harness error.
The JSON distinguishes observations from contract failures and harness errors.
It does NOT establish GPU state-copy correctness, logits, or TTFT.
"""
from __future__ import annotations

import argparse
import ast
import enum
import hashlib
import importlib.util
import json
import logging
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
import typing
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PIN = "sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce"
PATCHES = (
    "patch_scheduler_decode_floor.py",
    "patch_mamba_align_chunking.py",
    "patch_mamba_align_state_free.py",
    "patch_hybrid_prefix_hit.py",
    "patch_apc_per_group_retention.py",
    "patch_apc_no_store.py",
)
TARGETS = {
    "GLM53_SCHEDULER_PY": "v1/core/sched/scheduler.py",
    "GLM53_KV_COORDINATOR_PY": "v1/core/kv_cache_coordinator.py",
    "GLM53_SINGLE_TYPE_KV_CACHE_MANAGER_PY": "v1/core/single_type_kv_cache_manager.py",
    "GLM53_KV_CACHE_INTERFACE_PY": "v1/kv_cache_interface.py",
    "GLM53_BLOCK_POOL_PY": "v1/core/block_pool.py",
    "GLM53_SAMPLING_PARAMS_PY": "sampling_params.py",
    "GLM53_REQUEST_PY": "v1/request.py",
}
REPORT: dict = {"checks": [], "observations": [], "defects": [], "errors": []}


def check(condition, label, **details):
    REPORT["checks"].append({"pass": bool(condition), "label": label, **details})
    print(("PASS " if condition else "FAIL ") + label, flush=True)


def observe(label, **details):
    REPORT["observations"].append({"label": label, **details})
    print("OBS " + label + " " + json.dumps(details, default=str), flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_sources(source, manifest_path, stage):
    manifest = json.loads(manifest_path.read_text())
    if not manifest["image"].endswith("@" + PIN):
        raise ValueError("Manifest does not identify the Dockerfile-pinned image")
    if PIN not in (REPO / "Dockerfile").read_text():
        raise ValueError("Dockerfile image changed; acquire a new explicit source pin")
    for rel, digest in manifest["files"].items():
        if sha(source / rel) != digest:
            raise ValueError(f"Pristine source digest mismatch: {rel}")
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, target)
    REPORT["source_provenance"] = manifest
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GLM53_", "VLLM_"))}
    env.update({key: str(stage / rel) for key, rel in TARGETS.items()})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    composed = []
    for name in PATCHES:
        patch = REPO / "overlay" / name
        result = subprocess.run([sys.executable, str(patch)], env=env,
                                capture_output=True, text=True)
        composed.append({"patch": name, "sha256": sha(patch),
                         "returncode": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        REPORT["composition"] = composed
        if result.returncode:
            raise RuntimeError(f"Composition failed at {name}: {result.stderr}")
    REPORT["composed_sha256"] = {rel: sha(stage / rel)
                                 for rel in manifest["files"]}
    for name in PATCHES:
        result = subprocess.run([sys.executable, str(REPO / "overlay" / name)],
                                env=env, capture_output=True, text=True)
        check(result.returncode == 0, "composed patch reapplication succeeds", patch=name)
        check(all(sha(stage / rel) == value
                  for rel, value in REPORT["composed_sha256"].items()),
              "composed patch reapplication preserves bytes", patch=name)
    # A marked but altered target must fail without publishing a partial edit.
    for name, rel, old, new in (
        ("patch_mamba_align_chunking.py", TARGETS["GLM53_SCHEDULER_PY"],
         "block_size = self.block_size  # [glm53-mamba-align-chunking-v2]",
         "block_size = self.cache_config.block_size  # [glm53-mamba-align-chunking-v2]"),
        ("patch_mamba_align_chunking.py", TARGETS["GLM53_SCHEDULER_PY"],
         "if end < prefill_end and num_new_tokens >= block_size:",
         "if end < prefill_end:"),
        ("patch_mamba_align_chunking.py", TARGETS["GLM53_SCHEDULER_PY"],
         "for state_block in self.mamba_align_sub_block_sizes:",
         "for state_block in ():"),
        ("patch_mamba_align_chunking.py", TARGETS["GLM53_SCHEDULER_PY"],
         "if self.mamba_align_eagle_backoff:",
         "if self.use_eagle:"),
        ("patch_mamba_align_chunking.py", TARGETS["GLM53_SCHEDULER_PY"],
         "and group.kv_cache_spec.participates_in_prefix_caching",
         "and True"),
        ("patch_hybrid_prefix_hit.py", TARGETS["GLM53_KV_COORDINATOR_PY"],
         "if group.kv_cache_spec.participates_in_prefix_caching\n                and not",
         "if True\n                and not"),
        ("patch_hybrid_prefix_hit.py", TARGETS["GLM53_KV_COORDINATOR_PY"],
         "max_cache_hit_length + 1 - self.kpool_replay_tokens",
         "max_cache_hit_length + 1"),
    ):
        target = stage / rel
        original = target.read_text()
        if original.count(old) != 1:
            raise AssertionError(f"Drift fixture anchor missing: {old}")
        drifted = original.replace(old, new, 1)
        try:
            target.write_text(drifted)
            result = subprocess.run([sys.executable, str(REPO / "overlay" / name)],
                                    env=env, capture_output=True, text=True)
            check(result.returncode != 0 and target.read_text() == drifted,
                  "marked target drift fails closed without modifying file", patch=name)
        finally:
            target.write_text(original)
    # Migrate the pre-fix overlays on an otherwise fully composed image.
    for name, rel, prefixes in (
        ("patch_mamba_align_chunking.py", TARGETS["GLM53_SCHEDULER_PY"],
         ("INIT", "SPLIT", "GRANT", "STOP")),
        ("patch_hybrid_prefix_hit.py", TARGETS["GLM53_KV_COORDINATOR_PY"],
         ("CAPABILITY", "KPOOL_INIT", "KPOOL_HIT", "COARSE_RETRY")),
    ):
        spec = importlib.util.spec_from_file_location("_apc_migration_patch", REPO / "overlay" / name)
        patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch)
        target = stage / rel
        original = target.read_text()
        previous = original
        for prefix in prefixes:
            new = getattr(patch, prefix + "_NEW")
            old = getattr(patch, prefix + "_V1", getattr(patch, prefix + "_OLD"))
            if previous.count(new) != 1:
                raise AssertionError(f"Missing migration anchor {prefix}")
            previous = previous.replace(new, old, 1)
        try:
            target.write_text(previous)
            result = subprocess.run([sys.executable, str(REPO / "overlay" / name)],
                                    env=env, capture_output=True, text=True)
            check(result.returncode == 0 and target.read_text() == original,
                  "pre-fix composed image migrates to identical final behavior", patch=name)
        finally:
            target.write_text(original)


def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    sys.modules[name] = result
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(sys.modules[parent], child, result)
    return result


def forbidden(*args, **kwargs):
    raise AssertionError("Probe entered a GPU, plugin, metrics, or unsupported input path")


class ForbiddenDependency:
    def __init__(self, *args, **kwargs):
        forbidden(*args, **kwargs)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    parent, child = name.rsplit(".", 1)
    setattr(sys.modules[parent], child, result)
    return result


def method(path, cls, name):
    tree = ast.parse(path.read_text(), filename=str(path))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    # Preserve the entire function body, filename and line numbers; only postpone
    # annotations so importing the GPU scheduler is unnecessary.
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(unit)
    ns = {}
    exec(compile(unit, str(path), "exec"), ns)
    return ns[name]


def cpu_modules(stage):
    # Package placeholders prevent vLLM's __init__ from loading torch/CUDA.
    packages = ("vllm", "vllm.utils", "vllm.v1", "vllm.v1.core",
                "vllm.v1.attention", "vllm.v1.attention.backends",
                "vllm.v1.metrics", "vllm.distributed")
    for name in packages:
        module(name, __path__=[])
    logger = logging.getLogger("apc-cpu")
    logger.info_once = logger.info
    logger.warning_once = logger.warning
    logger.debug_once = logger.debug
    module("vllm.logger", init_logger=lambda name: logger)
    envs = module("vllm.envs", VLLM_PREFIX_CACHE_RETENTION_INTERVAL=None,
                  VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=False)
    module("vllm.config", VllmConfig=types.SimpleNamespace,
           get_current_vllm_config_or_none=lambda: None)
    module("vllm.platforms", current_platform=types.SimpleNamespace(
        register_custom_kv_cache_specs=lambda config: None))
    # Dtypes are inert labels: no tensor or device allocation is possible.
    torch = module("torch", dtype=str, float32="float32", float16="float16",
                   bfloat16="bfloat16", uint8="uint8")
    module("typing_extensions", Self=typing.Self)
    module("vllm.utils.torch_utils", get_dtype_size=lambda dtype: {
        "float32": 4, "float16": 2, "bfloat16": 2, "uint8": 1}[dtype],
        nvfp4_kv_cache_full_dim=forbidden)
    module("vllm.utils.mem_utils", format_gib=forbidden)
    # Actual request chaining calls the supplied sha256 callback below. CBOR
    # functions are unavailable and deliberately fail if called.
    module("vllm.utils.hashing", sha256_cbor=forbidden, xxhash_cbor=forbidden)
    module("vllm.v1.utils", tensor_data=forbidden)
    module("vllm.v1.attention.backends.registry",
           MambaAttentionBackendEnum=enum.Enum("MambaAttentionBackendEnum", ["MAMBA2"]))
    module("vllm.v1.core.kv_cache_metrics", KVCacheMetricsCollector=object)
    module("vllm.v1.metrics.stats", PrefixCacheStats=ForbiddenDependency)
    module("vllm.distributed.kv_events", MEDIUM_GPU="GPU",
           AllBlocksCleared=ForbiddenDependency, BlockRemoved=ForbiddenDependency,
           BlockStored=ForbiddenDependency, KVCacheEvent=object)
    statuses = enum.Enum("RequestStatus", ["WAITING", "RUNNING", "PREEMPTED"])
    module("vllm.v1.request", Request=types.SimpleNamespace, RequestStatus=statuses)
    load("vllm.utils.math_utils", stage / "utils/math_utils.py")
    load("vllm.v1.kv_cache_spec_registry", stage / "v1/kv_cache_spec_registry.py")
    specs = load("vllm.v1.kv_cache_interface", stage / "v1/kv_cache_interface.py")
    utils = load("vllm.v1.core.kv_cache_utils", stage / "v1/core/kv_cache_utils.py")
    pool = load("vllm.v1.core.block_pool", stage / "v1/core/block_pool.py")
    stm = load("vllm.v1.core.single_type_kv_cache_manager",
               stage / "v1/core/single_type_kv_cache_manager.py")
    coord = load("vllm.v1.core.kv_cache_coordinator", stage / "v1/core/kv_cache_coordinator.py")
    manager = load("vllm.v1.core.kv_cache_manager", stage / "v1/core/kv_cache_manager.py")
    sampling = module("vllm.sampling_params", logger=logger)
    text = (stage / "sampling_params.py").read_text()
    begin = text.index("# [glm53-apc-no-store] helper-begin")
    end = text.index("# [glm53-apc-no-store] helper-end", begin)
    exec(compile(text[begin:end], str(stage / "sampling_params.py"), "exec"),
         sampling.__dict__)
    resolve_no_store = method(stage / "v1/request.py", "Request",
                             "get_skip_writing_prefix_cache")
    utils.init_none_hash(digest)
    return types.SimpleNamespace(specs=specs, utils=utils, pool=pool, stm=stm,
        coord=coord, manager=manager, envs=envs, statuses=statuses,
        resolve_no_store=resolve_no_store, torch=torch)


def digest(value):
    # Standard-library hash backend supplied to the pinned request hasher.
    return hashlib.sha256(pickle.dumps(value, protocol=4)).digest()


def request(runtime, rid, tokens, no_store=False):
    tokens = list(tokens)
    req = types.SimpleNamespace(request_id=rid, all_token_ids=tokens,
        prompt_token_ids=tokens, num_tokens=len(tokens), num_prompt_tokens=len(tokens),
        num_computed_tokens=0, num_in_flight_tokens=0, shared_prefix_boundary=0,
        block_hashes=[], lora_request=None, mm_features=[], prompt_embeds=None,
        cache_salt=None, skip_reading_prefix_cache=False,
        sampling_params=types.SimpleNamespace(skip_writing_prefix_cache=no_store,
                                              extra_args=None),
        status=runtime.statuses.WAITING, num_preemptions=0)
    req.skip_writing_prefix_cache = runtime.resolve_no_store(req)
    req.block_hashes.extend(runtime.utils.get_request_block_hasher(64, digest)(req))
    return req


def layout(runtime, *, scratch=True, draft=64, mamba=(3584,) * 4):
    s = runtime.specs
    def attn(cls, block, **kwargs):
        return cls(block_size=block, num_kv_heads=1, head_size=1,
                   dtype=runtime.torch.float32, **kwargs)
    groups = [s.KVCacheGroupSpec(["mla"], attn(s.MLAAttentionSpec, 3584))]
    if scratch:
        groups.append(s.KVCacheGroupSpec(["kpool"],
                      attn(s.KpoolTailSpec, 4, sliding_window=4)))
    for idx, block in enumerate(mamba):
        groups.append(s.KVCacheGroupSpec([f"kda{idx}"], s.MambaSpec(
            block_size=block, shapes=((1, 1),), dtypes=(runtime.torch.float32,),
            mamba_cache_mode="align")))
    if draft:
        groups.append(s.KVCacheGroupSpec(["draft"],
                      attn(s.SlidingWindowSpec, draft, sliding_window=2048)))
    return s.KVCacheConfig(num_blocks=4096, kv_cache_tensors=[], kv_cache_groups=groups)


def geometry(runtime, stage, cfg, prefix_match_unit=None):
    config = types.SimpleNamespace(cache_config=types.SimpleNamespace(
        block_size=3584, enable_prefix_caching=True, prefix_match_unit=prefix_match_unit),
        parallel_config=types.SimpleNamespace(decode_context_parallel_size=1),
        kv_transfer_config=None)
    # Execute the exact engine assignments which set the post-grouping global
    # block size. Do not reproduce their min/participation logic in this probe.
    tree = ast.parse((stage / "v1/engine/core.py").read_text())
    owners = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and isinstance(node.test, ast.Name) and node.test.id == "kv_cache_groups"]
    owner = next(node for node in owners if any(isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "block_size" for t in n.targets)
        for n in node.body))
    assigns = [n for n in owner.body if isinstance(n, ast.Assign)]
    ns = {"vllm_config": config, "kv_cache_groups": cfg.kv_cache_groups}
    exec(compile(ast.Module(body=assigns, type_ignores=[]),
                 str(stage / "v1/engine/core.py"), "exec"), ns)
    scheduler, hash_size = runtime.utils.resolve_kv_cache_block_sizes(cfg, config)
    return config.cache_config.block_size, scheduler, hash_size


def new_manager(runtime, cfg, scheduler, global_retention=None, swa=None):
    runtime.envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL = global_retention
    if swa is None:
        os.environ.pop("VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA", None)
    else:
        os.environ["VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA"] = str(swa)
    return runtime.manager.KVCacheManager(cfg, max_model_len=32768,
        scheduler_block_size=scheduler, hash_block_size=64, enable_caching=True,
        use_eagle=True, max_in_flight_tokens=4096)


def scheduler_state(stage, cfg, manager, global_block, scheduler, hash_size=64):
    """Execute the dedicated checkpoint overlay's actual init assignments."""
    sched = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=global_block),
        block_size=scheduler, hash_block_size=hash_size, max_num_scheduled_tokens=7168,
        scheduler_config=types.SimpleNamespace(long_prefill_token_threshold=0),
        use_eagle=True, kv_cache_manager=manager, _glm53_align_prefill_limit=None,
        mamba_partial_cache_hit=manager.coordinator.enable_partial_hash_hits)
    path = stage / "v1/core/sched/scheduler.py"
    tree = ast.parse(path.read_text())
    names = {"mamba_align_sub_block_sizes", "mamba_align_eagle_backoff"}
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr in names for t in n.targets)]
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"),
         {"self": sched, "kv_cache_config": cfg,
          "MambaSpec": sys.modules["vllm.v1.kv_cache_interface"].MambaSpec})
    return sched

def alignment_probe(runtime, stage):
    split = method(stage / "v1/core/sched/scheduler.py", "Scheduler",
                   "_mamba_block_aligned_split")
    cfg = layout(runtime)
    check(geometry(runtime, stage, cfg) == geometry(runtime, stage, cfg, 64),
          "empty prefix unit resolves gcd64 identically to explicit64")
    examples = []
    for blocks in ((3584,) * 4, (896, 1792, 3584, 896)):
        cfg = layout(runtime, mamba=blocks)
        global_block, scheduler, hash_size = geometry(runtime, stage, cfg)
        check((global_block, scheduler, hash_size) == (64, 3584, 64),
              "real engine and resolver produce min64/LCM3584/hash64", mamba=blocks)
        manager = new_manager(runtime, cfg, scheduler)
        for budget in (2048, 4096, 7168, 8192):
            for start, count in (
                (start, count)
                for start in (0, 100, 2048, 3520, 3584, 5632, 7104)
                for count in (128, 256, 512, 1024, 2048, 4000, 5000, 7168, 8000)
            ):
                req = request(runtime, "align", range(32000))
                req.num_computed_tokens = start
                sched = scheduler_state(stage, cfg, manager, global_block, scheduler)
                sched.max_num_scheduled_tokens = budget
                sched.mamba_partial_cache_hit = False
                got = split(sched, req, count)
                end = start + got
                # A positive current grant can advance private state, regardless
                # of the configured budget. No existing partial Mamba slot may
                # be left behind while crossing its own state-page boundary.
                crossed = any(
                    start % b and start < (start // b + 1) * b < end for b in blocks)
                legal = not crossed and (
                    end % scheduler == 0 or
                    (start % scheduler != 0 and
                     end <= (start // scheduler + 1) * scheduler) or
                    (start % scheduler == 0 and got < scheduler))
                row = dict(global_block=global_block, scheduler_block=scheduler,
                           mamba_blocks=blocks, budget=budget, start=start,
                           proposed=count, returned=got, end=end, legal=legal,
                           crossed_checkpoints=crossed)
                examples.append(row)
                check(0 < got <= count, "positive grant makes bounded progress", **row)
                check(legal, "intermediate chunks respect shared state checkpoints", **row)
                if start % scheduler == 0 and count >= scheduler:
                    check(end % scheduler == 0,
                          "aligned start with above-page grant ends on checkpoint", **row)
                if not legal:
                    REPORT["defects"].append({"claim": "A", **row})
        # Decode and cached/external offsets must not accidentally become prefill.
        req = request(runtime, "decode", range(129))
        req.num_computed_tokens = 129
        check(split(sched, req, 8) == 8, "decode bypass preserves eight scheduled tokens")
        req.num_computed_tokens = 0
        check(split(sched, req, 8, 64, 65) == 8,
              "local plus external cached offsets reach the decode bypass")
        # Last prompt hash boundary: allow a fine tail only with partial hits enabled.
        req = request(runtime, "tail", range(7745))
        req.num_computed_tokens = 7680
        sched.mamba_partial_cache_hit = True
        tail = split(sched, req, 65)
        observe("last-prompt-hash-boundary", start=7680, prompt=7745,
                returned=tail, hash_boundary=7744, scheduler_block=scheduler)
        check(tail == 64, "fine-grained final prompt stop preserves hash64 boundary")
        for cap_kind in ("scheduler", "threshold", "fair", "remaining"):
            req = request(runtime, "progress", range(7745))
            sched.max_num_scheduled_tokens = 2048 if cap_kind == "scheduler" else 4096
            sched.scheduler_config.long_prefill_token_threshold = (
                2048 if cap_kind == "threshold" else 0)
            sched._glm53_align_prefill_limit = 2048 if cap_kind == "fair" else None
            trace = []
            cache = new_manager(runtime, cfg, scheduler)
            while req.num_computed_tokens < req.num_tokens:
                proposed = min(2048, req.num_tokens - req.num_computed_tokens)
                got = split(sched, req, proposed)
                check(0 < got <= proposed,
                      "intentional or remaining-budget cap makes bounded progress",
                      cap=cap_kind, start=req.num_computed_tokens, returned=got)
                if got <= 0 or got > proposed:
                    break
                check(not any(
                    req.num_computed_tokens % b
                    and (req.num_computed_tokens // b + 1) * b
                    < req.num_computed_tokens + got for b in blocks),
                    "progress never crosses a private Mamba state's page boundary",
                    start=req.num_computed_tokens, returned=got, mamba=blocks)
                cache.coordinator.new_step_starts()
                check(cache.allocate_slots(req, got) is not None,
                      "actual cache allocator accepts every scheduler progress step",
                      cap=cap_kind, start=req.num_computed_tokens, scheduled=got)
                req.num_computed_tokens += got
                trace.append(req.num_computed_tokens)
            check(all(boundary in trace for boundary in (3584, 7168, 7744, 7745)),
                  "chunked prefill visits shared checkpoints and exact final hash tail",
                  cap=cap_kind, trace=trace, mamba=blocks)
            cache.free(req)
            cache.coordinator.new_step_starts()
            _, hit, _ = cache.get_computed_blocks(
                request(runtime, "progress-reuse", range(7745)))
            check(hit >= 7168, "scheduler-driven prime retains a reusable exact checkpoint",
                  cap=cap_kind, hit=hit, mamba=blocks)
    observe("claim-A-real-scheduler-results", cases=examples)


def capability_probe(runtime, stage):
    rows = []
    for scratch, draft in ((False, 64), (True, 64), (False, 128), (True, 128)):
        cfg = layout(runtime, scratch=scratch, draft=draft)
        global_block, scheduler, hash_size = geometry(runtime, stage, cfg, 64)
        manager = new_manager(runtime, cfg, scheduler)
        coord = manager.coordinator
        facts = [dict(manager=type(m).__name__, block=m.block_size,
                      capable=m.supports_fine_grained_hash_lookup,
                      participates=g.kv_cache_spec.participates_in_prefix_caching)
                 for m, g in zip(coord.single_type_managers, cfg.kv_cache_groups)]
        rows.append(dict(scratch=scratch, draft=draft, geometry=[global_block,scheduler,hash_size],
                         partial=coord.enable_partial_hash_hits, managers=facts))
        if draft == 64:
            check(coord.enable_partial_hash_hits,
                  "SWA64 permits hash64 and nonparticipating scratch cannot veto it")
        if draft == 128:
            check(not coord.enable_partial_hash_hits,
                  "unsupported participating SWA128 continues to veto hash64")
        if scratch and draft == 64 and not coord.enable_partial_hash_hits:
            REPORT["defects"].append({"claim": "B-alternative",
                "reason": "nonparticipating KpoolTail4 vetoes otherwise-compatible hash64",
                "managers": facts})
        check(coord.eagle_group_ids == {len(cfg.kv_cache_groups) - 1},
              "composed EAGLE fallback narrows to the actual drafter")
    observe("claim-B-real-coordinator-results", cases=rows)


def cache_contract_probe(runtime, stage):
    tokens = list(range(10000, 17745))  # prompt 7745; last hash64 boundary 7744
    prime = request(runtime, "hash-prime", tokens)
    check(len(prime.block_hashes) == 121, "hash64 uses 121 full units, not an incomplete tail")
    for changed in (63, 64, 895, 896, 3583, 3584, 7743, 7744):
        edited = tokens.copy()
        edited[changed] += 99999
        query = request(runtime, "hash-edit", edited)
        prefix_units = changed // 64
        check(query.block_hashes[:prefix_units] == prime.block_hashes[:prefix_units],
              "token edit preserves only preceding complete hashes", changed=changed)
        if prefix_units < len(prime.block_hashes):
            check(all(a != b for a, b in zip(query.block_hashes[prefix_units:],
                      prime.block_hashes[prefix_units:])),
                  "chained hashes reject changed cached suffix", changed=changed)

    # Exercise real allocation/store/free/lookup, not a dictionary replica.
    # With scratch removed this additionally probes the already-supported fine
    # hit path; full seven-group runs remain the actual production-shape control.
    for scratch, mamba in ((False, (3584,) * 4), (True, (3584,) * 4),
                           (True, (896, 1792, 3584, 896))):
        for global_retention, swa in ((None, None), (0, 0), (14336, 14336)):
            cfg = layout(runtime, scratch=scratch, mamba=mamba)
            _, scheduler, _ = geometry(runtime, stage, cfg)
            manager = new_manager(runtime, cfg, scheduler, global_retention, swa)
            coord = manager.coordinator
            seed = request(runtime, "seed", tokens)
            # Every artificial compute step finishes at a real Mamba checkpoint,
            # then at the documented fine tail and final logits token. This
            # isolates cache correctness from the separately tested scheduler.
            ends = [3584, 7168, 7744, 7745]
            for end in ends:
                coord.new_step_starts()
                out = manager.allocate_slots(seed, end - seed.num_computed_tokens)
                check(out is not None, "CPU prime allocation succeeds", end=end)
                if out is None:
                    raise AssertionError("Unexpected CPU block-pool exhaustion")
                seed.num_computed_tokens = end
            coord.new_step_starts()
            manager.free(seed)
            coord.new_step_starts()
            for name, query_tokens, divergence in (
                ("identical", tokens, len(tokens)),
                ("append", tokens + [88, 89, 90], len(tokens)),
                ("changed-middle", tokens[:4000] + [999999] + tokens[4001:], 4000),
                ("short-branch", tokens[:3584] + [999999] * 2200, 3584),
            ):
                query = request(runtime, name, query_tokens, no_store=True)
                blocks, hit, junction = manager.get_computed_blocks(query)
                if name in ("identical", "append"):
                    expected_hit = 3584 if global_retention == 0 else 7168
                    check(hit >= expected_hit,
                          "fine lookup preserves the available coarse replay checkpoint",
                          query=name, hit=hit, retention=global_retention, swa=swa)
                _, ordinary_hit, _ = manager.get_computed_blocks(
                    request(runtime, "ordinary-" + name, query_tokens))
                check(hit == ordinary_hit, "no-store requests retain ordinary read-side hits")
                check(0 <= hit <= min(divergence, len(query_tokens) - 1),
                      "cached prefix cannot include changed suffix or logits token",
                      query=name, hit=hit, divergence=divergence)
                check(hit % coord._cache_hit_alignment_tokens == 0,
                      "returned hit obeys active alignment", hit=hit)
                check(query_tokens[:hit] == tokens[:hit],
                      "cached token prefix equals the producer prefix", query=name, hit=hit)
                # Host-side exact token partition, not a claim about GPU logits.
                check(tokens[:hit] + query_tokens[hit:] == query_tokens,
                      "cached prefix plus replay suffix reconstructs all query tokens")
                if hit:
                    draft = blocks.blocks[-1]
                    tail_count = math.ceil((2048 - 1) / 64)
                    complete_draft = (len(draft) == hit // 64 and len(draft) >= tail_count
                                      and all(not b.is_null for b in draft[-tail_count:]))
                    check(complete_draft or len(query_tokens) - hit >= 2048,
                          "short suffix requires complete retained EAGLE draft window",
                          query=name, hit=hit, fresh=len(query_tokens)-hit,
                          complete_draft=complete_draft)
                    if scratch:
                        check(not blocks.blocks[1], "Kpool scratch is never a reusable hit")
                        check(len(query_tokens) - hit >= 4,
                              "request-local Kpool suffix has at least four fresh tokens")
                observe("cache-lookup", scratch=scratch, retention=global_retention,
                        swa=swa, mamba=mamba, query=name, hit=hit, fresh=len(query_tokens)-hit,
                        junction=junction,
                        partial=coord.enable_partial_hash_hits,
                        group_block_counts=[len(g) for g in blocks.blocks])
                coord.new_step_starts()
                out = manager.allocate_slots(
                    query, len(query_tokens) - hit,
                    num_new_computed_tokens=hit, new_computed_blocks=blocks)
                check(out is not None, "cached prefix plus fresh suffix can allocate legally",
                      query=name, hit=hit)
                if out is None:
                    raise AssertionError("Unexpected warm continuation block-pool exhaustion")
                query.num_computed_tokens = len(query_tokens)
                manager.free(query)
                coord.new_step_starts()
            cold_tokens = list(range(90000, 97745))
            cold = request(runtime, "cold-no-store", cold_tokens, no_store=True)
            for end in ends:
                coord.new_step_starts()
                out = manager.allocate_slots(cold, end - cold.num_computed_tokens)
                check(out is not None, "no-store allocation remains available")
                if out is None:
                    raise AssertionError("Unexpected no-store block-pool exhaustion")
                cold.num_computed_tokens = end
            manager.free(cold)
            coord.new_step_starts()
            _, hit, _ = manager.get_computed_blocks(request(runtime, "cold-retry", cold_tokens))
            check(hit == 0, "no-store composed path publishes neither full nor partial hits")

    # Explicit legal/illegal SWA alignment and retained EAGLE lookahead. Use the
    # real pool's cache_full_blocks plus real mask and lookup functions.
    s = runtime.specs
    spec = s.SlidingWindowSpec(block_size=64, num_kv_heads=1, head_size=1,
                              dtype=runtime.torch.float32, sliding_window=2048)
    for retention in (None, 0, 14336):
        pool = runtime.pool.BlockPool(num_gpu_blocks=256, enable_caching=True, hash_block_size=64)
        req = request(runtime, "swa", range(8192))
        blocks = pool.get_new_blocks(128)
        mask = runtime.stm.SlidingWindowManager.reachable_block_mask(
            0, 128, 64, spec, True, retention, (7168,))
        pool.cache_full_blocks(req, blocks, 0, 128, 64, 0, block_mask=mask)
        hits, length = runtime.stm.SlidingWindowManager.find_longest_cache_hit(
            req.block_hashes, 7232, [0], pool, spec, True, 64)
        check(length == 7168 and all(not b.is_null for b in hits[0][-32:]),
              "SWA retention preserves complete window after one EAGLE pop",
              retention=retention, hit=length)
        illegal = s.SlidingWindowSpec(block_size=128, num_kv_heads=1, head_size=1,
                                    dtype=runtime.torch.float32, sliding_window=2048)
        try:
            runtime.stm.SlidingWindowManager.find_longest_cache_hit(
                req.block_hashes, 7232, [0], pool, illegal, True, 64)
        except AssertionError:
            check(True, "SWA128 rejects illegal hash64 lookup instead of class-name exemption")
        else:
            check(False, "SWA128 must reject an unsupported partial lookup")


def partial_tail_probe(runtime, stage):
    """Actual partial entry creation, CoW metadata, and EAGLE replay boundaries."""
    tokens = list(range(7745))
    for draft in (0, 64):
        cfg = layout(runtime, draft=draft)
        _, scheduler, _ = geometry(runtime, stage, cfg, 64)
        runtime.envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL = None
        os.environ.pop("VLLM_PREFIX_CACHE_RETENTION_INTERVAL_SWA", None)
        manager = runtime.manager.KVCacheManager(
            cfg, max_model_len=32768, scheduler_block_size=scheduler,
            hash_block_size=64, enable_caching=True, use_eagle=bool(draft))
        coord = manager.coordinator
        seed = request(runtime, "partial-seed", tokens)
        for end in (3584, 7168, 7744, 7745):
            coord.new_step_starts()
            assert manager.allocate_slots(seed, end - seed.num_computed_tokens) is not None
            seed.num_computed_tokens = end
        # Produce a real hash64 lookahead block after the original prompt tail.
        seed.all_token_ids.extend(range(7745, 7809))
        seed.num_tokens = len(seed.all_token_ids)
        seed.block_hashes.extend(runtime.utils.get_request_block_hasher(64, digest)(seed))
        coord.new_step_starts()
        assert manager.allocate_slots(seed, 64) is not None
        seed.num_computed_tokens += 64
        coord.new_step_starts()
        manager.free(seed)
        coord.new_step_starts()
        for length in (7745, 7746, 7747, 7748, 7809, 9792):
            query = request(runtime, "partial-query", range(length), no_store=True)
            blocks, hit, _ = manager.get_computed_blocks(query)
            check(length - hit >= 4, "partial entry cannot consume the Kpool replay suffix",
                  draft=draft, query_length=length, hit=hit)
            # Dense SWA retention still keeps coarse checkpoint windows; do not
            # expand it implicitly merely because target partial hits are legal.
            # A full fresh window makes that partial target boundary reusable.
            if length == 9792 or (not draft and length >= 7748):
                check(hit == 7744, "safe fine tail is reused rather than globally disabled",
                      draft=draft, query_length=length, hit=hit)
                for gid in range(2, 6):
                    tail = blocks.blocks[gid][-1]
                    check(tail.block_hash_num_tokens == hit,
                          "Mamba partial hit identifies the exact saved token boundary",
                          group=gid, hit=hit, stored=tail.block_hash_num_tokens)
                    check(runtime.utils.get_block_hash(tail.block_hash)
                          == query.block_hashes[hit // 64 - 1],
                          "Mamba partial state hash matches query prefix, not later suffix")
            check(hit <= length - 1 and hit % 64 == 0,
                  "fine tail respects final-token recomputation and hash granularity")
            coord.new_step_starts()
            assert manager.allocate_slots(query, length - hit,
                num_new_computed_tokens=hit, new_computed_blocks=blocks) is not None
            manager.free(query)
            coord.new_step_starts()
            observe("partial-tail-reuse", draft=draft, query_length=length, hit=hit)


def production_probe(runtime, stage):
    """Real production-grant splitter -> allocator -> free -> short-suffix hit."""
    split = method(stage / "v1/core/sched/scheduler.py", "Scheduler",
                   "_mamba_block_aligned_split")
    for prefix_unit in (None, 64):
        for retention in (None, 14336):
            cfg = layout(runtime)
            global_block, scheduler, hash_size = geometry(runtime, stage, cfg, prefix_unit)
            for length in range(3600, 32001, 448):
                cache = new_manager(runtime, cfg, scheduler, retention, retention)
                sched = scheduler_state(stage, cfg, cache, global_block, scheduler, hash_size)
                tokens = list(range(10000, 10000 + length))
                seed = request(runtime, "production-prime", tokens)
                trace = []
                while seed.num_computed_tokens < length:
                    grant = min(7168, length - seed.num_computed_tokens)
                    got = split(sched, seed, grant)
                    assert 0 < got <= grant
                    cache.coordinator.new_step_starts()
                    assert cache.allocate_slots(seed, got) is not None
                    seed.num_computed_tokens += got
                    trace.append(seed.num_computed_tokens)
                cache.coordinator.new_step_starts()
                cache.free(seed)
                cache.coordinator.new_step_starts()
                _, hit, _ = cache.get_computed_blocks(request(runtime, "production-followup",
                    tokens + list(range(500000, 500300))))
                checkpoint = (length - 1) // scheduler * scheduler
                check(checkpoint in trace,
                      "production grant preserves last full target checkpoint",
                      length=length, trace=trace, prefix_unit=prefix_unit, retention=retention)
                # Large grants can skip intermediate states; sparse retention
                # can then push draft replay to the preceding retained interval.
                # Exact comparison to main is an external A/B run.
                guard = max(7168, retention or 0)
                check(hit >= max(0, checkpoint - guard),
                      "production short suffix retains replay-safe checkpoint",
                      length=length, hit=hit, checkpoint=checkpoint)
                observe("production-short-suffix", length=length, trace=trace, hit=hit,
                        prefix_unit=prefix_unit, retention=retention)

    # No-SWA MTP must keep the upstream one-page scheduler backoff.
    cfg = layout(runtime, draft=0)
    global_block, scheduler, hash_size = geometry(runtime, stage, cfg, 64)
    cache = new_manager(runtime, cfg, scheduler)
    sched = scheduler_state(stage, cfg, cache, global_block, scheduler, hash_size)
    req = request(runtime, "mtp", range(12033))
    req.num_computed_tokens = 3584
    check(split(sched, req, 8449) == 3584,
          "no-SWA EAGLE preserves upstream mandatory backoff checkpoint")
    # Explicit group annotations take precedence over the SWA fallback.
    # Scratch never requires a target backoff; an EAGLE MLA or Mamba does.
    for group_id, expected_end in ((1, 10752), (0, 7168), (2, 7168)):
        cfg = layout(runtime)
        cfg.kv_cache_groups[group_id].is_eagle_group = True
        global_block, scheduler, hash_size = geometry(runtime, stage, cfg)
        cache = new_manager(runtime, cfg, scheduler)
        sched = scheduler_state(stage, cfg, cache, global_block, scheduler, hash_size)
        req = request(runtime, "annotated-eagle", range(12033))
        req.num_computed_tokens = 3584
        check(3584 + split(sched, req, 8449) == expected_end,
              "only participating non-SWA EAGLE needs target checkpoint backoff",
              eagle_group=group_id, expected_end=expected_end)


def compact_boundary_probe(runtime, stage):
    """Real retained compact-window lookup still reserves fresh target scratch."""
    cfg = layout(runtime, draft=896)
    _, scheduler, _ = geometry(runtime, stage, cfg, 64)
    cache = new_manager(runtime, cfg, scheduler, swa=0)
    seed = request(runtime, "compact-prime", range(7517))
    for end in (3584, 7168, 7517):
        cache.coordinator.new_step_starts()
        assert cache.allocate_slots(seed, end - seed.num_computed_tokens) is not None
        seed.num_computed_tokens = end
    cache.free(seed)
    cache.coordinator.new_step_starts()
    compact = os.environ["GLM53_DRAFT_KV_COMPACT"] == "1"
    for suffix in (1, 2, 3, 4, 349):
        query = request(runtime, "compact-followup", range(7168 + suffix))
        blocks, hit, _ = cache.get_computed_blocks(query)
        expected = 7168 if compact and suffix >= 4 else 3584
        check(hit == expected,
              "compact boundary window and independent scratch floor reconcile",
              compact=compact, suffix=suffix, hit=hit, expected=expected)
        check(query.num_tokens - hit >= 4,
              "retained compact drafter never replaces fresh target scratch")
        if hit == 7168:
            draft = blocks.blocks[-1]
            check(len(draft) == 8 and all(not b.is_null for b in draft[-3:]),
                  "short replay uses the complete retained compact draft window")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact", choices=("0", "1"), default="0")
    args = parser.parse_args()
    if sys.flags.optimize:
        raise SystemExit("Do not run with -O: upstream assertions are part of this probe")
    sys.dont_write_bytecode = True
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["GLM53_APC_NO_STORE"] = "1"
    os.environ["GLM53_DRAFT_KV_COMPACT"] = args.compact
    REPORT["compact_flag"] = args.compact
    REPORT["limitations"] = [
        "No tensor allocation, kernels, model forward, GPU copy, or service exercised",
        "Request input carrier and import-only dependencies are explicit stubs",
        "Hash backend is stdlib sha256/pickle; request chaining and block selection are pinned code",
        "Metadata/token consistency is not logits correctness or TTFT evidence",
        "Pinned manifest lacks worker/utils.py: compact allocator/backend composition is unsupported here",
        "Compact coordinator behavior uses explicit legal group geometry, not allocator or GPU evidence",
    ]
    with tempfile.TemporaryDirectory(prefix="glm53-apc-cpu-") as tmp:
        stage = Path(tmp)
        try:
            stage_sources(args.source_root, args.manifest, stage)
            runtime = cpu_modules(stage)
            for label, probe in (("alignment", alignment_probe),
                                 ("capability", capability_probe),
                                 ("cache_contract", cache_contract_probe),
                                 ("partial_tail", partial_tail_probe),
                                 ("compact_boundary", compact_boundary_probe),
                                 ("production", production_probe)):
                try:
                    probe(runtime, stage)
                except Exception:
                    REPORT["errors"].append({"probe": label, "traceback": traceback.format_exc()})
                    traceback.print_exc()
        except Exception:
            REPORT["errors"].append({"probe": "setup", "traceback": traceback.format_exc()})
            traceback.print_exc()
    failures = [row for row in REPORT["checks"] if not row["pass"]]
    code = 2 if REPORT["errors"] else (1 if failures or REPORT["defects"] else 0)
    REPORT["exit_code"] = code
    REPORT["summary"] = dict(checks=len(REPORT["checks"]), failures=len(failures),
                             defects=len(REPORT["defects"]), errors=len(REPORT["errors"]))
    args.output.write_text(json.dumps(REPORT, indent=2, default=str) + "\n")
    print(json.dumps(REPORT["summary"]), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
