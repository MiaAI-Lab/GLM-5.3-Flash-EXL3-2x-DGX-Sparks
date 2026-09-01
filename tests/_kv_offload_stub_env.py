#!/usr/bin/env python3
"""Stubbed-vllm exec environment for the kv-offload overlay tests.

Applies the overlay patchers to the pinned image fixtures
(tests/fixtures/image_487ecf187_*.py) and executes the PATCHED module text
under a minimal fake ``vllm`` namespace, so the tests drive the exact code the
container will run — not a replica (Codex OFFLOAD1 finding 6).

Only what the patched modules import is stubbed; everything else raises on
first touch. The 7-group GLM-5.3-Flash boot layout (stage-0 receipts) is the
canonical fake config: g0 mixed MLA+indexer (block 3584), g1 KpoolTail scratch
(block 4, not prefix-cacheable), g2-5 MambaSpec align (block 3584), g6 exact
SlidingWindowSpec drafter (block 64, EAGLE).
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIXTURES = HERE / "fixtures"

sys.path.insert(0, str(ROOT / "overlay"))

import patch_kv_offload_scope as scope_mod  # noqa: E402
import patch_kv_offload_store_local as store_mod  # noqa: E402
import patch_kv_offload_restore_g0 as restore_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Patched module text
# ---------------------------------------------------------------------------
def patched_texts(
    with_store: bool = True, with_restore: bool = False
) -> dict[str, str]:
    """Apply the patchers (prepare only, no writes) to the pinned fixtures."""
    kvo_cfg = (FIXTURES / "image_487ecf187_kv_offload_config.py").read_text()
    conn_cfg = (FIXTURES / "image_487ecf187_offloading_config.py").read_text()
    sched = (FIXTURES / "image_487ecf187_offloading_scheduler.py").read_text()
    common = (FIXTURES / "image_487ecf187_offloading_common.py").read_text()
    worker = (FIXTURES / "image_487ecf187_offloading_worker.py").read_text()

    kvo_cfg, _ = scope_mod.prepare(kvo_cfg, scope_mod.SITES_KVO_CONFIG, "t")
    conn_cfg, _ = scope_mod.prepare(conn_cfg, scope_mod.SITES_CONN_CONFIG, "t")
    sched, _ = scope_mod.prepare(sched, scope_mod.SITES_SCHED, "t")
    if with_store:
        common, _ = store_mod.prepare(common, store_mod.SITES_COMMON, "t")
        sched, _ = store_mod.prepare(sched, store_mod.SITES_SCHED, "t")
        worker, _ = store_mod.prepare(worker, store_mod.SITES_WORKER, "t")
    if with_restore:
        assert with_store, "the restore overlay stacks on the store overlay"
        common, _ = restore_mod.prepare(common, restore_mod.SITES_COMMON, "t")
        sched, _ = restore_mod.prepare(sched, restore_mod.SITES_SCHED, "t")
        worker, _ = restore_mod.prepare(worker, restore_mod.SITES_WORKER, "t")
    return {
        "kv_offload_config": kvo_cfg,
        "offloading_config": conn_cfg,
        "scheduler": sched,
        "common": common,
        "worker": worker,
    }


# ---------------------------------------------------------------------------
# Fake spec classes (names are load-bearing: the overlays key on them)
# ---------------------------------------------------------------------------
class KVCacheSpec:
    """[I] naming: the cacheability flag is the participates_in_prefix_caching
    PROPERTY (no prefix_cacheable attribute exists) — the stub mirrors that so
    the overlays' dual-name check is exercised against the fork name."""

    def __init__(self, block_size, prefix_cacheable=True):
        self.block_size = block_size
        self._cacheable = prefix_cacheable

    @property
    def participates_in_prefix_caching(self) -> bool:
        return self._cacheable


class AttentionSpec(KVCacheSpec):
    pass


class FullAttentionSpec(AttentionSpec):
    pass


class MLAAttentionSpec(FullAttentionSpec):
    pass


class ChunkedLocalAttentionSpec(AttentionSpec):
    pass


class SlidingWindowSpec(AttentionSpec):
    def __init__(self, block_size, sliding_window, **kw):
        super().__init__(block_size, **kw)
        self.sliding_window = sliding_window


class MambaSpec(KVCacheSpec):
    def __init__(self, block_size, mamba_cache_mode="align", shapes=None, dtypes=None):
        super().__init__(block_size)
        self.mamba_cache_mode = mamba_cache_mode
        # Stage-0 R13 KDA ABI at num_spec=7, TP=2.
        self.shapes = shapes if shapes is not None else ((10, 12288), (32, 128, 128))
        self.dtypes = (
            dtypes if dtypes is not None else ("torch.bfloat16", "torch.float32")
        )


class KpoolTailSpec(AttentionSpec):
    pass


class UniformTypeKVCacheSpecs(KVCacheSpec):
    """[I]'s wrapper: subclasses KVCacheSpec ONLY (not FullAttentionSpec),
    aggregates member cacheability — kv_cache_interface.py:920-937."""

    def __init__(self, block_size, kv_cache_specs):
        super().__init__(block_size)
        self.kv_cache_specs = kv_cache_specs

    @property
    def participates_in_prefix_caching(self) -> bool:
        return all(
            s.participates_in_prefix_caching for s in self.kv_cache_specs.values()
        )


@dataclass
class FakeKVCacheGroup:
    kv_cache_spec: object
    layer_names: tuple
    is_eagle_group: bool = False


@dataclass
class FakeKVCacheConfig:
    kv_cache_groups: list
    num_blocks: int = 804


def boot_layout() -> FakeKVCacheConfig:
    """The 7-group GLM-5.3-Flash layout (stage-0 receipts §C)."""
    g0_layers = tuple(f"mla.{i}" for i in range(11)) + tuple(
        f"idx.{i}" for i in range(11)
    )
    g0 = FakeKVCacheGroup(
        UniformTypeKVCacheSpecs(
            3584, {n: MLAAttentionSpec(3584) for n in g0_layers}
        ),
        g0_layers,
    )
    g1_layers = tuple(f"kpool.{i}" for i in range(11))
    g1 = FakeKVCacheGroup(
        UniformTypeKVCacheSpecs(
            4, {n: KpoolTailSpec(4, prefix_cacheable=False) for n in g1_layers}
        ),
        g1_layers,
    )
    mamba_groups = []
    for g, n in ((2, 9), (3, 9), (4, 8), (5, 8)):
        names = tuple(f"kda{g}.{i}" for i in range(n))
        mamba_groups.append(
            FakeKVCacheGroup(
                UniformTypeKVCacheSpecs(3584, {nm: MambaSpec(3584) for nm in names}),
                names,
            )
        )
    g6_layers = tuple(f"draft.{i}" for i in range(5))
    g6 = FakeKVCacheGroup(
        UniformTypeKVCacheSpecs(
            64, {n: SlidingWindowSpec(64, 2048) for n in g6_layers}
        ),
        g6_layers,
    )
    return FakeKVCacheConfig([g0, g1] + mamba_groups + [g6])


@dataclass
class FakeSpeculativeConfig:
    num_speculative_tokens: int = 7

    def use_eagle(self) -> bool:
        return True


@dataclass
class FakeParallelConfig:
    rank: int = 0
    world_size: int = 2
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    data_parallel_index: int = 0
    data_parallel_size: int = 1
    data_parallel_rank_local: int | None = 0
    nnodes_within_dp: int = 2
    distributed_executor_backend: str = "mp"


@dataclass
class FakeCacheConfig:
    enable_prefix_caching: bool = True
    prefix_caching_hash_algo: str = "sha256"
    cache_dtype: str = "fp8"


@dataclass
class FakeModelConfig:
    model: str = "Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw"
    dtype: str = "bfloat16"
    use_mla: bool = True

    def get_total_num_kv_heads(self) -> int:
        return 1


@dataclass
class FakeVllmConfig:
    parallel_config: FakeParallelConfig = field(default_factory=FakeParallelConfig)
    speculative_config: FakeSpeculativeConfig | None = field(
        default_factory=FakeSpeculativeConfig
    )
    cache_config: FakeCacheConfig = field(default_factory=FakeCacheConfig)
    model_config: FakeModelConfig = field(default_factory=FakeModelConfig)
    kv_transfer_config: object | None = None
    kv_events_config: object | None = None
    use_v2_model_runner: bool = False


# ---------------------------------------------------------------------------
# The fake vllm module tree
# ---------------------------------------------------------------------------
class _RecordingLogger:
    def __init__(self):
        self.lines: list[str] = []

    def _fmt(self, msg, *args):
        try:
            self.lines.append(msg % args if args else str(msg))
        except TypeError:
            self.lines.append(str(msg))

    info = info_once = debug = warning = warning_once = error = exception = _fmt


def _mod(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def install_fake_vllm(with_restore: bool = False) -> dict:
    """Install a minimal fake vllm namespace; returns handles for tests."""
    logger = _RecordingLogger()

    vllm = _mod("vllm")
    vllm.__version__ = "0.1.dev20051+g487ecf187"

    m = _mod("vllm.config")
    m.VllmConfig = FakeVllmConfig

    m = _mod("vllm.logger")
    m.init_logger = lambda name: logger

    m = _mod("vllm.utils.math_utils")
    m.cdiv = lambda a, b: -(-a // b)
    m.round_down = lambda x, y: (x // y) * y
    sys.modules["vllm.utils"] = types.ModuleType("vllm.utils")

    m = _mod("vllm.distributed.kv_events")
    m.KVCacheEvent = object

    m = _mod("vllm.distributed.parallel_state")

    def _tp_rank():
        raise RuntimeError("TP group not initialized in the stub env")

    m.get_tensor_model_parallel_rank = _tp_rank

    m = _mod("vllm.distributed.kv_transfer.kv_connector.utils")

    def yield_req_data(scheduler_output):
        for req_id, (blocks, preempted) in scheduler_output.req_data.items():
            yield req_id, blocks, preempted

    m.yield_req_data = yield_req_data

    m = _mod("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorMetadata:
        pass

    class KVConnectorWorkerMetadata:
        pass

    m.KVConnectorMetadata = KVConnectorMetadata
    m.KVConnectorWorkerMetadata = KVConnectorWorkerMetadata

    m = _mod("vllm.v1.kv_cache_interface")
    for cls in (
        AttentionSpec,
        FullAttentionSpec,
        MLAAttentionSpec,
        ChunkedLocalAttentionSpec,
        SlidingWindowSpec,
        MambaSpec,
        KVCacheSpec,
        UniformTypeKVCacheSpecs,
    ):
        setattr(m, cls.__name__, cls)
    m.KVCacheConfig = FakeKVCacheConfig
    m.KVCacheTensor = object

    m = _mod("vllm.v1.core.kv_cache_manager")

    @dataclass
    class KVCacheBlocks:
        blocks: tuple

    m.KVCacheBlocks = KVCacheBlocks

    m = _mod("vllm.v1.core.sched.output")
    m.SchedulerOutput = object

    m = _mod("vllm.v1.core.kv_cache_utils")
    m.resolve_kv_cache_block_sizes = lambda kv_cache_config, vllm_config: (
        [3584],
        64,
    )

    m = _mod("vllm.v1.outputs")

    @dataclass
    class KVConnectorOutput:
        kv_connector_stats: object = None
        invalid_block_ids: frozenset = frozenset()
        finished_sending: set | None = None
        finished_recving: set | None = None

    m.KVConnectorOutput = KVConnectorOutput

    m = _mod("vllm.v1.request")

    class RequestStatus:
        FINISHED_ABORTED = "aborted"

    class Request:
        pass

    m.RequestStatus = RequestStatus
    m.Request = Request

    m = _mod("vllm.v1.attention.backend")
    m.AttentionBackend = object

    # --- vllm.v1.kv_offload.base: real key packing, minimal everything else
    m = _mod("vllm.v1.kv_offload.base")

    def make_offload_key(block_hash: bytes, group_idx: int):
        return block_hash + group_idx.to_bytes(4, "big", signed=False)

    m.make_offload_key = make_offload_key
    m.get_offload_block_hash = lambda key: key[:-4]
    m.get_offload_group_idx = lambda key: int.from_bytes(key[-4:], "big")
    m.OffloadKey = bytes

    import enum

    class Medium(enum.Enum):
        CPU = "CPU"
        STORAGE = "STORAGE"

    class Locality(enum.Enum):
        LOCAL = "LOCAL"
        REMOTE = "REMOTE"

    class LookupResult(enum.Enum):
        HIT = 1
        HIT_PENDING = 2
        RETRY = 3
        MISS = 4

    class OffloadPolicy(enum.Enum):
        BLOCK_LEVEL = 1
        REQUEST_LEVEL = 2

    @dataclass
    class TierMatcher:
        medium: object = None
        locality: object = None

    class TierFilter:
        ALL = None

        def __init__(self, matchers=()):
            self.matchers = matchers

    TierFilter.ALL = TierFilter()

    @dataclass
    class ReqContext:
        req_id: str
        kv_transfer_params: dict | None = None
        load_tier_filter: object = None

    @dataclass
    class RequestOffloadingContext:
        policy: object = OffloadPolicy.BLOCK_LEVEL

    @dataclass
    class ScheduleEndContext:
        new_req_ids: list = None
        preempted_req_ids: object = ()

    import numpy as np

    class LoadStoreSpec:
        pass

    class BlockIDsLoadStoreSpec(LoadStoreSpec):
        def __init__(self, block_ids):
            self.block_ids = np.array(block_ids, dtype=np.int64)

    class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
        def __init__(self, block_ids, group_sizes, block_indices):
            super().__init__(block_ids)
            assert sum(group_sizes) == len(block_ids)
            assert len(block_indices) == len(group_sizes)
            self.group_sizes = group_sizes
            self.block_indices = block_indices

    class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
        pass

    class OffloadingManager:
        pass

    class OffloadingWorker:
        pass

    class OffloadingSpec:
        def __init__(self, config):
            self.config = config
            self.extra_config = config.extra_config
            self.tokens_per_block = tuple(
                g.tokens_per_block for g in config.groups
            )
            self.tokens_per_hash = config.cache.tokens_per_hash
            self.blocks_per_chunk = config.cache.blocks_per_chunk
            # Mirror upstream: DEFAULT TRUE — the launcher must explicitly
            # pass offload_prompt_only=false or decoded tokens are not stored.
            self.offload_prompt_only = bool(
                config.extra_config.get("offload_prompt_only", True)
            )

    for name, obj in dict(
        Medium=Medium,
        Locality=Locality,
        LookupResult=LookupResult,
        OffloadPolicy=OffloadPolicy,
        TierMatcher=TierMatcher,
        TierFilter=TierFilter,
        ReqContext=ReqContext,
        RequestOffloadingContext=RequestOffloadingContext,
        ScheduleEndContext=ScheduleEndContext,
        LoadStoreSpec=LoadStoreSpec,
        BlockIDsLoadStoreSpec=BlockIDsLoadStoreSpec,
        GPULoadStoreSpec=GPULoadStoreSpec,
        CPULoadStoreSpec=CPULoadStoreSpec,
        OffloadingManager=OffloadingManager,
        OffloadingWorker=OffloadingWorker,
        OffloadingSpec=OffloadingSpec,
        CanonicalKVCaches=object,
        CanonicalKVCacheRef=object,
        CanonicalKVCacheTensor=object,
        OffloadingCounterMetadata=object,
        OffloadingGaugeMetadata=object,
        OffloadingHistogramMetadata=object,
        OffloadingMetricMetadata=object,
    ).items():
        setattr(m, name, obj)

    # --- offloading sibling modules the scheduler/worker import
    m = _mod("vllm.distributed.kv_transfer.kv_connector.v1.offloading.events")

    @dataclass
    class OffloadingEventGroupSpec:
        name: str = "g"

    class OffloadingEventsTracker:
        def __init__(self, kv_events_config):
            self.records = []

        def record_lookup(self, *a, **k):
            self.records.append(("lookup", a))

        def record_partial_lookup(self, *a, **k):
            self.records.append(("partial_lookup", a))

        def record_store(self, *a, **k):
            self.records.append(("store", a))

        def record_partial_store(self, *a, **k):
            self.records.append(("partial_store", a))

        def take_events(self):
            return ()

        def reset(self):
            self.records.clear()

    m.OffloadingEventGroupSpec = OffloadingEventGroupSpec
    m.OffloadingEventsTracker = OffloadingEventsTracker
    m.get_offloading_event_group_spec = lambda group: OffloadingEventGroupSpec()

    m = _mod("vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics")

    class OffloadingConnectorStats:
        def __init__(self):
            self.hist = []

        def observe_histogram(self, *a, **k):
            self.hist.append(a)

        def increase_counter(self, *a, **k):
            pass

    class _ConnectorMetricName:
        LOOKUP_ASYNC_DELAY = "lookup_async"
        LOOKUP_SYNC_DELAY = "lookup_sync"
        ALLOCATION_FAILURE = "alloc_failure"

    class _TransferMetricName:
        pass

    m.OffloadingConnectorStats = OffloadingConnectorStats
    m._ConnectorMetricName = _ConnectorMetricName
    m._TransferMetricName = _TransferMetricName

    m = _mod(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping"
    )
    m.derive_canonical_mappings = lambda *a, **k: {}
    m.canonical_format_id = lambda: "v1-nhd"

    # --- exec the PATCHED overlay outputs as the real module names
    texts = patched_texts(with_store=True, with_restore=with_restore)

    kvo_cfg = _mod("vllm.v1.kv_offload.config")
    exec(compile(texts["kv_offload_config"], "kv_offload/config.py", "exec"), kvo_cfg.__dict__)

    common = _mod("vllm.distributed.kv_transfer.kv_connector.v1.offloading.common")
    exec(compile(texts["common"], "offloading/common.py", "exec"), common.__dict__)

    conn_cfg = _mod("vllm.distributed.kv_transfer.kv_connector.v1.offloading.config")
    exec(compile(texts["offloading_config"], "offloading/config.py", "exec"), conn_cfg.__dict__)

    sched = _mod("vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler")
    exec(compile(texts["scheduler"], "offloading/scheduler.py", "exec"), sched.__dict__)

    # worker.py imports torch at module level; provide a named stub only if
    # torch is genuinely absent (the writer path never touches it).
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            t = _mod("torch")
            t.Tensor = object
            t.Event = object

    worker = _mod("vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker")
    exec(compile(texts["worker"], "offloading/worker.py", "exec"), worker.__dict__)

    return {
        "logger": logger,
        "kv_offload_config": kvo_cfg,
        "offloading_config": conn_cfg,
        "scheduler": sched,
        "common": common,
        "worker": worker,
        "base": sys.modules["vllm.v1.kv_offload.base"],
        "kv_cache_manager": sys.modules["vllm.v1.core.kv_cache_manager"],
    }


# ---------------------------------------------------------------------------
# Convenience builders on top of the fake tree
# ---------------------------------------------------------------------------
@dataclass
class FakeKVTransferConfig:
    engine_id: str = "engine0"
    kv_connector_extra_config: dict = field(default_factory=dict)


def build_offloading_config(mods, vllm_config=None, kv_cache_config=None):
    vllm_config = vllm_config or FakeVllmConfig(
        kv_transfer_config=FakeKVTransferConfig()
    )
    kv_cache_config = kv_cache_config or boot_layout()
    # worker_kv_bytes_per_block path needs kv_cache_tensors; keep it inert.
    kv_cache_config.kv_cache_tensors = []
    kv_cache_config.num_blocks = 0
    return mods["offloading_config"].build_offloading_config(
        vllm_config, kv_cache_config
    )


class FakeManager:
    """Scheduler-side OffloadingManager stub that accepts every store."""

    def __init__(self, mods):
        self.base = mods["base"]
        self.lookup_calls = []
        self.prepare_load_calls = []
        self.touched = []

    def on_new_request(self, req_context):
        return self.base.RequestOffloadingContext()

    def lookup(self, key, req_context):
        self.lookup_calls.append(key)
        return self.base.LookupResult.MISS

    def touch(self, keys, req_context):
        self.touched.append(tuple(keys))

    def prepare_load(self, keys, req_context):
        self.prepare_load_calls.append(tuple(keys))
        return self.base.CPULoadStoreSpec(list(range(len(list(keys)))))

    def prepare_store(self, keys, req_context):
        keys = list(keys)

        class StoreOutput:
            pass

        out = StoreOutput()
        out.keys_to_store = list(keys)
        out.store_spec = self.base.CPULoadStoreSpec(
            list(range(100, 100 + len(keys)))
        )
        return out

    def complete_load(self, keys, req_context):
        self.completed_loads = getattr(self, "completed_loads", [])
        self.completed_loads.append(tuple(keys))

    def complete_store(self, keys, req_context):
        self.completed_stores = getattr(self, "completed_stores", [])
        self.completed_stores.append(tuple(keys))

    def on_request_finished(self, req_context):
        pass

    def on_schedule_end(self, schedule_end_context):
        pass

    def has_pending_work(self) -> bool:
        return False

    def reset_cache(self):
        pass

    def take_events(self):
        return ()

    def get_stats(self):
        return None


class FakeSchedSpec:
    """Duck-typed OffloadingSpec for the scheduler side."""

    def __init__(self, mods, offloading_config, manager=None):
        self.config = offloading_config
        self.tokens_per_block = tuple(
            g.tokens_per_block for g in offloading_config.groups
        )
        self.tokens_per_hash = offloading_config.cache.tokens_per_hash
        self.blocks_per_chunk = offloading_config.cache.blocks_per_chunk
        self.offload_prompt_only = bool(
            offloading_config.extra_config.get("offload_prompt_only", True)
        )
        self.kv_events_config = None
        self._manager = manager or FakeManager(mods)

    def get_manager(self):
        return self._manager


class FakeRequest:
    def __init__(
        self,
        num_tokens,
        tokens_per_hash=64,
        req_id="req0",
        salt=b"s",
        num_prompt_tokens=None,
    ):
        import hashlib

        self.request_id = req_id
        self.num_tokens = num_tokens
        self.num_prompt_tokens = (
            num_tokens if num_prompt_tokens is None else num_prompt_tokens
        )
        self.num_computed_tokens = 0
        self.kv_transfer_params = None
        self.skip_reading_prefix_cache = False
        self.status = None
        n_hashes = num_tokens // tokens_per_hash
        self.block_hashes = [
            hashlib.sha256(salt + i.to_bytes(4, "big")).digest()
            for i in range(n_hashes)
        ]

    def is_finished(self):
        return False
