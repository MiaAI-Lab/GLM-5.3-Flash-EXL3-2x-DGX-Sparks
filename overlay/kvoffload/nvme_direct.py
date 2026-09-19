# SPDX-License-Identifier: Apache-2.0
"""
NvmeDirectOffloadingSpec: NVMe-direct KV offload — NO resident CPU cache tier.

Offloaded KV streams between the GPU KV pool and files on NVMe:

  store: worker records a CUDA event on the compute stream; an IO thread
         makes its private stream wait that event, DMA-copies the chunk's
         pages into its own small pinned bounce buffer (sized to ONE chunk,
         reused forever — an implementation detail measured in tens of MB,
         not an LRU tier), then writes the file (tmp + atomic rename).
  load:  IO thread reads the file into its bounce buffer, fences pending GPU
         writes to the destination blocks via the recorded event, H2D-copies,
         and synchronizes its stream. The scheduler only uses the blocks
         after the load job is reported finished.

Note: a true zero-copy path (CPU dereferencing the KV pool) was probed on
DGX Spark (GB10) and segfaults — cudaMalloc memory is not CPU-mappable even
on this UMA part (cudaMallocManaged is, but swapping vLLM's KV allocator is
out of scope). On UMA the bounce DMA is DRAM->DRAM and costs ~nothing next
to NVMe bandwidth.

Scheduler-side manager state is just an in-flight-store set; lookups are
file-existence checks against rank 0's directory (ranks store symmetric
chunks). Files survive restarts => restart-restore for free. No eviction:
an external TTL sweeper owns cleanup.

Config (kv_connector_extra_config):
  spec_name: "NvmeDirectOffloadingSpec"
  spec_module_path: "vllm.v1.kv_offload.nvme_direct"
  root_dir: (required) directory on NVMe
  n_io_threads: (optional, default 4; each holds one chunk-sized
                 pinned bounce buffer)
"""

import os
import shutil
import threading
from collections import deque
from collections.abc import Collection
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    TransferResult,
)
from vllm.v1.kv_offload.config import OffloadingConfig

logger = init_logger(__name__)


def _key_relpath(key: OffloadKey) -> str:
    """Relative file path for an offload key (hash+group bytes)."""
    h = bytes(key).hex()
    return os.path.join(h[:3], h + ".bin")


class NvmeFileLoadStoreSpec(LoadStoreSpec):
    """Carries relative file names, ordered like the job's keys (which are
    ordered by KV group, matching GPULoadStoreSpec.block_ids)."""

    def __init__(self, relpaths: list[str]):
        self.relpaths = relpaths

    def __repr__(self) -> str:
        return f"NvmeFileLoadStoreSpec({len(self.relpaths)} files)"


class NvmeDirectManager(OffloadingManager):
    """Scheduler-side manager: file-existence lookups, no eviction."""

    def __init__(self, lookup_dir: str):
        self.medium = Medium.STORAGE
        self._lookup_dir = lookup_dir
        os.makedirs(lookup_dir, exist_ok=True)
        # Keys with an in-flight store job.
        self._pending_stores: set[OffloadKey] = set()
        # Keys confirmed present on disk (cache of positive exists() checks).
        self._exists: set[OffloadKey] = set()

    def _path(self, key: OffloadKey) -> str:
        return os.path.join(self._lookup_dir, _key_relpath(key))

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if key in self._pending_stores:
            return LookupResult.HIT_PENDING
        if key in self._exists:
            return LookupResult.HIT
        if os.path.exists(self._path(key)):
            self._exists.add(key)
            return LookupResult.HIT
        return LookupResult.MISS

    @override
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        # Files are never evicted while the server runs; nothing to pin.
        return NvmeFileLoadStoreSpec([_key_relpath(k) for k in keys])

    @override
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        keys_to_store = []
        for k in keys:
            if k in self._pending_stores or k in self._exists:
                continue
            if os.path.exists(self._path(k)):
                self._exists.add(k)
                continue
            keys_to_store.append(k)
        self._pending_stores.update(keys_to_store)
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=NvmeFileLoadStoreSpec(
                [_key_relpath(k) for k in keys_to_store]
            ),
            evicted_keys=[],
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        # Do not trust success blindly for lookups: _exists is also fed by
        # real exists() checks, so an IO-failed store decays back to MISS
        # after reset; within this process complete_store(True) marks it.
        self._pending_stores.difference_update(keys)
        if success:
            self._exists.update(keys)

    @override
    def reset_cache(self) -> None:
        # Files stay on disk (external TTL owns them); drop in-memory state.
        self._pending_stores.clear()
        self._exists.clear()


class NvmeDirectWorker(OffloadingWorker):
    """Worker-side GPU<->NVMe file streamer via per-thread bounce buffers."""

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        rank: int,
        root_dir: str,
        n_io_threads: int = 4,
    ):
        self._dir = os.path.join(root_dir, f"r{rank}")
        os.makedirs(self._dir, exist_ok=True)
        self._tensors = [t.tensor for t in kv_caches.tensors]
        self._group_refs = kv_caches.group_data_refs

        # Largest chunk = max over groups of the group's summed page sizes.
        self._max_chunk_bytes = max(
            (
                sum(ref.page_size_bytes for ref in refs)
                for refs in self._group_refs
                if refs
            ),
            default=0,
        )
        self._tls = threading.local()
        self._pool = ThreadPoolExecutor(
            max_workers=n_io_threads, thread_name_prefix="vllm_kv_nvme"
        )
        # job_id -> (future, num_bytes, is_load)
        self._jobs: dict[int, tuple[Future, int, bool]] = {}
        self._results: deque[TransferResult] = deque()
        logger.info(
            "NvmeDirectWorker rank=%d dir=%s tensors=%d threads=%d "
            "bounce=%.1f MiB/thread (max %.1f MiB total)",
            rank,
            self._dir,
            len(self._tensors),
            n_io_threads,
            self._max_chunk_bytes / (1 << 20),
            n_io_threads * self._max_chunk_bytes / (1 << 20),
        )

    def _thread_state(self):
        """Lazily create this IO thread's pinned bounce buffer + stream."""
        st = getattr(self._tls, "state", None)
        if st is None:
            buf = torch.empty(
                self._max_chunk_bytes, dtype=torch.uint8, pin_memory=True
            )
            st = (buf, buf.numpy(), torch.cuda.Stream())
            self._tls.state = st
        return st

    # -- job planning ------------------------------------------------------

    def _plan(self, gpu_spec: GPULoadStoreSpec, relpaths: list[str]):
        """Return ([(path, group_idx, block_id), ...], total_bytes).

        blocks_per_chunk == 1: one GPU block (chunk) per file. File layout:
        for each data-ref of the block's group in order, page_size_bytes of
        that ref's tensor row (mirrors the CPU direct layout, including
        duplicate refs to a shared tensor)."""
        block_ids = gpu_spec.block_ids
        assert len(relpaths) == len(block_ids), (
            f"{len(relpaths)} files vs {len(block_ids)} blocks"
        )
        plans = []
        total = 0
        i = 0
        for g_idx, group_size in enumerate(gpu_spec.group_sizes):
            if group_size == 0:
                continue
            group_bytes = sum(
                ref.page_size_bytes for ref in self._group_refs[g_idx]
            )
            for b in block_ids[i : i + group_size]:
                plans.append((os.path.join(self._dir, relpaths[i]), g_idx, int(b)))
                total += group_bytes
                i += 1
        assert i == len(block_ids)
        return plans, total

    # -- IO tasks ----------------------------------------------------------

    def _store_task(self, event: torch.cuda.Event, plans) -> None:
        buf, buf_np, stream = self._thread_state()
        stream.wait_event(event)
        for path, g_idx, b in plans:
            # DMA the chunk's pages into the bounce buffer
            off = 0
            with torch.cuda.stream(stream):
                for ref in self._group_refs[g_idx]:
                    n = ref.page_size_bytes
                    buf[off : off + n].copy_(
                        self._tensors[ref.tensor_idx][b, :n].view(torch.uint8),
                        non_blocking=True,
                    )
                    off += n
            stream.synchronize()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(fd, buf_np[:off].data)
                os.fsync(fd)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            except BaseException:
                os.close(fd)
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            os.close(fd)
            os.replace(tmp, path)

    def _load_task(self, event: torch.cuda.Event, plans) -> None:
        buf, buf_np, stream = self._thread_state()
        # Fence pending GPU writes (e.g. zeroing) to the destination blocks.
        stream.wait_event(event)
        for path, g_idx, b in plans:
            expected = sum(ref.page_size_bytes for ref in self._group_refs[g_idx])
            fd = os.open(path, os.O_RDONLY)
            try:
                got = os.readv(fd, [buf_np[:expected].data])
                if got != expected:
                    raise OSError(f"short read {got}/{expected} on {path}")
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            finally:
                os.close(fd)
            off = 0
            with torch.cuda.stream(stream):
                for ref in self._group_refs[g_idx]:
                    n = ref.page_size_bytes
                    self._tensors[ref.tensor_idx][b, :n].view(torch.uint8).copy_(
                        buf[off : off + n], non_blocking=True
                    )
                    off += n
            stream.synchronize()

    # -- OffloadingWorker interface ---------------------------------------

    def _submit(
        self,
        job_id: int,
        gpu_spec: GPULoadStoreSpec,
        relpaths: list[str],
        is_load: bool,
    ) -> bool:
        plans, total = self._plan(gpu_spec, relpaths)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        task = self._load_task if is_load else self._store_task
        fut = self._pool.submit(task, event, plans)
        self._jobs[job_id] = (fut, total, is_load)
        return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(dst_spec, NvmeFileLoadStoreSpec)
        return self._submit(job_id, src_spec, dst_spec.relpaths, is_load=False)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, NvmeFileLoadStoreSpec)
        return self._submit(job_id, dst_spec, src_spec.relpaths, is_load=True)

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        finished = [jid for jid, (f, _, _) in self._jobs.items() if f.done()]
        for jid in finished:
            fut, total, is_load = self._jobs.pop(jid)
            exc = fut.exception()
            if exc is not None:
                if is_load:
                    # A missing/short file on load is unrecoverable: the
                    # scheduler already committed to restoring these blocks.
                    logger.error("NVMe KV load job %d failed: %s", jid, exc)
                    raise exc
                # The connector asserts success; a failed store just leaves
                # no file behind (tmp cleaned up), so the key becomes a MISS
                # again after restart / cache reset. Log and move on.
                logger.warning(
                    "NVMe KV store job %d failed (block stays un-offloaded): %s",
                    jid,
                    exc,
                )
            results.append(
                TransferResult(
                    job_id=jid, success=True, transfer_size=total, transfer_time=None
                )
            )
        return results

    def wait(self, job_ids: set[int]) -> None:
        futs = [self._jobs[j][0] for j in job_ids if j in self._jobs]
        if futs:
            futures_wait(futs)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


class NvmeDirectOffloadingSpec(OffloadingSpec):
    """Spec wiring the NVMe-direct manager + worker."""

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        root_dir = self.extra_config.get("root_dir")
        if not root_dir:
            raise ValueError(
                "root_dir must be set in kv_connector_extra_config for "
                "NvmeDirectOffloadingSpec"
            )
        self.root_dir: str = root_dir
        self.n_io_threads = int(self.extra_config.get("n_io_threads", 4))
        os.makedirs(self.root_dir, exist_ok=True)
        # Refuse a cache target that cannot provide the requested per-node
        # persistence budget. This catches accidental placement on the OS
        # volume before a long soak silently produces a partial cache.
        self.capacity_bytes = int(
            self.extra_config.get("capacity_bytes", 500 * (1 << 30))
        )
        usage = shutil.disk_usage(self.root_dir)
        if usage.free < self.capacity_bytes:
            raise RuntimeError(
                f"NVMe prefix-cache target {self.root_dir} has "
                f"{usage.free / (1 << 30):.1f} GiB free; "
                f"{self.capacity_bytes / (1 << 30):.1f} GiB required"
            )
        logger.info(
            "NVMe prefix-cache capacity validated: %.1f GiB free (target %.1f GiB)",
            usage.free / (1 << 30),
            self.capacity_bytes / (1 << 30),
        )
        if self.blocks_per_chunk != 1:
            raise ValueError(
                "NvmeDirectOffloadingSpec requires blocks_per_chunk == 1"
            )
        self._manager: NvmeDirectManager | None = None
        self._worker: NvmeDirectWorker | None = None

    @override
    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            # Ranks store symmetric key sets; rank 0's dir is node-local to
            # the scheduler, so it is the lookup authority.
            self._manager = NvmeDirectManager(os.path.join(self.root_dir, "r0"))
            logger.info(
                "Created NvmeDirectManager (lookup dir %s/r0, no CPU tier)",
                self.root_dir,
            )
        return self._manager

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            self._worker = NvmeDirectWorker(
                kv_caches=kv_caches,
                rank=self.config.parallel.rank,
                root_dir=self.root_dir,
                n_io_threads=self.n_io_threads,
            )
        return self._worker
