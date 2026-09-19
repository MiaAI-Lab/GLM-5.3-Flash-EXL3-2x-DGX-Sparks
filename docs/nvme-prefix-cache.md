# NVMe prefix cache + UMA cold load (GB10 / 64 KiB pages)

Two changes to the TP=2 recipe, both opt-in through `.env` and both measured on
a 2× DGX Spark pair (GB10, 128 GB UMA, 1 TB NVMe, kernel `6.17.13-rocket64k`,
64 KiB pages).

## 1. Cold load at the NVMe ceiling (`overlay/patch_cold_load_uma.py`)

### The failure

On unified memory `torch.cuda.mem_get_info()[0]` is host `MemFree`, and the
page cache counts as *used*. After the 164 GiB rsync to the worker (or any
previous serve), `MemFree` on the head is ~2 GiB while `MemAvailable` is
~117 GiB. InstantTensor sizes its pinned ring buffer against
`free * max_free_mem_usage` and either

* raises before the first byte is read
  (`buffer_size (1268776960 B) exceeds device memory budget (1255964672 B)`,
  logs/head.log 2026-09-18 19:36), or
* survives with `io_depth` shrunk from 512 to double digits, well below the
  drive.

Reproduced on demand: fill the page cache with ~120 GB of buffered reads, then
run the stock `instanttensor_weights_iterator` on 24 shards:

| page cache | stock | patched |
|---|---|---|
| ~120 GB cached | `RuntimeError: buffer_size … exceeds device memory budget (644972544 B)` | 34.4 GiB in 7.3 s = **5.07 GB/s** |
| dropped | 34.4 GiB in 7.3 s = 5.08 GB/s | 82.9 GiB in 17.4 s = **5.12 GB/s** |

The single-reader `O_DIRECT` ceiling of this NVMe is 4.9 GB/s (`dd bs=16M
iflag=direct`); four parallel readers sum to ~5.5 GB/s. The load is at the
storage ceiling either way once the budget is honest.

### What the patch does

Applied to `model_executor/model_loader/weight_utils.py` at image build and
re-applied (idempotent) at boot from `GLM53_OVERLAY_ORDER`:

* `instanttensor_weights_iterator`: before `safe_open`, read `/proc/meminfo`.
  If the host is UMA (`cuda free ≈ MemFree`) and `MemFree` is short of the
  load window (4 GiB buffer + largest shard + 2 GiB) while `MemAvailable`
  suffices, try `drop_caches` (works only with `CAP_SYS_ADMIN`; the launcher
  does it on the host first) and hand InstantTensor an explicit
  `max_free_mem_usage` / `buffer_size` that keeps `io_depth` at the AIO/uring
  default. `INSTANTTENSOR_*` env vars still win when set. Non-UMA hosts and
  missing `instanttensor` take the stock path.
* `safetensors_weights_iterator`: when `sysconf(SC_PAGE_SIZE) != 4096`,
  `clone()` each tensor off the file-backed mmap into anonymous memory before
  yielding it. `cuMemcpyHtoDAsync` wedges on this driver when the source is a
  file-backed 64 KiB-page mapping (the rocket fork carries the same one-line
  fix). Byte-identical to stock on 4 KiB kernels. Kill switch
  `GLM53_COLD_LOAD_STAGE_MMAP=0`.

Kill switch for the whole patch: `GLM53_COLD_LOAD_UMA=0`. Host-only tests:
`tests/test_cold_load_uma.py` (also run in the Dockerfile before the patch is
applied).

### Launcher side (`GLM53_HOST_MEM_HYGIENE=1`, default)

`start.sh` drops clean page cache and cycles residual swap on **both** nodes
right before `docker run` (`sudo -n`; warns and continues without it). This is
the part the container cannot do for itself, and it is also the prevention
step from `docs/uvm-livelock-gb10.md`.

## 2. NVMe-direct prefix cache (`overlay/kvoffload/`, `OFFLOAD_NVME=1`)

The image ships upstream's `OffloadingConnector` stack; this overlay adds:

| file | what |
|---|---|
| `nvme_direct.py` | out-of-tree `NvmeDirectOffloadingSpec`: **no CPU tier** (UMA has none to spare). Each IO thread owns one chunk-sized pinned bounce buffer (~26 MiB; 8 threads ≈ 207 MiB total). Store = event-fenced D2H into the bounce → `tmp` write + `fsync` + atomic rename to `<root>/r<rank>/<hash[:3]>/<hash>.bin`. Load = file → bounce → H2D on a private stream. Lookups are `os.path.exists` against rank 0's tree (ranks store symmetric key sets). Files persist across restarts — no eviction, boot-time capacity gate only. |
| `offloading_scheduler.py` | draft-tower KV groups are detected by layer name (`mtp`/`dflash`/`drafter`) or, failing that, the root-prefix minority rule (DFlash2 registers under `model.`, the target under `language_model.`) instead of marking every group EAGLE-volatile; non-offloadable groups are skipped from every store/load/lookup path. |
| `offloading_config.py`, `kv_offload_config.py` | GLM5Next's `KpoolTailSpec` (4-token circular scratch, no block hashes, non-positional block table) is excluded from the `tokens_per_hash` divisibility assert and contributes zero blocks to every `GPULoadStoreSpec`. Restores land on 3584-token chunk boundaries, where the scratch is empty by construction, so skipping it is exact. |

Every diff against the image's copies is a pure addition (verified 2026-09-19
by `diff`; the image's three files are byte-identical to the fork pre-patch
versions the overlay was written against).

Measured (TRIAL.md, 2026-08-28, 524k ctx): 44,236-token prompt cold prefill
54.98 s → **9.54 s after a full engine restart** (43,008 cached tokens), GPU
prefix hit 2.88 s. 2,837 files / 1.7 GiB per rank on NVMe. Streaming a 4 GiB/s
NVMe at 656 B/token MLA + KDA state keeps up with prefill.

### Knobs (`.env`)

```
OFFLOAD_NVME=1                 # 0 = stock, no connector
OFFLOAD_FS_DIR=/kv-nvme        # in-container root
OFFLOAD_HOST_DIR=~/kv-cache-nvme          # head host dir (WORKER_OFFLOAD_HOST_DIR on the worker)
OFFLOAD_IO_THREADS=8
OFFLOAD_CAPACITY_BYTES=        # empty = min(free head, free worker) - OFFLOAD_RESERVE_GB
OFFLOAD_RESERVE_GB=24
OFFLOAD_TTL_MINUTES=0          # 0 = no sweeper; N = hourly cron deletes files older than N min
```

Side effects when on: `--enable-prompt-tokens-details` on the head (cached
token counts in `usage`), `PYTHONHASHSEED=0` in both containers (stable prefix
hashes across ranks and restarts), and `expandable_segments:True` stripped from
`PYTORCH_CUDA_ALLOC_CONF` (vLLM refuses any KV connector with it).

### Capacity

The spec refuses to boot unless `capacity_bytes` is *free* on the target at
start. With `OFFLOAD_CAPACITY_BYTES` empty the launcher computes it from the
smaller node's free space so both ranks pass. There is no runtime eviction:
when the drive fills, stores fail (logged, block stays un-offloaded) and
serving continues. Size the reserve so the OS and logs never starve.

## Rollback

`OFFLOAD_NVME=0` and `GLM53_COLD_LOAD_UMA=0` return the stock paths without a
rebuild (the offload files are inert unless the connector is configured; the
cold-load patch is a boot-time overlay with a kill switch).
