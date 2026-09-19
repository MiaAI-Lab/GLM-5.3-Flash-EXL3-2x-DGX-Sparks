# Cold load on GB10 UMA: boot with a full page cache, 64 KiB mmap staging

`overlay/patch_cold_load_uma.py` + `GLM53_HOST_MEM_HYGIENE`. Measured on a
2× DGX Spark pair (GB10, 128 GB UMA, 1 TB NVMe, kernel `6.17.13-rocket64k`,
64 KiB pages); verified inert on 4 KiB kernels and discrete GPUs.

## The failure

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
iflag=direct`). Full boot receipt: `Loading safetensors using InstantTensor
loader: 100% | 164G/164G [00:36, 4.88GB/s]`, `Model loading took 82.06 GiB and
43.1 seconds`.

## What this does and does not change (measured 2026-09-19)

**It does not make a clean-cache boot faster.** Once the page cache is clean,
InstantTensor already streams at the drive ceiling; the budget helper is a
no-op there. Fair A/B — upstream `main` ca85576 + published image + only the
mmap-clone half of this patch (the 64 KiB kernel needs it either way), page
cache dropped by hand on both nodes, CUDA free ≥ 110 GiB confirmed, vs this
PR with `GLM53_HOST_MEM_HYGIENE=0` (identical host state), same `.env`, 2 reps:

| clean cache | stock + mmap clone | this PR |
|---|---|---|
| weight stream | 36 s @ 4.79 / 35 s @ 4.94 GB/s | 36 s @ 4.82 / 35 s @ 4.89 |
| `Model loading took` | +65 / +68 s | +64 / +63 s |
| `/health` | 230 s / 230 s | 230 s / 230 s |
| back-to-back `./start.sh restart` | 256 s | 259 s |

**It makes the boot possible when the cache is not clean.** That is the state
every boot after the 164 GiB rsync or a previous serve is in, and the one an
operator hits unless they drop caches by hand. Same `.env`, page cache filled
to 118 GiB on both nodes, `./start.sh start` to `/health`:

| full page cache | stock + mmap clone | this PR |
|---|---|---|
| weight stream | `Shrink io_depth from 256 to 34`, then **`RuntimeError: buffer_size … exceeds device memory budget (586612736 B)`** | 164 GiB in **36 s @ 4.82 GB/s** |
| `/health` | **never** (`start.sh` exit 1 after 168 s) | **230 s** |

Iterator-level on one rank's share (60 shards, 82.9 GiB, same container):
clean cache stock 5.09 vs patched 5.11 GB/s; full cache stock raises, patched
5.20 GB/s.

The restart gate (`waiting for CUDA free … to reach …`) closes an intermittent
race: on this kit two consecutive restarts failed vLLM's startup check by
< 0.5 GiB (`107.45/123.72 GiB … less than desired … 107.64 GiB`) while the
driver was still returning the torn-down context; it did not reproduce in the
A/B above, so treat it as insurance, not a speedup.

## What the patch does

Applied to `model_executor/model_loader/weight_utils.py` at image build and
re-applied (idempotent) at boot from `GLM53_OVERLAY_ORDER`:

* `instanttensor_weights_iterator`: before `safe_open`, read `/proc/meminfo`.
  If the host is UMA (`cuda free ≈ MemFree`) and `MemFree` is short of the
  load window (4 GiB buffer + largest shard + 2 GiB) while `MemAvailable`
  suffices, try `drop_caches` (works only with `CAP_SYS_ADMIN`; the launcher
  does it on the host first) and hand InstantTensor an explicit
  `max_free_mem_usage` / `buffer_size` that keeps `io_depth` at the AIO/uring
  default. `INSTANTTENSOR_*` env vars still win when set.
* `safetensors_weights_iterator`: when `sysconf(SC_PAGE_SIZE) != 4096`,
  `clone()` each tensor off the file-backed mmap into anonymous memory before
  yielding it. `cuMemcpyHtoDAsync` wedges on this driver when the source is a
  file-backed 64 KiB-page mapping. Kill switch `GLM53_COLD_LOAD_STAGE_MMAP=0`.

Kill switch for the whole patch: `GLM53_COLD_LOAD_UMA=0`. Host-only tests:
`tests/test_cold_load_uma.py` (also run in the Dockerfile before the patch is
applied).

## Behaviour off this kit (verified 2026-09-19, same image)

| host | `_GLM53_UMA_STAGE_MMAP` | InstantTensor kwargs | effect |
|---|---|---|---|
| 4 KiB kernel (sysconf faked to 4096) | `False` | — | `safetensors_weights_iterator` yields the stock mmap views, no clone |
| discrete GPU (`cuda free` ≠ `MemFree`) | n/a | `max_free_mem_usage=None, buffer_size=None` | InstantTensor's own defaults / env; one INFO line |
| UMA, plentiful free memory | `True` on 64 KiB | `0.5, 4 GiB` | identical to InstantTensor's default budget (0.5) with the buffer pinned at the io_depth-preserving size |
| UMA, page cache full | `True` on 64 KiB | `0.95, ≤ free − 1 GiB` | loads instead of raising; warns to drop cache on the host |

The UMA test is `abs(cuda_free − MemFree) < 8 GiB`; a discrete card's free
memory and the host's MemFree never track within that band.

## Launcher side (`GLM53_HOST_MEM_HYGIENE=1`, default)

`start.sh` drops clean page cache and cycles residual swap on **both** nodes
right before `docker run` (`sudo -n`; warns and continues without it), then
waits until `torch.cuda.mem_get_info()` free in a throwaway container clears
`GPU_MEM_UTIL × total + 1.5 GiB`. That second part matters on `restart`: the
driver returns the torn-down context a few GiB behind `MemFree`, and vLLM's
startup check failed twice on this kit by < 0.5 GiB (`Free memory on device
cuda:0 (107.45/123.72 GiB) … less than desired … (0.87, 107.64 GiB)`). This is
also the prevention step from `docs/uvm-livelock-gb10.md`.

## Rollback

`GLM53_COLD_LOAD_UMA=0` and `GLM53_HOST_MEM_HYGIENE=0` return the stock paths
without a rebuild.
