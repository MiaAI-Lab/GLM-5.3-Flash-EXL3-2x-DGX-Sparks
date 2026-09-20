# Display-reserve KV: +1.75 GiB of KV cache per Spark from the framebuffer carveout

GB10 firmware reserves roughly 2 GiB of the unified 128 GB for display
scanout. CUDA's allocator never sees it, so on a headless Spark it is simply
lost. This overlay reclaims most of it as KV-cache backing, on top of the
ordinary profiled KV budget. Nothing else about the recipe changes.

Technique: [coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark)
(`release/runtime/sources/display_kv.c`, AGPL-3.0). Our port is
`overlay/display_kv/display_kv.c` + `glm53_display_kv.py`, wired in by
`overlay/patch_display_kv.py`.

## How it works

1. With `nvidia_drm modeset=1 fbdev=0` and no desktop, `/dev/dri/card0`
   accepts `DRM_IOCTL_MODE_CREATE_DUMB`. A 4096-px-wide, 32-bpp dumb buffer of
   `GLM53_DISPLAY_KV_MIB` is allocated **from the display reservation**, not
   from host RAM (`MemAvailable` moves by ~0 when it is touched).
2. `DRM_IOCTL_MODE_MAP_DUMB` + `mmap(MAP_SHARED)` maps it into the worker
   process; `cuMemHostRegister(DEVICEMAP | IOMEMORY)` registers it with CUDA.
   On the unified GB10 the device pointer equals the host pointer; the library
   asserts that and fails otherwise.
3. `gpu_worker.determine_available_memory` opens the pool once per rank and
   **adds** its size to `available_kv_cache_memory_bytes`. The ordinary budget
   (`GPU_MEM_UTIL`, profile run, CUDA-graph reserve) is untouched.
4. `gpu_model_runner._allocate_kv_cache_tensors` plans first-fit-decreasing
   over the KV tensor list and carves the largest tensors that fit from the
   pool (zero-copy `torch.as_tensor` views via `__cuda_array_interface__`);
   the rest come from `torch.zeros` as before. The stranded remainder is at
   most one small tensor (5 MiB on this kit).

The pool is process-lifetime: it is never unmapped while views or CUDA graphs
exist.

## Measured on this kit (driver 580.173.02, 2026-09-19)

Standalone probe (`overlay/display_kv/probe_main.cu`), 1792 MiB pool:

| Path | Read | Write |
|---|---:|---:|
| display pool (registered I/O) | 162 GB/s | 114 GB/s |
| ordinary pinned host (`cuMemHostRegister`) | 162 GB/s | 112 GB/s |
| `cudaMalloc` | 235 GB/s | 196 GB/s |

The pool behaves exactly like registered host memory: ~69% of `cudaMalloc`
read bandwidth. `cudaMemcpy` device→pool runs at ~60 GB/s. KV reads for the
blocks that land in the pool are correspondingly slower; the blocks are a
small fraction of the total pool (1.75 of ~15.5 GiB per rank here).

Largest dumb buffer that allocates: **2032 MiB** (2048 fails with ENOMEM).
The default stays at 1792 MiB, matching the DeepSeek recipe, leaving the
driver a margin.

## Host setup (one time, both Sparks)

DGX OS ships `/etc/modprobe.d/zz-nvidia-drm-override.conf` with
`options nvidia-drm modeset=0`, which disables dumb buffers. This kit's file
now reads:

```text
options nvidia_drm modeset=1 fbdev=0
```

followed by `sudo update-initramfs -u -k all`. The boot target is already
`multi-user.target`. To activate without a reboot, with no CUDA jobs and
`display-manager` inactive:

```bash
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" \
 && test "$(sudo cat /sys/module/nvidia_drm/refcnt)" = 0 \
 && sudo rmmod nvidia_drm && sudo modprobe nvidia_drm modeset=1 fbdev=0
```

Verify: `sudo cat /sys/module/nvidia_drm/parameters/{modeset,fbdev}` → `Y`, `N`.
Keep the UEFI display reservation at its default (2 GB); zero removes the region.

## Launcher knobs (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GLM53_DISPLAY_KV` | `auto` | `1` = required (boot fails if the pool cannot be opened), `auto` = use when both hosts report `modeset=Y fbdev=N` and the DRM node exists, `0` = never touch DRM |
| `GLM53_DISPLAY_KV_MIB` | `1792` | pool size per rank, multiple of 16 (≤ 2032 fits) |
| `HEAD_DRM_CARD` / `WORKER_DRM_CARD` | `/dev/dri/card0` | host DRM node, mapped to `/dev/dri/card0` inside each container |

Preflight checks the module parameters on both hosts (via `sudo -n` when
needed), builds `libglm53_display_kv.so` inside the serving image if it is
stale, and passes `--device <card>:/dev/dri/card0` to both containers. The
containers are not privileged and receive only that device node.

Boot log lines to look for (both ranks):

```text
[glm53-display-kv] {"stage": "pool_open", "bytes": 1879048192, "mib": 1792, ...}
[glm53-display-kv] {"stage": "plan", "tensors": 33, "from_pool": 11, "planned_mib": 1786, "stranded_mib": 5}
[glm53-display-kv] {"stage": "kv_allocated", ..., "tensors_ordinary": 22, "unused_pool_mib": 5}
```

`Available KV cache memory` in the worker log now includes the pool.

## Caveats

- It does not create memory: the ~256 MiB left of the reservation is not
  additional headroom, and host-RAM preflight is unchanged.
- Lower bandwidth than ordinary CUDA memory for the blocks that live there.
- A display workload (monitor, remote desktop, `display-manager`) competes for
  the same region; keep both Sparks headless.
- Driver-specific. Qualified on 580.173.02 only; on another driver use
  `GLM53_DISPLAY_KV=auto` and check the log.
