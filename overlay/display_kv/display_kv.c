// SPDX-License-Identifier: AGPL-3.0-only
// Display-reserve KV backing for GB10 (DGX Spark).
//
// Derived from coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark
// release/runtime/sources/display_kv.c (AGPL-3.0), simplified for the
// GLM-5.3 recipe: display-only span (no ordinary prefix), configurable
// DRM node and byte count, contiguous-UVA contract preserved.
//
// The UEFI display reservation (~2 GiB on a Spark) is invisible to the CUDA
// allocator. With nvidia_drm modeset=1 fbdev=0 and no desktop, a DRM "dumb"
// scanout buffer of that size can be created, mmapped, and registered with
// CUDA as device-mapped I/O memory. On the unified GB10 the driver returns a
// device pointer equal to the host pointer; we assert that so torch views
// can alias the mapping directly. Nothing here mode-sets, touches physical
// addresses, or creates a CUDA context.
#define _GNU_SOURCE
#include <cuda.h>
#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct {
    void *base;
    size_t size;
    CUdeviceptr gpu;
    CUcontext context;
    int fd, registered;
    unsigned handle;
} Pool;

static _Thread_local char error_text[512];
const char *glm53_display_error(void) { return error_text; }

static int check_cuda(const char *op, CUresult rc) {
    if (rc == CUDA_SUCCESS) return 1;
    const char *name = "unknown"; cuGetErrorName(rc, &name);
    snprintf(error_text, sizeof error_text, "%s: %s (%d)", op, name, rc); return 0;
}
static int check_sys(const char *op, int ok) {
    if (ok) return 1;
    snprintf(error_text, sizeof error_text, "%s: %s", op, strerror(errno)); return 0;
}

void glm53_display_destroy(Pool *p) {
    if (!p) return;
    CUcontext previous = NULL; cuCtxGetCurrent(&previous);
    if (p->context) cuCtxSetCurrent(p->context);
    if (p->registered) { cuCtxSynchronize(); cuMemHostUnregister(p->base); }
    if (p->base && p->base != MAP_FAILED) munmap(p->base, p->size);
    if (p->handle) { struct drm_gem_close c = {.handle = p->handle}; ioctl(p->fd, DRM_IOCTL_GEM_CLOSE, &c); }
    if (p->fd >= 0) close(p->fd);
    if (previous != p->context) cuCtxSetCurrent(previous);
    free(p);
}

// bytes must be a multiple of 16 MiB (4096 px * 4 B * 1024 rows) so the dumb
// buffer geometry is exact. Caller must hold a current CUDA context.
Pool *glm53_display_create(const char *drm_node, size_t bytes) {
    error_text[0] = 0;
    const size_t row = 4096 * 4;
    if (bytes == 0 || bytes % (row * 1024) || bytes / row > 0xFFFFFFFFu) {
        snprintf(error_text, sizeof error_text, "bytes must be a positive multiple of 16 MiB"); return NULL;
    }
    Pool *p = calloc(1, sizeof *p); if (!p) return NULL;
    p->base = MAP_FAILED; p->fd = -1; p->size = bytes;
    if (!check_cuda("cuCtxGetCurrent", cuCtxGetCurrent(&p->context))) goto fail;
    if (!p->context) { snprintf(error_text, sizeof error_text, "No current CUDA context on caller thread"); goto fail; }
    p->fd = open(drm_node ? drm_node : "/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (!check_sys("open DRM card", p->fd >= 0)) goto fail;
    struct drm_get_cap cap = {.capability = DRM_CAP_DUMB_BUFFER};
    if (!check_sys("DRM_CAP_DUMB_BUFFER", ioctl(p->fd, DRM_IOCTL_GET_CAP, &cap) == 0 && cap.value)) {
        if (!errno) snprintf(error_text, sizeof error_text, "DRM node lacks dumb-buffer support (nvidia_drm modeset=0?)");
        goto fail;
    }
    struct drm_mode_create_dumb c = {.width = 4096, .height = (unsigned)(bytes / row), .bpp = 32};
    if (!check_sys("DRM create dumb buffer", ioctl(p->fd, DRM_IOCTL_MODE_CREATE_DUMB, &c) == 0)) goto fail;
    p->handle = c.handle;
    if (c.size != bytes) { snprintf(error_text, sizeof error_text, "Unexpected dumb buffer size %llu", (unsigned long long)c.size); goto fail; }
    struct drm_mode_map_dumb m = {.handle = p->handle};
    if (!check_sys("DRM map dumb buffer", ioctl(p->fd, DRM_IOCTL_MODE_MAP_DUMB, &m) == 0)) goto fail;
    p->base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, p->fd, m.offset);
    if (!check_sys("mmap dumb buffer", p->base != MAP_FAILED)) goto fail;
    if (!check_cuda("cuMemHostRegister(IOMEMORY|DEVICEMAP)", cuMemHostRegister(p->base, bytes,
            CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY))) goto fail;
    p->registered = 1;
    CUdeviceptr gpu = 0;
    if (!check_cuda("cuMemHostGetDevicePointer", cuMemHostGetDevicePointer(&gpu, p->base, 0))) goto fail;
    if (gpu != (CUdeviceptr)(uintptr_t)p->base) {
        snprintf(error_text, sizeof error_text, "Driver did not preserve UVA: host=%p device=%" PRIx64, p->base, (uint64_t)gpu); goto fail;
    }
    p->gpu = gpu;
    return p;
fail:
    glm53_display_destroy(p); return NULL;
}

uint64_t glm53_display_pointer(Pool *p) { return p ? (uint64_t)p->gpu : 0; }
size_t   glm53_display_size(Pool *p)    { return p ? p->size : 0; }
