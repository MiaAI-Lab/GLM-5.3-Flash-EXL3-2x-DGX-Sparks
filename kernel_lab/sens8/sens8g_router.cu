// SENS8_V1 graph-safe router (SENS8G_V1): fixed launch, dynamic metadata.
//
// Provenance: generalized from kernel_lab/sens8/sens8_router.cu (phase-2
// port of donor 00548bd), same author/project. MATH UNCHANGED (see docs):
// identical utility/rank/coreset/top-8/renorm/tie-break; adds mode=STOCK
// (coreset bypassed == stock selection over all E) selected by device
// metadata so graph capture never freezes a branch.
//
// Launch: grid=4 CTAs (one per potential request block, MAX_NUM_SEQS=4).
// Metadata meta[6] int32: {mode, nb, len0..len3}; offsets are prefix sums.
// Full overwrite per step by the prepare path; kernel clamps defensively
// (never OOB) but relies on fresh metadata for correctness.
// Constraints: E=288, C=28, K=8, SCALE=2.5, block rows<=16, nb<=4, M<=128.
// No global temporaries; no host-device sync; one launch per router call.
#include <torch/extension.h>
#include <climits>  // INT_MAX
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sens8g {

constexpr int E = 288;
constexpr int C = 28;
constexpr int K = 8;
constexpr float SCALE = 2.5f;
constexpr int MAXN = 16;
constexpr int MAXB = 4;
constexpr int NTHREADS = 256;
constexpr int NGROUP = 4;
constexpr int JJSTEP = E / NGROUP;  // 72

constexpr int MODE_STOCK = 0;
constexpr int MODE_SENS8 = 1;

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__global__ __launch_bounds__(256) void sens8g_kernel(
    const T* __restrict__ logits,      // [M, E]
    const float* __restrict__ bias,   // [E]
    const int* __restrict__ meta,     // [6] int32 {mode, nb, len0..len3}
    int* __restrict__ ids,            // [M, K]
    float* __restrict__ w,            // [M, K]
    int* __restrict__ dbg,            // [2] int32 {stock_launches, sens8_launches} (nullable)
    int m, int stride_lm, int stride_le, int stride_om, int stride_ok) {
  const int q = blockIdx.x;  // fixed grid=MAXB; CTA q owns block q or exits.
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int mode = meta[0];
  const int nb = meta[1];
  const int grid = gridDim.x;
  if (dbg != nullptr && tid == 0 && q == 0) {
    atomicAdd(&dbg[(mode == MODE_SENS8) ? 1 : 0], 1);
  }
  if (mode == MODE_SENS8) {
    if (q >= nb || q >= MAXB) return;
  }
  // STOCK tiles stride across all CTAs of the (M-sized) grid.

  // Block rows from prefix sums of lens (defensive clamp: never OOB even
  // on stale metadata; correctness relies on fresh per-step metadata).
  int start = 0;
  for (int i = 0; i < q && i < MAXB; ++i) start += meta[2 + i];
  int n = (mode == MODE_SENS8) ? meta[2 + q] : m;
  if (start < 0) start = 0;
  if (start > m) start = m;
  if (n < 0) n = 0;
  if (start + n > m) n = m - start;
  if (n > MAXN) n = MAXN;
  if (n <= 0) return;

  __shared__ float sh_bias[E];
  __shared__ float sh_util[E];
  __shared__ float sh_b[MAXN][E];
  __shared__ float sh_s[MAXN][E];
  __shared__ int sh_core[C];
  __shared__ unsigned char sh_is_core[E];
  __shared__ float sh_part[NGROUP][E];

  for (int e = tid; e < E; e += NTHREADS) sh_bias[e] = bias[e];
  for (int e = tid; e < E; e += NTHREADS) sh_util[e] = 0.0f;
  __syncthreads();

  // ---- STOCK mode tiles all rows 16 at a time, strided across the grid
  // (prefill-safe). SENS8 mode handles one block (n<=16) directly.
  const int ntile = (mode == MODE_SENS8) ? 1 : ((m + MAXN - 1) / MAXN);
  const int tbeg = (mode == MODE_SENS8) ? 0 : q;
  const int tstep = (mode == MODE_SENS8) ? 1 : grid;
  for (int tile = tbeg; tile < ntile; tile += tstep) {
    const int t0 = (mode == MODE_SENS8) ? start : tile * MAXN;
    int tn = (mode == MODE_SENS8) ? n : m - t0;
    if (tn > MAXN) tn = MAXN;
    if (tn <= 0) continue;
  // ---- A1: sigmoid/b + stash (both modes; STOCK's dense scan reads it).
  // Coalesced lane mapping (e = lane + 32*j): same per-element ops as
  // lane*9+j, bitwise identical values, fewer sector transactions.
  for (int rr = warp; rr < tn; rr += 8) {
    for (int j = 0; j < 9; ++j) {
      int e = lane + 32 * j;
      float s = sigmoidf(to_f(logits[(t0 + rr) * stride_lm + e * stride_le]));
      sh_s[rr][e] = s;
      sh_b[rr][e] = s + sh_bias[e];
    }
  }
  __syncthreads();

  if (mode == MODE_SENS8) {
    // ---- A2: hierarchical exact ranks (4 groups x 72 jj), one row at a time.
    // Counting is exact (0/1 votes, exact partial sums); index-ascending
    // tie-break (jj < e) matches torch argsort. No atomics.
    const int grp = tid >> 6;
    const int loc = tid & 63;
    for (int rr = 0; rr < tn; ++rr) {
      for (int e = loc; e < E; e += 64) {
        float my_b = sh_b[rr][e];
        float part = 0.0f;
        const int j0 = grp * JJSTEP;
        for (int k = 0; k < JJSTEP; ++k) {
          int jj = j0 + k;
          float bj = sh_b[rr][jj];
          part += (my_b < bj) ? 1.0f : 0.0f;
          part += (my_b == bj && jj < e) ? 1.0f : 0.0f;
        }
        sh_part[grp][e] = part;
      }
      __syncthreads();
      for (int e = tid; e < E; e += NTHREADS) {
        float rank =
            (sh_part[0][e] + sh_part[1][e]) + (sh_part[2][e] + sh_part[3][e]);
        sh_util[e] += sh_s[rr][e] / (1.0f + rank);
      }
      __syncthreads();
    }

    // ---- top28 of block utility (warp 0 scan + shfl-reduce; strict > wins,
    // hence index-ascending order on ties).
    if (warp == 0) {
      for (int k = 0; k < C; ++k) {
        float lm = -INFINITY;
        int li = 0;
#pragma unroll
        for (int j = 0; j < 9; ++j) {
          int e = lane * 9 + j;
          float v = sh_util[e];
          if (v > lm) {
            lm = v;
            li = e;
          }
        }
        for (int d = 16; d > 0; d >>= 1) {
          float ov = __shfl_xor_sync(0xffffffff, lm, d);
          int oi = __shfl_xor_sync(0xffffffff, li, d);
          if (ov > lm || (ov == lm && oi < li)) {
            lm = ov;
            li = oi;
          }
        }
        if (lane == 0) {
          sh_core[k] = li;
          sh_util[li] = -INFINITY;
        }
      }
    }
    __syncthreads();
    // Mask is still built: the STOCK path below uses the original dense
    // masked scan (register pressure made the register-only STOCK path
    // slower; measured). SENS8 gathers the coreset directly instead.
    for (int e = tid; e < E; e += NTHREADS) {
      bool hit = false;
      for (int k = 0; k < C; ++k) hit |= (sh_core[k] == e);
      sh_is_core[e] = hit ? 1 : 0;
    }
    __syncthreads();
  } else {
    // STOCK: every expert eligible (mask bypassed for selection).
    for (int e = tid; e < E; e += NTHREADS) sh_is_core[e] = 1;
    __syncthreads();
  }

  // ---- B: per-row top8 + weights (one row per warp per wave).
  // SENS8 gathers the 28 coreset entries directly (one per lane): no
  // 288-scan, no mask reads. STOCK keeps the original dense masked scan
  // (the register-only STOCK path spilled and ran slower; measured).
  // Both preserve the pick sequence, tie-break, exclusion, and renorm, so
  // outputs are bitwise identical to the pre-reorg kernel.
  for (int rr = warp; rr < tn; rr += 8) {
    int ids_loc[8];
    float s_at[8];
    if (mode == MODE_SENS8) {
      float c_s;
      float c_b;
      int c_e;
      if (lane < C) {
        int ce = sh_core[lane];
        c_e = ce;
        c_b = sh_b[rr][ce];
        c_s = sh_s[rr][ce];
      } else {
        c_e = INT_MAX;
        c_b = -INFINITY;
        c_s = 0.0f;
      }
      for (int k = 0; k < 8; ++k) {
        bool taken = false;
        for (int t = 0; t < k; ++t) taken |= (ids_loc[t] == c_e);
        float lm = taken ? -INFINITY : c_b;
        int li = taken ? INT_MAX : c_e;
        for (int d = 16; d > 0; d >>= 1) {
          float ov = __shfl_xor_sync(0xffffffff, lm, d);
          int oi = __shfl_xor_sync(0xffffffff, li, d);
          if (ov > lm || (ov == lm && oi < li)) {
            lm = ov;
            li = oi;
          }
        }
        ids_loc[k] = li;
        // Broadcast the winner's s: every thread redundantly computes the
        // full sequence, so s_at must be lane-independent. Exactly one
        // lane holds li (unique ids).
        int iswin = (c_e == li) ? 1 : 0;
        unsigned winmask = __ballot_sync(0xffffffff, iswin);
        s_at[k] = (winmask == 0) ? 0.0f : __shfl_sync(
            0xffffffff, c_s, __ffs(winmask) - 1);
      }
    } else {
      for (int k = 0; k < 8; ++k) {
        float lm = -INFINITY;
        int li = 0;
        for (int j = 0; j < 9; ++j) {
          int e = lane * 9 + j;
          bool taken = false;
          for (int t = 0; t < k; ++t) taken |= (ids_loc[t] == e);
          float v = (!taken && sh_is_core[e]) ? sh_b[rr][e] : -INFINITY;
          if (v > lm) {
            lm = v;
            li = e;
          }
        }
        for (int d = 16; d > 0; d >>= 1) {
          float ov = __shfl_xor_sync(0xffffffff, lm, d);
          int oi = __shfl_xor_sync(0xffffffff, li, d);
          if (ov > lm || (ov == lm && oi < li)) {
            lm = ov;
            li = oi;
          }
        }
        ids_loc[k] = li;
      }
      for (int k = 0; k < 8; ++k) s_at[k] = sh_s[rr][ids_loc[k]];
    }
    float tot = ((((((s_at[0] + s_at[1]) + s_at[2]) + s_at[3]) + s_at[4]) +
                  s_at[5]) + s_at[6]) + s_at[7];
    if (tot < 1e-12f) tot = 1e-12f;
    if (lane < 8) {
      ids[(t0 + rr) * stride_om + lane * stride_ok] = ids_loc[lane];
      w[(t0 + rr) * stride_om + lane * stride_ok] =
          s_at[lane] / tot * SCALE;
    }
  }
  }  // tile
}

template <typename T>
void launch_fn(const T* logits, const float* bias, const int* meta, int* ids,
               float* w, int* dbg, int m, int slm, int sle, int som, int sok,
               cudaStream_t stream) {
  // Grid depends only on M (static per capture, graph-safe): 4 CTAs cover
  // any SENS8 partition; larger grids parallelize STOCK tiling (prefill).
  int grid = (m + MAXN - 1) / MAXN;
  if (grid < MAXB) grid = MAXB;
  if (grid > 64) grid = 64;
  sens8g_kernel<T><<<grid, NTHREADS, 0, stream>>>(logits, bias, meta, ids, w, dbg,
                                                  m, slm, sle, som, sok);
}

void sens8g_cuda_forward(at::Tensor logits, at::Tensor bias, at::Tensor meta,
                         at::Tensor ids, at::Tensor w, at::Tensor dbg) {
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
  TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == at::kFloat &&
                  bias.numel() == E,
              "bias must be CUDA fp32 [288]");
  TORCH_CHECK(meta.is_cuda() && meta.scalar_type() == at::kInt &&
                  meta.numel() == 6,
              "meta must be CUDA int32 [6] {mode, nb, len0..len3}");
  TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == at::kInt &&
                  w.is_cuda() && w.scalar_type() == at::kFloat,
              "ids int32 / w fp32 CUDA required");
  TORCH_CHECK(dbg.is_cuda() && dbg.scalar_type() == at::kInt &&
                  dbg.numel() == 2,
              "dbg must be CUDA int32 [2]");
  TORCH_CHECK(logits.size(1) == E && ids.size(1) == K && w.size(1) == K &&
                  ids.size(0) == logits.size(0) &&
                  w.size(0) == logits.size(0),
              "shapes: logits [M,288], ids/w [M,8]");
  const int m = (int)logits.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int slm = (int)logits.stride(0), sle = (int)logits.stride(1);
  const int som = (int)ids.stride(0), sok = (int)ids.stride(1);
  if (logits.scalar_type() == at::kFloat) {
    launch_fn((const float*)logits.data_ptr(), (const float*)bias.data_ptr(),
              (const int*)meta.data_ptr(), (int*)ids.data_ptr(),
              (float*)w.data_ptr(), (int*)dbg.data_ptr(), m, slm, sle, som,
              sok, stream);
  } else if (logits.scalar_type() == at::kBFloat16) {
    launch_fn((const __nv_bfloat16*)logits.data_ptr(),
              (const float*)bias.data_ptr(), (const int*)meta.data_ptr(),
              (int*)ids.data_ptr(), (float*)w.data_ptr(),
              (int*)dbg.data_ptr(), m, slm, sle, som, sok, stream);
  } else {
    TORCH_CHECK(false, "logits must be fp32 or bf16");
  }
}

}  // namespace sens8g

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sens8g::sens8g_cuda_forward, "graph-safe SENS8 router",
        py::arg("logits"), py::arg("bias"), py::arg("meta"), py::arg("ids"),
        py::arg("w"), py::arg("dbg"));
}
