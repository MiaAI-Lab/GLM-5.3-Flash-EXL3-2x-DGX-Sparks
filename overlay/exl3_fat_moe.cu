#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"
#include "exl3_fat_moe.cuh"

// Grouped fat-expert kernels. See exl3_fat_moe.cuh for the contract.
//
// Mainloop: TILE_M rows x 128 columns per CTA, K advanced 32 per pipeline
// stage with cp.async multi-buffering. B tiles (16x16 K4/MCG trellis tiles,
// 64 int16 words each) are staged through shared memory and dequantized in
// registers by the warp that owns that 16-column block, once per 16 K, and
// reused across all M blocks of the tile. The gate/up kernel runs two B
// streams (gate, up) for the same 128 intermediate columns so the SwiGLU and
// the down-projection input Hadamard fuse into its epilogue.

namespace {

constexpr int FM_THREADS = 256;                 // 8 warps
constexpr int FM_WARPS = FM_THREADS / 32;
constexpr int FM_TILE_N = 128;                  // one 16-col block per warp
constexpr int FM_TILE_K = 32;                   // per pipeline stage
constexpr int FM_STAGES = 4;
constexpr int FM_PACKED_WORDS = 64;             // int16 words per 16x16 K4 tile
constexpr int FM_B_STAGE_WORDS = 2 * FM_WARPS * FM_PACKED_WORDS;  // 2 k16 sub-steps
constexpr float FM_HAD_SCALE = 0.088388347648f; // 1/sqrt(128)

constexpr int FM_MB_GATEUP = 4;                 // 64-row tiles, 2 B streams
constexpr int FM_MB_DOWN = 4;                   // 64-row tiles, 2 B streams (256 cols)

template <int MB, int NS>
constexpr int fm_smem_bytes()
{
    constexpr int a_stage = MB * 16 * FM_TILE_K * 2;
    constexpr int b_stage = NS * FM_B_STAGE_WORDS * 2;
    constexpr int pipe = FM_STAGES * (a_stage + b_stage);
    constexpr int epi = 16 * NS * FM_TILE_N * 4;
    return pipe > epi ? pipe : epi;
}

// 128-element Hadamard over one row held as float4 per lane, then optional
// per-column scale. Same arithmetic as fat_had_ff_128 / had_ff_r_128_inner.
__device__ __forceinline__ void fm_had_row(float4& v, int lane)
{
    float s0 = v.x + v.y;
    float d0 = v.x - v.y;
    float s1 = v.z + v.w;
    float d1 = v.z - v.w;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;
    shuffle_had_f2x32(v.x, v.y, lane);
    shuffle_had_f2x32(v.z, v.w, lane);
    v.x *= FM_HAD_SCALE;
    v.y *= FM_HAD_SCALE;
    v.z *= FM_HAD_SCALE;
    v.w *= FM_HAD_SCALE;
}

__device__ __forceinline__ float4 fm_load_half4(const half* p)
{
    half4 h = *reinterpret_cast<const half4*>(p);
    return make_float4(__low2float(h.x), __high2float(h.x), __low2float(h.y), __high2float(h.y));
}

__device__ __forceinline__ void fm_mul_half4(float4& v, const half* p)
{
    float4 s = fm_load_half4(p);
    v.x *= s.x; v.y *= s.y; v.z *= s.z; v.w *= s.w;
}

__device__ __forceinline__ void fm_store_half4(half* p, const float4& v)
{
    half4 h(__floats2half2_rn(v.x, v.y), __floats2half2_rn(v.z, v.w));
    *reinterpret_cast<half4*>(p) = h;
}

// XOR swizzle of the 16-byte chunk column inside a 32-wide (64 B) A row so
// ldmatrix phases (8 consecutive rows, one chunk) hit 8 distinct bank groups.
__device__ __forceinline__ int fm_swz(int row, int chunk)
{
    return chunk ^ ((row >> 1) & 3);
}

// ---------------------------------------------------------------------------
// Gather + input Hadamard: h13[row] = had128(x[token[row]] * suh[expert[row]])
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(FM_THREADS)
void fm_gather_kernel(
    const half* __restrict__ x,
    const int64_t* __restrict__ row_token,
    const int* __restrict__ row_expert,
    const half* const* __restrict__ suh_ptrs,
    half* __restrict__ h13,
    const int* __restrict__ num_rows_ptr,
    int size_k)
{
    const int num_rows = *num_rows_ptr;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.y;             // 128-wide K block
    for (int row = blockIdx.x * FM_WARPS + warp; row < num_rows; row += gridDim.x * FM_WARPS)
    {
        const int64_t token = row_token[row];
        const half* suh = suh_ptrs[row_expert[row]] + blk * 128;
        const half* src = x + token * (int64_t) size_k + blk * 128;
        half* dst = h13 + (int64_t) row * size_k + blk * 128;
        // E2 boundary (had_hf_r_128_inner<pre_scale>): the input scale is a
        // fp16 x fp16 multiply, rounded, BEFORE the fp32 Hadamard.
        half4 hv = *reinterpret_cast<const half4*>(src + lane * 4);
        half4 hs = *reinterpret_cast<const half4*>(suh + lane * 4);
        hv.x = __hmul2(hv.x, hs.x);
        hv.y = __hmul2(hv.y, hs.y);
        float4 v = make_float4(__low2float(hv.x), __high2float(hv.x),
                               __low2float(hv.y), __high2float(hv.y));
        fm_had_row(v, lane);
        fm_store_half4(dst + lane * 4, v);
    }
}

// ---------------------------------------------------------------------------
// Shared mainloop: acc[NS][MB][2] += A(tile rows, K) @ B_ns(K, 128 cols)
// ---------------------------------------------------------------------------

// NB_STRIDE: 16-col block offset between streams (0 = different matrices
// over the same columns, 8 = adjacent 128-col halves of one matrix).
template <int MB, int NS, int NB_STRIDE>
__device__ __forceinline__ void fm_mainloop(
    const half* __restrict__ a,          // fat-row buffer [rows_cap, size_k]
    int size_k,
    int row0,
    int rows,
    const uint16_t* const (&packed)[NS], // per-stream trellis, (K/16, tiles_n, 64)
    int tiles_n,
    int n_block0,                        // first 16-col block of this tile
    half* sh_a,                          // FM_STAGES * MB*16*32 halves
    uint16_t* sh_b,                      // FM_STAGES * NS * FM_B_STAGE_WORDS
    FragC (&acc)[NS][MB][2])
{
    constexpr int TILE_M = MB * 16;
    constexpr int A_STAGE = TILE_M * FM_TILE_K;          // halves
    constexpr int A_CHUNKS = TILE_M * 4;                 // 16 B chunks per stage
    constexpr int A_ITERS = (A_CHUNKS + FM_THREADS - 1) / FM_THREADS;
    constexpr int B_STAGE = NS * FM_B_STAGE_WORDS;       // int16 words
    constexpr int B_CHUNKS = NS * 2 * FM_WARPS * 8;      // 16 B chunks per stage
    constexpr int B_ITERS = (B_CHUNKS + FM_THREADS - 1) / FM_THREADS;

    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int k_tiles = size_k / FM_TILE_K;

    #pragma unroll
    for (int s = 0; s < NS; ++s)
        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
        {
            acc[s][mb][0] = {};
            acc[s][mb][1] = {};
        }

    auto load_stage = [&](int stage, int kt)
    {
        half* sa = sh_a + stage * A_STAGE;
        #pragma unroll
        for (int i = 0; i < A_ITERS; ++i)
        {
            int c = i * FM_THREADS + t;
            if (c < A_CHUNKS)
            {
                int row = c >> 2;
                int chunk = c & 3;
                int src_row = row < rows ? row : rows - 1;
                const half* src = a + (int64_t) (row0 + src_row) * size_k + kt * FM_TILE_K + chunk * 8;
                half* dst = sa + row * FM_TILE_K + fm_swz(row, chunk) * 8;
                cp_async(dst, src);
            }
        }
        uint16_t* sb = sh_b + stage * B_STAGE;
        #pragma unroll
        for (int i = 0; i < B_ITERS; ++i)
        {
            int c = i * FM_THREADS + t;
            if (c < B_CHUNKS)
            {
                int s = c / (2 * FM_WARPS * 8);
                int r = c % (2 * FM_WARPS * 8);
                int j = r / (FM_WARPS * 8);          // k16 sub-step
                int nb = (r / 8) % FM_WARPS;         // 16-col block (= warp)
                int q = r % 8;                       // 16 B chunk of the 128 B tile
                const uint16_t* src = packed[s]
                    + ((int64_t) (kt * 2 + j) * tiles_n + n_block0 + s * NB_STRIDE + nb) * FM_PACKED_WORDS + q * 8;
                uint16_t* dst = sb + (s * 2 + j) * (FM_WARPS * FM_PACKED_WORDS) + nb * FM_PACKED_WORDS + q * 8;
                cp_async(dst, src);
            }
        }
    };

    #pragma unroll
    for (int s = 0; s < FM_STAGES - 1; ++s)
    {
        if (s < k_tiles) load_stage(s, s);
        cp_async_fence();
    }

    const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int a_chunk_hi = lane >> 4;

    for (int kt = 0; kt < k_tiles; ++kt)
    {
        cp_async_wait<FM_STAGES - 2>();
        __syncthreads();
        int nk = kt + FM_STAGES - 1;
        if (nk < k_tiles) load_stage(nk % FM_STAGES, nk);
        cp_async_fence();

        const int stage = kt % FM_STAGES;
        const half* sa = sh_a + stage * A_STAGE;
        const uint16_t* sb = sh_b + stage * B_STAGE;

        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            FragB fb[NS][2];
            #pragma unroll
            for (int s = 0; s < NS; ++s)
            {
                const uint32_t* wb = reinterpret_cast<const uint32_t*>(
                    sb + (s * 2 + j) * (FM_WARPS * FM_PACKED_WORDS) + warp * FM_PACKED_WORDS);
                dq_dispatch<4, 1>(wb, lane << 3, fb[s][0], fb[s][1]);
            }
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                FragA fa;
                int row = mb * 16 + a_row;
                int chunk = j * 2 + a_chunk_hi;
                ldsm4(fa, sa + row * FM_TILE_K + fm_swz(row, chunk) * 8);
                #pragma unroll
                for (int s = 0; s < NS; ++s)
                {
                    ptx_mma_m16n8k16(fa, fb[s][0], acc[s][mb][0]);
                    ptx_mma_m16n8k16(fa, fb[s][1], acc[s][mb][1]);
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();
}

// Stage one 16-row block of accumulators into sh_c[16][NS * 128] (fp32).
template <int NS>
__device__ __forceinline__ void fm_stage_acc(
    float* sh_c, FragC (&acc0)[2], FragC (&acc1)[2], int warp, int lane)
{
    constexpr int W = NS * FM_TILE_N;
    int r0 = lane >> 2;
    int col = (lane & 3) * 2 + warp * 16;
    {
        float* d0 = sh_c + r0 * W + col;
        float* d1 = sh_c + (r0 + 8) * W + col;
        d0[0] = acc0[0][0]; d0[1] = acc0[0][1]; d0[8] = acc0[1][0]; d0[9] = acc0[1][1];
        d1[0] = acc0[0][2]; d1[1] = acc0[0][3]; d1[8] = acc0[1][2]; d1[9] = acc0[1][3];
    }
    if constexpr (NS == 2)
    {
        float* d0 = sh_c + r0 * W + FM_TILE_N + col;
        float* d1 = sh_c + (r0 + 8) * W + FM_TILE_N + col;
        d0[0] = acc1[0][0]; d0[1] = acc1[0][1]; d0[8] = acc1[1][0]; d0[9] = acc1[1][1];
        d1[0] = acc1[0][2]; d1[1] = acc1[0][3]; d1[8] = acc1[1][2]; d1[9] = acc1[1][3];
    }
}

// ---------------------------------------------------------------------------
// gate/up GEMM + SwiGLU + down-input Hadamard
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(FM_THREADS, 2)
void fm_gateup_kernel(
    const half* __restrict__ h13,
    const uint16_t* const* __restrict__ gate_ptrs,
    const uint16_t* const* __restrict__ up_ptrs,
    const half* const* __restrict__ gate_svh_ptrs,
    const half* const* __restrict__ up_svh_ptrs,
    const half* const* __restrict__ down_suh_ptrs,
    half* __restrict__ h2,
    const int* __restrict__ seg_expert,
    const int* __restrict__ seg_row0,
    const int* __restrict__ seg_rows,
    const int* __restrict__ num_segs_ptr,
    int size_k,
    int size_n,
    float act_limit)
{
    constexpr int MB = FM_MB_GATEUP;
    constexpr int NS = 2;
    extern __shared__ __align__(16) unsigned char fm_smem[];
    half* sh_a = reinterpret_cast<half*>(fm_smem);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(sh_a + FM_STAGES * MB * 16 * FM_TILE_K);
    float* sh_c = reinterpret_cast<float*>(fm_smem);

    const int num_segs = *num_segs_ptr;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tiles_n = size_n / 16;
    const int n_base = blockIdx.x * FM_TILE_N;

    for (int seg = blockIdx.y; seg < num_segs; seg += gridDim.y)
    {
        const int e = seg_expert[seg];
        const int row0 = seg_row0[seg];
        const int rows = seg_rows[seg];
        const uint16_t* const packed[NS] = { gate_ptrs[e], up_ptrs[e] };
        const half* svh_g = gate_svh_ptrs[e] + n_base;
        const half* svh_u = up_svh_ptrs[e] + n_base;
        const half* suh_d = down_suh_ptrs[e] + n_base;

        FragC acc[NS][MB][2];
        fm_mainloop<MB, NS, 0>(h13, size_k, row0, rows, packed, tiles_n, n_base / 16, sh_a, sh_b, acc);

        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
        {
            int rows_mb = rows - mb * 16;
            if (rows_mb <= 0) break;
            fm_stage_acc<NS>(sh_c, acc[0][mb], acc[1][mb], warp, lane);
            __syncthreads();
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
                int r = warp + rr * FM_WARPS;
                if (r < rows_mb)
                {
                    const float* src = sh_c + r * (NS * FM_TILE_N) + lane * 4;
                    float4 g = *reinterpret_cast<const float4*>(src);
                    float4 u = *reinterpret_cast<const float4*>(src + FM_TILE_N);
                    fm_had_row(g, lane);
                    fm_mul_half4(g, svh_g + lane * 4);
                    fm_had_row(u, lane);
                    fm_mul_half4(u, svh_u + lane * 4);
                    // silu(min(g, limit)) * clamp(u, -limit, limit)
                    g.x = fminf(g.x, act_limit); g.y = fminf(g.y, act_limit);
                    g.z = fminf(g.z, act_limit); g.w = fminf(g.w, act_limit);
                    u.x = fminf(fmaxf(u.x, -act_limit), act_limit);
                    u.y = fminf(fmaxf(u.y, -act_limit), act_limit);
                    u.z = fminf(fmaxf(u.z, -act_limit), act_limit);
                    u.w = fminf(fmaxf(u.w, -act_limit), act_limit);
                    // E2 boundaries, in order: torch.sigmoid (1/(1+exp(-g)) at
                    // full precision) * g * u in fp32; act_h.copy_() rounds to
                    // fp16; had_hf_r_128_inner<pre_scale> multiplies by
                    // down.suh in fp16; fp32 Hadamard; fp16 store.
                    // exllamav3 builds with --use_fast_math (expf -> __expf,
                    // approximate division); the double-precision exp and the
                    // IEEE-rounded __fdiv_rn are immune to that flag, so the
                    // result is the same whether this file is compiled inside
                    // exllamav3_ext or as the standalone module.
                    float4 act;
                    act.x = __fdiv_rn(1.0f, 1.0f + (float) exp(-(double) g.x)) * g.x * u.x;
                    act.y = __fdiv_rn(1.0f, 1.0f + (float) exp(-(double) g.y)) * g.y * u.y;
                    act.z = __fdiv_rn(1.0f, 1.0f + (float) exp(-(double) g.z)) * g.z * u.z;
                    act.w = __fdiv_rn(1.0f, 1.0f + (float) exp(-(double) g.w)) * g.w * u.w;
                    half4 ha(__floats2half2_rn(act.x, act.y), __floats2half2_rn(act.z, act.w));
                    half4 hs = *reinterpret_cast<const half4*>(suh_d + lane * 4);
                    ha.x = __hmul2(ha.x, hs.x);
                    ha.y = __hmul2(ha.y, hs.y);
                    act = make_float4(__low2float(ha.x), __high2float(ha.x),
                                      __low2float(ha.y), __high2float(ha.y));
                    fm_had_row(act, lane);
                    half* dst = h2 + (int64_t) (row0 + mb * 16 + r) * size_n + n_base + lane * 4;
                    fm_store_half4(dst, act);
                }
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// down GEMM + output Hadamard + route weight + scatter-add
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(FM_THREADS, 2)
void fm_down_kernel(
    const half* __restrict__ h2,
    const uint16_t* const* __restrict__ down_ptrs,
    const half* const* __restrict__ down_svh_ptrs,
    float* __restrict__ out,
    const int64_t* __restrict__ row_token,
    const half* __restrict__ row_weight,
    const int* __restrict__ seg_expert,
    const int* __restrict__ seg_row0,
    const int* __restrict__ seg_rows,
    const int* __restrict__ num_segs_ptr,
    int size_k,
    int size_n)
{
    constexpr int MB = FM_MB_DOWN;
    constexpr int NS = 2;                       // two adjacent 128-col halves
    constexpr int TILE_N2 = NS * FM_TILE_N;
    extern __shared__ __align__(16) unsigned char fm_smem[];
    half* sh_a = reinterpret_cast<half*>(fm_smem);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(sh_a + FM_STAGES * MB * 16 * FM_TILE_K);
    float* sh_c = reinterpret_cast<float*>(fm_smem);

    const int num_segs = *num_segs_ptr;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tiles_n = size_n / 16;
    const int n_base = blockIdx.x * TILE_N2;

    for (int seg = blockIdx.y; seg < num_segs; seg += gridDim.y)
    {
        const int e = seg_expert[seg];
        const int row0 = seg_row0[seg];
        const int rows = seg_rows[seg];
        const uint16_t* const packed[NS] = { down_ptrs[e], down_ptrs[e] };
        const half* svh = down_svh_ptrs[e] + n_base;

        FragC acc[NS][MB][2];
        fm_mainloop<MB, NS, FM_TILE_N / 16>(h2, size_k, row0, rows, packed, tiles_n, n_base / 16, sh_a, sh_b, acc);

        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
        {
            int rows_mb = rows - mb * 16;
            if (rows_mb <= 0) break;
            fm_stage_acc<NS>(sh_c, acc[0][mb], acc[1][mb], warp, lane);
            __syncthreads();
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
                int r = warp + rr * FM_WARPS;
                if (r < rows_mb)
                {
                    const float* srow = sh_c + r * TILE_N2;
                    int frow = row0 + mb * 16 + r;
                    float w = __half2float(row_weight[frow]);
                    float* dst = out + row_token[frow] * (int64_t) size_n + n_base + lane * 4;
                    #pragma unroll
                    for (int s = 0; s < NS; ++s)
                    {
                        float4 v = *reinterpret_cast<const float4*>(srow + s * FM_TILE_N + lane * 4);
                        fm_had_row(v, lane);
                        fm_mul_half4(v, svh + s * FM_TILE_N + lane * 4);
                        v.x *= w; v.y *= w; v.z *= w; v.w *= w;
                        // Lane-contiguous 16 B vector atomics (sm_90+): one
                        // red.v4 per lane covers the warp's 512 B row span.
                        atomicAdd(reinterpret_cast<float4*>(dst + s * FM_TILE_N), v);
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

void check_ptr_table(const at::Tensor& t, const char* name, int64_t n)
{
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
    TORCH_CHECK(t.scalar_type() == at::kLong && t.dim() == 1 && t.size(0) >= n,
                name, " must be int64[n_exp]");
}

void check_seg(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kInt && t.dim() == 1,
                name, " must be a contiguous int32 CUDA vector");
}

int fm_grid_y(int64_t max_segs)
{
    // Grid-strided over segments: cap the launch so idle CTAs stay cheap.
    int64_t g = max_segs < 1 ? 1 : max_segs;
    if (g > 512) g = 512;
    return (int) g;
}

bool fm_attr_set[2] = { false, false };

}  // namespace

int64_t exl3_fat_moe_tile_rows_gateup() { return FM_MB_GATEUP * 16; }
int64_t exl3_fat_moe_tile_rows_down() { return FM_MB_DOWN * 16; }

void exl3_fat_moe_gather(
    at::Tensor x,
    at::Tensor row_token,
    at::Tensor row_expert,
    at::Tensor suh_ptrs,
    at::Tensor h13,
    at::Tensor num_rows)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kHalf && x.dim() == 2,
                "x must be a contiguous half [tokens, K] CUDA tensor");
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous() && h13.scalar_type() == at::kHalf && h13.dim() == 2,
                "h13 must be a contiguous half [rows_cap, K] CUDA tensor");
    TORCH_CHECK(h13.size(1) == x.size(1) && x.size(1) % 128 == 0, "K must match and be a multiple of 128");
    TORCH_CHECK(row_token.is_cuda() && row_token.scalar_type() == at::kLong && row_token.is_contiguous(),
                "row_token must be int64");
    TORCH_CHECK(row_expert.is_cuda() && row_expert.scalar_type() == at::kInt && row_expert.is_contiguous(),
                "row_expert must be int32");
    TORCH_CHECK(row_token.size(0) >= h13.size(0) && row_expert.size(0) >= h13.size(0),
                "row tables must cover rows_cap");
    check_ptr_table(suh_ptrs, "suh_ptrs", 1);
    check_seg(num_rows, "num_rows");
    int size_k = (int) x.size(1);
    int64_t rows_cap = h13.size(0);
    int64_t gx = (rows_cap + FM_WARPS - 1) / FM_WARPS;
    if (gx > 1024) gx = 1024;
    if (gx < 1) gx = 1;
    dim3 grid((unsigned) gx, size_k / 128);
    fm_gather_kernel<<<grid, FM_THREADS, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const int64_t*>(row_token.data_ptr()),
        reinterpret_cast<const int*>(row_expert.data_ptr()),
        reinterpret_cast<const half* const*>(suh_ptrs.data_ptr()),
        reinterpret_cast<half*>(h13.data_ptr()),
        reinterpret_cast<const int*>(num_rows.data_ptr()),
        size_k);
    cuda_check(cudaPeekAtLastError());
}

void exl3_fat_moe_gateup(
    at::Tensor h13,
    at::Tensor gate_ptrs,
    at::Tensor up_ptrs,
    at::Tensor gate_svh_ptrs,
    at::Tensor up_svh_ptrs,
    at::Tensor down_suh_ptrs,
    at::Tensor h2,
    at::Tensor seg_expert,
    at::Tensor seg_row0,
    at::Tensor seg_rows,
    at::Tensor num_segs,
    double act_limit)
{
    const at::cuda::OptionalCUDAGuard device_guard(h13.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous() && h13.scalar_type() == at::kHalf && h13.dim() == 2,
                "h13 must be a contiguous half [rows_cap, K] CUDA tensor");
    TORCH_CHECK(h2.is_cuda() && h2.is_contiguous() && h2.scalar_type() == at::kHalf && h2.dim() == 2,
                "h2 must be a contiguous half [rows_cap, N] CUDA tensor");
    TORCH_CHECK(h2.size(0) == h13.size(0), "h2 / h13 row capacity mismatch");
    int size_k = (int) h13.size(1);
    int size_n = (int) h2.size(1);
    TORCH_CHECK(size_k % FM_TILE_K == 0 && size_k >= FM_TILE_K, "K must be a multiple of 32");
    TORCH_CHECK(size_n % FM_TILE_N == 0, "N must be a multiple of 128");
    check_ptr_table(gate_ptrs, "gate_ptrs", 1);
    check_ptr_table(up_ptrs, "up_ptrs", gate_ptrs.size(0));
    check_ptr_table(gate_svh_ptrs, "gate_svh_ptrs", gate_ptrs.size(0));
    check_ptr_table(up_svh_ptrs, "up_svh_ptrs", gate_ptrs.size(0));
    check_ptr_table(down_suh_ptrs, "down_suh_ptrs", gate_ptrs.size(0));
    check_seg(seg_expert, "seg_expert");
    check_seg(seg_row0, "seg_row0");
    check_seg(seg_rows, "seg_rows");
    check_seg(num_segs, "num_segs");
    TORCH_CHECK(seg_row0.size(0) == seg_expert.size(0) && seg_rows.size(0) == seg_expert.size(0),
                "segment tables must share a length");

    constexpr int smem = fm_smem_bytes<FM_MB_GATEUP, 2>();
    if (!fm_attr_set[0])
    {
        cudaFuncSetAttribute(fm_gateup_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        fm_attr_set[0] = true;
    }
    dim3 grid(size_n / FM_TILE_N, fm_grid_y(seg_expert.size(0)));
    fm_gateup_kernel<<<grid, FM_THREADS, smem, stream>>>(
        reinterpret_cast<const half*>(h13.data_ptr()),
        reinterpret_cast<const uint16_t* const*>(gate_ptrs.data_ptr()),
        reinterpret_cast<const uint16_t* const*>(up_ptrs.data_ptr()),
        reinterpret_cast<const half* const*>(gate_svh_ptrs.data_ptr()),
        reinterpret_cast<const half* const*>(up_svh_ptrs.data_ptr()),
        reinterpret_cast<const half* const*>(down_suh_ptrs.data_ptr()),
        reinterpret_cast<half*>(h2.data_ptr()),
        reinterpret_cast<const int*>(seg_expert.data_ptr()),
        reinterpret_cast<const int*>(seg_row0.data_ptr()),
        reinterpret_cast<const int*>(seg_rows.data_ptr()),
        reinterpret_cast<const int*>(num_segs.data_ptr()),
        size_k,
        size_n,
        (float) act_limit);
    cuda_check(cudaPeekAtLastError());
}

void exl3_fat_moe_down(
    at::Tensor h2,
    at::Tensor down_ptrs,
    at::Tensor down_svh_ptrs,
    at::Tensor out,
    at::Tensor row_token,
    at::Tensor row_weight,
    at::Tensor seg_expert,
    at::Tensor seg_row0,
    at::Tensor seg_rows,
    at::Tensor num_segs)
{
    const at::cuda::OptionalCUDAGuard device_guard(h2.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(h2.is_cuda() && h2.is_contiguous() && h2.scalar_type() == at::kHalf && h2.dim() == 2,
                "h2 must be a contiguous half [rows_cap, K] CUDA tensor");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kFloat && out.dim() == 2,
                "out must be a contiguous float [tokens, N] CUDA tensor");
    int size_k = (int) h2.size(1);
    int size_n = (int) out.size(1);
    TORCH_CHECK(size_k % FM_TILE_K == 0 && size_k >= FM_TILE_K, "K must be a multiple of 32");
    TORCH_CHECK(size_n % FM_TILE_N == 0, "N must be a multiple of 128");
    TORCH_CHECK(row_token.is_cuda() && row_token.scalar_type() == at::kLong && row_token.is_contiguous()
                && row_token.size(0) >= h2.size(0), "row_token must be int64[rows_cap]");
    TORCH_CHECK(row_weight.is_cuda() && row_weight.scalar_type() == at::kHalf && row_weight.is_contiguous()
                && row_weight.size(0) >= h2.size(0), "row_weight must be half[rows_cap]");
    check_ptr_table(down_ptrs, "down_ptrs", 1);
    check_ptr_table(down_svh_ptrs, "down_svh_ptrs", down_ptrs.size(0));
    check_seg(seg_expert, "seg_expert");
    check_seg(seg_row0, "seg_row0");
    check_seg(seg_rows, "seg_rows");
    check_seg(num_segs, "num_segs");
    TORCH_CHECK(seg_row0.size(0) == seg_expert.size(0) && seg_rows.size(0) == seg_expert.size(0),
                "segment tables must share a length");

    TORCH_CHECK(size_n % (2 * FM_TILE_N) == 0, "N must be a multiple of 256");
    constexpr int smem = fm_smem_bytes<FM_MB_DOWN, 2>();
    if (!fm_attr_set[1])
    {
        cudaFuncSetAttribute(fm_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        fm_attr_set[1] = true;
    }
    dim3 grid(size_n / (2 * FM_TILE_N), fm_grid_y(seg_expert.size(0)));
    fm_down_kernel<<<grid, FM_THREADS, smem, stream>>>(
        reinterpret_cast<const half*>(h2.data_ptr()),
        reinterpret_cast<const uint16_t* const*>(down_ptrs.data_ptr()),
        reinterpret_cast<const half* const*>(down_svh_ptrs.data_ptr()),
        reinterpret_cast<float*>(out.data_ptr()),
        reinterpret_cast<const int64_t*>(row_token.data_ptr()),
        reinterpret_cast<const half*>(row_weight.data_ptr()),
        reinterpret_cast<const int*>(seg_expert.data_ptr()),
        reinterpret_cast<const int*>(seg_row0.data_ptr()),
        reinterpret_cast<const int*>(seg_rows.data_ptr()),
        reinterpret_cast<const int*>(num_segs.data_ptr()),
        size_k,
        size_n);
    cuda_check(cudaPeekAtLastError());
}
