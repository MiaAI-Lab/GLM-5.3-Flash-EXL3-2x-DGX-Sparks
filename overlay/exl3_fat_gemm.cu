#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"
#include "exl3_fat_gemm.cuh"

namespace {

constexpr int FAT_THREADS = 256;
constexpr int FAT_TILE_M = 128;
constexpr int FAT_TILE_K = 16;
constexpr int FAT_TILE_N = 128;
constexpr int FAT_M_BLOCKS = FAT_TILE_M / 16;
constexpr int FAT_N_BLOCKS = FAT_TILE_N / 16;
constexpr int FAT_PACKED_WORDS = 4 * 16;
constexpr float HAD_SCALE = 0.088388347648f;

__device__ inline void fat_had_ff_128(
    const float* input_ptr,
    float* output_ptr,
    const half* scale)
{
    int lane = threadIdx.x & 31;
    float4 v = reinterpret_cast<const float4*>(input_ptr)[lane];

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
    v.x *= HAD_SCALE;
    v.y *= HAD_SCALE;
    v.z *= HAD_SCALE;
    v.w *= HAD_SCALE;

    half4 scales = reinterpret_cast<const half4*>(scale)[lane];
    v.x *= __low2float(scales.x);
    v.y *= __high2float(scales.x);
    v.z *= __low2float(scales.y);
    v.w *= __high2float(scales.y);
    reinterpret_cast<float4*>(output_ptr)[lane] = v;
}

template <bool scatter>
__global__ __launch_bounds__(FAT_THREADS)
void exl3_fat_gemm_kernel(
    const half* __restrict__ a,
    const uint16_t* __restrict__ packed,
    float* __restrict__ out,
    const half* __restrict__ svh,
    const int64_t* __restrict__ token_idx,
    const half* __restrict__ route_weight,
    int size_m,
    int size_k,
    int size_n)
{
    extern __shared__ unsigned char shared_raw[];
    half* sh_a = reinterpret_cast<half*>(shared_raw);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(sh_a + FAT_TILE_M * FAT_TILE_K);
    float* sh_c = reinterpret_cast<float*>(sh_b + FAT_N_BLOCKS * FAT_PACKED_WORDS);

    int t = threadIdx.x;
    int warp = t / 32;
    int lane = t & 31;
    int m_base = blockIdx.y * FAT_TILE_M;
    int n_base = blockIdx.x * FAT_TILE_N;
    int tiles_n = size_n / 16;

    FragC frag_c[FAT_M_BLOCKS][2];
    #pragma unroll
    for (int mb = 0; mb < FAT_M_BLOCKS; ++mb)
    {
        frag_c[mb][0] = {};
        frag_c[mb][1] = {};
    }

    for (int k_block = 0; k_block < size_k / FAT_TILE_K; ++k_block)
    {
        int a_row = t / 2;
        int a_col8 = t & 1;
        int a_dst_col8 = a_col8 ^ ((a_row >> 2) & 1);
        int4 a_value = {};
        if (m_base + a_row < size_m)
        {
            const int4* a_src = reinterpret_cast<const int4*>(
                a + (m_base + a_row) * size_k + k_block * FAT_TILE_K);
            a_value = a_src[a_col8];
        }
        reinterpret_cast<int4*>(sh_a)[a_row * 2 + a_dst_col8] = a_value;

        if (t < 64)
        {
            const int4* b_src = reinterpret_cast<const int4*>(
                packed + (k_block * tiles_n + n_base / 16) * FAT_PACKED_WORDS);
            reinterpret_cast<int4*>(sh_b)[t] = b_src[t];
        }
        __syncthreads();

        FragB frag_b0;
        FragB frag_b1;
        const uint32_t* warp_b = reinterpret_cast<const uint32_t*>(
            sh_b + warp * FAT_PACKED_WORDS);
        dq_dispatch<4, 1>(warp_b, lane << 3, frag_b0, frag_b1);

        #pragma unroll
        for (int mb = 0; mb < FAT_M_BLOCKS; ++mb)
        {
            FragA frag_a;
            int row = (lane % 8) + 8 * ((lane / 8) % 2) + mb * 16;
            int base_col = lane / 16;
            int swizzled_col = base_col ^ ((row >> 2) & 1);
            ldsm4(frag_a, reinterpret_cast<int4*>(sh_a) + row * 2 + swizzled_col);
            ptx_mma_m16n8k16(frag_a, frag_b0, frag_c[mb][0]);
            ptx_mma_m16n8k16(frag_a, frag_b1, frag_c[mb][1]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mb = 0; mb < FAT_M_BLOCKS; ++mb)
    {
        int rows = min(16, size_m - (m_base + mb * 16));
        if (rows <= 0) break;
        int row0 = lane / 4;
        int row1 = row0 + 8;
        int col = (lane % 4) * 2;
        int n0 = warp * 16;
        if (row0 < rows)
        {
            float* dst0 = sh_c + row0 * FAT_TILE_N + n0 + col;
            dst0[0] = frag_c[mb][0][0];
            dst0[1] = frag_c[mb][0][1];
            dst0[8] = frag_c[mb][1][0];
            dst0[9] = frag_c[mb][1][1];
        }
        if (row1 < rows)
        {
            float* dst1 = sh_c + row1 * FAT_TILE_N + n0 + col;
            dst1[0] = frag_c[mb][0][2];
            dst1[1] = frag_c[mb][0][3];
            dst1[8] = frag_c[mb][1][2];
            dst1[9] = frag_c[mb][1][3];
        }
        __syncthreads();

        for (int row = warp; row < rows; row += 8)
        {
            fat_had_ff_128(
                sh_c + row * FAT_TILE_N,
                sh_c + row * FAT_TILE_N,
                svh + n_base);
        }
        __syncthreads();

        for (int i = t; i < rows * FAT_TILE_N; i += FAT_THREADS)
        {
            int row = i / FAT_TILE_N;
            int col_out = i % FAT_TILE_N;
            int source_row = m_base + mb * 16 + row;
            float value = sh_c[i];
            if constexpr (scatter)
            {
                int64_t destination = token_idx[source_row];
                value *= __half2float(route_weight[source_row]);
                // One route per token reaches a given expert, and expert
                // launches share this stream, so this accumulation is race-free.
                out[destination * size_n + n_base + col_out] += value;
            }
            else
            {
                out[source_row * size_n + n_base + col_out] = value;
            }
        }
        __syncthreads();
    }
}

void check_common(
    const at::Tensor& a,
    const at::Tensor& packed,
    const at::Tensor& out,
    const at::Tensor& svh,
    int64_t K,
    bool mcg,
    bool mul1)
{
    TORCH_CHECK(a.is_cuda() && packed.is_cuda() && out.is_cuda() && svh.is_cuda(),
                "exl3_fat_gemm tensors must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && packed.is_contiguous() && out.is_contiguous() && svh.is_contiguous(),
                "exl3_fat_gemm tensors must be contiguous");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "a must be float16");
    TORCH_CHECK(packed.scalar_type() == at::kShort, "packed must be int16");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
    TORCH_CHECK(svh.scalar_type() == at::kHalf, "svh must be float16");
    TORCH_CHECK(a.dim() == 2 && packed.dim() == 3 && out.dim() == 2 && svh.dim() == 1,
                "exl3_fat_gemm expects rank-2/rank-3 tensors");
    TORCH_CHECK(K == 4 && mcg && !mul1,
                "exl3_fat_gemm currently supports only K4 MCG tensors");
    TORCH_CHECK(a.size(1) == packed.size(0) * 16,
                "a K dimension does not match packed tensor");
    TORCH_CHECK(svh.numel() == packed.size(1) * 16,
                "svh N dimension does not match packed tensor");
    TORCH_CHECK(svh.numel() % FAT_TILE_N == 0,
                "output dimension must be divisible by 128");
    TORCH_CHECK(packed.size(2) == FAT_PACKED_WORDS,
                "packed K4 block width must be 64 int16 words");
    TORCH_CHECK(a.device() == packed.device() && a.device() == out.device() && a.device() == svh.device(),
                "exl3_fat_gemm tensors must share a device");
}

template <bool scatter>
void launch(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    int size_m = static_cast<int>(a.size(0));
    int size_k = static_cast<int>(a.size(1));
    int size_n = static_cast<int>(svh.numel());
    dim3 block(FAT_THREADS);
    dim3 grid(size_n / FAT_TILE_N, (size_m + FAT_TILE_M - 1) / FAT_TILE_M);
    size_t shared = FAT_TILE_M * FAT_TILE_K * sizeof(half)
                  + FAT_N_BLOCKS * FAT_PACKED_WORDS * sizeof(uint16_t)
                  + 16 * FAT_TILE_N * sizeof(float);
    exl3_fat_gemm_kernel<scatter><<<grid, block, shared, stream>>>(
        reinterpret_cast<const half*>(a.data_ptr()),
        reinterpret_cast<const uint16_t*>(packed.data_ptr()),
        reinterpret_cast<float*>(out.data_ptr()),
        reinterpret_cast<const half*>(svh.data_ptr()),
        scatter ? reinterpret_cast<const int64_t*>(token_idx.data_ptr()) : nullptr,
        scatter ? reinterpret_cast<const half*>(route_weight.data_ptr()) : nullptr,
        size_m,
        size_k,
        size_n);
    cuda_check(cudaPeekAtLastError());
}

}  // namespace

void exl3_fat_gemm(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    int64_t K,
    bool mcg,
    bool mul1)
{
    check_common(a, packed, out, svh, K, mcg, mul1);
    TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == svh.numel(),
                "out shape must be [M, N]");
    launch<false>(a, packed, out, svh, at::Tensor(), at::Tensor());
}

void exl3_fat_gemm_scatter(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight,
    int64_t K,
    bool mcg,
    bool mul1)
{
    check_common(a, packed, out, svh, K, mcg, mul1);
    TORCH_CHECK(out.size(1) == svh.numel(), "out N dimension mismatch");
    TORCH_CHECK(token_idx.is_cuda() && route_weight.is_cuda(),
                "routing tensors must be CUDA tensors");
    TORCH_CHECK(token_idx.is_contiguous() && route_weight.is_contiguous(),
                "routing tensors must be contiguous");
    TORCH_CHECK(token_idx.scalar_type() == at::kLong, "token_idx must be int64");
    TORCH_CHECK(route_weight.scalar_type() == at::kHalf, "route_weight must be float16");
    TORCH_CHECK(token_idx.numel() == a.size(0) && route_weight.numel() == a.size(0),
                "routing tensors must have M elements");
    TORCH_CHECK(token_idx.device() == a.device() && route_weight.device() == a.device(),
                "routing tensors must share a device with a");
    launch<true>(a, packed, out, svh, token_idx, route_weight);
}
