#pragma once

#include <torch/extension.h>

void exl3_fat_gemm(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_gemm_pair(
    at::Tensor a,
    at::Tensor packed_left,
    at::Tensor packed_right,
    at::Tensor out,
    at::Tensor svh_left,
    at::Tensor svh_right,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_swiglu(at::Tensor input, at::Tensor output, double limit);

void exl3_fat_gemm_scatter(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight,
    int64_t K,
    bool mcg,
    bool mul1);
