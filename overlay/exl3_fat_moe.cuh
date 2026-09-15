#pragma once

#include <torch/extension.h>

// Grouped fat-expert MoE for EXL3 K4/MCG trellis experts (prefill).
//
// One launch per phase covers every "fat" expert of a layer (an expert whose
// token count exceeds the fused exl3_moe temp rows). Work is described by
// device-side segment tables, so the host never synchronizes on routing.
//
//   exl3_fat_moe_gather   : h13[row] = had128(x[token[row]] * gate_suh[expert[row]])
//   exl3_fat_moe_gateup   : h2 = had128(silu(clamp(had(g)*svh_g)) * clamp(had(u)*svh_u) * down_suh)
//   exl3_fat_moe_down     : out[token] += had128(h2 @ W_down) * svh_d * route_weight
//
// Segment tables (int32, device): seg_expert / seg_row0 / seg_rows describe
// row tiles of the fat-row buffer (rows are grouped by expert, expert order).
// num_segs / num_rows are 1-element int32 device tensors; kernels are launched
// with capacity-sized, grid-strided grids and read the live counts.

void exl3_fat_moe_gather(
    at::Tensor x,            // [tokens, K] half
    at::Tensor row_token,    // [rows_cap] int64
    at::Tensor row_expert,   // [rows_cap] int32
    at::Tensor suh_ptrs,     // [n_exp] int64 (device pointers, half[K])
    at::Tensor h13,          // [rows_cap, K] half (out)
    at::Tensor num_rows);    // [1] int32 device

void exl3_fat_moe_gateup(
    at::Tensor h13,          // [rows_cap, K] half
    at::Tensor gate_ptrs,    // [n_exp] int64 trellis pointers (K/16, N/16, 64) int16
    at::Tensor up_ptrs,
    at::Tensor gate_svh_ptrs,
    at::Tensor up_svh_ptrs,
    at::Tensor down_suh_ptrs,
    at::Tensor h2,           // [rows_cap, N] half (out)
    at::Tensor seg_expert,
    at::Tensor seg_row0,
    at::Tensor seg_rows,
    at::Tensor num_segs,
    double act_limit);

void exl3_fat_moe_down(
    at::Tensor h2,           // [rows_cap, K] half
    at::Tensor down_ptrs,    // trellis pointers (K/16, N/16, 64)
    at::Tensor down_svh_ptrs,
    at::Tensor out,          // [tokens, N] float (accumulated)
    at::Tensor row_token,    // [rows_cap] int64
    at::Tensor row_weight,   // [rows_cap] half
    at::Tensor seg_expert,
    at::Tensor seg_row0,
    at::Tensor seg_rows,
    at::Tensor num_segs);

// Row tile sizes the segment tables must be built with.
int64_t exl3_fat_moe_tile_rows_gateup();
int64_t exl3_fat_moe_tile_rows_down();
