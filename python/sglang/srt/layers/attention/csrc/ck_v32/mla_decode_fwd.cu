// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "mla_decode_fwd_kernel.hpp"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace ck_mla_sparse_fp8 {

// Per-shape launch helper. Reshapes the KV buffer to the right inner dim and
// dispatches the templated kernel. Each instantiation is fully specialized at
// compile time (loop trip counts, LDS strides, MFMA tile counts).
template <int QK_HEAD_DIM_T>
static void launch_one(
    torch::Tensor q,
    torch::Tensor kv_buffer,
    torch::Tensor split_data,
    torch::Tensor split_lse,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float sm_scale,
    int num_kv_splits)
{
    const int batch = qo_indptr.size(0) - 1;
    const int nhead = q.size(1);
    const int num_head_groups = (nhead + TILE_HEADS - 1) / TILE_HEADS;

    auto kv_flat = kv_buffer.reshape({-1, QK_HEAD_DIM_T});

    MlaDecodeArgs args;
    args.q_ptr           = q.data_ptr();
    args.kv_ptr          = kv_flat.data_ptr();
    args.split_data_ptr  = split_data.data_ptr<float>();
    args.split_lse_ptr   = split_lse.data_ptr<float>();
    args.qo_indptr       = qo_indptr.data_ptr<int32_t>();
    args.kv_indptr       = kv_indptr.data_ptr<int32_t>();
    args.kv_indices      = kv_indices.data_ptr<int32_t>();
    args.sm_scale        = sm_scale;
    args.nhead           = nhead;
    args.num_kv_splits   = num_kv_splits;
    args.stride_q_s      = q.stride(0);
    args.stride_q_h      = q.stride(1);
    // KV stride in BYTES. fp8 is 1 byte per element, so stride_in_elems
    // (which torch returns) equals stride_in_bytes for fp8 contiguous rows.
    args.stride_kv       = kv_flat.stride(0);
    args.stride_sd_s     = split_data.stride(0);
    args.stride_sd_split = split_data.stride(1);
    args.stride_sd_h     = split_data.stride(2);
    args.stride_lse_s    = split_lse.stride(0);
    args.stride_lse_split = split_lse.stride(1);
    args.stride_lse_h    = split_lse.stride(2);

    dim3 grid(batch, num_head_groups, num_kv_splits);
    dim3 block(BLOCK_SIZE);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    hipLaunchKernelGGL(
        (mla_decode_fwd_kernel<QK_HEAD_DIM_T>),
        grid, block, lds_total<QK_HEAD_DIM_T>(), stream,
        args);
}

void mla_decode_fwd_launch(
    torch::Tensor q,            // bf16 [total_q, nhead, qk_head_dim]
    torch::Tensor kv_buffer,    // fp8_e4m3fnuz [num_page, ..., qk_head_dim]
    torch::Tensor split_data,
    torch::Tensor split_lse,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float sm_scale,
    int num_kv_splits)
{
    // Runtime dispatch on q's last dim. Both DSv4 attention shapes are
    // bundled in the same kernel module — V32 (576) and Flash-Base 2604
    // (512). Anything else is a hard error so the caller's adapter never
    // silently routes a mismatched shape into this kernel.
    const int qk_head_dim = q.size(-1);
    if (qk_head_dim == 576) {
        launch_one<576>(q, kv_buffer, split_data, split_lse,
                        qo_indptr, kv_indptr, kv_indices,
                        sm_scale, num_kv_splits);
    } else if (qk_head_dim == 512) {
        launch_one<512>(q, kv_buffer, split_data, split_lse,
                        qo_indptr, kv_indptr, kv_indices,
                        sm_scale, num_kv_splits);
    } else {
        TORCH_CHECK(false,
            "CK V32 sparse MLA: unsupported q_head_dim ", qk_head_dim,
            " (expected 576 for V32 or 512 for 2604)");
    }
}

} // namespace ck_mla_sparse_fp8

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mla_decode_fwd_ck_sparse_fp8", &ck_mla_sparse_fp8::mla_decode_fwd_launch,
          "CK-tile MLA decode forward FP8 sparse (Stage 1)",
          py::arg("q"), py::arg("kv_buffer"),
          py::arg("split_data"), py::arg("split_lse"),
          py::arg("qo_indptr"), py::arg("kv_indptr"), py::arg("kv_indices"),
          py::arg("sm_scale"), py::arg("num_kv_splits"));
}
