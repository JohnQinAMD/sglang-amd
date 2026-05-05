// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// HIP port of `make_swa_ring_buffer_indices`. One warp per query token; the
// batch_size scan computes cu_seqlens_q on-the-fly so callers can pass
// `extend_seq_lens` directly without a separate cumsum/pad. Templated on the
// seq-lens dtype to accept int32 or int64 without a host-side cast.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

namespace swa_indices_hip {

// gfx950 wave64. Use a compile-time constant matching the launch config.
constexpr int WARP_SIZE = 64;
constexpr int WARPS_PER_BLOCK = 4;  // 256 threads/block

template <typename SeqT, int SWA_WINDOW>
__global__ __launch_bounds__(WARP_SIZE * WARPS_PER_BLOCK)
void make_swa_indices_kernel(
    const SeqT*    __restrict__ seq_lens_k,
    const SeqT*    __restrict__ seq_lens_q,
    int32_t*       __restrict__ swa_indices,
    int batch_size,
    int num_q_tokens
) {
    static_assert(SWA_WINDOW % WARP_SIZE == 0,
                  "SWA_WINDOW must be a multiple of WARP_SIZE");
    constexpr int CELLS_PER_LANE = SWA_WINDOW / WARP_SIZE;

    const int warp_in_block = threadIdx.y;
    const int lane          = threadIdx.x;
    const int token_id      = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (token_id >= num_q_tokens) return;

    // Find seq_idx: first s such that cum_so_far <= token_id < cum_so_far + qo_len_s.
    // Computes cu_seqlens_q on-the-fly so the wrapper doesn't need a separate
    // cumsum+pad preamble (saves 2 kernel launches per call).
    int seq_idx = 0;
    int cum_qo_len = 0;
    int qo_at_seq_idx = 0;
    #pragma unroll 1
    for (int s = 0; s < batch_size; ++s) {
        int qo_len_s = (int)seq_lens_q[s];
        if (cum_qo_len + qo_len_s > token_id) {
            seq_idx = s;
            qo_at_seq_idx = qo_len_s;
            break;
        }
        cum_qo_len += qo_len_s;
    }

    const int kv_len      = (int)seq_lens_k[seq_idx];
    const int qo_len      = qo_at_seq_idx;
    const int prefix_len  = kv_len - qo_len;
    const int curr_qo_idx = token_id - cum_qo_len;
    const int end_abs_pos = prefix_len + curr_qo_idx + 1;
    const int start_abs_pos = end_abs_pos > SWA_WINDOW ? end_abs_pos - SWA_WINDOW : 0;
    const int old_kv_start = seq_idx * SWA_WINDOW;
    const int new_kv_start = batch_size * SWA_WINDOW + cum_qo_len;

    int32_t* row = swa_indices + (size_t)token_id * SWA_WINDOW;
    #pragma unroll
    for (int c = 0; c < CELLS_PER_LANE; ++c) {
        const int j = c * WARP_SIZE + lane;
        const int abs_pos = start_abs_pos + j;
        int32_t value;
        if (abs_pos < end_abs_pos) {
            if (abs_pos < prefix_len) {
                value = old_kv_start + (abs_pos % SWA_WINDOW);
            } else {
                value = new_kv_start + (abs_pos - prefix_len);
            }
        } else {
            value = -1;
        }
        row[j] = value;
    }
}

template <typename SeqT, int SWA_WINDOW>
static void launch_one_dtype_window(
    const torch::Tensor& seq_lens_k,
    const torch::Tensor& seq_lens_q,
    torch::Tensor&       swa_indices
) {
    const int num_q_tokens = (int)swa_indices.size(0);
    const int batch_size   = (int)seq_lens_k.size(0);
    if (num_q_tokens == 0) return;

    dim3 block(WARP_SIZE, WARPS_PER_BLOCK, 1);
    dim3 grid((num_q_tokens + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1);

    auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
    hipLaunchKernelGGL(
        (make_swa_indices_kernel<SeqT, SWA_WINDOW>),
        grid, block, 0, stream.stream(),
        seq_lens_k.data_ptr<SeqT>(),
        seq_lens_q.data_ptr<SeqT>(),
        swa_indices.data_ptr<int32_t>(),
        batch_size,
        num_q_tokens
    );
    TORCH_CHECK(hipGetLastError() == hipSuccess, "make_swa_indices_kernel launch failed");
}

template <int SWA_WINDOW>
static void launch_one_window(
    const torch::Tensor& seq_lens_k,
    const torch::Tensor& seq_lens_q,
    torch::Tensor&       swa_indices
) {
    const auto dt = seq_lens_k.scalar_type();
    if (dt == at::kInt) {
        launch_one_dtype_window<int32_t, SWA_WINDOW>(seq_lens_k, seq_lens_q, swa_indices);
    } else if (dt == at::kLong) {
        launch_one_dtype_window<int64_t, SWA_WINDOW>(seq_lens_k, seq_lens_q, swa_indices);
    } else {
        TORCH_CHECK(false, "seq_lens dtype must be int32 or int64");
    }
}

void make_swa_indices_launch(
    torch::Tensor seq_lens_k,    // int32/int64 [batch_size]
    torch::Tensor seq_lens_q,    // int32/int64 [batch_size]
    torch::Tensor swa_indices    // int32 [num_q_tokens, swa_window]   (output)
) {
    TORCH_CHECK(seq_lens_k.scalar_type() == seq_lens_q.scalar_type(),
                "seq_lens_k and seq_lens_q must have same dtype");
    TORCH_CHECK(swa_indices.scalar_type() == at::kInt, "swa_indices must be int32");
    TORCH_CHECK(seq_lens_k.is_cuda(), "seq_lens_k must be on device");
    TORCH_CHECK(seq_lens_q.is_cuda(), "seq_lens_q must be on device");
    TORCH_CHECK(swa_indices.is_cuda(), "swa_indices must be on device");
    TORCH_CHECK(swa_indices.dim() == 2, "swa_indices must be 2D");
    TORCH_CHECK(swa_indices.is_contiguous(), "swa_indices must be contiguous");
    TORCH_CHECK(seq_lens_k.size(0) == seq_lens_q.size(0),
                "seq_lens_k and seq_lens_q must have same length");

    const int swa_window = (int)swa_indices.size(1);
    if      (swa_window == 64)   launch_one_window<64>(seq_lens_k, seq_lens_q, swa_indices);
    else if (swa_window == 128)  launch_one_window<128>(seq_lens_k, seq_lens_q, swa_indices);
    else if (swa_window == 256)  launch_one_window<256>(seq_lens_k, seq_lens_q, swa_indices);
    else if (swa_window == 512)  launch_one_window<512>(seq_lens_k, seq_lens_q, swa_indices);
    else {
        TORCH_CHECK(false, "swa_window ", swa_window,
                    " not supported (expected 64/128/256/512)");
    }
}

} // namespace swa_indices_hip

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_swa_indices_hip", &swa_indices_hip::make_swa_indices_launch,
          "HIP fused kernel — replacement for the broken TileLang make_swa_prefill_indices",
          py::arg("seq_lens_k"),
          py::arg("seq_lens_q"),
          py::arg("swa_indices"));
}
