// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// HIP port of `expand_seq_lens` from
// /mnt/vast/john/sglang_v4_pr/python/sglang/srt/layers/attention/compressed/paged_prefill.py
//
// Replaces the CPU Python loop:
//   for i, (kv_len, qo_len) in enumerate(zip(seq_lens, extend_seq_lens)):
//       seq_lens_expanded[offset:offset+qo_len] = arange(kv_len-qo_len+1, kv_len+1)
//       expanded_idx[offset:offset+qo_len] = i
//       offset += qo_len
// + the trailing `.to(device, non_blocking=True)` H2D copies.
//
// Algorithm per output token t:
//   find seq_idx s.t. cum_qo[seq_idx] <= t < cum_qo[seq_idx+1]
//   seq_lens_expanded[t]    = (kv_len - qo_len) + 1 + (t - cum_qo[seq_idx])
//                          = kv_len - qo_len + 1 + curr_qo_idx
//   expanded_idx_to_unexpanded_idx[t] = seq_idx
//
// One thread = one output token. cu_seqlens_q is computed inline by the
// scan (same trick as make_swa_indices). seq-lens dtype is templated so
// callers don't need to cast.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

namespace expand_seq_lens_hip {

constexpr int BLOCK = 256;

template <typename SeqT>
__global__ __launch_bounds__(BLOCK)
void expand_seq_lens_kernel(
    const SeqT*    __restrict__ seq_lens,
    const SeqT*    __restrict__ extend_seq_lens,
    int32_t*       __restrict__ out_seq_lens_expanded,
    int32_t*       __restrict__ out_expanded_idx,
    int batch_size,
    int num_q_tokens
) {
    const int t = blockIdx.x * BLOCK + threadIdx.x;
    if (t >= num_q_tokens) return;

    // Per-token serial scan to find seq_idx + cum_qo. batch_size is small (1-8
    // in production), and the reads are uniform-ish across the warp so the
    // compiler keeps them scalar.
    int seq_idx = 0;
    int cum_qo_len = 0;
    int qo_at_seq_idx = 0;
    #pragma unroll 1
    for (int s = 0; s < batch_size; ++s) {
        int qo_len_s = (int)extend_seq_lens[s];
        if (cum_qo_len + qo_len_s > t) {
            seq_idx = s;
            qo_at_seq_idx = qo_len_s;
            break;
        }
        cum_qo_len += qo_len_s;
    }

    const int kv_len = (int)seq_lens[seq_idx];
    const int qo_len = qo_at_seq_idx;
    const int curr_qo_idx = t - cum_qo_len;

    // arange(kv_len - qo_len + 1, kv_len + 1)[curr_qo_idx]
    out_seq_lens_expanded[t] = kv_len - qo_len + 1 + curr_qo_idx;
    out_expanded_idx[t]      = seq_idx;
}

template <typename SeqT>
static void launch_one_dtype(
    const torch::Tensor& seq_lens,
    const torch::Tensor& extend_seq_lens,
    torch::Tensor&       out_seq_lens_expanded,
    torch::Tensor&       out_expanded_idx
) {
    const int num_q_tokens = (int)out_seq_lens_expanded.size(0);
    const int batch_size   = (int)seq_lens.size(0);
    if (num_q_tokens == 0) return;

    dim3 block(BLOCK, 1, 1);
    dim3 grid((num_q_tokens + BLOCK - 1) / BLOCK, 1, 1);

    auto stream = at::cuda::getCurrentHIPStreamMasqueradingAsCUDA();
    hipLaunchKernelGGL(
        (expand_seq_lens_kernel<SeqT>),
        grid, block, 0, stream.stream(),
        seq_lens.data_ptr<SeqT>(),
        extend_seq_lens.data_ptr<SeqT>(),
        out_seq_lens_expanded.data_ptr<int32_t>(),
        out_expanded_idx.data_ptr<int32_t>(),
        batch_size,
        num_q_tokens
    );
}

void expand_seq_lens_launch(
    torch::Tensor seq_lens,                // int32/int64 [batch_size]
    torch::Tensor extend_seq_lens,         // int32/int64 [batch_size]
    torch::Tensor out_seq_lens_expanded,   // int32 [num_q_tokens] (output)
    torch::Tensor out_expanded_idx         // int32 [num_q_tokens] (output)
) {
    TORCH_CHECK(seq_lens.scalar_type() == extend_seq_lens.scalar_type(),
                "seq_lens and extend_seq_lens must have the same dtype");
    TORCH_CHECK(out_seq_lens_expanded.scalar_type() == at::kInt,
                "out_seq_lens_expanded must be int32");
    TORCH_CHECK(out_expanded_idx.scalar_type() == at::kInt,
                "out_expanded_idx must be int32");
    TORCH_CHECK(seq_lens.is_cuda() && extend_seq_lens.is_cuda(),
                "inputs must be on device");
    TORCH_CHECK(out_seq_lens_expanded.is_cuda() && out_expanded_idx.is_cuda(),
                "outputs must be on device");
    TORCH_CHECK(out_seq_lens_expanded.numel() == out_expanded_idx.numel(),
                "output sizes must match");
    TORCH_CHECK(seq_lens.size(0) == extend_seq_lens.size(0),
                "seq_lens and extend_seq_lens must have same length");

    const auto dt = seq_lens.scalar_type();
    if (dt == at::kInt) {
        launch_one_dtype<int32_t>(seq_lens, extend_seq_lens,
                                  out_seq_lens_expanded, out_expanded_idx);
    } else if (dt == at::kLong) {
        launch_one_dtype<int64_t>(seq_lens, extend_seq_lens,
                                  out_seq_lens_expanded, out_expanded_idx);
    } else {
        TORCH_CHECK(false, "seq_lens dtype must be int32 or int64");
    }
}

} // namespace expand_seq_lens_hip

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("expand_seq_lens_hip", &expand_seq_lens_hip::expand_seq_lens_launch,
          "HIP fused kernel — replacement for the Python expand_seq_lens loop",
          py::arg("seq_lens"),
          py::arg("extend_seq_lens"),
          py::arg("out_seq_lens_expanded"),
          py::arg("out_expanded_idx"));
}
