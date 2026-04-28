// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// HIP port of `apply_rotary_emb_triton` from
// /mnt/vast/john/sglang_v4_pr/python/sglang/srt/layers/deepseek_v4_rope.py
//
// Goal: kill the per-launch CPU overhead of Triton's autotune-cache + JIT-cache
// + arg-serialization that dominates wall-time on AMD ROCm. Trace measured
// 194 us CPU per call x 462 calls = 89 ms / window. The GPU work itself is
// trivial (memory-bound, tiny grid at decode). PyTorch's cpp_extension dispatch
// path on ROCm typically lands at 30-50 us, leaving ~70%+ of the per-call cost
// recoverable.
//
// Kernel logic mirrors the Triton kernel exactly (verified bit-equivalent
// math): for each token / head / rope-pair, compute
//   out_real = x_real * freq_real - x_imag * freq_imag
//   out_imag = x_real * freq_imag + x_imag * freq_real    (forward)
//   out_real = x_real * freq_real + x_imag * freq_imag
//   out_imag = x_imag * freq_real - x_real * freq_imag    (inverse / conj)
// where x is bf16 in/out and freqs is fp32 (real/imag interleaved, the result
// of `torch.view_as_real(freqs_cis).flatten(-2)`).
//
// Templated on USE_POS / IS_INVERSE / IS_3D / ROPE_DIM_T at compile time so the
// hot path branches are constant-folded. ROPE_DIM_T = 64 covers both DSv4
// production rope dims (Q rope = 64; KV rope = 64). 128 is wired in for
// future-proofing.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

namespace rope_hip {

// One CTA processes BLOCK_M_T tokens for one head_block. Each thread handles
// one rope-pair (real+imag) within a token, looping over BLOCK_M_T tokens
// internally. Block size = ROPE_DIM_T / 2 (one thread per pair).
//
// Grid: (ceil(n_tokens / BLOCK_M_T), n_heads or 1).
// Thread layout: tid = pair index in [0, ROPE_DIM_T/2).
//
// Templated on x's storage dtype (XT) — covers bf16 (Q/KV rope) and fp32
// (kv_compressed from softmax * bf16 + sum in compress_decode_old).
template <typename XT, int ROPE_DIM_T, bool USE_POS, bool IS_INVERSE, bool IS_3D,
          int BLOCK_M_T, typename PosT>
__global__ void apply_rotary_emb_kernel(
    XT*             __restrict__ x_ptr,
    const float*    __restrict__ freqs_ptr,
    const PosT*     __restrict__ positions_ptr,  // may be nullptr if !USE_POS
    int64_t stride_x_batch,
    int64_t stride_x_head,
    int64_t stride_x_dim,
    int64_t stride_freq_pos,
    int64_t stride_freq_dim,
    int n_tokens
) {
    constexpr int ROPE_PAIRS = ROPE_DIM_T / 2;
    const int pid_m = blockIdx.x;
    const int pid_h = blockIdx.y;
    const int pair_idx = threadIdx.x;
    if (pair_idx >= ROPE_PAIRS) return;

    #pragma unroll
    for (int mi = 0; mi < BLOCK_M_T; ++mi) {
        const int m = pid_m * BLOCK_M_T + mi;
        if (m >= n_tokens) return;

        const int64_t pos = USE_POS ? (int64_t)positions_ptr[m] : (int64_t)m;

        const float* freq_base = freqs_ptr + pos * stride_freq_pos;
        const float fr = freq_base[(pair_idx * 2)     * stride_freq_dim];
        const float fi = freq_base[(pair_idx * 2 + 1) * stride_freq_dim];

        int64_t x_base = (int64_t)m * stride_x_batch;
        if (IS_3D) x_base += (int64_t)pid_h * stride_x_head;
        const int64_t x_real_off = x_base + (int64_t)(pair_idx * 2)     * stride_x_dim;
        const int64_t x_imag_off = x_base + (int64_t)(pair_idx * 2 + 1) * stride_x_dim;

        const float xr = (float)x_ptr[x_real_off];
        const float xi = (float)x_ptr[x_imag_off];

        float out_r, out_i;
        if (IS_INVERSE) {
            out_r = xr * fr + xi * fi;
            out_i = xi * fr - xr * fi;
        } else {
            out_r = xr * fr - xi * fi;
            out_i = xr * fi + xi * fr;
        }

        x_ptr[x_real_off] = (XT)out_r;
        x_ptr[x_imag_off] = (XT)out_i;
    }
}

// Per-shape launch helper. Dispatches the templated kernel based on x dtype
// (bf16 / fp32) and runtime params USE_POS / IS_INVERSE / IS_3D.
template <typename XT, int ROPE_DIM_T>
static void launch_one_xt(
    torch::Tensor x,
    torch::Tensor freqs_real,
    c10::optional<torch::Tensor> positions_opt,
    bool is_inverse,
    bool is_3d
) {
    constexpr int BLOCK_M_T = 64;
    constexpr int ROPE_PAIRS = ROPE_DIM_T / 2;

    const int n_tokens = x.size(0);
    const int n_heads = is_3d ? (int)x.size(1) : 1;

    const int64_t stride_x_batch = x.stride(0);
    const int64_t stride_x_head  = is_3d ? x.stride(1) : 0;
    const int64_t stride_x_dim   = x.stride(-1);
    const int64_t stride_freq_pos = freqs_real.stride(0);
    const int64_t stride_freq_dim = freqs_real.stride(1);

    const void* positions_ptr = nullptr;
    bool positions_is_int64 = false;
    if (positions_opt.has_value() && positions_opt->defined()) {
        const auto pdt = positions_opt->scalar_type();
        TORCH_CHECK(pdt == at::kLong || pdt == at::kInt,
                    "positions must be int32 or int64, got ", pdt);
        positions_is_int64 = (pdt == at::kLong);
        positions_ptr = positions_opt->data_ptr();
    }
    const bool use_pos = (positions_ptr != nullptr);

    auto* x_ptr = reinterpret_cast<XT*>(x.data_ptr());
    const float* freqs_ptr = freqs_real.data_ptr<float>();

    dim3 grid((n_tokens + BLOCK_M_T - 1) / BLOCK_M_T, n_heads, 1);
    dim3 block(ROPE_PAIRS, 1, 1);

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();

    #define LAUNCH_POS_T(POS_T, USE_POS, IS_INVERSE, IS_3D)                       \
        hipLaunchKernelGGL(                                                       \
            (apply_rotary_emb_kernel<XT, ROPE_DIM_T, USE_POS, IS_INVERSE, IS_3D,  \
                                     BLOCK_M_T, POS_T>),                          \
            grid, block, 0, stream,                                               \
            x_ptr, freqs_ptr, reinterpret_cast<const POS_T*>(positions_ptr),      \
            stride_x_batch, stride_x_head, stride_x_dim,                          \
            stride_freq_pos, stride_freq_dim, n_tokens)

    #define LAUNCH(USE_POS, IS_INVERSE, IS_3D)                                    \
        do {                                                                      \
            if (positions_is_int64) {                                             \
                LAUNCH_POS_T(int64_t, USE_POS, IS_INVERSE, IS_3D);                \
            } else {                                                              \
                LAUNCH_POS_T(int32_t, USE_POS, IS_INVERSE, IS_3D);                \
            }                                                                     \
        } while (0)

    if (use_pos) {
        if (is_inverse) {
            if (is_3d) LAUNCH(true,  true,  true);
            else       LAUNCH(true,  true,  false);
        } else {
            if (is_3d) LAUNCH(true,  false, true);
            else       LAUNCH(true,  false, false);
        }
    } else {
        if (is_inverse) {
            if (is_3d) LAUNCH(false, true,  true);
            else       LAUNCH(false, true,  false);
        } else {
            if (is_3d) LAUNCH(false, false, true);
            else       LAUNCH(false, false, false);
        }
    }
    #undef LAUNCH
    #undef LAUNCH_POS_T
}

// dtype dispatch — bf16 hot path + fp32 fallback (kv_compressed in
// compress_decode_old fires fp32 due to softmax * bf16 + sum chain).
template <int ROPE_DIM_T>
static void launch_one(
    torch::Tensor x,
    torch::Tensor freqs_real,
    c10::optional<torch::Tensor> positions_opt,
    bool is_inverse,
    bool is_3d
) {
    if (x.scalar_type() == at::kBFloat16) {
        launch_one_xt<__hip_bfloat16, ROPE_DIM_T>(
            x, freqs_real, positions_opt, is_inverse, is_3d);
    } else if (x.scalar_type() == at::kFloat) {
        launch_one_xt<float, ROPE_DIM_T>(
            x, freqs_real, positions_opt, is_inverse, is_3d);
    } else {
        TORCH_CHECK(false, "x dtype ", x.scalar_type(),
                    " not supported (expected bfloat16 or float32)");
    }
}

void apply_rotary_emb_launch(
    torch::Tensor x,            // bf16, 2D [B, rope_dim] or 3D [B, H, rope_dim]
    torch::Tensor freqs_real,   // fp32, [max_seqlen or B, rope_dim]; freqs_cis
                                //       in real/imag interleaved layout.
    c10::optional<torch::Tensor> positions,  // int64 [B] or undefined
    bool is_inverse
) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat,
                "x must be bfloat16 or float32, got ", x.scalar_type());
    TORCH_CHECK(freqs_real.scalar_type() == at::kFloat, "freqs_real must be float32");
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA/HIP device");
    TORCH_CHECK(freqs_real.is_cuda(), "freqs_real must be on CUDA/HIP device");

    const bool is_3d = (x.dim() == 3);
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be 2D or 3D");

    const int rope_dim = (int)x.size(-1);
    TORCH_CHECK(freqs_real.dim() == 2 && freqs_real.size(-1) == rope_dim,
                "freqs_real must be [_, rope_dim]; got ",
                freqs_real.sizes(), " vs rope_dim=", rope_dim);

    if (positions.has_value() && positions->defined()) {
        const auto pdt = positions->scalar_type();
        TORCH_CHECK(pdt == at::kLong || pdt == at::kInt,
                    "positions must be int32 or int64");
        TORCH_CHECK(positions->is_cuda(), "positions must be on CUDA/HIP device");
        TORCH_CHECK(positions->size(0) == x.size(0),
                    "positions size(0) must match x.size(0); got ",
                    positions->sizes(), " vs ", x.sizes());
    }

    if (rope_dim == 64) {
        launch_one<64>(x, freqs_real, positions, is_inverse, is_3d);
    } else if (rope_dim == 128) {
        launch_one<128>(x, freqs_real, positions, is_inverse, is_3d);
    } else {
        TORCH_CHECK(false, "rope_dim ", rope_dim,
                    " not supported (expected 64 or 128)");
    }
}

} // namespace rope_hip

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_rotary_emb_hip", &rope_hip::apply_rotary_emb_launch,
          "HIP RoPE kernel — drop-in replacement for apply_rotary_emb_triton",
          py::arg("x"),
          py::arg("freqs_real"),
          py::arg("positions") = c10::nullopt,
          py::arg("is_inverse") = false);
}
