// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "mla_decode_fwd_kernel.hpp"
#include "mla_combine_fwd_kernel.hpp"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>

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
    int num_kv_splits,
    float fp8_decode_scale)
{
    const int batch = qo_indptr.size(0) - 1;
    const int nhead = q.size(1);
    const int num_head_groups = (nhead + TILE_HEADS - 1) / TILE_HEADS;

    // KV buffer layout — Phase B+ (Apr 2026): support both legacy 2D contiguous
    // pools and 4D padded pools (e.g., c4/c128 caches where rows are padded to
    // a 576 B multiple, making `stride(0) > pages_per_row*slot_dim`). Avoiding
    // the previous `kv_buffer.reshape(...)` for the 4D case eliminates a 131 ms
    // float8_copy_kernel that fired on every dispatch in two-shot mode.
    int64_t kv_slot_stride;
    int64_t kv_outer_stride;
    int32_t kv_pages_per_row;
    const void* kv_data_ptr = nullptr;
    if (kv_buffer.dim() == 4) {
        TORCH_CHECK(kv_buffer.size(2) == 1,
                    "expected h_kv=1 for 4D kv_buffer, got ", kv_buffer.size(2));
        const int64_t slot_dim = kv_buffer.size(-1);
        TORCH_CHECK(slot_dim >= QK_HEAD_DIM_T,
                    "kv_buffer slot_dim ", slot_dim,
                    " < QK_HEAD_DIM=", QK_HEAD_DIM_T);
        kv_slot_stride   = kv_buffer.stride(1);
        kv_outer_stride  = kv_buffer.stride(0);
        kv_pages_per_row = static_cast<int32_t>(kv_buffer.size(1));
        kv_data_ptr      = kv_buffer.data_ptr();
    } else {
        const int64_t slot_stride = kv_buffer.size(-1);
        TORCH_CHECK(slot_stride >= QK_HEAD_DIM_T,
                    "kv_buffer slot stride ", slot_stride,
                    " < QK_HEAD_DIM=", QK_HEAD_DIM_T);
        const int64_t n_slots = kv_buffer.numel() / slot_stride;
        auto kv_flat = kv_buffer.reshape({n_slots, slot_stride});
        kv_slot_stride   = kv_flat.stride(0);
        kv_outer_stride  = kv_slot_stride;
        kv_pages_per_row = 1;
        kv_data_ptr      = kv_flat.data_ptr();
    }

    MlaDecodeArgs args;
    args.q_ptr           = q.data_ptr();
    args.kv_ptr          = kv_data_ptr;
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
    args.stride_kv          = kv_slot_stride;
    args.pool_outer_stride  = kv_outer_stride;
    args.pages_per_row      = kv_pages_per_row;
    args.stride_sd_s     = split_data.stride(0);
    args.stride_sd_split = split_data.stride(1);
    args.stride_sd_h     = split_data.stride(2);
    args.stride_lse_s    = split_lse.stride(0);
    args.stride_lse_split = split_lse.stride(1);
    args.stride_lse_h    = split_lse.stride(2);
    args.fp8_decode_scale = fp8_decode_scale;

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
    torch::Tensor kv_buffer,    // fp8 (e4m3fn on gfx950, e4m3fnuz on gfx942)
    torch::Tensor split_data,
    torch::Tensor split_lse,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    float sm_scale,
    int num_kv_splits,
    float fp8_decode_scale = 0.5f)
{
    // Runtime dispatch on q's last dim. Both DSv4 attention shapes are
    // bundled in the same kernel module — V32 (576) and Flash-Base 2604
    // (512). Anything else is a hard error so the caller's adapter never
    // silently routes a mismatched shape into this kernel.
    const int qk_head_dim = q.size(-1);
    if (qk_head_dim == 576) {
        launch_one<576>(q, kv_buffer, split_data, split_lse,
                        qo_indptr, kv_indptr, kv_indices,
                        sm_scale, num_kv_splits, fp8_decode_scale);
    } else if (qk_head_dim == 512) {
        launch_one<512>(q, kv_buffer, split_data, split_lse,
                        qo_indptr, kv_indptr, kv_indices,
                        sm_scale, num_kv_splits, fp8_decode_scale);
    } else {
        TORCH_CHECK(false,
            "CK V32 sparse MLA: unsupported q_head_dim ", qk_head_dim,
            " (expected 576 for V32 or 512 for 2604)");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase A combine kernel launch — N-way online-softmax merge across two
// sources' (split_data, split_lse) tensors, with optional attn_sink fold.
//
// Replaces the (2× aiter.mla_reduce_v1 + Triton merge_two_sparse_attn_outputs
// + Triton _sink_fold_inplace_kernel) pipeline with one CK launch. Caller
// supplies pre-allocated `out` [total_q, H, V] bf16 and `lse` [total_q, H]
// fp32 tensors (no internal allocation).
//
// `split_data_b` / `split_lse_b` may be undefined tensors (`.defined()`==false)
// for the single-source path, in which case S_b=0 and the kernel reduces over
// source-A splits only.
void mla_combine_fwd_launch(
    torch::Tensor split_data_a,                       // [total_q, S_a, H, V] fp32
    torch::Tensor split_lse_a,                        // [total_q, S_a, H, 1] fp32
    c10::optional<torch::Tensor> split_data_b_opt,    // [total_q, S_b, H, V] fp32 or undefined
    c10::optional<torch::Tensor> split_lse_b_opt,     // [total_q, S_b, H, 1] fp32 or undefined
    c10::optional<torch::Tensor> attn_sink_opt,       // [H] fp32 or undefined
    torch::Tensor out,                                // [total_q, H, V] bf16 (caller-allocated)
    torch::Tensor lse)                                // [total_q, H]    fp32 (caller-allocated)
{
    TORCH_CHECK(split_data_a.dim() == 4 && split_lse_a.dim() == 4,
                "split_data_a/split_lse_a must be 4D");
    const int total_q = split_data_a.size(0);
    const int S_a     = split_data_a.size(1);
    const int H       = split_data_a.size(2);
    const int V       = split_data_a.size(3);
    TORCH_CHECK(split_lse_a.size(0) == total_q && split_lse_a.size(1) == S_a
                && split_lse_a.size(2) == H && split_lse_a.size(3) == 1,
                "split_lse_a shape mismatch");
    TORCH_CHECK(out.dim() == 3 && out.size(0) == total_q && out.size(1) == H
                && out.size(2) == V,
                "out shape mismatch");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16,
                "out must be bf16");
    TORCH_CHECK(lse.dim() == 2 && lse.size(0) == total_q && lse.size(1) == H,
                "lse shape mismatch");
    TORCH_CHECK(lse.scalar_type() == at::kFloat,
                "lse must be fp32");
    TORCH_CHECK(out.stride(-1) == 1, "out V-stride must be 1");
    TORCH_CHECK(split_data_a.stride(-1) == 1, "split_data_a V-stride must be 1");

    const bool has_b = split_data_b_opt.has_value() && split_data_b_opt->defined();
    int S_b = 0;
    const float* split_data_b_ptr = nullptr;
    const float* split_lse_b_ptr  = nullptr;
    int64_t stride_sdb_q = 0, stride_sdb_s = 0, stride_sdb_h = 0;
    int64_t stride_lsb_q = 0, stride_lsb_s = 0, stride_lsb_h = 0;
    if (has_b) {
        TORCH_CHECK(split_lse_b_opt.has_value() && split_lse_b_opt->defined(),
                    "split_lse_b must be provided alongside split_data_b");
        const auto& sdb = *split_data_b_opt;
        const auto& slb = *split_lse_b_opt;
        TORCH_CHECK(sdb.dim() == 4 && sdb.size(0) == total_q && sdb.size(2) == H
                    && sdb.size(3) == V,
                    "split_data_b shape mismatch with split_data_a");
        TORCH_CHECK(slb.dim() == 4 && slb.size(0) == total_q && slb.size(2) == H
                    && slb.size(3) == 1,
                    "split_lse_b shape mismatch");
        TORCH_CHECK(sdb.stride(-1) == 1, "split_data_b V-stride must be 1");
        S_b = sdb.size(1);
        TORCH_CHECK(S_a + S_b <= COMBINE_MAX_SPLITS,
                    "S_a+S_b=", S_a + S_b, " exceeds COMBINE_MAX_SPLITS=",
                    COMBINE_MAX_SPLITS);
        split_data_b_ptr = sdb.data_ptr<float>();
        split_lse_b_ptr  = slb.data_ptr<float>();
        stride_sdb_q = sdb.stride(0); stride_sdb_s = sdb.stride(1); stride_sdb_h = sdb.stride(2);
        stride_lsb_q = slb.stride(0); stride_lsb_s = slb.stride(1); stride_lsb_h = slb.stride(2);
    } else {
        TORCH_CHECK(S_a <= COMBINE_MAX_SPLITS,
                    "S_a=", S_a, " exceeds COMBINE_MAX_SPLITS=", COMBINE_MAX_SPLITS);
    }

    const float* attn_sink_ptr = nullptr;
    if (attn_sink_opt.has_value() && attn_sink_opt->defined()) {
        const auto& s = *attn_sink_opt;
        TORCH_CHECK(s.dim() == 1 && s.size(0) == H && s.scalar_type() == at::kFloat,
                    "attn_sink must be [H] fp32");
        attn_sink_ptr = s.data_ptr<float>();
    }

    MlaCombineArgs args{};
    args.split_data_a = split_data_a.data_ptr<float>();
    args.split_lse_a  = split_lse_a.data_ptr<float>();
    args.S_a          = S_a;
    args.stride_sda_q = split_data_a.stride(0);
    args.stride_sda_s = split_data_a.stride(1);
    args.stride_sda_h = split_data_a.stride(2);
    args.stride_lsa_q = split_lse_a.stride(0);
    args.stride_lsa_s = split_lse_a.stride(1);
    args.stride_lsa_h = split_lse_a.stride(2);

    args.split_data_b = split_data_b_ptr;
    args.split_lse_b  = split_lse_b_ptr;
    args.S_b          = S_b;
    args.stride_sdb_q = stride_sdb_q;
    args.stride_sdb_s = stride_sdb_s;
    args.stride_sdb_h = stride_sdb_h;
    args.stride_lsb_q = stride_lsb_q;
    args.stride_lsb_s = stride_lsb_s;
    args.stride_lsb_h = stride_lsb_h;

    args.out_ptr      = reinterpret_cast<__bf16*>(out.data_ptr());
    args.stride_out_q = out.stride(0);
    args.stride_out_h = out.stride(1);
    args.lse_ptr      = lse.data_ptr<float>();
    args.stride_lse_q = lse.stride(0);
    args.stride_lse_h = lse.stride(1);

    args.attn_sink    = attn_sink_ptr;
    args.total_q      = total_q;
    args.H            = H;
    args.V            = V;

    dim3 grid(total_q * H, 1, 1);
    dim3 block(COMBINE_BLOCK_THREADS, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (V == 512) {
        hipLaunchKernelGGL((mla_combine_fwd_kernel<512>),
                           grid, block, 0, stream, args);
    } else {
        TORCH_CHECK(false,
            "CK V32 combine: unsupported V=", V, " (expected 512)");
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
          py::arg("sm_scale"), py::arg("num_kv_splits"),
          py::arg("fp8_decode_scale") = 0.5f);

    m.def("mla_combine_fwd_ck", &ck_mla_sparse_fp8::mla_combine_fwd_launch,
          "CK-tile MLA two-source combine + sink fold (Phase A; replaces "
          "Triton merge_two_sparse_attn_outputs + _sink_fold_inplace_kernel)",
          py::arg("split_data_a"), py::arg("split_lse_a"),
          py::arg("split_data_b") = c10::nullopt,
          py::arg("split_lse_b")  = c10::nullopt,
          py::arg("attn_sink")    = c10::nullopt,
          py::arg("out"),
          py::arg("lse"));
}
