"""HIP fallback for csrc/deepseek_v4/hisparse_transfer.cuh::offload_to_cpu.

The CUDA kernel walks `void**` arrays of per-layer GPU/CPU cache pointers and
issues per-token memcpys. tvm_ffi JIT is unavailable on HIP, and PyTorch has
no public Python API for "memcpy through a uint64 pointer to GPU memory" — so
we build a tiny `torch.utils.cpp_extension.load_inline` shim that calls
`hipMemcpyAsync` on the current stream. Cached after first compile (~3 s on
cold start; reused via `cache_once`).

Layout (from kvcacheio.cuh::transfer_item):
  Each "item" copies the per-token slot from one cache to another. The layout
  details (item_size_bytes, page_size, etc.) live in the cache tensors and
  are NOT visible to this transfer kernel — it just memcpys whatever blob
  the caller has set up. So our shim just needs item_size_bytes, which the
  caller knows.
"""
from __future__ import annotations

from typing import Any

import torch

from sglang.jit_kernel.utils import cache_once


@cache_once
def _load_hip_memcpy_module() -> Any:
    import torch.utils.cpp_extension as _cpp

    cpp_source = """
    #include <torch/extension.h>
    """

    # HIP source: take 4 int64 tensors (gpu_ptrs, cpu_ptrs, gpu_indices,
    # cpu_indices), num_items, num_layers, item_size_bytes; for each layer
    # j and each item i, hipMemcpyAsync per-item-slot from gpu to cpu.
    cuda_source = """
    #include <torch/extension.h>
    #include <hip/hip_runtime.h>
    #include <c10/hip/HIPStream.h>

    // Layout constants (must match include/sgl_kernel/deepseek_v4/kvcacheio.cuh)
    static constexpr int64_t kGPUPageSize   = 64;
    static constexpr int64_t kGPUPageBits   = 6;
    static constexpr int64_t kValueBytes    = 576;
    static constexpr int64_t kScaleBytes    = 8;
    static constexpr int64_t kCPUItemBytes  = kValueBytes + kScaleBytes;  // 584
    // GPU page = ceil(584*64/576)*576 = 64*576 + 64*8 padded to 576
    static constexpr int64_t kGPUPageBytes  = ((kCPUItemBytes * kGPUPageSize + 575) / 576) * 576;
    static constexpr int64_t kGPUScaleOffset = kValueBytes * kGPUPageSize;

    void offload_to_cpu_impl(
        torch::Tensor gpu_ptrs,    // uint64 [L]
        torch::Tensor cpu_ptrs,    // uint64 [L]
        torch::Tensor gpu_indices, // int64  [N]
        torch::Tensor cpu_indices  // int64  [N]
    ) {
        TORCH_CHECK(gpu_ptrs.dtype() == torch::kUInt64,
                    "gpu_ptrs must be uint64");
        TORCH_CHECK(cpu_ptrs.dtype() == torch::kUInt64,
                    "cpu_ptrs must be uint64");
        TORCH_CHECK(gpu_indices.dtype() == torch::kInt64,
                    "gpu_indices must be int64");
        TORCH_CHECK(cpu_indices.dtype() == torch::kInt64,
                    "cpu_indices must be int64");

        const int64_t L = gpu_ptrs.numel();
        const int64_t N = gpu_indices.numel();
        TORCH_CHECK(cpu_ptrs.numel() == L);
        TORCH_CHECK(cpu_indices.numel() == N);
        if (L == 0 || N == 0) return;

        // Pull pointer + index arrays to CPU once (small).
        auto gpu_ptrs_cpu = gpu_ptrs.to(torch::kCPU);
        auto cpu_ptrs_cpu = cpu_ptrs.to(torch::kCPU);
        auto gpu_idx_cpu  = gpu_indices.to(torch::kCPU);
        auto cpu_idx_cpu  = cpu_indices.to(torch::kCPU);

        const uint64_t* gp = gpu_ptrs_cpu.data_ptr<uint64_t>();
        const uint64_t* cp = cpu_ptrs_cpu.data_ptr<uint64_t>();
        const int64_t*  gi = gpu_idx_cpu.data_ptr<int64_t>();
        const int64_t*  ci = cpu_idx_cpu.data_ptr<int64_t>();

        hipStream_t stream = c10::hip::getCurrentHIPStream();

        // Per item: 2 DeviceToHost memcpys (value 576 B + scale 8 B), with the
        // GPU side paged (page = idx >> 6, page_offset = idx & 63) and the
        // CPU side linear ([idx*584 .. idx*584+584)).
        for (int64_t j = 0; j < L; ++j) {
            const uint64_t gpu_base = gp[j];
            const uint64_t cpu_base = cp[j];
            for (int64_t i = 0; i < N; ++i) {
                const int64_t gi_v = gi[i];
                const int64_t ci_v = ci[i];
                const int64_t page_num = gi_v >> kGPUPageBits;
                const int64_t page_off = gi_v & (kGPUPageSize - 1);
                const uint64_t gpu_page = gpu_base + (uint64_t)(page_num * kGPUPageBytes);
                const uint64_t gpu_value = gpu_page + (uint64_t)(page_off * kValueBytes);
                const uint64_t gpu_scale = gpu_page + (uint64_t)(kGPUScaleOffset + page_off * kScaleBytes);
                const uint64_t cpu_value = cpu_base + (uint64_t)(ci_v * kCPUItemBytes);
                const uint64_t cpu_scale = cpu_value + (uint64_t)kValueBytes;

                hipError_t e1 = hipMemcpyAsync(
                    (void*)cpu_value, (void*)gpu_value,
                    (size_t)kValueBytes, hipMemcpyDeviceToHost, stream
                );
                TORCH_CHECK(e1 == hipSuccess,
                            "hipMemcpyAsync(value) failed: ", hipGetErrorString(e1));
                hipError_t e2 = hipMemcpyAsync(
                    (void*)cpu_scale, (void*)gpu_scale,
                    (size_t)kScaleBytes, hipMemcpyDeviceToHost, stream
                );
                TORCH_CHECK(e2 == hipSuccess,
                            "hipMemcpyAsync(scale) failed: ", hipGetErrorString(e2));
            }
        }
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("offload_to_cpu_impl", &offload_to_cpu_impl,
              "HIP per-item DeviceToHost memcpy through pointer arrays");
    }
    """

    return _cpp.load_inline(
        name="hisparse_transfer_hip",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        with_cuda=True,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def offload_to_host_hip(
    gpu_ptrs: torch.Tensor,
    cpu_ptrs: torch.Tensor,
    gpu_indices: torch.Tensor,
    cpu_indices: torch.Tensor,
) -> None:
    """HIP entry point. Mirrors the CUDA kernel's contract — the layout
    constants (kValueBytes=576, kScaleBytes=8, kGPUPageSize=64, etc.) are
    baked into the C++ shim at compile time, matching kvcacheio.cuh.
    """
    if gpu_ptrs.dtype != torch.uint64:
        gpu_ptrs = gpu_ptrs.view(torch.uint64)
    if cpu_ptrs.dtype != torch.uint64:
        cpu_ptrs = cpu_ptrs.view(torch.uint64)
    if gpu_indices.dtype != torch.int64:
        gpu_indices = gpu_indices.to(torch.int64)
    if cpu_indices.dtype != torch.int64:
        cpu_indices = cpu_indices.to(torch.int64)

    module = _load_hip_memcpy_module()
    module.offload_to_cpu_impl(gpu_ptrs, cpu_ptrs, gpu_indices, cpu_indices)
