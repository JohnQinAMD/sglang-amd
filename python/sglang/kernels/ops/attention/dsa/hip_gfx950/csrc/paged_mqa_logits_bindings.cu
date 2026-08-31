// Torch pybind shim for csrc/paged_mqa_logits.cu .
//
// The kernel source exposes only a C entry point (`launch_logits`), which the
// KernelForge bench harness reaches through ctypes. Production must go through
// the same torch extension build as the rest of the stack, so this file adds
// the torch binding. It contains no kernel logic and no scheduling decision.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

extern "C" int launch_logits(const void *q, const void *kv, const void *wgt,
                             const void *seqlens, const void *ptable, void *out,
                             int batch, int heads, int max_pages,
                             int out_stride, int blocks_per_row, int warps,
                             void *stream, void *ghist, int hist_bits,
                             int topk);

static void logits_hist(at::Tensor q, at::Tensor kv, at::Tensor weights,
                        at::Tensor seqlens, at::Tensor page_table,
                        at::Tensor out, at::Tensor ghist,
                        int64_t blocks_per_row, int64_t hist_bits, int64_t topk,
                        int64_t warps) {
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(weights.is_contiguous() && weights.scalar_type() == at::kFloat,
              "weights must be contiguous fp32");
  TORCH_CHECK(out.scalar_type() == at::kFloat && out.stride(1) == 1,
              "logits out must be fp32 with unit row stride");
  TORCH_CHECK(kv.is_contiguous(), "kv cache must be contiguous");
  TORCH_CHECK(page_table.is_contiguous() &&
                  page_table.scalar_type() == at::kInt,
              "page_table_64 must be contiguous int32");
  TORCH_CHECK(seqlens.scalar_type() == at::kInt, "seqlens must be int32");
  TORCH_CHECK(ghist.scalar_type() == at::kInt, "ghist must be int32");
  TORCH_CHECK(q.dim() == 3, "q must be [rows, heads, 128]");
  TORCH_CHECK(q.size(0) == out.size(0) && q.size(0) == page_table.size(0),
              "rows must match across q / logits / page_table");

  const int rc = launch_logits(
      q.data_ptr(), kv.data_ptr(), weights.data_ptr(), seqlens.data_ptr(),
      page_table.data_ptr(), out.data_ptr(), (int)q.size(0), (int)q.size(1),
      (int)page_table.size(1), (int)out.stride(0), (int)blocks_per_row,
      (int)warps, (void *)at::cuda::getCurrentCUDAStream().stream(),
      ghist.data_ptr(), (int)hist_bits, (int)topk);
  TORCH_CHECK(rc == 0, "launch_logits failed rc=", rc);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("logits_hist", &logits_hist,
        "paged MQA fp8 logits + fine/coarse histogram (gfx950)");
}
