"""Probe gfx950 v_cvt_pk_f32_fp8 decode behavior vs torch native fp8 cast.

Build a tiny kernel that decodes fp8 bytes and writes to f32 output. Compare
to torch.float8_e4m3fnuz.float() to verify the bias.
"""
import os, sys, torch
from torch.utils.cpp_extension import load_inline

src = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

typedef __attribute__((ext_vector_type(2))) float float2_native;

__global__ void decode_kernel(const uint32_t* __restrict__ in,
                              float* __restrict__ out, int n_u32) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_u32) return;
    uint32_t v = in[t];
    float2_native lo = __builtin_amdgcn_cvt_pk_f32_fp8(v, false);
    float2_native hi = __builtin_amdgcn_cvt_pk_f32_fp8(v, true);
    out[t * 4 + 0] = lo[0];
    out[t * 4 + 1] = lo[1];
    out[t * 4 + 2] = hi[0];
    out[t * 4 + 3] = hi[1];
}

void decode_fp8(torch::Tensor in_u32, torch::Tensor out_f32) {
    int n = in_u32.numel();
    int bs = 256;
    int gs = (n + bs - 1) / bs;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    hipLaunchKernelGGL(decode_kernel, dim3(gs), dim3(bs), 0, stream,
        in_u32.data_ptr<uint32_t>(), out_f32.data_ptr<float>(), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_fp8", &decode_fp8);
}
"""

mod = load_inline(name="fp8_decode_probe", cpp_sources="", cuda_sources=src,
                  extra_cuda_cflags=["-O3", "-std=c++20"], verbose=False)

device = torch.device("cuda")

# Build all 256 fp8 byte values
all_bytes = torch.arange(256, dtype=torch.uint8, device=device)
# Pack into uint32 (4 bytes each)
all_u32 = all_bytes.view(64, 4).to(torch.int32).contiguous()
# Pack 4 bytes into uint32: byte0 + byte1<<8 + byte2<<16 + byte3<<24
packed = (all_u32[:, 0] | (all_u32[:, 1] << 8) | (all_u32[:, 2] << 16) | (all_u32[:, 3] << 24)).view(torch.uint32)

out_hw = torch.zeros((packed.numel() * 4,), dtype=torch.float32, device=device)
mod.decode_fp8(packed.contiguous(), out_hw)
torch.cuda.synchronize()

# Reorder back: byte i in original sequence -> out_hw[i]
# packed[t] -> bytes [4t, 4t+1, 4t+2, 4t+3] -> out_hw[4t+0..3] in order lo[0],lo[1],hi[0],hi[1]
out_hw_r = out_hw.view(64, 4)  # [64 u32, 4 floats from each]
# byte order in output: lo decodes bytes (0,1) of u32, hi decodes bytes (2,3)
# So out[4t+0]=byte 4t, out[4t+1]=byte 4t+1, out[4t+2]=byte 4t+2, out[4t+3]=byte 4t+3
out_seq = out_hw.view(256)

# Reference: torch native fp8_e4m3fnuz cast
ref_e4m3fnuz = all_bytes.view(torch.float8_e4m3fnuz).float()

# Reference: torch native fp8_e4m3fn cast
ref_e4m3fn = all_bytes.view(torch.float8_e4m3fn).float()

# Compare
diff_fnuz = (out_seq - ref_e4m3fnuz).abs()
diff_fn = (out_seq - ref_e4m3fn).abs()

print("HW decode vs torch e4m3fnuz: max_abs_diff =", diff_fnuz.max().item(),
      "n_mismatch (>1e-6) =", (diff_fnuz > 1e-6).sum().item())
print("HW decode vs torch e4m3fn:   max_abs_diff =", diff_fn.max().item(),
      "n_mismatch (>1e-6) =", (diff_fn > 1e-6).sum().item())

# Print first 16 mismatches against fnuz
mm = (diff_fnuz > 1e-6).nonzero(as_tuple=True)[0]
if mm.numel() > 0:
    print("First mismatches vs e4m3fnuz:")
    for i in mm[:16].tolist():
        print(f"  byte={i:3d} hw={out_seq[i].item():12.6e} fnuz={ref_e4m3fnuz[i].item():12.6e} fn={ref_e4m3fn[i].item():12.6e}")
else:
    print("FULL MATCH: HW decoder == torch e4m3fnuz cast")
