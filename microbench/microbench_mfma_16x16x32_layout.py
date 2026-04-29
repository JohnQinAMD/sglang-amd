"""Probe gfx950 mfma_bf16_16x16x32 per-lane output layout via known-input trick.

Issue: the CK V32 sparse MLA decode kernel assumes per-lane output is
`output[M=kgrp*4+c, N=lane_id%16]` (M=head, N=kv_col), but the drill-down
shows a labeled-transpose bug — kernel always picks the OPPOSITE valid X
from oracle on production data.

This probe synthesizes a one-hot input where the only non-zero output
element is at a specific (M, N), runs a single mfma_bf16_16x16x32, and
inspects which lane's which c-register holds the non-zero. That gives the
ground-truth (M, N) → (lane_id, c) mapping.

Build via torch.utils.cpp_extension; emit one __global__ kernel per probe.
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

CPP = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>

using bf16 = __bf16;
typedef float fp32x4 __attribute__((ext_vector_type(4)));

// Single-MFMA probe. A is [M=16, K=32] of bf16 with a 1.0 at (m_one, k_one)
// and zeros elsewhere. B is [K=32, N=16] of bf16 with 1.0 at (k_one, n_one).
// All other inputs zero. Output C[M, N] should have 1.0 at (m_one, n_one)
// only. We dump per-lane c-register to inspect which lane holds the nonzero.
//
// Wave size 64. C[M=16, N=16] = 256 fp32 / 64 lanes = 4 fp32 per lane.

__global__ void probe_mfma(const bf16* __restrict__ A,  // [16, 32]
                           const bf16* __restrict__ B,  // [32, 16]
                           float* __restrict__ out) {   // [64 lanes, 4 c]
    int lane = threadIdx.x;
    typedef bf16 bf16x8 __attribute__((ext_vector_type(8)));

    // Load A row segment for this lane: each lane reads 8 bf16 elements
    // covering some (m, k) chunk. The MFMA hardware handles the shuffle.
    // For mfma_bf16_16x16x32 on gfx950, the operand layout is:
    //   A: lane holds A[m=lane%16, k=(lane/16)*8 .. (lane/16)*8+7]
    //   B: lane holds B[k=(lane/16)*8 .. (lane/16)*8+7, n=lane%16]
    bf16x8 a_vec, b_vec;
    int m_a = lane % 16;
    int k_base = (lane / 16) * 8;
    int n_b = lane % 16;
    for (int i = 0; i < 8; ++i) {
        a_vec[i] = A[m_a * 32 + (k_base + i)];
        b_vec[i] = B[(k_base + i) * 16 + n_b];
    }

    fp32x4 c = {0, 0, 0, 0};
    c = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a_vec, b_vec, c, 0, 0, 0);

    // Dump the per-lane c register.
    for (int i = 0; i < 4; ++i)
        out[lane * 4 + i] = c[i];
}

torch::Tensor probe(torch::Tensor A, torch::Tensor B) {
    auto out = torch::zeros({64, 4}, A.options().dtype(torch::kFloat32));
    probe_mfma<<<dim3(1), dim3(64)>>>(
        reinterpret_cast<const bf16*>(A.data_ptr()),
        reinterpret_cast<const bf16*>(B.data_ptr()),
        out.data_ptr<float>()
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe", &probe);
}
"""

ext = load_inline(
    name="mfma_layout_probe",
    cpp_sources=[],
    cuda_sources=[CPP],
    extra_cuda_cflags=["-O3", "-std=c++17", "--offload-arch=gfx950"],
    verbose=False,
)


def run_one(m_one: int, n_one: int, k_one: int = 0):
    """Set A[m_one, k_one]=1, B[k_one, n_one]=1, all else 0. Output should
    have 1 at C[m_one, n_one] only. Find which lane/c-register holds it."""
    A = torch.zeros(16, 32, dtype=torch.bfloat16, device="cuda")
    B = torch.zeros(32, 16, dtype=torch.bfloat16, device="cuda")
    A[m_one, k_one] = 1.0
    B[k_one, n_one] = 1.0
    out = ext.probe(A, B)  # [64, 4]
    nonzero = (out.abs() > 0.5).nonzero(as_tuple=False).cpu().tolist()
    return nonzero


def main():
    print("=== gfx950 mfma_bf16_16x16x32 per-lane layout probe ===\n")
    print("Probing all (M, N) pairs to determine per-lane (lane_id, c) mapping.")
    print(f"{'M':>3s} {'N':>3s}  →  (lane, c)")

    layout_map = {}  # (M, N) → (lane, c)
    for m in range(16):
        for n in range(16):
            nz = run_one(m, n)
            if len(nz) != 1:
                print(f"{m:>3d} {n:>3d}  → UNEXPECTED nonzero count: {len(nz)}")
                continue
            lane, c = nz[0]
            layout_map[(m, n)] = (lane, c)
            if m < 4 and n < 8:
                print(f"{m:>3d} {n:>3d}  →  (lane={lane:2d}, c={c})")

    # Derive the formula: try lane = f(m, n), c = g(m, n).
    print("\n=== Inferred layout formulas ===")

    # Hypothesis A: lane%16=m (head), (lane/16)*4+c=n (kv_col).
    #   For (m, n): lane = m + ((n // 4) * 16) = m + (n>>2)*16; c = n%4.
    matches_A = 0
    for (m, n), (lane, c) in layout_map.items():
        exp_lane = m + (n // 4) * 16
        exp_c = n % 4
        if lane == exp_lane and c == exp_c:
            matches_A += 1
    print(f"Hypothesis A — lane=m+(n//4)*16, c=n%4: {matches_A}/{len(layout_map)} match")

    # Hypothesis B (kernel's current assumption):
    #   lane%16 = n (kv_col); (lane/16)*4 + c = m (head). Inverse:
    #   lane = n + ((m // 4) * 16); c = m % 4.
    matches_B = 0
    for (m, n), (lane, c) in layout_map.items():
        exp_lane = n + (m // 4) * 16
        exp_c = m % 4
        if lane == exp_lane and c == exp_c:
            matches_B += 1
    print(f"Hypothesis B (kernel's) — lane=n+(m//4)*16, c=m%4: {matches_B}/{len(layout_map)} match")

    # Hypothesis C: variant with row-blocked output. Try:
    #   m = (lane // 16) * 4 + (c // 1) ??? — explore programmatically.
    # Just dump first 8 rows of the layout for inspection.
    print("\n=== First 8 (M, N) → (lane, c) entries ===")
    for k in sorted(layout_map.keys())[:32]:
        print(f"  M={k[0]:>2d} N={k[1]:>2d} → lane={layout_map[k][0]:>2d} c={layout_map[k][1]}")

    if matches_A == len(layout_map):
        print("\n>>> CONFIRMED: layout is Hypothesis A (head=lane%16, kv_col=(lane//16)*4+c)")
        print(">>> Kernel's current labeling (Hypothesis B) is WRONG.")
        print(">>> Fix: swap labels in LDS write + softmax mask + reductions.")
    elif matches_B == len(layout_map):
        print("\n>>> Layout is Hypothesis B (kernel's current assumption is correct).")
        print(">>> The Layer-3 bug is NOT in the MFMA layout — look elsewhere.")
    else:
        print("\n>>> Neither hypothesis matches. Custom layout — see dump above.")


if __name__ == "__main__":
    main()
