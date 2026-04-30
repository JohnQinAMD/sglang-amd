"""HIP Triton port of csrc/elementwise/kvcache.cuh — replaces the
`k_cache[indices] = k; v_cache[indices] = v` torch fallback in
`_set_kv_buffer_impl` (memory_pool.py) which lowers to two separate
`index_elementwise_kernel<128,4>` launches per layer-pass on HIP.

Lever 5 v2: 2D grid (BLOCK_M tokens × BLOCK_D cols per program) + prefill-only
gate. v1 (1 program per token) regressed at decode batch=6 because Triton's
fixed per-launch overhead exceeds torch index_put_'s tiled execution at small N.
v2 amortizes launch overhead across BLOCK_M=16 rows per program AND skips the
kernel entirely when N is small (< MIN_N) so decode keeps the torch fallback.

Mirrors the Triton port pattern established by topk_transform_512_triton and
fused_store_cache_triton (the JIT-CUDA path is unavailable on HIP because
tvm_ffi.cpp.load_inline requires CUDA_HOME / nvcc).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# Below this N, torch index_put_ wins on per-launch overhead; skip Triton.
# Decode batch sizes (≤ max-running-request, typically ≤ 8) fall under this.
# Prefill chunked-prefill at 8192 tokens easily clears it.
MIN_N_FOR_TRITON = 32


@triton.jit
def _store_kv_kernel_v2(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    N,
    row_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = rows < N
    col_mask = cols < row_dim
    mask = row_mask[:, None] & col_mask[None, :]

    idx = tl.load(indices_ptr + rows, mask=row_mask, other=0)
    src_off = rows[:, None] * row_dim + cols[None, :]
    dst_off = idx[:, None].to(tl.int64) * row_dim + cols[None, :]

    k = tl.load(k_ptr + src_off, mask=mask)
    v = tl.load(v_ptr + src_off, mask=mask)
    tl.store(k_cache_ptr + dst_off, k, mask=mask)
    tl.store(v_cache_ptr + dst_off, v, mask=mask)


def store_kv_hip(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Single-launch K+V scatter for HIP. Returns True if Triton fired,
    False if the caller must fall back to torch index_put_.

    Skipped (returns False) when N < MIN_N_FOR_TRITON — at small N torch's
    index_elementwise_kernel beats Triton on launch overhead.

    Shapes: k/v are [N, row_dim]; k_cache/v_cache are [num_pages, row_dim];
    indices is [N].
    """
    assert k.shape == v.shape, f"k/v shape mismatch: {k.shape} vs {v.shape}"
    assert k.dtype == k_cache.dtype, f"k dtype mismatch: {k.dtype} vs {k_cache.dtype}"
    assert v.dtype == v_cache.dtype, f"v dtype mismatch: {v.dtype} vs {v_cache.dtype}"
    assert k.is_contiguous() and v.is_contiguous()
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    N, row_dim = k.shape
    if N == 0:
        return True
    if N < MIN_N_FOR_TRITON:
        return False
    if indices.dtype not in (torch.int32, torch.int64):
        indices = indices.to(torch.int32)

    BLOCK_M = 16
    # BLOCK_D heuristic: small row_dim → single tile; larger → 256-wide tiles.
    if row_dim <= 256:
        BLOCK_D = triton.next_power_of_2(row_dim)
    else:
        BLOCK_D = 256
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(row_dim, BLOCK_D))
    _store_kv_kernel_v2[grid](
        k, v, k_cache, v_cache, indices, N,
        row_dim=row_dim, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
    )
    return True
