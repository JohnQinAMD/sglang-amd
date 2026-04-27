import torch
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.fused_qk_concat import fused_qk_rope_cat
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm_a16w16_atomic import gemm_a16w16_atomic

from sglang.srt.utils import BumpAllocator

__all__ = ["fused_qk_rope_cat", "fused_qk_rope_cat_and_cache_mla"]


# Hand-tuned config for Pro-Base routed-MoE router gemm at decode shapes
# (M small, N=384 routed experts, K=7168 hidden). aiter's default
# (BLOCK_M=256, BLOCK_N=256, NUM_KSPLIT=1) under-tiles the problem on MI355X
# (only 2 active CTAs across 304 CUs). Microbench shows 3.96× speedup with
# (BLOCK_M=16, BLOCK_N=64, NUM_KSPLIT=8). Confirmed cos_sim 1.0000.
_PRO_BASE_ROUTER_OVERRIDE = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "NUM_KSPLIT": 8,
    "cache_modifier": None,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 32,
    "kpack": 1,
}
def aiter_dsv3_router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    gemm_output_zero_allocator: BumpAllocator = None,
):
    M = hidden_states.shape[0]
    N = weight.shape[0]
    K = hidden_states.shape[1]
    y = None

    if M <= 256:
        # TODO (cagri): convert to bfloat16 as part of another kernel to save time
        # for now it is also coupled with zero allocator.
        if gemm_output_zero_allocator != None:
            y = gemm_output_zero_allocator.allocate(M * N).view(M, N)
        else:
            y = torch.zeros((M, N), dtype=torch.float32, device=hidden_states.device)

    if y is not None:
        # Pro-Base routed-MoE router has N=384, K=7168. At decode (M ≤ 64) the
        # default config (BLOCK_M=256, BLOCK_N=256, NUM_KSPLIT=1) leaves 2 of 304
        # CUs busy; the hand-tuned override below gets ~96 tiles → ~3.96× speedup.
        cfg = (
            _PRO_BASE_ROUTER_OVERRIDE
            if (N == 384 and K == 7168 and M <= 64)
            else None
        )
        logits = gemm_a16w16_atomic(
            hidden_states, weight, y=y, config=cfg
        ).to(hidden_states.dtype)
    else:
        logits = gemm_a16w16(hidden_states, weight)

    return logits


def get_dsv3_gemm_output_zero_allocator_size(
    n_routed_experts: int, num_moe_layers: int, allocate_size: int, embedding_dim: int
):
    if embedding_dim != 7168 or n_routed_experts != 256:
        return 0

    per_layer_size = 256 * (allocate_size + n_routed_experts)

    return num_moe_layers * per_layer_size
