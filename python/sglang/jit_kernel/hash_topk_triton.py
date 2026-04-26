"""HIP fallback for csrc/deepseek_v4/hash_topk.cuh.

Port of `HashTopKKernel<act_sqrt_softplus>::run`. For each token:
  1. Read `topk` expert IDs from `tid2eid[input_id[i]]` (token's pre-assigned
     experts — the "hash" is at the dataset level, not in this kernel).
  2. Compute routed_weight = sqrt(softplus(router_logits[i, expert_id])).
  3. Normalize by the per-token sum.
  4. For each shared expert (lane in [topk, topk_fused)), append expert
     `num_routed + (lane - topk)` with weight `1 / routed_scaling_factor`.

Caller in `_jit_hash_topk_module` only registers `act_sqrt_softplus`, so
that's the only scoring function we implement here.

Shapes (matching the wrapper at deepseek_v4.py:hash_topk):
  router_logits: f32 [B, num_routed]
  input_id:      i64 [B]                   (token id into tid2eid)
  tid2eid:       i32 [vocab, topk_routed]  (per-token expert assignments)
  topk_weights:  f32 [B, topk_fused]   (output)
  topk_ids:      i32 [B, topk_fused]   (output)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hash_topk_kernel(
    router_logits_ptr,        # f32 [B, E]
    input_id_ptr,             # i64 [B]
    tid2eid_ptr,              # i32 [V, TOPK]   (we only need indexing by input_id)
    topk_weights_ptr,         # f32 [B, TOPK_FUSED]
    topk_ids_ptr,             # i32 [B, TOPK_FUSED]
    routed_scaling_factor,
    num_routed,
    TOPK: tl.constexpr,
    TOPK_FUSED: tl.constexpr,
    NUM_SHARED: tl.constexpr,
    BLOCK_K: tl.constexpr,        # next_pow2(TOPK_FUSED), <= 32
):
    bid = tl.program_id(0)
    token_id = tl.load(input_id_ptr + bid).to(tl.int64)

    lane = tl.arange(0, BLOCK_K)
    routed_mask = lane < TOPK
    fused_mask = lane < TOPK_FUSED

    # Per-routed-lane: load expert_id from tid2eid[token_id, lane]
    eid_offs = token_id * TOPK + lane
    expert_id = tl.load(tid2eid_ptr + eid_offs, mask=routed_mask, other=0).to(tl.int32)

    # router_logits[bid, expert_id]
    rl_offs = bid * num_routed + expert_id.to(tl.int64)
    logits = tl.load(router_logits_ptr + rl_offs, mask=routed_mask, other=0.0).to(tl.float32)

    # sqrt(softplus(x)) = sqrt(max(x,0) + log1p(exp(-|x|)))
    softplus = tl.maximum(logits, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(logits)))
    routed_weight = tl.sqrt(softplus)
    routed_weight = tl.where(routed_mask, routed_weight, 0.0)

    routed_sum = tl.sum(routed_weight, axis=0)

    # For shared lanes (lane >= TOPK): expert = num_routed + (lane - TOPK), weight = 1/scale
    shared_eid = num_routed + (lane - TOPK)
    is_shared = lane >= TOPK
    out_eid = tl.where(is_shared, shared_eid, expert_id)
    out_weight = tl.where(
        is_shared,
        1.0 / routed_scaling_factor,
        routed_weight / routed_sum,
    )

    out_off = bid * TOPK_FUSED + lane
    tl.store(topk_ids_ptr + out_off, out_eid, mask=fused_mask)
    tl.store(topk_weights_ptr + out_off, out_weight, mask=fused_mask)


_HASH_TOPK_CALLS = 0


def hash_topk_triton(
    router_logits: torch.Tensor,
    input_id: torch.Tensor,
    tid2eid: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    routed_scaling_factor: float,
) -> None:
    """In-place fill of topk_weights and topk_ids."""
    global _HASH_TOPK_CALLS
    _HASH_TOPK_CALLS += 1
    if _HASH_TOPK_CALLS == 1 or _HASH_TOPK_CALLS % 100 == 0:
        print(f"[HASH_TOPK_TRITON] HIP-Triton port call #{_HASH_TOPK_CALLS} "
              f"(router_logits={tuple(router_logits.shape)}/{router_logits.dtype})",
              flush=True)
    # Production passes bf16/fp16 router_logits; CUDA reference upcasts internally.
    # Triton kernel converts to fp32 inside; cast at entry to handle bf16/fp16.
    assert router_logits.dtype in (torch.float32, torch.bfloat16, torch.float16), (
        f"unexpected router_logits dtype {router_logits.dtype}"
    )
    if router_logits.dtype != torch.float32:
        router_logits = router_logits.to(torch.float32)
    if input_id.dtype != torch.int64:
        input_id = input_id.to(torch.int64)
    if tid2eid.dtype != torch.int32:
        tid2eid = tid2eid.to(torch.int32)
    assert topk_weights.dtype == torch.float32, f"got {topk_weights.dtype}"
    assert topk_ids.dtype == torch.int32, f"got {topk_ids.dtype}"

    B, num_routed = router_logits.shape
    TOPK = tid2eid.shape[1]
    TOPK_FUSED = topk_ids.shape[1]
    NUM_SHARED = TOPK_FUSED - TOPK
    assert TOPK <= TOPK_FUSED, f"TOPK={TOPK} > TOPK_FUSED={TOPK_FUSED}"
    assert TOPK_FUSED <= 32, f"TOPK_FUSED={TOPK_FUSED} exceeds warp size 32"

    BLOCK_K = max(8, triton.next_power_of_2(TOPK_FUSED))

    _hash_topk_kernel[(B,)](
        router_logits, input_id, tid2eid,
        topk_weights, topk_ids,
        float(routed_scaling_factor), num_routed,
        TOPK=TOPK, TOPK_FUSED=TOPK_FUSED, NUM_SHARED=NUM_SHARED,
        BLOCK_K=BLOCK_K,
    )
