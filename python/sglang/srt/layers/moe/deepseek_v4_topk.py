from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_compiler_backend,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
)

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


from sglang.srt.layers.moe.topk import StandardTopKOutput, _mask_topk_ids_padded_region


# ----------------------------------------------------------------------------
# Triton port of biased_topk for ROCm/AMD.
#
# The CUDA C++ JIT path (`moe_fused_gate`) hardcodes CUDA_HOME and fails to
# build on AMD ROCm. This Triton kernel implements the same op:
#   scores = sqrt(softplus(gating_output))
#   topk on (scores + correction_bias) along dim=-1
#   weights = scores.gather(topk_ids).renormalize()
#
# Microbench at Pro M=8192, N=384, TOPK=6: 281us (eager) -> 37us (this), 7.65x.
# At Flash M=8192, N=256, TOPK=6: 162us -> 33us, 4.86x.
# cos_sim_sorted=1.0000, top-k id overlap 6/6 vs eager.
# ----------------------------------------------------------------------------
import triton
import triton.language as tl


@triton.jit
def _biased_topk_sqrtsoftplus_kernel(
    gating_ptr,    # fp32 [M, N]
    bias_ptr,      # fp32 [N]
    out_w_ptr,     # fp32 [M, TOPK]
    out_i_ptr,     # i32 [M, TOPK]
    M,
    N,
    N_PADDED: tl.constexpr,   # next pow2 >= N
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    n_offs = tl.arange(0, N_PADDED)
    n_valid = n_offs < N
    g_offs = m_offs[:, None] * N + n_offs[None, :]
    valid_mask = m_mask[:, None] & n_valid[None, :]
    g = tl.load(gating_ptr + g_offs, mask=valid_mask, other=-float("inf"))
    bias = tl.load(bias_ptr + n_offs, mask=n_valid, other=-float("inf"))

    # Stable sqrtsoftplus: softplus = max(x,0) + log1p(exp(-|x|))
    abs_g = tl.abs(g)
    softplus_g = tl.maximum(g, 0.0) + tl.log(1.0 + tl.exp(-abs_g))
    scores = tl.sqrt(softplus_g)
    scores_for_choice = tl.where(n_valid[None, :], scores + bias[None, :], -float("inf"))

    sf = scores_for_choice
    for k in tl.static_range(0, TOPK):
        max_idx = tl.argmax(sf, axis=1)
        gather_offs = m_offs * N + max_idx
        weight = tl.load(gating_ptr + gather_offs, mask=m_mask, other=0.0)
        abs_w = tl.abs(weight)
        softplus_w = tl.maximum(weight, 0.0) + tl.log(1.0 + tl.exp(-abs_w))
        score_w = tl.sqrt(softplus_w)
        out_offs = m_offs * TOPK + k
        tl.store(out_w_ptr + out_offs, score_w, mask=m_mask)
        tl.store(out_i_ptr + out_offs, max_idx.to(tl.int32), mask=m_mask)
        mask_eq = (n_offs[None, :] == max_idx[:, None])
        sf = tl.where(mask_eq, -float("inf"), sf)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _biased_topk_sqrtsoftplus_triton(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton fused: sqrt(softplus(gating)) + bias + iterative-argmax + renorm.

    Used as the AMD/ROCm fast path for `biased_topk_impl` when scoring_func is
    "sqrtsoftplus", num_fused_shared_experts == 0, and shapes are supported.
    Returns (topk_weights[fp32], topk_ids[int32]).
    """
    M, N = gating_output.shape
    N_padded = _next_pow2(N)
    out_w = torch.empty(M, topk, dtype=torch.float32, device=gating_output.device)
    out_i = torch.empty(M, topk, dtype=torch.int32, device=gating_output.device)
    BLOCK_M = 16 if M >= 256 else (4 if M > 1 else 1)
    grid = (triton.cdiv(M, BLOCK_M),)
    _biased_topk_sqrtsoftplus_kernel[grid](
        gating_output.contiguous(), correction_bias.contiguous(),
        out_w, out_i,
        M, N, N_PADDED=N_padded, TOPK=topk, BLOCK_M=BLOCK_M,
    )
    if renormalize:
        out_w = out_w / out_w.sum(dim=-1, keepdim=True)
    return out_w, out_i


class HashTopK(nn.Module):
    def __init__(
        self,
        topk,
        num_experts,
        num_fused_shared_experts,
        vocab_size,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        apply_routed_scaling_factor_on_output=False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.routed_scaling_factor = routed_scaling_factor
        self.num_fused_shared_experts = num_fused_shared_experts
        self.score_func = scoring_func
        self.tid2eid = nn.Parameter(
            torch.empty(vocab_size, topk - num_fused_shared_experts, dtype=torch.int32),
            requires_grad=False,
        )

        if get_bool_env_var("SGLANG_HACK_TID2EID_INIT_ZERO"):
            print("hack: tid2eid init to zero")
            nn.init.constant_(self.tid2eid, 0)

        assert not apply_routed_scaling_factor_on_output, "not implemented"


    def empty_topk_output(self, device: torch.device):
        topk = self.topk - self.num_fused_shared_experts
        topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_ids = torch.full((0, topk), -1, dtype=torch.int32, device=device)
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)

    def _forward_torch(
        self, router_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.score_func == "softmax":
            scores = router_logits.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = router_logits.sigmoid()
        else:
            scores = torch.nn.functional.softplus(router_logits).sqrt()

        num_token = scores.shape[0]

        topk_ids = torch.zeros(
            (num_token, self.topk), dtype=torch.int32, device=scores.device
        )
        topk_weights = torch.zeros(
            (num_token, self.topk), dtype=scores.dtype, device=scores.device
        )

        if self.num_fused_shared_experts == 1:
            # Hash MoE: get routed expert IDs and weights
            topk_ids[:, :-1] = self.tid2eid[input_ids]
            topk_weights[:, :-1] = scores.gather(1, topk_ids[:, :-1])

            if self.score_func != "softmax":
                topk_weights[:, :-1] /= topk_weights[:, :-1].sum(dim=-1, keepdim=True)

            # reference: biased_grouped_topk_impl in topk.py
            topk_ids[:, -1] = torch.randint(
                low=self.num_experts,
                high=self.num_experts + self.num_fused_shared_experts,
                size=(num_token,),
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            )

            # don't apply routed scaling factor here
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / self.routed_scaling_factor
            )
        else:
            topk_ids[:, :] = self.tid2eid[input_ids]
            topk_weights[:, :] = scores.gather(1, topk_ids[:, :])
            if self.score_func != "softmax":
                topk_weights[:, :] /= topk_weights[:, :].sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids

    @torch.compiler.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ):
        assert (
            input_ids.shape[0] == hidden_states.shape[0] == router_logits.shape[0]
        ), f"{input_ids.shape=} {hidden_states.shape=} {router_logits.shape=}"

        if envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.get():
            from sglang.jit_kernel.deepseek_v4 import hash_topk

            topk_weights, topk_ids = hash_topk(
                router_logits=router_logits,
                input_ids=input_ids,
                tid2eid=self.tid2eid,
                num_fused_shared_experts=self.num_fused_shared_experts,
                routed_scaling_factor=self.routed_scaling_factor,
                scoring_func=self.score_func,
            )
        else:
            topk_weights, topk_ids = self._forward_torch(router_logits, input_ids)

        if is_hip():
            topk_weights = topk_weights.to(torch.float32)

        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
        )
        return topk_output


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def biased_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    # ROCm fast path: fused sqrt(softplus)+bias+topk+renorm Triton kernel.
    # 7.65x at Pro M=8192 / 4.86x at Flash M=8192 vs the eager path below.
    # Falls back to eager for unsupported shapes/configs (multi-shared-experts,
    # apply_routed_scaling_factor_on_output, dispatch_info).
    if (
        _is_hip
        and scoring_func == "sqrtsoftplus"
        and num_fused_shared_experts == 0
        and not apply_routed_scaling_factor_on_output
        and gating_output.dtype == torch.float32
        and correction_bias.dtype == torch.float32
        and gating_output.is_contiguous()
    ):
        topk_weights, topk_ids = _biased_topk_sqrtsoftplus_triton(
            gating_output, correction_bias, topk, renormalize,
        )
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
        _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
        return topk_weights, topk_ids

    if scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(gating_output).sqrt()

    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    _, topk_ids = torch.topk(
        scores_for_choice,
        k=topk,
        dim=-1,
        sorted=(True if num_fused_shared_experts > 0 else False),
    )
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        if routed_scaling_factor is not None:
            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
            )

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def biased_topk_jit_kernel_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    from sglang.jit_kernel.moe_fused_gate import moe_fused_gate

    topk_weights, topk_ids = moe_fused_gate(
        gating_output,
        correction_bias,
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_fused_shared_experts,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids
