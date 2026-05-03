"""
Triton sparse-attention decode kernel for DeepSeek-V4 on AMD MI355X (gfx950).

Design: FlashAttention-2 style online-softmax decode over pre-gathered K.
  grid = (B*Sq, ceil(Hq / BLOCK_H))
  Per program: load q[BLOCK_H, D_QK], stream K over Topk in BLOCK_T chunks,
    compute QK -> softmax -> accumulate PV.

D_QK = 576 is not a power of 2; we tile along D in BLOCK_D chunks with masking
so the dot work stays proportional to the true D.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_decode_kernel_v3(
    Q_ptr, KV_ptr, INVMASK_ptr, SINK_ptr, O_ptr, LSE_ptr,
    sQ_b, sQ_h, sQ_d,
    sK_b, sK_t, sK_d,
    sM_b, sM_t,
    sO_b, sO_h, sO_d,
    sL_b, sL_h,
    sm_scale,
    TOPK: tl.int32,
    Hq: tl.constexpr,
    D_QK_ACT: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_SINK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < Hq

    m_i = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)

    NUM_D_TILES: tl.constexpr = (D_QK_ACT + BLOCK_D - 1) // BLOCK_D

    offs_dv = tl.arange(0, D_V)

    for t_start in range(0, TOPK, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < TOPK

        # Load invalid mask FIRST so K/V loads at invalid rows can be masked
        # (KV may contain NaN at unused slots — loading would poison the dot).
        inv_ptrs = INVMASK_ptr + pid_b * sM_b + offs_t * sM_t
        inv = tl.load(inv_ptrs, mask=mask_t, other=1).to(tl.int1)
        invalid = inv | (~mask_t)
        valid_t = ~invalid

        qk = tl.zeros([BLOCK_H, BLOCK_T], dtype=tl.float32)

        for d_tile in tl.static_range(NUM_D_TILES):
            d_start = d_tile * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_QK_ACT

            q_ptrs = Q_ptr + pid_b * sQ_b + offs_h[:, None] * sQ_h + offs_d[None, :] * sQ_d
            q_mask = mask_h[:, None] & mask_d[None, :]
            q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

            k_ptrs = KV_ptr + pid_b * sK_b + offs_t[:, None] * sK_t + offs_d[None, :] * sK_d
            # Mask OUT invalid rows so NaN (if any) never enters the dot.
            k_mask = valid_t[:, None] & mask_d[None, :]
            k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

            qk = tl.dot(q_tile, tl.trans(k_tile), acc=qk, out_dtype=tl.float32)

        qk = qk * sm_scale
        qk = tl.where(invalid[None, :], -float("inf"), qk)

        # Online softmax with all-masked-row guard
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        row_all_masked = m_new == -float("inf")
        m_i_safe = tl.where(row_all_masked, 0.0, m_i)
        m_new_safe = tl.where(row_all_masked, 0.0, m_new)
        alpha = tl.exp(m_i_safe - m_new_safe)
        alpha = tl.where(row_all_masked, 1.0, alpha)
        p = tl.exp(qk - m_new_safe[:, None])
        # qk is -inf for masked cols -> exp(-inf - finite) = 0; fine.

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = KV_ptr + pid_b * sK_b + offs_t[:, None] * sK_t + offs_dv[None, :] * sK_d
        v_mask = valid_t[:, None]
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc, out_dtype=tl.float32)

        m_i = m_new

    # Epilogue
    lonely = l_i == 0.0
    lse = tl.where(lonely, float("inf"), m_i + tl.log(l_i))
    inv_l = tl.where(lonely, 0.0, 1.0 / l_i)
    out = acc * inv_l[:, None]

    if USE_SINK:
        sink = tl.load(SINK_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        scale = 1.0 / (1.0 + tl.exp(sink - lse))
        scale = tl.where(lonely, 1.0, scale)
        out = out * scale[:, None]

    o_ptrs = O_ptr + pid_b * sO_b + offs_h[:, None] * sO_h + offs_dv[None, :] * sO_d
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(LSE_ptr + pid_b * sL_b + offs_h * sL_h, lse, mask=mask_h)


def triton_sparse_attn_decode(
    q: torch.Tensor,
    gathered_kv: torch.Tensor,
    invalid_mask: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    sm_scale: float,
    d_v: int,
    *,
    BLOCK_H: int = 4,
    BLOCK_T: int = 128,
    BLOCK_D: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
    waves_per_eu: Optional[int] = None,
    matrix_instr_nonkdim: Optional[int] = None,
    force_single_kernel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (output[BS, Hq, d_v] bf16, lse[BS, Hq] fp32).

    gfx950 (MI355X) DSv4 decode shape (Hq=64, D_QK=576, D_V=512, Topk in {256,512}):

      Single-kernel v3 default (used when force_single_kernel=True, or when
      explicit waves_per_eu / matrix_instr_nonkdim is passed, or when the
      v3 grid alone already saturates CUs):
        BLOCK_H=4, BLOCK_T=128, BLOCK_D=128, num_warps=4, num_stages=1
      Perf: ~0.060 ms at B=1 Topk=512 on torch 2.10-dev + triton 3.6.

      Split-K auto-dispatch (default path for small B):
        BLOCK_H=4, BLOCK_T=64, BLOCK_D=128, SPLIT_K=8, num_warps=4.
      Measured wins (torch 2.10-dev + triton 3.6 on MI355X):
        B=1  Topk=512: 0.0330 ms vs v3 0.0602 ms  (1.82x)
        B=1  Topk=256: 0.0324 ms vs v3 0.0359 ms  (1.11x)
        B=4  Topk=512: 0.0340 ms vs v3 0.0598 ms  (1.76x)
        B=16 Topk=512: 0.0588 ms vs v3 0.0639 ms  (1.09x)
      Triggered when V3_GRID = BS * ceil(Hq/BLOCK_H) < 2 * N_CU_MI355X (512).
      Above that, v3 is already CU-saturated and split-K adds pointless
      stage-2 overhead.
    """
    assert q.dtype == torch.bfloat16, f"q must be bf16, got {q.dtype}"
    assert gathered_kv.dtype == torch.bfloat16
    assert invalid_mask.dtype == torch.bool
    BS, Hq, D_QK = q.shape
    BS2, Topk, D_QK2 = gathered_kv.shape
    assert BS == BS2 and D_QK == D_QK2
    assert invalid_mask.shape == (BS, Topk)
    assert d_v <= D_QK

    # Auto-dispatch to split-K when the single-kernel grid under-utilizes CUs.
    # MI355X has 256 CUs. v3 launches BS * ceil(Hq / BLOCK_H) programs. At
    # DSv4 decode shape with default BLOCK_H=4, that is BS * 16. Below
    # 2 waves/CU (grid < 512) split-K wins on both B=1..16 that we measured.
    V3_GRID = BS * ((Hq + BLOCK_H - 1) // BLOCK_H)
    N_CU_MI355X = 256
    if (
        not force_single_kernel
        and Topk >= 64
        and V3_GRID < 2 * N_CU_MI355X
        and waves_per_eu is None
        and matrix_instr_nonkdim is None
    ):
        # Per-shape autotuned config (2026-04-29 sweep on chi2774, MI355X).
        # Sweep at /sgl-pr/microbench/triton_sparse_decode_sweep_results.md
        # documents the search space + winners. Universal pattern:
        # `waves_per_eu=2 matrix_instr_nonkdim=16` + BLOCK_T=32 + larger
        # SPLIT_K + BLOCK_D matched to D_QK = avg 1.10× decode, 1.32× big-topk.
        if D_QK == 512:
            block_d_sk = 256  # Flash mxfp4 / Flash-Base FP8: 1 clean D-tile
        elif D_QK == 576:
            block_d_sk = 128  # Pro: 5 tiles with masking; BD=192 was non-improving
        else:
            block_d_sk = 128  # safe default
        if BS <= 2:
            sk_block_h, sk_split_k = 4, 16  # B=1: lots of SPLIT_K to fill CUs
        elif Topk >= 1024:
            sk_block_h, sk_split_k = 8, 16  # big topk: wider H + max SPLIT_K
        else:
            sk_block_h, sk_split_k = 8, 8   # B≥4 small topk: wider H, moderate SK
        return triton_sparse_attn_decode_split_k(
            q, gathered_kv, invalid_mask, attn_sink, sm_scale, d_v,
            BLOCK_H=sk_block_h, BLOCK_T=32, BLOCK_D=block_d_sk,
            SPLIT_K=sk_split_k, num_warps=4, num_stages=1,
            waves_per_eu=2, matrix_instr_nonkdim=16,
        )

    # 2026-05-03 Phase A3: skip `.contiguous()` when the tensor is already
    # contiguous (the production case: ref.py:75-82 already runs `.contiguous()`
    # on q + gathered_kv before calling here). Preserves correctness via the
    # `is_contiguous()` guard while cutting Python overhead and any redundant
    # bf16_copy launches if upstream ever produces a non-contig input.
    if not q.is_contiguous():
        q = q.contiguous()
    if not gathered_kv.is_contiguous():
        gathered_kv = gathered_kv.contiguous()
    # bool tensor and uint8 share itemsize=1; use .view() (bit-reinterpret, free)
    # instead of .to(uint8) (allocates + copies). Skip the trailing .contiguous()
    # too — bool→uint8 view preserves contiguity, and upstream callers always
    # produce contiguous masks (M3 megakernel buf, ref.py compute path).
    invalid_mask_u8 = invalid_mask.view(torch.uint8)
    if not invalid_mask_u8.is_contiguous():
        invalid_mask_u8 = invalid_mask_u8.contiguous()

    out = torch.empty((BS, Hq, d_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((BS, Hq), dtype=torch.float32, device=q.device)

    USE_SINK = attn_sink is not None
    sink_ptr = attn_sink if USE_SINK else q  # dummy

    grid = (BS, triton.cdiv(Hq, BLOCK_H))

    extra = {}
    if waves_per_eu is not None:
        extra["waves_per_eu"] = waves_per_eu
    if matrix_instr_nonkdim is not None:
        extra["matrix_instr_nonkdim"] = matrix_instr_nonkdim

    _sparse_attn_decode_kernel_v3[grid](
        q, gathered_kv, invalid_mask_u8, sink_ptr, out, lse,
        q.stride(0), q.stride(1), q.stride(2),
        gathered_kv.stride(0), gathered_kv.stride(1), gathered_kv.stride(2),
        invalid_mask_u8.stride(0), invalid_mask_u8.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        sm_scale,
        Topk,
        Hq=Hq,
        D_QK_ACT=D_QK,
        D_V=d_v,
        BLOCK_H=BLOCK_H,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        USE_SINK=1 if USE_SINK else 0,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra,
    )
    return out, lse


# ---------------------------------------------------------------------------
# Split-K variant
#
# Stage 1: each program handles a contiguous slice of Topk for a (batch, head-block).
#   Grid = (BS, ceil(Hq / BLOCK_H), SPLIT_K)
#   Writes partial (m, l, acc) to staging tensors.
# Stage 2: reduce across the SPLIT_K axis, produce final O and lse.
# ---------------------------------------------------------------------------

@triton.jit
def _sparse_attn_decode_stage1(
    Q_ptr, KV_ptr, INVMASK_ptr,
    M_part_ptr, L_part_ptr, ACC_part_ptr,  # staging
    sQ_b, sQ_h, sQ_d,
    sK_b, sK_t, sK_d,
    sM_b, sM_t,
    sMP_b, sMP_h, sMP_k,
    sLP_b, sLP_h, sLP_k,
    sAP_b, sAP_h, sAP_k, sAP_d,
    sm_scale,
    TOPK: tl.int32,
    Hq: tl.constexpr,
    D_QK_ACT: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < Hq

    # Slice of Topk this program owns — contiguous range.
    # Divide TOPK evenly-ish by ceil; last program may have fewer blocks.
    # Use a ceil-div so we cover all of TOPK exactly.
    t_per_split = tl.cdiv(TOPK, SPLIT_K)
    t_begin = pid_k * t_per_split
    t_end = tl.minimum(t_begin + t_per_split, TOPK)

    m_i = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)

    NUM_D_TILES: tl.constexpr = (D_QK_ACT + BLOCK_D - 1) // BLOCK_D
    offs_dv = tl.arange(0, D_V)

    for t_start in range(t_begin, t_end, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < t_end

        inv_ptrs = INVMASK_ptr + pid_b * sM_b + offs_t * sM_t
        inv = tl.load(inv_ptrs, mask=mask_t, other=1).to(tl.int1)
        invalid = inv | (~mask_t)
        valid_t = ~invalid

        qk = tl.zeros([BLOCK_H, BLOCK_T], dtype=tl.float32)

        for d_tile in tl.static_range(NUM_D_TILES):
            d_start = d_tile * BLOCK_D
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_QK_ACT

            q_ptrs = Q_ptr + pid_b * sQ_b + offs_h[:, None] * sQ_h + offs_d[None, :] * sQ_d
            q_mask = mask_h[:, None] & mask_d[None, :]
            q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

            k_ptrs = KV_ptr + pid_b * sK_b + offs_t[:, None] * sK_t + offs_d[None, :] * sK_d
            k_mask = valid_t[:, None] & mask_d[None, :]
            k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

            qk = tl.dot(q_tile, tl.trans(k_tile), acc=qk, out_dtype=tl.float32)

        qk = qk * sm_scale
        qk = tl.where(invalid[None, :], -float("inf"), qk)

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        row_all_masked = m_new == -float("inf")
        m_i_safe = tl.where(row_all_masked, 0.0, m_i)
        m_new_safe = tl.where(row_all_masked, 0.0, m_new)
        alpha = tl.exp(m_i_safe - m_new_safe)
        alpha = tl.where(row_all_masked, 1.0, alpha)
        p = tl.exp(qk - m_new_safe[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = KV_ptr + pid_b * sK_b + offs_t[:, None] * sK_t + offs_dv[None, :] * sK_d
        v_mask = valid_t[:, None]
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc, out_dtype=tl.float32)

        m_i = m_new

    # Store partials.
    # M_part[pid_b, offs_h, pid_k], L_part same, ACC_part[pid_b, offs_h, pid_k, :]
    mp_ptrs = M_part_ptr + pid_b * sMP_b + offs_h * sMP_h + pid_k * sMP_k
    lp_ptrs = L_part_ptr + pid_b * sLP_b + offs_h * sLP_h + pid_k * sLP_k
    tl.store(mp_ptrs, m_i, mask=mask_h)
    tl.store(lp_ptrs, l_i, mask=mask_h)

    ap_ptrs = (
        ACC_part_ptr
        + pid_b * sAP_b
        + offs_h[:, None] * sAP_h
        + pid_k * sAP_k
        + offs_dv[None, :] * sAP_d
    )
    tl.store(ap_ptrs, acc, mask=mask_h[:, None])


@triton.jit
def _sparse_attn_decode_stage2(
    M_part_ptr, L_part_ptr, ACC_part_ptr,
    SINK_ptr, O_ptr, LSE_ptr,
    sMP_b, sMP_h, sMP_k,
    sLP_b, sLP_h, sLP_k,
    sAP_b, sAP_h, sAP_k, sAP_d,
    sO_b, sO_h, sO_d,
    sL_b, sL_h,
    Hq: tl.constexpr,
    D_V: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_SINK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h_row = tl.program_id(1)   # single head per program, simpler reduction

    if pid_h_row >= Hq:
        return

    offs_k = tl.arange(0, SPLIT_K)
    offs_dv = tl.arange(0, D_V)

    # Load all SPLIT_K partials for this (pid_b, pid_h_row)
    mp_ptrs = M_part_ptr + pid_b * sMP_b + pid_h_row * sMP_h + offs_k * sMP_k
    lp_ptrs = L_part_ptr + pid_b * sLP_b + pid_h_row * sLP_h + offs_k * sLP_k
    m_parts = tl.load(mp_ptrs)        # [SPLIT_K]
    l_parts = tl.load(lp_ptrs)        # [SPLIT_K]

    # Global max
    m_global = tl.max(m_parts, axis=0)
    # If all -inf, lonely.
    lonely = m_global == -float("inf")
    m_global_safe = tl.where(lonely, 0.0, m_global)

    # Rescale factors and weights
    scale = tl.exp(m_parts - m_global_safe)
    scale = tl.where(m_parts == -float("inf"), 0.0, scale)
    l_global = tl.sum(l_parts * scale, axis=0)  # [scalar]

    # Weighted acc sum
    # ACC_part[pid_b, pid_h_row, :, :]   shape [SPLIT_K, D_V]
    ap_ptrs = (
        ACC_part_ptr
        + pid_b * sAP_b
        + pid_h_row * sAP_h
        + offs_k[:, None] * sAP_k
        + offs_dv[None, :] * sAP_d
    )
    acc_parts = tl.load(ap_ptrs)      # [SPLIT_K, D_V]  fp32
    weighted = acc_parts * scale[:, None]
    acc = tl.sum(weighted, axis=0)    # [D_V]

    inv_l = tl.where(lonely | (l_global == 0.0), 0.0, 1.0 / l_global)
    lse = tl.where(lonely | (l_global == 0.0), float("inf"), m_global_safe + tl.log(l_global))
    out = acc * inv_l

    if USE_SINK:
        sink = tl.load(SINK_ptr + pid_h_row).to(tl.float32)
        scl = 1.0 / (1.0 + tl.exp(sink - lse))
        scl = tl.where(lonely, 1.0, scl)
        out = out * scl

    o_ptrs = O_ptr + pid_b * sO_b + pid_h_row * sO_h + offs_dv * sO_d
    tl.store(o_ptrs, out.to(tl.bfloat16))
    tl.store(LSE_ptr + pid_b * sL_b + pid_h_row * sL_h, lse)


def triton_sparse_attn_decode_split_k(
    q: torch.Tensor,
    gathered_kv: torch.Tensor,
    invalid_mask: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    sm_scale: float,
    d_v: int,
    *,
    BLOCK_H: int = 16,
    BLOCK_T: int = 64,
    BLOCK_D: int = 64,
    SPLIT_K: int = 4,
    num_warps: int = 4,
    num_stages: int = 1,
    waves_per_eu: Optional[int] = None,
    matrix_instr_nonkdim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert q.dtype == torch.bfloat16
    assert gathered_kv.dtype == torch.bfloat16
    assert invalid_mask.dtype == torch.bool
    BS, Hq, D_QK = q.shape
    BS2, Topk, _ = gathered_kv.shape
    assert BS == BS2

    # 2026-05-03 Phase A3: skip `.contiguous()` when the tensor is already
    # contiguous (the production case: ref.py:75-82 already runs `.contiguous()`
    # on q + gathered_kv before calling here). Preserves correctness via the
    # `is_contiguous()` guard while cutting Python overhead and any redundant
    # bf16_copy launches if upstream ever produces a non-contig input.
    if not q.is_contiguous():
        q = q.contiguous()
    if not gathered_kv.is_contiguous():
        gathered_kv = gathered_kv.contiguous()
    # bool tensor and uint8 share itemsize=1; use .view() (bit-reinterpret, free)
    # instead of .to(uint8) (allocates + copies). Skip the trailing .contiguous()
    # too — bool→uint8 view preserves contiguity, and upstream callers always
    # produce contiguous masks (M3 megakernel buf, ref.py compute path).
    invalid_mask_u8 = invalid_mask.view(torch.uint8)
    if not invalid_mask_u8.is_contiguous():
        invalid_mask_u8 = invalid_mask_u8.contiguous()

    # Staging buffers
    m_part = torch.empty((BS, Hq, SPLIT_K), dtype=torch.float32, device=q.device)
    l_part = torch.empty((BS, Hq, SPLIT_K), dtype=torch.float32, device=q.device)
    acc_part = torch.empty((BS, Hq, SPLIT_K, d_v), dtype=torch.float32, device=q.device)

    out = torch.empty((BS, Hq, d_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((BS, Hq), dtype=torch.float32, device=q.device)

    USE_SINK = attn_sink is not None
    sink_ptr = attn_sink if USE_SINK else q  # dummy

    extra = {}
    if waves_per_eu is not None: extra["waves_per_eu"] = waves_per_eu
    if matrix_instr_nonkdim is not None: extra["matrix_instr_nonkdim"] = matrix_instr_nonkdim

    grid1 = (BS, triton.cdiv(Hq, BLOCK_H), SPLIT_K)
    _sparse_attn_decode_stage1[grid1](
        q, gathered_kv, invalid_mask_u8,
        m_part, l_part, acc_part,
        q.stride(0), q.stride(1), q.stride(2),
        gathered_kv.stride(0), gathered_kv.stride(1), gathered_kv.stride(2),
        invalid_mask_u8.stride(0), invalid_mask_u8.stride(1),
        m_part.stride(0), m_part.stride(1), m_part.stride(2),
        l_part.stride(0), l_part.stride(1), l_part.stride(2),
        acc_part.stride(0), acc_part.stride(1), acc_part.stride(2), acc_part.stride(3),
        sm_scale,
        Topk,
        Hq=Hq,
        D_QK_ACT=D_QK,
        D_V=d_v,
        BLOCK_H=BLOCK_H,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        SPLIT_K=SPLIT_K,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra,
    )

    grid2 = (BS, Hq)
    _sparse_attn_decode_stage2[grid2](
        m_part, l_part, acc_part,
        sink_ptr, out, lse,
        m_part.stride(0), m_part.stride(1), m_part.stride(2),
        l_part.stride(0), l_part.stride(1), l_part.stride(2),
        acc_part.stride(0), acc_part.stride(1), acc_part.stride(2), acc_part.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        lse.stride(0), lse.stride(1),
        Hq=Hq,
        D_V=d_v,
        SPLIT_K=SPLIT_K,
        USE_SINK=1 if USE_SINK else 0,
        num_warps=4,
        num_stages=1,
    )
    return out, lse
