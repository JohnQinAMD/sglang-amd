"""MEGA-3': compress_extend_old per-batch-loop fusion megakernel.

STATUS (2026-05-04): Stage 1 kernel implemented (extend setup: state-clear + cat
+ state-writeback). Stage 2 (compute fusion: APE + overlap_transform + compress
+ rmsnorm + rope) deferred to next session as a chained kernel.

Targets the per-request loop in `compress_extend_old` at deepseek_v4.py:1102-1235.

Trace evidence (post-MEGA-1, chi2811): the per-request loop body is the dominant
remaining elementwise source per decode iter (when continuous-batching mixes
extend with decode forwards):
  - 210x copy_(4, 256), 210x copy_(4, 1024)         — temp_buffer slice-assigns
  - 168x fill_(8, 256), 168x fill_(8, 1024)         — KVAndScoreOld.clear()
  - 84x copy_(856, 256) prefill-batch state-cat
  - 86x mul((856, 24), (856, 1))                    — APE/score broadcasts
  - 43x where((856, 1, 64, 512))                    — overlap_transform masking
With bs=4 × 60 layers, that's ~6,000+ captured launches in the extend graph.

Stage 1 design (this session): single Triton launch absorbing per-request
{clear-state-if-prefix-zero, cat-prefix-state-with-current-kv, write-back-state}.
Eliminates ~6 launches per request × bs × 60 layers per extend forward.

Stage 2 (next session): chain to a second Triton launch absorbing the compute
(overlap_transform + softmax + sum + RMSNorm + RoPE). Or fully fuse into one
kernel if shape constraints allow.

INPUT contract for Stage 1:
  kv_and_score_states  : [num_reqs, T_state, D]  rw  (T_state = ratio*coff)
  kv_and_scores        : [total_q_tokens, D]     read
  temp_buffer          : [max_buf, D]            write (caller pre-allocated)
  req_pool_indices     : [bs] int32              read
  prefix_lens          : [bs] int32              read
  extend_lens          : [bs] int32              read
  pt_offsets           : [bs] int32              cumsum of extend_lens (CPU-built)
  buf_offsets          : [bs] int32              cumsum of valid_kv_lens (CPU-built)
                                                  (where valid_kv_len = pre_state + extend)
  pre_state_lens       : [bs] int32              compute_state_len(prefix_lens)
  post_state_lens      : [bs] int32              compute_state_len(prefix_lens + extend_lens)

Per-request scratch is laid out CONSECUTIVELY in temp_buffer:
  request 0: temp_buffer[0:valid_kv_len_0]
  request 1: temp_buffer[buf_offsets[1]:buf_offsets[1]+valid_kv_len_1]
  ...

This allows parallel-over-requests grid scheduling without locks.

CONSTRAINTS (Stage 1):
  - D ≤ 4096 (head_dim*coff*2 for the kv+score backing tensor)
  - T_state ≤ 16 (small for c4; large for c128 — Stage 1 c128 uses simpler path)
  - cuda-graph compatible: all input shapes static at capture time
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _mega3_prime_extend_setup_kernel(
    # ---- Pool / state (rw) ----
    kv_score_states_ptr,    # [num_reqs, T_state, D] — last dim is D = 2 * head_dim_coff
    # ---- Inputs (read) ----
    kv_score_input_ptr,     # [total_q_tokens, D]
    # ---- Output (write) ----
    temp_buffer_ptr,        # [max_buf, D]
    # ---- Per-request descriptors ----
    req_pool_indices_ptr,   # [bs] int32 — index into kv_score_states first dim
    prefix_lens_ptr,        # [bs] int32
    extend_lens_ptr,        # [bs] int32
    pt_offsets_ptr,         # [bs] int32
    buf_offsets_ptr,        # [bs] int32
    pre_state_lens_ptr,     # [bs] int32
    post_state_lens_ptr,    # [bs] int32
    # ---- Strides ----
    state_s_req, state_s_t, state_s_d,
    inp_s_token, inp_s_d,
    buf_s_token, buf_s_d,
    # ---- Sizes ----
    bs,
    T_state,                # max state length = ratio * coff
    HEAD_DIM_KV_HALF,       # = D // 2; first half is kv (zero-fill), second is score (-inf-fill)
    # ---- Compile-time ----
    D: tl.constexpr,                # last dim
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,          # tile size on T_state dim
    BLOCK_TOK: tl.constexpr,        # tile size on token dim for cat
):
    """One program per (request_id, dim_tile). Executes the 4 phases SEQUENTIALLY
    inside the program to avoid inter-program data races on the state buffer.

    Phases (in order):
      1. Clear state[req] (zero kv-half, -inf score-half) IF prefix_len[req] == 0
      2. Cat prefix-state into temp_buffer[buf_off:buf_off+pre_state_len]
      3. Cat current-kv into temp_buffer[buf_off+pre_state_len:buf_off+valid_kv_len]
      4. Write-back temp_buffer[..valid_kv_len-post_state_len:valid_kv_len] -> state[req][:post_state_len]

    Grid: (bs, n_d_tiles).
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_b >= bs:
        return

    # Load per-request descriptors
    req = tl.load(req_pool_indices_ptr + pid_b)
    prefix_len = tl.load(prefix_lens_ptr + pid_b)
    extend_len = tl.load(extend_lens_ptr + pid_b)
    pt_off = tl.load(pt_offsets_ptr + pid_b)
    buf_off = tl.load(buf_offsets_ptr + pid_b)
    pre_state_len = tl.load(pre_state_lens_ptr + pid_b)
    post_state_len = tl.load(post_state_lens_ptr + pid_b)

    valid_kv_len = pre_state_len + extend_len

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # ---- Phase 1: Clear state[req] if prefix_len == 0 ----
    if prefix_len == 0:
        for t_start in range(0, T_state, BLOCK_T):
            t_offs = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offs < T_state
            tile_mask = t_mask[:, None] & d_mask[None, :]

            is_score_half = d_offs >= HEAD_DIM_KV_HALF
            clear_val = tl.where(is_score_half, float("-inf"), 0.0)
            clear_val_2d = clear_val[None, :]   # broadcast over T_state rows

            state_off = (
                req.to(tl.int64) * state_s_req
                + t_offs[:, None].to(tl.int64) * state_s_t
                + d_offs[None, :].to(tl.int64) * state_s_d
            )
            tl.store(kv_score_states_ptr + state_off, clear_val_2d, mask=tile_mask)

    # ---- Phase 2: Cat prefix-state into temp_buffer[buf_off:buf_off+pre_state_len] ----
    for t_start in range(0, pre_state_len, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < pre_state_len
        tile_mask = t_mask[:, None] & d_mask[None, :]

        state_off = (
            req.to(tl.int64) * state_s_req
            + t_offs[:, None].to(tl.int64) * state_s_t
            + d_offs[None, :].to(tl.int64) * state_s_d
        )
        val = tl.load(kv_score_states_ptr + state_off, mask=tile_mask, other=0.0)

        buf_dst_off = (
            (buf_off + t_offs[:, None]).to(tl.int64) * buf_s_token
            + d_offs[None, :].to(tl.int64) * buf_s_d
        )
        tl.store(temp_buffer_ptr + buf_dst_off, val, mask=tile_mask)

    # ---- Phase 3: Cat current-kv into temp_buffer[buf_off+pre_state_len:valid_kv_len] ----
    for tk_start in range(0, extend_len, BLOCK_TOK):
        tk_offs = tk_start + tl.arange(0, BLOCK_TOK)
        tk_mask = tk_offs < extend_len
        tile_mask = tk_mask[:, None] & d_mask[None, :]

        inp_off = (
            (pt_off + tk_offs[:, None]).to(tl.int64) * inp_s_token
            + d_offs[None, :].to(tl.int64) * inp_s_d
        )
        val = tl.load(kv_score_input_ptr + inp_off, mask=tile_mask, other=0.0)

        buf_dst_off = (
            (buf_off + pre_state_len + tk_offs[:, None]).to(tl.int64) * buf_s_token
            + d_offs[None, :].to(tl.int64) * buf_s_d
        )
        tl.store(temp_buffer_ptr + buf_dst_off, val, mask=tile_mask)

    # ---- Phase 4: Write-back state[req][:post_state_len] from temp_buffer tail ----
    for t_start in range(0, post_state_len, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < post_state_len
        tile_mask = t_mask[:, None] & d_mask[None, :]

        buf_src_off = (
            (buf_off + valid_kv_len - post_state_len + t_offs[:, None]).to(tl.int64) * buf_s_token
            + d_offs[None, :].to(tl.int64) * buf_s_d
        )
        val = tl.load(temp_buffer_ptr + buf_src_off, mask=tile_mask, other=0.0)

        state_off = (
            req.to(tl.int64) * state_s_req
            + t_offs[:, None].to(tl.int64) * state_s_t
            + d_offs[None, :].to(tl.int64) * state_s_d
        )
        tl.store(kv_score_states_ptr + state_off, val, mask=tile_mask)


def mega3_prime_extend_setup_old_triton(
    kv_states: torch.Tensor,              # [num_reqs, T_state, D_channel] (rw)  — KV channel only
    score_states: torch.Tensor,           # [num_reqs, T_state, D_channel] (rw)  — SCORE channel only
    kv_input: torch.Tensor,               # [total_q_tokens, D_channel]            — KV channel only
    score_input: torch.Tensor,            # [total_q_tokens, D_channel]            — SCORE channel only
    temp_buffer_kv: torch.Tensor,         # [max_buf, D_channel] (write)
    temp_buffer_score: torch.Tensor,      # [max_buf, D_channel] (write)
    req_pool_indices: torch.Tensor,       # [bs]
    prefix_lens: torch.Tensor,            # [bs]
    extend_lens: torch.Tensor,            # [bs]
) -> Optional[tuple]:
    """KVAndScoreOld variant of Stage 1 for the production OLD compressor path.

    Calls the underlying kernel TWICE (once per channel) with the appropriate
    fill_value (0 for kv, -inf for score). Each call uses HEAD_DIM_KV_HALF=D_channel
    so the entire D_channel is filled (kv with 0, score with -inf) — no half-split.

    Returns the same descriptor tuple as the packed-layout variant.
    """
    bs = req_pool_indices.shape[0]
    if bs == 0:
        return None
    num_reqs, T_state, D_channel = kv_states.shape
    if D_channel > 4096 or T_state > 16:
        return None
    if not all(t.is_contiguous() for t in (kv_states, score_states, kv_input, score_input,
                                            temp_buffer_kv, temp_buffer_score)):
        return None

    # CPU descriptors
    if prefix_lens.is_cuda:
        prefix_lens_cpu = prefix_lens.cpu()
    else:
        prefix_lens_cpu = prefix_lens
    if extend_lens.is_cuda:
        extend_lens_cpu = extend_lens.cpu()
    else:
        extend_lens_cpu = extend_lens

    RATIO = 4
    pre_state_lens_cpu = prefix_lens_cpu % RATIO + RATIO
    post_state_lens_cpu = (prefix_lens_cpu + extend_lens_cpu) % RATIO + RATIO
    valid_kv_lens_cpu = pre_state_lens_cpu + extend_lens_cpu

    pt_offsets_cpu = torch.zeros(bs, dtype=torch.int32)
    buf_offsets_cpu = torch.zeros(bs, dtype=torch.int32)
    pt_running, buf_running = 0, 0
    for i in range(bs):
        pt_offsets_cpu[i] = pt_running
        buf_offsets_cpu[i] = buf_running
        pt_running += int(extend_lens_cpu[i])
        buf_running += int(valid_kv_lens_cpu[i])

    if buf_running > temp_buffer_kv.shape[0] or buf_running > temp_buffer_score.shape[0]:
        return None

    device = kv_states.device
    pt_offsets = pt_offsets_cpu.to(device, non_blocking=True)
    buf_offsets = buf_offsets_cpu.to(device, non_blocking=True)
    pre_state_lens = pre_state_lens_cpu.to(torch.int32).to(device, non_blocking=True)
    post_state_lens = post_state_lens_cpu.to(torch.int32).to(device, non_blocking=True)
    if prefix_lens.is_cuda:
        prefix_lens_dev = prefix_lens.to(torch.int32)
    else:
        prefix_lens_dev = prefix_lens.to(torch.int32).to(device, non_blocking=True)
    if extend_lens.is_cuda:
        extend_lens_dev = extend_lens.to(torch.int32)
    else:
        extend_lens_dev = extend_lens.to(torch.int32).to(device, non_blocking=True)
    if req_pool_indices.dtype != torch.int32:
        req_pool_indices = req_pool_indices.to(torch.int32)

    BLOCK_D = min(triton.next_power_of_2(D_channel), 256)
    n_d_tiles = triton.cdiv(D_channel, BLOCK_D)
    BLOCK_T = 8
    BLOCK_TOK = 16

    grid = (bs, n_d_tiles)

    # KV channel: HEAD_DIM_KV_HALF = D_channel means is_score_half is always False
    # (d < D_channel always), so clear_val = 0.0 throughout. Correct for KV.
    _mega3_prime_extend_setup_kernel[grid](
        kv_states, kv_input, temp_buffer_kv,
        req_pool_indices, prefix_lens_dev, extend_lens_dev,
        pt_offsets, buf_offsets, pre_state_lens, post_state_lens,
        kv_states.stride(0), kv_states.stride(1), kv_states.stride(2),
        kv_input.stride(0), kv_input.stride(1),
        temp_buffer_kv.stride(0), temp_buffer_kv.stride(1),
        bs, T_state,
        D_channel,   # HEAD_DIM_KV_HALF = D_channel → all cols treated as kv (fill=0)
        D=D_channel, BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T, BLOCK_TOK=BLOCK_TOK,
        num_warps=2, num_stages=1,
    )

    # SCORE channel: HEAD_DIM_KV_HALF = 0 means is_score_half is True for all d
    # (d >= 0 always), so clear_val = -inf throughout. Correct for SCORE.
    _mega3_prime_extend_setup_kernel[grid](
        score_states, score_input, temp_buffer_score,
        req_pool_indices, prefix_lens_dev, extend_lens_dev,
        pt_offsets, buf_offsets, pre_state_lens, post_state_lens,
        score_states.stride(0), score_states.stride(1), score_states.stride(2),
        score_input.stride(0), score_input.stride(1),
        temp_buffer_score.stride(0), temp_buffer_score.stride(1),
        bs, T_state,
        0,           # HEAD_DIM_KV_HALF = 0 → all cols treated as score (fill=-inf)
        D=D_channel, BLOCK_D=BLOCK_D, BLOCK_T=BLOCK_T, BLOCK_TOK=BLOCK_TOK,
        num_warps=2, num_stages=1,
    )

    return (
        pt_offsets, buf_offsets,
        pre_state_lens, post_state_lens, valid_kv_lens_cpu.to(torch.int32),
    )




@triton.jit
def _mega3_prime_overlap_xform_drop_kernel(
    in_ptr,                  # [max_buf, 2*D] (one channel — kv OR score) — row=token, col=D-cat
    out_ptr,                 # [total_out_blocks, 2*R, D]
    ape_ptr,                 # [R, 2*D] fp32 — only used when ADD_APE
    buf_offsets_ptr,         # [bs] int32 — start row in temp_buffer per request
    n_blocks_ptr,            # [bs] int32 — n_in_blocks (= compress_len // R) per request
    out_block_offsets_ptr,   # [bs] int32 — cumsum of (n_in_blocks - 1) per request
    in_s_row, in_s_col,
    out_s_block, out_s_t, out_s_d,
    ape_s_t, ape_s_d,
    bs,
    R: tl.constexpr,         # ratio (4 typical)
    D: tl.constexpr,         # head_dim
    BLOCK_D: tl.constexpr,
    ADD_APE: tl.constexpr,   # True for score channel, False for kv
):
    """Per-channel overlap_transform + (optional APE add) + drop-first-block.

    Reads input shape [n_in_blocks, R, 2*D] (viewed as [buf_off + n_in*R, 2*D]) and
    writes output shape [n_in_blocks - 1, 2*R, D] for one channel (kv OR score).

    Per-program (req_id, output_block_id, dim_tile, half_id):
      half_id == 0: lower-half output slots [0, R)
                    out[o, t<R, :]  = in[block o, slot t, cols [0:D]]   + (ape[t, 0:D] if score)
      half_id == 1: upper-half output slots [R, 2R)
                    out[o, t>=R, :] = in[block o+1, slot t-R, cols [D:2D]] + (ape[t-R, D:2D] if score)
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_d_half = tl.program_id(2)   # encodes (dim_tile, half) as pid_d_half = dim_tile*2 + half

    pid_d = pid_d_half // 2
    pid_h = pid_d_half % 2

    if pid_b >= bs:
        return

    n_in = tl.load(n_blocks_ptr + pid_b)
    if pid_o >= n_in - 1:
        return

    buf_off = tl.load(buf_offsets_ptr + pid_b)
    out_block_base = tl.load(out_block_offsets_ptr + pid_b)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    t_offs = tl.arange(0, R)   # local slot indices within one half (0..R-1)

    if pid_h == 0:
        # Lower half: out[o, t<R, :] = in[block o, slot t, cols [0:D]]
        in_row = buf_off + pid_o * R + t_offs                              # (R,)
        in_col = d_offs                                                     # (BLOCK_D,) — lower cols
        in_off = (
            in_row[:, None].to(tl.int64) * in_s_row
            + in_col[None, :].to(tl.int64) * in_s_col
        )
        val = tl.load(in_ptr + in_off, mask=d_mask[None, :], other=0.0)
        if ADD_APE:
            ape_off = (
                t_offs[:, None].to(tl.int64) * ape_s_t
                + d_offs[None, :].to(tl.int64) * ape_s_d   # ape cols [0:D]
            )
            ape = tl.load(ape_ptr + ape_off, mask=d_mask[None, :], other=0.0)
            val = val + ape.to(val.dtype)
        out_t = t_offs                                                       # output slots [0, R)
        out_off = (
            (out_block_base + pid_o).to(tl.int64) * out_s_block
            + out_t[:, None].to(tl.int64) * out_s_t
            + d_offs[None, :].to(tl.int64) * out_s_d
        )
        tl.store(out_ptr + out_off, val, mask=d_mask[None, :])
    else:
        # Upper half: out[o, t>=R, :] = in[block o+1, slot t-R, cols [D:2D]]
        in_row = buf_off + (pid_o + 1) * R + t_offs                         # (R,) — block o+1
        in_col = d_offs + D                                                  # (BLOCK_D,) — upper cols
        in_off = (
            in_row[:, None].to(tl.int64) * in_s_row
            + in_col[None, :].to(tl.int64) * in_s_col
        )
        val = tl.load(in_ptr + in_off, mask=d_mask[None, :], other=0.0)
        if ADD_APE:
            ape_off = (
                t_offs[:, None].to(tl.int64) * ape_s_t
                + (d_offs + D)[None, :].to(tl.int64) * ape_s_d   # ape cols [D:2D]
            )
            ape = tl.load(ape_ptr + ape_off, mask=d_mask[None, :], other=0.0)
            val = val + ape.to(val.dtype)
        out_t = t_offs + R                                                   # output slots [R, 2R)
        out_off = (
            (out_block_base + pid_o).to(tl.int64) * out_s_block
            + out_t[:, None].to(tl.int64) * out_s_t
            + d_offs[None, :].to(tl.int64) * out_s_d
        )
        tl.store(out_ptr + out_off, val, mask=d_mask[None, :])


def mega3_prime_overlap_ape_drop_triton(
    temp_buffer_kv: torch.Tensor,        # [max_buf, 2*D] (Stage 1 output, kv channel)
    temp_buffer_score: torch.Tensor,     # [max_buf, 2*D] (Stage 1 output, score channel)
    out_kv: torch.Tensor,                # [total_out_blocks, 2*R, D]  (preallocated)
    out_score: torch.Tensor,             # [total_out_blocks, 2*R, D]  (preallocated)
    ape: torch.Tensor,                   # [R, 2*D] fp32
    buf_offsets: torch.Tensor,           # [bs] int32 (from Stage 1)
    n_blocks: torch.Tensor,              # [bs] int32 — n_in_blocks per request (= compress_len_i // R)
    out_block_offsets: torch.Tensor,     # [bs] int32 — cumsum of (n_in_blocks - 1) per request
    ratio: int,
    head_dim: int,
) -> bool:
    """MEGA-3' Stage 2: per-channel overlap_transform + APE add (score only) +
    drop-first-block fusion. Two Triton launches (one per channel) replace 14
    aten launches (3-launch overlap_transform × 2 channels + 1 APE add per request × bs requests).

    Output `out_kv`/`out_score` shape: (total_out_blocks, 2*ratio, head_dim).
    Caller chains a SINGLE compress_decode_full_triton on the flat output
    (instead of per-request calls), saving (bs - 1) compress launches.

    Returns True if launched, False if shape gate failed.
    """
    bs = n_blocks.shape[0]
    if bs == 0:
        return False
    if ratio != 4:   # tested only for ratio=4 (production)
        return False
    if head_dim & 1 != 0 or head_dim > 1024:
        return False
    if temp_buffer_kv.shape[1] != 2 * head_dim or temp_buffer_score.shape[1] != 2 * head_dim:
        return False
    if ape.shape != (ratio, 2 * head_dim):
        return False
    if out_kv.shape[1] != 2 * ratio or out_kv.shape[2] != head_dim:
        return False

    BLOCK_D = min(triton.next_power_of_2(head_dim), 256)
    n_d_tiles = triton.cdiv(head_dim, BLOCK_D)

    # Max output blocks per request: bound by the largest n_blocks - 1.
    max_n_in = int(n_blocks.max().item()) if n_blocks.is_cuda else int(n_blocks.max())
    max_out_per_req = max(0, max_n_in - 1)
    if max_out_per_req == 0:
        return True   # nothing to compress; output left untouched (caller masks)

    grid = (bs, max_out_per_req, n_d_tiles * 2)   # last dim: dim_tile*2 + half_id

    # KV channel
    _mega3_prime_overlap_xform_drop_kernel[grid](
        temp_buffer_kv, out_kv, ape,
        buf_offsets, n_blocks, out_block_offsets,
        temp_buffer_kv.stride(0), temp_buffer_kv.stride(1),
        out_kv.stride(0), out_kv.stride(1), out_kv.stride(2),
        ape.stride(0), ape.stride(1),
        bs,
        R=ratio, D=head_dim, BLOCK_D=BLOCK_D,
        ADD_APE=False,
        num_warps=2, num_stages=1,
    )

    # SCORE channel
    _mega3_prime_overlap_xform_drop_kernel[grid](
        temp_buffer_score, out_score, ape,
        buf_offsets, n_blocks, out_block_offsets,
        temp_buffer_score.stride(0), temp_buffer_score.stride(1),
        out_score.stride(0), out_score.stride(1), out_score.stride(2),
        ape.stride(0), ape.stride(1),
        bs,
        R=ratio, D=head_dim, BLOCK_D=BLOCK_D,
        ADD_APE=True,
        num_warps=2, num_stages=1,
    )

    return True

