"""HIP fallback for csrc/deepseek_v4/c4.cuh and c128.cuh.

Both kernels implement "softmax-weighted attention over N history slots" with
paged buffer management:

  c4:   N=8 slots,   page_size=4 (Page4Align) or 8 (RingBuffer),
        kv_score_buffer last-dim layout = [kv_overlap | kv | score_overlap | score]
        each segment of size `head_dim`, total `head_dim * 4` per page slot.
        4 of the 8 slots use the "overlap" segments, 4 use the "current" segments.

  c128: N=128 slots, page_size=128 (RingBuffer only),
        kv_score_buffer last-dim layout = [kv | score]
        each segment of size `head_dim`, total `head_dim * 2` per page slot.

Per-output math (per head_dim lane):
  scaled[j] = score[j] + bias[j]                   (j over N slots)
  weights = softmax(scaled, dim=slot)
  out = sum_j(weights[j] * kv[j])

The torch port focuses on correctness; performance is dominated by the
indexing — a batch of 1 decode call costs ~1ms in torch vs ~10us in the CUDA
kernel. DSv4-Flash on HIP today bypasses the compressor in production, so
this fallback exists primarily to make the `is_hip()` dispatch path complete
and well-defined rather than to be the production path.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# c4 — 8 slots with overlap
# ---------------------------------------------------------------------------


def c4_forward_decode_torch(
    kv_score_buffer: torch.Tensor,        # [num_indices, page_size, head_dim*4]
    kv_score_input: torch.Tensor,         # [B, head_dim*4]
    kv_compressed_output: torch.Tensor,   # [B, head_dim]
    ape: torch.Tensor,                    # [8, head_dim]
    indices: torch.Tensor,                # [B] int32
    seq_lens: torch.Tensor,               # [B] int32
    extra: torch.Tensor | None,           # [B, 1] int32 or None — Page4Align prev page
) -> None:
    """c4 decode kernel — torch port.

    Steps per batch row b:
      1. Write `kv_score_input[b]` into `kv_score_buffer[indices[b], (seq_len-1) % page_size]`.
      2. If `seq_len % 4 == 0`: gather 8 history slots and run softmax-weighted
         attention; write to `kv_compressed_output[b]`.
    """
    page_size = 4 if extra is not None else 8
    assert kv_score_buffer.shape[1] == page_size, \
        f"kv_score_buffer page_size mismatch: {kv_score_buffer.shape[1]} vs {page_size}"
    B = kv_score_input.shape[0]
    head_dim = ape.shape[1]
    assert ape.shape == (8, head_dim)
    assert kv_score_input.shape[-1] == head_dim * 4

    indices_l = indices.to(torch.long)
    seq_lens_l = seq_lens.to(torch.long)
    device = kv_score_buffer.device

    # Step 1: write current input row to ring/page buffer
    write_pos = (seq_lens_l - 1) % page_size                     # [B]
    kv_score_buffer[indices_l, write_pos, :] = kv_score_input

    # Step 2: only batches with seq_len % 4 == 0 emit a compressed output
    do_compress = (seq_lens_l % 4 == 0)
    active = do_compress.nonzero(as_tuple=False).flatten()
    if active.numel() == 0:
        return
    A = active.numel()

    # Reshape buffer last dim into 4 segments of head_dim:
    #   seg 0 = kv_overlap, seg 1 = kv, seg 2 = score_overlap, seg 3 = score
    buf4 = kv_score_buffer.view(*kv_score_buffer.shape[:-1], 4, head_dim)
    act_indices = indices_l[active]                              # [A]
    act_seq = seq_lens_l[active]                                 # [A]

    # 8 slots: slots 0..3 use overlap segments (0, 2); slots 4..7 use current (1, 3).
    # The slot-to-page-position mapping differs:
    #   RingBuffer: k = (seq_len + slot) % 8 — both overlap and non-overlap from
    #               the SAME page (indices[b]) just at different page positions.
    #   Page4Align: overlap from extra[b,0] page, non-overlap from indices[b];
    #               position within page is slot % 4.

    if extra is None:
        # RingBuffer: gather full page once per active row, then permute.
        buf_act = buf4[act_indices]                              # [A, 8, 4, head_dim]
        # Page positions for slots 0..7
        slot_pos = (act_seq.unsqueeze(1) +
                    torch.arange(8, device=device)) % 8          # [A, 8]
        slot_buf = torch.gather(
            buf_act, 1,
            slot_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, head_dim),
        )                                                        # [A, 8, 4, head_dim]
        # Pick segments: overlap (0,2) for first 4 slots, current (1,3) for last 4
        kv_stack = torch.empty(A, 8, head_dim, dtype=torch.float32, device=device)
        sc_stack = torch.empty(A, 8, head_dim, dtype=torch.float32, device=device)
        kv_stack[:, :4] = slot_buf[:, :4, 0].float()             # kv_overlap
        kv_stack[:, 4:] = slot_buf[:, 4:, 1].float()             # kv
        sc_stack[:, :4] = slot_buf[:, :4, 2].float()             # score_overlap
        sc_stack[:, 4:] = slot_buf[:, 4:, 3].float()             # score
    else:
        # Page4Align: two pages per active row (prev + current), 4 slots each.
        prev_idx = extra[active, 0].to(torch.long)               # [A]
        buf_prev = buf4[prev_idx]                                # [A, 4, 4, head_dim]
        buf_curr = buf4[act_indices]                             # [A, 4, 4, head_dim]
        kv_stack = torch.empty(A, 8, head_dim, dtype=torch.float32, device=device)
        sc_stack = torch.empty(A, 8, head_dim, dtype=torch.float32, device=device)
        kv_stack[:, :4] = buf_prev[:, :, 0].float()              # kv_overlap
        kv_stack[:, 4:] = buf_curr[:, :, 1].float()              # kv
        sc_stack[:, :4] = buf_prev[:, :, 2].float()              # score_overlap
        sc_stack[:, 4:] = buf_curr[:, :, 3].float()              # score

    sc_stack = sc_stack + ape.float().unsqueeze(0)               # [A, 8, head_dim]

    # When seq_len == 4 the kernel zeros kv[0..3] and sets score[0..3] = -inf
    # so they contribute nothing to the softmax. Mirror that without forcing
    # a sync (use mask broadcast).
    seq4 = (act_seq == 4).view(A, 1, 1)                          # [A, 1, 1]
    slot_mask = (torch.arange(8, device=device) < 4).view(1, 8, 1)
    suppress = seq4 & slot_mask
    sc_stack = torch.where(suppress, sc_stack.new_full((), -1e9), sc_stack)
    kv_stack = torch.where(suppress, kv_stack.new_zeros(()), kv_stack)

    weights = torch.softmax(sc_stack, dim=1)                     # [A, 8, head_dim]
    weighted = (weights * kv_stack).sum(dim=1)                   # [A, head_dim]
    kv_compressed_output[active] = weighted.to(kv_compressed_output.dtype)


# ---------------------------------------------------------------------------
# c128 — 128 slots, no overlap
# ---------------------------------------------------------------------------


def c128_forward_decode_torch(
    kv_score_buffer: torch.Tensor,        # [num_indices, 128, head_dim*2]
    kv_score_input: torch.Tensor,         # [B, head_dim*2]
    kv_compressed_output: torch.Tensor,   # [B, head_dim]
    ape: torch.Tensor,                    # [128, head_dim]
    indices: torch.Tensor,                # [B] int32
    seq_lens: torch.Tensor,               # [B] int32
) -> None:
    """c128 decode — torch port. Buffer last-dim is [kv | score]."""
    assert kv_score_buffer.shape[1] == 128
    B = kv_score_input.shape[0]
    head_dim = ape.shape[1]
    assert ape.shape == (128, head_dim)
    assert kv_score_input.shape[-1] == head_dim * 2

    indices_l = indices.to(torch.long)
    seq_lens_l = seq_lens.to(torch.long)

    # Step 1: write current input row to ring buffer position (seq_len-1) % 128
    write_pos = (seq_lens_l - 1) % 128
    kv_score_buffer[indices_l, write_pos, :] = kv_score_input

    # Step 2: only seq_len % 128 == 0 fires a compress
    do_compress = (seq_lens_l % 128 == 0)
    active = do_compress.nonzero(as_tuple=False).flatten()
    if active.numel() == 0:
        return

    # Gather full 128-slot history for each active batch row
    # buf[indices[b]]: [128, head_dim*2] → split into [kv, score]
    buf2 = kv_score_buffer[indices_l[active]]                     # [A, 128, head_dim*2]
    kv_all = buf2[..., :head_dim].to(torch.float32)               # [A, 128, head_dim]
    score_all = buf2[..., head_dim:].to(torch.float32)            # [A, 128, head_dim]

    scaled = score_all + ape.to(torch.float32).unsqueeze(0)       # [A, 128, head_dim]
    weights = torch.softmax(scaled, dim=1)
    out = (weights * kv_all).sum(dim=1)                           # [A, head_dim]
    kv_compressed_output[active] = out.to(kv_compressed_output.dtype)


# ---------------------------------------------------------------------------
# Dispatch wrapper that mirrors `module.decode` / `module.prefill`
# ---------------------------------------------------------------------------


def compress_decode_hip(
    compress_ratio: int,
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    kv_compressed_output: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extra: torch.Tensor | None,
) -> None:
    if compress_ratio == 4:
        c4_forward_decode_torch(
            kv_score_buffer, kv_score_input, kv_compressed_output, ape,
            indices, seq_lens, extra,
        )
    elif compress_ratio == 128:
        if extra is not None:
            raise NotImplementedError(
                "c128 with Page4Align (extra != None) — c128 only supports "
                "RingBuffer mode in the production CUDA kernel."
            )
        c128_forward_decode_torch(
            kv_score_buffer, kv_score_input, kv_compressed_output, ape,
            indices, seq_lens,
        )
    else:
        raise ValueError(f"compress_ratio must be 4 or 128, got {compress_ratio}")


# ---------------------------------------------------------------------------
# Prefill — driven by compress_plan / write_plan (PrefillPlan rows)
# ---------------------------------------------------------------------------


def _decode_plan(plan_bytes: torch.Tensor) -> torch.Tensor:
    """[N, 16] uint8 → [N, 4] int64 (ragged_id, batch_id, position, window_len).

    Plans are written as 4× uint32; we read as int64 so callers can do tensor
    arithmetic without overflow. ragged_id == 0xFFFFFFFF marks an invalid row.
    """
    if plan_bytes.dtype != torch.uint8:
        plan_bytes = plan_bytes.view(torch.uint8)
    return plan_bytes.view(-1, 16).view(torch.int32).to(torch.int64)


def _c4_write_one(
    buf4: torch.Tensor,                   # [num_indices, page_size, 4, head_dim]
    kv_score_input: torch.Tensor,         # [num_q_tokens, 4, head_dim]
    plans_i64: torch.Tensor,              # [N, 4]
    indices_l: torch.Tensor,              # [batch_size]
    page_size: int,
    extra: torch.Tensor | None,           # [batch_size, 4] int32 (Page4Align prefill) or None
) -> None:
    """Apply the kWrite half of c4 prefill: copy kv_score_input[ragged_id] to
    the right page slot. Handles invalid plans (ragged_id == 0xFFFFFFFF).
    """
    head_dim = buf4.shape[-1]
    rid = plans_i64[:, 0]
    valid = rid != 0xFFFFFFFF
    if not valid.any():
        return
    rid = rid[valid]
    bid = plans_i64[valid, 1]
    pos = plans_i64[valid, 2]

    if extra is None:
        # RingBuffer: page = indices[bid], pos % 8 within page
        page = indices_l[bid]
        write_pos = pos % page_size
    else:
        # Page4Align: pick write_first_page or write_second_page based on last_pos
        last_pos = extra[bid, 3].to(torch.int64)
        write_first = extra[bid, 2].to(torch.int64)
        write_second = indices_l[bid]
        page = torch.where(pos < last_pos, write_first, write_second)
        write_pos = pos % page_size

    buf4[page, write_pos, :, :] = kv_score_input[rid]


def _c4_compress_one(
    buf4: torch.Tensor,                   # [num_indices, page_size, 4, head_dim]
    kv_score_input: torch.Tensor,         # [num_q_tokens, 4, head_dim]
    kv_compressed_output: torch.Tensor,   # [num_q_tokens, head_dim]
    ape: torch.Tensor,                    # [8, head_dim]
    plans_i64: torch.Tensor,              # [N, 4]
    indices_l: torch.Tensor,              # [batch_size]
    page_size: int,
    extra: torch.Tensor | None,
) -> None:
    """Apply the !kWrite half of c4 prefill: 8-slot softmax-weighted attention,
    write to kv_compressed_output[ragged_id]."""
    device = buf4.device
    head_dim = buf4.shape[-1]
    rid = plans_i64[:, 0]
    valid = rid != 0xFFFFFFFF
    if not valid.any():
        return
    rid = rid[valid]
    bid = plans_i64[valid, 1]
    pos = plans_i64[valid, 2]
    wl = plans_i64[valid, 3]
    A = rid.numel()

    # Build per-(active, slot) source: shape [A, 8, head_dim] for kv and score.
    # Slot j in [0, wl): from buffer; slot j in [wl, 8): from kv_score_input
    # at offset (rid + (j - 7)).
    slot_arange = torch.arange(8, device=device, dtype=torch.int64)            # [8]
    in_window = slot_arange.unsqueeze(0) < wl.unsqueeze(1)                     # [A, 8]
    is_overlap = slot_arange < 4                                                # [8]

    # Source from buffer (in-window slots)
    if extra is None:
        # RingBuffer: page = indices[bid], page_pos = (seq_len + slot) % 8
        seq_len = pos + 1                                                       # [A]
        page_buf = indices_l[bid].unsqueeze(1).expand(A, 8)                    # [A, 8]
        page_pos = (seq_len.unsqueeze(1) + slot_arange.unsqueeze(0)) % page_size
    else:
        # Page4Align: overlap from load_first_page (or load_second_page if wl<=4),
        # current from load_second_page. page_pos = slot % 4.
        load_first = extra[bid, 0]
        load_second = extra[bid, 1]
        # When wl <= 4, all overlap slots come from load_second
        wl_le_4 = (wl <= 4).unsqueeze(1)                                       # [A, 1]
        overlap_page = torch.where(wl_le_4, load_second, load_first).long()    # [A]
        normal_page = load_second.long()                                       # [A]
        page_buf = torch.where(
            is_overlap.unsqueeze(0),                                            # [1, 8]
            overlap_page.unsqueeze(1),                                          # [A, 1]
            normal_page.unsqueeze(1),
        )                                                                       # [A, 8]
        page_pos = (slot_arange % page_size).unsqueeze(0).expand(A, 8)
    page_buf = page_buf.long()
    page_pos = page_pos.long()

    # Source from kv_score_input (out-of-window slots): rid + (slot - 7)
    src_rid = (rid.unsqueeze(1) + slot_arange.unsqueeze(0) - 7).clamp(min=0)
    src_rid = src_rid.long()                                                    # [A, 8]

    # Build flat kv_stack/sc_stack [A, 8, head_dim]
    # For each slot, choose between buffer-row and input-row based on in_window.
    # Use advanced indexing per source then where-select.
    A_idx = torch.arange(A, device=device).unsqueeze(1).expand(A, 8)
    slot_idx = slot_arange.unsqueeze(0).expand(A, 8)

    # buffer reads: pick segment based on slot (overlap=0/2, current=1/3)
    kv_seg = torch.where(is_overlap, torch.zeros_like(slot_arange),
                         torch.ones_like(slot_arange))                          # [8]
    sc_seg = torch.where(is_overlap,
                         torch.full_like(slot_arange, 2),
                         torch.full_like(slot_arange, 3))                       # [8]
    # buf_kv: buf4[page_buf, page_pos, kv_seg, :]   shape [A, 8, head_dim]
    # advanced index: buf4 is [N, P, 4, D]; need (page, pos, seg) per-slot.
    seg_kv_b = kv_seg.unsqueeze(0).expand(A, 8)                                 # [A, 8]
    seg_sc_b = sc_seg.unsqueeze(0).expand(A, 8)
    buf_kv = buf4[page_buf, page_pos, seg_kv_b, :]                              # [A, 8, D]
    buf_sc = buf4[page_buf, page_pos, seg_sc_b, :]                              # [A, 8, D]

    # input reads: kv_score_input[src_rid, kv_seg, :] / [src_rid, sc_seg, :]
    inp_kv = kv_score_input[src_rid, seg_kv_b, :]                               # [A, 8, D]
    inp_sc = kv_score_input[src_rid, seg_sc_b, :]                               # [A, 8, D]

    in_window_e = in_window.unsqueeze(-1)                                       # [A, 8, 1]
    kv_stack = torch.where(in_window_e, buf_kv, inp_kv).float()
    sc_stack = torch.where(in_window_e, buf_sc, inp_sc).float()
    sc_stack = sc_stack + ape.float().unsqueeze(0)                              # [A, 8, D]

    # seq_len == 4 special case (see decode): zero kv[0..3], -inf score[0..3]
    seq_len_a = pos + 1
    seq4 = (seq_len_a == 4).view(A, 1, 1)
    slot_lt4 = (slot_arange < 4).view(1, 8, 1)
    suppress = seq4 & slot_lt4
    sc_stack = torch.where(suppress, sc_stack.new_full((), -1e9), sc_stack)
    kv_stack = torch.where(suppress, kv_stack.new_zeros(()), kv_stack)

    weights = torch.softmax(sc_stack, dim=1)
    out = (weights * kv_stack).sum(dim=1)                                       # [A, D]
    kv_compressed_output[rid] = out.to(kv_compressed_output.dtype)


def c4_forward_prefill_torch(
    kv_score_buffer: torch.Tensor,        # [num_indices, page_size, head_dim*4]
    kv_score_input: torch.Tensor,         # [num_q_tokens, head_dim*4]
    kv_compressed_output: torch.Tensor,   # [num_q_tokens, head_dim]
    ape: torch.Tensor,                    # [8, head_dim]
    indices: torch.Tensor,                # [batch_size]
    compress_plan: torch.Tensor,          # uint8 [num_compress, 16]
    write_plan: torch.Tensor,             # uint8 [num_write, 16]
    extra: torch.Tensor | None,           # [batch_size, 4] int32 (Page4Align) or None
) -> None:
    """c4 prefill — torch port. Two phases:
      1. Apply write_plan: scatter kv_score_input rows into kv_score_buffer
         at the right page positions.
      2. Apply compress_plan: 8-slot softmax-weighted attention per row,
         write to kv_compressed_output.
    """
    page_size = 4 if extra is not None else 8
    head_dim = ape.shape[1]
    indices_l = indices.to(torch.long)
    buf4 = kv_score_buffer.view(*kv_score_buffer.shape[:-1], 4, head_dim)
    inp4 = kv_score_input.view(kv_score_input.shape[0], 4, head_dim)

    write_plans = _decode_plan(write_plan)
    _c4_write_one(buf4, inp4, write_plans, indices_l, page_size, extra)

    compress_plans = _decode_plan(compress_plan)
    _c4_compress_one(
        buf4, inp4, kv_compressed_output, ape, compress_plans,
        indices_l, page_size, extra,
    )


def c128_forward_prefill_torch(
    kv_score_buffer: torch.Tensor,        # [num_indices, 128, head_dim*2]
    kv_score_input: torch.Tensor,         # [num_q_tokens, head_dim*2]
    kv_compressed_output: torch.Tensor,   # [num_q_tokens, head_dim]
    ape: torch.Tensor,                    # [128, head_dim]
    indices: torch.Tensor,                # [batch_size]
    compress_plan: torch.Tensor,          # uint8 [num_compress, 16]
    write_plan: torch.Tensor,             # uint8 [num_write, 16]
) -> None:
    """c128 prefill — torch port. Same shape as c4 prefill but 128 slots,
    no overlap (single segment layout [kv | score])."""
    head_dim = ape.shape[1]
    device = kv_score_buffer.device
    indices_l = indices.to(torch.long)

    # ---- Write phase ----
    write_plans = _decode_plan(write_plan)
    rid_w = write_plans[:, 0]
    valid_w = rid_w != 0xFFFFFFFF
    if valid_w.any():
        rid_w = rid_w[valid_w]
        bid_w = write_plans[valid_w, 1]
        pos_w = write_plans[valid_w, 2]
        page_w = indices_l[bid_w]
        write_pos = pos_w % 128
        kv_score_buffer[page_w, write_pos, :] = kv_score_input[rid_w]

    # ---- Compress phase ----
    compress_plans = _decode_plan(compress_plan)
    rid_c = compress_plans[:, 0]
    valid_c = rid_c != 0xFFFFFFFF
    if not valid_c.any():
        return
    rid_c = rid_c[valid_c]
    bid_c = compress_plans[valid_c, 1]
    pos_c = compress_plans[valid_c, 2]
    wl_c = compress_plans[valid_c, 3]
    A = rid_c.numel()

    # Build buf-source vs input-source per slot.
    slot_arange = torch.arange(128, device=device, dtype=torch.int64)
    in_window = slot_arange.unsqueeze(0) < wl_c.unsqueeze(1)            # [A, 128]
    seq_len = pos_c + 1
    page = indices_l[bid_c]                                              # [A]
    page_pos = (seq_len.unsqueeze(1) + slot_arange.unsqueeze(0)) % 128   # [A, 128]
    src_rid = (rid_c.unsqueeze(1) + slot_arange.unsqueeze(0) - 127).clamp(min=0)

    # Buffer reads
    buf_view = kv_score_buffer.view(*kv_score_buffer.shape[:-1], 2, head_dim)
    page_buf = page.unsqueeze(1).expand(A, 128).long()
    page_pos_l = page_pos.long()
    src_rid_l = src_rid.long()
    buf_kv = buf_view[page_buf, page_pos_l, 0, :]                        # [A, 128, D]
    buf_sc = buf_view[page_buf, page_pos_l, 1, :]

    # Input reads
    inp_view = kv_score_input.view(kv_score_input.shape[0], 2, head_dim)
    inp_kv = inp_view[src_rid_l, 0, :]                                   # [A, 128, D]
    inp_sc = inp_view[src_rid_l, 1, :]

    iw = in_window.unsqueeze(-1)
    kv_stack = torch.where(iw, buf_kv, inp_kv).float()
    sc_stack = torch.where(iw, buf_sc, inp_sc).float() + ape.float().unsqueeze(0)

    weights = torch.softmax(sc_stack, dim=1)
    out = (weights * kv_stack).sum(dim=1)
    kv_compressed_output[rid_c] = out.to(kv_compressed_output.dtype)


def compress_prefill_hip(
    compress_ratio: int,
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    kv_compressed_output: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_plan: torch.Tensor,
    write_plan: torch.Tensor,
    extra: torch.Tensor | None,
) -> None:
    if compress_ratio == 4:
        c4_forward_prefill_torch(
            kv_score_buffer, kv_score_input, kv_compressed_output, ape,
            indices, compress_plan, write_plan, extra,
        )
    elif compress_ratio == 128:
        if extra is not None:
            raise NotImplementedError(
                "c128 prefill with Page4Align — not used in production"
            )
        c128_forward_prefill_torch(
            kv_score_buffer, kv_score_input, kv_compressed_output, ape,
            indices, compress_plan, write_plan,
        )
    else:
        raise ValueError(f"compress_ratio must be 4 or 128, got {compress_ratio}")
