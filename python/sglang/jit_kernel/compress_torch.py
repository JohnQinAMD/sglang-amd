"""HIP fallback for csrc/deepseek_v4/common.cuh and the prefill-plan portion
of compress_forward.

`plan_compress_prefill` is a pure indexing kernel that emits two PrefillPlan
arrays from (extend_lens, seq_lens, compress_ratio, is_overlap). It's small
and self-contained — pure-torch port works fine.

`compress_forward` (c4.cuh / c128.cuh) does softmax-weighted compression
across an 8-slot history buffer with paged or ring-buffer addressing modes;
that's ~20K lines of CUDA per ratio with overlap windows and is non-trivial
to port. The HIP path is intentionally a clear failure (NotImplementedError)
since DSv4-Flash on HIP today bypasses the compressor entirely.
"""
from __future__ import annotations

from typing import Tuple

import torch


# 16-byte PrefillPlan: 4 × uint32 (ragged_id, batch_id, position, window_len)
_kPrefillPlanFields = 4
_kInvalid = 0xFFFFFFFF


def plan_compress_prefill_torch(
    extend_lens: torch.Tensor,                # int64 [B]
    seq_lens: torch.Tensor,                   # int64 [B]
    compress_plan: torch.Tensor,              # uint8 [num_tokens, 16]
    write_plan: torch.Tensor,                 # uint8 [num_tokens, 16]
    compress_ratio: int,
    is_overlap: bool,
    use_cuda_graph: bool,
) -> Tuple[int, int]:
    """Mirrors `plan_prefill_host` in common.cuh.

    For each batch i and each token j in [0, extend_len[i]):
      position = prefix_len + j         where prefix_len = seq_len - extend_len
      ratio = compress_ratio * (1 + is_overlap)
      window_len = ratio - min(j+1, ratio)
      plan = (ragged_id=counter+j, batch_id=i, position, window_len)

      if (position + 1) % compress_ratio == 0:  → emit to compress_plan
      if position >= start_write_pos:            → emit to write_plan

    where start_write_pos = (seq_len // compress_ratio) * compress_ratio
    minus compress_ratio if is_overlap (clamped to ≥ 0).

    Returns (num_compress, num_write). If use_cuda_graph: pads both to
    num_tokens with kInvalidPlan and returns (num_tokens, num_tokens).
    """
    assert compress_plan.dtype == torch.uint8
    assert write_plan.dtype == torch.uint8
    assert compress_plan.shape == write_plan.shape
    num_tokens, plan_dim = compress_plan.shape
    assert plan_dim == 16

    device = extend_lens.device
    extend_lens64 = extend_lens.to(torch.int64)
    seq_lens64 = seq_lens.to(torch.int64)
    B = extend_lens64.numel()

    # Build per-token (batch_id, j_within_batch) by repeat_interleave
    counts = extend_lens64
    batch_id = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int64), counts
    )                                                         # [N]
    cum = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int64),
        torch.cumsum(counts, dim=0),
    ])                                                        # [B+1]
    j = torch.arange(cum[-1].item(), device=device, dtype=torch.int64) - cum[batch_id]
    ragged_id = torch.arange(cum[-1].item(), device=device, dtype=torch.int64)

    extend_len = extend_lens64[batch_id]
    seq_len = seq_lens64[batch_id]
    prefix_len = seq_len - extend_len
    position = prefix_len + j
    ratio = compress_ratio * (2 if is_overlap else 1)
    window_len = ratio - torch.minimum(
        j + 1, torch.full_like(j, ratio)
    )

    if is_overlap:
        start_write_pos = torch.maximum(
            (seq_len // compress_ratio) * compress_ratio - compress_ratio,
            torch.zeros_like(seq_len),
        )
    else:
        start_write_pos = (seq_len // compress_ratio) * compress_ratio

    compress_mask = ((position + 1) % compress_ratio == 0)
    write_mask = position >= start_write_pos

    # Build the per-position plan rows as 4 × uint32, then byte-view as uint8 [N, 16]
    plan_words = torch.stack([
        ragged_id.to(torch.int64),
        batch_id.to(torch.int64),
        position.to(torch.int64),
        window_len.to(torch.int64),
    ], dim=-1).to(torch.uint32).contiguous()                   # [N, 4] uint32
    plan_bytes = plan_words.view(torch.uint8).view(-1, 16)     # [N, 16]

    # Compaction (no atomic — host-side: stable order matches CPU loop)
    compress_idx = compress_mask.nonzero(as_tuple=False).flatten()
    write_idx = write_mask.nonzero(as_tuple=False).flatten()
    num_compress = int(compress_idx.numel())
    num_write = int(write_idx.numel())

    assert num_compress <= num_tokens, \
        f"compress_plan overflow: {num_compress} > {num_tokens}"
    assert num_write <= num_tokens, \
        f"write_plan overflow: {num_write} > {num_tokens}"

    compress_plan[:num_compress] = plan_bytes[compress_idx]
    write_plan[:num_write] = plan_bytes[write_idx]

    if use_cuda_graph:
        invalid_word = torch.tensor([_kInvalid] * 4, dtype=torch.uint32,
                                    device=device).view(torch.uint8)
        compress_plan[num_compress:] = invalid_word
        write_plan[num_write:] = invalid_word
        return num_tokens, num_tokens
    return num_compress, num_write


def compress_forward_hip_unsupported(*args, **kwargs):
    raise NotImplementedError(
        "compress_forward (csrc/deepseek_v4/c4.cuh, c128.cuh) has no HIP port. "
        "DSv4-Flash on HIP runs the no-compressor path; if you reach this "
        "from a code path that needs compression, port the c4/c128 kernels "
        "to Triton (8-slot softmax-weighted attention with paged or ring "
        "buffer addressing) or route around the compressor."
    )
