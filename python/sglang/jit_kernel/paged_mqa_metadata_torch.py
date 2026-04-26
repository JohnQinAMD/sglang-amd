"""HIP fallback for csrc/deepseek_v4/paged_mqa_metadata.cuh.

Port of `IndexerMetadataKernel::run`. The kernel distributes per-batch work
(`ceil(seq_lens[b] / kSplitKV)` units per batch) across `num_sm` SMs, writing
`schedule_metadata[i] = (start_batch_q, offset_within_batch)` for each SM in
`[0, num_sm]`.

The original CUDA kernel uses a single threadblock + warp reductions; the
output is tiny (`(num_sm+1, 2)` int32) and the input is small (one int32 per
batch). Pure-torch on the GPU is fine and capture-safe.
"""
from __future__ import annotations

import torch


_METADATA_CALLS = 0


def get_paged_mqa_logits_metadata_torch(
    seq_lens: torch.Tensor, page_size: int, num_sm: int
) -> torch.Tensor:
    """Returns int32 [num_sm + 1, 2].

    Matches the CUDA kernel's contract:
      schedule_metadata[i, 0] = batch index `q` where SM i starts (B if past end)
      schedule_metadata[i, 1] = `target - cumsum_before_q` (work offset into batch q)
    """
    global _METADATA_CALLS
    _METADATA_CALLS += 1
    if _METADATA_CALLS == 1 or _METADATA_CALLS % 100 == 0:
        print(f"[PAGED_MQA_METADATA_TORCH] HIP fallback call #{_METADATA_CALLS} "
              f"(seq_lens={tuple(seq_lens.shape)}, num_sm={num_sm})", flush=True)
    assert page_size == 64, f"page_size must be 64, got {page_size}"
    assert seq_lens.dtype == torch.int32, f"seq_lens must be int32, got {seq_lens.dtype}"
    kSplitKV = 256

    seq_lens64 = seq_lens.to(torch.int64)
    work = (seq_lens64 + kSplitKV - 1) // kSplitKV          # [B]
    cumsum = torch.cumsum(work, dim=0)                      # [B] inclusive
    B = work.numel()

    if B == 0:
        return torch.zeros(num_sm + 1, 2, dtype=torch.int32, device=seq_lens.device)

    total = cumsum[-1]                                      # scalar tensor
    avg = total // num_sm
    ret = total - avg * num_sm                              # = total % num_sm

    i_grid = torch.arange(num_sm + 1, device=seq_lens.device, dtype=torch.int64)
    targets = i_grid * avg + torch.minimum(i_grid, ret.expand_as(i_grid))   # [num_sm+1]

    # q[i] = first batch where cumsum[q] > target[i]
    q = torch.searchsorted(cumsum, targets, right=True)     # [num_sm+1]
    past = q >= B
    q_clamped = torch.where(past, torch.full_like(q, B - 1), q)

    # cumsum_before_q = cumsum[q-1] if q>0 else 0
    #                = cumsum[q] - work[q] when q < B
    cum_before = torch.where(
        q < B,
        cumsum[q_clamped] - work[q_clamped],
        torch.zeros_like(q),
    )

    out_q = torch.where(past, torch.full_like(q, B), q)
    out_off = torch.where(past, torch.zeros_like(q), targets - cum_before)

    metadata = torch.stack([out_q, out_off], dim=-1).to(torch.int32)        # [num_sm+1, 2]
    return metadata
