"""TileLang kernel for SWA prefill index table.

Lives in its own module so that `from __future__ import annotations` (which
deepseek_v4.py uses) does NOT apply here. With PEP 563, TileLang's
`@tilelang.jit` calls `typing.get_type_hints` on string annotations like
`T.Tensor[(batch_size,), T.int32]`, evaluates them to a TileLang `Buffer`
instance, and then `typing._type_check` rejects it ("Forward references must
evaluate to types. Got buffer."). Defining the kernel here keeps annotations
as actual TileLang objects at definition time, sidestepping the bug.
"""

from typing import Any, Optional

import torch

from sglang.jit_kernel.utils import cache_once


@cache_once
def _tilelang_make_swa_indices_kernel(swa_window_size: int, threads: int = 128) -> Any:
    import tilelang
    import tilelang.language as T

    batch_size = T.dynamic("batch_size")
    batch_size_plus_1 = T.dynamic("batch_size_plus_1")
    num_q_tokens = T.dynamic("num_q_tokens")
    num_warps = threads // 32
    assert swa_window_size % 32 == 0

    @tilelang.jit
    def make_swa_prefill_indices(
        seq_lens_k: T.Tensor[(batch_size,), T.int32],
        seq_lens_q: T.Tensor[(batch_size,), T.int32],
        cu_seqlens_q: T.Tensor[(batch_size_plus_1,), T.int32],
        swa_indices: T.Tensor[(num_q_tokens, swa_window_size), T.int32],
    ):
        _ = batch_size_plus_1  # unused, but don't remove it
        with T.Kernel(T.ceildiv(num_q_tokens, num_warps), threads=threads) as bx:
            tx = T.get_thread_binding()
            warp_id = tx // 32
            lane_id = tx % 32
            s_batch_id = T.alloc_shared((num_warps,), dtype=T.int32)

            token_id = warp_id + bx * num_warps
            if token_id >= num_q_tokens:
                return
            # Bounds-check `j < batch_size` — without this, lanes 0..31 always
            # read cu_seqlens_q[j] / cu_seqlens_q[j+1] regardless of batch_size.
            # When batch_size < 32 (e.g. c=1 single-stream), lanes 1..31 read
            # OOB garbage, which can spuriously satisfy the comparison and pin
            # s_batch_id[warp_id] to an invalid seq_idx → garbage swa_indices →
            # downstream HIP IMA in attention. The ref Python loop at
            # paged_prefill.py is unaffected because it iterates `range(batch_size)`.
            for i in T.serial(0, batch_size, step=32):
                j = i + lane_id
                if j < batch_size:
                    if cu_seqlens_q[j] <= token_id < cu_seqlens_q[j + 1]:
                        s_batch_id[warp_id] = j
            T.sync_warp()

            seq_idx = s_batch_id[warp_id]
            kv_len = seq_lens_k[seq_idx]
            qo_len = seq_lens_q[seq_idx]
            cum_qo_len = cu_seqlens_q[seq_idx]
            prefix_len = kv_len - qo_len
            curr_seq_qo_idx = token_id - cum_qo_len
            end_abs_pos = prefix_len + curr_seq_qo_idx + 1
            start_abs_pos = T.max(end_abs_pos - swa_window_size, 0)
            old_kv_start = seq_idx * swa_window_size
            new_kv_start = batch_size * swa_window_size + cum_qo_len

            for i in T.unroll(0, swa_window_size, step=32):
                j = i + lane_id
                abs_pos = start_abs_pos + j
                swa_indices[token_id, j] = T.if_then_else(
                    abs_pos < end_abs_pos,
                    T.if_then_else(
                        abs_pos < prefix_len,
                        old_kv_start + abs_pos % swa_window_size,
                        new_kv_start + (abs_pos - prefix_len),
                    ),
                    -1,
                )

    return make_swa_prefill_indices


def tilelang_make_swa_prefill_indices(
    seq_lens_k: torch.Tensor,
    seq_lens_q: torch.Tensor,
    swa_indices: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.cumsum(seq_lens_q, dim=0, dtype=torch.int32)
        cu_seqlens_q = torch.nn.functional.pad(cu_seqlens_q, (1, 0), value=0)
    swa_window_size = swa_indices.shape[1]
    kernel = _tilelang_make_swa_indices_kernel(swa_window_size)
    kernel(seq_lens_k, seq_lens_q, cu_seqlens_q, swa_indices)
    return swa_indices
