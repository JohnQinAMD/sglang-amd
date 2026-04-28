"""HIP fused kernel for `paged_prefill.expand_seq_lens`.

Drop-in replacement for the CPU Python loop that fills `seq_lens_expanded`
and `expanded_idx_to_unexpanded_idx` per-sequence on pinned host memory and
then `.to(device, non_blocking=True)`s them. The HIP fast-path takes the
already-on-device `forward_batch.seq_lens` / `forward_batch.extend_seq_lens`
and produces the two int32 output tensors with one GPU launch.

Bundled HIP source at `csrc/expand_seq_lens_hip/expand_seq_lens.cu`.
JIT-compiled via `torch.utils.cpp_extension.load` on first import.

Gate via env: `SGLANG_HIP_EXPAND_SEQ_LENS=1` to enable; default off.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch


_BUNDLED_KERNEL_SRC = os.path.join(
    os.path.dirname(__file__), "csrc", "expand_seq_lens_hip"
)

_esl_mod = None


def _kernel_src_dir() -> str:
    return os.environ.get("SGLANG_HIP_EXPAND_SEQ_LENS_KERNEL_SRC_DIR", _BUNDLED_KERNEL_SRC)


def _get_esl_mod():
    """JIT-build the HIP expand_seq_lens module (cached per-process)."""
    global _esl_mod
    if _esl_mod is not None:
        return _esl_mod
    from torch.utils.cpp_extension import load

    src_dir = _kernel_src_dir()
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"HIP expand_seq_lens kernel src dir not found at {src_dir!r}. "
            "Set SGLANG_HIP_EXPAND_SEQ_LENS_KERNEL_SRC_DIR to override."
        )

    old_arch = os.environ.get("PYTORCH_ROCM_ARCH", None)
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    _esl_mod = load(
        name="expand_seq_lens_hip_kernel",
        sources=[os.path.join(src_dir, "expand_seq_lens.cu")],
        extra_include_paths=[src_dir],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )
    if old_arch is not None:
        os.environ["PYTORCH_ROCM_ARCH"] = old_arch
    else:
        os.environ.pop("PYTORCH_ROCM_ARCH", None)
    return _esl_mod


def hip_expand_seq_lens(
    seq_lens: torch.Tensor,         # int32/int64 [batch_size] — on device
    extend_seq_lens: torch.Tensor,  # int32/int64 [batch_size] — on device
    extend_num_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (seq_lens_expanded, expanded_idx_to_unexpanded_idx).

    Both are int32, length `extend_num_tokens`, on the same device as
    `seq_lens`. seq_lens and extend_seq_lens must share dtype.
    """
    if extend_seq_lens.dtype != seq_lens.dtype:
        extend_seq_lens = extend_seq_lens.to(seq_lens.dtype)
    device = seq_lens.device
    seq_lens_expanded = torch.empty(extend_num_tokens, dtype=torch.int32, device=device)
    expanded_idx_to_unexpanded_idx = torch.empty(
        extend_num_tokens, dtype=torch.int32, device=device
    )
    mod = _get_esl_mod()
    mod.expand_seq_lens_hip(
        seq_lens=seq_lens,
        extend_seq_lens=extend_seq_lens,
        out_seq_lens_expanded=seq_lens_expanded,
        out_expanded_idx=expanded_idx_to_unexpanded_idx,
    )
    return seq_lens_expanded, expanded_idx_to_unexpanded_idx
