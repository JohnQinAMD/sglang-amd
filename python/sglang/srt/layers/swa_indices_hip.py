"""HIP fused kernel for `make_swa_ring_buffer_indices`.

Drop-in replacement for the broken TileLang fast-path
(`sglang.jit_kernel.swa_indices_tilelang.tilelang_make_swa_prefill_indices`)
which compiles to an empty kernel body on this gfx950/TVM stack — see
`microbench_swa_indices.py` for the repro.

Replaces the CPU Python double-loop in `make_swa_ring_buffer_indices` (~5,120
inner iterations / prefill batch for production shapes) and the final
pinned-memory H2D copy with a single GPU dispatch.

Bundled HIP source at `csrc/swa_indices_hip/make_swa_indices.cu`. JIT-compiled
via `torch.utils.cpp_extension.load` on first import.

Gate via env: `SGLANG_HIP_SWA_PREPARE=1` to enable; default off.
"""
from __future__ import annotations

import os
from typing import Optional

import torch


_BUNDLED_KERNEL_SRC = os.path.join(
    os.path.dirname(__file__), "csrc", "swa_indices_hip"
)

_swa_mod = None


def _kernel_src_dir() -> str:
    return os.environ.get("SGLANG_HIP_SWA_PREPARE_KERNEL_SRC_DIR", _BUNDLED_KERNEL_SRC)


def _get_swa_mod():
    """JIT-build the HIP swa-indices module (cached per-process)."""
    global _swa_mod
    if _swa_mod is not None:
        return _swa_mod
    from torch.utils.cpp_extension import load

    src_dir = _kernel_src_dir()
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"HIP swa-indices kernel src dir not found at {src_dir!r}. "
            "Set SGLANG_HIP_SWA_PREPARE_KERNEL_SRC_DIR to override."
        )

    old_arch = os.environ.get("PYTORCH_ROCM_ARCH", None)
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    _swa_mod = load(
        name="swa_indices_hip_make_swa_indices",
        sources=[os.path.join(src_dir, "make_swa_indices.cu")],
        extra_include_paths=[src_dir],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )
    if old_arch is not None:
        os.environ["PYTORCH_ROCM_ARCH"] = old_arch
    else:
        os.environ.pop("PYTORCH_ROCM_ARCH", None)
    return _swa_mod


def hip_make_swa_prefill_indices(
    seq_lens_k: torch.Tensor,
    seq_lens_q: torch.Tensor,
    swa_indices: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,  # ignored; kept for back-compat
) -> torch.Tensor:
    """In-place fill of `swa_indices`. Drop-in for the TileLang wrapper.

    The kernel computes cu_seqlens_q on-the-fly inside the GPU launch, so the
    caller can pass `seq_lens_q` directly (int32 or int64) without first
    cumsum/pad'ing it. This drops 2 small kernel launches per call. The
    `cu_seqlens_q` arg is accepted for backward-compat but ignored.

    Args:
        seq_lens_k:   int32/int64 [batch_size] — total KV length per sequence.
        seq_lens_q:   int32/int64 [batch_size] — query length per sequence
                      (extend_seq_lens). Must match seq_lens_k's dtype.
        swa_indices:  int32 [num_q_tokens, swa_window_size] (output, contiguous).

    Returns:
        The same `swa_indices` tensor, filled in-place.
    """
    del cu_seqlens_q  # silenced for back-compat; kernel computes it internally
    mod = _get_swa_mod()
    mod.make_swa_indices_hip(
        seq_lens_k=seq_lens_k,
        seq_lens_q=seq_lens_q,
        swa_indices=swa_indices,
    )
    return swa_indices
