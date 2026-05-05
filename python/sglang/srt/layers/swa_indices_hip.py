"""HIP fused kernel for `make_swa_ring_buffer_indices`.

Replaces the CPU Python double-loop in `make_swa_ring_buffer_indices` and the
final pinned-memory H2D copy with a single GPU dispatch. Bundled HIP source at
`csrc/swa_indices_hip/make_swa_indices.cu`, JIT-compiled via
`torch.utils.cpp_extension.load` on first import.
"""
from __future__ import annotations

import os

import torch


_BUNDLED_KERNEL_SRC = os.path.join(
    os.path.dirname(__file__), "csrc", "swa_indices_hip"
)

_swa_mod = None


def _get_swa_mod():
    """JIT-build the HIP swa-indices module (cached per-process)."""
    global _swa_mod
    if _swa_mod is not None:
        return _swa_mod
    from torch.utils.cpp_extension import load

    src_dir = _BUNDLED_KERNEL_SRC
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    old_arch = os.environ.get("PYTORCH_ROCM_ARCH", None)
    os.environ["PYTORCH_ROCM_ARCH"] = arch
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
) -> torch.Tensor:
    """In-place fill of `swa_indices`.

    Args:
        seq_lens_k:   int32/int64 [batch_size] — total KV length per sequence.
        seq_lens_q:   int32/int64 [batch_size] — query length per sequence.
                      Must match seq_lens_k's dtype.
        swa_indices:  int32 [num_q_tokens, swa_window_size] (output, contiguous).
    """
    mod = _get_swa_mod()
    mod.make_swa_indices_hip(
        seq_lens_k=seq_lens_k,
        seq_lens_q=seq_lens_q,
        swa_indices=swa_indices,
    )
    return swa_indices
