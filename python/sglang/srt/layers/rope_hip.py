"""HIP RoPE kernel — drop-in replacement for `apply_rotary_emb_triton`.

Goal: reduce the per-launch CPU overhead of the existing Triton RoPE kernel.
Trace measured 194 us CPU x 462 calls = 89 ms / window on DSv4 Flash-Base FP8
(MI355X). The Triton autotune-cache + JIT-cache + arg-serialization paths
dominate; the actual GPU work is trivial (memory-bound, tiny grid at decode).

Bundled HIP source at `csrc/rope_hip/apply_rotary_emb.cu`. JIT-compiled via
`torch.utils.cpp_extension.load` on first import.

Gate via env: `SGLANG_HIP_ROPE=1` to enable; default off.
"""
from __future__ import annotations

import os
from typing import Optional

import torch


_BUNDLED_KERNEL_SRC = os.path.join(
    os.path.dirname(__file__), "csrc", "rope_hip"
)

_rope_mod = None


def _kernel_src_dir() -> str:
    return os.environ.get("SGLANG_HIP_ROPE_KERNEL_SRC_DIR", _BUNDLED_KERNEL_SRC)


def _get_rope_mod():
    """JIT-build the HIP rope module (cached per-process)."""
    global _rope_mod
    if _rope_mod is not None:
        return _rope_mod
    from torch.utils.cpp_extension import load

    src_dir = _kernel_src_dir()
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"HIP rope kernel src dir not found at {src_dir!r}. "
            "Set SGLANG_HIP_ROPE_KERNEL_SRC_DIR to override."
        )

    old_arch = os.environ.get("PYTORCH_ROCM_ARCH", None)
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    _rope_mod = load(
        name="rope_hip_apply_rotary_emb",
        sources=[os.path.join(src_dir, "apply_rotary_emb.cu")],
        extra_include_paths=[src_dir],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )
    if old_arch is not None:
        os.environ["PYTORCH_ROCM_ARCH"] = old_arch
    else:
        os.environ.pop("PYTORCH_ROCM_ARCH", None)
    return _rope_mod


def apply_rotary_emb_hip(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    inverse: bool = False,
) -> None:
    """In-place RoPE rotation. Drop-in replacement for `apply_rotary_emb_triton`.

    Args:
        x: 2d [B, rope_dim] or 3d [B, n_heads, rope_dim] bf16
        freqs_cis: complex64
            - if positions is None: [B, rope_dim // 2] (already indexed)
            - if positions is not None: [max_seqlen, rope_dim // 2]
        positions: int64 [B] or None (eager: index into freqs_cis)
        inverse: bool, if True applies the conjugate rotation
    """
    # `view_as_real(complex64).flatten(-2)` matches the layout the kernel reads:
    # freqs[pos, 2*pair_idx]   = real part
    # freqs[pos, 2*pair_idx+1] = imag part
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)

    mod = _get_rope_mod()
    mod.apply_rotary_emb_hip(
        x=x,
        freqs_real=freqs_real,
        positions=positions,
        is_inverse=inverse,
    )
