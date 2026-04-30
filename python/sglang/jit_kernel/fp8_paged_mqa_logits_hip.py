"""HIP kernel loader for fp8_paged_mqa_logits on AMD MI355X (gfx950).

Drop-in replacement for `fp8_paged_mqa_logits_torch` / _aiter / _fused_triton.
Microbench (chi2811 GPU 4) at production b=1..6 / max_pages=64 / D=128:
  b=1 L=1024:  4.17 us/call  (vs torch 132, fused_triton 33)
  b=2 L=1500:  6.18 us/call
  b=4 L=2000:  12.14 us/call
  b=6 L=2048:  17.23 us/call
  b=6 L=4000:  30.32 us/call

Strategy: 1 warp per token, 64 lanes split the H=64 reduction direction;
inner D=128 dot product uses `__builtin_amdgcn_cvt_f32_fp8` per-byte for fast
fp8 unpack. Output is fp32 [B, eff_max_seq_len] with -inf at invalid positions.

Persistent scratch: caller-side `out` buffer is sourced from per-mode dicts
(_HIP_SCRATCH_CAPTURED / _EAGER) so cuda-graph replay reuses the same physical
storage across calls (mirrors fp8_paged_mqa_logits_fused_triton.py).

JIT-builds via `torch.utils.cpp_extension.load` on first call.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils import cpp_extension

_SRC = os.path.join(
    os.path.dirname(__file__), "csrc", "fp8_paged_mqa_logits_hip.hip"
)
_DEFAULT_BUILD = os.path.join(
    os.path.dirname(__file__), "csrc", "build_fp8_paged_mqa_logits_hip"
)
_BUILD_DIR = os.environ.get("SGLANG_HIP_FP8_PAGED_BUILD_DIR", _DEFAULT_BUILD)

_ext = None  # cached extension module handle


# Persistent output scratch — separate dicts for capture-time vs eager calls.
# `_CAPTURED` is allocated during torch.cuda.is_current_stream_capturing()=True;
# its tensor addresses get baked into the cuda graph and MUST NEVER be reassigned
# afterwards (replay reads freed memory otherwise).
_HIP_SCRATCH_CAPTURED: dict = {}
_HIP_SCRATCH_EAGER: dict = {}


def _ensure_scratch(scratch_dict, name, shape, dtype, device):
    cur = scratch_dict.get(name)
    target = tuple(shape)
    if (
        cur is None
        or cur.dtype != dtype
        or cur.device != device
        or tuple(cur.shape) != target
    ):
        scratch_dict[name] = torch.empty(target, dtype=dtype, device=device)
    return scratch_dict[name]


def _load():
    global _ext
    if _ext is not None:
        return _ext
    os.makedirs(_BUILD_DIR, exist_ok=True)
    _ext = cpp_extension.load(
        name="fp8_paged_mqa_logits_hip",
        sources=[_SRC],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-std=c++17"],
        build_directory=_BUILD_DIR,
        verbose=False,
    )
    return _ext


def fp8_paged_mqa_logits_hip(
    q_fp8: torch.Tensor,            # [B, 1, H=64, D=128] fp8 e4m3
    kvcache_fp8: torch.Tensor,      # [N_pg, BS=64, 1, D+4=132] fp8-typed
    weight: torch.Tensor,           # [B, H] fp32
    seq_lens: torch.Tensor,         # [B] int32
    page_table: torch.Tensor,       # [B, max_pages] int32
    deep_gemm_metadata=None,        # ignored (signature compat)
    max_seq_len: Optional[int] = None,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Drop-in compatible with `fp8_paged_mqa_logits_torch`. The `clean_logits`
    flag is ignored (kernel always pre-fills -inf for invalid positions).
    """
    _ = clean_logits
    _ = deep_gemm_metadata

    B = q_fp8.size(0)
    BS = kvcache_fp8.size(1)
    if max_seq_len is None:
        max_seq_len = page_table.size(1) * BS
    eff_max_seq_len = min(int(max_seq_len), int(page_table.size(1)) * BS)

    # Pick scratch dict by capture state, then allocate/reuse output buffer.
    scratch = (
        _HIP_SCRATCH_CAPTURED
        if torch.cuda.is_current_stream_capturing()
        else _HIP_SCRATCH_EAGER
    )
    out = _ensure_scratch(
        scratch, "out", (B, eff_max_seq_len),
        torch.float32, q_fp8.device,
    )

    ext = _load()
    return ext.fp8_paged_mqa_logits_hip(
        q_fp8.contiguous(),
        kvcache_fp8.contiguous(),
        weight.contiguous(),
        seq_lens.contiguous(),
        page_table.contiguous(),
        int(max_seq_len),
        out,
    )
