"""TurboQuant core engine: codebook, rotation, quantize, pack/unpack.

Implements the TurboQuant algorithm from "TurboQuant: Online Vector Quantization
with Near-optimal Distortion Rate" (ICLR 2026). Data-oblivious quantization via
random rotation + Lloyd-Max scalar quantization for N(0,1).

This module is framework-agnostic (no SGLang/vLLM dependencies).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Lloyd-Max codebook for N(0,1)
# ---------------------------------------------------------------------------

_CODEBOOK_CACHE: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


def _compute_lloyd_max_gaussian(
    n_levels: int, n_iters: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max optimal centroids and boundaries for N(0,1)."""
    from scipy.stats import norm

    boundaries = np.linspace(-3.5, 3.5, n_levels + 1)
    boundaries[0] = -1e10
    boundaries[-1] = 1e10
    centroids = np.zeros(n_levels)

    for _ in range(n_iters):
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            p = norm.cdf(hi) - norm.cdf(lo)
            if p > 1e-15:
                centroids[i] = (norm.pdf(lo) - norm.pdf(hi)) / p
            else:
                centroids[i] = (max(lo, -3.5) + min(hi, 3.5)) / 2

        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    return centroids, boundaries


def get_codebook(bit_width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get (centroids, inner_boundaries) for given bit-width, cached globally.

    Returns:
        centroids: (2^bit_width,) float32
        boundaries: (2^bit_width - 1,) float32  (inner boundaries only)
    """
    if bit_width not in _CODEBOOK_CACHE:
        n_levels = 2 ** bit_width
        centroids, boundaries = _compute_lloyd_max_gaussian(n_levels)
        _CODEBOOK_CACHE[bit_width] = (
            torch.tensor(centroids, dtype=torch.float32),
            torch.tensor(boundaries[1:-1], dtype=torch.float32),
        )
    return _CODEBOOK_CACHE[bit_width]


# ---------------------------------------------------------------------------
# Rotation matrix generation (Haar-distributed random orthogonal)
# ---------------------------------------------------------------------------

_ROTATION_CACHE: Dict[int, torch.Tensor] = {}


def generate_rotation_matrix(d: int, seed: int = 42) -> torch.Tensor:
    """Generate Haar-distributed random orthogonal matrix via QR of Gaussian.

    Cached by (d, seed) key. Result is always float32 on CPU;
    caller should .to(device) as needed.
    """
    key = (d, seed)
    if key not in _ROTATION_CACHE:
        gen = torch.Generator().manual_seed(seed)
        G = torch.randn(d, d, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        Q = Q * diag_sign.unsqueeze(0)
        _ROTATION_CACHE[key] = Q
    return _ROTATION_CACHE[key]


def clear_rotation_cache():
    _ROTATION_CACHE.clear()


# ---------------------------------------------------------------------------
# 4-bit packing / unpacking
# ---------------------------------------------------------------------------


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices (0-15) into uint8, 2 per byte.
    Layout: byte = lo_nibble | (hi_nibble << 4).
    """
    assert indices.shape[-1] % 2 == 0, "Last dim must be even"
    lo = indices[..., 0::2].to(torch.uint8)
    hi = indices[..., 1::2].to(torch.uint8)
    return lo | (hi << 4)


def unpack_4bit(packed: torch.Tensor, orig_last_dim: int) -> torch.Tensor:
    """Unpack uint8 -> 4-bit indices as int32."""
    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)
    result = torch.stack([lo, hi], dim=-1)
    return result.reshape(*packed.shape[:-1], packed.shape[-1] * 2)[
        ..., :orig_last_dim
    ]


# ---------------------------------------------------------------------------
# Single-pass quantization
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_quantize_packed(
    W: torch.Tensor,
    bit_width: int = 4,
    group_size: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Quantize weight matrix and return packed representation.

    Args:
        W: (out_features, in_features) weight matrix
        bit_width: bits per element (4-bit packing only)
        group_size: group size along in_features (None = full row)
        seed: rotation seed

    Returns dict with indices_packed, codebook, norms, seed, group_size, shape, bit_width.
    """
    assert bit_width == 4, "Packed format currently supports 4-bit only"
    M, N = W.shape
    if group_size is None:
        group_size = N

    W = W.float()
    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    all_norms = []
    all_indices = []

    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(W.device)
        Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    full_indices = torch.cat(all_indices, dim=1)
    norms_out = (
        torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]
    )

    if N % 2 != 0:
        full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)

    packed = pack_4bit(full_indices)

    return {
        "indices_packed": packed,
        "codebook": centroids.cpu(),
        "norms": norms_out.cpu(),
        "seed": seed,
        "group_size": group_size,
        "shape": (M, N),
        "bit_width": bit_width,
    }


@torch.no_grad()
def turboquant_dequantize(packed_data: dict, device: torch.device) -> torch.Tensor:
    """Reconstruct full weight from packed representation."""
    M, N = packed_data["shape"]
    group_size = packed_data["group_size"]
    seed = packed_data["seed"]

    indices_packed = packed_data["indices_packed"].to(device)
    codebook = packed_data["codebook"].to(device)
    norms = packed_data["norms"].to(device)

    padded_N = N if N % 2 == 0 else N + 1
    indices = unpack_4bit(indices_packed, padded_N)[:, :N]

    n_groups = math.ceil(N / group_size)
    W_approx = torch.zeros(M, N, dtype=torch.float32, device=device)

    for g in range(n_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start

        Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
        scale = math.sqrt(g_dim)

        Y_g = codebook[indices[:, g_start:g_end].long()] / scale
        W_g = Y_g @ Pi

        if norms.dim() == 1:
            W_g = W_g * norms.unsqueeze(1)
        else:
            W_g = W_g * norms[:, g].unsqueeze(1)

        W_approx[:, g_start:g_end] = W_g

    return W_approx


# ---------------------------------------------------------------------------
# Residual (two-pass) quantization
# ---------------------------------------------------------------------------


@torch.no_grad()
def residual_quantize_packed(
    W: torch.Tensor,
    bit_width_1: int = 4,
    bit_width_2: int = 4,
    group_size: Optional[int] = None,
    seed_1: int = 42,
    seed_2: int = 1042,
) -> dict:
    """Two-pass residual TurboQuant: returns packed reps for both passes."""
    pass1 = turboquant_quantize_packed(
        W, bit_width=bit_width_1, group_size=group_size, seed=seed_1
    )

    W_hat1 = turboquant_dequantize(pass1, device=W.device)
    residual = W.float() - W_hat1

    pass2 = turboquant_quantize_packed(
        residual, bit_width=bit_width_2, group_size=group_size, seed=seed_2
    )

    return {
        "pass1": pass1,
        "pass2": pass2,
        "total_bits": bit_width_1 + bit_width_2,
    }


# ---------------------------------------------------------------------------
# On-the-fly forward pass (PyTorch fallback, no Triton)
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_matmul_pytorch(
    x: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    weight_norms: torch.Tensor,
    in_features: int,
    group_size: int,
    seed: int,
) -> torch.Tensor:
    """On-the-fly dequant matmul: y = x @ W^T using packed representation.

    Approach C: rotate input instead of dequantizing full weight.
    """
    B = x.shape[0]
    N = indices_packed.shape[0]
    K = in_features
    device = x.device
    n_groups = math.ceil(K / group_size)
    scale = math.sqrt(group_size)

    padded_K = K if K % 2 == 0 else K + 1
    indices = unpack_4bit(indices_packed, padded_K)[:, :K]

    output = torch.zeros(B, N, dtype=torch.float32, device=device)

    for g in range(n_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, K)
        g_dim = g_end - g_start

        Pi_g = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
        x_rot_g = x[:, g_start:g_end].float() @ Pi_g.T

        idx_g = indices[:, g_start:g_end]
        W_g = codebook[idx_g.long()]

        out_g = x_rot_g @ W_g.T

        if weight_norms.dim() == 1:
            norms_g = weight_norms
        else:
            norms_g = weight_norms[:, g]

        out_g = out_g * (norms_g[None, :] / scale)
        output += out_g

    return output
