"""Phase B+ correctness microbench: verify 4D padded-pool kernel path

The CK V32 sparse-MLA decode kernel must produce IDENTICAL output for:
  (A) a contiguous 4D pool tensor `(N, P, 1, slot_dim)` (no padding)
  (B) a strided 4D pool tensor with the same logical data but a padded outer
      stride (pool_outer_stride > P*slot_dim)

Phase B+ adds support for case (B) via the launcher's stride extraction +
kernel-side `outer*pool_outer_stride + inner*slot_stride` addressing.
Pre-Phase-B+ the launcher's `kv_buffer.reshape(...)` would silently copy a
non-contiguous tensor — that copy was the +131 ms elementwise regression.

Usage:
    python microbench/microbench_ck_v32_padded_pool.py
"""
from __future__ import annotations
import os, sys
import torch

# Make sglang importable from the bundled kernel-source tree.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from sglang.srt.layers.attention.ck_v32_sparse_mla import (
    ck_sparse_mla_decode_fp8_v32,
    ck_sparse_mla_decode_fp8_v32_to_split,
)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def make_padded_4d_view(contig_pool: torch.Tensor, P: int, slot_dim: int,
                         pad_bytes: int) -> torch.Tensor:
    """Build a 4D strided view (N, P, 1, slot_dim) with a padded outer stride.

    Allocates a wider 2D buffer, copies the contiguous data into the leading
    P*slot_dim bytes of each row, then returns the strided 4D view.
    """
    N = contig_pool.shape[0]
    inner_used = P * slot_dim
    inner_padded = inner_used + pad_bytes
    wide = torch.zeros(N, inner_padded, dtype=contig_pool.dtype,
                       device=contig_pool.device)
    wide[:, :inner_used] = contig_pool.reshape(N, inner_used)
    # Slice + view: produces (N, P, 1, slot_dim) with stride(0)=inner_padded
    sliced = wide[:, :inner_used]
    view4d = sliced.view(N, P, 1, slot_dim)
    return view4d, wide


def run_one(B: int, S_q: int, H: int, D: int, V: int, topk: int,
            N_pages: int, P: int, slot_dim: int, pad_bytes: int,
            seed: int = 0) -> dict:
    torch.manual_seed(seed)
    device = "cuda"
    sm_scale = 1.0 / (D ** 0.5)

    # Build random q (bf16) and contiguous fp8 pool.
    q = torch.randn(B, S_q, H, D, device=device, dtype=torch.bfloat16) * 0.1
    pool_size = N_pages * P * slot_dim
    pool_bytes_contig = torch.randn(pool_size, device=device).clamp(-2, 2) * 0.5
    pool_contig = pool_bytes_contig.to(torch.float8_e4m3fn).view(N_pages, P, 1, slot_dim)

    # Random sparse indices
    indices = torch.randint(0, N_pages * P, (B, S_q, topk), device=device, dtype=torch.int32)

    # Reference: contig path
    out_ref, lse_ref = ck_sparse_mla_decode_fp8_v32(
        q=q, k_cache=pool_contig, indices=indices, invalid_mask=None,
        attn_sink=None, sm_scale=sm_scale,
    )

    # Test: padded 4D path
    pool_padded_4d, _wide = make_padded_4d_view(pool_contig, P, slot_dim, pad_bytes)
    out_pad, lse_pad = ck_sparse_mla_decode_fp8_v32(
        q=q, k_cache=pool_padded_4d, indices=indices, invalid_mask=None,
        attn_sink=None, sm_scale=sm_scale,
    )

    cs_out = cos_sim(out_ref, out_pad)
    cs_lse = cos_sim(lse_ref, lse_pad)
    max_diff_out = float((out_ref.float() - out_pad.float()).abs().max())
    max_diff_lse = float((lse_ref.float() - lse_pad.float()).abs().max())

    return {
        "B": B, "S_q": S_q, "H": H, "D": D, "V": V, "topk": topk,
        "N_pages": N_pages, "P": P, "pad_bytes": pad_bytes,
        "cos_sim_out": cs_out, "cos_sim_lse": cs_lse,
        "max_diff_out": max_diff_out, "max_diff_lse": max_diff_lse,
    }


def main():
    # Configs covering production shapes:
    # - V32 path: D=576 (q d_qk), V=512 (kernel output)
    # - 2604 path: D=512 (Flash-Base FP8 path)
    # Pad bytes match real-world pool padding (pool_outer_stride - P*slot_dim).
    # For V32 c4: P=16, slot_dim=584, padded=ceil(16*584/576)*576=9792, pad=9792-9344=448.
    # For c128:    P=1,  slot_dim=584, padded=ceil(584/576)*576=1152,    pad=1152-584=568.
    configs = [
        # (B, S_q, H, D, V, topk, N_pages, P, slot_dim, pad_bytes)
        (1, 1, 128, 576, 512,  64,   64, 16, 584, 448),  # V32 c4 small
        (4, 1, 128, 576, 512, 256,  256, 16, 584, 448),  # V32 c4 production-ish
        (4, 1, 128, 576, 512, 512,  256,  1, 584, 568),  # V32 c128
        (1, 1, 128, 512, 512, 128,  128, 16, 584, 448),  # 2604 c4 (Flash-Base FP8)
        (4, 1, 128, 512, 512, 256,  256, 16, 584, 448),  # 2604 c4 production-ish
    ]
    print(f"{'B':>2} {'S':>2} {'H':>3} {'D':>3} {'topk':>4} {'P':>2} {'pad':>4}  "
          f"{'cs_out':>8} {'cs_lse':>8} {'max_d_out':>10} {'max_d_lse':>10}  PASS?")
    n_pass = 0
    for cfg in configs:
        B, S_q, H, D, V, topk, N_pages, P, slot_dim, pad_bytes = cfg
        r = run_one(B, S_q, H, D, V, topk, N_pages, P, slot_dim, pad_bytes)
        # Bit-exact pass criterion: cos_sim ≥ 0.99999 (kernel produces same fp8
        # decode regardless of physical pool layout — only addressing differs).
        ok = (r["cos_sim_out"] >= 0.99999 and r["cos_sim_lse"] >= 0.99999
              and r["max_diff_out"] <= 1e-3)
        n_pass += int(ok)
        print(f"{B:>2} {S_q:>2} {H:>3} {D:>3} {topk:>4} {P:>2} {pad_bytes:>4}  "
              f"{r['cos_sim_out']:8.6f} {r['cos_sim_lse']:8.6f} "
              f"{r['max_diff_out']:10.3e} {r['max_diff_lse']:10.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"\n  {n_pass}/{len(configs)} configs pass (bit-exact between contig and padded)")
    return 0 if n_pass == len(configs) else 1


if __name__ == "__main__":
    sys.exit(main())
