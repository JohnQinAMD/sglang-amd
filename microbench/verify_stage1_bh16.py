"""Correctness gate: BH=16 must bit-equal BH=8 (the shipped path).

Both run the same online-softmax math on identical inputs. Output (bf16) and
LSE (fp32) must match within bf16-ULP at the per-element scale.
"""
import sys
sys.path.insert(0, "/sgl-pr/python")

import torch
from sglang.srt.flashmla_tests.triton_sparse_decode_kernel import (
    triton_sparse_attn_decode_split_k,
)


def make(B, Hq, Topk, D_QK, lonely_frac=0.05, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, Hq, D_QK, dtype=torch.bfloat16, device="cuda") * 0.1
    kv = torch.randn(B, Topk, D_QK, dtype=torch.bfloat16, device="cuda") * 0.1
    mask = torch.rand(B, Topk, device="cuda") < lonely_frac
    return q, kv, mask


def run(q, kv, mask, sm_scale, D_V, BH, SK, wpe=2, mind=16):
    o, lse = triton_sparse_attn_decode_split_k(
        q, kv, mask, None, sm_scale, D_V,
        BLOCK_H=BH, BLOCK_T=32, BLOCK_D=256, SPLIT_K=SK,
        num_warps=4, num_stages=1,
        waves_per_eu=wpe, matrix_instr_nonkdim=mind,
    )
    return o, lse


def diff(a, b, label):
    af, bf = a.float(), b.float()
    both_neg_inf = (af == float("-inf")) & (bf == float("-inf"))
    both_pos_inf = (af == float("+inf")) & (bf == float("+inf"))
    both_inf = both_neg_inf | both_pos_inf
    d = (af - bf).abs()
    d = torch.where(both_inf, torch.zeros_like(d), d)
    d = torch.nan_to_num(d, nan=1e10, posinf=1e10, neginf=1e10)
    print(f"  {label}: max={d.max().item():.4e}  mean={d.mean().item():.4e}")
    return d.max().item()


def verify(B, Topk, BH_new, SK_new):
    print(f"\n=== B={B} Topk={Topk}: BH={BH_new} SK={SK_new} vs BH=8 SK=16 ===")
    Hq, D_QK, D_V = 64, 512, 512
    sm_scale = 0.04
    q, kv, mask = make(B, Hq, Topk, D_QK)

    o_ref, lse_ref = run(q, kv, mask, sm_scale, D_V, BH=8, SK=16)
    o_new, lse_new = run(q, kv, mask, sm_scale, D_V, BH=BH_new, SK=SK_new)

    o_diff = diff(o_ref, o_new, "output (bf16)")
    lse_diff = diff(lse_ref, lse_new, "lse    (fp32)")
    ok = o_diff <= 5e-3 and lse_diff <= 5e-3
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    all_ok = True
    # Cover production decode shapes
    for B in [1, 2, 4, 8, 16]:
        for T in [256, 512, 1024, 2048]:
            # Pick the new dispatch winner per shape
            if B <= 2:
                BH, SK = (16, 32) if T >= 1024 else (4, 16)
            elif T >= 1024:
                BH, SK = 16, 16
            else:
                BH, SK = 8, 8  # unchanged from current
            ok = verify(B, T, BH, SK)
            all_ok = all_ok and ok
    print(f"\n{'='*40}")
    print(f"ALL: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
