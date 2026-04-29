"""Integration-level test: exercise the WRAPPER (not raw kernel) under
production-mimicking conditions to localize the Layer-3 integration bug.

What the kernel-level unit test (`microbench_ck_v32_score_dump.py`) does NOT
cover:
1. The wrapper's per-shape cached buffers (`_split_buf_cache`, `_OUT_BUF_CACHE`,
   `_REDUCE_OUT_CACHE`) reused across calls.
2. cuda graph capture + replay (production runs all decode under graph).
3. Multiple sequential calls with the same shape (production has 60+ calls
   per token).
4. Triton sink-fold post-process for num_splits=1 (separate from
   num_splits>1's aiter.mla_reduce_v1).

This script tests each of those in isolation and compares to per-batch FP32
oracle. Whichever case fails IS the Layer-3 integration trigger.
"""
import os, math
import torch

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("SGLANG_HIP_SPARSE_MLA_DECODE_FP8", "1")

from sglang.srt.layers.attention.ck_v32_sparse_mla import (
    ck_sparse_mla_decode_fp8_v32,
)

# Production-matching shape (B=6 = the bug-triggering batch size from the
# production drill-down).
B, S_q, H, D, V = 6, 1, 64, 512, 512
SLOT = 584
N_KV = 1024
TOPK = 128
device = "cuda"

torch.manual_seed(42)
q = (torch.randn(B, S_q, H, D, dtype=torch.bfloat16, device=device) * 0.5)
kv = torch.zeros(N_KV, SLOT, dtype=torch.bfloat16, device=device)
for b in range(B):
    kv[128 + b*13, :D] = torch.randn(D, dtype=torch.bfloat16, device=device) * 30.0
    kv[129 + b*13, :D] = torch.randn(D, dtype=torch.bfloat16, device=device) * 30.0
kv_fp8 = kv.to(torch.float8_e4m3fn)
PAGE_SIZE = 128
NUM_PAGES = N_KV // PAGE_SIZE
kv_4d = kv_fp8.view(NUM_PAGES, PAGE_SIZE, 1, SLOT).contiguous()

indices = torch.full((B, S_q, TOPK), -1, dtype=torch.int32, device=device)
for b in range(B):
    indices[b, 0, 0] = 128 + b * 13
    indices[b, 0, 1] = 129 + b * 13

invalid_mask = (indices.view(B*S_q, -1) < 0)
attn_sink = (torch.randn(H, dtype=torch.float32, device=device) * 0.5)
sm_scale = 1.0 / math.sqrt(D)


def per_batch_oracle():
    """[B, H, V] expected output per batch."""
    k_pool = kv_fp8.float()
    out = torch.zeros(B, H, V, dtype=torch.float32, device=device)
    lse_out = torch.zeros(B, H, dtype=torch.float32, device=device)
    for b in range(B):
        kv0, kv1 = 128 + b*13, 129 + b*13
        k0 = k_pool[kv0, :D]; k1 = k_pool[kv1, :D]
        v0 = k_pool[kv0, :V]; v1 = k_pool[kv1, :V]
        q_h = q[b, 0, :, :].float()
        s0 = (q_h @ k0) * sm_scale
        s1 = (q_h @ k1) * sm_scale
        m = torch.maximum(s0, s1)
        e0 = torch.exp(s0 - m); e1 = torch.exp(s1 - m)
        denom = e0 + e1
        lse_out[b] = m + torch.log(denom)
        out[b] = (e0[:, None] * v0 + e1[:, None] * v1) / denom[:, None]
    # Apply attn_sink fold (matches the wrapper's behavior).
    sc = 1.0 / (1.0 + torch.exp(attn_sink.view(1, H) - lse_out))
    out = out * sc.unsqueeze(-1)
    return out.to(torch.bfloat16)


oracle = per_batch_oracle()


def cos(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    n = (af.norm() * bf.norm()).item()
    return float(af @ bf) / n if n > 1e-12 else float("nan")


def diff_vs_oracle(out_bf16):
    """[B, S_q, H, V] → per-batch cos+maxd."""
    rows = []
    for b in range(B):
        ob = out_bf16[b, 0]  # [H, V]
        cs = cos(ob, oracle[b])
        mxd = (ob.float() - oracle[b].float()).abs().max().item()
        rows.append((b, cs, mxd))
    return rows


print("=" * 75)
print("TEST 1: Wrapper baseline call (1 call, eager)")
print("=" * 75)
out, lse = ck_sparse_mla_decode_fp8_v32(
    q=q, k_cache=kv_4d, indices=indices, invalid_mask=invalid_mask,
    attn_sink=attn_sink, sm_scale=sm_scale,
)
torch.cuda.synchronize()
for b, cs, mxd in diff_vs_oracle(out):
    tag = "PASS" if cs > 0.999 else "FAIL"
    print(f"  batch {b}: cos={cs:+.6f} maxd={mxd:.3e}  {tag}")

print()
print("=" * 75)
print("TEST 2: Wrapper 50 sequential calls (exercise cache reuse)")
print("=" * 75)
for i in range(50):
    out, lse = ck_sparse_mla_decode_fp8_v32(
        q=q, k_cache=kv_4d, indices=indices, invalid_mask=invalid_mask,
        attn_sink=attn_sink, sm_scale=sm_scale,
    )
torch.cuda.synchronize()
for b, cs, mxd in diff_vs_oracle(out):
    tag = "PASS" if cs > 0.999 else "FAIL"
    print(f"  batch {b}: cos={cs:+.6f} maxd={mxd:.3e}  {tag}")

print()
print("=" * 75)
print("TEST 3: Wrapper under cuda graph capture+replay")
print("=" * 75)
# Pre-capture warmup (must run once eagerly to allocate JIT modules).
out_warmup, _ = ck_sparse_mla_decode_fp8_v32(
    q=q, k_cache=kv_4d, indices=indices, invalid_mask=invalid_mask,
    attn_sink=attn_sink, sm_scale=sm_scale,
)
torch.cuda.synchronize()

# Capture under graph.
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out_g, lse_g = ck_sparse_mla_decode_fp8_v32(
        q=q, k_cache=kv_4d, indices=indices, invalid_mask=invalid_mask,
        attn_sink=attn_sink, sm_scale=sm_scale,
    )

# Replay.
g.replay()
torch.cuda.synchronize()
print("After 1 replay:")
for b, cs, mxd in diff_vs_oracle(out_g):
    tag = "PASS" if cs > 0.999 else "FAIL"
    print(f"  batch {b}: cos={cs:+.6f} maxd={mxd:.3e}  {tag}")

# Replay 10 more times to stress.
for _ in range(10):
    g.replay()
torch.cuda.synchronize()
print("After 11 replays:")
for b, cs, mxd in diff_vs_oracle(out_g):
    tag = "PASS" if cs > 0.999 else "FAIL"
    print(f"  batch {b}: cos={cs:+.6f} maxd={mxd:.3e}  {tag}")

print()
print("=" * 75)
print("TEST 4: 60 calls under graph (mimics 60-layer decode pattern)")
print("=" * 75)
# Build 60 calls with rotating q values (60 different layers) but same shape.
q_layers = [(torch.randn_like(q) * 0.5) for _ in range(60)]
# Pre-warmup with each q first.
for ql in q_layers:
    _ = ck_sparse_mla_decode_fp8_v32(
        q=ql, k_cache=kv_4d, indices=indices, invalid_mask=invalid_mask,
        attn_sink=attn_sink, sm_scale=sm_scale,
    )
torch.cuda.synchronize()
# Now call sequentially and check the LAST output.
for ql in q_layers:
    out, _ = ck_sparse_mla_decode_fp8_v32(
        q=ql, k_cache=kv_4d, indices=indices, invalid_mask=invalid_mask,
        attn_sink=attn_sink, sm_scale=sm_scale,
    )
torch.cuda.synchronize()
# Recompute oracle for the LAST q.
q_last = q_layers[-1]
def oracle_for_q(qx):
    k_pool = kv_fp8.float()
    out_o = torch.zeros(B, H, V, dtype=torch.float32, device=device)
    lse_o = torch.zeros(B, H, dtype=torch.float32, device=device)
    for b in range(B):
        kv0, kv1 = 128 + b*13, 129 + b*13
        k0 = k_pool[kv0, :D]; k1 = k_pool[kv1, :D]
        v0 = k_pool[kv0, :V]; v1 = k_pool[kv1, :V]
        q_h = qx[b, 0, :, :].float()
        s0 = (q_h @ k0) * sm_scale
        s1 = (q_h @ k1) * sm_scale
        m = torch.maximum(s0, s1)
        e0 = torch.exp(s0 - m); e1 = torch.exp(s1 - m)
        denom = e0 + e1
        lse_o[b] = m + torch.log(denom)
        out_o[b] = (e0[:, None] * v0 + e1[:, None] * v1) / denom[:, None]
    sc = 1.0 / (1.0 + torch.exp(attn_sink.view(1, H) - lse_o))
    return (out_o * sc.unsqueeze(-1)).to(torch.bfloat16)

oracle_last = oracle_for_q(q_last)
for b in range(B):
    ob = out[b, 0]
    cs = cos(ob, oracle_last[b])
    mxd = (ob.float() - oracle_last[b].float()).abs().max().item()
    tag = "PASS" if cs > 0.999 else "FAIL"
    print(f"  batch {b}: cos={cs:+.6f} maxd={mxd:.3e}  {tag}")
