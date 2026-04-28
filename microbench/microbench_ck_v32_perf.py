"""Performance bench: CK V32 single-shot path vs torch BF16 ref.

Tests if the kernel actually delivers a speedup over the fall-through ref
path (which is what production uses today via the Layer-3 stopgap).
"""
import os, time, math
import torch
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("SGLANG_HIP_SPARSE_MLA_DECODE_FP8", "1")

# Import the wrapper (uses cached production kernel build, no DEBUG flag).
from sglang.srt.layers.attention.ck_v32_sparse_mla import (
    ck_sparse_mla_decode_fp8_v32,
)

# Production-matching shape (Flash mxfp4 c=4 decode).
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


def bench_ck_v32(iters=200):
    for _ in range(10):
        out, lse = ck_sparse_mla_decode_fp8_v32(
            q=q, k_cache=kv_4d, indices=indices,
            invalid_mask=invalid_mask, attn_sink=attn_sink, sm_scale=sm_scale,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out, lse = ck_sparse_mla_decode_fp8_v32(
            q=q, k_cache=kv_4d, indices=indices,
            invalid_mask=invalid_mask, attn_sink=attn_sink, sm_scale=sm_scale,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def bench_torch_ref(iters=200):
    out_bf16 = torch.empty((B*S_q, H, V), dtype=torch.bfloat16, device=device)
    def step():
        k_pool_bf16 = kv_fp8.float().to(torch.bfloat16)
        for b in range(B):
            idx_raw = indices[b, 0].long()
            invalid = idx_raw < 0
            idx_safe = torch.clamp(idx_raw, min=0)
            k_g = k_pool_bf16[idx_safe, :D]
            v_g = k_pool_bf16[idx_safe, :V]
            q_h = q[b, 0]
            scores = (q_h.float() @ k_g.float().T) * sm_scale
            scores = torch.where(invalid.view(1, -1),
                                 torch.tensor(float("-inf"), device=device), scores)
            attn = torch.softmax(scores, dim=-1)
            out_bf16[b] = (attn @ v_g.float()).to(torch.bfloat16)
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


us_ck = bench_ck_v32()
us_ref = bench_torch_ref()

print(f"=== Performance: CK V32 (production fast path) vs torch BF16 ref ===")
print(f"Shape: B={B}, H={H}, topk={TOPK}, num_splits=auto (4 for B=6)")
print(f"  CK V32 + reduce + sink-fold:  {us_ck:8.2f} µs/call")
print(f"  Torch BF16 ref (dequant+attn): {us_ref:8.2f} µs/call")
if us_ref > us_ck:
    print(f"  Speedup:                       {us_ref / us_ck:6.2f}x  <-- CK V32 wins")
else:
    print(f"  Slowdown:                      {us_ck / us_ref:6.2f}x  <-- CK V32 LOSES")
print(f"  Per-call delta:                {us_ref - us_ck:+8.2f} us")
print(f"  At 60 layers/forward x 1 forward/token: {(us_ref - us_ck) * 60 / 1000:+.2f} ms/token TPOT delta")
