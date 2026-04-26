"""Microbench: CK V32 sparse MLA (qk_head_dim=512) vs torch reference.

Goal: identify why CK V32 produces wrong output for Flash's qk_head_dim=512.
Hypothesis: FP8 decode uses fnuz bias-8 fold but actual storage is fn (bias=7).

Setup matches Flash decode: B=1, S_q=1, H=128 (q heads), D=512 (qk_head_dim),
V=512 (v_head_dim), topk=64 (sparse routing).
"""
import torch
import os

os.environ["SGLANG_HIP_SPARSE_MLA_DECODE_FP8"] = "1"  # gate the CK V32 path

# Shape: parametrize via env (D=qk_head_dim 512=Flash 576=Pro V32)
import os as _os
B, S_q, H, V = 1, 1, 128, 512
D = int(_os.environ.get("D", "512"))
TOPK = 64
N_KV = 1024  # KV cache slots

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
device = "cuda"

q = (torch.randn(B, S_q, H, D, dtype=torch.bfloat16, device=device) * 0.1)
# Production stores 584-byte slot (nope=512 + rope=64 + scale=8). Flash uses
# only the first 512 bytes (no rope). Allocate matching layout.
SLOT = 584
k_cache_bf16 = torch.randn(N_KV, SLOT, dtype=torch.bfloat16, device=device) * 0.1

# Convert to fp8 with both formats, observe difference.
import torch as _t
FN = _t.float8_e4m3fn
# FNUZ may not exist — fall back to FN if not.
FNUZ = getattr(_t, "float8_e4m3fnuz", FN)

# Quantize to FN (what MI355X actually uses)
k_cache_fn = k_cache_bf16.to(FN)

# Quantize to FNUZ (what the kernel was designed around)
k_cache_fnuz = k_cache_bf16.to(FNUZ) if FNUZ is not FN else k_cache_fn

# We pass the float8 tensors DIRECTLY (no uint8 view) so the wrapper can
# detect the storage format from .dtype and choose the right decode scale.
print(f"  fn dtype: {k_cache_fn.dtype}, fnuz dtype: {k_cache_fnuz.dtype}")
print(f"  fn byte[0,:8]:   {k_cache_fn.view(_t.uint8)[0,:8].tolist()}")
print(f"  fnuz byte[0,:8]: {k_cache_fnuz.view(_t.uint8)[0,:8].tolist()}")

# Test the kernel
indices = torch.randint(0, N_KV, (B, S_q, TOPK), dtype=torch.int32, device=device)
invalid_mask = torch.zeros(B * S_q, TOPK, dtype=torch.bool, device=device)
sm_scale = 1.0 / (D ** 0.5)

from sglang.srt.layers.attention.ck_v32_sparse_mla import ck_sparse_mla_decode_fp8_v32

def run_ck(k_view):
    """Run CK kernel with k_cache stored as uint8 (the kernel reads bytes)."""
    out, lse = ck_sparse_mla_decode_fp8_v32(
        q=q, k_cache=k_view, indices=indices, invalid_mask=invalid_mask,
        attn_sink=None, sm_scale=sm_scale,
    )
    return out, lse


def torch_ref(k_cache_bf):
    """Reference: k_cache as bf16, sparse gather + dense attention."""
    B_, S_q_, _, _ = q.shape
    out = torch.zeros(B_, S_q_, H, V, dtype=torch.float32, device=device)
    lse_out = torch.zeros(B_, H, S_q_, dtype=torch.float32, device=device)
    for b in range(B_):
        idx = indices[b, 0]  # [TOPK]
        k_gathered = k_cache_bf[idx, :D].float()  # [TOPK, D]
        v_gathered = k_cache_bf[idx, :V].float()  # [TOPK, V] — note V=D here
        for h in range(H):
            q_vec = q[b, 0, h, :].float()  # [D]
            scores = (q_vec @ k_gathered.T) * sm_scale  # [TOPK]
            lse = torch.logsumexp(scores, dim=0)  # scalar
            attn = torch.softmax(scores, dim=0)
            out[b, 0, h, :] = attn @ v_gathered
            lse_out[b, h, 0] = lse
    return out.to(torch.bfloat16), lse_out


print("\n=== Computing torch reference (bf16 storage) ===")
ref_out, ref_lse = torch_ref(k_cache_bf16)
print(f"  ref_out[0,0,0,:5]={ref_out[0,0,0,:5].float().tolist()}")
print(f"  ref_lse[0,0,0]={ref_lse[0,0,0].item():.4f}")

print("\n=== CK V32 (512) on FN-quantized KV ===")
ck_out_fn, ck_lse_fn = run_ck(k_cache_fn)
diff_fn = (ck_out_fn.float() - ref_out.float())
print(f"  ck_out[0,0,0,:5]={ck_out_fn[0,0,0,:5].float().tolist()}")
print(f"  ck_lse[0,0,0]={ck_lse_fn[0,0,0].item():.4f}")
print(f"  max_diff={diff_fn.abs().max().item():.4e}")
print(f"  mean_diff={diff_fn.abs().mean().item():.4e}")
ratio_fn = (ck_out_fn.float() / (ref_out.float() + 1e-9))
print(f"  ratio (ck/ref) median={ratio_fn.median().item():.4f}, mean={ratio_fn.mean().item():.4f}")
cs = (ck_out_fn.float().flatten() @ ref_out.float().flatten()) / (ck_out_fn.float().norm() * ref_out.float().norm() + 1e-9)
print(f"  cos_sim={cs.item():+.4f}")

if FNUZ is not FN:
    print("\n=== CK V32 (512) on FNUZ-quantized KV ===")
    ck_out_fnuz, ck_lse_fnuz = run_ck(k_cache_fnuz)
    diff_fnuz = (ck_out_fnuz.float() - ref_out.float())
    print(f"  max_diff={diff_fnuz.abs().max().item():.4e}")
    cs2 = (ck_out_fnuz.float().flatten() @ ref_out.float().flatten()) / (ck_out_fnuz.float().norm() * ref_out.float().norm() + 1e-9)
    print(f"  cos_sim={cs2.item():+.4f}")

# Test if ck_out is just scaled relative to ref
print("\n=== Check if CK is uniform-scaled vs reference ===")
median_ratio = (ck_out_fn.float() / (ref_out.float().abs() + 1e-9)).abs().median()
print(f"  abs median ratio = {median_ratio.item():.4f}")
print(f"  if ratio==2.0 → CK output is 2x too big (kernel forgot 0.5 fold or applied twice)")
print(f"  if ratio==0.5 → CK output is 0.5x (extra 0.5 fold for fn storage)")
print(f"  if ratio==1.0 → CK is correct, divergence is non-uniform")
