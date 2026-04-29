"""Direct in-kernel score dump for the smallest failing config.

Build the CK V32 kernel with -DSGLANG_CK_V32_DEBUG_DUMP, call it once with
B=1, S_q=1, H=64, topk=128, valid indices ONLY at X=0 (kv_idx=128) and
X=1 (kv_idx=129) — the production-failure shape from the drill-down. The
kernel will printf its post-scale + post-mask s_acc[nt][c] for every
(lane, c, nt) it computes; we then compare those against the oracle's
q[h] · k_pool[idx[X]] * sm_scale * log2e for X in {0, 1}.

If kernel-printed scores match oracle: bug is in softmax/argmax pipeline.
If kernel scores ≠ oracle: it's the operand input layout (kb/qa load).

Usage:
  PYTHONPATH=/sgl-pr/python python3 microbench/microbench_ck_v32_score_dump.py
"""
import math
import os
import torch

# Force a kernel rebuild with debug-dump on.
os.environ["SGLANG_CK_V32_KERNEL_SRC_DIR"] = (
    "/sgl-pr/python/sglang/srt/layers/attention/csrc/ck_v32"
)
os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/ck_v32_dbg_ext"
# Add -DSGLANG_CK_V32_DEBUG_DUMP. We can't pass it via the existing wrapper's
# load() call which hardcodes flags, so we'll patch the wrapper's _get_ck_mod
# at import time. Simplest: pass as an env-prefixed compile flag via the build
# system. We bypass the cached wrapper and build a debug variant here.
os.environ.setdefault("SGLANG_HIP_SPARSE_MLA_DECODE_FP8", "1")

from torch.utils.cpp_extension import load
src_dir = os.environ["SGLANG_CK_V32_KERNEL_SRC_DIR"]
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import shutil
if os.path.isdir("/tmp/ck_v32_dbg_ext"):
    shutil.rmtree("/tmp/ck_v32_dbg_ext")
os.makedirs("/tmp/ck_v32_dbg_ext", exist_ok=True)

ck = load(
    name="ck_mla_decode_sparse_fp8_DBG",
    sources=[os.path.join(src_dir, "mla_decode_fwd.cu")],
    extra_include_paths=[src_dir],
    extra_cuda_cflags=[
        "-O3", "-std=c++20", "-DSGLANG_CK_V32_DEBUG_DUMP=1",
    ],
    build_directory="/tmp/ck_v32_dbg_ext",
    verbose=True,
)

# ───── Setup: smallest failing config ─────
B, S_q, H, D, V = 1, 1, 64, 512, 512
SLOT = 584
N_KV = 256
TOPK = 128
device = "cuda"

torch.manual_seed(42)
q = (torch.randn(B, S_q, H, D, dtype=torch.bfloat16, device=device) * 0.5)
# Build a KV pool with non-trivial values at idx=128 and idx=129.
kv = torch.zeros(N_KV, SLOT, dtype=torch.bfloat16, device=device)
kv[128, :D] = torch.randn(D, dtype=torch.bfloat16, device=device) * 30.0
kv[129, :D] = torch.randn(D, dtype=torch.bfloat16, device=device) * 30.0
kv_fp8 = kv.to(torch.float8_e4m3fn)

# Indices: valid only at X=0 (kv=128) and X=1 (kv=129); rest are -1.
indices = torch.full((B, S_q, TOPK), -1, dtype=torch.int32, device=device)
indices[0, 0, 0] = 128
indices[0, 0, 1] = 129

# Oracle expectation: per-head score for X=0 and X=1.
sm_scale = 1.0 / math.sqrt(D)
log2e = 1.0 / math.log(2.0)
sm_scale_log2e = sm_scale * log2e

q_h = q[0, 0, :, :].float()  # [H, D]
k0 = kv_fp8[128, :D].float()  # post-fn-decode
k1 = kv_fp8[129, :D].float()
score_X0 = (q_h @ k0).cpu()  # [H]
score_X1 = (q_h @ k1).cpu()
score_X0_scaled = score_X0 * sm_scale_log2e
score_X1_scaled = score_X1 * sm_scale_log2e

print("=== Oracle expected post-scale post-mask scores ===")
print(f"  Head 0..7  X=0 (kv_idx=128): {[f'{s:+.4f}' for s in score_X0_scaled[:8].tolist()]}")
print(f"  Head 0..7  X=1 (kv_idx=129): {[f'{s:+.4f}' for s in score_X1_scaled[:8].tolist()]}")
print()
print(f"For (head=0, kv_col=0): expect kernel s_acc ≈ {score_X0_scaled[0].item():+.4f}")
print(f"For (head=0, kv_col=1): expect kernel s_acc ≈ {score_X1_scaled[0].item():+.4f}")
print(f"For (head=0, kv_col=2..15): expect kernel s_acc = -1e30 (masked, pidx<-1)")
print()

# Set up split tensors and call the kernel directly.
num_splits = 1  # match microbench's pick_num_splits for B=1, topk=128 → 4
# Actually we want to keep it simple — single split.
split_data = torch.empty((B*S_q, num_splits, H, V), dtype=torch.float32, device=device)
split_lse = torch.empty((B*S_q, num_splits, H, 1), dtype=torch.float32, device=device)
qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
kv_indptr = torch.tensor([0, TOPK], dtype=torch.int32, device=device)
idx_flat = indices.reshape(-1).to(torch.int32)
q_2d = q.view(B*S_q, H, D).contiguous()

print("=== Calling kernel (debug printf will fire) ===")
ck.mla_decode_fwd_ck_sparse_fp8(
    q_2d, kv_fp8, split_data, split_lse,
    qo_indptr, kv_indptr, idx_flat,
    float(sm_scale), int(num_splits), 1.0,  # fp8_decode_scale=1.0 for fn
)
torch.cuda.synchronize()
print("=== Kernel call complete ===")
print()
print("Inspect [CK_DBG] lines above:")
print("  - For (lane=0, c=0, nt=0): expect head=0, kv_col=0, s_acc ≈", f"{score_X0_scaled[0].item():+.4f}")
print("  - For (lane=1, c=0, nt=0): expect head=0, kv_col=1, s_acc ≈", f"{score_X1_scaled[0].item():+.4f}")
print("  - For (lane=2..15, c=0, nt=0): expect head=0, kv_col=2..15, s_acc = -1e30")
print()
print("If kernel s_acc at (lane=0, c=0, nt=0) ≠ oracle's score for (head=0, kv_col=0):")
print("  → A/B operand layout bug — kernel is pairing wrong q with wrong k.")
print("If kernel s_acc matches oracle at every (lane, c, nt):")
print("  → bug is in softmax/argmax/PV (downstream).")
