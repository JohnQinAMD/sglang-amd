"""Quick launch-count comparator: torch_op vs triton_op, 100 iterations each."""
import os, sys
sys.path.insert(0, "/sgl-pr/python")
sys.path.insert(0, os.path.dirname(__file__))

import torch
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dbm_block_a",
    "/sgl-pr/python/sglang/jit_kernel/decode_body_mqa_prologue_triton.py",
)
dbm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dbm)

torch.manual_seed(0)
device = "cuda:0"
FP8 = torch.float8_e4m3fn
HIDDEN, Q_LORA_RANK, HEAD_DIM, GROUP_SIZE, EPS = 8192, 1536, 64, 128, 1e-6
bs = 6

hidden = torch.randn(bs, HIDDEN, device=device, dtype=torch.bfloat16) * 0.1
rmsnorm_w = torch.randn(HIDDEN, device=device, dtype=torch.float32)
wq_a = (torch.randn(Q_LORA_RANK, HIDDEN, device=device, dtype=torch.bfloat16) * 0.1).to(FP8)
wkv_a = (torch.randn(HEAD_DIM, HIDDEN, device=device, dtype=torch.bfloat16) * 0.1).to(FP8)
n_g_q = (Q_LORA_RANK + GROUP_SIZE - 1) // GROUP_SIZE
n_g_k = (HEAD_DIM + GROUP_SIZE - 1) // GROUP_SIZE
n_g_h = (HIDDEN + GROUP_SIZE - 1) // GROUP_SIZE
wq_a_scale = torch.rand(n_g_q, n_g_h, device=device, dtype=torch.float32) + 0.5
wkv_a_scale = torch.rand(max(1, n_g_k), n_g_h, device=device, dtype=torch.float32) + 0.5
qlo = torch.empty(bs, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
kvo = torch.empty(bs, HEAD_DIM, dtype=torch.bfloat16, device=device)


def torch_op():
    h = hidden
    var = h.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + EPS)
    normed = (h.to(torch.float32) * rstd * rmsnorm_w).to(torch.bfloat16)
    n_groups = HIDDEN // GROUP_SIZE
    normed_g = normed.view(bs, n_groups, GROUP_SIZE).to(torch.float32)
    scales = normed_g.abs().amax(dim=-1) / 448.0
    fp8_in = (normed_g / scales.unsqueeze(-1)).to(FP8).view(bs, HIDDEN)
    fp8_w = wq_a.to(torch.bfloat16)
    qlo.copy_((fp8_in.to(torch.float32) @ fp8_w.t().to(torch.float32)).to(torch.bfloat16))
    fp8_kv = wkv_a.to(torch.bfloat16)
    kvo.copy_((fp8_in.to(torch.float32) @ fp8_kv.t().to(torch.float32)).to(torch.bfloat16))


def triton_op():
    dbm.mqa_prologue_megakernel(
        hidden, rmsnorm_w, wq_a, wq_a_scale, wkv_a, wkv_a_scale, eps=EPS,
        q_lora_out=qlo, kv_out=kvo,
    )


# Warmup
for _ in range(20):
    torch_op()
    triton_op()
torch.cuda.synchronize()

mode = sys.argv[1] if len(sys.argv) > 1 else "torch"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
op = torch_op if mode == "torch" else triton_op
print(f"running {ITERS} iters of {mode}_op", flush=True)
for _ in range(ITERS):
    op()
torch.cuda.synchronize()
print("done", flush=True)
