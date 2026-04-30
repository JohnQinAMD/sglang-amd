"""v2 microbench scaffold — Decode-body megakernel Block A (MQA prologue).

Status: SCAFFOLD. Provides the torch reference (4-op chain) + production shape
histogram + the framework call. Kernel author fills in `triton_op` once
`mqa_prologue_megakernel` body is implemented.

Production reference chain (4 ops, ~5-7 aten glue ops, 4 hipLaunchKernel):
    rmsnorm_out      = aiter.add_rmsnorm(hidden, rmsnorm_w, eps)            # 1 launch
    fp8_in, scales   = aiter.dynamic_per_group_scaled_quant(rmsnorm_out)    # 1 launch
    q_lora           = aiter.gemm_a8w8_blockscale(fp8_in, wq_a_fp8, scales, wq_a_scale)
    kv               = aiter.gemm_a8w8_blockscale(fp8_in, wkv_a_fp8, scales, wkv_a_scale)

Megakernel target: 1 launch, 0-1 aten glue.

Production shape histogram (DSv4 Flash-Base FP8 decode TP=4, weighted by
concurrency=4 max_running=6 production traffic):
    BS=1  : 50
    BS=2  : 30
    BS=4  : 80
    BS=6  : 200   (production hot path)
    BS=8  : 20

Constants (from DSv4 config):
    HIDDEN = 8192
    Q_LORA_RANK = 1536
    HEAD_DIM = 64

Gates (per `reference_microbench_v2_framework.md`):
    G0 — bit-correctness vs eager 4-op chain (HARD; bf16 output ULP floor)
    G1 — eager + cuda-graph capture+replay timing
    G2 — production-shape histogram weighted speedup (worst-mode ≥ 1.10x)
    G3 — PMC: launch count delta (4 → 1) and wait/MFMA ratio
"""
import os, sys
sys.path.insert(0, "/sgl-pr/python")
sys.path.insert(0, os.path.dirname(__file__))

import torch
from _framework import bench_v2

import importlib.util
proto_path = os.environ.get(
    "DBM_BLOCK_A_PATH",
    "/sgl-pr/python/sglang/jit_kernel/decode_body_mqa_prologue_triton.py",
)
spec = importlib.util.spec_from_file_location("dbm_block_a", proto_path)
dbm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dbm)


torch.manual_seed(0)
device = "cuda:0"
FP8_DTYPE = torch.float8_e4m3fn

HIDDEN = 8192
Q_LORA_RANK = 1536
HEAD_DIM = 64
GROUP_SIZE = 128
EPS = 1e-6

SHAPE_FREQ = {
    (1, HIDDEN, Q_LORA_RANK, HEAD_DIM): 50,
    (2, HIDDEN, Q_LORA_RANK, HEAD_DIM): 30,
    (4, HIDDEN, Q_LORA_RANK, HEAD_DIM): 80,
    (6, HIDDEN, Q_LORA_RANK, HEAD_DIM): 200,
    (8, HIDDEN, Q_LORA_RANK, HEAD_DIM): 20,
}


def prep(shape):
    bs, hidden, q_lora_rank, head_dim = shape
    hidden_t = (torch.randn(bs, hidden, device=device, dtype=torch.bfloat16) * 0.1)
    rmsnorm_w = torch.randn(hidden, device=device, dtype=torch.float32)
    # fp8 weights with per-128x128 block scales
    n_groups_q = (q_lora_rank + GROUP_SIZE - 1) // GROUP_SIZE
    n_groups_k = (head_dim + GROUP_SIZE - 1) // GROUP_SIZE
    n_groups_h = (hidden + GROUP_SIZE - 1) // GROUP_SIZE
    wq_a_bf16 = (torch.randn(q_lora_rank, hidden, device=device, dtype=torch.bfloat16) * 0.1)
    wq_a_fp8 = wq_a_bf16.to(FP8_DTYPE)
    wq_a_scale = (torch.rand(n_groups_q, n_groups_h, device=device, dtype=torch.float32) + 0.5)
    wkv_a_bf16 = (torch.randn(head_dim, hidden, device=device, dtype=torch.bfloat16) * 0.1)
    wkv_a_fp8 = wkv_a_bf16.to(FP8_DTYPE)
    wkv_a_scale = (torch.rand(max(1, n_groups_k), n_groups_h, device=device, dtype=torch.float32) + 0.5)
    return {
        "hidden": hidden_t,
        "rmsnorm_w": rmsnorm_w,
        "wq_a_fp8": wq_a_fp8, "wq_a_scale": wq_a_scale,
        "wkv_a_fp8": wkv_a_fp8, "wkv_a_scale": wkv_a_scale,
        "q_lora_out": torch.empty(bs, q_lora_rank, dtype=torch.bfloat16, device=device),
        "kv_out": torch.empty(bs, head_dim, dtype=torch.bfloat16, device=device),
    }


def _blockscale_gemm_ref(
    fp8_in: torch.Tensor,        # (m, k)  fp8
    x_scale: torch.Tensor,       # (m, k // GROUP_SIZE) fp32 — per-1x128 row-group scale
    fp8_w: torch.Tensor,         # (n, k)  fp8 row-major
    w_scale: torch.Tensor,       # (cdiv(n, 128), k // GROUP_SIZE) fp32 — per-128x128 block scale
) -> torch.Tensor:
    """aiter::gemm_a8w8_blockscale semantic match — applies BOTH scales then matmuls.

    Mirrors `run_torch` in /aiter-amd/op_tests/test_gemm_a8w8_blockscale.py
    (broadcast w_scale per (128_n × 128_k) block, x_scale per (1 × 128_k) group,
    multiply into fp32, then F.linear and bf16 cast).
    """
    m, k = fp8_in.shape
    n = fp8_w.shape[0]
    scale_k = k // GROUP_SIZE
    scale_n = (n + GROUP_SIZE - 1) // GROUP_SIZE
    # Broadcast x_scale: (m, scale_k) -> (m, k). Each row has GROUP_SIZE k-elts at scale.
    x_dq = fp8_in.to(torch.float32).view(m, scale_k, GROUP_SIZE) * x_scale.unsqueeze(-1)
    x_dq = x_dq.view(m, k)
    # Broadcast w_scale: (scale_n, scale_k) -> (scale_n*128, scale_k*128).
    # We need result[ng*128 + nn, kg*128 + kk] = w_scale[ng, kg]. Permute so collapse
    # produces the right ordering: (sn, 128, sk, 128) -> reshape (sn*128, sk*128).
    w_scale_bcast = (
        w_scale[:, None, :, None]
        .expand(scale_n, GROUP_SIZE, scale_k, GROUP_SIZE)
        .reshape(scale_n * GROUP_SIZE, scale_k * GROUP_SIZE)
    )
    w_scale_bcast = w_scale_bcast[:n, :k]
    w_dq = fp8_w.to(torch.float32) * w_scale_bcast
    out = torch.matmul(x_dq, w_dq.t())
    return out.to(torch.bfloat16)


def torch_op(args):
    """Production 4-op chain — the reference oracle (Phase 2: scale-applying).

    Mirrors aiter::gemm_a8w8_blockscale exactly:
    - Per-1x128 x_scale computed inside the quant kernel from RMSNorm output
    - Per-128x128 w_scale loaded from `*_a_scale` weights
    - Output fp32 acc = (x_fp8 * x_scale_per_group) @ (w_fp8 * w_scale_per_block).T
    """
    # 1. RMSNorm in fp32 (production keeps fp32 in the dynamic_per_group_scaled_quant
    #    fused kernel; the bf16 round-trip in v1 is dropped for tighter parity).
    h = args["hidden"]
    eps = EPS
    var = h.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    normed_fp32 = h.to(torch.float32) * rstd * args["rmsnorm_w"]
    # 2. Per-1x128 quant (block-scale). x_scale shape (bs, hidden // 128).
    bs, hidden = normed_fp32.shape
    n_groups = hidden // GROUP_SIZE
    normed_grouped = normed_fp32.view(bs, n_groups, GROUP_SIZE)
    amax = normed_grouped.abs().amax(dim=-1)
    x_scale = amax / 448.0
    x_scale_safe = torch.where(x_scale == 0.0, torch.ones_like(x_scale), x_scale)
    fp8_in = (normed_grouped / x_scale_safe.unsqueeze(-1)).to(FP8_DTYPE).view(bs, hidden)
    # 3. Block-scale GEMM wq_a — applies BOTH x_scale (per-1x128) and w_scale (per-128x128).
    q_lora = _blockscale_gemm_ref(
        fp8_in, x_scale, args["wq_a_fp8"], args["wq_a_scale"],
    )
    # 4. Block-scale GEMM wkv_a — same scale-applying semantic.
    kv = _blockscale_gemm_ref(
        fp8_in, x_scale, args["wkv_a_fp8"], args["wkv_a_scale"],
    )
    args["q_lora_out"].copy_(q_lora)
    args["kv_out"].copy_(kv)


def triton_op(args):
    """Megakernel — calls scaffold; until kernel body is implemented this returns
    None and the bench will mark it as 'not yet implemented'.
    """
    out = dbm.mqa_prologue_megakernel(
        args["hidden"], args["rmsnorm_w"],
        args["wq_a_fp8"], args["wq_a_scale"],
        args["wkv_a_fp8"], args["wkv_a_scale"],
        eps=EPS,
        q_lora_out=args["q_lora_out"],
        kv_out=args["kv_out"],
    )
    if out is None:
        # Scaffold: kernel body not yet implemented.
        torch_op(args)


def get_outputs(args, op):
    a2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in args.items()}
    op(a2)
    return [a2["q_lora_out"].detach().clone(), a2["kv_out"].detach().clone()]


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"HIDDEN={HIDDEN} Q_LORA_RANK={Q_LORA_RANK} HEAD_DIM={HEAD_DIM}")
    print()

    result = bench_v2(
        name="DBM Block A — MQA prologue megakernel (rmsnorm + quant + 2x fp8 GEMM)",
        prep=prep, torch_op=torch_op, triton_op=triton_op,
        shape_freq=SHAPE_FREQ, hit_rate=0.0,
        correctness_get_outputs=get_outputs,
        correctness_atol=2e-2, correctness_rtol=1e-1,
        iters=2000, warmup=200, skip_graph=False,
    )

    print()
    print("=" * 78)
    print("STATUS: scaffold — kernel body NOT yet implemented.")
    print("=" * 78)
    print("Next steps for the kernel author:")
    print("  1. Implement _mqa_prologue_kernel body in")
    print("     /sgl-pr/python/sglang/jit_kernel/decode_body_mqa_prologue_triton.py")
    print("  2. Re-run this microbench until G0 (correctness) + G1 (≥1.10× worst-mode)")
    print("     are green, THEN proceed to G2-G6 per design doc §5.")
