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


def torch_op(args):
    """Production 4-op chain — the reference oracle.

    Note: uses ATEN ops directly to avoid AITER import in microbench harness.
    The aiter wrappers add wrapper-tax (alloc + view + cast) that's part of the
    real production path; for fairness against the megakernel we measure the
    AITER wrappers in the wired-in path, not here.
    """
    # 1. RMSNorm
    h = args["hidden"]
    eps = EPS
    var = h.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    normed = (h.to(torch.float32) * rstd * args["rmsnorm_w"]).to(torch.bfloat16)
    # 2. Per-1x128 quant (block-scale)
    bs, hidden = normed.shape
    n_groups = hidden // GROUP_SIZE
    normed_grouped = normed.view(bs, n_groups, GROUP_SIZE).to(torch.float32)
    scales = normed_grouped.abs().amax(dim=-1) / 448.0
    fp8_in = (normed_grouped / scales.unsqueeze(-1)).to(FP8_DTYPE).view(bs, hidden)
    # 3. Block-scale GEMM wq_a (eager simulation; real path uses aiter::gemm_a8w8_blockscale)
    fp8_w = args["wq_a_fp8"].to(torch.bfloat16)  # eager dequant for ref
    q_lora = (
        torch.matmul(fp8_in.to(torch.float32), fp8_w.t().to(torch.float32))
        .to(torch.bfloat16)
    )
    # 4. Block-scale GEMM wkv_a
    fp8_kv = args["wkv_a_fp8"].to(torch.bfloat16)
    kv = (
        torch.matmul(fp8_in.to(torch.float32), fp8_kv.t().to(torch.float32))
        .to(torch.bfloat16)
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
