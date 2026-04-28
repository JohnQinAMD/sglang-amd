"""Replay actual production-captured tensors through the wrapper +
compare to per-(b,h) FP32 oracle. The kernel/wrapper unit tests with
synthetic data ALL pass; this checks the production captures specifically.
"""
import os, sys, math
import torch
import glob

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("SGLANG_HIP_SPARSE_MLA_DECODE_FP8", "1")

from sglang.srt.layers.attention.ck_v32_sparse_mla import (
    ck_sparse_mla_decode_fp8_v32,
)

device = "cuda"
DUMP_DIR = "/sgl-pr/_ck_v32_dumps_flash_mxfp4"
paths = sorted(glob.glob(os.path.join(DUMP_DIR, "ck_v32_call_*single_shot.pt")))
if not paths:
    print(f"No dumps in {DUMP_DIR}")
    sys.exit(1)


def cos(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    n = (af.norm() * bf.norm()).item()
    return float(af @ bf) / n if n > 1e-12 else float("nan")


def per_batch_oracle(q, kv, indices, attn_sink, sm_scale):
    """[B, S_q, H, V] expected output via FP32 dense attention with -inf masking.

    CRITICAL: production K cache is torch.uint8 (raw bytes). Must view as
    torch.float8_e4m3fn before .float() to get the actual FP8-dequantized
    values. Without this view, .float() on uint8 just returns the byte
    values (0..255) and the oracle silently disagrees with the kernel
    (which correctly does cvt_pk_f32_fp8 with fn semantics).
    """
    B, S_q, H, D = q.shape
    V = 512
    if kv.dim() == 4:
        npg, ps, _, slot = kv.shape
        kv2d = kv.reshape(npg * ps, slot)
    elif kv.dim() == 3:
        kv2d = kv.reshape(-1, kv.shape[-1])
    else:
        kv2d = kv
    # FIX: ensure FP8 fn dequant when kv is uint8.
    if kv2d.dtype == torch.uint8:
        kv2d = kv2d.view(torch.float8_e4m3fn)
    k_full = kv2d.float()
    out = torch.zeros(B, S_q, H, V, dtype=torch.float32, device=device)
    lse_out = torch.full((B, H, S_q), float("-inf"),
                         dtype=torch.float32, device=device)
    for b in range(B):
        for s in range(S_q):
            idx_raw = indices[b, s].long()
            invalid = idx_raw < 0
            if invalid.all():
                continue
            idx_safe = torch.clamp(idx_raw, min=0)
            k_g = k_full[idx_safe, :D]
            v_g = k_full[idx_safe, :V]
            q_h = q[b, s, :, :].float()
            scores = (q_h @ k_g.T) * sm_scale
            scores = torch.where(invalid.view(1, -1),
                                 torch.tensor(float("-inf"), device=device),
                                 scores)
            lse_out[b, :, s] = torch.logsumexp(scores, dim=-1)
            attn = torch.softmax(scores, dim=-1)
            out[b, s, :, :] = attn @ v_g
    if attn_sink is not None:
        sc = 1.0 / (1.0 + torch.exp(
            attn_sink.view(1, 1, H, 1) - lse_out.transpose(1, 2).unsqueeze(-1)))
        sc = torch.where(torch.isfinite(sc), sc, torch.zeros_like(sc))
        out = out * sc
    return torch.nan_to_num(out, nan=0.0).to(torch.bfloat16)


print(f"Replaying {len(paths)} production captures through ck_sparse_mla_decode_fp8_v32...")
print()
print(f"{'name':<55s} {'B':>3s} {'valid%':>6s} {'cos':>9s} {'maxd':>10s}  V")

for p in paths:
    payload = torch.load(p, map_location=device, weights_only=False)
    q_t = payload["q"].to(device)
    kv_cpu = payload["k_cache"]
    dtype_str = payload["k_cache_dtype"]
    # Production saves kv as torch.uint8. Pass through as-is (the wrapper
    # accepts uint8 and the kernel does cvt_pk_f32_fp8 with fn semantics).
    kv_t = kv_cpu.to(device)
    indices_t = payload["indices"].to(device).to(torch.int32)
    sink_t = payload["attn_sink"]
    if sink_t is not None:
        sink_t = sink_t.to(device).float()
    sm = float(payload["sm_scale"])
    B, S_q, H, D = q_t.shape
    valid_pct = (indices_t >= 0).float().mean().item() * 100

    # Skip warmup placeholder captures (q all zeros).
    if q_t.float().abs().max().item() == 0:
        continue

    invalid_mask = (indices_t.view(B*S_q, -1) < 0)
    out, lse = ck_sparse_mla_decode_fp8_v32(
        q=q_t.contiguous(), k_cache=kv_t, indices=indices_t,
        invalid_mask=invalid_mask, attn_sink=sink_t, sm_scale=sm,
    )
    torch.cuda.synchronize()

    oracle = per_batch_oracle(q_t, kv_t, indices_t, sink_t, sm)
    cs = cos(out, oracle)
    mxd = (out.float() - oracle.float()).abs().max().item()
    verdict = "PASS" if cs > 0.999 else "FAIL"
    print(f"{os.path.basename(p):<55s} {B:>3d} {valid_pct:>5.1f}% "
          f"{cs:>+.5f} {mxd:>10.3e}  {verdict}")
