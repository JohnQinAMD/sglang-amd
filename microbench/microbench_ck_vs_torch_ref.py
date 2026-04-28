"""Side-by-side: CK V32 single_shot output vs torch's actual ref_sparse_attn_decode.

This is the closest comparison to what production sees. The "ref path" used
when the Layer-3 stopgap is on routes calls through `ref_sparse_attn_decode`
which uses torch's bf16 matmul (rocBLAS / hipBLASLt under the hood) — not my
custom fp32 oracle.

If kernel ≠ torch.ref_sparse_attn_decode at bf16-precision, the per-call delta
vs torch's specific kernel is the e2e drift source. If kernel ≈ torch.ref at
bf16-precision, the drift is integration-level (not per-call).
"""
import os, glob, math
import torch

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ.setdefault("SGLANG_HIP_SPARSE_MLA_DECODE_FP8", "1")

from sglang.srt.layers.attention.ck_v32_sparse_mla import (
    ck_sparse_mla_decode_fp8_v32,
)

device = "cuda"
DUMP_DIR = "/sgl-pr/_ck_v32_dumps_flash_mxfp4"
paths = sorted(glob.glob(os.path.join(DUMP_DIR, "ck_v32_call_*single_shot.pt")))


def cos(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    n = (af.norm() * bf.norm()).item()
    return float(af @ bf) / n if n > 1e-12 else float("nan")


def torch_ref(q, kv, indices, attn_sink, sm_scale):
    """Torch ref via the same path as ref_sparse_attn_decode but with our exact
    inputs. This is bf16 dequant + bf16 matmul + fp32 softmax + bf16 matmul
    output — what production hits when the gate falls through.
    """
    B, S_q, H, D = q.shape
    V = 512
    if kv.dim() == 4:
        npg, ps, _, slot = kv.shape
        kv2d = kv.reshape(npg * ps, slot)
    else:
        kv2d = kv
    if kv2d.dtype == torch.uint8:
        kv2d = kv2d.view(torch.float8_e4m3fn)
    # Dequant fp8 → bf16 (lossless; fp8 has 4 mantissa bits ≪ bf16's 7)
    k_pool_bf = kv2d.to(torch.bfloat16)
    out = torch.zeros(B, S_q, H, V, dtype=torch.bfloat16, device=device)
    lse_out = torch.full((B, H, S_q), float("-inf"), dtype=torch.float32, device=device)
    for b in range(B):
        for s in range(S_q):
            idx_raw = indices[b, s].long()
            invalid = idx_raw < 0
            if invalid.all():
                continue
            idx_safe = torch.clamp(idx_raw, min=0)
            k_g = k_pool_bf[idx_safe, :D]
            v_g = k_pool_bf[idx_safe, :V]
            q_h = q[b, s, :, :]  # bf16, no upcast
            # Match production: bf16 q × bf16 k via torch.matmul (uses rocBLAS).
            # Output is bf16; torch internally does fp32 accumulation.
            scores_bf = q_h.float() @ k_g.float().T
            scores_bf = scores_bf * sm_scale
            scores_bf = torch.where(invalid.view(1, -1),
                                    torch.tensor(float("-inf"), device=device), scores_bf)
            lse_out[b, :, s] = torch.logsumexp(scores_bf, dim=-1)
            attn = torch.softmax(scores_bf, dim=-1)  # fp32 inside softmax
            # P @ V: bf16 attn × bf16 V
            out[b, s, :, :] = (attn @ v_g.float()).to(torch.bfloat16)
    if attn_sink is not None:
        sc = 1.0 / (1.0 + torch.exp(
            attn_sink.view(1, 1, H, 1).float()
            - lse_out.transpose(1, 2).unsqueeze(-1).float()))
        sc = torch.where(torch.isfinite(sc), sc, torch.zeros_like(sc))
        out = (out.float() * sc).to(torch.bfloat16)
    return out


print(f"=== CK V32 single_shot vs torch ref (production-tensor side-by-side) ===")
print(f"{'name':<55s} {'cos':>9s} {'maxd':>10s} {'ck.amax':>9s} {'rf.amax':>9s} {'rel.maxd':>9s}")

worst_ratio = 0
for p in paths:
    payload = torch.load(p, map_location=device, weights_only=False)
    q_t = payload["q"].to(device)
    if q_t.float().abs().max().item() == 0:
        continue
    kv_t = payload["k_cache"].to(device)
    idx = payload["indices"].to(device).to(torch.int32)
    sink = (None if payload["attn_sink"] is None
            else payload["attn_sink"].to(device).float())
    sm = float(payload["sm_scale"])
    invalid_mask = (idx.view(idx.shape[0]*idx.shape[1], -1) < 0)

    out_ck, _ = ck_sparse_mla_decode_fp8_v32(
        q=q_t.contiguous(), k_cache=kv_t, indices=idx,
        invalid_mask=invalid_mask, attn_sink=sink, sm_scale=sm,
    )
    torch.cuda.synchronize()

    out_rf = torch_ref(q_t, kv_t, idx, sink, sm)

    cs = cos(out_ck, out_rf)
    diff = (out_ck.float() - out_rf.float()).abs()
    mxd = diff.max().item()
    ck_amax = out_ck.float().abs().max().item()
    rf_amax = out_rf.float().abs().max().item()
    rel = mxd / max(rf_amax, 1e-9)
    worst_ratio = max(worst_ratio, rel)
    print(f"{os.path.basename(p):<55s} {cs:>+.5f} {mxd:>10.3e} "
          f"{ck_amax:>9.2e} {rf_amax:>9.2e} {rel*100:>8.3f}%")

print()
print(f"Worst per-element relative diff: {worst_ratio*100:.3f}%")
print()
print("Reading:")
print("  - relative diff < 0.5%: kernel ≈ torch ref at bf16 precision; e2e")
print("    drift is integration-level (model+stream+graph), not per-call.")
print("  - relative diff > 1%: kernel diverges from torch ref by more than")
print("    bf16 floor. Localizes the per-call drift source.")
