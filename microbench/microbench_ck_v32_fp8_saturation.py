"""Microbench: CK V32 sparse MLA at FP8-saturation KV magnitudes.

`microbench_ck_v32_512.py` uses `randn() * 0.1` so kv.abs.max ≈ 3 — never
reaches the FP8 e4m3 saturation regime that production hits. The Flash mxfp4
e2e bug (garbage tokens) reproduces only when kv.abs.max is at the FP8
saturation max (~255 for fn). This file extends the sweep to that regime so
the bug reproduces in isolation.

Sweeps:
  kv_scale         ∈ {0.1, 1.0, 30.0, 80.0}    (1.0 ≈ raw randn; 80 → satur.)
  kv_clamp         ∈ {None, ±240}              (clamp to near FP8-fn max)
  B                ∈ {1, 2, 6, 16, 64}         (2 PASS / 6+ FAIL in production)
  TOPK             ∈ {64, 128, 256}            (production hits 128)
  invalid_frac     ∈ {0.0, 0.95}               (production has up to 99% -1)

PASS = cos_sim ≥ 0.999 vs FP32 oracle AND output is NaN-free.

Run: PYTHONPATH=/sgl-pr/python python3 microbench/microbench_ck_v32_fp8_saturation.py
"""
import os
import torch

os.environ["SGLANG_HIP_SPARSE_MLA_DECODE_FP8"] = "1"
os.environ.setdefault("SGLANG_CK_V32_DUMP_DIR", "")  # disable hook

S_q, H, V = 1, 64, 512  # Flash mxfp4 production heads-per-rank
D = int(os.environ.get("D", "512"))
N_KV = int(os.environ.get("N_KV", "8192"))
SLOT = 584

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
device = "cuda"
FN = torch.float8_e4m3fn


def cos_sim(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    n = (af.norm() * bf.norm()).item()
    if n < 1e-12:
        return float("nan")
    return float(af @ bf) / n


def make_kv(scale, clamp_max):
    bf = torch.randn(N_KV, SLOT, dtype=torch.bfloat16, device=device) * scale
    if clamp_max is not None:
        bf = torch.clamp(bf, -clamp_max, clamp_max)
    return bf.to(FN)


def make_indices(B, topk, invalid_frac):
    raw = torch.randint(0, N_KV, (B, S_q, topk), dtype=torch.int32, device=device)
    if invalid_frac > 0:
        n_invalid = int(round(topk * invalid_frac))
        if n_invalid > 0:
            raw[..., -n_invalid:] = -1
    return raw


def torch_ref_fn(q, kv_fn, indices, attn_sink, sm_scale):
    B = q.shape[0]
    k_full = kv_fn.float()
    out = torch.zeros(B, S_q, H, V, dtype=torch.float32, device=device)
    lse_out = torch.zeros(B, H, S_q, dtype=torch.float32, device=device)
    for b in range(B):
        idx_raw = indices[b, 0]
        invalid = idx_raw < 0
        idx_safe = torch.clamp(idx_raw, min=0).to(torch.long)
        k_g = k_full[idx_safe, :D]
        v_g = k_full[idx_safe, :V]
        for h in range(H):
            q_vec = q[b, 0, h, :].float()
            scores = (q_vec @ k_g.T) * sm_scale
            scores = torch.where(invalid, torch.tensor(float("-inf"), device=device), scores)
            if invalid.all():
                # No valid keys: defined-behavior output is zeros, lse=-inf.
                continue
            lse_out[b, h, 0] = torch.logsumexp(scores, dim=0)
            attn = torch.softmax(scores, dim=0)
            out[b, 0, h, :] = attn @ v_g
    out_bf16 = out.to(torch.bfloat16)
    if attn_sink is not None:
        sink = attn_sink.view(1, 1, H, 1).float()
        lse_b = lse_out.transpose(1, 2).unsqueeze(-1).float()
        scale = 1.0 / (1.0 + torch.exp(sink - lse_b))
        out_bf16 = (out.float() * scale).to(torch.bfloat16)
    return out_bf16


from sglang.srt.layers.attention.ck_v32_sparse_mla import (
    ck_sparse_mla_decode_fp8_v32, pick_num_splits,
)


def run_ck(q, kv_fn, indices, attn_sink, sm_scale):
    B = q.shape[0]
    invalid_2d = (indices.view(B * S_q, -1) < 0)
    out, _ = ck_sparse_mla_decode_fp8_v32(
        q=q.contiguous(), k_cache=kv_fn,
        indices=indices, invalid_mask=invalid_2d,
        attn_sink=attn_sink, sm_scale=sm_scale,
    )
    return out


def main():
    sm_scale = 1.0 / (D ** 0.5)
    attn_sink = (torch.randn(H, dtype=torch.float32, device=device) * 0.5)

    print(f"{'kv_scale':>9s} {'clamp':>6s} {'kv.amax':>8s} {'B':>3s} {'topk':>5s} "
          f"{'inv':>5s} {'spl':>4s} {'cos':>9s} {'maxd':>10s} {'NaN':>4s}  PASS")
    n_pass = 0
    n_total = 0

    for kv_scale, kv_clamp in [(0.1, None), (1.0, None), (30.0, None),
                                (80.0, None), (80.0, 240.0)]:
        kv_fn = make_kv(kv_scale, kv_clamp)
        kv_amax = kv_fn.float().abs().max().item()

        for B in (1, 2, 6, 16, 64):
            for topk in (64, 128, 256):
                if topk > N_KV:
                    continue
                spl = pick_num_splits(B, topk)
                for invalid_frac in (0.0, 0.95):
                    q = (torch.randn(B, S_q, H, D, dtype=torch.bfloat16, device=device) * 0.5)
                    idx = make_indices(B, topk, invalid_frac)

                    try:
                        ck = run_ck(q, kv_fn, idx, attn_sink, sm_scale).view(B, S_q, H, V)
                    except Exception as e:
                        print(f"{kv_scale:>9.2f} {str(kv_clamp):>6s} {kv_amax:>8.1f} "
                              f"{B:>3d} {topk:>5d} {invalid_frac:>5.2f} {spl:>4d}   ERROR: {e}")
                        continue

                    nan_count = torch.isnan(ck).sum().item()
                    ref = torch_ref_fn(q, kv_fn, idx, attn_sink, sm_scale)
                    cs = cos_sim(ck, ref)
                    mxd = (ck.float() - ref.float()).abs().max().item()
                    pass_ = (nan_count == 0) and (cs >= 0.999) and (mxd < 0.5)
                    n_total += 1
                    if pass_:
                        n_pass += 1
                    print(f"{kv_scale:>9.2f} {str(kv_clamp):>6s} {kv_amax:>8.1f} "
                          f"{B:>3d} {topk:>5d} {invalid_frac:>5.2f} {spl:>4d} "
                          f"{cs:>+9.5f} {mxd:>10.3e} {nan_count:>4d}  "
                          f"{'YES' if pass_ else 'no'}")

    print(f"\n=== {n_pass}/{n_total} configs PASS ===")
    print("If 'low kv_scale' rows pass and 'high kv_scale' rows fail with cos<0.999"
          " (not NaN), Layer-2 wrong-arithmetic bug reproduces here. Use this"
          " sweep to bisect: try clamp at lower thresholds, or scale q smaller,"
          " until you find the boundary at which the kernel diverges from the"
          " FP32 oracle.")


if __name__ == "__main__":
    main()
