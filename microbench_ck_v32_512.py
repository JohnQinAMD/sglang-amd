"""Microbench: CK V32 sparse MLA (qk_head_dim=512) vs torch reference.

Goal v1 (2026-04-26 morning): identify why CK V32 produced 2× scaled output for
Flash's qk_head_dim=512. Hypothesis confirmed: FP8 decode used fnuz bias-8 fold
but actual storage is fn (bias=7). Fixed in commit eb373f796 by parametrizing
`fp8_decode_scale` and the wrapper auto-picking 1.0 for fn / 0.5 for fnuz.

Goal v2 (2026-04-26 evening — this version): the cos_sim 0.999998 from v1's
TOPK=64 / attn_sink=None microbench is misleading — end-to-end Flash mxfp4
inference still returns garbage tokens with `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1`
even after the fp8_decode_scale fix. Sweep more shapes (TOPK, attn_sink,
invalid-mask coverage) to expose where the kernel breaks.

Setup matches Flash decode: B=1, S_q=1, H=128 (q heads), D=512 (qk_head_dim),
V=512 (v_head_dim).

Sweeps:
  TOPK             ∈ {64, 128, 256, 512, 1024}   (production hits 256+)
  attn_sink        ∈ {None, random fp32 [H]}     (production passes per-head sink)
  invalid_frac     ∈ {0.0, 0.25, 0.50}           (production has some -1 indices)

PASS/FAIL: cos_sim ≥ 0.999 vs FP32 oracle (FP32 oracle, not FP8-quantized
reference, since the kernel reads FP8 — we want to prove the kernel matches the
attention math even with the FP8 quant noise).
"""
import os
import torch

os.environ["SGLANG_HIP_SPARSE_MLA_DECODE_FP8"] = "1"  # gate the CK V32 path

# ───── Shape ────────────────────────────────────────────────────────────────
B, S_q, H, V = 1, 1, 128, 512
D = int(os.environ.get("D", "512"))         # 512=Flash 2604, 576=Pro V32
N_KV = int(os.environ.get("N_KV", "8192"))  # KV cache slots; production has thousands
SLOT = 584                                   # production KV slot stride (nope=512+rope=64+scale=8)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
device = "cuda"
FN = torch.float8_e4m3fn
FNUZ = getattr(torch, "float8_e4m3fnuz", FN)

# Production-realistic q + KV. Scale 0.1 keeps values inside FP8 range.
q = (torch.randn(B, S_q, H, D, dtype=torch.bfloat16, device=device) * 0.1)
k_cache_bf16 = torch.randn(N_KV, SLOT, dtype=torch.bfloat16, device=device) * 0.1
k_cache_fn = k_cache_bf16.to(FN)

print(f"=== Setup ===")
print(f"  B={B}, S_q={S_q}, H={H}, D={D}, V={V}, N_KV={N_KV}, SLOT={SLOT}")
print(f"  fn dtype: {k_cache_fn.dtype}, sample bytes [0,:8]: {k_cache_fn.view(torch.uint8)[0,:8].tolist()}")

from sglang.srt.layers.attention.ck_v32_sparse_mla import (
    ck_sparse_mla_decode_fp8_v32, pick_num_splits,
)


def make_indices(topk, invalid_frac):
    """Build [B, S_q, topk] indices with `invalid_frac` of them set to -1.

    Production decode marks invalid positions as -1; the kernel treats them as
    masked rows (zero-fill in the LDS tile loader).
    """
    raw = torch.randint(0, N_KV, (B, S_q, topk), dtype=torch.int32, device=device)
    if invalid_frac > 0:
        n_invalid = int(round(topk * invalid_frac))
        if n_invalid > 0:
            # Mark a contiguous tail as invalid (matches sglang's topk_length
            # truncation pattern: `indices[mask] = -1` where mask is the suffix).
            raw[..., -n_invalid:] = -1
    return raw


def torch_ref_fn(indices, attn_sink):
    """FP32 oracle — re-quantize to FP8 first to match what the kernel sees, then
    upcast to fp32 and run dense attention.

    Critical: this dequant uses the SAME fp8 bytes as `k_cache_fn`. The kernel's
    cvt_pk_f32_fp8 instruction also reads fn bytes → both should agree exactly
    after the parametric `fp8_decode_scale=1.0` fix.
    """
    # Re-dequant fp8→fp32 the same way the kernel will, ignoring scale (test
    # data has unit scale).
    k_full = k_cache_fn.float()  # [N_KV, SLOT]

    sm_scale = 1.0 / (D ** 0.5)
    out = torch.zeros(B, S_q, H, V, dtype=torch.float32, device=device)
    lse_out = torch.zeros(B, H, S_q, dtype=torch.float32, device=device)
    for b in range(B):
        idx_raw = indices[b, 0]                  # [topk] int32, -1 = invalid
        invalid = idx_raw < 0
        # Clamp to 0 for gather; mask the result with -inf in score space.
        idx_safe = torch.clamp(idx_raw, min=0).to(torch.long)
        k_gathered = k_full[idx_safe, :D]        # [topk, D] fp32
        v_gathered = k_full[idx_safe, :V]        # [topk, V] fp32
        for h in range(H):
            q_vec = q[b, 0, h, :].float()         # [D]
            scores = (q_vec @ k_gathered.T) * sm_scale  # [topk]
            scores = torch.where(invalid, torch.tensor(float("-inf"), device=device), scores)
            lse = torch.logsumexp(scores, dim=0)  # scalar
            attn = torch.softmax(scores, dim=0)
            out[b, 0, h, :] = attn @ v_gathered
            lse_out[b, h, 0] = lse
    out_bf16 = out.to(torch.bfloat16)
    if attn_sink is not None:
        # Match the wrapper: out *= 1 / (1 + exp(sink - lse)).
        # lse_out: [B, H, S_q]; need broadcast to [B, S_q, H, V].
        sink = attn_sink.view(1, 1, H, 1).float()
        lse_b = lse_out.transpose(1, 2).unsqueeze(-1).float()  # [B, S_q, H, 1]
        scale = 1.0 / (1.0 + torch.exp(sink - lse_b))
        out_bf16 = (out.float() * scale).to(torch.bfloat16)
    return out_bf16, lse_out


def run_ck(indices, attn_sink):
    invalid_2d = (indices.view(B * S_q, -1) < 0)
    sm_scale = 1.0 / (D ** 0.5)
    out, lse = ck_sparse_mla_decode_fp8_v32(
        q=q, k_cache=k_cache_fn,
        indices=indices, invalid_mask=invalid_2d,
        attn_sink=attn_sink, sm_scale=sm_scale,
    )
    return out, lse


def cos_sim(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    return (af @ bf / (af.norm() * bf.norm() + 1e-9)).item()


# ───── Sweep ────────────────────────────────────────────────────────────────
print("\n=== Sweep: TOPK × attn_sink × invalid_frac ===")
print(f"{'TOPK':>5}  {'splits':>6}  {'sink':>5}  {'inv_frac':>9}  "
      f"{'cos_sim':>9}  {'max_diff':>10}  {'ratio_med':>10}  PASS?")

attn_sink_v = (torch.randn(H, dtype=torch.float32, device=device) * 0.5)  # production sink scale
results = []
for topk in [64, 128, 256, 512, 1024]:
    splits = pick_num_splits(B, topk)
    indices_all = {f: make_indices(topk, f) for f in (0.0, 0.25, 0.50)}
    for sink_label, sink_t in (("None", None), ("real", attn_sink_v)):
        for inv_frac, idx in indices_all.items():
            ref_out, ref_lse = torch_ref_fn(idx, sink_t)
            ck_out, ck_lse = run_ck(idx, sink_t)

            cs = cos_sim(ck_out, ref_out)
            mxd = (ck_out.float() - ref_out.float()).abs().max().item()
            # Median ratio on |ref| > 1e-3 to avoid div-by-zero noise on tiny outputs
            mask = ref_out.float().abs() > 1e-3
            if mask.any():
                ratio = (ck_out.float()[mask] / ref_out.float()[mask]).abs().median().item()
            else:
                ratio = float("nan")

            ok = cs >= 0.999 and mxd < 0.5
            tag = "PASS" if ok else "FAIL"
            print(f"{topk:>5d}  {splits:>6d}  {sink_label:>5s}  {inv_frac:>9.2f}  "
                  f"{cs:>+9.5f}  {mxd:>10.4e}  {ratio:>10.4f}  {tag}")
            results.append((topk, splits, sink_label, inv_frac, cs, mxd, ratio, ok))

# ───── Summary ──────────────────────────────────────────────────────────────
print("\n=== Summary ===")
n_pass = sum(1 for r in results if r[-1])
n_fail = len(results) - n_pass
print(f"  {n_pass}/{len(results)} configs pass (cos_sim ≥ 0.999, max_diff < 0.5)")
if n_fail > 0:
    print(f"  {n_fail} FAIL configs — first 5:")
    for topk, splits, sink, inv, cs, mxd, ratio, _ in [r for r in results if not r[-1]][:5]:
        print(f"    TOPK={topk:4d} splits={splits:2d} sink={sink:>5s} inv={inv:.2f}: "
              f"cos_sim={cs:+.5f} max_diff={mxd:.4e} ratio_med={ratio:.4f}")

# ───── Diagnostic dump for a single failing config ──────────────────────────
fails = [r for r in results if not r[-1]]
if fails:
    topk, splits, sink_label, inv_frac, *_ = fails[0]
    print(f"\n=== Diagnostic: first failing config TOPK={topk} sink={sink_label} inv={inv_frac:.2f} ===")
    sink_t = attn_sink_v if sink_label == "real" else None
    idx = make_indices(topk, inv_frac)
    ref_out, _ = torch_ref_fn(idx, sink_t)
    ck_out, _ = run_ck(idx, sink_t)
    print(f"  ref_out[0,0,0,:8]={ref_out[0,0,0,:8].float().tolist()}")
    print(f"  ck_out [0,0,0,:8]={ck_out[0,0,0,:8].float().tolist()}")
    print(f"  ratio per-head (h ∈ {{0,1,...,7}}): "
          + ", ".join(f"{(ck_out[0,0,h,:].float() / (ref_out[0,0,h,:].float().abs() + 1e-9)).abs().median().item():.3f}"
                      for h in range(8)))

# ───── Note on what this microbench does NOT cover ──────────────────────────
# As of 2026-04-26 evening, the sweep above passes 30/30 at both D=512 and
# D=576, yet end-to-end Flash mxfp4 inference with
# `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1` produces silent garbage tokens.
#
# Therefore the kernel is correct *in isolation* and the e2e bug lives
# elsewhere in the pipeline. Things this microbench does NOT replicate:
#
#   1. Multi-layer compounding. Per-call max_diff is ~1e-3 with FP8 quant
#      noise; over 60+ decoder layers, repeated dependence on the
#      attention output through residual connections + RMSnorm + MoE can
#      drift the next layer's q distribution enough to push tokens off
#      the manifold. The torch-ref path (cos_sim 1.0 vs the same FP8
#      kernel arithmetic) avoids this because it uses bf16 matmul which
#      has lower per-call noise than the FP8 path.
#
#   2. Production KV distribution. We use `randn() * 0.1` here. Real KV
#      after running through the model has heavy tails, occasional
#      outliers near FP8_MAX, and per-token correlation across the
#      nope/rope split that randn doesn't capture.
#
#   3. Production index distribution. We use `randint(0, N_KV)` here.
#      Real indices come from the indexer scoring + `topk_transform_512`
#      (page_id << 8 | offset), so they cluster around recently-used
#      pages. Could expose paging-related stride bugs the random
#      indices miss.
#
#   4. Multi-stream overlap / cuda graph capture interactions. The kernel
#      is called from inside a captured graph in production; any state
#      that depends on stream/event ordering won't surface here.
#
# Suggested next debug step: add a `SGLANG_DUMP_CK_V32_TENSORS=1` hook in
# `ck_sparse_mla_decode_fp8_v32` that snapshots (q, k_cache, indices,
# attn_sink, sm_scale) on the first call after server warmup, then replay
# them through both ck_sparse_mla_decode_fp8_v32 and a bf16 dense
# reference here. If outputs disagree on captured tensors but agree on
# random tensors, distribution is the cause; if they agree on captured
# tensors too, the bug is multi-layer compounding (and the practical fix
# is to keep `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0` until we can budget
# the per-call FP8 noise more tightly).
