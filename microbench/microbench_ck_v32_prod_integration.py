"""Integration test using REAL production tensors instead of synthetic data.

Loads the captures from _ck_v32_dumps_flash_mxfp4/ and replays them through
the wrapper under 4 conditions:
  1. Single eager call (baseline).
  2. 50 sequential calls (cache reuse — the wrapper's _get_split_buffers,
     _OUT_BUF_CACHE, _REDUCE_OUT_CACHE).
  3. cuda graph capture + replay.
  4. 60-call rotating-q (mimics 60-layer decode pattern with rotating real q
     tensors from the captures).

If all 4 PASS at bf16 precision, the residual e2e bug is in production-only
state (model weights × forward pass × cuda graph multi-stream) — beyond
isolated kernel/wrapper testing. If any FAIL, that condition reproduces
the production garbage pattern.
"""
import os, sys, glob, math
import torch

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
    sys.exit(2)

# Load all production captures into device memory.
captures = []
for p in paths:
    payload = torch.load(p, map_location=device, weights_only=False)
    q_t = payload["q"].to(device)
    if q_t.float().abs().max().item() == 0:
        continue  # skip cuda-graph-capture placeholder
    captures.append({
        "name": os.path.basename(p),
        "q": q_t,
        "k_cache": payload["k_cache"].to(device),
        "indices": payload["indices"].to(device).to(torch.int32),
        "attn_sink": (None if payload["attn_sink"] is None
                      else payload["attn_sink"].to(device).float()),
        "sm_scale": float(payload["sm_scale"]),
    })
print(f"Loaded {len(captures)} non-degenerate production captures")
print()


def cos(a, b):
    af, bf = a.float().flatten(), b.float().flatten()
    n = (af.norm() * bf.norm()).item()
    return float(af @ bf) / n if n > 1e-12 else float("nan")


def per_batch_oracle(q, kv, indices, attn_sink, sm_scale):
    B, S_q, H, D = q.shape
    V = 512
    if kv.dim() == 4:
        npg, ps, _, slot = kv.shape
        kv2d = kv.reshape(npg * ps, slot)
    elif kv.dim() == 3:
        kv2d = kv.reshape(-1, kv.shape[-1])
    else:
        kv2d = kv
    if kv2d.dtype == torch.uint8:
        kv2d = kv2d.view(torch.float8_e4m3fn)
    k_full = kv2d.float()
    out = torch.zeros(B, S_q, H, V, dtype=torch.float32, device=device)
    lse_out = torch.full((B, H, S_q), float("-inf"), dtype=torch.float32, device=device)
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
                                 torch.tensor(float("-inf"), device=device), scores)
            lse_out[b, :, s] = torch.logsumexp(scores, dim=-1)
            attn = torch.softmax(scores, dim=-1)
            out[b, s, :, :] = attn @ v_g
    if attn_sink is not None:
        sc = 1.0 / (1.0 + torch.exp(
            attn_sink.view(1, 1, H, 1) - lse_out.transpose(1, 2).unsqueeze(-1)))
        sc = torch.where(torch.isfinite(sc), sc, torch.zeros_like(sc))
        out = out * sc
    return torch.nan_to_num(out, nan=0.0).to(torch.bfloat16)


def diff(out, oracle):
    cs = cos(out, oracle)
    mxd = (out.float() - oracle.float()).abs().max().item()
    return cs, mxd


def is_pass(cs, mxd, oracle):
    """Treat 'both zero' (kernel and oracle agree at zero) as PASS even when
    cos is NaN (which happens for cos(0,0)=0/0). For non-degenerate cases,
    require cos > 0.999."""
    oracle_norm = oracle.float().norm().item()
    if oracle_norm < 1e-3:
        # Oracle output is all-zero (degenerate, all-invalid head case).
        # Kernel agrees iff its output is also near-zero AND maxd is tiny.
        return mxd < 0.5
    return cs > 0.999 and mxd < 5.0


def call_wrapper(c):
    invalid_mask = (c["indices"].view(c["indices"].shape[0]
                                      * c["indices"].shape[1], -1) < 0)
    out, _ = ck_sparse_mla_decode_fp8_v32(
        q=c["q"].contiguous(), k_cache=c["k_cache"], indices=c["indices"],
        invalid_mask=invalid_mask, attn_sink=c["attn_sink"],
        sm_scale=c["sm_scale"],
    )
    return out


# Pre-compute all per-batch oracles.
oracles = [per_batch_oracle(c["q"], c["k_cache"], c["indices"],
                             c["attn_sink"], c["sm_scale"]) for c in captures]


def report(label, results):
    print(f"=== {label} ===")
    n_pass = sum(1 for r in results if r["pass"])
    print(f"  {n_pass}/{len(results)} captures PASS")
    for r in results:
        tag = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['name']:<55s} cos={r['cos']:+.5f} maxd={r['maxd']:.3e} {tag}")
    print()
    return n_pass == len(results)


# ───── TEST 1: single call per capture (baseline) ─────
results = []
for c, ora in zip(captures, oracles):
    out = call_wrapper(c)
    cs, mxd = diff(out, ora)
    results.append({"name": c["name"], "cos": cs, "maxd": mxd, "pass": is_pass(cs, mxd, ora)})
torch.cuda.synchronize()
test1_ok = report("TEST 1 — Single eager call per capture", results)

# ───── TEST 2: 50 sequential calls of the SAME capture (cache reuse) ─────
results = []
for c, ora in zip(captures, oracles):
    for _ in range(50):
        out = call_wrapper(c)
    cs, mxd = diff(out, ora)
    results.append({"name": c["name"], "cos": cs, "maxd": mxd, "pass": is_pass(cs, mxd, ora)})
torch.cuda.synchronize()
test2_ok = report("TEST 2 — 50 sequential calls same capture (cache reuse)", results)

# ───── TEST 3: rotate captures across 60 calls (mimics 60 layers) ─────
# Use modulo to cycle. Check the LAST capture's output against its oracle.
results = []
for tgt in range(len(captures)):
    seq = [(tgt + i) % len(captures) for i in range(60)]
    # Reorder so the LAST call is captures[tgt].
    seq = seq[:-1] + [tgt]
    for s_idx in seq:
        out = call_wrapper(captures[s_idx])
    # Check the final out (== captures[tgt]'s output).
    cs, mxd = diff(out, oracles[tgt])
    results.append({"name": captures[tgt]["name"],
                     "cos": cs, "maxd": mxd,
                     "pass": is_pass(cs, mxd, oracles[tgt])})
torch.cuda.synchronize()
test3_ok = report("TEST 3 — 60-call rotating across captures (60-layer mimic)", results)

# ───── TEST 4: cuda graph capture + replay per capture ─────
results = []
for c, ora in zip(captures, oracles):
    # Warmup (must run eagerly to allocate JIT-built modules).
    _ = call_wrapper(c)
    torch.cuda.synchronize()
    # Capture.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = call_wrapper(c)
    # Replay multiple times to stress.
    for _ in range(11):
        g.replay()
    torch.cuda.synchronize()
    cs, mxd = diff(out_g, ora)
    results.append({"name": c["name"], "cos": cs, "maxd": mxd, "pass": is_pass(cs, mxd, ora)})
test4_ok = report("TEST 4 — cuda graph capture + 11 replays", results)

print("=" * 75)
print(f"Summary: TEST1={'PASS' if test1_ok else 'FAIL'} "
      f"TEST2={'PASS' if test2_ok else 'FAIL'} "
      f"TEST3={'PASS' if test3_ok else 'FAIL'} "
      f"TEST4={'PASS' if test4_ok else 'FAIL'}")
if all([test1_ok, test2_ok, test3_ok, test4_ok]):
    print("All 4 tests PASS — bug is in production-only state (model × graph × stream)")
else:
    failed = [n for n, ok in zip([1,2,3,4], [test1_ok, test2_ok, test3_ok, test4_ok]) if not ok]
    print(f"FAILED tests: {failed} — that condition reproduces the production drift")
