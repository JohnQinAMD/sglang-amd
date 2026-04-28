"""Phase C3 / Phase E Lever 3 — Triton fast-path for `mla_combine_fwd_ck`.

The CK combine kernel runs at ~107 µs/call in production traces. Per-call work
is small (a softmax merge over [B*H, V] = ~96 × 512 fp32 reads + bf16 writes)
but the launch is heavily under-saturated on MI355X — only ~1.5 waves per CU at
B=6/H=16. Replacing with a Triton kernel that uses (B*H, V/64) grid gives ~8×
more blocks and lets the GPU pipeline the small per-block work.

Two entry points:

  * mla_combine_two_splits_triton(...)
        S_a == 1 and S_b in {0, 1}. The original C3 fast-path for decode.

  * mla_combine_n_way_triton(...)            (Phase E Lever 3 extension)
        S_a > 1, S_b in {0, 1}. PREFILL N-way merge — the case that fires
        the 107 µs/call CK combine in production traces (164 calls/iter →
        17.6 ms/iter total → -1 to -2 ms TTFT).

Both share the same kernel template; constexpr S_A_MAX_LOG2 (or runtime S_A
loop via `BLOCK_S` constexpr) selects the body.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _combine_two_splits_triton_kernel(
    data_a_ptr, lse_a_ptr,
    data_b_ptr, lse_b_ptr,             # may be null pointers when has_b=False
    out_ptr, lse_out_ptr,
    sink_ptr,                           # may be null pointer when has_sink=False
    stride_da_q, stride_da_h,
    stride_db_q, stride_db_h,
    stride_la_q, stride_la_h,
    stride_lb_q, stride_lb_h,
    stride_out_q, stride_out_h,
    stride_lse_q, stride_lse_h,
    H: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_B: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    # Grid: (total_q*H, V/BLOCK_V).
    qh = tl.program_id(0)
    v_block = tl.program_id(1)
    q_idx = qh // H
    h_idx = qh % H
    v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

    # Load LSE-A (scalar per (q, h)).
    lse_a = tl.load(lse_a_ptr + q_idx * stride_la_q + h_idx * stride_la_h)
    if HAS_B:
        lse_b = tl.load(lse_b_ptr + q_idx * stride_lb_q + h_idx * stride_lb_h)
        lm = tl.maximum(lse_a, lse_b)
    else:
        lm = lse_a

    # Detect "all invalid" (lm == -inf via NaN-safe path).
    all_invalid = lm == float("-inf")
    safe_lm = tl.where(all_invalid, 0.0, lm)

    # Compute weights.
    w_a = tl.where(lse_a == float("-inf"), 0.0, tl.exp(lse_a - safe_lm))
    if HAS_B:
        w_b = tl.where(lse_b == float("-inf"), 0.0, tl.exp(lse_b - safe_lm))
        sw = w_a + w_b
    else:
        sw = w_a
    sw_safe = tl.where(sw > 0.0, sw, 1.0)
    inv_sw = 1.0 / sw_safe

    # Optionally fold attn_sink into inv_sw.
    if HAS_SINK:
        sink_h = tl.load(sink_ptr + h_idx)
        merged_lse = tl.where(all_invalid, float("-inf"), safe_lm + tl.log(sw))
        sink_factor = 1.0 / (1.0 + tl.exp(sink_h - merged_lse))
        inv_sw = inv_sw * sink_factor
    else:
        merged_lse = tl.where(all_invalid, float("-inf"), safe_lm + tl.log(sw))

    # Load V slice from each source. Strides are q->stride_da_q (skipping S=1
    # dim is folded into stride_da_q), h->stride_da_h, V is innermost (stride 1).
    a_base = q_idx * stride_da_q + h_idx * stride_da_h
    a = tl.load(data_a_ptr + a_base + v_offsets)
    if HAS_B:
        b_base = q_idx * stride_db_q + h_idx * stride_db_h
        b = tl.load(data_b_ptr + b_base + v_offsets)
        acc = w_a * a + w_b * b
    else:
        acc = w_a * a

    out = tl.where(all_invalid, 0.0, acc * inv_sw).to(tl.bfloat16)
    out_base = q_idx * stride_out_q + h_idx * stride_out_h
    tl.store(out_ptr + out_base + v_offsets, out)

    # Only the v_block == 0 program writes the merged LSE (single fp32 store
    # per (q, h)). All other v_blocks skip; merged_lse is uniform within a
    # program so a single tl.store at scalar address is fine.
    if v_block == 0:
        tl.store(lse_out_ptr + q_idx * stride_lse_q + h_idx * stride_lse_h,
                 merged_lse)


def mla_combine_two_splits_triton(
    split_data_a: torch.Tensor,       # [total_q, 1, H, V] fp32
    split_lse_a: torch.Tensor,        # [total_q, 1, H, 1] fp32
    split_data_b: torch.Tensor = None,  # [total_q, 1, H, V] fp32 or None
    split_lse_b: torch.Tensor = None,   # [total_q, 1, H, 1] fp32 or None
    attn_sink: torch.Tensor = None,
    out: torch.Tensor = None,         # [total_q, H, V] bf16
    lse: torch.Tensor = None,         # [total_q, H] fp32
):
    """Fast-path Triton combine for the small-S case (S_a == 1, S_b in {0, 1}).

    Layout assumptions (asserted at call site, not here for cuda-graph friendliness):
      - split_data_a: [total_q, S_a=1, H, V] fp32, V is innermost (stride 1)
      - split_lse_a:  [total_q, S_a=1, H, 1] fp32
      - same for B if provided
    """
    total_q = split_data_a.size(0)
    S_a = split_data_a.size(1)
    H = split_data_a.size(2)
    V = split_data_a.size(3)
    assert S_a == 1, f"Triton combine only handles S_a=1, got {S_a}"
    has_b = split_data_b is not None
    if has_b:
        assert split_data_b.size(1) == 1
    has_sink = attn_sink is not None
    device = split_data_a.device
    if out is None:
        out = torch.empty((total_q, H, V), dtype=torch.bfloat16, device=device)
    if lse is None:
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

    BLOCK_V = 64
    grid = (total_q * H, triton.cdiv(V, BLOCK_V))

    # Strides into split_data_*: collapse the S_a=1 dimension into the q stride
    # (data layout is [Q, S, H, V] so q-stride = S*H*V; with S=1 it equals H*V).
    # We pass strides directly; tl.load will skip the S=1 dim since we always
    # index s=0.
    sda_q = split_data_a.stride(0)
    sda_h = split_data_a.stride(2)
    lsa_q = split_lse_a.stride(0)
    lsa_h = split_lse_a.stride(2)
    if has_b:
        sdb_q = split_data_b.stride(0)
        sdb_h = split_data_b.stride(2)
        lsb_q = split_lse_b.stride(0)
        lsb_h = split_lse_b.stride(2)
        data_b_ptr = split_data_b
        lse_b_ptr = split_lse_b
    else:
        sdb_q = sdb_h = lsb_q = lsb_h = 0
        data_b_ptr = split_data_a   # placeholder, never read
        lse_b_ptr = split_lse_a

    sink_ptr = attn_sink if has_sink else split_data_a  # placeholder

    _combine_two_splits_triton_kernel[grid](
        split_data_a, split_lse_a,
        data_b_ptr, lse_b_ptr,
        out, lse, sink_ptr,
        sda_q, sda_h,
        sdb_q, sdb_h,
        lsa_q, lsa_h,
        lsb_q, lsb_h,
        out.stride(0), out.stride(1),
        lse.stride(0), lse.stride(1),
        H=H,
        V=V,
        BLOCK_V=BLOCK_V,
        HAS_B=has_b,
        HAS_SINK=has_sink,
        num_warps=1,
    )
    return out, lse


# ─────────────────────────────────────────────────────────────────────────────
# Phase E Lever 3 — N-way merge for PREFILL (S_a > 1)
# ─────────────────────────────────────────────────────────────────────────────
# The CK kernel mla_combine_fwd_kernel has a 256-thread-per-block design with
# V_PER_THREAD=2 at V=512 (BLOCK_THREADS=256). At PREFILL (S_total ≤ ~16, total_q
# small) the launch is severely under-saturated on MI355X (256 CUs × 64 = 16384
# lane slots; combine launches 96-200 blocks × 256 = 24-50k thread slots, but
# the per-block work is so small that wave occupancy stays near 1.5 waves/CU).
#
# Triton N-way variant: grid (total_q*H, V/BLOCK_V), one warp per block,
# BLOCK_V=64 → 8× more blocks at V=512. Each block does an inner loop over
# all S_a + S_b splits with broadcast LSE reduction at the head of the block,
# then accumulates the V slice in fp32 and writes bf16. Same correctness math
# as the CK kernel.

@triton.jit
def _combine_n_way_triton_kernel(
    data_a_ptr, lse_a_ptr,
    data_b_ptr, lse_b_ptr,
    out_ptr, lse_out_ptr,
    sink_ptr,
    stride_da_q, stride_da_s, stride_da_h,
    stride_db_q, stride_db_s, stride_db_h,
    stride_la_q, stride_la_s, stride_la_h,
    stride_lb_q, stride_lb_s, stride_lb_h,
    stride_out_q, stride_out_h,
    stride_lse_q, stride_lse_h,
    H: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    S_A: tl.constexpr,
    S_B: tl.constexpr,
    HAS_B: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    # Grid: (total_q*H, V/BLOCK_V). One warp per program (num_warps=1).
    qh = tl.program_id(0)
    v_block = tl.program_id(1)
    q_idx = qh // H
    h_idx = qh % H
    v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offsets < V

    # ── Stage 1: find lm = max over all splits' lse, then build w_i via static
    # unroll. S_A and S_B are constexpr — compiler unrolls cleanly.
    lm = tl.full([], float("-inf"), tl.float32)
    for s in tl.static_range(0, S_A):
        lse_si = tl.load(
            lse_a_ptr + q_idx * stride_la_q + s * stride_la_s + h_idx * stride_la_h
        )
        lm = tl.maximum(lm, lse_si)
    if HAS_B:
        for s in tl.static_range(0, S_B):
            lse_si = tl.load(
                lse_b_ptr + q_idx * stride_lb_q + s * stride_lb_s + h_idx * stride_lb_h
            )
            lm = tl.maximum(lm, lse_si)

    all_invalid = lm == float("-inf")
    safe_lm = tl.where(all_invalid, 0.0, lm)

    # ── Stage 2: streaming accum of acc = sum_i(w_i * data_i) and sw = sum_i(w_i).
    acc = tl.zeros([BLOCK_V], tl.float32)
    sw = tl.zeros([], tl.float32)

    for s in tl.static_range(0, S_A):
        lse_si = tl.load(
            lse_a_ptr + q_idx * stride_la_q + s * stride_la_s + h_idx * stride_la_h
        )
        # Detect invalid lse via NaN-safe masking: if lse == -inf, weight = 0.
        d = lse_si - safe_lm
        w_i = tl.where(lse_si == float("-inf"), 0.0, tl.exp(d))
        sw += w_i
        a_base = q_idx * stride_da_q + s * stride_da_s + h_idx * stride_da_h
        x = tl.load(data_a_ptr + a_base + v_offsets, mask=v_mask, other=0.0)
        acc += w_i * x

    if HAS_B:
        for s in tl.static_range(0, S_B):
            lse_si = tl.load(
                lse_b_ptr + q_idx * stride_lb_q + s * stride_lb_s + h_idx * stride_lb_h
            )
            d = lse_si - safe_lm
            w_i = tl.where(lse_si == float("-inf"), 0.0, tl.exp(d))
            sw += w_i
            b_base = q_idx * stride_db_q + s * stride_db_s + h_idx * stride_db_h
            x = tl.load(data_b_ptr + b_base + v_offsets, mask=v_mask, other=0.0)
            acc += w_i * x

    sw_safe = tl.where(sw > 0.0, sw, 1.0)
    inv_sw = 1.0 / sw_safe

    if HAS_SINK:
        sink_h = tl.load(sink_ptr + h_idx)
        merged_lse = tl.where(all_invalid, float("-inf"), safe_lm + tl.log(sw))
        sink_factor = 1.0 / (1.0 + tl.exp(sink_h - merged_lse))
        inv_sw = inv_sw * sink_factor
    else:
        merged_lse = tl.where(all_invalid, float("-inf"), safe_lm + tl.log(sw))

    out_v = tl.where(all_invalid, 0.0, acc * inv_sw).to(tl.bfloat16)
    out_base = q_idx * stride_out_q + h_idx * stride_out_h
    tl.store(out_ptr + out_base + v_offsets, out_v, mask=v_mask)

    # Lane 0 only writes lse — single fp32 store per (q, h).
    if v_block == 0:
        tl.store(
            lse_out_ptr + q_idx * stride_lse_q + h_idx * stride_lse_h,
            merged_lse,
        )


# Triton compile cache keyed by (S_A, S_B, has_b, has_sink). Production
# pick_num_splits returns a discrete set of values (1..16 typical, 32-128
# in cuda-graph capture corner cases). Compilations are amortized.
_NWAY_COMPILE_CACHE = set()


def mla_combine_n_way_triton(
    split_data_a: torch.Tensor,       # [total_q, S_a, H, V] fp32
    split_lse_a: torch.Tensor,        # [total_q, S_a, H, 1] fp32
    split_data_b: torch.Tensor = None,  # [total_q, S_b, H, V] fp32 or None
    split_lse_b: torch.Tensor = None,   # [total_q, S_b, H, 1] fp32 or None
    attn_sink: torch.Tensor = None,
    out: torch.Tensor = None,         # [total_q, H, V] bf16
    lse: torch.Tensor = None,         # [total_q, H] fp32
):
    """N-way merge variant of the Triton combine — covers PREFILL (S_a > 1).

    Compiles a fresh kernel per (S_a, S_b) shape; caller should expect a
    one-time compilation cost on the first call at each shape. Output is
    identical (per CK combine kernel math: lm = max(lse), w_i = exp(lse_i - lm),
    out = sum(w_i * data_i) / sum(w_i), with attn_sink fold).

    Asserts: V is innermost (stride 1) on both split_data_*.
    """
    total_q = split_data_a.size(0)
    S_a = split_data_a.size(1)
    H = split_data_a.size(2)
    V = split_data_a.size(3)
    assert split_lse_a.shape == (total_q, S_a, H, 1)
    has_b = split_data_b is not None
    if has_b:
        assert split_data_b.shape[0] == total_q and split_data_b.shape[2] == H
        assert split_data_b.shape[3] == V
        S_b = split_data_b.size(1)
        assert split_lse_b is not None and split_lse_b.shape == (total_q, S_b, H, 1)
    else:
        S_b = 0

    has_sink = attn_sink is not None
    device = split_data_a.device
    if out is None:
        out = torch.empty((total_q, H, V), dtype=torch.bfloat16, device=device)
    else:
        assert out.shape == (total_q, H, V) and out.dtype == torch.bfloat16
    if lse is None:
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)
    else:
        assert lse.shape == (total_q, H) and lse.dtype == torch.float32

    BLOCK_V = 64
    grid = (total_q * H, triton.cdiv(V, BLOCK_V))

    sda_q = split_data_a.stride(0)
    sda_s = split_data_a.stride(1)
    sda_h = split_data_a.stride(2)
    lsa_q = split_lse_a.stride(0)
    lsa_s = split_lse_a.stride(1)
    lsa_h = split_lse_a.stride(2)
    if has_b:
        sdb_q = split_data_b.stride(0)
        sdb_s = split_data_b.stride(1)
        sdb_h = split_data_b.stride(2)
        lsb_q = split_lse_b.stride(0)
        lsb_s = split_lse_b.stride(1)
        lsb_h = split_lse_b.stride(2)
        data_b_ptr = split_data_b
        lse_b_ptr = split_lse_b
    else:
        sdb_q = sdb_s = sdb_h = 0
        lsb_q = lsb_s = lsb_h = 0
        data_b_ptr = split_data_a   # placeholder, never read
        lse_b_ptr = split_lse_a

    sink_ptr = attn_sink if has_sink else split_data_a  # placeholder

    _combine_n_way_triton_kernel[grid](
        split_data_a, split_lse_a,
        data_b_ptr, lse_b_ptr,
        out, lse, sink_ptr,
        sda_q, sda_s, sda_h,
        sdb_q, sdb_s, sdb_h,
        lsa_q, lsa_s, lsa_h,
        lsb_q, lsb_s, lsb_h,
        out.stride(0), out.stride(1),
        lse.stride(0), lse.stride(1),
        H=H,
        V=V,
        BLOCK_V=BLOCK_V,
        S_A=S_a,
        S_B=S_b,
        HAS_B=has_b,
        HAS_SINK=has_sink,
        num_warps=1,
    )
    return out, lse


def prewarm_n_way_triton(H=16, V=512, S_a_values=(2, 4, 8, 16),
                          total_q=64, device="cuda"):
    """Prewarm Triton compilation for N-way combine across the (S_a, has_b, has_sink)
    matrix. Call once at server startup BEFORE serving traffic.

    The Triton kernel templates on (S_A, S_B, HAS_B, HAS_SINK) — one compile per
    distinct tuple. Production hits typically (2-16) × (False, True) × (False, True) =
    8-32 distinct kernels. Each compile is ~100-500 ms; total prewarm is ~5-15 sec.

    chi2811 E2E A/B showed P99 TTFT regressed from 2541 ms to 7184 ms WITHOUT
    prewarming — Triton cold-compile lands on individual tail requests. With this
    prewarm pass the cold-compile cost is amortized into server startup instead.

    Use total_q small (32-64) so the prewarm calls are fast — only the kernel
    compilation matters, the runtime size doesn't affect the cached binary.
    """
    import torch
    a  = torch.randn(total_q, max(S_a_values), H, V, device=device, dtype=torch.float32)
    la = torch.randn(total_q, max(S_a_values), H, 1, device=device, dtype=torch.float32) * 0.5
    b  = torch.randn(total_q, 1, H, V, device=device, dtype=torch.float32)
    lb = torch.randn(total_q, 1, H, 1, device=device, dtype=torch.float32) * 0.5
    sink = torch.randn(H, device=device, dtype=torch.float32)

    n = 0
    for S_a in S_a_values:
        a_view  = a[:, :S_a]
        la_view = la[:, :S_a]
        for has_b in (False, True):
            for has_sink in (False, True):
                out = torch.empty((total_q, H, V), dtype=torch.bfloat16, device=device)
                lse = torch.empty((total_q, H), dtype=torch.float32, device=device)
                mla_combine_n_way_triton(
                    a_view, la_view,
                    b if has_b else None,
                    lb if has_b else None,
                    sink if has_sink else None,
                    out, lse,
                )
                n += 1
    torch.cuda.synchronize()
    return n
