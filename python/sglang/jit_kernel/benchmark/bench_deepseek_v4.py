"""Microbenchmark for deepseek_v4 jit_kernel functions at Pro / Flash shapes.

Shapes are derived from
  /mnt/vast/john/huggingface/DeepSeek-V4-Flash/config.json
  /mnt/vast/john/huggingface/DeepSeek-V4-Pro/config.json

Each kernel is timed in isolation with `triton.testing.do_bench` (median over
warm-up + repeat). Kernels that have no HIP path are skipped on ROCm with a
short note. The bench prints a markdown table per kernel.

Usage:
    python -m sglang.jit_kernel.benchmark.bench_deepseek_v4
    python -m sglang.jit_kernel.benchmark.bench_deepseek_v4 --variant flash
    python -m sglang.jit_kernel.benchmark.bench_deepseek_v4 --variant pro
    python -m sglang.jit_kernel.benchmark.bench_deepseek_v4 --kernel topk fused_rope
    python -m sglang.jit_kernel.benchmark.bench_deepseek_v4 --quick   # one shape per kernel
    python -m sglang.jit_kernel.benchmark.bench_deepseek_v4 --csv out.csv
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import triton

def _detect_hip() -> bool:
    """Detect HIP without importing sglang.srt (avoids heavy deps at parse)."""
    try:
        return bool(torch.version.hip)
    except Exception:
        return False


IS_HIP = _detect_hip()


# ---------------------------------------------------------------------------
# Model variants (from config.json)
# ---------------------------------------------------------------------------


@dataclass
class ModelVariant:
    name: str
    hidden_size: int
    head_dim: int                   # MLA head_dim (kv compress projects to this)
    qk_rope_head_dim: int
    num_attention_heads: int
    num_routed_experts: int
    n_shared_experts: int
    moe_intermediate_size: int
    num_experts_per_tok: int
    index_topk: int
    index_head_dim: int
    index_n_heads: int
    sliding_window: int
    q_lora_rank: int
    o_groups: int
    quant_block_size: int = 128


FLASH = ModelVariant(
    name="flash",
    hidden_size=4096,
    head_dim=512,
    qk_rope_head_dim=64,
    num_attention_heads=64,
    num_routed_experts=256,
    n_shared_experts=1,
    moe_intermediate_size=2048,
    num_experts_per_tok=6,
    index_topk=512,
    index_head_dim=128,
    index_n_heads=64,
    sliding_window=128,
    q_lora_rank=1024,
    o_groups=8,
)

PRO = ModelVariant(
    name="pro",
    hidden_size=7168,
    head_dim=512,
    qk_rope_head_dim=64,
    num_attention_heads=128,
    num_routed_experts=384,
    n_shared_experts=1,
    moe_intermediate_size=3072,
    num_experts_per_tok=6,
    index_topk=1024,
    index_head_dim=128,
    index_n_heads=64,
    sliding_window=128,
    q_lora_rank=1536,
    o_groups=16,
)

VARIANTS = {"flash": FLASH, "pro": PRO}


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


@dataclass
class Row:
    kernel: str
    variant: str
    shape: str
    backend: str
    us_median: float
    extra: str = ""


@dataclass
class Suite:
    rows: List[Row] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.rows.append(Row(**kw))

    def report(self) -> None:
        if not self.rows:
            print("(no rows recorded)")
            return
        by_kernel: dict = {}
        for r in self.rows:
            by_kernel.setdefault(r.kernel, []).append(r)

        print()
        for kernel, rs in by_kernel.items():
            print(f"### {kernel}")
            print("| variant | shape | backend | us | extra |")
            print("|---|---|---|---:|---|")
            for r in rs:
                us = "nan" if math.isnan(r.us_median) else f"{r.us_median:.2f}"
                print(
                    f"| {r.variant} | {r.shape} | {r.backend} | {us} | {r.extra} |"
                )
            print()


def _do_bench(fn: Callable[[], None], warmup: int = 25, rep: int = 100) -> float:
    """Return median runtime in microseconds."""
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    return float(ms) * 1000.0


def _device() -> torch.device:
    return torch.device("cuda")


# ---------------------------------------------------------------------------
# Torch references (apples-to-apples baselines for the ported kernels)
# ---------------------------------------------------------------------------


def _torch_ref_hash_topk(
    router_logits: torch.Tensor, input_id: torch.Tensor, tid2eid: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    routed_scaling_factor: float,
) -> None:
    """Vectorized torch baseline: gather → sqrt(softplus) → normalize."""
    B, E = router_logits.shape
    K = tid2eid.shape[1]
    KF = topk_ids.shape[1]
    NS = KF - K
    expert_ids = tid2eid[input_id.long()]                                  # [B, K]
    rl = torch.gather(router_logits, 1, expert_ids.long())                 # [B, K]
    softplus = torch.clamp(rl, min=0) + torch.log1p(torch.exp(-rl.abs()))
    routed_w = torch.sqrt(softplus)
    routed_w = routed_w / routed_w.sum(dim=-1, keepdim=True)
    if NS > 0:
        shared_eid = E + torch.arange(NS, device=router_logits.device,
                                      dtype=torch.int32).expand(B, NS)
        topk_ids.copy_(torch.cat([expert_ids.to(torch.int32), shared_eid], dim=-1))
        shared_w = torch.full((B, NS), 1.0 / routed_scaling_factor,
                              device=router_logits.device, dtype=torch.float32)
        topk_weights.copy_(torch.cat([routed_w, shared_w], dim=-1))
    else:
        topk_ids.copy_(expert_ids.to(torch.int32))
        topk_weights.copy_(routed_w)


def _torch_ref_paged_mqa_metadata(
    seq_lens: torch.Tensor, page_size: int, num_sm: int,
) -> torch.Tensor:
    """Naive python-loop baseline for paged_mqa_metadata (matches the
    sequential CUDA-kernel logic). Slow on purpose — measures what a
    no-vectorization torch implementation would cost."""
    assert page_size == 64
    kSplitKV = 256
    work = ((seq_lens.to(torch.int64) + kSplitKV - 1) // kSplitKV).tolist()
    B = len(work)
    total = sum(work)
    avg = total // num_sm
    ret = total % num_sm
    out = torch.zeros(num_sm + 1, 2, dtype=torch.int32, device=seq_lens.device)
    q = 0
    sum_work = work[0] if B else 0
    num_work = sum_work
    for i in range(num_sm + 1):
        target = i * avg + min(i, ret)
        while sum_work <= target:
            q += 1
            if q >= B:
                break
            num_work = work[q]
            sum_work += num_work
        if q >= B:
            out[i, 0] = B
            out[i, 1] = 0
        else:
            out[i, 0] = q
            out[i, 1] = target - (sum_work - num_work)
    return out


def _torch_ref_c4_decode_naive(
    kv_score_buffer: torch.Tensor, kv_score_input: torch.Tensor,
    kv_compressed_output: torch.Tensor, ape: torch.Tensor,
    indices: torch.Tensor, seq_lens: torch.Tensor,
    extra: torch.Tensor | None,
) -> None:
    """Naive python-loop torch baseline for c4 decode.

    Mirrors the kernel's algorithm element-wise: for each batch row, write to
    the ring buffer, then if seq_len % 4 == 0 gather 8 history slots and run
    softmax-weighted attention. The "vectorized" port batches across active
    rows; this naive version processes one row at a time.
    """
    page_size = 4 if extra is not None else 8
    B = kv_score_input.shape[0]
    head_dim = ape.shape[1]

    indices_l = indices.to(torch.long)
    seq_lens_l = seq_lens.to(torch.long)
    buf4 = kv_score_buffer.view(*kv_score_buffer.shape[:-1], 4, head_dim)

    for b in range(B):
        idx = int(indices_l[b].item())
        sl = int(seq_lens_l[b].item())
        write_pos = (sl - 1) % page_size
        kv_score_buffer[idx, write_pos] = kv_score_input[b]
        if sl % 4 != 0:
            continue
        kvs = []
        scs = []
        for slot in range(8):
            is_overlap = slot < 4
            kv_seg = 0 if is_overlap else 1
            sc_seg = 2 if is_overlap else 3
            if extra is None:
                k = (sl + slot) % 8
                kvs.append(buf4[idx, k, kv_seg].float())
                scs.append(buf4[idx, k, sc_seg].float())
            else:
                page = int(extra[b, 0].item()) if is_overlap else idx
                k = slot % 4
                kvs.append(buf4[page, k, kv_seg].float())
                scs.append(buf4[page, k, sc_seg].float())
        kv_stack = torch.stack(kvs)                       # [8, head_dim]
        sc_stack = torch.stack(scs) + ape.float()         # [8, head_dim]
        if sl == 4:
            sc_stack[:4] = -1e9
            kv_stack[:4] = 0.0
        weights = torch.softmax(sc_stack, dim=0)
        out = (weights * kv_stack).sum(dim=0)
        kv_compressed_output[b] = out.to(kv_compressed_output.dtype)


def _torch_ref_c128_decode_naive(
    kv_score_buffer: torch.Tensor, kv_score_input: torch.Tensor,
    kv_compressed_output: torch.Tensor, ape: torch.Tensor,
    indices: torch.Tensor, seq_lens: torch.Tensor,
) -> None:
    """Naive python-loop torch baseline for c128 decode."""
    B = kv_score_input.shape[0]
    head_dim = ape.shape[1]
    indices_l = indices.to(torch.long)
    seq_lens_l = seq_lens.to(torch.long)
    for b in range(B):
        idx = int(indices_l[b].item())
        sl = int(seq_lens_l[b].item())
        write_pos = (sl - 1) % 128
        kv_score_buffer[idx, write_pos] = kv_score_input[b]
        if sl % 128 != 0:
            continue
        buf = kv_score_buffer[idx].float()                 # [128, 2*head_dim]
        kv_all = buf[:, :head_dim]
        score_all = buf[:, head_dim:]
        scaled = score_all + ape.float()
        weights = torch.softmax(scaled, dim=0)
        out = (weights * kv_all).sum(dim=0)
        kv_compressed_output[b] = out.to(kv_compressed_output.dtype)


def _torch_ref_silu_mul_quant_transposed_ue8m0(
    input_: torch.Tensor, output: torch.Tensor, output_scale: torch.Tensor,
    quant_group_size: int, masked_m: torch.Tensor,
) -> None:
    """Vectorized torch baseline for the transposed+ue8m0 silu+mul+fp8 quant.
    Uses int32 (not uint32) bit ops because rshift_cuda for UInt32 is missing
    on HIP. Per-token scale dtype is int32-packed (4 ue8m0 / int32) into
    [E, G/4, T] layout — same as the production kernel's output."""
    FP8_MAX = 240.0
    E, T, D2 = input_.shape
    D = D2 // 2
    G = D // quant_group_size
    gate = input_[..., :D].to(torch.float32)
    up = input_[..., D:].to(torch.float32)
    silu = gate / (1.0 + torch.exp(-gate))
    silu_bf16 = silu.to(torch.bfloat16).to(torch.float32)
    val = up * silu_bf16                                       # [E, T, D]
    val_g = val.view(E, T, G, quant_group_size)                # group last dim
    abs_max = val_g.abs().amax(dim=-1).clamp(min=1e-10)        # [E, T, G]
    raw_scale = abs_max / FP8_MAX
    bits = raw_scale.view(torch.int32)
    exp_field = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    ue8m0 = (exp_field + (mant != 0).to(torch.int32)).to(torch.int32)  # [E, T, G]
    inv_scale_bits = (127 + 127 - ue8m0).to(torch.int32) << 23
    inv_scale = inv_scale_bits.view(torch.float32)
    quantized = (val_g * inv_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    fp8 = quantized.to(torch.float8_e4m3fn).view(E, T, D)
    output.copy_(fp8)
    # mask out unused tokens (tid >= masked_m[e]) — kernel skips them.
    # For perf comparison we don't need to exactly match (output is scratch
    # for those rows), but ue8m0 packing layout must match.
    # Pack 4 ue8m0 per int32 in [E, G/4, T] layout:
    #   out[e, g//4, t] |= ue8m0[e, t, g] << (8 * (g % 4))
    packed = output_scale
    packed.zero_()
    for k in range(4):
        # group dimension stride: take every 4th group starting at k
        ue8m0_k = ue8m0[:, :, k::4].to(torch.int32)                 # [E, T, G/4]
        # transpose to [E, G/4, T]
        packed |= (ue8m0_k.permute(0, 2, 1).contiguous() & 0xFF) << (8 * k)


def _torch_ref_topk_transform_512(
    scores: torch.Tensor, seq_lens: torch.Tensor, page_tables: torch.Tensor,
    out_page_indices: torch.Tensor, page_size: int,
) -> None:
    """Pure-torch baseline (works for any K, not just 512). Matches the kernel's
    page-table transform: page_idx = pos >> page_bits, offset = pos & page_mask,
    out = (page_table[page_idx] << page_bits) | offset. Invalid → -1.
    """
    B, L = scores.shape
    K = out_page_indices.shape[1]
    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    pos = torch.arange(L, device=scores.device).unsqueeze(0).expand(B, -1)
    valid = pos < seq_lens.unsqueeze(1)
    masked = torch.where(valid, scores, scores.new_full((), float("-inf")))
    actual_k = min(K, L)
    _, raw = torch.topk(masked, k=actual_k, dim=1, largest=True, sorted=False)
    raw = raw.to(torch.int32)
    if actual_k < K:
        pad = torch.full((B, K - actual_k), -1, dtype=torch.int32,
                         device=scores.device)
        raw = torch.cat([raw, pad], dim=1)
    bidx = torch.arange(B, device=scores.device).unsqueeze(1).expand(-1, K)
    gathered = scores[bidx.flatten(), raw.clamp(min=0).long().flatten()].view(B, K)
    valid_topk = gathered != float("-inf")
    page_idx = raw >> page_bits
    page_idx_clamped = torch.clamp(page_idx, min=0).long()
    phys = torch.gather(page_tables, dim=1, index=page_idx_clamped)
    out = (phys << page_bits) | (raw & page_mask)
    out_page_indices.copy_(torch.where(valid_topk, out.to(torch.int32),
                                       out.new_full((), -1)))


def _torch_ref_fused_rope(
    q: torch.Tensor, k: torch.Tensor,
    freqs_cis: torch.Tensor, positions: torch.Tensor,
) -> None:
    """In-place rope using torch complex math, mirroring the kernel."""
    rope_dim = q.shape[-1]
    pos = positions.long()
    freqs = freqs_cis[pos]  # [B, rope_dim/2] complex64
    for t in (q, k):
        if t is None:
            continue
        # [B, H, rope_dim] -> [B, H, rope_dim/2] complex
        x = t.to(torch.float32).view(*t.shape[:-1], rope_dim // 2, 2)
        xc = torch.complex(x[..., 0], x[..., 1])
        yc = xc * freqs.unsqueeze(1)
        out = torch.stack([yc.real, yc.imag], dim=-1).flatten(-2)
        t.copy_(out.to(t.dtype))


def _ue8m0_from_fp32(x: torch.Tensor) -> torch.Tensor:
    """Extract ue8m0 (rounded-up exponent) from fp32 — uses int32 math because
    PyTorch on HIP does not implement uint32 rshift (`rshift_cuda` for UInt32).
    """
    bits = x.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    return exp + (mant != 0).to(torch.int32)


def _torch_ref_fused_store_cache(
    input_: torch.Tensor, cache: torch.Tensor, indices: torch.Tensor,
    *, page_size: int, type_: str,
) -> None:
    """Apples-to-apples PyTorch baseline: per-token fp8_e4m3 quant + scatter
    into the same paged byte-cache layout the kernel writes.

    Vectorized over batch where possible. Uses int32 (not uint32) bit ops
    because PyTorch on HIP doesn't implement `rshift_cuda` for UInt32.
    """
    FP8_MAX = 240.0
    indices_l = indices.to(torch.long)
    pages = indices_l // page_size
    offsets = indices_l % page_size
    N = input_.shape[0]

    if type_ == "flashmla":
        x = input_.to(torch.float32).view(N, 8, 64)
        # Rows 0..6: fp8 with per-row ue8m0 scale
        r = x[:, :7, :]                              # [N, 7, 64]
        abs_max = torch.clamp(r.abs().amax(dim=2), min=1e-4)  # [N, 7]
        scale_raw = abs_max / FP8_MAX
        scale_ue8m0 = _ue8m0_from_fp32(scale_raw)              # int32 [N, 7]
        inv_scale_bits = (127 + 127 - scale_ue8m0).to(torch.int32) << 23
        inv_scale = inv_scale_bits.view(torch.float32)
        q = (r * inv_scale.unsqueeze(2)).clamp(-FP8_MAX, FP8_MAX)
        q_fp8 = q.to(torch.float8_e4m3fn).view(torch.uint8)    # [N, 7, 64]
        # Row 7: bf16 raw bytes
        bf16_bytes = x[:, 7, :].to(torch.bfloat16).view(torch.uint8).view(N, 128)
        scale_bytes = scale_ue8m0.to(torch.uint8)              # [N, 7]
        # Scatter (vectorized over N)
        for b in range(N):
            base = offsets[b] * 576
            cache[pages[b], base : base + 7 * 64] = q_fp8[b].reshape(-1)
            cache[pages[b], base + 448 : base + 576] = bf16_bytes[b]
            so = 576 * page_size + offsets[b] * 8
            cache[pages[b], so : so + 7] = scale_bytes[b]
    elif type_ == "indexer":
        x = input_.to(torch.float32)
        abs_max = torch.clamp(x.abs().amax(dim=1), min=1e-4)
        scale = abs_max / FP8_MAX
        inv_scale = 1.0 / scale
        q = (x * inv_scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX)
        q_fp8 = q.to(torch.float8_e4m3fn).view(torch.uint8)    # [N, 128]
        scale_bytes = scale.view(torch.float32).contiguous().view(
            torch.uint8).view(N, 4)
        for b in range(N):
            cache[pages[b], offsets[b] * 128 : offsets[b] * 128 + 128] = q_fp8[b]
            so = 128 * page_size + offsets[b] * 4
            cache[pages[b], so : so + 4] = scale_bytes[b]
    else:
        raise ValueError(f"unknown type {type_!r}")


def _torch_ref_swa_indices(
    seq_lens_k: torch.Tensor, seq_lens_q: torch.Tensor,
    swa_indices: torch.Tensor, cu_seqlens_q: torch.Tensor,
) -> None:
    """Vectorized torch reference for the TileLang SWA index builder.

    For each (token_id, j) we map the token to its seq via searchsorted on
    cu_seqlens_q, then compute the same conditional formula as the kernel.
    All ops are tensor-level — fair perf comparison against TileLang.
    """
    device = swa_indices.device
    B = seq_lens_q.shape[0]
    swa = swa_indices.shape[1]
    num_q = swa_indices.shape[0]

    # token_id -> seq_idx via prefix-sum bucket lookup
    token_ids = torch.arange(num_q, device=device, dtype=torch.int32)
    seq_idx = torch.searchsorted(cu_seqlens_q, token_ids, right=True) - 1
    seq_idx = seq_idx.clamp(min=0, max=B - 1)
    cum_qo_len = cu_seqlens_q[seq_idx]
    kv_len = seq_lens_k[seq_idx]
    qo_len = seq_lens_q[seq_idx]
    prefix_len = kv_len - qo_len
    curr_seq_qo_idx = token_ids - cum_qo_len
    end_abs = prefix_len + curr_seq_qo_idx + 1
    start_abs = torch.clamp(end_abs - swa, min=0)
    old_kv_start = seq_idx * swa
    new_kv_start = B * swa + cum_qo_len

    j_grid = torch.arange(swa, device=device, dtype=torch.int32)
    abs_pos = start_abs.unsqueeze(1) + j_grid.unsqueeze(0)        # [num_q, swa]
    end_abs_b = end_abs.unsqueeze(1).expand_as(abs_pos)
    prefix_b = prefix_len.unsqueeze(1).expand_as(abs_pos)
    old_b = old_kv_start.unsqueeze(1).expand_as(abs_pos)
    new_b = new_kv_start.unsqueeze(1).expand_as(abs_pos)

    in_old = abs_pos < prefix_b
    val_old = old_b + (abs_pos % swa)
    val_new = new_b + (abs_pos - prefix_b)
    val = torch.where(in_old, val_old, val_new)
    val = torch.where(abs_pos < end_abs_b, val,
                      torch.full_like(val, -1))
    swa_indices.copy_(val.to(swa_indices.dtype))


# ---------------------------------------------------------------------------
# Per-kernel benches
# ---------------------------------------------------------------------------


def bench_topk_transform_512(
    suite: Suite, variant: ModelVariant, decode_configs: List[Tuple[int, int]]
) -> None:
    """topk_transform_512: extract top-K positions per batch row over a paged page-table.

    Inputs:
      scores:           [B, max_seq_len] f32
      seq_lens:         [B] i32
      page_tables:      [B, num_pages] i32
      out_page_indices: [B, K] i32  (K = index_topk)

    Compares the ported Triton kernel (or jit-cuda on CUDA) against a pure-
    torch baseline that uses torch.topk + paged transform.
    """
    from sglang.jit_kernel.deepseek_v4 import topk_transform_512

    K = variant.index_topk
    page_size = 64  # c4 indexer page size for DSv4
    port_backend = "triton" if IS_HIP else "jit-cuda"

    for B, seq_len in decode_configs:
        device = _device()
        max_seq_len = seq_len
        num_pages = max(1, math.ceil(max_seq_len / page_size))
        scores = torch.randn(B, max_seq_len, dtype=torch.float32, device=device)
        seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
        page_tables = torch.randint(
            0, num_pages, (B, num_pages), dtype=torch.int32, device=device
        )
        out_port = torch.empty(B, K, dtype=torch.int32, device=device)
        out_torch = torch.empty(B, K, dtype=torch.int32, device=device)

        for backend, fn in (
            (port_backend, lambda: topk_transform_512(
                scores, seq_lens, page_tables, out_port, page_size)),
            ("torch", lambda: _torch_ref_topk_transform_512(
                scores, seq_lens, page_tables, out_torch, page_size)),
        ):
            try:
                fn()
                us = _do_bench(fn)
            except Exception as e:
                suite.add(
                    kernel="topk_transform_512", variant=variant.name,
                    shape=f"B={B} L={seq_len} K={K}", backend=backend,
                    us_median=float("nan"), extra=f"skip: {e.__class__.__name__}",
                )
                continue
            suite.add(
                kernel="topk_transform_512", variant=variant.name,
                shape=f"B={B} L={seq_len} K={K}", backend=backend, us_median=us,
            )


def bench_fused_rope(
    suite: Suite, variant: ModelVariant, decode_configs: List[Tuple[int, int]]
) -> None:
    """fused_rope: applies RoPE to Q ([B, H_q, D]) and K ([B, H_k, D]).

    Per deepseek_v4.py:
      q,k:       bfloat16
      freqs_cis: complex64
      positions: int32 or int64

    Triton fallback on HIP applies the rotation twice per call (once for Q,
    once for K). Workload doesn't depend on ctx — dedupe by B only.
    """
    from sglang.jit_kernel.deepseek_v4 import fused_rope

    rope_dim = variant.qk_rope_head_dim
    H_q = variant.num_attention_heads
    H_k = 1   # NSA-style: single key head per token
    backend = "triton" if IS_HIP else "jit-cuda"

    seen_B: set = set()
    for B, _ in decode_configs:
        if B in seen_B:
            continue
        seen_B.add(B)
        device = _device()
        max_pos = 65536
        q_orig = torch.randn(B, H_q, rope_dim, dtype=torch.bfloat16, device=device)
        k_orig = torch.randn(B, H_k, rope_dim, dtype=torch.bfloat16, device=device)
        freqs_real = torch.randn(
            max_pos, rope_dim // 2, dtype=torch.float32, device=device
        )
        freqs_cis = torch.complex(freqs_real, torch.randn_like(freqs_real))
        positions = torch.randint(0, max_pos, (B,),
                                  dtype=torch.int32, device=device)

        # Two clones for the two backends — fused_rope is in-place.
        q_port = q_orig.clone()
        k_port = k_orig.clone()
        q_torch = q_orig.clone()
        k_torch = k_orig.clone()

        for be, fn in (
            (backend, lambda: fused_rope(q_port, k_port, freqs_cis, positions)),
            ("torch", lambda: _torch_ref_fused_rope(
                q_torch, k_torch, freqs_cis, positions)),
        ):
            try:
                fn()
                us = _do_bench(fn)
            except Exception as e:
                suite.add(
                    kernel="fused_rope", variant=variant.name,
                    shape=f"B={B} H_q={H_q} H_k={H_k} D={rope_dim}",
                    backend=be, us_median=float("nan"),
                    extra=f"skip: {e.__class__.__name__}",
                )
                continue
            suite.add(
                kernel="fused_rope", variant=variant.name,
                shape=f"B={B} H_q={H_q} H_k={H_k} D={rope_dim}",
                backend=be, us_median=us,
            )


def bench_triton_paged_compress_data(
    suite: Suite, variant: ModelVariant, prefill_configs: List[Tuple[int, int]]
) -> None:
    """triton_create_paged_compress_data: per-batch index transforms for paged
    compression. Pure Triton — works on HIP.
    """
    from sglang.jit_kernel.deepseek_v4 import triton_create_paged_compress_data

    swa_page_size = variant.sliding_window
    ring_size = 16  # c4 ring size from get_compress_state_ring_size

    for B, num_q_tokens in prefill_configs:
        device = _device()
        A = max(1024, B * 4)
        Bdim = max(1024, num_q_tokens * 4)
        req_pool_indices = torch.arange(B, dtype=torch.int32, device=device)
        seq_lo = max(1, num_q_tokens // max(1, B))
        seq_hi = max(seq_lo + 64, 128)
        seq_lens = torch.randint(seq_lo, seq_hi, (B,),
                                 dtype=torch.int32, device=device)
        ext_hi = max(2, num_q_tokens // max(1, B) + 1)
        extend_seq_lens = torch.randint(1, ext_hi, (B,),
                                        dtype=torch.int32, device=device)
        # extend_seq_lens cannot exceed seq_lens elementwise
        extend_seq_lens = torch.minimum(extend_seq_lens, seq_lens)
        req_to_token = torch.randint(0, A * Bdim, (A, Bdim),
                                     dtype=torch.int32, device=device)
        full_to_swa_index_mapping = torch.randint(
            0, A * Bdim, (A * Bdim,), dtype=torch.int32, device=device
        )

        for compress_ratio, is_overlap in [(4, True), (128, False)]:
            try:
                def fn(cr=compress_ratio, ov=is_overlap):
                    triton_create_paged_compress_data(
                        compress_ratio=cr,
                        is_overlap=ov,
                        swa_page_size=swa_page_size,
                        ring_size=ring_size,
                        req_pool_indices=req_pool_indices,
                        seq_lens=seq_lens,
                        extend_seq_lens=extend_seq_lens,
                        req_to_token=req_to_token,
                        full_to_swa_index_mapping=full_to_swa_index_mapping,
                    )
                us = _do_bench(fn)
            except Exception as e:
                suite.add(
                    kernel="paged_compress_data", variant=variant.name,
                    shape=f"B={B} q={num_q_tokens} cr={compress_ratio}",
                    backend="triton", us_median=float("nan"),
                    extra=f"skip: {e.__class__.__name__}",
                )
                continue
            suite.add(
                kernel="paged_compress_data", variant=variant.name,
                shape=f"B={B} q={num_q_tokens} cr={compress_ratio} ov={int(is_overlap)}",
                backend="triton", us_median=us,
            )


def bench_tilelang_swa_indices(
    suite: Suite, variant: ModelVariant, prefill_configs: List[Tuple[int, int]]
) -> None:
    """tilelang_make_swa_prefill_indices: SWA neighbor-table builder."""
    try:
        from sglang.jit_kernel.deepseek_v4 import tilelang_make_swa_prefill_indices
    except Exception as e:
        suite.add(
            kernel="swa_prefill_indices", variant=variant.name, shape="-",
            backend="-", us_median=float("nan"),
            extra=f"skip: import failed ({e.__class__.__name__})",
        )
        return

    swa = variant.sliding_window
    for B, num_q_tokens in prefill_configs:
        device = _device()
        per = max(1, num_q_tokens // max(1, B))
        seq_lens_q = torch.full((B,), per, dtype=torch.int32, device=device)
        diff = num_q_tokens - int(seq_lens_q.sum().item())
        if B > 0:
            seq_lens_q[-1] = max(1, int(seq_lens_q[-1].item()) + diff)
        seq_lens_k = seq_lens_q + torch.randint(
            0, 4096, (B,), dtype=torch.int32, device=device
        )
        cu_seqlens_q = torch.cumsum(seq_lens_q, dim=0, dtype=torch.int32)
        cu_seqlens_q = torch.nn.functional.pad(cu_seqlens_q, (1, 0), value=0)
        actual_q = int(cu_seqlens_q[-1].item())
        swa_indices = torch.empty(actual_q, swa, dtype=torch.int32, device=device)

        swa_torch = torch.empty_like(swa_indices)
        for be, fn in (
            ("tilelang", lambda: tilelang_make_swa_prefill_indices(
                seq_lens_k, seq_lens_q, swa_indices, cu_seqlens_q)),
            ("torch", lambda: _torch_ref_swa_indices(
                seq_lens_k, seq_lens_q, swa_torch, cu_seqlens_q)),
        ):
            try:
                fn()
                us = _do_bench(fn, warmup=5, rep=20)
                suite.add(
                    kernel="swa_prefill_indices", variant=variant.name,
                    shape=f"B={B} q={actual_q} swa={swa}",
                    backend=be, us_median=us,
                )
            except Exception as e:
                suite.add(
                    kernel="swa_prefill_indices", variant=variant.name,
                    shape=f"B={B} q={actual_q}",
                    backend=be, us_median=float("nan"),
                    extra=f"skip: {e.__class__.__name__}",
                )


def bench_linear_bf16_fp32(
    suite: Suite, variant: ModelVariant, decode_configs: List[Tuple[int, int]]
) -> None:
    """linear_bf16_fp32: bf16 x bf16 -> fp32 GEMM (used for indexer scores).

    Per deepseek_v4.py:
      x, y: bfloat16
      out:  float32

    Algos: cublas/deep_gemm on CUDA, torch fp32 on HIP. Workload depends on M
    (= num_q_tokens), so dedupe by B only.
    """
    from sglang.jit_kernel.deepseek_v4 import linear_bf16_fp32

    K_dim = variant.index_n_heads * variant.index_head_dim  # 8192
    out_dim = variant.index_topk

    algos = ["torch"] if IS_HIP else ["cublas", "deep_gemm", "torch"]

    seen_B: set = set()
    for B, _ in decode_configs:
        if B in seen_B:
            continue
        seen_B.add(B)
        device = _device()
        x = torch.randn(B, K_dim, dtype=torch.bfloat16, device=device)
        w = torch.randn(out_dim, K_dim, dtype=torch.bfloat16, device=device)

        for algo in algos:
            os.environ["SGLANG_OPT_BF16_FP32_GEMM_ALGO"] = algo
            try:
                def fn():
                    linear_bf16_fp32(x, w)
                fn()  # warm
                us = _do_bench(fn)
                suite.add(
                    kernel="linear_bf16_fp32", variant=variant.name,
                    shape=f"M={B} K={K_dim} N={out_dim}",
                    backend=algo, us_median=us,
                )
            except Exception as e:
                suite.add(
                    kernel="linear_bf16_fp32", variant=variant.name,
                    shape=f"M={B} K={K_dim} N={out_dim}",
                    backend=algo, us_median=float("nan"),
                    extra=f"skip: {e.__class__.__name__}: {str(e)[:64]}",
                )


def bench_hash_topk(
    suite: Suite, variant: ModelVariant, decode_configs: List[Tuple[int, int]]
) -> None:
    """hash_topk: fused MoE topk + sqrtsoftplus + share-expert append."""
    from sglang.jit_kernel.deepseek_v4 import hash_topk
    backend = "triton" if IS_HIP else "jit-cuda"

    topk_routed = variant.num_experts_per_tok
    num_routed = variant.num_routed_experts
    num_shared = variant.n_shared_experts
    vocab = 129280

    seen_B: set = set()
    for B, _ in decode_configs:
        if B in seen_B:
            continue
        seen_B.add(B)
        device = _device()
        router_logits = torch.randn(B, num_routed, dtype=torch.float32, device=device)
        input_ids = torch.randint(0, vocab, (B,), dtype=torch.int64, device=device)
        tid2eid = torch.randint(
            0, num_routed, (vocab, topk_routed), dtype=torch.int32, device=device
        )
        topk_fused = topk_routed + num_shared

        for be, fn in (
            (backend, lambda: hash_topk(
                router_logits=router_logits, input_ids=input_ids,
                tid2eid=tid2eid, num_fused_shared_experts=num_shared,
                routed_scaling_factor=1.5, scoring_func="sqrtsoftplus",
            )),
            ("torch", None),  # placeholder; built per-iteration below
        ):
            if be == "torch":
                tw = torch.empty(B, topk_fused, dtype=torch.float32, device=device)
                ti = torch.empty(B, topk_fused, dtype=torch.int32, device=device)
                fn = lambda: _torch_ref_hash_topk(
                    router_logits, input_ids, tid2eid, tw, ti, 1.5,
                )
            try:
                fn()
                us = _do_bench(fn)
            except Exception as e:
                suite.add(
                    kernel="hash_topk", variant=variant.name,
                    shape=f"B={B} E={num_routed} k={topk_routed}",
                    backend=be, us_median=float("nan"),
                    extra=f"skip: {e.__class__.__name__}",
                )
                continue
            suite.add(
                kernel="hash_topk", variant=variant.name,
                shape=f"B={B} E={num_routed} k={topk_routed}",
                backend=be, us_median=us,
            )


def bench_fused_store_cache(
    suite: Suite, variant: ModelVariant, decode_configs: List[Tuple[int, int]]
) -> None:
    """fused_store_cache: bf16/fp32 -> fp8 quant + scatter into paged KV cache.

    Production input dtypes (from compressor.py / deepseek_v4_backend_radix.py):
      flashmla — bf16  (raw key activations from attention output)
      indexer  — fp32  (post-RoPE compressed kv from compress_forward)

    Triton port (fused_store_cache_triton.py) accepts bf16/fp16/fp32 and casts
    internally; CUDA JIT templates on input.dtype. Workload depends on N=B
    only — dedupe by B.
    """
    from sglang.jit_kernel.deepseek_v4 import fused_store_cache

    backend = "triton" if IS_HIP else "jit-cuda"

    for kind, item, page_size, page_bytes_per_token, in_dtype in [
        ("flashmla", 512, 256, 584, torch.bfloat16),
        ("indexer",  128, 64,  132, torch.float32),
    ]:
        seen_B: set = set()
        for B, _ in decode_configs:
            if B in seen_B:
                continue
            seen_B.add(B)
            device = _device()
            # cache must hold all token indices in [0, num_pages*page_size)
            num_pages = max(8, math.ceil(B / max(1, page_size)) + 8)
            input_ = torch.randn(B, item, dtype=in_dtype, device=device)
            cache = torch.zeros(
                num_pages, page_size * page_bytes_per_token,
                dtype=torch.uint8, device=device,
            )
            indices = torch.randint(
                0, num_pages * page_size, (B,),
                dtype=torch.int32, device=device,
            )
            cache_torch = cache.clone()
            for be, fn in (
                (backend, lambda: fused_store_cache(
                    input=input_, cache=cache, indices=indices,
                    page_size=page_size, type=kind)),
                ("torch", lambda: _torch_ref_fused_store_cache(
                    input_, cache_torch, indices,
                    page_size=page_size, type_=kind)),
            ):
                try:
                    fn()  # warm + verify
                    us = _do_bench(fn, warmup=10, rep=50)
                except Exception as e:
                    suite.add(
                        kernel="fused_store_cache", variant=variant.name,
                        shape=f"{kind} B={B} item={item}", backend=be,
                        us_median=float("nan"),
                        extra=f"skip: {e.__class__.__name__}: {str(e)[:64]}",
                    )
                    continue
                suite.add(
                    kernel="fused_store_cache", variant=variant.name,
                    shape=f"{kind} B={B} dtype={str(in_dtype).split('.')[-1]}",
                    backend=be, us_median=us,
                )


def bench_silu_and_mul_masked_post_quant(
    suite: Suite, variant: ModelVariant, prefill_configs: List[Tuple[int, int]]
) -> None:
    """silu_and_mul_masked_post_quant: gated-MoE silu+mul with per-group fp8 quant.

    Per srt/layers/moe/moe_runner/deep_gemm.py:599-672 and
    csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh:33-42
    the production DSv4 call shape is:
      input  (gateup_output): bf16 [E, N, 2*moe_inter]
      output (down_input):    fp8_e4m3fn [E, N, moe_inter]
      output_scale:           int32 [E, G/4, N]   (transposed=True, ue8m0)
      masked_m:               int32 [E]
      group_size: 128, scale_ue8m0=True, transposed=True
      topk = num_experts_per_tok

    Note on mxfp4: DSv4-Flash's hybrid mxfp4 checkpoint is a *weight-only*
    quantization (per-expert weights are int4-packed with e8m0 scales — see
    srt/models/deepseek_v4.py:_try_consume_mxfp4_expert_weight). The activation
    pipeline through this kernel still produces fp8_e4m3fn; this kernel does
    not have a separate mxfp4 mode.
    """
    from sglang.jit_kernel.deepseek_v4 import silu_and_mul_masked_post_quant
    backend = "triton" if IS_HIP else "jit-cuda"

    G = variant.quant_block_size
    inter = variant.moe_intermediate_size
    hidden_dim = 2 * inter
    expert_num = 32  # typical per-rank EP
    topk = variant.num_experts_per_tok

    for token_pad, _ in prefill_configs:
        token_num_padded = max(G * 4, token_pad)  # need N % 4 == 0 for transposed
        if token_num_padded % 4:
            token_num_padded += 4 - (token_num_padded % 4)
        device = _device()
        input_ = torch.randn(
            expert_num, token_num_padded, hidden_dim,
            dtype=torch.bfloat16, device=device,
        )
        out = torch.empty(
            expert_num, token_num_padded, inter,
            dtype=torch.float8_e4m3fn, device=device,
        )
        # Production layout: int32 packed scales, transposed: [E, G/4, N]
        groups = inter // G
        assert groups % 4 == 0, f"groups={groups} must be % 4 == 0"
        out_scale = torch.empty(
            expert_num, groups // 4, token_num_padded,
            dtype=torch.int32, device=device,
        )
        masked_m = torch.randint(
            1, token_num_padded, (expert_num,),
            dtype=torch.int32, device=device,
        )

        out_torch = torch.empty_like(out)
        out_scale_torch = torch.empty_like(out_scale)
        for be, fn in (
            (backend, lambda: silu_and_mul_masked_post_quant(
                input_, out, out_scale, G, masked_m,
                scale_ue8m0=True, topk=topk, transposed=True,
            )),
            ("torch", lambda: _torch_ref_silu_mul_quant_transposed_ue8m0(
                input_, out_torch, out_scale_torch, G, masked_m,
            )),
        ):
            try:
                fn()
                us = _do_bench(fn, warmup=10, rep=50)
            except Exception as e:
                suite.add(
                    kernel="silu_mul_quant", variant=variant.name,
                    shape=f"E={expert_num} T={token_num_padded} D={hidden_dim}",
                    backend=be, us_median=float("nan"),
                    extra=f"skip: {e.__class__.__name__}",
                )
                continue
            suite.add(
                kernel="silu_mul_quant", variant=variant.name,
                shape=f"E={expert_num} T={token_num_padded} D={hidden_dim} G={G} ue8m0+T",
                backend=be, us_median=us,
            )


def bench_paged_mqa_metadata(
    suite: Suite, variant: ModelVariant, decode_configs: List[Tuple[int, int]]
) -> None:
    """get_paged_mqa_logits_metadata: per-SM work-balancer for paged MQA."""
    from sglang.jit_kernel.deepseek_v4 import get_paged_mqa_logits_metadata
    backend = "torch" if IS_HIP else "jit-cuda"

    num_sm = 256 if IS_HIP else 132   # MI355X CU count vs H100/H200 SM count

    seen_B: set = set()
    for B, ctx in decode_configs:
        if B in seen_B:
            continue
        seen_B.add(B)
        device = _device()
        seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
        for be, fn in (
            (backend, lambda: get_paged_mqa_logits_metadata(seq_lens, 64, num_sm)),
            ("torch-naive", lambda: _torch_ref_paged_mqa_metadata(
                seq_lens, 64, num_sm)),
        ):
            try:
                fn()
                us = _do_bench(fn, warmup=10, rep=50)
            except Exception as e:
                suite.add(
                    kernel="paged_mqa_metadata", variant=variant.name,
                    shape=f"B={B} L={ctx}", backend=be,
                    us_median=float("nan"),
                    extra=f"skip: {e.__class__.__name__}",
                )
                continue
            suite.add(
                kernel="paged_mqa_metadata", variant=variant.name,
                shape=f"B={B} L={ctx} SMs={num_sm}",
                backend=be, us_median=us,
            )


def bench_compress_forward(
    suite: Suite, variant: ModelVariant, prefill_configs: List[Tuple[int, int]]
) -> None:
    """compress_forward: c4 / c128 KV compression — decode path.

    HIP path uses a torch fallback (softmax-weighted attention over 8 / 128
    history slots). Production DSv4-Flash on HIP doesn't exercise this — the
    bench is for completeness / regression detection.
    """
    from sglang.jit_kernel.deepseek_v4 import (
        CompressorDecodePlan, compress_forward,
    )
    backend = "torch" if IS_HIP else "jit-cuda"

    # Decode-shaped: B = num batches, head_dim = compressed kv lane (Pro/Flash 128 typically)
    # The compress_forward signature for decode wants:
    #   kv_score_buffer: [num_indices, page_size, head_dim*K]
    #     where K=4 for c4 and K=2 for c128, page_size=8 for c4 RingBuffer / 128 for c128
    #   kv_score_input:  [B, head_dim*K]
    #   plan = CompressorDecodePlan(compress_ratio, seq_lens [B] int32)
    head_dim = 128                      # c-block compressed head dim
    for B, _ in prefill_configs:
        for compress_ratio in (4, 128):
            device = _device()
            num_indices = max(8, B + 4)
            page_size = 8 if compress_ratio == 4 else 128
            K = 4 if compress_ratio == 4 else 2
            kv_score_buffer = torch.randn(
                num_indices, page_size, head_dim * K,
                dtype=torch.float32, device=device,
            )
            kv_score_input = torch.randn(
                B, head_dim * K, dtype=torch.float32, device=device,
            )
            ape = torch.randn(page_size, head_dim,
                              dtype=torch.float32, device=device)
            indices = torch.randint(0, num_indices, (B,),
                                    dtype=torch.int32, device=device)
            # All seq_lens % compress_ratio == 0 → trigger compression on every
            # batch row (worst case for HIP torch path).
            seq_lens = torch.full(
                (B,), compress_ratio * 4, dtype=torch.int32, device=device,
            )
            plan = CompressorDecodePlan(compress_ratio, seq_lens)
            buf_naive = kv_score_buffer.clone()
            out_naive = torch.empty(B, head_dim,
                                    dtype=torch.float32, device=device)

            def port_fn():
                compress_forward(
                    kv_score_buffer=kv_score_buffer,
                    kv_score_input=kv_score_input, ape=ape,
                    indices=indices, plan=plan,
                    compress_ratio=compress_ratio, head_dim=head_dim,
                )

            def naive_fn(cr=compress_ratio):
                if cr == 4:
                    _torch_ref_c4_decode_naive(
                        buf_naive, kv_score_input, out_naive, ape,
                        indices, seq_lens, None,
                    )
                else:
                    _torch_ref_c128_decode_naive(
                        buf_naive, kv_score_input, out_naive, ape,
                        indices, seq_lens,
                    )

            for be, fn in (
                (backend, port_fn),
                ("torch-naive", naive_fn),
            ):
                try:
                    fn()
                    us = _do_bench(fn, warmup=5, rep=20)
                except Exception as e:
                    suite.add(
                        kernel="compress_forward", variant=variant.name,
                        shape=f"B={B} cr={compress_ratio} D={head_dim}",
                        backend=be, us_median=float("nan"),
                        extra=f"skip: {e.__class__.__name__}",
                    )
                    continue
                suite.add(
                    kernel="compress_forward", variant=variant.name,
                    shape=f"B={B} cr={compress_ratio} D={head_dim} ring",
                    backend=be, us_median=us,
                )


# ---------------------------------------------------------------------------
# fused_norm_rope (mode=2, DefaultForward)
# ---------------------------------------------------------------------------


def _torch_ref_fused_norm_rope_default(
    kv: torch.Tensor, weight: torch.Tensor, eps: float,
    freqs_cis: torch.Tensor, positions: torch.Tensor, rope_dim: int,
) -> None:
    """Vectorized torch baseline: RMSNorm + RoPE on the last `rope_dim` lanes,
    in-place. Mirrors the kernel's DefaultForward (mode=2) contract.
    """
    head_dim = kv.shape[-1]
    x = kv.float()
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x * rms * weight.float()
    nope_len = head_dim - rope_dim
    rope = y[:, nope_len:].view(*y.shape[:-1], rope_dim // 2, 2)
    rc = torch.complex(rope[..., 0], rope[..., 1])
    fc = freqs_cis[positions.long()]
    out = rc * fc
    rope_out = torch.stack([out.real, out.imag], dim=-1).flatten(-2)
    y[:, nope_len:] = rope_out
    kv.copy_(y.to(kv.dtype))


def bench_fused_norm_rope(
    suite: Suite, variant: ModelVariant, decode_configs: List[Tuple[int, int]]
) -> None:
    """fused_norm_rope (DefaultForward mode=2): RMSNorm + RoPE on the last
    rope_dim lanes of [N, head_dim]. Compares Triton port vs vectorized torch.
    """
    from sglang.jit_kernel.deepseek_v4 import fused_norm_rope_inplace

    rope_dim = variant.qk_rope_head_dim
    head_dim = variant.head_dim         # MLA head_dim (compressed kv lane)
    backend = "triton" if IS_HIP else "jit-cuda"

    seen_B: set = set()
    for B, _ in decode_configs:
        if B in seen_B:
            continue
        seen_B.add(B)
        device = _device()
        kv_orig = torch.randn(B, head_dim, dtype=torch.bfloat16, device=device)
        weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
        freqs_real = torch.randn(2048, rope_dim // 2, dtype=torch.float32, device=device)
        freqs_cis = torch.complex(freqs_real, torch.randn_like(freqs_real))
        positions = torch.randint(0, 2048, (B,), dtype=torch.int64, device=device)

        kv_port = kv_orig.clone()
        kv_torch = kv_orig.clone()
        # `fused_norm_rope_inplace` flattens freqs_cis internally; the torch ref
        # works directly on the complex tensor.

        for be, fn in (
            (backend, lambda: fused_norm_rope_inplace(
                kv_port, weight, 1e-6, freqs_cis, positions)),
            ("torch", lambda: _torch_ref_fused_norm_rope_default(
                kv_torch, weight, 1e-6, freqs_cis, positions, rope_dim)),
        ):
            try:
                fn()
                us = _do_bench(fn)
            except Exception as e:
                suite.add(
                    kernel="fused_norm_rope", variant=variant.name,
                    shape=f"B={B} D={head_dim} rope={rope_dim}",
                    backend=be, us_median=float("nan"),
                    extra=f"skip: {e.__class__.__name__}",
                )
                continue
            suite.add(
                kernel="fused_norm_rope", variant=variant.name,
                shape=f"B={B} D={head_dim} rope={rope_dim}",
                backend=be, us_median=us,
            )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


KERNELS: dict = {
    "topk":              bench_topk_transform_512,
    "fused_rope":        bench_fused_rope,
    "paged_compress":    bench_triton_paged_compress_data,
    "swa_indices":       bench_tilelang_swa_indices,
    "linear":            bench_linear_bf16_fp32,
    "hash_topk":         bench_hash_topk,
    "fused_store":       bench_fused_store_cache,
    "silu_mul_quant":    bench_silu_and_mul_masked_post_quant,
    "paged_mqa_meta":    bench_paged_mqa_metadata,
    "compress_forward":  bench_compress_forward,
    "fused_norm_rope":   bench_fused_norm_rope,
}

_PREFILL_KERNELS = {
    bench_triton_paged_compress_data,
    bench_tilelang_swa_indices,
    bench_silu_and_mul_masked_post_quant,
    bench_compress_forward,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=["flash", "pro", "both"], default="both",
        help="model variant",
    )
    parser.add_argument(
        "--kernel", nargs="+",
        choices=list(KERNELS.keys()) + ["all"], default=["all"],
        help="kernels to bench",
    )
    parser.add_argument("--quick", action="store_true",
                        help="one shape per kernel")
    parser.add_argument("--csv", type=str, default=None,
                        help="optional path to write rows as CSV")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA/HIP runtime not available; aborting.", file=sys.stderr)
        return 1

    name = torch.cuda.get_device_name(0) or "(unnamed)"
    cap = ""
    try:
        cap = f" cc={torch.cuda.get_device_capability(0)}"
    except Exception:
        pass
    print(f"Device: {name}{cap}  is_hip={IS_HIP}  device_count={torch.cuda.device_count()}")
    print()

    if args.quick:
        decode_configs = [(8, 4096)]
        prefill_configs = [(2, 1024)]
    else:
        decode_configs = [
            (1, 1024), (1, 16384),
            (8, 4096), (8, 16384),
            (32, 4096), (32, 16384),
        ]
        prefill_configs = [
            (1, 512), (2, 1024), (4, 4096),
        ]

    if args.variant == "both":
        variants = [FLASH, PRO]
    else:
        variants = [VARIANTS[args.variant]]

    if args.kernel == ["all"] or "all" in args.kernel:
        selected = list(KERNELS.values())
    else:
        selected = [KERNELS[k] for k in args.kernel]

    suite = Suite()
    for variant in variants:
        for fn in selected:
            print(f"==> {variant.name} :: {fn.__name__}")
            try:
                cfgs = prefill_configs if fn in _PREFILL_KERNELS else decode_configs
                fn(suite, variant, cfgs)
            except Exception as e:
                suite.add(
                    kernel=fn.__name__, variant=variant.name, shape="-",
                    backend="-", us_median=float("nan"),
                    extra=f"crashed: {e.__class__.__name__}: {e}",
                )

    suite.report()

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["kernel", "variant", "shape", "backend", "us_median",
                        "extra"])
            for r in suite.rows:
                us = "nan" if math.isnan(r.us_median) else f"{r.us_median:.3f}"
                w.writerow([r.kernel, r.variant, r.shape, r.backend, us, r.extra])
        print(f"wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
