from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.debug_utils.deepseek_v4_debug_utils import deepseek_v4_moe_code_path_checker

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def make_name(name: str) -> str:
    return f"dpsk_v4_{name}"


@cache_once
def _jit_common_module() -> Module:
    return load_jit(
        make_name(f"common"),
        cuda_files=[f"deepseek_v4/common.cuh"],
        cuda_wrappers=[("plan_compress_prefill", "plan_compress_prefill")],
    )


@cache_once
def _jit_topk_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk"),
        *args,
        cuda_files=["deepseek_v4/topk.cuh"],
        cuda_wrappers=[("topk_transform", f"TopK512Kernel<{args}>::transform")],
    )


@cache_once
def _jit_topk_v2_module() -> Module:
    return load_jit(
        make_name("topk_v2"),
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=[("topk_transform", "TopK512Kernel::transform")],
    )


@cache_once
def _jit_hash_topk_module() -> Module:
    args = make_cpp_args("act_sqrt_softplus", is_arch_support_pdl())
    return load_jit(
        make_name("hash_topk"),
        *args,
        cuda_files=["deepseek_v4/hash_topk.cuh"],
        cuda_wrappers=[("hash_topk", f"HashTopKKernel<{args}>::run")],
    )


@cache_once
def _jit_compress_module(
    head_dim: int,
    dtype_in: torch.dtype,
    dtype_out: torch.dtype,
    ratio: Literal[4, 128],
) -> Module:
    args = make_cpp_args(head_dim, dtype_in, dtype_out, is_arch_support_pdl())
    kernel_class = f"FlashCompress{ratio}Kernel<{args}>"
    return load_jit(
        make_name(f"compress_{ratio}"),
        *args,
        cuda_files=[f"deepseek_v4/c{ratio}.cuh"],
        cuda_wrappers=[
            ("decode", f"{kernel_class}::run_decode"),
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
    )


@cache_once
def _jit_fused_rope_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("fused_rope"),
        *args,
        cuda_files=["deepseek_v4/rope.cuh"],
        cuda_wrappers=[("forward", f"FusedQKRopeKernel<{args}>::forward")],
    )


@cache_once
def _jit_norm_rope_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
) -> Module:
    args = make_cpp_args(dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        make_name(f"fused_norm_rope"),
        *args,
        cuda_files=[f"deepseek_v4/fused_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_fused_store_module(
    name: Literal["flashmla", "indexer"],
    input_dtype: torch.dtype,
    index_dtype: torch.dtype,
    page_size: int,
) -> Module:
    args = make_cpp_args(input_dtype, index_dtype, page_size, is_arch_support_pdl())
    cname = "FlashMLA" if name == "flashmla" else "Indexer"
    kernel_class = f"FusedStoreCache{cname}Kernel<{args}>"
    return load_jit(
        make_name("store_" + name),
        *args,
        cuda_files=["deepseek_v4/store.cuh"],
        cuda_wrappers=[("run", f"{kernel_class}::run")],
    )


@cache_once
def _jit_metadata_module():
    return load_jit(
        make_name("metadata"),
        cuda_files=["deepseek_v4/paged_mqa_metadata.cuh"],
        cuda_wrappers=[("run", "IndexerMetadataKernel::run")],
    )


_TOPK_HIP_FALLBACK_WARNED = False


def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
    ver: Literal[1, 2] = 1,
) -> None:
    """Output to page_indices tensor, optionally also output raw abs position indices.

    On ROCm/HIP the JIT-CUDA path is unavailable (`tvm_ffi.cpp.load_inline`
    requires `CUDA_HOME`/`nvcc` and has no HIP build path). To make
    `SGLANG_TOPK_TRANSFORM_512_TORCH=0` not crash on HIP, we transparently
    fall back to the vectorized torch implementation. The torch path is
    capture-safe after the `scores.clone()` -> `torch.where` fix at
    indexer.py:271.

    TODO(rocm): write a Triton port of TopK512Kernel (it runs natively on
    AMDGCN without needing nvcc) and wire it in here as a third option,
    e.g. via `SGLANG_TOPK_TRANSFORM_512_BACKEND={torch,jit,triton}`. Hook
    point: dispatch on `is_hip()` below.
    """
    from sglang.srt.utils import is_hip

    if is_hip():
        # Native HIP path: Triton port at jit_kernel/topk_transform_512_triton.py.
        # Microbenched 4-9x faster than the torch fallback at DSv4 decode shapes
        # (B=1..16, L=256..4096); see aiter_capture_repro/bench_topk_triton.py.
        # Set SGLANG_TOPK_TRANSFORM_512_BACKEND=torch to force the torch path
        # (e.g., for debugging numerical mismatches).
        import os
        backend = os.environ.get("SGLANG_TOPK_TRANSFORM_512_BACKEND", "triton").lower()
        if backend == "triton":
            from sglang.jit_kernel.topk_transform_512_triton import (
                topk_transform_512_triton,
            )
            topk_transform_512_triton(
                scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices,
            )
            return
        # torch fallback (lazy import to avoid circular dependency).
        global _TOPK_HIP_FALLBACK_WARNED
        if not _TOPK_HIP_FALLBACK_WARNED:
            import warnings
            warnings.warn(
                "[HIP] topk_transform_512 forced to torch path via "
                "SGLANG_TOPK_TRANSFORM_512_BACKEND=torch (8x slower than the "
                "Triton port). Unset to use the Triton kernel.",
                RuntimeWarning,
            )
            _TOPK_HIP_FALLBACK_WARNED = True
        from sglang.srt.layers.attention.compressed.indexer import (
            topk_transform_512_pytorch_vectorized,
        )
        topk_transform_512_pytorch_vectorized(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices,
        )
        return

    module = _jit_topk_v2_module() if ver == 2 else _jit_topk_module()
    module.topk_transform(
        scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
    )


def hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert scoring_func == "sqrtsoftplus"
    num_tokens = router_logits.size(0)
    topk_routed = tid2eid.size(1)
    topk_fused = topk_routed + num_fused_shared_experts
    topk_ids = torch.empty(
        (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
    )
    topk_weights = torch.empty(
        (num_tokens, topk_fused), dtype=torch.float32, device=router_logits.device
    )
    from sglang.srt.utils import is_hip
    if is_hip():
        from sglang.jit_kernel.hash_topk_triton import hash_topk_triton
        hash_topk_triton(
            router_logits, input_ids, tid2eid,
            topk_weights, topk_ids, routed_scaling_factor,
        )
        return topk_weights, topk_ids
    module = _jit_hash_topk_module()
    module.hash_topk(
        router_logits,
        input_ids,
        tid2eid,
        topk_weights,
        topk_ids,
        routed_scaling_factor,
    )
    return topk_weights, topk_ids


class CompressorPrefillPlan(NamedTuple):
    compress_ratio: int
    compress_plan: torch.Tensor
    write_plan: torch.Tensor

    def copy_(self, other: CompressorPrefillPlan) -> None:
        assert self.compress_ratio == other.compress_ratio
        self.compress_plan.copy_(other.compress_plan)
        self.write_plan.copy_(other.write_plan)

    @staticmethod
    def generate(
        compress_ratio: Literal[4, 128],
        num_q_tokens: int,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        device: torch.device,
        use_cuda_graph: bool = False,
    ) -> CompressorPrefillPlan:
        assert seq_lens.device == extend_lens.device
        seq_lens = seq_lens.to(torch.int64)
        extend_lens = extend_lens.to(torch.int64)
        plan_tensor = torch.empty(
            (2, num_q_tokens, 16),
            dtype=torch.uint8,
            device=seq_lens.device,
            pin_memory=seq_lens.is_cpu,
        )
        is_overlap = compress_ratio == 4
        from sglang.srt.utils import is_hip
        if is_hip():
            from sglang.jit_kernel.compress_torch import plan_compress_prefill_torch
            plan_lens = plan_compress_prefill_torch(
                extend_lens, seq_lens,
                plan_tensor[0], plan_tensor[1],
                compress_ratio, is_overlap, use_cuda_graph,
            )
        else:
            module = _jit_common_module()
            # NOTE: when seq_lens on CUDA device or use_cuda_graph = True,
            # the C++/CUDA implementation will pad up to num_q_tokens
            plan_lens = module.plan_compress_prefill(
                extend_lens,
                seq_lens,
                plan_tensor[0],
                plan_tensor[1],
                compress_ratio,
                is_overlap,
                use_cuda_graph,
            )
        return CompressorPrefillPlan(
            compress_ratio,
            plan_tensor[0, : plan_lens[0]].to(device, non_blocking=True),
            plan_tensor[1, : plan_lens[1]].to(device, non_blocking=True),
        )


# NOTE: only decode plan is compatible with cuda graph
class CompressorDecodePlan(NamedTuple):
    compress_ratio: int
    seq_lens: torch.Tensor

    def copy_(self, other: CompressorDecodePlan) -> None:
        assert self.compress_ratio == other.compress_ratio
        self.seq_lens.copy_(other.seq_lens)


def compress_plan(
    compress_ratio: Literal[4, 128],
    num_q_tokens: int,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor],
    device: torch.device,
) -> Union[CompressorDecodePlan, CompressorPrefillPlan]:
    if extend_lens is not None:
        return CompressorPrefillPlan.generate(
            compress_ratio,
            num_q_tokens,
            seq_lens,
            extend_lens,
            device,
        )
    else:
        assert num_q_tokens == len(seq_lens)
        seq_lens = seq_lens.to(device, non_blocking=True)
        return CompressorDecodePlan(compress_ratio, seq_lens)


def compress_forward(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan, None] = None,
    extra_data: Optional[torch.Tensor] = None,
    *,
    head_dim: int,
    compress_ratio: Literal[4, 128],
    out: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    extend_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # TODO(dark): support dynamic plan and dispatch for decode kernel
    # Currently, there's no load-balancing for compression kernel
    # In worst cases, few SM will be overloaded with most compression work.
    # For C4, this may not be a big issue, since the compression is fast enough,
    # and the compression is quite common (with an probability of 1/4 in average).
    # For C128, the compression involves CTA reduction, which is relatively slow,
    # and the compression is rare (with an probability of 1/128 in average).
    # We may need to implement dynamic dispatch to better balance the load among SMs.
    # We may need some interface like `module.plan(...)` to prepare before forward pass.
    assert head_dim % 128 == 0
    num_q_tokens = kv_score_input.shape[0]
    if out is None:
        out = kv_score_input.new_empty((num_q_tokens, head_dim))
    if plan is None:
        assert seq_lens is not None
        plan = compress_plan(
            compress_ratio,
            num_q_tokens,
            seq_lens,
            extend_lens,
            kv_score_input.device,
        )
    assert plan.compress_ratio == compress_ratio, "Mismatched compress ratio in plan!"
    from sglang.srt.utils import is_hip
    if is_hip():
        from sglang.jit_kernel.compress_c4_c128_torch import (
            compress_decode_hip, compress_prefill_hip,
        )
        if isinstance(plan, CompressorDecodePlan):
            compress_decode_hip(
                compress_ratio, kv_score_buffer, kv_score_input, out, ape,
                indices, plan[1], extra_data,
            )
        else:
            # Prefill: plan = (compress_ratio, compress_plan, write_plan)
            compress_prefill_hip(
                compress_ratio, kv_score_buffer, kv_score_input, out, ape,
                indices, plan[1], plan[2], extra_data,
            )
        return out
    module = _jit_compress_module(
        head_dim,
        kv_score_input.dtype,
        out.dtype,
        compress_ratio,
    )
    F = module.decode if isinstance(plan, CompressorDecodePlan) else module.prefill
    F(kv_score_buffer, kv_score_input, out, ape, indices, *plan[1:], extra_data)
    return out


def compress_fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
) -> None:
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    mode = 1 if isinstance(plan, CompressorDecodePlan) else 0
    from sglang.srt.utils import is_hip
    if is_hip():
        from sglang.jit_kernel.fused_norm_rope_triton import fused_norm_rope_inplace_hip
        fused_norm_rope_inplace_hip(
            kv, weight, plan[1], freq_cis, mode, eps, plan.compress_ratio,
        )
        return
    module = _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_cis.shape[-1])
    module.forward(
        kv,
        weight,
        plan[1],  # decode: seq_lens, prefill: compress_plan
        freq_cis,
        mode,
        eps,
        plan.compress_ratio,
    )


def compress_fused_norm_rope_quant(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    plan: "CompressorDecodePlan",
    fp8_dtype: Optional[torch.dtype] = None,
):
    """A2-#3: fused norm+rope+per-1x128 fp8 quant — decode mode only.

    Equivalent to ``compress_fused_norm_rope_inplace`` followed by
    ``aiter_per1x128_quant``, in a single Triton launch. Returns
    ``(fp8_out [N,head_dim], scale_out [N,head_dim/128])``.

    Falls back to the two-step path on non-HIP backends or for prefill plans.
    """
    if fp8_dtype is None:
        fp8_dtype = torch.float8_e4m3fn
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    from sglang.srt.utils import is_hip
    if is_hip() and isinstance(plan, CompressorDecodePlan):
        from sglang.jit_kernel.fused_norm_rope_triton import (
            fused_norm_rope_quant_decode_hip,
        )
        return fused_norm_rope_quant_decode_hip(
            kv, weight, plan[1], freq_cis, eps, plan.compress_ratio,
            fp8_dtype=fp8_dtype,
        )
    # Fallback: two-step path
    compress_fused_norm_rope_inplace(kv, weight, eps, torch.view_as_complex(
        freq_cis.view(*freq_cis.shape[:-1], -1, 2)
    ), plan)
    # Caller should run aiter_per1x128_quant separately in the fallback path.
    return None


def fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    from sglang.srt.utils import is_hip
    if is_hip():
        from sglang.jit_kernel.fused_norm_rope_triton import fused_norm_rope_inplace_hip
        fused_norm_rope_inplace_hip(
            kv, weight, positions, freq_cis, 2, eps, 0,
        )
        return
    module = _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_cis.shape[-1])
    module.forward(
        kv,
        weight,
        positions,
        freq_cis,
        2,  # mode
        eps,
        0,  # compress_ratio (no use in this mode)
    )


def fused_rope(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    """Apply rotary embeddings to both Q and K in a single fused CUDA kernel.

    Args:
        q: [batch_size, num_q_heads, rope_dim] bfloat16
        k: [batch_size, num_k_heads, rope_dim] bfloat16 or None
        freqs_cis: [max_seq_len, rope_dim // 2] complex64 (full table)
        positions: [batch_size] int32 or int64, indices into freqs_cis
        inverse: if True, apply inverse rotation (conjugate freqs)
    """
    from sglang.srt.utils import is_hip

    if is_hip():
        from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton

        apply_rotary_emb_triton(q, freqs_cis, positions=positions, inverse=inverse)
        if k is not None:
            apply_rotary_emb_triton(k, freqs_cis, positions=positions, inverse=inverse)
        return

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    module = _jit_fused_rope_module()
    module.forward(q, k, freqs_real, positions, inverse)


# NOTE: TileLang's `@tilelang.jit` does not work in modules with
# `from __future__ import annotations` — `typing.get_type_hints` evaluates
# `T.Tensor[(...), T.int32]` to a TileLang Buffer instance which is not a
# Python type, and `typing._type_check` rejects it with "Forward references
# must evaluate to types. Got buffer." The kernel therefore lives in a
# sibling module without future-annotations.
from sglang.jit_kernel.swa_indices_tilelang import (  # noqa: F401
    _tilelang_make_swa_indices_kernel,
    tilelang_make_swa_prefill_indices,
)


@triton.jit
def create_paged_compress_data_kernel(
    req_pool_indices_ptr,  # int32 [batch]
    seq_lens_ptr,  # int32 [batch]
    extend_seq_lens_ptr,  # int32 [batch]
    req_to_token_ptr,  # int32 [A, B]
    full_to_swa_index_mapping_ptr,  # int32 [C]
    out_0_ptr,  # int32 [batch]
    out_1_ptr,  # int32 [batch, out_dim]
    batch_size,
    stride_req_to_token_0,
    stride_req_to_token_1: tl.constexpr,  # 1
    stride_out_1_0,
    stride_out_1_1: tl.constexpr,  # 1
    compress_ratio: tl.constexpr,
    is_overlap: tl.constexpr,  # 0/1
    swa_page_size: tl.constexpr,
    ring_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < batch_size

    # load per-batch
    rid = tl.load(req_pool_indices_ptr + offs, mask=mask, other=0).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    extend_len = tl.load(extend_seq_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    prefix_len = seq_len - extend_len

    cr = compress_ratio
    write_pos = ((seq_len - 1) // cr) * cr
    load_pos = ((prefix_len - 1) // cr) * cr
    write_overlap_pos = write_pos - cr
    load_overlap_pos = load_pos - cr
    v0 = tl.zeros([BLOCK], tl.int32)
    v1 = tl.zeros([BLOCK], tl.int32)
    v2 = tl.zeros([BLOCK], tl.int32)
    v3 = tl.zeros([BLOCK], tl.int32)

    for i in tl.static_range(4):
        if i == 0:
            pos = load_pos
        elif i == 1:
            pos = write_pos
        elif i == 2:
            pos = load_overlap_pos
        else:
            pos = write_overlap_pos
        pos = tl.maximum(pos, 0)
        # req_to_token[rid, pos]
        loc = tl.load(
            req_to_token_ptr
            + rid * stride_req_to_token_0
            + pos * stride_req_to_token_1,
            mask=mask,
            other=0,
        ).to(tl.int32)
        swa_loc = tl.load(full_to_swa_index_mapping_ptr + loc, mask=mask, other=0).to(
            tl.int32
        )
        swa_page = swa_loc // swa_page_size
        state_loc = swa_page * ring_size + (swa_loc % ring_size)
        state_loc = state_loc // cr
        if i == 0:
            v0 = state_loc
        elif i == 1:
            v1 = state_loc
        elif i == 2:
            v2 = state_loc
        else:
            v3 = state_loc

    tl.store(out_0_ptr + offs, v1, mask=mask)

    if is_overlap:
        base = out_1_ptr + offs * stride_out_1_0
        tl.store(base + 0 * stride_out_1_1, v2, mask=mask)
        tl.store(base + 1 * stride_out_1_1, v0, mask=mask)
        tl.store(base + 2 * stride_out_1_1, v3, mask=mask)
        tl.store(base + 3 * stride_out_1_1, write_pos.to(tl.int32), mask=mask)
    else:
        base = out_1_ptr + offs * stride_out_1_0
        tl.store(base + 0 * stride_out_1_1, v0, mask=mask)


def triton_create_paged_compress_data(
    *,
    compress_ratio: int,
    is_overlap: bool,
    swa_page_size: int,
    ring_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    block: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = req_pool_indices.shape[0]
    out_dim = 4 if is_overlap else 1
    device_args: dict = dict(device=req_pool_indices.device, dtype=torch.int32)
    out_0 = torch.empty((batch_size,), **device_args)
    out_1 = torch.empty((batch_size, out_dim), **device_args)
    grid = (triton.cdiv(batch_size, block),)
    create_paged_compress_data_kernel[grid](
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        req_to_token,
        full_to_swa_index_mapping,
        out_0,
        out_1,
        batch_size=batch_size,  # type: ignore
        stride_req_to_token_0=req_to_token.stride(0),  # type: ignore
        stride_req_to_token_1=req_to_token.stride(1),  # type: ignore
        stride_out_1_0=out_1.stride(0),  # type: ignore
        stride_out_1_1=out_1.stride(1),  # type: ignore
        compress_ratio=compress_ratio,  # type: ignore
        is_overlap=1 if is_overlap else 0,  # type: ignore
        swa_page_size=swa_page_size,  # type: ignore
        ring_size=ring_size,  # type: ignore
        BLOCK=block,  # type: ignore
    )
    if not is_overlap:
        out_1.squeeze_(1)
    return out_0, out_1


def fused_store_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: Literal["flashmla", "indexer"],
) -> None:
    # On ROCm/HIP the JIT-CUDA path is unavailable (`tvm_ffi.cpp.load_inline`
    # requires `CUDA_HOME`/`nvcc` and has no HIP build path). To make
    # `SGLANG_OPT_USE_FUSED_STORE_CACHE=true` work on HIP we route through
    # a Triton port (flashmla) and a torch fallback (indexer) — same pattern
    # as topk_transform_512_triton.py at lines 154-180 above.
    from sglang.srt.utils import is_hip
    if is_hip():
        from sglang.jit_kernel.fused_store_cache_triton import fused_store_cache_hip
        fused_store_cache_hip(input, cache, indices, page_size=page_size, type=type)
        return
    module = _jit_fused_store_module(
        name=type,
        input_dtype=input.dtype,
        index_dtype=indices.dtype,
        page_size=page_size,
    )
    module.run(input, cache, indices)


@cache_once
def _jit_silu_mul_quant_module(
    quant_group_size: int, scale_ue8m0: bool, apply_swiglu_limit: bool
) -> Module:
    args = make_cpp_args(
        quant_group_size, scale_ue8m0, is_arch_support_pdl(), apply_swiglu_limit
    )
    return load_jit(
        make_name("silu_mul_quant"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulMaskedPostQuantKernel<{args}>::run")],
    )


def silu_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
) -> None:
    """
    Fused SiLU-and-mul with per-group FP8 quantization for expert-parallel MoE.

    input shape:        [expert_num, token_num_padded, hidden_dim]
    output shape:       [expert_num, token_num_padded, hidden_dim // 2], dtype fp8_e4m3
    output_scale shape: [expert_num, token_num_padded, hidden_dim // 2 // quant_group_size], dtype float32
    masked_m shape:     [expert_num], dtype int32. i.e. actual token count per expert
    topk:               max routed experts per token (grid = token_num_padded * topk blocks)
    swiglu_limit:       Optional. When None (default), use the original fast path (no clamp).
                        When set, JIT-compiles a separate kernel variant that clamps gate to
                        [-inf, L] and up to [-L, L] before silu (fused).
    """
    apply_swiglu_limit = swiglu_limit is not None
    if apply_swiglu_limit:
        deepseek_v4_moe_code_path_checker.observed += 1
    from sglang.srt.utils import is_hip
    if is_hip():
        from sglang.jit_kernel.silu_and_mul_masked_post_quant_triton import (
            silu_and_mul_masked_post_quant_hip,
        )
        silu_and_mul_masked_post_quant_hip(
            input, output, output_scale, quant_group_size, masked_m,
            scale_ue8m0=scale_ue8m0, topk=topk, transposed=transposed,
            swiglu_limit=float(swiglu_limit) if apply_swiglu_limit else 0.0,
        )
        return
    module = _jit_silu_mul_quant_module(
        quant_group_size, scale_ue8m0, apply_swiglu_limit
    )
    module.run(
        input,
        output,
        output_scale,
        masked_m,
        topk,
        transposed,
        float(swiglu_limit) if apply_swiglu_limit else 0.0,
    )


def get_paged_mqa_logits_metadata(seq_lens: torch.Tensor, page_size: int, num_sm: int):
    assert page_size == 64
    seq_lens = seq_lens.to(torch.int32)
    from sglang.srt.utils import is_hip
    if is_hip():
        from sglang.jit_kernel.paged_mqa_metadata_torch import (
            get_paged_mqa_logits_metadata_torch,
        )
        return get_paged_mqa_logits_metadata_torch(seq_lens, page_size, num_sm)
    metadata = seq_lens.new_empty(num_sm + 1, 2)
    module = _jit_metadata_module()
    module.run(seq_lens, metadata)
    return metadata


@cache_once
def _jit_torch_cublas_bf16_fp32() -> Any:
    import torch.utils.cpp_extension

    source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

torch::Tensor linear_bf16_fp32(
    torch::Tensor X,
    torch::Tensor W)
{
    int batch = X.size(0);
    int in_features = X.size(1);
    int out_features = W.size(0);

    auto Y = torch::empty(
        {batch, out_features},
        torch::dtype(torch::kFloat32).device(X.device()));

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_features,
        batch,
        in_features,
        &alpha,
        W.data_ptr(), CUDA_R_16BF, in_features,
        X.data_ptr(), CUDA_R_16BF, in_features,
        &beta,
        Y.data_ptr(), CUDA_R_32F, out_features,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_bf16_fp32", &linear_bf16_fp32, "BF16xBF16 -> FP32 linear (no bias)");
}
"""
    module = torch.utils.cpp_extension.load_inline(
        name="linear_bf16_fp32",
        cpp_sources="",
        cuda_sources=source,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return module


def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    from sglang.srt.environ import envs

    algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()

    if algo == "cublas":
        module = _jit_torch_cublas_bf16_fp32()
        return module.linear_bf16_fp32(x, y)
    elif algo == "deep_gemm":
        import deep_gemm

        z = x.new_empty(x.size(0), y.size(0), dtype=torch.float32)
        deep_gemm.bf16_gemm_nt(x, y, z)
        return z
    else:  # fall back to torch fp32 GEMM
        return torch.nn.functional.linear(x.float(), y.float())


def _compile_one(*input_tuple) -> None:
    name, job_fn, *args = input_tuple
    print(f"Compiling {name}...", flush=True)
    job_fn(*args)
    print(f"Finished compiling {name}.", flush=True)


def compile_aot():
    c_dtype = torch.float32  # compress uses float32
    jobs = [
        ("cublas", _jit_torch_cublas_bf16_fp32),
        ("common", _jit_common_module),
        ("topk", _jit_topk_module),
        ("hash_topk", _jit_hash_topk_module),
        ("rope", _jit_fused_rope_module),
        ("metadata", _jit_metadata_module),
        (
            "compress_128_4",
            _jit_compress_module,
            128,
            c_dtype,
            c_dtype,
            4,
        ),
        (
            "compress_512_4",
            _jit_compress_module,
            512,
            c_dtype,
            c_dtype,
            4,
        ),
        (
            "compress_512_128",
            _jit_compress_module,
            512,
            c_dtype,
            c_dtype,
            128,
        ),
        (
            "norm_rope_128_64",
            _jit_norm_rope_module,
            c_dtype,
            128,
            64,
        ),
        (
            "norm_rope_512_64",
            _jit_norm_rope_module,
            c_dtype,
            512,
            64,
        ),
        (
            "store_flashmla_bf16_swa_256",
            _jit_fused_store_module,
            "flashmla",
            torch.bfloat16,
            torch.int32,
            256,
        ),
        (
            "store_flashmla_fp32_c4_64",
            _jit_fused_store_module,
            "flashmla",
            torch.float32,
            torch.int32,
            64,
        ),
        (
            "store_flashmla_fp32_c128_2",
            _jit_fused_store_module,
            "flashmla",
            torch.float32,
            torch.int32,
            2,
        ),
        (
            "store_indexer_fp32_c4_64",
            _jit_fused_store_module,
            "indexer",
            torch.float32,
            torch.int32,
            64,
        ),
    ]
    # use multiprocess to speed up compilation
    import multiprocessing

    max_parallel_jobs = min(len(jobs), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
        pool.starmap(_compile_one, jobs)


if __name__ == "__main__":
    compile_aot()
