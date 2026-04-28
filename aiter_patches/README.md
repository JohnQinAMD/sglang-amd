# Aiter upstream patches for piecewise CG correctness

These three files are patched copies of `aiter/` source from inside the
`rocm/sgl-dev:rocm720-deepseek-v4-mi35x` container, fixing the
`mutates_args="unknown"` schema bug documented in
[ROCm/aiter issue #2780](https://github.com/ROCm/aiter/issues/2780). The bug
makes `torch.library.infer_schema` mark every tensor input as mutated
(`Tensor(a0!)`), torch.compile then wraps the op in `auto_functionalized_v2`,
which is incompatible with cuda graph capture and produces low-address
memory access faults during piecewise CG capture on MI355X.

## What's patched

| File | Change |
|---|---|
| `quant.py` | `per_group_quant_hip`: `@torch_compile_guard()` → `@torch_compile_guard(mutates_args=[])` |
| `gemm_op_a8w8.py` | `gemm_a8w8_blockscale_ck`: added `mutates_args=["Out"]` to its `@compile_ops(...)` |
| `core.py` | `compile_ops()` now accepts and forwards `mutates_args` to the inner `torch_compile_guard` call |

These mirror [PR #2781](https://github.com/ROCm/aiter/pull/2781) (open) which
applies the same pattern to `gemm_a4w4`. The two ops here (`per_group_quant_hip`,
`gemm_a8w8_blockscale_ck`) are on the DSv4-Pro mxfp4 EP=1 piecewise CG path
and need the same fix.

## How to apply

Bind-mount the patched files over the container's aiter source at
`docker run` time:

```bash
docker run ... \
  -v /path/to/sglang_v4_pr/aiter_patches/quant.py:/sgl-workspace/aiter/aiter/ops/quant.py:ro \
  -v /path/to/sglang_v4_pr/aiter_patches/gemm_op_a8w8.py:/sgl-workspace/aiter/aiter/ops/gemm_op_a8w8.py:ro \
  -v /path/to/sglang_v4_pr/aiter_patches/core.py:/sgl-workspace/aiter/aiter/jit/core.py:ro \
  ...
```

See `launch_dsv4_pro_mxfp4.sh` callers for a working example.

## Upstream status

These are local copies, not a clean diff against upstream. To file PRs to
ROCm/aiter, extract the three changes above as a patch series modeled on
PR #2781. Submitting upstream removes the need for the bind-mount.

## Verified effect

Without these patches: container MAFs at the first piecewise CG capture step
on MI355X with addresses in `{0x2000, 0x4000, 0x6000, 0xc000, NULL}` —
classic fake-tensor-address leakage from `auto_functionalized_v2` wrap.

With these patches + `register_custom_op + register_split_op` pattern in
`fp8_utils.py:aiter_w8a8_block_fp8_linear`: capture succeeds at 13.2 GB/rank
(vs 144 GB standard CG = 11× memory reduction), server stable, 220 ms TPOT.
Output tokens are still garbage due to other unrelated leak sites in the
DSv4 forward path; investigation continued in `ck_v32_sparse_mla.py:_OUT_BUF_CACHE`
and elsewhere. See plan file
`/home/yanyuqin/.claude/plans/quirky-purring-llama.md` for full context.
