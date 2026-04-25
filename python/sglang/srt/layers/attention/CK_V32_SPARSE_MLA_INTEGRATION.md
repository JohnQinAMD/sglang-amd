# CK Tile FP8 Sparse MLA Decode — SGLang Integration

This integrates a hand-written CK Tile sparse MLA decode kernel for the
V32 (DSv4-Pro / V3) shape on AMD MI355X (gfx950). The kernel beats AMD's
asm `.co` baseline by **1.7-3.4×** across the typical decode workload
grid and the SGLang Triton fallback by **2.0-4.6×**.

## Files (all in SGLang tree)

```
python/sglang/srt/layers/attention/
├── ck_v32_sparse_mla.py                      Python adapter (this PR)
├── debug_flash_mla_adapter.py                env-gate dispatch (modified)
└── csrc/ck_v32/
    ├── mla_decode_fwd.cu                     pybind launcher
    └── mla_decode_fwd_kernel.hpp             CK Tile FP8 sparse kernel

test/srt/
└── test_ck_v32_sparse_mla.py                 9 pytest parity + invariant cases

benchmark/bench_ck_v32_sparse_mla/
└── bench_ck_v32_sparse_mla.py                B×topk perf sweep
```

The kernel source bundled at `csrc/ck_v32/` is the only required artifact
for runtime. Override the lookup path via
`SGLANG_CK_V32_KERNEL_SRC_DIR=<path>` for in-place kernel iteration.

## Activation

Set the env var when launching the SGLang server:

```bash
export SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1
# Optional: override default kernel src dir
export SGLANG_CK_V32_KERNEL_SRC_DIR=/path/to/ck_mla_decode_sparse_fp8
```

The dispatch in `debug_flash_mla_adapter.py:flash_mla_with_kvcache_torch`
routes on `head_dim_v`:

| Model | d_v | Path |
|---|---|---|
| DSv4-Pro / V3 (V32 layout) | **512** | **CK Tile FP8 sparse** (this) |
| DSv4-Flash 2604 (MODEL1 layout) | 448 | TileLang MODEL1 (existing) |
| Other | * | falls through to BF16 dequant + Triton sparse |

## Build prerequisites

- ROCm 7+, hipcc, gfx950 GPU
- aiter Python package importable (we use `aiter.mla_reduce_v1` for stage-2)
- The kernel C++ source visible at `SGLANG_CK_V32_KERNEL_SRC_DIR`

First import compiles the kernel via `torch.utils.cpp_extension.load`
(~5 s; cached at `~/.cache/torch_extensions/...` for subsequent imports).

## Performance (B=1 H=128 topk=512 on MI355X chi2811)

| Path | Wall-clock | Kernel launches/iter | GPU time |
|---|---|---|---|
| **CK Tile (this)** | **47 µs** | **3** | 42.7 µs |
| asm `.co` (aiter) | 108 µs | ~5 | ~100 µs |
| Triton fallback | 140 µs | 20 | 73 µs |
| TileLang MODEL1 (V32 forced) | 508 µs | ~5 | ~500 µs |

Full B×topk grid (post-fusion, adapter end-to-end): wins **15/16 vs asm**,
**16/16 vs Triton fallback**. To re-run on a target machine:

```bash
PYTHONPATH=python python3 benchmark/bench_ck_v32_sparse_mla/bench_ck_v32_sparse_mla.py
```

## Correctness

Parity GREEN against the FP32 oracle (`_sparse_attn_decode_inner` from
`sglang.srt.flashmla_tests.ref`) at cos_sim 0.999998 across all tested
shapes. The test suite lives in-tree at
[`test/srt/test_ck_v32_sparse_mla.py`](../../../../../test/srt/test_ck_v32_sparse_mla.py)
— 9 cases covering 7 (B × topk) parity shapes plus lonely-Q-zeros and
attn-sink-correction invariants.

```bash
PYTHONPATH=python pytest test/srt/test_ck_v32_sparse_mla.py -v
```

## Known limitations

1. **B≥8 t≥2048 regression vs asm**: at very high batch + topk, CK loses
   to AMD's asm `.co` (e.g. B=8 t=2048: CK 141 µs vs asm 108 µs = 0.76×).
   Caused by per-batch workgroup imbalance — asm uses a different
   work-distribution scheme that scales better at large workloads.
   **CK still beats the existing Triton fallback (151 µs) at every shape**,
   so enabling CK is never worse than disabling it. Wiring asm `.co` as
   a fallback for the regression band is a future optimization (requires
   `aiter.mla.mla_decode_fwd` metadata plumbing).

2. **Stage-2 reduce metadata is uniform-decode-only**: assumes max_seqlen_q=1
   and uniform splits per query. Multi-token speculative decode would need
   the full aiter `get_mla_metadata_v1` path.

3. **Attention sink** is folded post-reduce in PyTorch (extra ~2-3 µs).
   Could be moved into the CK reduce kernel for complete fusion.

4. **No FP8 native MFMA yet**: kernel decodes FP8→BF16 in the LDS tile
   loader and uses BF16 MFMA (16x16x32_bf16, 512 FLOPs/issue). A
   `mfma_f32_16x16x32_fp8_fp8` (1024 FLOPs/issue) variant could halve
   the MFMA pass; deferred until a re-bench shows MFMA-bound behavior.

## Deployment notes

For production deployment outside this dev tree:
1. Bundle the kernel source (`mla_decode_fwd.cu`, `mla_decode_fwd_kernel.hpp`)
   into the SGLang Python package under `sglang/srt/layers/attention/csrc/ck_v32/`
2. Update `_kernel_src_dir()` default to that bundled path
3. Or: register as an aiter JIT module in `aiter/jit/optCompilerConfig.json`
   and call `aiter.mla_decode_fwd_ck_sparse_fp8(...)` directly (cleaner but
   requires aiter-amd patch)
