# Triton sparse decode kernel autotune sweep — 2026-04-29 chi2774 MI355X

Curated 180-config-per-shape sweep across all DSv4 production shapes. Bench
under cuda-graph capture+replay (per `feedback_microbench_vs_e2e_cuda_graph.md`).
Source: `/tmp/triton_sparse_decode_sweep.py`.

## Result table

| Shape | Path | Base ms | Best ms | Speedup | Best config |
|---|---|---:|---:|---:|---|
| Flash B=1 T=512 | V3 | 0.0445 | 0.0438 | 1.02× | BH=4 BT=128 BD=256 nw=4 |
| Flash B=1 T=512 | SK | 0.0196 | 0.0185 | **1.06×** | BH=4 BT=32 BD=256 SK=16 wpe=2 mind=16 |
| Flash B=4 T=512 | V3 | 0.0448 | 0.0442 | 1.01× | BH=4 BT=128 BD=256 nw=4 |
| Flash B=4 T=512 | SK | 0.0231 | 0.0209 | **1.10×** | BH=8 BT=32 BD=256 SK=8 wpe=2 mind=16 |
| Pro B=1 T=512 | V3 | 0.0483 | 0.0483 | 1.00× | (default optimal) |
| Pro B=1 T=512 | SK | 0.0200 | 0.0192 | **1.04×** | BH=4 BT=32 BD=128 SK=16 wpe=2 mind=16 |
| Pro B=4 T=512 | V3 | 0.0484 | 0.0484 | 1.00× | (default optimal) |
| Pro B=4 T=512 | SK | 0.0239 | 0.0218 | **1.10×** | BH=8 BT=64 BD=128 SK=8 nw=4 |
| Flash B=4 T=2048 | V3 | 0.1396 | 0.1368 | 1.02× | BH=4 BT=128 BD=256 nw=4 |
| Flash B=4 T=2048 | SK | 0.0415 | 0.0314 | **1.32×** | BH=8 BT=32 BD=256 SK=16 wpe=2 mind=16 |

## Universal patterns

1. **`waves_per_eu=2 matrix_instr_nonkdim=16`** consistently wins on AMD/MI355X.
   Default (None, None) lets the compiler pick — typically 4 waves/EU which
   over-subscribes register-heavy MFMA pipes.

2. **`BLOCK_T=32`** beats `BLOCK_T=64` for split-K decode. Smaller T-tile keeps
   the online-softmax loop tight and lets `SPLIT_K` saturate CUs better. V3
   (single-kernel prefill chunks) prefers BLOCK_T=128 (more amortization).

3. **`BLOCK_D` matched to `D_QK`** matters: Flash (D=512) uses BD=256 (1 clean
   D-tile, no mask waste); Pro (D=576) uses BD=128 (5 tiles, last has 50% mask
   waste — but BD=192 non-improving in sweep, so keep 128). Default BD=128
   left 50% of the last D-tile masked-out on Flash.

4. **`SPLIT_K` scales with grid pressure**: B=1 wants SK=16 (more programs to
   fill 256 CUs); B=4+ small-topk wants SK=8; B=4 big-topk (T=2048) wants
   SK=16 for max parallel reduction.

5. **`BLOCK_H=8`** wins for B≥4: denser MFMA per program, fewer programs to
   launch. B=1 stays at BH=4 (already grid-bound, BH=8 just halves program
   count).

## Implementation

Applied as per-shape dispatch table in
[python/sglang/srt/flashmla_tests/triton_sparse_decode_kernel.py](python/sglang/srt/flashmla_tests/triton_sparse_decode_kernel.py)
inside `triton_sparse_attn_decode`'s split-K auto-dispatch branch (replaces
the single hardcoded `BLOCK_H=4, BLOCK_T=64, BLOCK_D=128, SPLIT_K=8`). V3
single-kernel left alone (gains <2% are dominated by JIT compile cost).

## Estimated TPOT impact on DSv4

Sparse decode in trace was ~2.7 ms per 5 decode steps (43 stage1 calls × 63
us). At ~1.10× avg speedup on the typical decode shape, ~0.05–0.10 ms TPOT
reduction per step. Bigger lift on Flash big-topk (chunked-prefill where
T=2048): ~0.10 ms per call × ~5 calls per chunk = 0.5 ms TTFT.

Post-fix expected TPOT on chi2774 Flash mxfp4 c=4 num=8: 33.69 → ~33.5 ms
(within noise of the just-shipped MHC_PRE=0 baseline). Lift compounds with
Lever 4 (decode-body megakernel) when it lands.

---

## v3b sweep — 2026-05-03 chi2811 (post-stack baseline)

The 04-29 round only swept up to Topk=2048 with `BLOCK_H ∈ {4, 8}`. v3b adds
`BLOCK_H ∈ {16, 32, 64}` against the post-2026-05-03 baseline (T1-T7 + A1/A3
/B3/C1 + MEGA-3' Stage 1+2 + fused_lonely_q + Lever G all default-on).

### Result table (production decode, Flash D_QK=D_V=512, Hq=64)

| Shape | 04-29 winner | 04-29 us | v3b winner | v3b us | Speedup vs 04-29 |
|---|---|---:|---|---:|---:|
| B=2 T=2048 | BH=8  SK=16 | 26.18 | **BH=16 SK=32** | 19.07 | **1.37×** |
| B=4 T=2048 | BH=8  SK=16 | 26.71 | **BH=16 SK=16** | 21.75 | **1.23×** |
| B=8 T=2048 | BH=8  SK=16 | 42.41 | BH=32 SK=16 | 27.37 | 1.55× (BH=16 SK=16: 27.93, near-tie) |
| B=4 T=1024 | BH=8  SK=16 | 19.81 | **BH=16 SK=16** | 17.00 | **1.17×** |
| B=8 T=1024 | BH=8  SK=16 | 28.96 | BH=32 SK=16 | 20.05 | 1.45× (BH=16 SK=16: 20.38, near-tie) |

### Why BH=16 was missed in 04-29

04-29 set BH=8 as the universal big-topk choice based on B=4 T=512 results
(where BH=16 wasn't materially better). At T=2048 the per-program work is 4×
larger and BH=16's H-axis MFMA amortization wins decisively — but T=2048 wasn't
swept at BH=16 in 04-29. Same root cause as the 04-29 BLOCK_D fix: hardcoded
defaults from a small-shape sweep don't generalize to big-topk decode.

### Universal v3b pattern

**BH=16 SK=16 within 6% of best on every shape tested.** Adopted as the
universal Topk≥1024 path (vs picking BH=32 only at B=8 — more dispatch
complexity for ≤2% extra). Dispatch table at
[python/sglang/srt/flashmla_tests/triton_sparse_decode_kernel.py:185-220](python/sglang/srt/flashmla_tests/triton_sparse_decode_kernel.py#L185).

Env knob `SGLANG_SPARSE_DECODE_BH16_BIG_TOPK=0` rolls back to 04-29 dispatch.

### Correctness gate

20 (B, Topk) shape combinations vs BH=8 SK=16 reference. All PASS. Production
decode shapes (B≥4, Topk≥1024) **bit-exact** (max_diff = 0.0). Other shapes
within bf16-ULP rounding noise (max_diff ≤ 1e-4).

### Estimated TPOT delta

40 stage1 calls/iter × 5 us/call savings (B=4 T=2048: 26.71→21.75 = -4.96 us)
= 198 us/iter gross. Critical-path-adjusted (75% credit for kernel overlap)
≈ 0.15 ms TPOT. **Above the 0.13 ms median noise floor**. Aligned E2E bench
on chi2811 confirms (see daily-updates/2026-05-03.md "stage1 BH=16" entry).
