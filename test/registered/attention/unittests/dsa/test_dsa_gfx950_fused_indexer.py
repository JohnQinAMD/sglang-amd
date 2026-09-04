"""Numerical checks for the gfx950 fused DSA-indexer kernels.

Covered here, each against an implementation that was already in the tree:

  node 1  dual_gemv_bf16   -> torch.mm on the same operands
  node 4  topk_transform   -> aiter's dsa_topk_transform, selecting from the
                              same logits, compared as exact sets of physical
                              slots

NOT covered, stated plainly rather than left to be assumed:

  node 2  qk_rope_hadamard_quant_and_cache
  node 3  logits_hist

Both read or write the FP8 index-K cache in aiter's preshuffled page layout,
where each 64-token page interleaves the 128 K bytes and the 4 ue8m0 scale
bytes rather than storing them as [128 K][4 scale] per token. A test cannot
synthesise that content: random bytes give scale exponents spanning 2**+-127
and fp8 NaN patterns, and a comparison then measures denormal handling instead
of the dot product (measured: 1.2 dB SNR against aiter, with individual entries
agreeing to 7 digits and others differing in sign -- the signature of garbage
inputs, not of a wrong kernel). Producing a valid cache needs either a torch
model of the layout or the in-tree ``fused_store_index_k_cache``, whose JIT does
not build in the ROCm image. Until one of those exists, treat nodes 2 and 3 as
untested; node 4's inputs below are therefore synthetic on purpose, which is
sound because both sides of that comparison read the same logits.

Requires a gfx950 device and a successful build of the extension modules;
skipped otherwise.
"""

import unittest

import torch

HEAD_DIM = 128
N_HEADS = 32
PAGE_SIZE = 64
TOPK = 2048
Q_LORA_RANK = 2048
HIDDEN_SIZE = 6144
KW_ROWS = HEAD_DIM + N_HEADS  # [wk ; weights_proj] merged output width
FP8 = torch.float8_e4m3fn


def _skip_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0]
    if arch != "gfx950":
        return f"kernels are gfx950-only, got {arch}"
    try:
        from sglang.kernels.ops.attention.dsa.hip_gfx950 import loader
    except ImportError as exc:  # pragma: no cover - build-environment dependent
        return f"extension package unavailable: {exc}"
    if loader.modules_or_none() is None:
        return "gfx950 fused indexer modules failed to build"
    return None


SKIP = _skip_reason()


def _snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref = ref.float()
    err = got.float() - ref
    power = ref.pow(2).mean()
    noise = err.pow(2).mean()
    if noise == 0:
        return float("inf")
    return float(10 * torch.log10(power / noise))


@unittest.skipIf(SKIP is not None, SKIP or "")
class TestGfx950FusedIndexerKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.dsa.hip_gfx950 import loader

        cls.loader = loader
        cls.gemv, cls.qk, cls.logits_mod, cls.topk_mod = loader.modules()
        cls.dev = torch.device("cuda", 0)
        torch.manual_seed(0)

    # -- node 1 --------------------------------------------------------------
    def test_dual_gemv_matches_torch_mm(self):
        """One launch, two independent projections; torch.mm is the definition.

        bf16 inputs with fp32 accumulation on both sides, so this is a numerical
        comparison, not bit equality. The dual GEMV is exercised at every row
        count it accepts in a single launch.
        """
        for rows in (1, 2, 4, 8):
            with self.subTest(rows=rows):
                q_lora = torch.randn(
                    rows, Q_LORA_RANK, dtype=torch.bfloat16, device=self.dev
                )
                w_q_b = torch.randn(
                    N_HEADS * HEAD_DIM,
                    Q_LORA_RANK,
                    dtype=torch.bfloat16,
                    device=self.dev,
                )
                x = torch.randn(
                    rows, HIDDEN_SIZE, dtype=torch.bfloat16, device=self.dev
                )
                w_kw = torch.randn(
                    KW_ROWS, HIDDEN_SIZE, dtype=torch.bfloat16, device=self.dev
                )
                q_proj = torch.empty(
                    rows, N_HEADS * HEAD_DIM, dtype=torch.bfloat16, device=self.dev
                )
                kw = torch.empty(rows, KW_ROWS, dtype=torch.bfloat16, device=self.dev)

                self.gemv.dual_gemv_bf16(
                    q_lora, w_q_b, q_proj, x, w_kw, kw, self.loader.DUAL_GEMV_CFG
                )
                torch.cuda.synchronize()

                self.assertGreater(
                    _snr_db(q_lora.float() @ w_q_b.float().t(), q_proj), 35.0
                )
                self.assertGreater(_snr_db(x.float() @ w_kw.float().t(), kw), 35.0)

    # -- node 4 --------------------------------------------------------------
    def _fused_logits(self, rows: int, ctx: int):
        """Produce logits and the paired histogram for the node-4 tests.

        The cache content is synthetic and its numerical meaning is not asserted
        anywhere -- see the module docstring. What node 4 needs from it is a
        populated ``logits`` row and the ``ghist`` that node 3 leaves behind,
        because the two kernels are coupled through that histogram. The fp8
        values come from ``randn`` rather than random bytes so no NaN patterns
        enter and the logits stay finite, which is all the selection kernel
        requires.
        """
        pages_per_row = ctx // PAGE_SIZE
        total_pages = rows * pages_per_row
        q_fp8 = torch.randn(
            rows, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=self.dev
        ).to(FP8)
        kv = torch.randn(
            total_pages, PAGE_SIZE * 132, dtype=torch.bfloat16, device=self.dev
        ).to(FP8)
        gate = torch.rand(rows, N_HEADS, dtype=torch.float32, device=self.dev)
        seqlens = torch.full((rows,), ctx, dtype=torch.int32, device=self.dev)
        page_table_64 = (
            torch.arange(total_pages, dtype=torch.int32, device=self.dev)
            .reshape(rows, pages_per_row)
            .contiguous()
        )
        logits = torch.zeros(rows, ctx, dtype=torch.float32, device=self.dev)
        ghist = torch.zeros(
            rows, self.topk_mod.hist_stride(), dtype=torch.int32, device=self.dev
        )
        self.logits_mod.logits_hist(
            q_fp8,
            kv,
            gate,
            seqlens,
            page_table_64,
            logits,
            ghist,
            self.loader.LOGITS_BLOCKS_PER_ROW,
            self.loader.LOGITS_HIST_BITS,
            TOPK,
            self.loader.LOGITS_WARPS,
        )
        torch.cuda.synchronize()
        return logits, ghist, (q_fp8, kv, gate, seqlens, page_table_64, ctx)

    def test_topk_transform_matches_aiter(self):
        """Node 4 against aiter's dsa_topk_transform on identical logits.

        The two kernels take different page tables by design -- the fused one
        takes the compact page_size=64 table and derives the physical slot
        in-kernel as ``pt64[row, p >> 6] * 64 + (p & 63)``, which is the
        definition of page_table_1 -- so the reference is given the equivalent
        wide table. Both must therefore return the same physical slots.

        Compared as per-row sets: both select the top-k by score, and neither
        contract fixes the order among the winners.
        """
        import aiter

        rows, ctx = 8, 8192
        logits, ghist, (_, _, _, seqlens, page_table_64, _) = self._fused_logits(
            rows, ctx
        )

        cap = max(TOPK, ctx)
        out = torch.empty(rows, TOPK, dtype=torch.int32, device=self.dev)
        self.topk_mod.topk_transform(
            logits,
            seqlens,
            page_table_64,
            out,
            ghist,
            torch.zeros(rows, 32, dtype=torch.int32, device=self.dev),
            torch.zeros(rows, dtype=torch.int32, device=self.dev),
            torch.empty(rows, cap, dtype=torch.int32, device=self.dev),
            torch.empty(rows, cap, dtype=torch.float32, device=self.dev),
            self.loader.TOPK_G,
            PAGE_SIZE,
        )
        torch.cuda.synchronize()

        # pt1[row, p] = pt64[row, p >> 6] * 64 + (p & 63)
        page_table_1 = (
            page_table_64.to(torch.int64).repeat_interleave(PAGE_SIZE, dim=1)
            * PAGE_SIZE
            + torch.arange(PAGE_SIZE, device=self.dev)
            .repeat(page_table_64.shape[1])
            .unsqueeze(0)
        ).to(torch.int32)
        ref = logits.new_full((rows, TOPK), -1, dtype=torch.int32)
        aiter.dsa_topk_transform(
            logits, None, seqlens, page_table_1, ref, 1, TOPK, ptRowMap=None
        )
        torch.cuda.synchronize()

        for r in range(rows):
            got_r = set(out[r].tolist()) - {-1}
            ref_r = set(ref[r].tolist()) - {-1}
            self.assertEqual(
                len(got_r),
                min(TOPK, ctx),
                f"row {r}: fused top-k returned {len(got_r)} distinct slots",
            )
            self.assertEqual(
                got_r, ref_r, f"row {r}: fused and aiter selected different slots"
            )

    def test_topk_transform_restores_its_histogram(self):
        """The zero-in/zero-out invariant the shared workspace depends on.

        One ``ghist`` serves all 79 layers back to back, so a kernel that leaves
        it dirty corrupts the next layer rather than failing here.
        """
        rows, ctx = 4, 4096
        logits, ghist, (_, _, _, seqlens, page_table_64, _) = self._fused_logits(
            rows, ctx
        )
        cap = max(TOPK, ctx)
        self.topk_mod.topk_transform(
            logits,
            seqlens,
            page_table_64,
            torch.empty(rows, TOPK, dtype=torch.int32, device=self.dev),
            ghist,
            torch.zeros(rows, 32, dtype=torch.int32, device=self.dev),
            torch.zeros(rows, dtype=torch.int32, device=self.dev),
            torch.empty(rows, cap, dtype=torch.int32, device=self.dev),
            torch.empty(rows, cap, dtype=torch.float32, device=self.dev),
            self.loader.TOPK_G,
            PAGE_SIZE,
        )
        torch.cuda.synchronize()
        self.assertEqual(
            int(ghist.abs().sum()), 0, "topk_transform left its histogram dirty"
        )


if __name__ == "__main__":
    unittest.main()
