"""Target verify belongs at the FlyDSL decode gate, not behind the prefill gate.

Under MTP5 a verify batch has shape T = bs * num_draft_tokens (24 at bs=4), one
page-table row per token and topk 2048. That is the decode contract, not the
prefill one. The prefill gate's T >= 256 floor was measured on prefill shapes
and rejects every verify batch below bs=43, which left the hot path on all 78
layers falling back to TileLang.
"""

import unittest

import torch

from sglang.srt.layers.attention import dsa_backend as B


def _fp8(*shape):
    return torch.zeros(*shape, dtype=torch.float8_e4m3fn, device="cuda")


class TestVerifyDispatch(unittest.TestCase):
    """Only the two gates' verdicts on one set of shapes; no kernel is launched."""

    def setUp(self):
        if not B._IS_GFX950:
            self.skipTest("gfx950 only")
        self._decode = B._DSA_FLYDSL_DECODE
        self._prefill = B._DSA_FLYDSL_PREFILL
        B._DSA_FLYDSL_DECODE = True
        B._DSA_FLYDSL_PREFILL = True
        self.kv = _fp8(4096, 576)

    def tearDown(self):
        B._DSA_FLYDSL_DECODE = self._decode
        B._DSA_FLYDSL_PREFILL = self._prefill

    def _gates(self, T):
        q_all = _fp8(T, 16, 576)
        q_nope, q_rope = q_all[:, :, :512], q_all[:, :, 512:]
        pt = torch.zeros(T, 2048, dtype=torch.int32, device="cuda")
        dec = B._can_use_flydsl_sparse_mla_decode(
            q_all, self.kv, pt, head_dim=576, v_head_dim=512
        )
        pre = B._can_use_flydsl_sparse_mla_prefill(q_nope, q_rope, self.kv, pt)
        return dec, pre

    def test_verify_shapes_hit_the_decode_gate(self):
        # bs = 1, 3, 4, 8 under MTP5 -> T = bs * 6, all within the cap
        for bs in (1, 3, 4, 8):
            T = bs * 6
            dec, pre = self._gates(T)
            self.assertTrue(dec, f"verify T={T} should pass the decode gate")
            self.assertLessEqual(T, B._FLYDSL_VERIFY_MAX_ROWS)
            self.assertFalse(pre, f"verify T={T} was never going to pass the prefill gate")

    def test_verify_cap_matches_where_flydsl_still_wins(self):
        """The cap has to sit before the measured crossover.

        Measured at width 2048, fp8, device time inside a HIP graph:
            T=24  FlyDSL 15.41 vs TileLang 22.97 = 0.67x  saves 3.07% of a step
                                                          over 78 layers
            T=48  FlyDSL 25.09 vs TileLang 28.26 = 0.89x  saves 1.29%
            T=96  FlyDSL 42.75 vs TileLang 41.83 = 1.02x  loses 0.37%
        Hence 48: at T=96 (bs=16) this dispatch would become a regression.
        """
        self.assertEqual(B._FLYDSL_VERIFY_MAX_ROWS, 48)
        self.assertLess(
            B._FLYDSL_VERIFY_MAX_ROWS,
            96,
            "FlyDSL is slower than TileLang at T=96, so it must not be under the cap",
        )

    def test_prefill_shapes_still_go_to_the_prefill_gate(self):
        for T in (384, 2048, 8192, 32768):
            dec, pre = self._gates(T)
            self.assertTrue(pre, f"prefill T={T} should pass the prefill gate")
            if T > 96:
                self.assertFalse(dec, f"prefill T={T} is past the decode gate's seq <= 96")

    def test_the_two_gates_do_not_both_claim_a_shape(self):
        for T in (6, 24, 96, 256, 384, 2048):
            dec, pre = self._gates(T)
            self.assertFalse(dec and pre, f"T={T} is claimed by both gates, so the dispatch is ambiguous")


if __name__ == "__main__":
    unittest.main(verbosity=2)
