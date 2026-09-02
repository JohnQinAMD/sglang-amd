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
        self.kv = _fp8(4096, 576)

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
        # bs = 1, 3, 4, 8 under MTP5 -> T = bs * 6
        for bs in (1, 3, 4, 8):
            T = bs * 6
            dec, pre = self._gates(T)
            self.assertTrue(dec, f"verify T={T} should pass the decode gate")
            self.assertFalse(
                pre, f"verify T={T} was never going to pass the prefill gate"
            )

    def test_the_decode_gate_bound_is_the_only_verify_bound(self):
        """Verify rows are limited by seq <= 96, not by a separate cap.

        FlyDSL beats TileLang across the whole 1..96 range, so the dispatch has
        no reason to stop earlier than the kernel contract does.
        """
        self.assertTrue(self._gates(96)[0], "bs=16 under MTP5 must still dispatch")
        self.assertFalse(self._gates(102)[0], "past the kernel's seq <= 96 contract")

    def test_prefill_shapes_still_go_to_the_prefill_gate(self):
        for T in (384, 2048, 8192, 32768):
            dec, pre = self._gates(T)
            self.assertTrue(pre, f"prefill T={T} should pass the prefill gate")
            if T > 96:
                self.assertFalse(
                    dec, f"prefill T={T} is past the decode gate's seq <= 96"
                )

    def test_the_two_gates_do_not_both_claim_a_shape(self):
        for T in (6, 24, 96, 256, 384, 2048):
            dec, pre = self._gates(T)
            self.assertFalse(
                dec and pre,
                f"T={T} is claimed by both gates, so the dispatch is ambiguous",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
