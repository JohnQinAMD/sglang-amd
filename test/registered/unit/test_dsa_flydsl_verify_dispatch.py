"""target-verify 应当走 FlyDSL 的 decode 门，而不是被 prefill 门挡掉。

MTP5 下 verify 的形状是 T = bs * num_draft_tokens（bs=4 时 24），每 token 一行
page table，topk 2048 —— 这是 decode 契约，不是 prefill 契约。prefill 门的
T>=256 下限是按 prefill 形状测出来的，会把 bs<43 的 verify 全部挡掉，于是
78 层的重路径一直在回落到 TileLang。
"""

import unittest

import torch

from sglang.srt.layers.attention import dsa_backend as B


def _fp8(*shape):
    return torch.zeros(*shape, dtype=torch.float8_e4m3fn, device="cuda")


class TestVerifyDispatch(unittest.TestCase):
    """只测两个门对同一批形状的判定，不触发实际 kernel。"""

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
            self.assertTrue(dec, f"verify T={T} 应命中 decode 门")
            self.assertLessEqual(T, B._FLYDSL_VERIFY_MAX_ROWS)
            self.assertFalse(pre, f"verify T={T} 本来就过不了 prefill 门")

    def test_verify_cap_matches_where_flydsl_still_wins(self):
        """上界必须落在实测的交叉点之前。

        实测（width 2048, fp8, HIP graph 内 device time）：
            T=24  FlyDSL 15.41 vs TileLang 22.97  = 0.67x   78 层省 3.07% 步时
            T=48  FlyDSL 25.09 vs TileLang 28.26  = 0.89x   省 1.29%
            T=96  FlyDSL 42.75 vs TileLang 41.83  = 1.02x   亏 0.37%
        所以上界取 48：T=96（bs=16）会让这条分派变成回归。
        """
        self.assertEqual(B._FLYDSL_VERIFY_MAX_ROWS, 48)
        self.assertLess(
            B._FLYDSL_VERIFY_MAX_ROWS,
            96,
            "T=96 时 FlyDSL 比 TileLang 慢，不能落在上界之内",
        )

    def test_prefill_shapes_still_go_to_the_prefill_gate(self):
        for T in (384, 2048, 8192, 32768):
            dec, pre = self._gates(T)
            self.assertTrue(pre, f"prefill T={T} 应命中 prefill 门")
            if T > 96:
                self.assertFalse(dec, f"prefill T={T} 超出 decode 门的 seq<=96")

    def test_the_two_gates_do_not_both_claim_a_shape(self):
        for T in (6, 24, 96, 256, 384, 2048):
            dec, pre = self._gates(T)
            self.assertFalse(dec and pre, f"T={T} 被两个门同时认领，分派有歧义")


if __name__ == "__main__":
    unittest.main(verbosity=2)
