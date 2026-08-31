"""FlyDSL decode 的持久 scratch。

aiter 要求 partial_output/partial_lse 连续且形状精确等于 [seq,ng,H,DV] /
[seq,ng,H] —— 对最大尺寸张量做二维切片得不到连续张量，所以实现用的是一维
扁平 buffer 加前缀 view。这些测试钉住那个契约。
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention import dsa_backend as B

H, DV = B._FLYDSL_DECODE_H, B._FLYDSL_DECODE_DV


class TestFlydslDecodeScratch(unittest.TestCase):
    def setUp(self):
        B._FLYDSL_DECODE_SCRATCH.clear()
        self.dev = torch.device("cuda:0")

    def test_shape_and_contiguity(self):
        for seq, ng in ((1, 32), (4, 32), (24, 32), (96, 33), (6, 1)):
            po, pl = B._flydsl_decode_scratch(seq, ng, self.dev)
            self.assertIsNotNone(po, f"seq={seq} ng={ng} 不应回落")
            self.assertEqual(tuple(po.shape), (seq, ng, H, DV))
            self.assertEqual(tuple(pl.shape), (seq, ng, H))
            self.assertTrue(po.is_contiguous(), "aiter 要求连续")
            self.assertTrue(pl.is_contiguous(), "aiter 要求连续")
            self.assertEqual(po.dtype, torch.bfloat16)
            self.assertEqual(pl.dtype, torch.float32)

    def test_one_allocation_serves_every_call(self):
        first, _ = B._flydsl_decode_scratch(4, 32, self.dev)
        base = B._FLYDSL_DECODE_SCRATCH[self.dev][0].data_ptr()
        for seq in range(1, 97):
            po, _ = B._flydsl_decode_scratch(seq, 32, self.dev)
            self.assertEqual(B._FLYDSL_DECODE_SCRATCH[self.dev][0].data_ptr(), base)
        self.assertEqual(len(B._FLYDSL_DECODE_SCRATCH), 1)

    def test_beyond_capacity_falls_back(self):
        B._flydsl_decode_scratch(4, 32, self.dev)          # 先建好 buffer
        po, pl = B._flydsl_decode_scratch(97, 33, self.dev)  # 超出门的上界
        self.assertIsNone(po)
        self.assertIsNone(pl)

    def test_capture_never_allocates(self):
        with mock.patch.object(
            torch.cuda, "is_current_stream_capturing", lambda: True
        ):
            po, pl = B._flydsl_decode_scratch(4, 32, self.dev)
        self.assertIsNone(po)
        self.assertIsNone(pl)
        self.assertEqual(len(B._FLYDSL_DECODE_SCRATCH), 0)

    def test_capacity_covers_the_gate(self):
        # 门允许 1<=seq<=96 且 64<=width<=2112 步长 64 -> ng 1..33
        B._flydsl_decode_scratch(1, 1, self.dev)
        cap_out = B._FLYDSL_DECODE_SCRATCH[self.dev][0].numel()
        self.assertGreaterEqual(cap_out, 96 * 33 * H * DV)

    def test_prealloc_creates_the_buffer_before_capture(self):
        self.assertEqual(len(B._FLYDSL_DECODE_SCRATCH), 0)
        B._flydsl_decode_scratch_prealloc(self.dev)
        self.assertEqual(len(B._FLYDSL_DECODE_SCRATCH), 1)
        # 预分配之后，捕获期也能拿到 buffer 而不是回落
        with mock.patch.object(
            torch.cuda, "is_current_stream_capturing", lambda: True
        ):
            po, pl = B._flydsl_decode_scratch(4, 32, self.dev)
        self.assertIsNotNone(po)
        self.assertIsNotNone(pl)

    def test_prealloc_is_a_noop_under_capture(self):
        with mock.patch.object(
            torch.cuda, "is_current_stream_capturing", lambda: True
        ):
            B._flydsl_decode_scratch_prealloc(self.dev)
        self.assertEqual(len(B._FLYDSL_DECODE_SCRATCH), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
