"""Persistent scratch for the FlyDSL decode kernel.

aiter requires partial_output/partial_lse to be contiguous and to have exactly
the shapes [seq,ng,H,DV] and [seq,ng,H]. A 2-D slice of a max-sized tensor is
not contiguous, so the implementation keeps one flat 1-D buffer and views a
prefix of it. These tests pin that contract down.
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
            self.assertIsNotNone(po, f"seq={seq} ng={ng} should not fall back")
            self.assertEqual(tuple(po.shape), (seq, ng, H, DV))
            self.assertEqual(tuple(pl.shape), (seq, ng, H))
            self.assertTrue(po.is_contiguous(), "aiter requires contiguous buffers")
            self.assertTrue(pl.is_contiguous(), "aiter requires contiguous buffers")
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
        B._flydsl_decode_scratch(4, 32, self.dev)  # build the buffer first
        po, pl = B._flydsl_decode_scratch(97, 33, self.dev)  # past the gate's upper bound
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
        # The gate admits 1 <= seq <= 96 and 64 <= width <= 2112 in steps of
        # 64, i.e. ng 1..33.
        B._flydsl_decode_scratch(1, 1, self.dev)
        cap_out = B._FLYDSL_DECODE_SCRATCH[self.dev][0].numel()
        self.assertGreaterEqual(cap_out, 96 * 33 * H * DV)

    def test_prealloc_creates_the_buffer_before_capture(self):
        self.assertEqual(len(B._FLYDSL_DECODE_SCRATCH), 0)
        B._flydsl_decode_scratch_prealloc(self.dev)
        self.assertEqual(len(B._FLYDSL_DECODE_SCRATCH), 1)
        # After preallocation a capture gets the buffer instead of falling back
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
