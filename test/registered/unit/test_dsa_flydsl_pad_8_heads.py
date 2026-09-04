"""TP=8 gives 8 q heads per rank; the FlyDSL kernel's H is fixed at 16.

The 8 are padded to 16 and the upper half dropped, which is sound only if the
padded heads cannot reach the real ones. That is what this file pins.
"""

import math
import unittest

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

DV = 512
ROPE = 64
DIM = DV + ROPE  # 576
WIDTH = 2048
CTX = 8192


def _flydsl_ready() -> bool:
    if not torch.cuda.is_available() or not dsa_backend._IS_GFX950:
        return False
    try:
        from aiter.ops.flydsl import flydsl_sparse_mla_decode  # noqa: F401
    except ImportError:
        return False
    return True


def _inputs(seq, heads, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn((seq, heads, DIM), device="cuda", generator=g) * 0.25).clamp_(
        -2.0, 2.0
    )
    kv = (torch.randn((CTX, DIM), device="cuda", generator=g) * 0.25).clamp_(-2.0, 2.0)
    idx = torch.empty((seq, WIDTH), dtype=torch.int32, device="cuda")
    base = torch.arange(WIDTH, dtype=torch.int32, device="cuda")
    for row in range(seq):
        idx[row] = (base + row * 337) % CTX
    return (
        q.to(torch.float8_e4m3fn).contiguous(),
        kv.to(torch.float8_e4m3fn).contiguous(),
        idx.contiguous(),
    )


def _run_flydsl(q, kv, idx):
    from aiter.ops.flydsl import flydsl_sparse_mla_decode

    out = torch.empty(
        (q.shape[0], q.shape[1], DV), dtype=torch.bfloat16, device=q.device
    )
    flydsl_sparse_mla_decode(
        q=q, kv=kv, indices=idx, out=out, sm_scale=1.0 / math.sqrt(DIM)
    )
    return out


def _snr_db(actual, ref):
    a, r = actual.double(), ref.double()
    noise = torch.linalg.vector_norm((a - r).reshape(-1)).item()
    signal = torch.linalg.vector_norm(r.reshape(-1)).item()
    if noise == 0.0:
        return float("inf")
    return 20.0 * math.log10(signal / noise)


class TestFlydslGateAcceptsEightHeads(CustomTestCase):
    def _reason(self, heads):
        from unittest.mock import MagicMock

        def t(shape, dtype):
            m = MagicMock(spec=torch.Tensor)
            m.shape = torch.Size(shape)
            m.ndim = len(shape)
            m.dtype = dtype
            m.device = torch.device("cuda:0")
            m.is_contiguous.return_value = True
            return m

        return dsa_backend._flydsl_decode_decline_reason(
            t((6, heads, DIM), torch.bfloat16),
            t((CTX, 1, DIM), torch.float8_e4m3fn),
            t((6, WIDTH), torch.int32),
            DIM,
            DV,
        )

    def test_eight_heads_is_not_the_clause_that_fails(self):
        # Before the padding path this named the shape and TileLang ran.
        self.assertNotIn("q shape", self._reason(8))

    def test_sixteen_heads_still_accepted(self):
        self.assertNotIn("q shape", self._reason(16))

    def test_twelve_heads_still_rejected_by_shape(self):
        self.assertIn("q shape", self._reason(12))


@unittest.skipUnless(_flydsl_ready(), "needs gfx950 with FlyDSL installed")
class TestPaddingCannotReachTheRealHeads(CustomTestCase):
    def test_padded_half_does_not_change_the_first_eight(self):
        """Same real heads, two different upper halves -- same answer."""
        for seq in (1, 6, 24):
            with self.subTest(seq=seq):
                q8, kv, idx = _inputs(seq, 8, 400 + seq)
                zeros = torch.zeros(
                    (seq, 16, DIM), dtype=torch.float8_e4m3fn, device="cuda"
                )
                zeros[:, :8] = q8
                loud = torch.full(
                    (seq, 16, DIM), 6.0, device="cuda", dtype=torch.bfloat16
                ).to(torch.float8_e4m3fn)
                loud[:, :8] = q8

                a = _run_flydsl(zeros, kv, idx)[:, :8]
                b = _run_flydsl(loud, kv, idx)[:, :8]
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_padded_output_is_finite(self):
        q8, kv, idx = _inputs(6, 8, 401)
        padded = torch.zeros((6, 16, DIM), dtype=torch.float8_e4m3fn, device="cuda")
        padded[:, :8] = q8
        self.assertTrue(torch.isfinite(_run_flydsl(padded, kv, idx)[:, :8]).all())

    def test_matches_the_tilelang_path_it_displaces(self):
        """Against the answer production gets today at TP=8."""
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        seq = 6
        q8, kv, idx = _inputs(seq, 8, 402)
        padded = torch.zeros((seq, 16, DIM), dtype=torch.float8_e4m3fn, device="cuda")
        padded[:, :8] = q8
        mine = _run_flydsl(padded, kv, idx)[:, :8]
        theirs = tilelang_sparse_fwd(
            q=q8,
            kv=kv.unsqueeze(1),
            indices=idx.unsqueeze(1),
            sm_scale=1.0 / math.sqrt(DIM),
            d_v=DV,
        )
        # Same fp8 products in a different order.
        self.assertGreater(_snr_db(mine, theirs), 30.0)


@unittest.skipUnless(_flydsl_ready(), "needs gfx950 with FlyDSL installed")
class TestBackendTakesTheEightHeadPath(CustomTestCase):
    """Through the code that pads, not around it."""

    @staticmethod
    def _call(seq, heads, seed):
        from types import SimpleNamespace

        q, kv, idx = _inputs(seq, heads, seed)
        layer = SimpleNamespace(
            head_dim=DIM,
            v_head_dim=DV,
            tp_q_head_num=heads,
            scaling=1.0 / math.sqrt(DIM),
        )
        return dsa_backend.DeepseekSparseAttnBackend._try_flydsl_sparse_mla_decode(
            q, kv.unsqueeze(1), idx, layer
        )

    def test_eight_heads_returns_eight_heads(self):
        out = self._call(6, 8, 500)
        self.assertIsNotNone(out, "the backend declined 8 heads instead of padding")
        self.assertEqual(tuple(out.shape), (6, 8, DV))
        self.assertTrue(out.is_contiguous())
        self.assertTrue(torch.isfinite(out).all())

    def test_eight_head_answer_matches_tilelang(self):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        q, kv, idx = _inputs(6, 8, 501)
        mine = self._call(6, 8, 501)
        theirs = tilelang_sparse_fwd(
            q=q,
            kv=kv.unsqueeze(1),
            indices=idx.unsqueeze(1),
            sm_scale=1.0 / math.sqrt(DIM),
            d_v=DV,
        )
        self.assertGreater(_snr_db(mine, theirs), 30.0)


if __name__ == "__main__":
    unittest.main()
