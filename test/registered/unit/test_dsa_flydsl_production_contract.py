"""What the FlyDSL gates must accept, taken from what production produces.

The existing gate tests all passed while both gates declined every call in a
real server for twenty-seven hours. They asserted the gates reject what they
should reject, and they built their accepted case out of mocks that happened to
carry the dtype and layout the gate wanted -- never the dtype and layout the
serving path actually hands over. This file pins that second thing.

Two regressions it would have caught:

  the pool width -- calculate_mla_kv_cache_dim whitelists the HIP backends that
  read the raw 576 MLA layout, and flydsl was added to DSA_CHOICES without
  being added there, so the pool came out 656 and every kernel on the platform
  refused it.

  the q dtype -- q reaches forward_extend as bf16 from the MLA absorb bmm, and
  both gates required float8_e4m3fn.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

KV_LORA_RANK = 512
ROPE_DIM = 64
RAW_MLA_WIDTH = KV_LORA_RANK + ROPE_DIM  # 576


def _tensor(shape, dtype, *, device="cuda:0", contiguous=True):
    t = MagicMock(spec=torch.Tensor)
    t.shape = torch.Size(shape)
    t.ndim = len(shape)
    t.dtype = dtype
    t.device = torch.device(device)
    t.is_contiguous.return_value = contiguous
    return t


class TestFlydslPoolWidth(CustomTestCase):
    """The pool a FlyDSL server gets must be the one its kernels can read."""

    @staticmethod
    def _dim(prefill_backend, decode_backend, *, is_hip=True):
        from sglang.srt.mem_cache import kv_cache_configurator as cfg

        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(),
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
        )
        server_args = SimpleNamespace(
            dsa_prefill_backend=prefill_backend,
            dsa_decode_backend=decode_backend,
        )
        with patch.object(cfg, "_is_hip", is_hip), patch.object(
            cfg, "is_deepseek_dsa", return_value=True
        ):
            return cfg.calculate_mla_kv_cache_dim(
                model_config=model_config,
                kv_cache_dtype=torch.float8_e4m3fn,
                server_args=server_args,
            )

    def test_flydsl_gets_the_raw_layout_like_tilelang_and_aiter(self):
        for backend in ("tilelang", "aiter", "flydsl"):
            with self.subTest(backend=backend):
                self.assertEqual(self._dim(backend, backend), RAW_MLA_WIDTH)

    def test_one_side_on_flydsl_is_enough(self):
        # The whitelist is an `or`: a run with only the decode backend on
        # flydsl still needs the raw layout, because the pool is shared.
        self.assertEqual(self._dim("tilelang", "flydsl"), RAW_MLA_WIDTH)
        self.assertEqual(self._dim("flydsl", "tilelang"), RAW_MLA_WIDTH)

    def test_the_scaled_layout_is_what_we_are_avoiding(self):
        # Not a HIP platform: the CUDA scaled layout is correct there, and it
        # is 656 -- the number that reached the HIP kernels as a mismatch.
        scaled = self._dim("flashmla_sparse", "flashmla_sparse", is_hip=False)
        self.assertEqual(scaled, KV_LORA_RANK + KV_LORA_RANK // 128 * 4 + ROPE_DIM * 2)
        self.assertNotEqual(scaled, RAW_MLA_WIDTH)


class TestFlydslAcceptsProductionQ(CustomTestCase):
    """q arrives bf16. Both gates must take it; the caller casts."""

    def setUp(self):
        self.kv = _tensor((62000, 1, RAW_MLA_WIDTH), torch.float8_e4m3fn)
        self.page_table = _tensor((6, 2048), torch.int32)

    def _decode_gate(self, q):
        with patch.object(dsa_backend, "_IS_GFX950", True):
            return dsa_backend._can_use_flydsl_sparse_mla_decode(
                q, self.kv, self.page_table, head_dim=576, v_head_dim=512
            )

    def test_decode_gate_takes_bf16_q(self):
        q = _tensor((6, 16, RAW_MLA_WIDTH), torch.bfloat16)
        self.assertTrue(self._decode_gate(q))

    def test_decode_gate_still_takes_fp8_q(self):
        q = _tensor((6, 16, RAW_MLA_WIDTH), torch.float8_e4m3fn)
        self.assertTrue(self._decode_gate(q))

    def test_decode_gate_refuses_fp16_q(self):
        # Widening the dtype set is not the same as removing the check.
        q = _tensor((6, 16, RAW_MLA_WIDTH), torch.float16)
        self.assertFalse(self._decode_gate(q))

    def test_prefill_gate_takes_bf16_q(self):
        q_nope = _tensor((512, 16, KV_LORA_RANK), torch.bfloat16)
        q_rope = _tensor((512, 16, ROPE_DIM), torch.bfloat16)
        page_table = _tensor((512, 2048), torch.int32)
        with patch.object(dsa_backend, "_IS_GFX950", True), patch.object(
            dsa_backend, "_is_flydsl_prefill_q_layout", return_value=True
        ):
            self.assertTrue(
                dsa_backend._can_use_flydsl_sparse_mla_prefill(
                    q_nope, q_rope, self.kv, page_table
                )
            )

    def test_the_decline_reason_names_the_clause(self):
        # A gate that says only "outside the gate" is what let the pool bug
        # survive a 3600 s round. Each reason has to be specific enough to act
        # on without reading the gate.
        wrong_dtype = _tensor((6, 16, RAW_MLA_WIDTH), torch.float16)
        with patch.object(dsa_backend, "_IS_GFX950", True):
            reason = dsa_backend._flydsl_decode_decline_reason(
                wrong_dtype, self.kv, self.page_table, 576, 512
            )
        self.assertIn("dtype", reason)
        self.assertIn("float16", reason)

        wide_kv = _tensor((62000, 1, 656), torch.float8_e4m3fn)
        q = _tensor((6, 16, RAW_MLA_WIDTH), torch.bfloat16)
        with patch.object(dsa_backend, "_IS_GFX950", True):
            reason = dsa_backend._flydsl_decode_decline_reason(
                q, wide_kv, self.page_table, 576, 512
            )
        self.assertIn("656", reason)
        self.assertIn("calculate_mla_kv_cache_dim", reason)


if __name__ == "__main__":
    unittest.main()
