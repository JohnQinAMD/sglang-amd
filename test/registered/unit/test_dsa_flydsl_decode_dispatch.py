"""CPU coverage for the FlyDSL sparse-MLA decode dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDsaFlydslDecodeDispatch(CustomTestCase):
    @staticmethod
    def _tensor(shape, dtype, *, device="cuda:0", contiguous=True):
        tensor = MagicMock(spec=torch.Tensor)
        tensor.shape = torch.Size(shape)
        tensor.ndim = len(shape)
        tensor.dtype = dtype
        tensor.device = torch.device(device)
        tensor.is_contiguous.return_value = contiguous
        return tensor

    def setUp(self):
        self.q = self._tensor((6, 16, 576), torch.float8_e4m3fn)
        self.kv = self._tensor((62000, 1, 576), torch.float8_e4m3fn)
        self.indices = self._tensor((6, 2048), torch.int32)

    def _can_use(self, **replacements):
        head_dim = replacements.pop("head_dim", 576)
        v_head_dim = replacements.pop("v_head_dim", 512)
        values = {
            "q_all": self.q,
            "kv_cache": self.kv,
            "page_table": self.indices,
        }
        values.update(replacements)
        return dsa_backend._can_use_flydsl_sparse_mla_decode(
            **values,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
        )

    @staticmethod
    def _backend_for_forward_test(kv):
        backend = dsa_backend.DeepseekSparseAttnBackend.__new__(
            dsa_backend.DeepseekSparseAttnBackend
        )
        backend.dsa_decode_impl = "flydsl"
        backend.dsa_prefill_impl = "tilelang"
        backend.use_mha = False
        backend.use_fused_topk = True
        backend.hisparse_coordinator = None
        backend.forward_metadata = SimpleNamespace()
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=MagicMock(return_value=kv)
        )
        backend.get_topk_transform_method = MagicMock(return_value=None)
        backend._get_fused_topk_page_table = MagicMock(side_effect=lambda x: x)
        return backend

    @patch.object(dsa_backend, "_IS_GFX950", True)
    def test_validated_shapes_are_accepted(self):
        self.assertTrue(self._can_use())
        self.assertTrue(
            self._can_use(
                q_all=self._tensor((1, 16, 576), torch.float8_e4m3fn),
                page_table=self._tensor((1, 2048), torch.int32),
            )
        )
        for seq in (24, 30, 60, 96):
            self.assertTrue(
                self._can_use(
                    q_all=self._tensor((seq, 16, 576), torch.float8_e4m3fn),
                    page_table=self._tensor((seq, 2048), torch.int32),
                )
            )
        self.assertTrue(self._can_use(page_table=self._tensor((6, 64), torch.int32)))
        self.assertTrue(self._can_use(page_table=self._tensor((6, 2112), torch.int32)))

    @patch.object(dsa_backend, "_IS_GFX950", True)
    def test_unvalidated_inputs_fall_back(self):
        cases = {
            "seq above scope": {
                "q_all": self._tensor((97, 16, 576), torch.float8_e4m3fn),
                "page_table": self._tensor((97, 2048), torch.int32),
            },
            "wrong qk dim": {"head_dim": 640},
            "wrong v dim": {"v_head_dim": 448},
            "bf16 q": {"q_all": self._tensor((6, 16, 576), torch.bfloat16)},
            "noncontiguous q": {
                "q_all": self._tensor(
                    (6, 16, 576), torch.float8_e4m3fn, contiguous=False
                )
            },
            "fnuz kv": {
                "kv_cache": self._tensor((62000, 1, 576), torch.float8_e4m3fnuz)
            },
            "CPU q": {
                "q_all": self._tensor((6, 16, 576), torch.float8_e4m3fn, device="cpu")
            },
            "different device": {
                "kv_cache": self._tensor(
                    (62000, 1, 576), torch.float8_e4m3fn, device="cuda:1"
                )
            },
            "unaligned topk": {"page_table": self._tensor((6, 2047), torch.int32)},
            "too-wide topk": {"page_table": self._tensor((6, 2176), torch.int32)},
            "noncontiguous indices": {
                "page_table": self._tensor((6, 2048), torch.int32, contiguous=False)
            },
        }
        for name, replacements in cases.items():
            with self.subTest(name=name):
                self.assertFalse(self._can_use(**replacements))

    def test_flydsl_is_an_opt_in_backend_choice(self):
        from sglang.srt.server_args import DSA_CHOICES

        self.assertIn("flydsl", DSA_CHOICES)
        self.assertIsNone(ServerArgs.dsa_decode_backend)

    @patch.object(dsa_backend, "_IS_GFX950", False)
    def test_non_gfx950_falls_back(self):
        self.assertFalse(self._can_use())

    def test_speculative_extend_modes_use_decode_kernel(self):
        q = torch.empty((6, 16, 576), dtype=torch.bfloat16)
        kv = torch.empty((64, 1, 576), dtype=torch.bfloat16)
        indices = torch.zeros((6, 2048), dtype=torch.int32)
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=0,
            tp_q_head_num=16,
            head_dim=576,
            v_head_dim=512,
            scaling=1.0,
        )
        expected = torch.empty((6, 16, 512), dtype=torch.bfloat16)

        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            with self.subTest(mode=mode):
                backend = self._backend_for_forward_test(kv)
                backend._try_flydsl_sparse_mla_decode = MagicMock(return_value=expected)

                result = backend.forward(
                    q,
                    None,
                    None,
                    layer,
                    SimpleNamespace(forward_mode=mode),
                    save_kv_cache=False,
                    q_rope=None,
                    topk_indices=indices,
                )

                self.assertIs(result, expected)
                backend._try_flydsl_sparse_mla_decode.assert_called_once()

    def test_speculative_extend_preserves_tilelang_fallback(self):
        q = torch.empty((6, 16, 576), dtype=torch.bfloat16)
        kv = torch.empty((64, 1, 576), dtype=torch.bfloat16)
        indices = torch.zeros((6, 2048), dtype=torch.int32)
        expected = torch.empty((6, 16, 512), dtype=torch.bfloat16)
        backend = self._backend_for_forward_test(kv)
        backend._try_flydsl_sparse_mla_decode = MagicMock(return_value=None)
        backend._forward_tilelang = MagicMock(return_value=expected)

        result = backend.forward(
            q,
            None,
            None,
            SimpleNamespace(
                is_cross_attention=False,
                layer_id=0,
                tp_q_head_num=16,
                head_dim=576,
                v_head_dim=512,
                scaling=1.0,
            ),
            SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
            save_kv_cache=False,
            q_rope=None,
            topk_indices=indices,
        )

        self.assertIs(result, expected)
        backend._forward_tilelang.assert_called_once()


if __name__ == "__main__":
    unittest.main()
