"""CPU coverage for the opt-in FlyDSL sparse-MLA decode dispatch gate."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention import dsa_backend
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

    @patch.object(dsa_backend, "_IS_GFX950", True)
    @patch.object(dsa_backend, "_DSA_FLYDSL_DECODE", True)
    def test_validated_shapes_are_accepted(self):
        self.assertTrue(self._can_use())
        self.assertTrue(
            self._can_use(
                q_all=self._tensor((1, 16, 576), torch.float8_e4m3fn),
                page_table=self._tensor((1, 2048), torch.int32),
            )
        )
        for seq in (48, 96):
            self.assertTrue(
                self._can_use(
                    q_all=self._tensor((seq, 16, 576), torch.float8_e4m3fn),
                    page_table=self._tensor((seq, 2048), torch.int32),
                )
            )
        self.assertTrue(self._can_use(page_table=self._tensor((6, 64), torch.int32)))
        self.assertTrue(self._can_use(page_table=self._tensor((6, 2112), torch.int32)))

    @patch.object(dsa_backend, "_IS_GFX950", True)
    @patch.object(dsa_backend, "_DSA_FLYDSL_DECODE", True)
    def test_unvalidated_inputs_fall_back(self):
        cases = {
            "seq above scope": {
                "q_all": self._tensor((25, 16, 576), torch.float8_e4m3fn),
                "page_table": self._tensor((25, 2048), torch.int32),
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

    @patch.object(dsa_backend, "_IS_GFX950", True)
    @patch.object(dsa_backend, "_DSA_FLYDSL_DECODE", False)
    def test_default_off_falls_back(self):
        self.assertFalse(self._can_use())

    @patch.object(dsa_backend, "_IS_GFX950", False)
    @patch.object(dsa_backend, "_DSA_FLYDSL_DECODE", True)
    def test_non_gfx950_falls_back(self):
        self.assertFalse(self._can_use())


if __name__ == "__main__":
    unittest.main()
