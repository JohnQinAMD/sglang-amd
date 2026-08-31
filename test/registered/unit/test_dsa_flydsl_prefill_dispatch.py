"""CPU coverage for the opt-in FlyDSL sparse-MLA prefill dispatch gate."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDsaFlydslPrefillDispatch(CustomTestCase):
    @staticmethod
    def _tensor(shape, dtype, *, device="cuda:0", contiguous=True, strides=None):
        tensor = MagicMock(spec=torch.Tensor)
        tensor.shape = torch.Size(shape)
        tensor.ndim = len(shape)
        tensor.dtype = dtype
        tensor.device = torch.device(device)
        tensor.is_contiguous.return_value = contiguous
        if strides is None:
            strides = []
            stride = 1
            for size in reversed(shape):
                strides.append(stride)
                stride *= size
            strides = tuple(reversed(strides))
        tensor.stride.side_effect = lambda dim=None: (
            strides if dim is None else strides[dim]
        )
        return tensor

    def setUp(self):
        self.q_nope = self._tensor((512, 16, 512), torch.float8_e4m3fn)
        self.q_rope = self._tensor((512, 16, 64), torch.float8_e4m3fn)
        self.kv = self._tensor((4096, 1, 576), torch.float8_e4m3fn)
        self.indices = self._tensor((512, 2048), torch.int32)

    def _can_use(self, **replacements):
        values = {
            "q_nope": self.q_nope,
            "q_rope": self.q_rope,
            "kv_cache": self.kv,
            "page_table": self.indices,
        }
        values.update(replacements)
        return dsa_backend._can_use_flydsl_sparse_mla_prefill(**values)

    @patch.object(dsa_backend, "_IS_GFX950", True)
    @patch.object(dsa_backend, "_DSA_FLYDSL_PREFILL", True)
    def test_validated_shape_is_accepted(self):
        self.assertTrue(self._can_use())
        production_strides = (16 * 576, 576, 1)
        self.assertTrue(
            self._can_use(
                q_nope=self._tensor(
                    (512, 16, 512),
                    torch.float8_e4m3fn,
                    contiguous=False,
                    strides=production_strides,
                ),
                q_rope=self._tensor(
                    (512, 16, 64),
                    torch.float8_e4m3fn,
                    contiguous=False,
                    strides=production_strides,
                ),
            )
        )

    @patch.object(dsa_backend, "_IS_GFX950", True)
    @patch.object(dsa_backend, "_DSA_FLYDSL_PREFILL", True)
    def test_unvalidated_inputs_fall_back(self):
        cases = {
            "short sequence": {
                "q_nope": self._tensor((255, 16, 512), torch.float8_e4m3fn),
                "q_rope": self._tensor((255, 16, 64), torch.float8_e4m3fn),
                "page_table": self._tensor((255, 2048), torch.int32),
            },
            "bf16 q": {"q_nope": self._tensor((512, 16, 512), torch.bfloat16)},
            "unsupported inner stride": {
                "q_nope": self._tensor(
                    (512, 16, 512),
                    torch.float8_e4m3fn,
                    contiguous=False,
                    strides=(16 * 576 * 2, 576 * 2, 2),
                )
            },
            "fnuz kv": {
                "kv_cache": self._tensor((4096, 1, 576), torch.float8_e4m3fnuz)
            },
            "CPU q": {
                "q_nope": self._tensor(
                    (512, 16, 512), torch.float8_e4m3fn, device="cpu"
                )
            },
            "different device": {
                "q_rope": self._tensor(
                    (512, 16, 64), torch.float8_e4m3fn, device="cuda:1"
                )
            },
            "wrong topk": {"page_table": self._tensor((512, 2047), torch.int32)},
            "noncontiguous indices": {
                "page_table": self._tensor((512, 2048), torch.int32, contiguous=False)
            },
        }
        for name, replacements in cases.items():
            with self.subTest(name=name):
                self.assertFalse(self._can_use(**replacements))

    @patch.object(dsa_backend, "_IS_GFX950", True)
    @patch.object(dsa_backend, "_DSA_FLYDSL_PREFILL", False)
    def test_default_off_falls_back(self):
        self.assertFalse(self._can_use())

    @patch.object(dsa_backend, "_IS_GFX950", False)
    @patch.object(dsa_backend, "_DSA_FLYDSL_PREFILL", True)
    def test_non_gfx950_falls_back(self):
        self.assertFalse(self._can_use())


if __name__ == "__main__":
    unittest.main()
