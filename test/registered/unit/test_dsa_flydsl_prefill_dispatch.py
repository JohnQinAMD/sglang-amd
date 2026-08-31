"""CPU coverage for the opt-in FlyDSL sparse-MLA prefill dispatch gate."""

import unittest
from unittest.mock import patch

import torch
from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDsaFlydslPrefillDispatch(CustomTestCase):
    def setUp(self):
        self.q_nope = torch.empty((512, 16, 512), dtype=torch.float8_e4m3fn)
        self.q_rope = torch.empty((512, 16, 64), dtype=torch.float8_e4m3fn)
        self.kv = torch.empty((4096, 1, 576), dtype=torch.float8_e4m3fn)
        self.indices = torch.empty((512, 2048), dtype=torch.int32)

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

    @patch.object(dsa_backend, "_IS_GFX950", True)
    @patch.object(dsa_backend, "_DSA_FLYDSL_PREFILL", True)
    def test_unvalidated_inputs_fall_back(self):
        cases = {
            "short sequence": {"q_nope": self.q_nope[:511]},
            "bf16 q": {"q_nope": self.q_nope.bfloat16()},
            "fnuz kv": {"kv_cache": self.kv.to(torch.float8_e4m3fnuz)},
            "wrong topk": {"page_table": self.indices[:, :-1]},
            "noncontiguous indices": {"page_table": self.indices.t()},
        }
        for name, replacements in cases.items():
            with self.subTest(name=name):
                self.assertFalse(self._can_use(**replacements))

    @patch.object(dsa_backend, "_IS_GFX950", True)
    @patch.object(dsa_backend, "_DSA_FLYDSL_PREFILL", False)
    def test_default_off_falls_back(self):
        self.assertFalse(self._can_use())


if __name__ == "__main__":
    unittest.main()
