"""CPU coverage for the opt-in FlyDSL sparse-MLA decode dispatch gate."""

import unittest
from unittest.mock import patch

import torch
from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDsaFlydslDecodeDispatch(CustomTestCase):
    def setUp(self):
        self.q = torch.empty((6, 16, 576), dtype=torch.float8_e4m3fn)
        self.kv = torch.empty((62000, 1, 576), dtype=torch.float8_e4m3fn)
        self.indices = torch.empty((6, 2048), dtype=torch.int32)

    def _can_use(self, **replacements):
        values = {
            "q_all": self.q,
            "kv_cache": self.kv,
            "page_table": self.indices,
        }
        values.update(replacements)
        return dsa_backend._can_use_flydsl_sparse_mla_decode(**values)

    @patch.object(dsa_backend, "_IS_GFX950", True)
    @patch.object(dsa_backend, "_DSA_FLYDSL_DECODE", True)
    def test_validated_shapes_are_accepted(self):
        self.assertTrue(self._can_use())
        self.assertTrue(
            self._can_use(
                q_all=self.q[:1],
                page_table=self.indices[:1],
            )
        )
        self.assertTrue(self._can_use(page_table=self.indices[:, :64].contiguous()))
        padded = torch.empty((6, 2112), dtype=torch.int32)
        self.assertTrue(self._can_use(page_table=padded))

    @patch.object(dsa_backend, "_IS_GFX950", True)
    @patch.object(dsa_backend, "_DSA_FLYDSL_DECODE", True)
    def test_unvalidated_inputs_fall_back(self):
        cases = {
            "wrong seq": {"q_all": self.q[:2], "page_table": self.indices[:2]},
            "bf16 q": {"q_all": self.q.bfloat16()},
            "fnuz kv": {"kv_cache": self.kv.to(torch.float8_e4m3fnuz)},
            "unaligned topk": {"page_table": self.indices[:, :-1]},
            "too-wide topk": {"page_table": torch.empty((6, 2176), dtype=torch.int32)},
            "noncontiguous indices": {"page_table": self.indices.t()},
        }
        for name, replacements in cases.items():
            with self.subTest(name=name):
                self.assertFalse(self._can_use(**replacements))

    @patch.object(dsa_backend, "_IS_GFX950", True)
    @patch.object(dsa_backend, "_DSA_FLYDSL_DECODE", False)
    def test_default_off_falls_back(self):
        self.assertFalse(self._can_use())


if __name__ == "__main__":
    unittest.main()
