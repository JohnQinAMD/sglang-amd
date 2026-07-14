"""Gate-routing test for EAGLE verify: sampling vs. greedy (argmax).

Guards ``eagle_utils._verify_uses_greedy``: verify commits greedy (argmax) iff the
batch is all-greedy or the platform has no sampling-verify kernel (CPU/NPU/XPU). A
sampling-capable platform -- CUDA, and now ROCm/HIP (which has the pure-Triton chain
and tree samplers) -- takes the sampling path whenever the batch isn't all-greedy.
The guarded regressions: re-adding HIP (or CUDA) to the greedy predicate would
silently drop temperature/top_p on that platform; letting CPU/NPU/XPU sample would
call a kernel that isn't there. Pure-boolean logic, runs on CPU CI.
"""

import unittest

from sglang.srt.speculative.eagle_utils import _verify_uses_greedy
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# is_cpu/is_npu/is_xpu flags. CUDA and HIP set none of them -> sampling-capable.
_NO_KERNEL = {
    "cpu": (True, False, False),
    "npu": (False, True, False),
    "xpu": (False, False, True),
}


def _gate(is_all_greedy, is_cpu=False, is_npu=False, is_xpu=False):
    return _verify_uses_greedy(
        is_all_greedy=is_all_greedy, is_cpu=is_cpu, is_npu=is_npu, is_xpu=is_xpu
    )


class TestEagleGateRouting(CustomTestCase):
    def test_sampling_platform_samples_iff_not_all_greedy(self):
        # CUDA/HIP (no kernel-missing flag): sample when not all-greedy, greedy when it is.
        self.assertFalse(_gate(is_all_greedy=False))
        self.assertTrue(_gate(is_all_greedy=True))

    def test_no_kernel_platforms_always_greedy(self):
        # CPU/NPU/XPU have no sampling-verify kernel -> always greedy, even non-greedy batch.
        for name, (c, n, x) in _NO_KERNEL.items():
            self.assertTrue(_gate(False, c, n, x), f"{name} must force greedy")
            self.assertTrue(_gate(True, c, n, x), f"{name} must force greedy")


if __name__ == "__main__":
    unittest.main()
