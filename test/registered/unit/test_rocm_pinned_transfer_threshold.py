import os
import unittest
from unittest import mock

from sglang.srt import server_args as sa
from sglang.srt.server_args import ServerArgs

VAR = "GPU_PINNED_MIN_XFER_SIZE"


class TestRocmPinnedTransferThreshold(unittest.TestCase):
    def _build(self, hip):
        p = mock.patch.object(sa, "is_hip", lambda: hip)
        p.start()
        self.addCleanup(p.stop)
        return ServerArgs(model_path="dummy")

    def setUp(self):
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop(VAR, None)

    def test_set_on_rocm(self):
        self._build(hip=True)
        self.assertEqual(os.environ[VAR], str(4 * 1024 * 1024))

    def test_untouched_on_cuda(self):
        self._build(hip=False)
        self.assertNotIn(VAR, os.environ)

    def test_explicit_value_wins(self):
        os.environ[VAR] = "1234"
        self._build(hip=True)
        self.assertEqual(os.environ[VAR], "1234")


if __name__ == "__main__":
    unittest.main()
