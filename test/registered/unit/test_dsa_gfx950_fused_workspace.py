"""Bookkeeping for the gfx950 fused DSA indexer workspace.

The buffers are pure scratch, so one workspace per (device, fp8 dtype) serves
every layer. These tests pin that down without a GPU by standing a counting
double in for _Workspace: only ensure_workspace's bookkeeping is under test.
"""

import unittest
from unittest import mock

from sglang.kernels.ops.attention.dsa.hip_gfx950 import fused_decode as fd

# GLM-5.2: 78 target layers + 1 draft layer, each with its own Indexer.
NUM_LAYERS = 79


class CountingWorkspace:
    made = 0

    def __init__(self, *, device, max_cols, fp8_dtype):
        CountingWorkspace.made += 1
        self.max_cols = max_cols
        self.page_table = None


class TestFusedIndexerWorkspace(unittest.TestCase):
    def setUp(self):
        CountingWorkspace.made = 0
        fd._WORKSPACES.clear()
        fd._FRESH_ALLOCATION = False
        self._patches = [
            mock.patch.object(fd, "_Workspace", CountingWorkspace),
            mock.patch.object(
                fd.torch.cuda, "is_current_stream_capturing", self._capturing
            ),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)
        self.capturing = False

    def _capturing(self):
        return self.capturing

    def _indexer(self):
        state = fd.Gfx950FusedIndexer.__new__(fd.Gfx950FusedIndexer)
        state.workspace = None
        state._constants = lambda: None
        return state

    def test_one_allocation_serves_every_layer(self):
        layers = [self._indexer() for _ in range(NUM_LAYERS)]
        for state in layers:
            self.assertTrue(
                state.ensure_workspace(device=0, max_cols=4096, fp8_dtype="f8")
            )
        self.assertEqual(CountingWorkspace.made, 1)
        self.assertIs(layers[0].workspace, layers[-1].workspace)

    def test_fresh_allocation_signals_once(self):
        first, second = self._indexer(), self._indexer()
        first.ensure_workspace(device=0, max_cols=4096, fp8_dtype="f8")
        self.assertTrue(fd.consume_fresh_allocation())
        self.assertFalse(fd.consume_fresh_allocation())
        second.ensure_workspace(device=0, max_cols=4096, fp8_dtype="f8")
        self.assertFalse(fd.consume_fresh_allocation())

    def test_devices_do_not_share(self):
        state = self._indexer()
        state.ensure_workspace(device=0, max_cols=4096, fp8_dtype="f8")
        state.ensure_workspace(device=1, max_cols=4096, fp8_dtype="f8")
        self.assertEqual(CountingWorkspace.made, 2)

    def test_wider_request_falls_back_rather_than_reallocating(self):
        state = self._indexer()
        self.assertTrue(
            state.ensure_workspace(device=0, max_cols=4096, fp8_dtype="f8")
        )
        self.assertFalse(
            state.ensure_workspace(device=0, max_cols=8192, fp8_dtype="f8")
        )
        self.assertEqual(CountingWorkspace.made, 1)

    def test_capture_never_allocates(self):
        self.capturing = True
        state = self._indexer()
        self.assertFalse(
            state.ensure_workspace(device=0, max_cols=4096, fp8_dtype="f8")
        )
        self.assertEqual(CountingWorkspace.made, 0)


if __name__ == "__main__":
    unittest.main()


class TestWorkspaceWidthRefusalIsLoud(unittest.TestCase):
    """A workspace narrower than the request must refuse *and* say so.

    Regression test for a silent disable: capping the preallocated width below
    the model context made ensure_workspace return False on every decode call,
    so the fused path never ran for an entire 1800 s benchmark while the startup
    log still reported "gfx950 fused DSA indexer enabled". The A/B that run
    produced compared two identical arms.
    """

    def setUp(self):
        fd._WORKSPACES.clear()
        fd._FRESH_ALLOCATION = False
        fd._WARNED_TOO_NARROW = False
        self._patches = [
            mock.patch.object(fd, "_Workspace", CountingWorkspace),
            mock.patch.object(
                fd.torch.cuda, "is_current_stream_capturing", lambda: False
            ),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _state(self):
        state = fd.Gfx950FusedIndexer.__new__(fd.Gfx950FusedIndexer)
        state.workspace = None
        state._constants = lambda: None
        return state

    def test_narrow_workspace_refuses_and_warns(self):
        fd.prealloc_workspace(device=0, max_cols=131072, fp8_dtype="fp8")
        state = self._state()
        with self.assertLogs(fd.logger, level="WARNING") as cm:
            ok = state.ensure_workspace(device=0, max_cols=1048576, fp8_dtype="fp8")
        self.assertFalse(ok)
        self.assertIn("131072", "".join(cm.output))
        self.assertIn("1048576", "".join(cm.output))

    def test_wide_enough_workspace_is_silent(self):
        fd.prealloc_workspace(device=0, max_cols=1048576, fp8_dtype="fp8")
        state = self._state()
        self.assertTrue(
            state.ensure_workspace(device=0, max_cols=1048576, fp8_dtype="fp8")
        )
        self.assertFalse(fd._WARNED_TOO_NARROW)
