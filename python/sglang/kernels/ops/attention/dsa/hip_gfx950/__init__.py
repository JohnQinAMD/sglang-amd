"""gfx950 (MI355X) fused DSA indexer decode path.

Four raw-HIP kernels replacing the 12-launch ROCm indexer decode chain. Opt-in
behind ``--enable-dsa-fused-indexer``; see
``sglang.srt.layers.attention.dsa.utils.gfx950_fused_indexer_runtime_ok``.

Each kernel source under ``csrc/`` ships exactly ONE instantiation: the
configuration is baked into the source rather than selected at runtime, and the
search space that chose it is not shipped. That is why the launches read as
fully-specified template arguments (``logits_hist_m<8, 12>``,
``k_scatter<2, true, true, true, false, false, false, 0, true, 16>``) with no
dispatcher around them -- there is nothing else to dispatch to. Changing a
configuration therefore means editing the kernel, not passing a different flag.
"""

from sglang.kernels.ops.attention.dsa.hip_gfx950.fused_decode import (  # noqa: F401
    MAX_ROWS,
    PAGE_SIZE,
    Gfx950FusedIndexer,
    consume_fresh_allocation,
    model_shape_supported,
    prealloc_workspace,
)
