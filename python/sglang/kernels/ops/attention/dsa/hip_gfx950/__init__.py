"""gfx950 (MI355X) fused DSA indexer decode path.

Four raw-HIP kernels replacing the 12-launch ROCm indexer decode chain. Opt-in
behind ``SGLANG_DSA_HIP_FUSED_INDEXER_GFX950``; see
``sglang.srt.layers.attention.dsa.utils.gfx950_fused_indexer_runtime_ok``.

Each kernel source under ``csrc/`` ships exactly one accepted instantiation,
distilled from its KernelForge campaign source; see ``PROVENANCE``.
"""

from sglang.kernels.ops.attention.dsa.hip_gfx950.fused_decode import (  # noqa: F401
    MAX_ROWS,
    PAGE_SIZE,
    Gfx950FusedIndexer,
    model_shape_supported,
)
