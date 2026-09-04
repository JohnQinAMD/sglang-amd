"""JIT build + load of the four gfx950 DSA-indexer kernels.

Each kernel source under ``csrc/`` ships exactly one accepted instantiation
(see the package docstring); this module only builds them and hands back
the extension objects. Four extensions rather than one, because three of the
sources carry their own ``PYBIND11_MODULE`` and would collide in a single
extension.

Everything is built at *import* time of the first gated call — never inside a
timed region and never inside a HIP graph capture.

Build flags are the ones the sources were measured with:
  * ``--offload-arch=gfx950`` only (the default torch arch list also emits
    gfx942, doubling compile time for a target this path refuses to run on).
  * the logits kernel additionally needs ``-mllvm -amdgpu-mfma-vgpr-form`` and
    ``-fno-honor-nans``; that combination is ``liblogits_fast.so``, which is the
    library every accepted section-B measurement was taken with.  Dropping
    either flag changes the measured kernel, so they are not optional.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "csrc")

# Section B's accepted configuration: logits_hist_m<8, 12, ...>, 48 blocks/row.
# The kernel rejects any other (warps, hist_bits) pair rather than mis-launching.
LOGITS_HIST_BITS = 12
LOGITS_BLOCKS_PER_ROW = 48
LOGITS_WARPS = 8
# Section A's LSU4 Q reduction with two-MFMA shifted-load lookahead, plus the
# accepted KW decomposition and K3 epilogue.
DUAL_GEMV_CFG = 61
# Section C blocks per row.  Its configuration is baked into topk_transform.cu.
TOPK_G = 64


def _build_dir() -> str:
    d = os.environ.get("SGLANG_DSA_GFX950_BUILD_DIR")
    if not d:
        d = os.path.join(os.path.expanduser("~"), ".cache", "sglang", "dsa_gfx950")
    os.makedirs(d, exist_ok=True)
    return d


def _aiter_include_paths() -> List[str]:
    """``qk_rope_hadamard_quant.cu`` includes aiter's ``hip_reduce.h`` / ``opus/opus.hpp``.

    It includes them rather than vendoring them on purpose: the numeric helpers
    must be the same code aiter's own fused indexer op uses.
    """
    import aiter

    root = os.path.abspath(os.path.join(os.path.dirname(aiter.__file__), ".."))
    inc = os.path.join(root, "csrc", "include")
    if not os.path.isfile(os.path.join(inc, "hip_reduce.h")):
        raise RuntimeError(
            f"aiter C++ headers not found under {inc}; the gfx950 fused DSA "
            "indexer needs aiter's csrc/include on the include path"
        )
    return [inc]


_NAMES = (
    "sglang_dsa_gfx950_gemv",
    "sglang_dsa_gfx950_qk",
    "sglang_dsa_gfx950_logits",
    "sglang_dsa_gfx950_topk",
)


def _stage(build_dir: str, sources: List[str]) -> List[str]:
    """Compile from a copy in the build directory, not from the package.

    torch's ROCm build hipifies each source IN PLACE: it writes the rewritten
    file next to the original and compiles that one. Pointing it at the
    installed package would litter it with generated files, and would fail
    outright wherever the install is read-only. Copying only on a content
    change keeps ninja from rebuilding on every import.
    """
    out = []
    for s in sources:
        src = os.path.join(_CSRC, s)
        dst = os.path.join(build_dir, s)
        with open(src, "rb") as f:
            want = f.read()
        try:
            with open(dst, "rb") as f:
                stale = f.read() != want
        except FileNotFoundError:
            stale = True
        if stale:
            with open(dst, "wb") as f:
                f.write(want)
        out.append(dst)
    return out


def _load(
    name: str,
    sources: List[str],
    extra_flags: Optional[List[str]] = None,
    std: str = "c++17",
    include_aiter: bool = False,
):
    from torch.utils.cpp_extension import load

    build_dir = os.path.join(_build_dir(), name)
    os.makedirs(build_dir, exist_ok=True)
    staged = _stage(build_dir, sources)
    prev = os.environ.get("PYTORCH_ROCM_ARCH")
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    try:
        return load(
            name=name,
            sources=staged,
            build_directory=build_dir,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950", f"-std={std}"]
            + (extra_flags or []),
            extra_include_paths=_aiter_include_paths() if include_aiter else [],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("PYTORCH_ROCM_ARCH", None)
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = prev


@lru_cache(maxsize=1)
def modules():
    """(gemv, qk, logits, topk) extension modules, or raise."""
    gemv = _load(_NAMES[0], ["dual_gemv_bf16.cu"])
    # qk_rope_hadamard_quant.cu needs C++20 (aiter's opus.hpp) and aiter's headers.
    qk = _load(
        _NAMES[1],
        ["qk_rope_hadamard_quant.cu"],
        extra_flags=["-Wno-unused-result"],
        std="c++20",
        include_aiter=True,
    )
    logits = _load(
        _NAMES[2],
        ["paged_mqa_logits.cu", "paged_mqa_logits_bindings.cu"],
        extra_flags=[
            "-fno-honor-nans",
            "-mllvm",
            "-amdgpu-mfma-vgpr-form",
            "-mllvm",
            "-amdgpu-early-inline-all=true",
            "-mllvm",
            "-amdgpu-function-calls=false",
            "-Wno-unused-result",
        ],
    )
    topk = _load(
        _NAMES[3],
        ["topk_transform.cu"],
        extra_flags=[f"-DDSA_TOPK_HIST_BITS={LOGITS_HIST_BITS}"],
    )
    return gemv, qk, logits, topk


@lru_cache(maxsize=1)
def modules_or_none():
    """Build once; on any failure log and return None so callers fall back."""
    try:
        return modules()
    except Exception as e:  # noqa: BLE001 - a build failure must never be fatal
        logger.warning(
            "gfx950 fused DSA indexer unavailable (build failed: %s); "
            "falling back to the standard ROCm indexer path",
            e,
        )
        return None
