"""Python wrapper for the topk_select CUDA kernel.

Lazy-loads the topk_select extension (same pattern as dsv4/ops/topk.py).
This is the general top-k primitive reused by the router and the CSA indexer.
"""

import os
import torch

_kernel_module = None


def _get_kernel_module():
    """Lazy-load the topk_select CUDA extension."""
    global _kernel_module
    if _kernel_module is not None:
        return _kernel_module

    from torch.utils.cpp_extension import load
    kernel_dir = os.path.join(os.path.dirname(__file__), "kernels", "cuda")
    _kernel_module = load(
        name="topk_select",
        sources=[os.path.join(kernel_dir, "topk_select.cu")],
        extra_cuda_cflags=["-O3", "--generate-code=arch=arch=compute_100a,code=[sm_100a]"],
        verbose=False,
    )
    return _kernel_module


def topk_select(
    scores: torch.Tensor,    # [num_rows, E] float32, row-major contiguous
    k: int,                  # number to select (currently only k=6 supported)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k indices and values from each row of scores.

    Returns (values, indices) where:
      values: [num_rows, k] float32 — top-k scores in descending order
      indices: [num_rows, k] int32 — top-k indices (0-based, lower index wins on ties)

    One block per row, 64 threads per block, per-thread register min-heap
    with shared-memory merge. O(E * log k) per row.
    """
    mod = _get_kernel_module()
    return mod.topk_select(scores, k)
