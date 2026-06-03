"""Python wrapper for the hash_router CUDA kernel.

Lazy-loads the hash_router extension (same pattern as dsv4/ops/topk.py).
"""

import os
import torch

_kernel_module = None


def _get_kernel_module():
    """Lazy-load the hash_router CUDA extension."""
    global _kernel_module
    if _kernel_module is not None:
        return _kernel_module

    from torch.utils.cpp_extension import load
    kernel_dir = os.path.join(os.path.dirname(__file__))
    _kernel_module = load(
        name="hash_router",
        sources=[os.path.join(kernel_dir, "hash_router.cu")],
        extra_cuda_cflags=["-O3", "--generate-code=arch=compute_100a,code=[sm_100a]"],
        verbose=False,
    )
    return _kernel_module


def run_hash_router(
    token_ids: torch.Tensor,       # [N] int32
    hash_lut: torch.Tensor,        # [vocab_size, k] int32
    top_k: int,
    out_weights: torch.Tensor,     # [N, k] float32, pre-allocated
    out_ids: torch.Tensor,         # [N, k] int32, pre-allocated
):
    """Run the hash router kernel: gather expert IDs from LUT, write 1/k weights."""
    mod = _get_kernel_module()
    return mod.hash_router(token_ids, hash_lut, top_k, out_weights, out_ids)
