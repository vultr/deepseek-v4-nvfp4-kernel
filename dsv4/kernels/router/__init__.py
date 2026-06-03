"""DSV4 Router kernels — dispatch and CUDA kernel wrappers.

Exports:
  dense_router_dispatch: BF16 GEMM + fused activation + top-k (fallback)
  dense_router_dispatch_nvfp4: NVFP4 GEMM + fused activation + top-k (2-kernel)
  dense_router_dispatch_nvfp4_fused: NVFP4 fused single-kernel GEMM + router epilogue
  hash_router_dispatch: Hash routing via precomputed LUT gather
"""

from dsv4.kernels.router.dense_router_decode import (
    dense_router_dispatch,
    dense_router_dispatch_nvfp4,
    dense_router_dispatch_nvfp4_fused,
)


def hash_router_dispatch(
    token_ids,       # [N] int32
    hash_lut,        # [vocab_size, k] int32
    top_k,           # k=6
    out_weights,     # [N, k] float32, pre-allocated
    out_ids,         # [N, k] int32, pre-allocated
):
    """Hash router dispatch: gather expert IDs from precomputed LUT.

    Wraps the hash_router CUDA kernel (dsv4/kernels/cuda/hash_router.cu).
    One kernel launch, no intermediate buffers, no CPU-GPU sync.
    """
    from dsv4.kernels.cuda._hash_router import run_hash_router
    return run_hash_router(token_ids, hash_lut, top_k, out_weights, out_ids)
