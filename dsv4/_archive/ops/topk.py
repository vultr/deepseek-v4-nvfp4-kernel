"""
Sparse topk metadata kernels for DeepSeek-V4 Blackwell decode attention.

Own kernels — no FlashMLA, no Triton from vLLM.

C128A: position-based compressed KV slot lookup via block table.
C4A: local topk index to global slot ID mapping via block table.
"""

import os
import torch
from typing import Optional

_kernel_module = None

def _get_kernel_module():
    """Lazy-load the CUDA extension."""
    global _kernel_module
    if _kernel_module is not None:
        return _kernel_module
    
    from torch.utils.cpp_extension import load
    kernel_dir = os.path.join(os.path.dirname(__file__), "kernels")
    _kernel_module = load(
        name="sparse_topk_metadata",
        sources=[os.path.join(kernel_dir, "sparse_topk_metadata.cu")],
        extra_cuda_cflags=["-O3", "--generate-code=arch=compute_100a,code=[sm_100a]"],
        verbose=False,
    )
    return _kernel_module


def build_c128a_topk_metadata(
    positions: torch.Tensor,
    compress_ratio: int,
    num_decode_tokens: int,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    global_decode_buffer: torch.Tensor,
    decode_lens_buffer: torch.Tensor,
    prefill_buffer: torch.Tensor,
    max_compressed_tokens: int = 8192,
) -> tuple:
    """Build C128A topk metadata for decode and prefill tokens.
    
    For decode tokens: maps compressed KV positions to global slot IDs
    via block table lookup. Returns (global_decode, decode_lens, prefill_local).
    """
    mod = _get_kernel_module()
    return mod.build_c128a_topk_metadata(
        positions,
        compress_ratio,
        num_decode_tokens,
        token_to_req,
        block_table,
        block_size,
        slot_mapping,
        global_decode_buffer,
        decode_lens_buffer,
        prefill_buffer,
        max_compressed_tokens,
    )


def compute_c4a_global_topk(
    local_topk: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple:
    """Map local C4A topk indices to global KV cache slots.
    
    For each token, takes local compressed indices (from the indexer)
    and maps them to global slot IDs via block table lookup.
    Returns (global_topk_indices, topk_lens).
    """
    mod = _get_kernel_module()
    if is_valid_token.dtype == torch.bool:
        is_valid_token = is_valid_token.to(torch.int32)
    return mod.compute_c4a_global_topk(
        local_topk,
        token_to_req,
        block_table,
        block_size,
        is_valid_token,
    )
