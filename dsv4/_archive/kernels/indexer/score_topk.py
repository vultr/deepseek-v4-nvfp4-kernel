"""Python launcher for the indexer score+topk kernel.

Provides run_indexer_score_topk() which takes FP32 query tensors
and an IndexerView from the cache, runs the fused score + ReLU +
weighted sum + top-k kernel, and returns [T, top_k] compressed
entry indices.

Phase 1: FP32 dot products. Correct, testable.
Phase 2: FP4 tcgen05 MMA swap (optimization on known-correct base).
"""
from __future__ import annotations
import os
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dsv4.cache.handle import IndexerView

_kernel_module = None


def _get_kernel_module():
    global _kernel_module
    if _kernel_module is not None:
        return _kernel_module
    from dsv4.kernels.cuda.loader import get_cuda_module
    _kernel_module = get_cuda_module("indexer_score_topk", ["indexer_score_topk.cu"])
    return _kernel_module


def run_indexer_score_topk(
    q_I: torch.Tensor,      # [T, n_heads * head_dim] BF16 — indexer queries
    w_h: torch.Tensor,      # [T, n_heads] FP32 — per-head weights
    indexer_view: "IndexerView",
    num_heads: int,
    head_dim: int,
    top_k: int,
    entries_per_block: int,
) -> torch.Tensor:
    """Returns [T, top_k] int32 of selected compressed entry indices.

    The kernel computes (MQA — shared key across indexer heads):
      I[t,s] = Σ_h w_h[t,h] · ReLU(q_I[t,h] · K^IComp[s])
      topk_indices = argtopk(I[t,:], k=top_k)

    Note: K^IComp[s] is shared across heads (MQA), NOT per-head K^IComp[s,h].
    This matches the .cu kernel and the production Indexer.forward() einsum.
    The paper (eq. 16) uses the shared-key form.

    q_I is passed as BF16 and dequantized to FP32 before the kernel.
    The indexer keys are stored FP4 in the cache and dequantized
    inside the kernel.
    """
    mod = _get_kernel_module()
    T = q_I.shape[0]

    # Dequantize q_I from BF16 to FP32 and reshape to [T, n_heads, head_dim]
    q_I_f32 = q_I.float().reshape(T, num_heads, head_dim).contiguous()

    # Compute valid lens from block_lens
    valid_lens = indexer_view.block_lens * entries_per_block  # [B] int32
    # We need per-query valid lens. block_lens is [B] where B = batch.
    # For a single request, this is just the one value.
    # For batched, repeat across tokens belonging to the same request.
    # Simplification: assume T == B for now (one token per request in decode).
    if valid_lens.shape[0] != T:
        # Prefill: T > B. We need to map tokens to requests.
        # WARNING: broadcasting request 0's valid_lens is WRONG for batched
        # or multi-request prefill — it selects from wrong key ranges per token.
        # This is only correct for single-request bring-up.
        # TODO: proper per-token valid_lens from request_ids mapping.
        valid_lens = valid_lens[:1].expand(T).contiguous()

    out = torch.full((T, top_k), -1, dtype=torch.int32, device=q_I.device)

    mod.indexer_score_topk_fp32(
        q_I_f32, w_h,
        indexer_view.keys_fp4, indexer_view.scale, indexer_view.global_scale,
        indexer_view.block_table, valid_lens,
        out,
        num_heads, head_dim, top_k, entries_per_block,
    )
    return out
