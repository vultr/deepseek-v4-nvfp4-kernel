"""Python wrappers for cache gather kernels.

Provides the bridge between LayerCacheHandle and the raw CUDA gather ops.
Loaded via torch.utils.cpp_extension (JIT, sm_100a).

Three gather paths:
1. gather_compressed_kv: CSA — top-k selected compressed entries
2. gather_all_compressed_kv: HCA — all compressed entries (no indexer)
3. gather_swa_kv: SWA window entries from state cache

All outputs are dense BF16 tensors ready for FMHA consumption.
"""
import os
import torch
from torch.utils.cpp_extension import load

_kernel_module = None
_kernel_dir = os.path.join(os.path.dirname(__file__), "..", "cuda")


def _get_kernel_module():
    global _kernel_module
    if _kernel_module is not None:
        return _kernel_module
    _kernel_module = load(
        name="cache_gather",
        sources=[
            os.path.join(_kernel_dir, "gather_kv.cu"),
            os.path.join(_kernel_dir, "gather_swa.cu"),
        ],
        extra_cuda_cflags=["-O3", "--generate-code=arch=compute_100a,code=[sm_100a]"],
        verbose=False,
    )
    return _kernel_module


def gather_compressed_kv(
    entries_fp8: torch.Tensor,      # [num_blocks, epb, fp8_dim] uint8
    entries_rope: torch.Tensor,     # [num_blocks, epb, rope_dim] BF16
    inv_scale: torch.Tensor,        # [num_blocks, epb] FP32
    topk_indices: torch.Tensor,     # [T, top_k] int32
    block_table: torch.Tensor,      # [batch, max_logical_blocks] int32
    entries_per_block: int,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Gather top-k compressed KV entries into a dense BF16 tile.

    Returns: (T, top_k, head_dim) BF16
    """
    T = topk_indices.size(0)
    top_k = topk_indices.size(1)
    output = torch.zeros(T, top_k, head_dim, dtype=torch.bfloat16, device=entries_fp8.device)

    mod = _get_kernel_module()
    mod.gather_kv(
        entries_fp8, entries_rope, inv_scale,
        topk_indices, block_table, output,
        entries_per_block, rope_dim,
    )
    return output


def gather_all_compressed_kv(
    entries_fp8: torch.Tensor,      # [num_blocks, epb, fp8_dim] uint8
    entries_rope: torch.Tensor,     # [num_blocks, epb, rope_dim] BF16
    inv_scale: torch.Tensor,        # [num_blocks, epb] FP32
    block_table: torch.Tensor,      # [batch, max_logical_blocks] int32
    block_lens: torch.Tensor,       # [batch] int32
    entries_per_block: int,
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Gather ALL compressed KV entries for HCA (dense attention).

    Returns: (batch, total_entries, head_dim) BF16 where
    total_entries = block_lens.sum() * entries_per_block (padded to max)
    """
    batch = block_table.size(0)
    max_blocks = block_lens.max().item()
    total_entries = max_blocks * entries_per_block
    output = torch.zeros(batch, total_entries, head_dim, dtype=torch.bfloat16, device=entries_fp8.device)

    mod = _get_kernel_module()
    mod.gather_all_compressed(
        entries_fp8, entries_rope, inv_scale,
        block_table, block_lens, output,
        entries_per_block, rope_dim,
    )
    return output


def gather_swa_kv(
    swa_fp8: torch.Tensor,          # [max_req, n_win, fp8_dim] uint8
    swa_rope: torch.Tensor,         # [max_req, n_win, rope_dim] BF16
    swa_inv: torch.Tensor,          # [max_req, n_win] FP32
    swa_pos: torch.Tensor,          # [max_req, n_win] int32
    request_slots: torch.Tensor,    # [batch] int32
    head_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Gather SWA window entries into a dense BF16 tile.

    Returns: (batch, n_win, head_dim) BF16
    """
    batch = request_slots.size(0)
    n_win = swa_fp8.size(1)
    output = torch.zeros(batch, n_win, head_dim, dtype=torch.bfloat16, device=swa_fp8.device)

    mod = _get_kernel_module()
    mod.gather_swa(
        swa_fp8, swa_rope, swa_inv, swa_pos, request_slots,
        output, head_dim, rope_dim,
    )
    return output
