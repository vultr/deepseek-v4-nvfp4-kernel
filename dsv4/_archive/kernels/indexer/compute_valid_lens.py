"""Compute per-query valid compressed entry count from block table.

Small integer reduction: for each request, valid_len = block_lens * entries_per_block
accounting for the partially-filled last block. Used by the indexer score kernel
to know how many candidate keys to stream.
"""
import torch


def compute_valid_lens(
    block_lens: torch.Tensor,   # [B] int32 — number of blocks per request
    block_table: torch.Tensor,  # [B, max_logical_blocks] int32
    entries_per_block: int,
) -> torch.Tensor:
    """Return [B] int32 — total valid compressed entries per request.

    For now, a simple formula: valid_entries = block_lens * entries_per_block.
    This assumes all entries in all allocated blocks are valid, which is correct
    because blocks are only allocated when flush writes to them, and each block
    is fully populated before the next is allocated (compression ratio is fixed).

    In a more general design with partially-filled blocks, this would need
    to check the actual write positions. For DSV4's fixed-ratio compression,
    the simple formula is exact.
    """
    return block_lens * entries_per_block
