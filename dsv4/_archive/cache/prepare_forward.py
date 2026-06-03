"""Pre-forward block allocation.

Runs between captured graphs. Computes how many new compressed entries
will be produced by this forward (deterministic from positions), allocates
the required physical blocks, and updates block tables.

After this runs, the captured graph can perform flushes by writing to
already-resolved (request, layer, logical_block) -> physical_block
mappings. No allocation inside the graph.
"""
from __future__ import annotations
from typing import List
import torch

from dsv4.model.layer_schedule import LayerSpec, AttentionType
from dsv4.cache.manager import KVCacheManager


def prepare_forward(
    manager: KVCacheManager,
    request_slots: torch.Tensor,      # [B] state cache slots
    positions_before: torch.Tensor,    # [B] absolute position BEFORE this forward
    positions_after: torch.Tensor,     # [B] absolute position AFTER this forward
) -> None:
    """Pre-allocate any blocks that will be needed by flushes in this forward.

    Pure CPU/GPU bookkeeping — runs between captures, not in hot path.
    For each compressed layer, works out how many flushes happen per
    request and allocates blocks to cover them.
    """
    for layer_idx, spec in enumerate(manager.schedule):
        if spec.attn == AttentionType.SWA:
            continue  # No classical pool, no flushes.

        schema = manager.schemas[layer_idx]
        alloc = manager.allocators[layer_idx]
        if alloc is None:
            continue

        m = (manager.config.csa_compression_ratio
             if spec.attn == AttentionType.CSA
             else manager.config.hca_compression_ratio)
        epb = schema.entries_per_block

        # How many compressed entries are NEWLY produced per request?
        # = floor(positions_after / m) - floor(positions_before / m)
        entries_after = (positions_after // m).to(torch.int64)
        entries_before = (positions_before // m).to(torch.int64)
        new_entries = entries_after - entries_before  # [B] int64

        # For each request, figure out how many new blocks are needed.
        # A block holds `epb` entries. If there are already some entries
        # in the current (open) block, they take some slots.
        for b in range(request_slots.numel()):
            n_new = int(new_entries[b])
            if n_new == 0:
                continue
            req_slot = int(request_slots[b])

            # How many entries are already in the current open block?
            existing_blocks = int(manager.block_lens[layer_idx][req_slot])
            entries_in_open_block = int(entries_before[b]) % epb if existing_blocks > 0 else 0
            slots_remaining_in_open = epb - entries_in_open_block if entries_in_open_block > 0 else 0

            # How many new blocks do we need?
            if entries_in_open_block == 0 and existing_blocks == 0:
                # Fresh — no open block yet
                blocks_needed = (n_new + epb - 1) // epb
            elif slots_remaining_in_open >= n_new:
                # Fits in the current open block
                blocks_needed = 0
            else:
                # Need additional blocks beyond the current open one
                overflow = n_new - slots_remaining_in_open
                blocks_needed = (overflow + epb - 1) // epb

            if blocks_needed == 0:
                continue

            ids = alloc.acquire(blocks_needed)
            existing = int(manager.block_lens[layer_idx][req_slot])
            manager.block_tables[layer_idx][req_slot, existing:existing + blocks_needed] = ids
            manager.block_lens[layer_idx][req_slot] = existing + blocks_needed
