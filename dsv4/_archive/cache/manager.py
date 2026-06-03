"""KVCacheManager — owns all KV cache state for one model instance.

Responsibilities:
  - Build per-layer pools and allocators at startup.
  - Hand out state-cache slots when requests are admitted.
  - Hand out classical blocks when layers need to flush compression.
  - Compose LayerCacheHandle for each layer per forward call.
  - Reclaim slots and blocks on request completion.

Not on the manager:
  - On-disk prefix storage. (Paper §3.5.2 — deferred entirely.)
  - Eviction policies. (Single-instance; requests run to completion.)
  - Cross-instance coordination.
"""
from __future__ import annotations
from typing import List, Optional, Dict
import torch

from dsv4.model.config import DSV4Config
from dsv4.model.layer_schedule import LayerSpec, AttentionType
from dsv4.cache.schema import LayerCacheSchema, build_schema, compute_block_budget
from dsv4.cache.allocator import BlockAllocator
from dsv4.cache.paged_cache import PagedKVPool
from dsv4.cache.state_cache import StateCachePool
from dsv4.cache.handle import LayerCacheHandle


class KVCacheManager:
    def __init__(
        self,
        config: DSV4Config,
        schedule: List[LayerSpec],
        max_concurrent_requests: int,
        max_context_tokens: int = 1_000_000,
        # Per-layer-type block budget. If None, computed from
        # max_context_tokens and max_concurrent_requests.
        num_blocks_per_csa_layer: Optional[int] = None,
        num_blocks_per_hca_layer: Optional[int] = None,
        device: str = "cuda",
    ):
        self.config = config
        self.schedule = schedule
        self.max_concurrent_requests = max_concurrent_requests
        self.device = device

        # ---- Per-layer schemas ----
        self.schemas: Dict[int, LayerCacheSchema] = {
            spec.layer_idx: build_schema(config, spec) for spec in schedule
        }

        # ---- Compute block budgets if not provided ----
        if num_blocks_per_csa_layer is None or num_blocks_per_hca_layer is None:
            budget = compute_block_budget(config, schedule, max_context_tokens,
                                          max_concurrent_requests)
            num_blocks_per_csa_layer = num_blocks_per_csa_layer or budget.get("csa", 0)
            num_blocks_per_hca_layer = num_blocks_per_hca_layer or budget.get("hca", 0)

        # ---- Per-layer pools ----
        # State cache exists for every layer.
        self.state_pools: Dict[int, StateCachePool] = {
            i: StateCachePool(schema, max_concurrent_requests, device)
            for i, schema in self.schemas.items()
        }

        # Classical paged pool only for compressed layers.
        self.paged_pools: Dict[int, Optional[PagedKVPool]] = {}
        self.allocators: Dict[int, Optional[BlockAllocator]] = {}
        for i, schema in self.schemas.items():
            if schema.entries_per_block == 0:
                self.paged_pools[i] = None
                self.allocators[i] = None
            else:
                nb = (num_blocks_per_csa_layer
                      if schema.attn_type == AttentionType.CSA
                      else num_blocks_per_hca_layer)
                self.paged_pools[i] = PagedKVPool(schema, nb, device)
                self.allocators[i] = BlockAllocator(nb, device)

        # ---- Request state ----
        # Slot index per request, into state cache pools (same index in
        # every layer). -1 = slot free.
        self.request_slot_map: torch.Tensor = torch.full(
            (max_concurrent_requests,), -1, dtype=torch.int32, device=device,
        )

        # Block table per request per layer:
        # block_tables[layer_idx][request_slot, logical_block_idx]
        # -> physical_block_idx
        max_blocks = max_context_tokens // 128  # BLOCK_SIZE_ORIGINAL_TOKENS
        self.max_blocks_per_request = max_blocks
        self.block_tables: Dict[int, torch.Tensor] = {}
        self.block_lens: Dict[int, torch.Tensor] = {}
        for i, schema in self.schemas.items():
            if schema.entries_per_block > 0:
                self.block_tables[i] = torch.full(
                    (max_concurrent_requests, max_blocks), -1,
                    dtype=torch.int32, device=device,
                )
                self.block_lens[i] = torch.zeros(
                    (max_concurrent_requests,), dtype=torch.int32, device=device,
                )

    # ------------------------------------------------------------------
    # Request lifecycle (called between captured graphs)
    # ------------------------------------------------------------------
    def admit_request(self) -> int:
        """Allocate a state cache slot. Returns the slot index."""
        free = (self.request_slot_map == -1).nonzero(as_tuple=False)
        if free.numel() == 0:
            raise RuntimeError("max concurrent requests exceeded")
        slot = int(free[0])
        self.request_slot_map[slot] = slot
        return slot

    def release_request(self, slot: int) -> None:
        """Return state cache slot and all associated blocks to free lists."""
        for layer_idx, alloc in self.allocators.items():
            if alloc is None:
                continue
            table = self.block_tables[layer_idx]
            lens = self.block_lens[layer_idx]
            valid = int(lens[slot])
            if valid > 0:
                alloc.release(table[slot, :valid].clone())
                lens[slot] = 0
                table[slot].fill_(-1)
        # Reset state cache slot.
        for state in self.state_pools.values():
            state.reset_slot(slot)
        self.request_slot_map[slot] = -1

    # ------------------------------------------------------------------
    # Block allocation for compression flush (called between captures)
    # ------------------------------------------------------------------
    def allocate_block(self, layer_idx: int, request_slot: int) -> int:
        """Allocate one new classical block for a request. Returns block ID."""
        alloc = self.allocators[layer_idx]
        assert alloc is not None, f"layer {layer_idx} has no classical pool"
        block_id = alloc.acquire(1)
        bid = int(block_id[0])
        # Append to the request's block table.
        table = self.block_tables[layer_idx]
        lens = self.block_lens[layer_idx]
        pos = int(lens[request_slot])
        assert pos < self.max_blocks_per_request, "block table overflow"
        table[request_slot, pos] = bid
        lens[request_slot] = pos + 1
        return bid

    # ------------------------------------------------------------------
    # Per-forward handle construction (called INSIDE captured graph)
    # ------------------------------------------------------------------
    def acquire(
        self,
        layer_idx: int,
        request_slots: torch.Tensor,  # [batch] int32
        positions: torch.Tensor,       # [tokens] int32
        request_ids: torch.Tensor,     # [tokens] int32
    ) -> LayerCacheHandle:
        """Build the LayerCacheHandle for one layer's forward.

        No allocation happens here — critical for cudagraph safety.
        """
        paged = self.paged_pools[layer_idx]
        state = self.state_pools[layer_idx]

        if paged is not None:
            # Pass the full tensors — no indexing, no allocation.
            # The attention kernel indexes by request_slots internally.
            block_table = self.block_tables[layer_idx]
            block_lens = self.block_lens[layer_idx]
        else:
            block_table = None
            block_lens = None

        handle = LayerCacheHandle(
            paged=paged,
            state=state,
            schema=self.schemas[layer_idx],
            request_slots=request_slots,
            positions=positions,
            request_ids=request_ids,
            block_table=block_table,
            block_lens=block_lens,
        )
        handle.num_query_heads = self.config.num_query_heads
        return handle

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def memory_bytes(self) -> int:
        """Total GPU memory used by all pools."""
        total = 0
        for pool in self.state_pools.values():
            total += pool.memory_bytes()
        for pool in self.paged_pools.values():
            if pool is not None:
                total += pool.memory_bytes()
        for i, table in self.block_tables.items():
            total += table.numel() * table.element_size()
            total += self.block_lens[i].numel() * self.block_lens[i].element_size()
        return total
