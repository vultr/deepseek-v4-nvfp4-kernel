"""Per-layer KV cache shape.

Computed once per layer at engine startup from the LayerSpec. The
schema is what tells the allocator how big each pool slot is and what
sub-regions exist (compressed entries / indexer keys / SWA window /
uncompressed tail).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from dsv4.model.config import DSV4Config
from dsv4.model.layer_schedule import LayerSpec, AttentionType


# Block size is invariant for DSV4 — derived from compression ratios.
# lcm(m, m') = lcm(4, 128) = 128 original tokens per block.
# Holds 128/4 = 32 CSA entries OR 128/128 = 1 HCA entry per block.
BLOCK_SIZE_ORIGINAL_TOKENS = 128


@dataclass(frozen=True)
class LayerCacheSchema:
    """Cache layout for one transformer layer.

    Fields with `_per_block` are the dimensions of one block in the
    classical paged pool. `_per_state_slot` are dimensions of one
    request's slot in the state cache.

    All sizes are in number of entries — bytes come from the dtypes.
    """
    layer_idx: int
    attn_type: AttentionType

    # ---- Classical paged cache (compressed entries) ----
    entries_per_block: int
    entry_head_dim: int
    rope_dim: int

    # ---- Indexer pool (CSA only) ----
    indexer_entries_per_block: int
    indexer_head_dim: int

    # ---- State cache (SWA window + uncompressed tail) ----
    swa_window_size: int

    # CSA: paper eq.11-12, the i-th flush uses Ca[m*i:m*(i+1)] and
    # Cb[m*(i-1):m*i]. After flush, current a-stream becomes next b-stream.
    # So we need m entries for current a-stream AND m entries for previous
    # b-stream. Total tail = 2*m for CSA.
    tail_buffer_size_a: int  # m (CSA) or m' (HCA) — current tokens
    tail_buffer_size_b: int  # m (CSA only) — previous block's a-stream kept as b-input

    # Per-token inverse scale storage (for FP8 dequant).
    needs_inv_scale: bool = True

    @property
    def tail_buffer_size(self) -> int:
        """Total tail entries (for backward compat with schema consumers)."""
        return self.tail_buffer_size_a + self.tail_buffer_size_b


def build_schema(config: DSV4Config, spec: LayerSpec) -> LayerCacheSchema:
    """Derive cache schema for a single layer from architectural config."""
    if spec.attn == AttentionType.CSA:
        return LayerCacheSchema(
            layer_idx=spec.layer_idx,
            attn_type=AttentionType.CSA,
            entries_per_block=BLOCK_SIZE_ORIGINAL_TOKENS // config.csa_compression_ratio,
            entry_head_dim=config.head_dim,
            rope_dim=config.rope_dim,
            indexer_entries_per_block=BLOCK_SIZE_ORIGINAL_TOKENS // config.csa_compression_ratio,
            indexer_head_dim=config.indexer_head_dim,
            swa_window_size=config.sliding_window,
            tail_buffer_size_a=config.csa_compression_ratio,  # m=4 current
            tail_buffer_size_b=config.csa_compression_ratio,  # m=4 previous (b-stream)
        )
    elif spec.attn == AttentionType.HCA:
        return LayerCacheSchema(
            layer_idx=spec.layer_idx,
            attn_type=AttentionType.HCA,
            entries_per_block=BLOCK_SIZE_ORIGINAL_TOKENS // config.hca_compression_ratio,
            entry_head_dim=config.head_dim,
            rope_dim=config.rope_dim,
            indexer_entries_per_block=0,
            indexer_head_dim=0,
            swa_window_size=config.sliding_window,
            tail_buffer_size_a=config.hca_compression_ratio,  # m'=128 current
            tail_buffer_size_b=0,  # HCA has no b-stream
        )
    else:  # SWA-only
        return LayerCacheSchema(
            layer_idx=spec.layer_idx,
            attn_type=AttentionType.SWA,
            entries_per_block=0,
            entry_head_dim=config.head_dim,
            rope_dim=config.rope_dim,
            indexer_entries_per_block=0,
            indexer_head_dim=0,
            swa_window_size=config.sliding_window,
            tail_buffer_size_a=0,
            tail_buffer_size_b=0,
        )


def compute_block_budget(
    config: DSV4Config,
    schedule: list[LayerSpec],
    max_context_tokens: int,
    max_concurrent_requests: int,
) -> dict[str, int]:
    """Compute per-layer-type block counts for the allocator."""
    blocks_per_request = max_context_tokens // BLOCK_SIZE_ORIGINAL_TOKENS
    headroom = 1.10
    result = {}
    for spec in schedule:
        if spec.attn == AttentionType.CSA:
            key = "csa"
        elif spec.attn == AttentionType.HCA:
            key = "hca"
        else:
            continue
        total = int(max_concurrent_requests * blocks_per_request * headroom)
        result[key] = max(result.get(key, 0), total)
    return result
