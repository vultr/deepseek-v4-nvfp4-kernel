"""Storage for one layer's classical paged KV cache.

Layout per block:
  entries:     [num_blocks, entries_per_block, head_dim - rope_dim] FP8 (uint8 view)
  entries_r:   [num_blocks, entries_per_block, rope_dim]            BF16
  inv_scale:   [num_blocks, entries_per_block]                     FP32

The FP8/BF16 split mirrors paper §2.3.4 ("BF16 for RoPE dims, FP8 for
the rest"). The kernel reads both halves and concatenates in registers.

For CSA layers, a parallel pool stores indexer keys at the same block
granularity — same block ID maps to a block in both pools.
"""
from __future__ import annotations
from typing import Optional
import torch

from dsv4.cache.schema import LayerCacheSchema


class PagedKVPool:
    """Per-layer classical paged KV storage.

    Indexed by [physical_block_id, slot_in_block, ...].
    Both compressed entries and indexer keys (if applicable) are
    indexed by the SAME physical_block_id so a CSA layer's two pools
    share the block table.
    """

    def __init__(
        self,
        schema: LayerCacheSchema,
        num_blocks: int,
        device: str = "cuda",
    ):
        self.schema = schema
        self.num_blocks = num_blocks
        self.device = device

        nb = num_blocks
        epb = schema.entries_per_block
        hd = schema.entry_head_dim
        rd = schema.rope_dim
        fp8_dim = hd - rd

        # ---- Compressed entries ----
        # FP8 stored as uint8 (we view as float8_e4m3fn at read time).
        self.entries_fp8 = torch.zeros(
            (nb, epb, fp8_dim), dtype=torch.uint8, device=device,
        )
        # BF16 RoPE'd half — no quantization.
        self.entries_rope = torch.zeros(
            (nb, epb, rd), dtype=torch.bfloat16, device=device,
        )
        # Per-entry inverse scale (for FP8 dequant in attention kernel).
        self.inv_scale = torch.ones(
            (nb, epb), dtype=torch.float32, device=device,
        )

        # ---- Indexer keys (CSA only) ----
        if schema.indexer_entries_per_block > 0:
            i_epb = schema.indexer_entries_per_block
            i_hd = schema.indexer_head_dim
            # Indexer QK is FP4 per paper §2.3.4 — but we store the keys
            # post-quant. uint8 = 2 FP4 packed per byte.
            self.indexer_keys_fp4 = torch.zeros(
                (nb, i_epb, i_hd // 2), dtype=torch.uint8, device=device,
            )
            # Per-block-vector scale for the FP4 (one E4M3 scalar per
            # 16-element group, per the NVFP4 quantization scheme).
            self.indexer_scale = torch.ones(
                (nb, i_epb, i_hd // 16),
                dtype=torch.float8_e4m3fn, device=device,
            )
            self.indexer_global_scale = torch.ones(
                (nb,), dtype=torch.float32, device=device,
            )
        else:
            self.indexer_keys_fp4 = None
            self.indexer_scale = None
            self.indexer_global_scale = None

    def memory_bytes(self) -> int:
        """Total GPU memory used by this pool."""
        total = 0
        for name in ("entries_fp8", "entries_rope", "inv_scale",
                      "indexer_keys_fp4", "indexer_scale", "indexer_global_scale"):
            t = getattr(self, name)
            if t is not None:
                total += t.numel() * t.element_size()
        return total
