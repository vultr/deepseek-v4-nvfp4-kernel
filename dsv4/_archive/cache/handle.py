"""LayerCacheHandle — typed per-call view onto one layer's cache.

Constructed by KVCacheManager.acquire() once per layer per forward.
Holds tensor references and integer indices; no allocation. Methods
expose the operations AttentionSubBlock needs without exposing the
underlying storage layout.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from dsv4.cache.paged_cache import PagedKVPool
    from dsv4.cache.state_cache import StateCachePool
    from dsv4.cache.schema import LayerCacheSchema


@dataclass
class LayerCacheHandle:
    """Read/write interface for one layer's cache.

    The fields are the resolved indices and tensor refs for THIS call's
    batch of requests. AttentionSubBlock never sees raw pool tensors.
    """
    # Pool references (shared across handles — never mutated).
    paged: Optional["PagedKVPool"]
    state: "StateCachePool"
    schema: "LayerCacheSchema"

    # Per-call indices.
    request_slots: torch.Tensor   # [batch] int32 — state cache slot per request
    positions: torch.Tensor       # [tokens] int32 — absolute position per token
    request_ids: torch.Tensor     # [tokens] int32 — which request each token belongs to

    # Block table for the classical pool (None for SWA-only layers).
    # Shape: [batch, max_logical_blocks] int32. -1 padding for unused entries.
    block_table: Optional[torch.Tensor]
    # Number of valid blocks per request (excludes padding).
    block_lens: Optional[torch.Tensor]

    # ------------------------------------------------------------------
    # Properties called by AttentionSubBlock
    # ------------------------------------------------------------------

    @property
    def num_query_heads(self) -> int:
        """Number of query heads (from schema)."""
        # The schema doesn't store n_q directly — derive from the config.
        # For now, store on the handle at construction.
        return self._num_query_heads

    @num_query_heads.setter
    def num_query_heads(self, value: int):
        self._num_query_heads = value

    @property
    def head_dim(self) -> int:
        """Head dimension (from schema)."""
        return self.schema.entry_head_dim

    # ------------------------------------------------------------------
    # Methods called by AttentionSubBlock
    # ------------------------------------------------------------------

    def write_swa(
        self,
        raw_kv: torch.Tensor,  # (T, head_dim) BF16
    ) -> None:
        """Write raw KV into the SWA ring buffer AND tail compression buffer.

        Both regions get the same tokens — SWA consumes the last n_win,
        the tail accumulates until it can flush.
        """
        from dsv4.kernels.cache.append_swa import append_swa_kernel
        append_swa_kernel(
            raw_kv=raw_kv,
            request_slots=self.request_slots,
            positions=self.positions,
            swa_fp8=self.state.swa_fp8,
            swa_rope=self.state.swa_rope,
            swa_inv=self.state.swa_inv,
            swa_pos=self.state.swa_pos,
            swa_head=self.state.swa_head,
            rope_dim=self.schema.rope_dim,
        )

    def flush_compression(
        self,
        compressed: torch.Tensor,  # (T_flush, head_dim) BF16 — newly produced
        indexer_keys: Optional[torch.Tensor] = None,
    ) -> None:
        """Promote pending tail tokens into the classical pool.

        Called by the compressor when the tail buffer has enough tokens.
        Allocates a new block if the latest block is full.

        Block allocation requires going outside the captured graph — in
        a fully-captured decode this is rare (once per m or m' tokens),
        so we make it explicit. The manager has the contract.
        """
        raise NotImplementedError("see kernels/cache/flush_compression.py")

    def gather_compressed_kv(
        self,
        selected_indices: torch.Tensor,  # (T, top_k) int64 — from indexer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CSA: gather top-k compressed KV entries into dense BF16 tensors.

        Returns:
            (k_compressed, v_compressed) each of shape (1, n_comp, head_dim) BF16.
            The leading dim=1 is for the single KV head (MQA in DSV4).
        """
        assert self.paged is not None, "CSA gather requires paged pool"
        from dsv4.kernels.cache.gather import gather_compressed_kv

        hd = self.head_dim
        rd = self.schema.rope_dim
        epb = self.schema.entries_per_block

        # selected_indices is int64, gather kernel needs int32
        indices_i32 = selected_indices.to(torch.int32)

        # block_table for CSA: [batch, max_logical_blocks]
        # For per-request gather, use the first request's block_table
        # (decode: batch=1, so this is trivial)
        if self.block_table.dim() == 1:
            bt = self.block_table.unsqueeze(0)
        else:
            bt = self.block_table

        k_out = gather_compressed_kv(
            entries_fp8=self.paged.entries_fp8,
            entries_rope=self.paged.entries_rope,
            inv_scale=self.paged.inv_scale,
            topk_indices=indices_i32,
            block_table=bt,
            entries_per_block=epb,
            head_dim=hd,
            rope_dim=rd,
        )
        # k_out: (T, top_k, hd) — for FMHA we need (1, n_comp, hd)
        # At decode T=1: squeeze to (top_k, hd) then unsqueeze for KV head dim
        n_comp = k_out.shape[1]
        k_compressed = k_out.squeeze(0).unsqueeze(0)  # (1, n_comp, hd)
        # V shares the same storage but is transposed — DSV4 uses K=V for
        # the compressed KV (same entries, different projection weights applied
        # before compression). For now, return the same gathered tensor.
        # TODO: verify if K and V are stored separately or shared.
        v_compressed = k_compressed.clone()
        return k_compressed, v_compressed

    def gather_all_compressed_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """HCA: gather ALL compressed KV entries into dense BF16 tensors.

        No indexer — dense attention over the short compressed sequence.

        Returns:
            (k_compressed, v_compressed) each of shape (1, n_comp, head_dim) BF16.
        """
        assert self.paged is not None, "HCA gather requires paged pool"
        from dsv4.kernels.cache.gather import gather_all_compressed_kv

        hd = self.head_dim
        rd = self.schema.rope_dim
        epb = self.schema.entries_per_block

        if self.block_table.dim() == 1:
            bt = self.block_table.unsqueeze(0)
            bl = self.block_lens.unsqueeze(0) if self.block_lens is not None else None
        else:
            bt = self.block_table
            bl = self.block_lens

        if bl is None:
            # Default: all blocks valid
            bl = torch.full((bt.shape[0],), bt.shape[1], dtype=torch.int32, device=bt.device)

        k_out = gather_all_compressed_kv(
            entries_fp8=self.paged.entries_fp8,
            entries_rope=self.paged.entries_rope,
            inv_scale=self.paged.inv_scale,
            block_table=bt,
            block_lens=bl,
            entries_per_block=epb,
            head_dim=hd,
            rope_dim=rd,
        )
        # k_out: (batch, total_entries, hd) — for FMHA we need (1, n_comp, hd)
        n_comp = k_out.shape[1]
        k_compressed = k_out.squeeze(0).unsqueeze(0)  # (1, n_comp, hd)
        v_compressed = k_compressed.clone()
        return k_compressed, v_compressed

    def gather_swa_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather SWA window entries into dense BF16 tensors.

        Returns:
            (k_swa, v_swa) each of shape (1, swa_len, head_dim) BF16.
        """
        from dsv4.kernels.cache.gather import gather_swa_kv

        hd = self.head_dim
        rd = self.schema.rope_dim

        k_out = gather_swa_kv(
            swa_fp8=self.state.swa_fp8,
            swa_rope=self.state.swa_rope,
            swa_inv=self.state.swa_inv,
            swa_pos=self.state.swa_pos,
            request_slots=self.request_slots,
            head_dim=hd,
            rope_dim=rd,
        )
        # k_out: (batch, n_win, hd) — for FMHA we need (1, swa_len, hd)
        k_swa = k_out.squeeze(0).unsqueeze(0)  # (1, swa_len, hd)
        v_swa = k_swa.clone()
        return k_swa, v_swa

    def read_swa_view(self) -> "SWAView":
        """Return a typed view of the SWA window for this batch."""
        return SWAView(
            fp8=self.state.swa_fp8,
            rope=self.state.swa_rope,
            inv_scale=self.state.swa_inv,
            positions=self.state.swa_pos,
            head=self.state.swa_head,
            slots=self.request_slots,
        )

    def read_classical_view(self) -> "ClassicalView":
        """Return a typed view of compressed entries for this batch."""
        assert self.paged is not None, "SWA-only layers have no classical cache"
        return ClassicalView(
            entries_fp8=self.paged.entries_fp8,
            entries_rope=self.paged.entries_rope,
            inv_scale=self.paged.inv_scale,
            block_table=self.block_table,
            block_lens=self.block_lens,
        )

    def read_indexer_view(self) -> "IndexerView":
        """CSA-only. Returns FP4 indexer keys with their scales."""
        assert self.paged is not None and self.paged.indexer_keys_fp4 is not None
        return IndexerView(
            keys_fp4=self.paged.indexer_keys_fp4,
            scale=self.paged.indexer_scale,
            global_scale=self.paged.indexer_global_scale,
            block_table=self.block_table,
            block_lens=self.block_lens,
        )

    def __post_init__(self):
        # Initialize _num_query_heads (must be set by the manager at construction)
        if not hasattr(self, '_num_query_heads'):
            self._num_query_heads = 0


# Typed views — simple dataclasses, no logic. The FMHA / indexer / SWA
# kernels accept these to keep their signatures clean.
@dataclass
class SWAView:
    fp8: torch.Tensor
    rope: torch.Tensor
    inv_scale: torch.Tensor
    positions: torch.Tensor
    head: torch.Tensor
    slots: torch.Tensor


@dataclass
class ClassicalView:
    entries_fp8: torch.Tensor
    entries_rope: torch.Tensor
    inv_scale: torch.Tensor
    block_table: torch.Tensor
    block_lens: torch.Tensor


@dataclass
class IndexerView:
    keys_fp4: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    block_table: torch.Tensor
    block_lens: torch.Tensor
