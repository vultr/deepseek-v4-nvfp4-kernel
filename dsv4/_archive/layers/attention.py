"""DSV4 attention sub-block — CSA / HCA / SWA variants.

The constructor decides which variant this instance is by reading the
LayerSpec passed in. After construction, forward() is fixed — no runtime
branching on attention type, so torch.compile can fold the dispatch
completely.

What this class is responsible for:
  - Holding the projection weights (Q-down/up, KV-down, output groups).
  - Calling Nvfp4Linear for the projections.
  - Running RoPE (partial, last rope_dim dimensions).
  - Calling the compressor (CSA or HCA) to write into the KV cache.
  - For CSA only: calling the indexer + top-k to select sparse entries.
  - Calling the FMHA kernel with the right K/V source.
  - Calling inverse RoPE on the output.
  - Calling Nvfp4GroupedLinear (wo_a) + Nvfp4Linear (wo_b).

What this class does NOT touch:
  - mHC pre/post — done by TransformerLayer.
  - Pre-norm — done by TransformerLayer.
  - KV cache allocation / eviction — done by the cache handle.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from dsv4.layers.linear import Nvfp4Linear
from dsv4.layers.grouped_linear import Nvfp4GroupedLinear
from dsv4.model.layer_schedule import AttentionType, LayerSpec

if TYPE_CHECKING:
    from dsv4.model.config import DSV4Config
    from dsv4.cache.paged_cache import LayerCacheHandle


class AttentionSubBlock:
    def __init__(self, config: "DSV4Config", spec: LayerSpec):
        self.config = config
        self.spec = spec
        self.layer_idx = spec.layer_idx
        self.attn_type = spec.attn  # frozen for this layer's lifetime

        # ---- Projections (all NVFP4) ----

        # Query path is low-rank: hidden -> dc -> head_dim * num_heads.
        self.q_down = Nvfp4Linear(
            in_features=config.hidden_size,
            out_features=config.query_compression_dim,
        )
        self.q_up = Nvfp4Linear(
            in_features=config.query_compression_dim,
            out_features=config.head_dim * config.num_query_heads,
        )

        # KV projection — single head shared across queries (MQA).
        # CSA needs two streams (a, b) for the overlapping compression;
        # HCA needs one. SWA needs raw KV with no compression.
        if self.attn_type == AttentionType.CSA:
            # 2 * head_dim for (Ca, Cb), 2 * head_dim for compression weights (Za, Zb)
            self.kv_down = Nvfp4Linear(
                in_features=config.hidden_size,
                out_features=4 * config.head_dim,
            )
        elif self.attn_type == AttentionType.HCA:
            # head_dim for C, head_dim for Z
            self.kv_down = Nvfp4Linear(
                in_features=config.hidden_size,
                out_features=2 * config.head_dim,
            )
        else:  # SWA
            self.kv_down = Nvfp4Linear(
                in_features=config.hidden_size,
                out_features=config.head_dim,
            )

        # ---- Indexer (CSA only) ----
        if self.attn_type == AttentionType.CSA:
            # Indexer queries share the q_down latent c_Q with main queries.
            # Up-projection: c_Q -> indexer_num_heads * indexer_head_dim.
            self.indexer_q_up = Nvfp4Linear(
                in_features=config.query_compression_dim,
                out_features=config.indexer_num_heads * config.indexer_head_dim,
            )
            # Per-head weight vector w_h: hidden -> indexer_num_heads.
            self.indexer_head_weights = Nvfp4Linear(
                in_features=config.hidden_size,
                out_features=config.indexer_num_heads,
                max_num_tokens=1,  # scalar per head, not per-token projection
            )

        # ---- Output projection ----
        # wo_a: grouped, splits the n_heads outputs into n_groups and projects
        # each group from (heads_per_group * head_dim) to o_lora_rank.
        heads_per_group = config.num_query_heads // config.num_output_groups
        self.wo_a = Nvfp4GroupedLinear(
            n_local_groups=config.num_output_groups,
            heads_per_group=heads_per_group,
            head_dim=config.head_dim,
            o_lora_rank=config.output_group_dim,
        )
        # wo_b: dense, concatenated group outputs back to hidden_size.
        self.wo_b = Nvfp4Linear(
            in_features=config.output_group_dim * config.num_output_groups,
            out_features=config.hidden_size,
        )

        # Attention sink: per-head learnable logit added to the softmax
        # denominator (paper §2.3.3). One scalar per query head.
        self.sink_logits: torch.Tensor | None = None  # set by load_weights

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,            # (T, hidden_size) BF16, post-RMSNorm
        cache: "LayerCacheHandle",   # per-layer cache slice
    ) -> torch.Tensor:               # (T, hidden_size) BF16
        # Branch on attention type. Frozen at construction so this is a
        # constant in any traced graph.
        if self.attn_type == AttentionType.CSA:
            return self._forward_csa(x, cache)
        elif self.attn_type == AttentionType.HCA:
            return self._forward_hca(x, cache)
        else:
            return self._forward_swa(x, cache)

    # ------------------------------------------------------------------
    # CSA: compressed + sparse attention, with indexer + top-k.
    # ------------------------------------------------------------------
    def _forward_csa(self, x, cache):
        # 1) Low-rank query projection. c_Q is shared with the indexer.
        c_Q = self.q_down(x)       # (T, dc)
        q = self.q_up(c_Q)          # (T, n_h * head_dim)

        # 2) KV down-projection — produces (Ca, Cb, Za, Zb) interleaved.
        kv_raw = self.kv_down(x)    # (T, 4 * head_dim)

        # 3) Apply partial RoPE to q and the to-be-compressed keys.
        q_roped = self._apply_rope(q, positions=cache.positions)

        # 4) Compress + write to cache. Returns nothing — the cache holds it.
        from dsv4.kernels.compressor import csa_compress_and_store
        csa_compress_and_store(
            kv_raw=kv_raw,
            cache=cache,  # writes C_comp + Z + indexer keys
            positions=cache.positions,
            compression_ratio=self.config.csa_compression_ratio,
        )

        # 5) Indexer: compute index scores, select top-k compressed entries.
        q_indexer = self.indexer_q_up(c_Q)            # (T, n_I_h * c_I)
        w_indexer = self.indexer_head_weights(x)        # (T, n_I_h)

        from dsv4.kernels.indexer import compute_index_scores_topk
        topk_indices = compute_index_scores_topk(
            q_indexer=q_indexer,
            w_indexer=w_indexer,
            cache=cache,  # provides compressed indexer keys
            top_k=self.config.csa_top_k,
        )

        # 6) FMHA over selected sparse entries + SWA branch, fused sink merge.
        from dsv4.kernels.attention import sparse_fmha_with_swa
        attn_out = sparse_fmha_with_swa(
            q=q_roped,
            cache=cache,
            selected_indices=topk_indices,
            sink_logits=self.sink_logits,
            sliding_window=self.config.sliding_window,
        )  # (T, n_h * head_dim)

        # 7) Inverse RoPE on outputs (paper §2.3.3).
        from dsv4.ops.rope import inverse_rope_bf16
        attn_out = inverse_rope_bf16(attn_out, positions=cache.positions)

        # 8) Grouped output projection: wo_a then wo_b.
        grouped = self.wo_a(attn_out)   # (T, n_groups * d_g)
        return self.wo_b(grouped)       # (T, hidden_size)

    # ------------------------------------------------------------------
    # HCA: heavy compression, dense attention over compressed entries.
    # ------------------------------------------------------------------
    def _forward_hca(self, x, cache):
        c_Q = self.q_down(x)
        q = self.q_up(c_Q)
        kv_raw = self.kv_down(x)    # (T, 2 * head_dim)
        q_roped = self._apply_rope(q, positions=cache.positions)

        from dsv4.kernels.compressor import hca_compress_and_store
        hca_compress_and_store(
            kv_raw=kv_raw,
            cache=cache,
            positions=cache.positions,
            compression_ratio=self.config.hca_compression_ratio,
        )

        # Dense FMHA over ALL compressed entries (not top-k) plus SWA.
        from dsv4.kernels.attention import dense_fmha_with_swa
        attn_out = dense_fmha_with_swa(
            q=q_roped,
            cache=cache,
            sink_logits=self.sink_logits,
            sliding_window=self.config.sliding_window,
        )

        from dsv4.ops.rope import inverse_rope_bf16
        attn_out = inverse_rope_bf16(attn_out, positions=cache.positions)

        grouped = self.wo_a(attn_out)
        return self.wo_b(grouped)

    # ------------------------------------------------------------------
    # SWA: sliding window only, no compression branch.
    # First 2 layers of Flash. Acts as the "local context" foundation.
    # ------------------------------------------------------------------
    def _forward_swa(self, x, cache):
        c_Q = self.q_down(x)
        q = self.q_up(c_Q)
        kv_raw = self.kv_down(x)    # (T, head_dim)
        q_roped = self._apply_rope(q, positions=cache.positions)

        # Write raw KV to the SWA window in the cache. No compressor.
        cache.write_swa(kv_raw)

        # Dense FMHA over the sliding window only.
        from dsv4.kernels.attention import swa_only_fmha
        attn_out = swa_only_fmha(
            q=q_roped,
            cache=cache,
            sink_logits=self.sink_logits,
            sliding_window=self.config.sliding_window,
        )

        from dsv4.ops.rope import inverse_rope_bf16
        attn_out = inverse_rope_bf16(attn_out, positions=cache.positions)

        grouped = self.wo_a(attn_out)
        return self.wo_b(grouped)

    # ------------------------------------------------------------------
    def _apply_rope(self, q: torch.Tensor, positions: torch.Tensor):
        """Apply RoPE to the last `rope_dim` dimensions of each head."""
        from dsv4.ops.rope import forward_rope_partial
        return forward_rope_partial(
            q, positions=positions, rope_dim=self.config.rope_dim,
        )
