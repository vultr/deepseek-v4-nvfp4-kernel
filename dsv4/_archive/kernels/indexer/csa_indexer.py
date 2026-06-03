"""CSA indexer — sparse top-k selection from compressed KV cache.

Paper §2.3.1, eq. 13–17:
  c_Q = h_t · W_DQ    (shared with main queries)
  q^I_t = c_Q · W_IUQ  (low-rank indexer queries)
  w^I_t = h_t · W_w    (per-head weights)
  I[t,s] = Σ_h w^I_t,h · ReLU(q^I_t,h · K^IComp[s])  (MQA: shared key K)
  Selected = TopK(I[t,:])

Key layout: K^IComp[s] is shared across indexer heads (MQA, NOT per-head).
The dot product is: q^I_t,h (per-head) · K^IComp[s] (shared).
This matches the production Indexer.forward() einsum 'tnd,cd->tnc'.

RoPE: Neither indexer queries nor keys have RoPE applied.
The indexer is a lightweight scoring mechanism for block selection,
not a full attention layer. If the HF reference applies RoPE to
indexer keys, the stored FP4 keys would need it baked in at
compression time. VERIFY THIS AGAINST THE REFERENCE BEFORE PRODUCTION.

The indexer only exists in CSA layers. HCA and SWA layers don't have
an indexer (they do dense attention).
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from dsv4.model.config import DSV4Config
    from dsv4.cache.handle import LayerCacheHandle


class CSAIndexer:
    """Lightning indexer for CSA layers.

    Composed by AttentionSubBlock when layer is CSA. Owns W_IUQ and W_w.
    The shared c_Q comes from the main query path; this class does NOT
    own W_DQ.
    """

    def __init__(self, config: "DSV4Config"):
        self.config = config
        self._runner_id = None

    def __call__(
        self,
        c_Q: torch.Tensor,     # [T, d_c] BF16 — shared latent
        h_t: torch.Tensor,     # [T, d] BF16 — hidden states
        cache: "LayerCacheHandle",
    ) -> torch.Tensor:
        """Return top-k compressed-block indices per query token.

        Returns [T, csa_top_k] int32 indices into the compressed pool.
        """
        from dsv4.kernels.indexer.score_topk import run_indexer_score_topk

        # Kernel A: indexer query up-projection (c_Q -> q_I)
        # For now, use a simple torch linear; will swap to Nvfp4Linear
        # with FP4 output in Phase 2.
        if not hasattr(self, '_q_up_weight'):
            # WARNING: USING RANDOM WEIGHTS — csa_indexer.py has NO weight loading.
            # The production path uses the Indexer class in single_shot_inference.py
            # which loads real weights from the checkpoint via Nvfp4Linear.
            # This CSAIndexer class should NOT be used for production inference.
            # If you see this message, you need to wire up checkpoint weight loading
            # or use the production Indexer instead.
            raise RuntimeError(
                "CSAIndexer has no checkpoint weight loading. "
                "Use the production Indexer class (single_shot_inference.py) instead, "
                "or implement weight loading for CSAIndexer.")
            # Old code (random weights — removed to prevent silent incorrect behavior):
            # d_c = self.config.query_compression_dim
            # n_ih = self.config.indexer_num_heads
            # c_i = self.config.indexer_head_dim
            # self._q_up_weight = torch.randn(d_c, n_ih * c_i, ...) * 0.02
            # self._w_head_weight = torch.randn(hidden_size, n_ih, ...) * 0.02

        q_I = torch.nn.functional.linear(c_Q, self._q_up_weight.T)  # [T, n_ih * c_i] BF16
        w_h = torch.nn.functional.linear(h_t, self._w_head_weight.T).float()  # [T, n_ih] FP32

        view = cache.read_indexer_view()
        return run_indexer_score_topk(
            q_I=q_I,
            w_h=w_h,
            indexer_view=view,
            num_heads=self.config.indexer_num_heads,
            head_dim=self.config.indexer_head_dim,
            top_k=self.config.csa_top_k,
            entries_per_block=cache.paged.schema.entries_per_block,
        )
