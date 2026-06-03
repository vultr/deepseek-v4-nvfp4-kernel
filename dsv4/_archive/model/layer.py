"""A single DSV4 transformer layer.

Structure (paper Figure 2):
    X_l ─→ mHC.pre_block ─→ RMSNorm ─→ Attention ─→ mHC.post_block (using F_attn)
     │
     ▼
    mHC.pre_block ─→ RMSNorm ─→ FFN ─→ mHC.post_block (using F_ffn)
     │
     ▼
    X_{l+1}

Each layer owns:
  - One LayerSpec (from build_schedule).
  - Two mHC instances (one per sub-block).
  - One AttentionSubBlock (type fixed by spec.attn).
  - One FFNSubBlock (router mode fixed by spec.router_mode).
  - Two RMSNorm weight tensors.

The layer is otherwise pure orchestration: no learned params live
directly on TransformerLayer, only on its components.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from dsv4.layers.mhc import mHCLayer
from dsv4.layers.attention import AttentionSubBlock
from dsv4.layers.ffn import FFNSubBlock
from dsv4.layers.norm import RMSNorm  # PyTorch ref for now, fused later
from dsv4.model.layer_schedule import LayerSpec

if TYPE_CHECKING:
    from dsv4.model.config import DSV4Config
    from dsv4.cache.paged_cache import LayerCacheHandle


class TransformerLayer:
    def __init__(self, config: "DSV4Config", spec: LayerSpec):
        self.config = config
        self.spec = spec
        self.layer_idx = spec.layer_idx

        # Two mHC wrappers — one per sub-block. mHCLayer holds its own
        # projection weights (W_pre, W_post, W_comb) and static biases.
        self.mhc_attn = mHCLayer(
            hidden_dim=config.hidden_size,
            n_hc=config.n_hc,
            t_max_sinkhorn=config.sinkhorn_iters,
        )
        self.mhc_ffn = mHCLayer(
            hidden_dim=config.hidden_size,
            n_hc=config.n_hc,
            t_max_sinkhorn=config.sinkhorn_iters,
        )

        # Pre-block norms (one per sub-block).
        self.norm_attn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_ffn = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Sub-blocks — type-frozen at construction.
        self.attn = AttentionSubBlock(config, spec)
        self.ffn = FFNSubBlock(config, spec)

    def forward(
        self,
        X: torch.Tensor,             # (T, n_hc, hidden_size) BF16 — residual streams
        token_ids: torch.Tensor,      # (T,) int32 — for hash routing
        cache: "LayerCacheHandle",
    ) -> torch.Tensor:
        # ---- Attention sub-block ----
        x_attn_in, ctx_attn = self.mhc_attn.pre_block(X)
        x_attn_in = self.norm_attn(x_attn_in)
        F_attn = self.attn.forward(x_attn_in, cache)
        X = self.mhc_attn.post_block(X, F_attn, ctx_attn)

        # ---- FFN sub-block ----
        x_ffn_in, ctx_ffn = self.mhc_ffn.pre_block(X)
        x_ffn_in = self.norm_ffn(x_ffn_in)
        F_ffn = self.ffn.forward(x_ffn_in, token_ids)
        X = self.mhc_ffn.post_block(X, F_ffn, ctx_ffn)

        return X
