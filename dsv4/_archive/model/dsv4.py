"""Full DSV4 model — embedding → N×DSV4Layer → final norm → prediction head.

Supports both Flash and Pro variants via DSV4Config.
MTP (multi-token prediction) is wired but optional (off by default).
"""
from __future__ import annotations
from typing import Optional, List
import torch

from dsv4.model.config import DSV4Config
from dsv4.model.layer import TransformerLayer
from dsv4.model.layer_schedule import build_schedule, LayerSpec
from dsv4.layers.norm import RMSNorm
from dsv4.layers.linear import Nvfp4Linear
from dsv4.cache.manager import KVCacheManager


class DSV4Model:
    """Full DeepSeek-V4 model for inference.

    Construction:
        config = DSV4Config.pro()
        model = DSV4Model(config, cache_manager)
        model.load_weights(checkpoint_path)

    Decode step:
        logits = model.decode_step(token_ids, positions, request_ids)
        next_token = sampler(logits)
    """

    def __init__(
        self,
        config: DSV4Config,
        cache_manager: KVCacheManager,
        device: str = "cuda",
    ):
        self.config = config
        self.cache_manager = cache_manager
        self.device = device
        self.schedule = build_schedule(config)

        # ---- Token embedding ----
        self.token_embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device,
        )

        # ---- Transformer layers ----
        self.layers: List[TransformerLayer] = [
            TransformerLayer(config, spec) for spec in self.schedule
        ]

        # ---- Final norm ----
        self.final_norm = RMSNorm(config.hidden_size, device=device)

        # ---- Prediction head (tied with embedding or separate) ----
        self.lm_head = Nvfp4Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
        )
        self._weights_tied = False

        # mHC state shape: (batch, n_hc, hidden_size) per layer
        self.n_hc = config.n_hc

    def load_weights(self, checkpoint_path: str) -> None:
        """Load weights from a checkpoint directory.

        TODO: implement HF checkpoint loader (dsv4/loader/hf_checkpoint.py).
        For now, weights must be set manually.
        """
        raise NotImplementedError(
            "Weight loading not yet implemented. "
            "See dsv4/loader/hf_checkpoint.py"
        )

    def tie_weights(self) -> None:
        """Tie lm_head weights to token_embedding (common in LLMs)."""
        self.lm_head.weight = self.token_embedding.weight
        self._weights_tied = True

    def decode_step(
        self,
        token_ids: torch.Tensor,    # (batch,) int64
        positions: torch.Tensor,     # (batch,) int64
        request_ids: torch.Tensor,   # (batch,) int32
        mhc_states: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """Single decode step: token_ids → logits.

        Args:
            token_ids: (batch,) int64 — token IDs for this step
            positions: (batch,) int64 — absolute positions
            request_ids: (batch,) int32 — which request each token belongs to
            mhc_states: optional list of (batch, n_hc, hidden_size) BF16 per layer.
                If None, initialized from embedding (first step).

        Returns:
            (logits, updated_mhc_states)
            logits: (batch, vocab_size) BF16
            mhc_states: List[Tensor] per layer
        """
        batch = token_ids.shape[0]
        T = 1  # decode

        # Embed → (batch, hidden_size) → (T, batch, hidden_size) → mHC state
        emb = self.token_embedding(token_ids)  # (batch, hidden_size)

        # mHC state: X_0 = expand to (T, n_hc, hidden_size)
        # At layer 0, the first mHC state is initialized from the embedding.
        # X[0, i, :] = emb for all i (paper: identity initialization before Sinkhorn)
        if mhc_states is None:
            mhc_states = []
            for _ in self.layers:
                x = torch.zeros(batch, self.n_hc, self.config.hidden_size,
                                dtype=torch.bfloat16, device=self.device)
                # First layer: broadcast embedding into all n_hc slots
                x[:, :, :] = emb.unsqueeze(1)
                mhc_states.append(x)

        # Get cache handles for each layer
        # request_slots are the state cache slots for these requests
        request_slots = self.cache_manager.request_slot_map[:batch].clone()

        for layer_idx, (layer, spec) in enumerate(zip(self.layers, self.schedule)):
            cache = self.cache_manager.acquire(
                layer_idx, request_slots, positions, request_ids,
            )

            X = mhc_states[layer_idx]  # (batch, n_hc, hidden_size)
            # TransformerLayer expects (T, n_hc, hidden_size)
            X = X.unsqueeze(0)  # (1, n_hc, hidden_size) — T=1

            # token_ids needed for hash routing
            X = layer.forward(X, token_ids, cache)

            mhc_states[layer_idx] = X.squeeze(0)  # (batch, n_hc, hidden_size)

        # Final output: take the last mHC channel and apply norm + head
        # X has been updated in-place by the last layer's mHC.post_block
        # The output is from the first channel (paper: identity residual)
        x_out = mhc_states[-1][:, 0, :]  # (batch, hidden_size)
        x_out = self.final_norm(x_out)
        logits = self.lm_head(x_out)  # (batch, vocab_size)
        return logits, mhc_states

    def forward(
        self,
        token_ids: torch.Tensor,    # (T,) int64 — prefill tokens
        positions: torch.Tensor,     # (T,) int64
        request_ids: torch.Tensor,   # (T,) int32
        request_slots: torch.Tensor, # (batch,) int32
    ) -> torch.Tensor:
        """Prefill: process a sequence of tokens.

        Returns:
            (T, vocab_size) BF16 logits
        """
        T = token_ids.shape[0]
        emb = self.token_embedding(token_ids)  # (T, hidden_size)

        for layer_idx, (layer, spec) in enumerate(zip(self.layers, self.schedule)):
            cache = self.cache_manager.acquire(
                layer_idx, request_slots, positions, request_ids,
            )
            # Initialize mHC state from embedding
            X = torch.zeros(T, self.n_hc, self.config.hidden_size,
                            dtype=torch.bfloat16, device=self.device)
            X[:, :, :] = emb.unsqueeze(1)

            X = layer.forward(X, token_ids, cache)

        # Output from last layer
        x_out = X[:, 0, :]  # (T, hidden_size)
        x_out = self.final_norm(x_out)
        logits = self.lm_head(x_out)  # (T, vocab_size)
        return logits
