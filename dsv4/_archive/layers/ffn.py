"""DSV4 FFN sub-block — routed MoE + shared expert.

The router instance encapsulates hash-vs-dense; this sub-block doesn't
have to care, it just calls router(x, token_ids) and feeds the result
to Nvfp4MoE. Shared expert runs in parallel (logically — kernels
can overlap).
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from dsv4.layers.router import Router
from dsv4.layers.moe import Nvfp4MoE
from dsv4.layers.shared_expert import Nvfp4SharedExpert
from dsv4.model.layer_schedule import LayerSpec, RouterMode

if TYPE_CHECKING:
    from dsv4.model.config import DSV4Config


class FFNSubBlock:
    def __init__(self, config: "DSV4Config", spec: LayerSpec):
        self.config = config
        self.spec = spec

        self.router = Router(
            hidden_size=config.hidden_size,
            num_experts=config.num_routed_experts,
            top_k=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            mode="hash" if spec.router_mode == RouterMode.HASH else "dense",
            vocab_size=config.vocab_size if spec.router_mode == RouterMode.HASH else None,
        )
        self.moe = Nvfp4MoE(
            num_experts=config.num_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            top_k=config.num_experts_per_tok,
        )
        self.shared = Nvfp4SharedExpert(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )

    def forward(
        self,
        x: torch.Tensor,          # (T, hidden_size) BF16, post-RMSNorm
        token_ids: torch.Tensor,   # (T,) int32 — needed only for hash routing
    ) -> torch.Tensor:
        topk_w, topk_ids = self.router(x, token_ids=token_ids)
        routed_out = self.moe.run(x, topk_w, topk_ids)
        shared_out = self.shared.run(x)
        return routed_out + shared_out
