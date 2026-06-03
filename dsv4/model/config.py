"""DSV4 architecture configuration.

One config class for both Flash and Pro — they differ only in numeric
parameters, not structure. Construct via .flash() / .pro() classmethods
or load from a checkpoint's config.json.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DSV4Config:
    # Identity
    variant: str  # "flash" | "pro"
    vocab_size: int

    # Transformer
    num_layers: int  # Flash: 43, Pro: 61
    hidden_size: int  # Flash: 4096, Pro: 7168

    # Attention (shared across CSA/HCA/SWA)
    num_query_heads: int  # Flash: 64, Pro: 128
    head_dim: int  # Both: 512
    rope_dim: int  # Both: 64 (partial RoPE on last 64 dims)
    query_compression_dim: int  # Flash: 1024, Pro: 1536

    # CSA-specific
    csa_compression_ratio: int  # m, both: 4
    csa_top_k: int  # Flash: 512, Pro: 1024
    indexer_num_heads: int  # Both: 64
    indexer_head_dim: int  # Both: 128

    # HCA-specific
    hca_compression_ratio: int  # m', both: 128

    # SWA branch
    sliding_window: int  # n_win, both: 128

    # Output projection grouping
    num_output_groups: int  # Flash: 8, Pro: 16
    output_group_dim: int  # Both: 1024

    # MoE
    num_routed_experts: int  # Flash: 256, Pro: 384
    num_shared_experts: int  # Both: 1
    num_experts_per_tok: int  # Both: 6
    moe_intermediate_size: int  # Flash: 2048, Pro: 3072
    num_hash_routing_layers: int  # Both: 3
    routed_scaling_factor: float  # Verify per checkpoint

    # mHC
    n_hc: int  # Both: 4
    sinkhorn_iters: int  # Both: 20

    # Architecture schedule — first N layers use a non-standard attention
    # pattern (paper §4.2.1).
    # Flash: layers 0-1 = SWA only
    # Pro: layers 0-1 = HCA only
    num_special_first_layers: int = 2

    # Norm
    rms_norm_eps: float = 1e-6

    @classmethod
    def flash(cls) -> "DSV4Config":
        return cls(
            variant="flash",
            vocab_size=128_000,
            num_layers=43,
            hidden_size=4096,
            num_query_heads=64,
            head_dim=512,
            rope_dim=64,
            query_compression_dim=1024,
            csa_compression_ratio=4,
            csa_top_k=512,
            indexer_num_heads=64,
            indexer_head_dim=128,
            hca_compression_ratio=128,
            sliding_window=128,
            num_output_groups=8,
            output_group_dim=1024,
            num_routed_experts=256,
            num_shared_experts=1,
            num_experts_per_tok=6,
            moe_intermediate_size=2048,
            num_hash_routing_layers=3,
            routed_scaling_factor=2.5,
            n_hc=4,
            sinkhorn_iters=20,
        )

    @classmethod
    def pro(cls) -> "DSV4Config":
        return cls(
            variant="pro",
            vocab_size=128_000,
            num_layers=61,
            hidden_size=7168,
            num_query_heads=128,
            head_dim=512,
            rope_dim=64,
            query_compression_dim=1536,
            csa_compression_ratio=4,
            csa_top_k=1024,
            indexer_num_heads=64,
            indexer_head_dim=128,
            hca_compression_ratio=128,
            sliding_window=128,
            num_output_groups=16,
            output_group_dim=1024,
            num_routed_experts=384,
            num_shared_experts=1,
            num_experts_per_tok=6,
            moe_intermediate_size=3072,
            num_hash_routing_layers=3,
            routed_scaling_factor=2.5,
            n_hc=4,
            sinkhorn_iters=20,
        )
