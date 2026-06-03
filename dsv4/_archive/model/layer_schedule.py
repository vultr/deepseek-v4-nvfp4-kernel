"""Per-layer-index architecture schedule for DSV4.

Encodes the attention / FFN / routing pattern from the paper. Kept
separate from layer.py so the KV cache allocator can query it without
instantiating any layers.
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import List

from dsv4.model.config import DSV4Config


class AttentionType(Enum):
    SWA = "swa"   # Sliding window only (Flash layers 0-1)
    HCA = "hca"   # Heavily Compressed Attention (Pro layers 0-1, alternating later)
    CSA = "csa"   # Compressed Sparse Attention (alternating from layer 2)


class FFNType(Enum):
    MOE = "moe"   # DeepSeekMoE — both variants use MoE in every layer


class RouterMode(Enum):
    HASH = "hash"   # Layers 0-2: deterministic by token ID
    DENSE = "dense"  # Layers 3+: sqrt(softplus) + top-k


@dataclass(frozen=True)
class LayerSpec:
    """The complete architectural type of a single transformer layer."""
    layer_idx: int
    attn: AttentionType
    ffn: FFNType
    router_mode: RouterMode


def build_schedule(config: DSV4Config) -> List[LayerSpec]:
    """Produce the full per-layer schedule for a given config.

    Paper §4.2.1:
      Flash: first 2 layers = pure SWA, then alternating CSA / HCA.
      Pro: first 2 layers = HCA, then alternating CSA / HCA.
    Both: first 3 MoE layers use hash routing.

    'Alternating CSA / HCA' starts with CSA at layer 2 in both variants.
    """
    schedule: List[LayerSpec] = []
    first_special = config.num_special_first_layers  # 2

    for i in range(config.num_layers):
        # Attention type
        if i < first_special:
            if config.variant == "flash":
                attn = AttentionType.SWA
            else:  # pro
                attn = AttentionType.HCA
        else:
            # Layer 2 is CSA, layer 3 is HCA, layer 4 is CSA, ...
            attn = AttentionType.CSA if (i - first_special) % 2 == 0 else AttentionType.HCA

        # FFN type — DSV4 is MoE in every layer.
        ffn = FFNType.MOE

        # Router mode — hash for the first N MoE layers.
        router = RouterMode.HASH if i < config.num_hash_routing_layers else RouterMode.DENSE

        schedule.append(LayerSpec(layer_idx=i, attn=attn, ffn=ffn, router_mode=router))

    return schedule


def validate_schedule(schedule: List[LayerSpec], config: DSV4Config) -> None:
    """Sanity checks. Wrong schedule = silent garbage, so be loud here."""
    assert len(schedule) == config.num_layers, \
        f"schedule has {len(schedule)} layers, config says {config.num_layers}"

    # First N hash routing layers
    for i in range(config.num_hash_routing_layers):
        assert schedule[i].router_mode == RouterMode.HASH, \
            f"layer {i} should be HASH-routed"
    for i in range(config.num_hash_routing_layers, config.num_layers):
        assert schedule[i].router_mode == RouterMode.DENSE, \
            f"layer {i} should be DENSE-routed"

    # First two layers are special
    expected_first = AttentionType.SWA if config.variant == "flash" else AttentionType.HCA
    for i in range(config.num_special_first_layers):
        assert schedule[i].attn == expected_first, \
            f"layer {i} should be {expected_first} for {config.variant}"

    # Alternation from layer 2
    for i in range(config.num_special_first_layers, config.num_layers):
        expected = AttentionType.CSA if (i - config.num_special_first_layers) % 2 == 0 \
            else AttentionType.HCA
        assert schedule[i].attn == expected, \
            f"layer {i} should be {expected}, got {schedule[i].attn}"

    # Every layer is MoE
    for i, spec in enumerate(schedule):
        assert spec.ffn == FFNType.MOE, f"layer {i} should be MOE"
