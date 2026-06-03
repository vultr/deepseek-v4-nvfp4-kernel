"""CSA/HCA compressor — functional API for flush pipeline.

The compressor runs token-level softmax over m entries (CSA) or m' entries (HCA)
to produce compressed KV entries. The compressed entries are then written to the
paged pool by the flush_write kernel.

CSA (paper eq. 11-12):
  entry_i = sum_j (softmax(Z_a[i,:], Z_b[i,:]) * concat(K_a[i,:], K_b[i,:]))
  where j ranges over the 2*m tokens (m from a-stream + m from b-stream)

HCA:
  entry_i = sum_j (softmax(Z_a[i,:]) * K_a[i,:])
  where j ranges over m' tokens (no b-stream)
"""
import torch
from typing import Optional


def csa_compress_tail(
    tail_ka: torch.Tensor,      # (max_req, m, head_dim) BF16 — current a-stream KV
    tail_za: torch.Tensor,      # (max_req, m, head_dim) BF16 — a-stream Z weights
    tail_kb: torch.Tensor,      # (max_req, m, head_dim) BF16 — previous b-stream KV
    tail_zb: torch.Tensor,      # (max_req, m, head_dim) BF16 — b-stream Z weights
    tail_len: torch.Tensor,     # (max_req,) int32 — valid entries in a-stream
    request_slots: torch.Tensor, # (B,) int32
    m: int = 4,                 # compression ratio
) -> tuple[torch.Tensor, torch.Tensor]:
    """CSA: compress tail entries into one compressed entry per request.

    Args:
        tail_ka, tail_za: a-stream (current block's tokens)
        tail_kb, tail_zb: b-stream (previous block's tokens)
        tail_len: number of valid entries
        request_slots: which request slots to process
        m: compression ratio (4 for CSA)

    Returns:
        (entry, indexer_key)
        entry: (B, head_dim) BF16 — compressed KV entry
        indexer_key: (B, indexer_head_dim) BF16 — key for indexer scoring
    """
    B = request_slots.shape[0]
    head_dim = tail_ka.shape[-1]

    entries = []
    indexer_keys = []

    for b in range(B):
        slot = request_slots[b].item()
        valid_len = tail_len[slot].item()

        if valid_len < m:
            # Not enough tokens — zero fill
            entries.append(torch.zeros(head_dim, dtype=torch.bfloat16, device=tail_ka.device))
            indexer_keys.append(torch.zeros(head_dim, dtype=torch.bfloat16, device=tail_ka.device))
            continue

        # Gather a-stream and b-stream entries
        ka = tail_ka[slot, :m].float()   # (m, head_dim)
        za = tail_za[slot, :m].float()   # (m, head_dim)
        kb = tail_kb[slot, :m].float()   # (m, head_dim)
        zb = tail_zb[slot, :m].float()   # (m, head_dim)

        # Concatenate a-stream and b-stream
        k_cat = torch.cat([ka, kb], dim=0)  # (2m, head_dim)
        z_cat = torch.cat([za, zb], dim=0)  # (2m, head_dim)

        # Token-level softmax: for each head dimension d,
        # compute softmax over the 2m tokens
        # Z values are the logits for the softmax
        # The paper uses learned Z projections; here we treat Z as the
        # pre-softmax logits.
        # softmax over dim=0 (token dimension) for each head dim
        z_max = z_cat.max(dim=0, keepdim=True).values  # (1, head_dim)
        z_exp = torch.exp(z_cat - z_max)  # (2m, head_dim)
        z_sum = z_exp.sum(dim=0, keepdim=True)  # (1, head_dim)
        weights = z_exp / z_sum  # (2m, head_dim) — per-token, per-dim weights

        # Weighted sum: entry = sum_j (weights[j] * k_cat[j])
        entry = (weights * k_cat).sum(dim=0)  # (head_dim)
        entries.append(entry.bfloat16())

        # Indexer key: same compression but on a different projection.
        # For now, use the same entry as the indexer key.
        # The real implementation would use a separate Q_indexer projection.
        indexer_keys.append(entry.bfloat16())

    return torch.stack(entries), torch.stack(indexer_keys)


def hca_compress_tail(
    tail_ka: torch.Tensor,      # (max_req, m_prime, head_dim) BF16
    tail_za: torch.Tensor,      # (max_req, m_prime, head_dim) BF16
    tail_len: torch.Tensor,     # (max_req,) int32
    request_slots: torch.Tensor, # (B,) int32
    m: int = 128,               # HCA compression ratio
) -> torch.Tensor:
    """HCA: compress tail entries into one compressed entry per request.

    No b-stream, no overlap. Dense attention over the compressed sequence.

    Returns:
        entry: (B, head_dim) BF16 — compressed KV entry
    """
    B = request_slots.shape[0]
    head_dim = tail_ka.shape[-1]

    entries = []

    for b in range(B):
        slot = request_slots[b].item()
        valid_len = tail_len[slot].item()

        if valid_len < m:
            entries.append(torch.zeros(head_dim, dtype=torch.bfloat16, device=tail_ka.device))
            continue

        ka = tail_ka[slot, :m].float()  # (m, head_dim)
        za = tail_za[slot, :m].float()  # (m, head_dim)

        z_max = za.max(dim=0, keepdim=True).values
        z_exp = torch.exp(za - z_max)
        z_sum = z_exp.sum(dim=0, keepdim=True)
        weights = z_exp / z_sum  # (m, head_dim)

        entry = (weights * ka).sum(dim=0)  # (head_dim)
        entries.append(entry.bfloat16())

    return torch.stack(entries)
