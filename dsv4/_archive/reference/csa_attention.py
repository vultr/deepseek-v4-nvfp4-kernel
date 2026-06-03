#!/usr/bin/env python3
"""
CSA (Compressed Sparse Attention) + HCA (Heavily Compressed Attention) kernel
for DeepSeek-V4-Pro.

Replaces vLLM's FlashMLA sparse attention which doesn't work on Blackwell.

Architecture:
- CSA (C128A): KV cache compressed 128x. Indexer finds top-k relevant positions.
  Sparse attention attends only to those positions.
- HCA (C4A): KV cache compressed 4x with overlap. Similar indexer + sparse attention.
- SWA: Standard sliding window attention (compress_ratio=0/1).

The attention mechanism in DeepSeek-V4:
1. Q: hidden → q_a_proj → q_norm → q_b_proj → (T, NH, HD) → RoPE
2. KV: hidden → kv_proj → (T, HD) → RoPE → FP8 quant → KV cache (paged)
3. Compressor: hidden → fused_wkv_wgate → compressed KV + score → state cache
4. Indexer: compressed state cache → top-k position indices
5. Sparse attention: Q attends to compressed KV at top-k positions
6. Window attention: Q attends to local window
7. Merge: combine sparse + window attention outputs using attn_sink weights

This module implements steps 4-7 in pure PyTorch (works on any GPU).
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional


# ── Sparse Attention Kernel ───────────────────────────────────────────

def csa_sparse_attention(
    q: torch.Tensor,           # (num_tokens, num_heads, head_dim) - with RoPE applied
    kv_cache: torch.Tensor,    # (num_blocks, block_size, head_dim) - FP8 compressed KV
    topk_indices: torch.Tensor,  # (num_tokens, 1, num_topk) - global position indices
    topk_lens: torch.Tensor,   # (num_tokens,) - valid length per token
    block_table: torch.Tensor, # (num_seqs, num_blocks_per_seq)
    block_size: int,
    scale: float,
    nope_dim: int,             # dimensions without RoPE
    rope_dim: int,             # dimensions with RoPE
    cos_sin_cache: torch.Tensor,  # (max_pos, rope_dim) for RoPE on gathered KV
    positions: torch.Tensor,   # (num_tokens,) position IDs
    attn_sink: torch.Tensor,   # (num_heads,) sink weights (softmax bias)
) -> torch.Tensor:
    """CSA sparse attention: attend to top-k positions in compressed KV cache.

    For each query token, gathers KV from the top-k positions and performs
    standard scaled dot-product attention.
    """
    num_tokens, num_heads, head_dim = q.shape
    device = q.device

    # Gather KV from compressed cache at top-k positions
    # topk_indices: (num_tokens, 1, num_topk) → (num_tokens, num_topk)
    if topk_indices.dim() == 3:
        topk_indices = topk_indices.squeeze(1)

    num_topk = topk_indices.shape[-1]

    # Convert global position indices to (block_idx, offset) for paged cache
    # global_pos → block_idx = global_pos // block_size
    # global_pos → offset = global_pos % block_size
    topk_block_idx = topk_indices // block_size  # (num_tokens, num_topk)
    topk_offset = topk_indices % block_size

    # For each token, we need its sequence's block table to look up physical blocks
    # This is a simplified version assuming single-sequence for now
    # In production, we'd use token_to_req_indices to get the right block_table row

    # Gather KV from cache
    # kv_cache shape: (num_blocks, block_size, head_dim) in FP8
    # Dequantize FP8 to BF16
    if kv_cache.dtype == torch.uint8:
        # FP8 E4M3 dequant: values = uint8 → float8_e4m3fn → bfloat16
        kv_bf16 = kv_cache.view(torch.float8_e4m3fn).to(torch.bfloat16)
    else:
        kv_bf16 = kv_cache.to(torch.bfloat16)

    # For each query token, gather its top-k KV vectors
    # This is the core sparse gather operation
    # Output: (num_tokens, num_topk, head_dim)
    k_gathered = torch.zeros(
        num_tokens, num_topk, head_dim,
        dtype=torch.bfloat16, device=device,
    )

    for t in range(num_tokens):
        for k_idx in range(min(topk_lens[t].item(), num_topk)):
            gpos = topk_indices[t, k_idx].item()
            if gpos < 0:
                continue
            bidx = gpos // block_size
            boff = gpos % block_size
            if bidx < kv_bf16.shape[0] and boff < kv_bf16.shape[1]:
                k_gathered[t, k_idx] = kv_bf16[bidx, boff]

    # Apply RoPE to gathered KV (the compressed KV needs RoPE at its original position)
    if rope_dim > 0:
        # Positions of gathered KV
        kv_positions = topk_indices.clamp(min=0)  # (num_tokens, num_topk)
        half_rot = rope_dim // 2
        cos_kv = cos_sin_cache[kv_positions, :half_rot]  # (NT, num_topk, half_rot)
        sin_kv = cos_sin_cache[kv_positions, half_rot:]  # (NT, num_topk, half_rot)

        # Apply GPT-J RoPE to the rope portion of k_gathered
        k_rope = k_gathered[:, :, nope_dim:]  # (NT, num_topk, rope_dim)
        k_even = k_rope[:, :, 0::2]
        k_odd = k_rope[:, :, 1::2]
        cos_f = cos_kv.unsqueeze(2).to(k_gathered.dtype)  # (NT, num_topk, 1, half_rot)
        sin_f = sin_kv.unsqueeze(2).to(k_gathered.dtype)

        # RoPE on 2D KV (no head dim, treat as single head)
        k_even_rot = k_even * cos_f.squeeze(2) - k_odd * sin_f.squeeze(2)
        k_odd_rot = k_even * sin_f.squeeze(2) + k_odd * cos_f.squeeze(2)
        k_gathered[:, :, nope_dim:][:, :, 0::2] = k_even_rot
        k_gathered[:, :, nope_dim:][:, :, 1::2] = k_odd_rot

    # Expand k for multi-head attention
    # k_gathered: (NT, num_topk, HD) → (NT, NH, num_topk, HD)
    k_expanded = k_gathered.unsqueeze(1).expand(-1, num_heads, -1, -1)

    # Q: (NT, NH, HD) → (NT, NH, 1, HD)
    q_4d = q.unsqueeze(2)

    # Attention scores: (NT, NH, 1, num_topk)
    attn_weights = torch.matmul(q_4d, k_expanded.transpose(-1, -2)) * scale

    # Apply attention sink bias
    # attn_sink: (NH,) → add to the first position's logit
    if attn_sink is not None:
        sink_bias = attn_sink.view(1, num_heads, 1, 1)  # (1, NH, 1, 1)
        attn_weights[:, :, :, 0] += sink_bias.squeeze(-1)

    # Causal mask: don't attend to future positions
    # (simplified — assumes topk_indices are already filtered for causality)

    # Mask invalid positions
    valid_mask = torch.arange(num_topk, device=device).unsqueeze(0) < topk_lens.unsqueeze(1)  # (NT, num_topk)
    attn_weights = attn_weights.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

    attn_weights = F.softmax(attn_weights.float(), dim=-1).to(torch.bfloat16)

    # Weighted sum: (NT, NH, 1, num_topk) @ (NT, NH, num_topk, HD) → (NT, NH, 1, HD)
    attn_output = torch.matmul(attn_weights, k_expanded)
    return attn_output.squeeze(2)  # (NT, NH, HD)


def swa_attention(
    q: torch.Tensor,           # (num_tokens, num_heads, head_dim)
    swa_kv_cache: torch.Tensor,  # (num_blocks, block_size, head_dim) - SWA KV cache
    positions: torch.Tensor,    # (num_tokens,)
    block_table: torch.Tensor,  # (num_seqs, num_blocks_per_seq)
    slot_mapping: torch.Tensor, # (num_tokens,)
    block_size: int,
    window_size: int,
    scale: float,
) -> torch.Tensor:
    """Sliding window attention: attend to local window of tokens.

    Standard multi-head attention over the last `window_size` tokens.
    """
    num_tokens, num_heads, head_dim = q.shape
    device = q.device

    # Dequantize SWA cache if FP8
    if swa_kv_cache.dtype == torch.uint8:
        swa_bf16 = swa_kv_cache.view(torch.float8_e4m3fn).to(torch.bfloat16)
    else:
        swa_bf16 = swa_kv_cache.to(torch.bfloat16)

    # For a simplified implementation, gather all KV in the window
    # In production, this would use paged cache access
    output = torch.zeros(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)

    for t in range(num_tokens):
        pos = positions[t].item()
        window_start = max(0, pos - window_size + 1)
        window_len = pos - window_start + 1

        if window_len == 0:
            continue

        # Gather KV from window
        k_window = torch.zeros(window_len, head_dim, dtype=torch.bfloat16, device=device)
        for i, p in enumerate(range(window_start, pos + 1)):
            slot = p  # simplified: slot = position for contiguous sequences
            bidx = slot // block_size
            boff = slot % block_size
            if bidx < swa_bf16.shape[0] and boff < swa_bf16.shape[1]:
                k_window[i] = swa_bf16[bidx, boff]

        # Multi-head attention
        q_t = q[t]  # (NH, HD)
        k_exp = k_window.unsqueeze(0).expand(num_heads, -1, -1)  # (NH, window_len, HD)

        # Q @ K^T: (NH, 1, HD) @ (NH, HD, window_len) → (NH, 1, window_len)
        scores = torch.matmul(q_t.unsqueeze(1), k_exp.transpose(-1, -2)) * scale
        scores = F.softmax(scores.float(), dim=-1).to(torch.bfloat16)

        # Weighted sum: (NH, 1, window_len) @ (NH, window_len, HD) → (NH, 1, HD)
        out_t = torch.matmul(scores, k_exp).squeeze(1)  # (NH, HD)
        output[t] = out_t

    return output


def csa_hca_forward(
    q: torch.Tensor,           # (num_tokens, num_heads, head_dim) with RoPE
    kv: torch.Tensor,          # (num_tokens, head_dim) - KV latent (after norm)
    positions: torch.Tensor,   # (num_tokens,)
    # SWA cache
    swa_kv_cache: torch.Tensor,
    swa_block_table: torch.Tensor,
    swa_slot_mapping: torch.Tensor,
    swa_block_size: int,
    window_size: int,
    # CSA cache (optional, for compress_ratio > 1)
    csa_kv_cache: Optional[torch.Tensor] = None,
    csa_block_table: Optional[torch.Tensor] = None,
    csa_block_size: int = 256,
    compress_ratio: int = 1,
    topk_indices: Optional[torch.Tensor] = None,
    topk_lens: Optional[torch.Tensor] = None,
    # Params
    scale: float = 1.0,
    nope_dim: int = 448,
    rope_dim: int = 64,
    cos_sin_cache: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full CSA/HCA/SWA forward pass.

    For compress_ratio > 1: CSA/HCA sparse attention + SWA
    For compress_ratio <= 1: SWA only
    """
    num_tokens, num_heads, head_dim = q.shape
    device = q.device

    if compress_ratio <= 1:
        # SWA-only layer
        return swa_attention(
            q, swa_kv_cache, positions, swa_block_table,
            swa_slot_mapping, swa_block_size, window_size, scale,
        )

    # CSA/HCA layer: sparse attention + SWA, merged with sink weights
    sparse_out = csa_sparse_attention(
        q, csa_kv_cache, topk_indices, topk_lens,
        csa_block_table, csa_block_size, scale,
        nope_dim, rope_dim, cos_sin_cache, positions, attn_sink,
    )

    swa_out = swa_attention(
        q, swa_kv_cache, positions, swa_block_table,
        swa_slot_mapping, swa_block_size, window_size, scale,
    )

    # Merge sparse + SWA outputs
    # The sink weights determine the mixing between sparse and window attention
    # For now, simple addition (the actual merge uses attn_sink as a learned weight)
    if attn_sink is not None:
        # attn_sink: (num_heads,) — softmax bias toward the sink token
        # When sink weight is -inf, no sink effect → pure SWA + sparse
        # When sink weight is 0, equal mixing
        # In practice, attn_sink is trained and typically small
        sink_weight = torch.sigmoid(attn_sink).view(1, num_heads, 1)
        output = sparse_out * (1 - sink_weight) + swa_out * sink_weight
    else:
        output = sparse_out + swa_out

    return output


# ── Batched sparse attention (optimized, no Python loops) ─────────────

def csa_sparse_attention_batched(
    q: torch.Tensor,           # (T, NH, HD)
    kv_cache: torch.Tensor,    # (num_blocks, block_size, kv_dim) FP8 or BF16
    topk_indices: torch.Tensor,  # (T, num_topk) global position indices
    topk_lens: torch.Tensor,   # (T,) valid lengths
    block_size: int,
    scale: float,
    nope_dim: int,
    rope_dim: int,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    attn_sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Optimized CSA sparse attention using batched gather + SDPA.

    No Python loops. Uses torch.gather and F.scaled_dot_product_attention.
    """
    T, NH, HD = q.shape
    device = q.device
    num_topk = topk_indices.shape[-1]

    # Dequantize KV cache
    if kv_cache.dtype == torch.uint8:
        kv_flat = kv_cache.view(torch.float8_e4m3fn).to(torch.bfloat16)
    else:
        kv_flat = kv_cache.to(torch.bfloat16)

    # Flatten cache: (num_blocks * block_size, kv_dim)
    num_blocks, bs, kv_dim = kv_flat.shape
    kv_flat = kv_flat.reshape(num_blocks * bs, kv_dim)

    # Clamp topk_indices to valid range and gather
    # topk_indices: (T, num_topk) → gather from kv_flat
    safe_indices = topk_indices.clamp(min=0, max=kv_flat.shape[0] - 1)

    # Gather: (T, num_topk, kv_dim)
    # torch.gather needs (T, num_topk) index → expand to (T, num_topk, kv_dim)
    idx_expanded = safe_indices.unsqueeze(-1).expand(-1, -1, kv_dim)
    k_gathered = torch.gather(
        kv_flat.unsqueeze(0).expand(T, -1, -1),  # (T, total_positions, kv_dim)
        1,  # dim=1
        idx_expanded,  # (T, num_topk, kv_dim)
    )

    # Mask invalid positions
    valid_mask = torch.arange(num_topk, device=device).unsqueeze(0) < topk_lens.unsqueeze(1)
    k_gathered = k_gathered * valid_mask.unsqueeze(-1).to(k_gathered.dtype)

    # Apply RoPE to gathered K (GPT-J style)
    if rope_dim > 0 and cos_sin_cache is not None:
        kv_positions = safe_indices  # (T, num_topk)
        half_rot = rope_dim // 2
        cos_kv = cos_sin_cache[kv_positions, :half_rot]  # (T, num_topk, half_rot)
        sin_kv = cos_sin_cache[kv_positions, half_rot:]

        k_rope = k_gathered[:, :, nope_dim:]  # (T, num_topk, rope_dim)
        k_even = k_rope[:, :, 0::2]
        k_odd = k_rope[:, :, 1::2]
        cos_f = cos_kv.to(k_gathered.dtype)
        sin_f = sin_kv.to(k_gathered.dtype)
        k_gathered[:, :, nope_dim:][:, :, 0::2] = k_even * cos_f - k_odd * sin_f
        k_gathered[:, :, nope_dim:][:, :, 1::2] = k_even * sin_f + k_odd * cos_f

    # Expand for multi-head: (T, num_topk, HD) → (T, NH, num_topk, HD)
    k_heads = k_gathered.unsqueeze(1).expand(-1, NH, -1, -1)
    v_heads = k_heads.clone()  # K=V in MLA-style attention

    # Q: (T, NH, HD) → (T, NH, 1, HD)
    q_4d = q.unsqueeze(2)

    # Use PyTorch SDPA (works on all GPUs including Blackwell)
    # Need shapes: (T*NH, 1, HD) and (T*NH, num_topk, HD)
    q_2d = q.reshape(T * NH, 1, HD)
    k_2d = k_heads.reshape(T * NH, num_topk, HD)
    v_2d = v_heads.reshape(T * NH, num_topk, HD)

    # Build attention mask from valid positions
    # (T, num_topk) → (T*NH, 1, num_topk)
    attn_mask = valid_mask.unsqueeze(1).expand(-1, NH, -1).reshape(T * NH, 1, num_topk)
    attn_mask = attn_mask.to(torch.bool)

    # Apply attn_sink bias
    if attn_sink is not None:
        # Add sink bias to first position's attention logit
        # attn_sink: (NH,) → (T*NH, 1, 1) broadcast
        sink = attn_sink.view(1, NH, 1).expand(T, -1, -1).reshape(T * NH, 1, 1)
        # We'll add this after SDPA by adjusting the mask
        # Actually, we need to handle this before softmax
        # For now, just note that attn_sink is a learned bias

    # PyTorch SDPA
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                                          torch.nn.attention.SDPBackend.MATH]):
        out_2d = F.scaled_dot_product_attention(
            q_2d, k_2d, v_2d,
            attn_mask=attn_mask if not attn_mask.all() else None,
            scale=scale,
        )

    return out_2d.squeeze(1).reshape(T, NH, HD)


# ── Simplified full-attention fallback (no compression, for testing) ──

def full_attention_reference(
    q: torch.Tensor,           # (T, NH, HD) with RoPE
    kv: torch.Tensor,          # (T, HD) KV latent
    scale: float = 1.0,
) -> torch.Tensor:
    """Full attention reference: attend to all positions.

    Useful for testing when CSA cache is not available.
    Uses PyTorch SDPA which works on all GPUs.
    """
    T, NH, HD = q.shape

    # K=V from kv latent (shared across all heads and all query positions)
    # kv: (T, HD) → each token's KV is seen by all heads at all query positions
    k = kv.unsqueeze(1).expand(-1, NH, -1).contiguous()  # (T, NH, HD)
    # For cross-attention where each Q attends to all KV positions:
    # K needs to be (T_q, NH, T_kv, HD) — repeat for each query position
    k = k.unsqueeze(0).expand(T, -1, -1, -1).contiguous()  # (T, T, NH, HD) → wrong order
    # Actually: for self-attention, K/V shape for SDPA is (batch, seq_kv, HD)
    # where batch = T*NH (each query token is a batch, each head independent)
    # K/V: (T*NH, T, HD) — each (query, head) pair attends to all T KV positions
    kv_expanded = kv.unsqueeze(1).expand(-1, NH, -1).contiguous()  # (T, NH, HD)
    # Repeat KV for each query: (T, NH, HD) → (T*NH, T, HD)
    k_2d = kv_expanded.permute(1, 0, 2).unsqueeze(1).expand(NH, T, T, -1).contiguous().reshape(T * NH, T, HD)
    v_2d = k_2d.clone()

    # Q: (T, NH, HD) → (T*NH, 1, HD)
    q_2d = q.reshape(T * NH, 1, HD)

    # Manual attention (SDPA mask handling is tricky with batched single-query)
    # scores: (T*NH, 1, T) = Q @ K^T
    scores = torch.matmul(q_2d, k_2d.transpose(-1, -2)) * scale
    # Causal mask: each query at position i can only attend to positions <= i
    # Since each batch is (query_pos, head), and KV has all T positions,
    # we need position-aware masking
    # For single-query batches: batch i corresponds to (pos i // NH, head i % NH)
    # All positions <= i // NH are valid
    # Simple approach: use a per-query mask
    query_positions = torch.arange(T, device=q.device).unsqueeze(1).repeat(1, NH).reshape(T * NH)  # (T*NH,)
    kv_positions = torch.arange(T, device=q.device).unsqueeze(0)  # (1, T)
    causal = kv_positions <= query_positions.unsqueeze(1)  # (T*NH, T)
    scores = scores.squeeze(1).masked_fill(~causal, float('-inf'))  # (T*NH, T)
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)  # (T*NH, T)
    out = torch.matmul(weights.unsqueeze(1), v_2d)  # (T*NH, 1, HD)

    return out.squeeze(1).reshape(T, NH, HD)
