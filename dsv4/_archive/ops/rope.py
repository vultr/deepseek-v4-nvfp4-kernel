"""Inverse RoPE + NVFP4 wo_a grouped GEMM for DeepSeek V4 attention.

Replaces:
  1. fused_inv_rope_fp8_quant (CUDA kernel) → inverse_rope_bf16 (Python)
  2. deepseek_v4_fp8_einsum (DeepGEMM) → CuTeDSL NVFP4 grouped GEMM

The inverse RoPE is the conjugate rotation that undoes the RoPE applied
during attention. DeepSeek V4 uses GPT-J style (interleaved) RoPE.

For the RoPE portion of each head (last rope_dim=64 dims):
  - Pair elements (x[2i], x[2i+1]) — interleaved (GPT-J style)
  - Inverse (conjugate rotation):
    x[2i]   =  x'[2i] * cos(θ_i) + x'[2i+1] * sin(θ_i)
    x[2i+1] = -x'[2i] * sin(θ_i) + x'[2i+1] * cos(θ_i)
"""

import torch


def forward_rope_partial(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int = 64,
    head_dim: int = 512,
) -> torch.Tensor:
    """Apply partial RoPE to the last rope_dim dimensions of each head.

    DSV4 uses GPT-J style (interleaved) RoPE on the last rope_dim=64 dims.
    The first nope_dim=448 dims are left unchanged.

    For the RoPE portion (last 64 dims of each head):
      - Pair elements (x[2i], x[2i+1]) — interleaved
      - Forward rotation:
        x'[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
        x'[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)

    Args:
        x: (T, n_h * head_dim) BF16 — flat across heads
        positions: (T,) int64 token positions
        rope_dim: number of RoPE dims per head
        head_dim: total head dimension

    Returns:
        (T, n_h * head_dim) BF16 with forward RoPE applied to last rope_dim dims
    """
    T = x.shape[0]
    n_h = x.shape[1] // head_dim
    nope_dim = head_dim - rope_dim
    half_rope = rope_dim // 2

    # Build cos/sin cache (simple theta = 1/10000^(2i/d))
    # This should match the model's cos_sin_cache, but for now compute inline
    freqs = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=x.device) / rope_dim))
    pos_float = positions.float()  # (T,)
    angles = torch.outer(pos_float, freqs)  # (T, half_rope)
    cos_vals = torch.cos(angles).unsqueeze(1)  # (T, 1, half_rope) FP32
    sin_vals = torch.sin(angles).unsqueeze(1)  # FP32 for accuracy

    # Reshape x to (T, n_h, head_dim)
    x_heads = x.reshape(T, n_h, head_dim)

    # Extract RoPE portion — compute in FP32
    x_rope = x_heads[:, :, nope_dim:].float()  # (T, n_h, rope_dim) FP32
    x_even = x_rope[:, :, 0::2]  # (T, n_h, half_rope)
    x_odd = x_rope[:, :, 1::2]   # (T, n_h, half_rope)

    # Forward rotation
    rot_even = x_even * cos_vals - x_odd * sin_vals
    rot_odd = x_even * sin_vals + x_odd * cos_vals

    # Interleave back
    x_rot = torch.empty_like(x_rope)
    x_rot[:, :, 0::2] = rot_even
    x_rot[:, :, 1::2] = rot_odd

    # Copy NoPE portion unchanged
    result = x_heads.clone()
    result[:, :, nope_dim:] = x_rot.to(torch.bfloat16)
    return result.reshape(T, n_h * head_dim)


def inverse_rope_bf16(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    nope_dim: int = 448,
    rope_dim: int = 64,
) -> torch.Tensor:
    """Apply inverse RoPE to attention output in BF16.

    This is a pure-Python replacement for vLLM's
    fused_inv_rope_fp8_quant CUDA kernel. It only does the inverse
    RoPE (no FP8 quantization) since we quantize to NVFP4 instead.

    Args:
        o: (num_tokens, n_local_heads, head_dim) BF16 attention output
        positions: (num_tokens,) int64 token positions
        cos_sin_cache: (max_pos, rope_dim) float32 — cos||sin concatenated
        nope_dim: number of non-RoPE dims per head (448)
        rope_dim: number of RoPE dims per head (64)

    Returns:
        (num_tokens, n_local_heads, head_dim) BF16 with inverse RoPE applied
    """
    num_tokens, num_heads, head_dim = o.shape
    half_rope = rope_dim // 2

    # Get cos/sin for each position: (num_tokens, half_rope)
    cos_all = cos_sin_cache[positions, :half_rope]  # (T, 32)
    sin_all = cos_sin_cache[positions, half_rope:]   # (T, 32)

    # Expand for broadcasting: (T, 1, 32) → broadcasts over heads
    # CRITICAL: compute in FP32, not BF16! BF16 cos/sin destroys cos²+sin²=1
    cos_all = cos_all.unsqueeze(1).float()
    sin_all = sin_all.unsqueeze(1).float()

    # Extract RoPE portion: (T, H, rope_dim) — compute in FP32
    o_rope = o[:, :, nope_dim:].float()

    # Split into even/odd pairs (interleaved GPT-J style)
    o_even = o_rope[:, :, 0::2]  # (T, H, 32)
    o_odd = o_rope[:, :, 1::2]   # (T, H, 32)

    # Inverse rotation (conjugate):
    # inv[2i]   =  x[2i] * cos + x[2i+1] * sin
    # inv[2i+1] = -x[2i] * sin + x[2i+1] * cos
    inv_even = o_even * cos_all + o_odd * sin_all
    inv_odd = -o_even * sin_all + o_odd * cos_all

    # Interleave back
    o_inv = torch.empty_like(o_rope)
    o_inv[:, :, 0::2] = inv_even
    o_inv[:, :, 1::2] = inv_odd

    # Copy NoPE portion unchanged, replace RoPE portion
    result = o.clone()
    result[:, :, nope_dim:] = o_inv.to(torch.bfloat16)

    return result
