"""Tensor layout helpers: scale swizzle, gate/up interleave, K-major, offsets."""
import torch

from dsv4.kernels.gemm.grouped import (
    pad_and_swizzle_single,
    assemble_raw_scales_2d3d_2d_side,
    assemble_raw_scales_2d3d_3d_side,
)

def ceil_div(a, b):
    return (a + b - 1) // b


def round_up(a, b):
    return ceil_div(a, b) * b

def interleave_l1_weights(w_ekn, granularity_bf16=8):
    """Interleave gate/up weights at granularity 8 in BF16 (4 in FP4).
    
    The fused SwiGLU epilogue requires gate/up pairs to be adjacent in the
    MMA accumulator. With interleaved weights, the MMA tile produces
    gate[i*8..i*8+7] and up[i*8..i*8+7] next to each other in registers,
    enabling a single-register SwiGLU without SMEM round-trips.
    
    Before:  [gate_0..gate_N/2-1 | up_0..up_N/2-1]
    After:   [gate_0..gate_7, up_0..up_7, gate_8..gate_15, up_8..up_15, ...]
    
    The interleave operates along the N dimension, where each column = 1 BF16
    (FP4 packing is along K, not N). So g = granularity_bf16 directly.
    
    Args:
        w_ekn: (E, K_packed, N_packed) FP4 weight tensor in K-major layout
                N_packed = 2*intermediate/2 = intermediate (gate+up fused)
        granularity_bf16: interleave group size in BF16 elements (default 8)
    
    Returns:
        (E, K_packed, N_packed) FP4 weight tensor with interleaved gate/up
    """
    E, K, N = w_ekn.shape
    N_half = N // 2  # gate and up each have N/2 FP4 columns
    g = granularity_bf16  # N-axis interleave: each N-col = 1 BF16 col (packing is along K)
    
    gate = w_ekn[:, :, :N_half].reshape(E, K, N_half // g, g)
    up = w_ekn[:, :, N_half:].reshape(E, K, N_half // g, g)
    return torch.stack([gate, up], dim=3).reshape(E, K, N)


def deinterleave_l1_weights(w_ekn, granularity_bf16=8):
    """De-interleave gate/up weights (inverse of interleave_l1_weights).
    
    Used for testing/verification only.
    """
    g = granularity_bf16  # N-axis: each N-col = 1 BF16 col
    E, K, N = w_ekn.shape
    w_reshaped = w_ekn.reshape(E, K, N // (2 * g), 2, g)
    gate = w_reshaped[:, :, :, 0, :].reshape(E, K, N // 2)
    up = w_reshaped[:, :, :, 1, :].reshape(E, K, N // 2)
    return torch.cat([gate, up], dim=2)


def assemble_scales_2d_side(raw_scales):
    """Assemble activation scale factors for the 2Dx3D scenario.
    
    Args:
        raw_scales: list of (M_e, K_sf) float8_e4m3fn tensors, one per expert
    
    Returns:
        Assembled and swizzled scale tensor
    """
    return assemble_raw_scales_2d3d_2d_side(raw_scales)


def assemble_scales_3d_side(raw_scales):
    """Assemble weight scale factors for the 2Dx3D scenario.
    
    Args:
        raw_scales: list of (K_sf, N) float8_e4m3fn tensors, one per expert
        NOTE: These will be transposed to (N, K_sf) before swizzling,
        since the kernel expects N as the non-K dimension.
    
    Returns:
        Assembled and swizzled scale tensor
    """
    # Kernel expects (N, K_sf) — transpose before swizzling
    transposed = [sf.T.contiguous() for sf in raw_scales]
    return assemble_raw_scales_2d3d_3d_side(transposed)


# ── Tensor Layout Conversion ──────────────────────────────────────────

def make_b_k_major(b_tensor):
    """Convert B tensor from N-major to K-major layout.
    
    The kernel expects B with stride (E*K*N, 1, K) — K is contiguous.
    torch.stack produces stride (E*K*N, N, 1) — N is contiguous.
    
    Args:
        b_tensor: (experts, K_packed, N_packed) float4_e2m1fn_x2, N-major
    
    Returns:
        Same shape, K-major strides
    """
    return b_tensor.permute(0, 2, 1).contiguous().permute(0, 2, 1)


def compute_expert_offsets(tokens_per_expert, num_experts, device="cuda"):
    """Compute cumulative token offsets for the grouped GEMM.
    
    Args:
        tokens_per_expert: list of int, one per expert
    
    Returns:
        offs: (num_experts,) int32 — cumulative sum
    """
    offs = torch.tensor(
        [sum(tokens_per_expert[:e+1]) for e in range(num_experts)],
        dtype=torch.int32, device=device,
    )
    return offs


# ── Kernel Launch ─────────────────────────────────────────────────────

