"""Production compressor: NVFP4 GEMM projections + CUDA softmax/reduce kernel.

Pipeline:
  1. NVFP4 GEMM: hidden_states @ kv_proj → kv (T, kv_dim)
  2. NVFP4 GEMM: hidden_states @ gate_proj → gate (T, kv_dim)
  3. CUDA kernel: token-level softmax(gate) * kv → compressed entries
  4. CUDA kernel: kv_norm (unweighted RMSNorm + weight)

KV-1/KV-2: NVFP4 output variants compress + quantize in a single kernel.
No intermediate BF16. Stored as FP4 data + E4M3 block scales + FP32 global scale.

No PyTorch softmax. No reference fallback. All on the GPU.
"""

from __future__ import annotations

import os
import torch
from typing import Optional

_kernel_module = None


def _get_kernel():
    global _kernel_module
    if _kernel_module is not None:
        return _kernel_module
    from torch.utils.cpp_extension import load
    kernel_dir = os.path.join(os.path.dirname(__file__), "..", "cuda")
    _kernel_module = load(
        name="compressor_reduce",
        sources=[os.path.join(kernel_dir, "compressor_reduce.cu")],
        extra_cuda_cflags=["-O3", "--generate-code=arch=compute_100a,code=[sm_100a]"],
        verbose=False,
    )
    return _kernel_module


def csa_compress_production(
    kv_proj_out: torch.Tensor,      # (T, 2*hd) FP32 — output of NVFP4 GEMM
    gate_proj_out: torch.Tensor,    # (T, 2*hd) FP32 — output of NVFP4 GEMM
    position_bias: Optional[torch.Tensor],  # (m, 2*hd) BF16 or None
    kv_norm_weight: Optional[torch.Tensor], # (hd) BF16 or None
    m: int = 4,
) -> torch.Tensor:
    """CSA compress: softmax + weighted sum + kv_norm. Returns BF16."""
    return csa_compress_production_fp32(
        kv_proj_out, gate_proj_out, position_bias, kv_norm_weight, m
    ).bfloat16()


def csa_compress_production_fp32(
    kv_proj_out: torch.Tensor,
    gate_proj_out: torch.Tensor,
    position_bias: Optional[torch.Tensor],
    kv_norm_weight: Optional[torch.Tensor],
    m: int = 4,
) -> torch.Tensor:
    """CSA compress: softmax + weighted sum + kv_norm. Returns FP32."""
    T = kv_proj_out.shape[0]
    hd = kv_proj_out.shape[1] // 2
    n_blocks = T // m
    if n_blocks == 0:
        return torch.zeros(0, hd, dtype=torch.float32, device=kv_proj_out.device)

    mod = _get_kernel()

    pos_bias_f32 = torch.empty(0, dtype=torch.float32, device=kv_proj_out.device)
    if position_bias is not None:
        pos_bias_f32 = position_bias.float()

    norm_f32 = torch.empty(0, dtype=torch.float32, device=kv_proj_out.device)
    if kv_norm_weight is not None:
        norm_f32 = kv_norm_weight.float()

    compressed = torch.zeros(n_blocks, hd, dtype=torch.float32, device=kv_proj_out.device)

    mod.csa_compress_reduce(
        kv_proj_out.contiguous(),
        gate_proj_out.contiguous(),
        pos_bias_f32.contiguous(),
        norm_f32.contiguous(),
        compressed,
        m, n_blocks,
    )

    return compressed


def hca_compress_production(
    kv_proj_out: torch.Tensor,      # (T, hd) FP32
    gate_proj_out: torch.Tensor,    # (T, hd) FP32
    position_bias: Optional[torch.Tensor],  # (m, hd) BF16 or None
    kv_norm_weight: Optional[torch.Tensor], # (hd) BF16 or None
    m: int = 128,
) -> torch.Tensor:
    """HCA compress: softmax + weighted sum + kv_norm. Returns BF16."""
    return hca_compress_production_fp32(
        kv_proj_out, gate_proj_out, position_bias, kv_norm_weight, m
    ).bfloat16()


def hca_compress_production_fp32(
    kv_proj_out: torch.Tensor,
    gate_proj_out: torch.Tensor,
    position_bias: Optional[torch.Tensor],
    kv_norm_weight: Optional[torch.Tensor],
    m: int = 128,
) -> torch.Tensor:
    """HCA compress: softmax + weighted sum + kv_norm. Returns FP32."""
    T = kv_proj_out.shape[0]
    hd = kv_proj_out.shape[1]
    n_blocks = T // m
    if n_blocks == 0:
        return torch.zeros(0, hd, dtype=torch.float32, device=kv_proj_out.device)

    mod = _get_kernel()

    pos_bias_f32 = torch.empty(0, dtype=torch.float32, device=kv_proj_out.device)
    if position_bias is not None:
        pos_bias_f32 = position_bias.float()

    norm_f32 = torch.empty(0, dtype=torch.float32, device=kv_proj_out.device)
    if kv_norm_weight is not None:
        norm_f32 = kv_norm_weight.float()

    compressed = torch.zeros(n_blocks, hd, dtype=torch.float32, device=kv_proj_out.device)

    mod.hca_compress_reduce(
        kv_proj_out.contiguous(),
        gate_proj_out.contiguous(),
        pos_bias_f32.contiguous(),
        norm_f32.contiguous(),
        compressed,
        m, n_blocks,
    )

    return compressed


# ===========================================================================
# KV-1/KV-2: NVFP4 output — two proven kernels, no BF16 intermediate
#
# Architecture:
#   1. CUDA compress kernel (compressor_reduce.cu) → FP32 compressed output
#   2. CUDA amax_gsa_fp32 → per-row gsa (GPU-only, no CPU sync)
#   3. CUDA quantize_nvfp4_from_fp32 → NVFP4 triple (fp4 + sf + gsa)
#
# This is the same two-kernel pattern that works everywhere else in the
# pipeline (quantize_nvfp4_gpu_fused). The previous single-kernel fused
# approach had shared memory corruption bugs. Two kernels is correct.
#
# Storage: NVFP4 (E2M1 data + E4M3 block scales + FP32 global scale)
# Read path: dequant_nvfp4 / dequant_nvfp4_selective → BF16 for FMHA
# ===========================================================================

def _quantize_fp32_to_nvfp4(compressed_fp32: torch.Tensor) -> tuple:
    """Quantize FP32 compressed output → NVFP4. Two-kernel, GPU-only.

    Uses the same proven pattern as quantize_nvfp4_gpu_fused (amax_gsa +
    quantize_from_buffer) but with FP32 input instead of BF16.
    No BF16 intermediate. No CPU sync.

    Returns: (fp4_data, block_scales, global_scales) — NVFP4 triple.
    """
    from dsv4.kernels.cuda.loader import get_cuda_module
    mod = get_cuda_module("kv_quantize", ["kv_quantize.cu"])
    # Kernel 1: Compute per-row gsa from FP32 input (GPU-only)
    gsa = mod.compute_amax_gsa_fp32(compressed_fp32.contiguous(), 6.0 * 448.0)
    # Kernel 2: Quantize FP32 → NVFP4 using GPU gsa buffer
    fp4, sf = mod.quantize_nvfp4_from_fp32(compressed_fp32.contiguous(), gsa)
    return fp4, sf, gsa


def csa_compress_production_nvfp4(
    kv_proj_out: torch.Tensor,
    gate_proj_out: torch.Tensor,
    position_bias: Optional[torch.Tensor],
    kv_norm_weight: Optional[torch.Tensor],
    m: int = 4,
) -> tuple:
    """CSA compress → NVFP4. No BF16 intermediate.

    KV-1: Production path. Compressed KV stored as NVFP4.
    Pipeline: compress (FP32) → amax_gsa (GPU) → quantize (GPU) → NVFP4 triple.
    Returns: (fp4_data, block_scales, global_scales) — NVFP4 triple.
    """
    # Step 1: Compress → FP32 (same proven kernel as BF16 path)
    compressed_fp32 = csa_compress_production_fp32(
        kv_proj_out, gate_proj_out, position_bias, kv_norm_weight, m)
    if compressed_fp32.shape[0] == 0:
        dev = kv_proj_out.device
        hd = kv_proj_out.shape[1] // 2
        return (torch.zeros(0, hd // 2, dtype=torch.float4_e2m1fn_x2, device=dev),
                torch.zeros(0, hd // 16, dtype=torch.float8_e4m3fn, device=dev),
                torch.zeros(0, dtype=torch.float32, device=dev))
    # Step 2-3: FP32 → NVFP4 (two proven kernels)
    return _quantize_fp32_to_nvfp4(compressed_fp32)


def hca_compress_production_nvfp4(
    kv_proj_out: torch.Tensor,
    gate_proj_out: torch.Tensor,
    position_bias: Optional[torch.Tensor],
    kv_norm_weight: Optional[torch.Tensor],
    m: int = 128,
) -> tuple:
    """HCA compress → NVFP4. No BF16 intermediate.

    KV-2: Production path. Compressed KV stored as NVFP4.
    Pipeline: compress (FP32) → amax_gsa (GPU) → quantize (GPU) → NVFP4 triple.
    Returns: (fp4_data, block_scales, global_scales) — NVFP4 triple.
    """
    # Step 1: Compress → FP32
    compressed_fp32 = hca_compress_production_fp32(
        kv_proj_out, gate_proj_out, position_bias, kv_norm_weight, m)
    if compressed_fp32.shape[0] == 0:
        dev = kv_proj_out.device
        hd = kv_proj_out.shape[1]
        return (torch.zeros(0, hd // 2, dtype=torch.float4_e2m1fn_x2, device=dev),
                torch.zeros(0, hd // 16, dtype=torch.float8_e4m3fn, device=dev),
                torch.zeros(0, dtype=torch.float32, device=dev))
    # Step 2-3: FP32 → NVFP4
    return _quantize_fp32_to_nvfp4(compressed_fp32)
