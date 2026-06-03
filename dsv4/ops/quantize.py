"""NVFP4 quantization: BF16 <-> NVFP4 conversion, scale factor computation."""
import math
import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from dsv4.ops.layouts import ceil_div

from dsv4.kernels.gemm.grouped import (
    cat_byte_reinterpretable_tensors,
    stack_byte_reinterpretable_tensors,
)

E2M1_MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# Cache compiled kernels + pre-allocated workspace by cache_key
# Each entry: {'compiled': callable, 'workspace': Tensor, 'workspace_size': int}
#
# Key design decisions (Bug #1 fix):
# - cute.compile does NOT corrupt GPU memory (verified 2026-05-20 on B200).
#   The original _needs_token_refill hack was a misdiagnosis. The real bug
#   was elsewhere (likely OOB write or weight loading).
# - Workspace is pre-allocated per cache entry during warmup_compilation()
#   and reused on subsequent calls. No torch.full() in the hot path.
# - CuTe tensor wrappers (from_dlpack + mark_layout_dynamic) are cheap
#   metadata wrappers. We re-create them per call from real tensors.
#   Caching them would hold stale references to tensors that get freed.

# Cached LUT for E2M1 quantization (created once per device, cudagraph-safe)
_NVFP4_STEP_LUT_CACHE = {}
def _get_step_to_idx_lut(device):
    """Get or create the E2M1 step-to-index LUT for the given device.
    
    Cached per device to avoid CPU->CUDA copies during cudagraph capture.
    Must be pre-populated during warmup (before torch.compile/cudagraph capture)
    so the lock is never entered on the compiled path.
    """
    # Fast path: already cached — no lock needed (torch.compile-safe)
    if device in _NVFP4_STEP_LUT_CACHE:
        return _NVFP4_STEP_LUT_CACHE[device]
    # Slow path: first call, create the LUT
    lut = torch.as_tensor(
        [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7],
        dtype=torch.int8, device=device,
    )
    _NVFP4_STEP_LUT_CACHE[device] = lut
    return lut
SF_VEC_SIZE = 16  # NVFP4 block size

def quantize_to_nvfp4(x_bf16, block_size=SF_VEC_SIZE):
    """Quantize BF16 tensor to NVFP4.
    
    Args:
        x_bf16: (..., D) BF16 tensor
    
    Returns:
        x_fp4: (..., D//2) float4_e2m1fn_x2 — native PyTorch FP4
        x_sf: (..., D//16) float8_e4m3fn — block scales
        global_scale: float32 scalar
    """
    x_f32 = x_bf16.float()
    amax = x_f32.abs().max().clamp(min=1e-8).float()
    global_scale = amax / (6.0 * 448.0)
    x_norm = x_f32 / global_scale

    last_dim = x_norm.shape[-1]
    n_blocks = ceil_div(last_dim, block_size)

    if last_dim % block_size != 0:
        pad_size = n_blocks * block_size - last_dim
        x_norm = torch.nn.functional.pad(x_norm, (0, pad_size))

    x_reshaped = x_norm.reshape(*x_norm.shape[:-1], n_blocks, block_size)
    block_amax = x_reshaped.abs().amax(dim=-1)
    # Detect zero blocks and underflow blocks (amax > 0 but too small for FP8).
    # Smallest positive FP8 e4m3fn is 2^-9 ≈ 1.95e-3. If amax/6 < this,
    # the block scale underflows to 0, and dividing x by the clamped 1e-8
    # inflates values into nonzero FP4 buckets — producing wrong results.
    zero_block = block_amax < (6.0 * 2.0 ** -9)  # < ~0.0117
    # Zero out x for zero/underflow blocks before division.
    # This ensures x_scaled = 0 → FP4 nibbles = 0.
    x_reshaped = torch.where(zero_block.unsqueeze(-1),
                              torch.zeros_like(x_reshaped), x_reshaped)
    block_amax = block_amax.clamp(min=1e-8)
    block_scale = (block_amax / 6.0).to(torch.float8_e4m3fn)
    # Force zero/underflow blocks: FP8 scale = 0 (exact zero).
    block_scale = torch.where(zero_block, torch.zeros_like(block_scale), block_scale)

    # Nearest E2M1
    block_sf_expanded = block_scale.float().unsqueeze(-1)
    x_scaled = x_reshaped / block_sf_expanded.clamp(min=1e-8)

    signs = torch.sign(x_scaled)
    abs_scaled = x_scaled.abs().clamp(max=6.0)
    
    half_steps = (abs_scaled * 2.0).round().clamp(0, 12).to(torch.int8)
    step_to_idx = _get_step_to_idx_lut(x_bf16.device)
    indices = step_to_idx[half_steps.long()]

    nibbles = torch.where(signs < 0, indices + 8, indices).to(torch.uint8)
    even = nibbles[..., ::2]
    odd = nibbles[..., 1::2]
    packed = (odd << 4) | even

    packed_shape = list(x_bf16.shape)
    packed_shape[-1] = last_dim // 2
    x_fp4 = packed.view(torch.float4_e2m1fn_x2).reshape(packed_shape)

    sf_shape = list(x_bf16.shape[:-1]) + [n_blocks]
    block_scale = block_scale.reshape(sf_shape)

    return x_fp4, block_scale, global_scale


def quantize_activation_nvfp4(x_bf16, global_scale, block_size=SF_VEC_SIZE):
    """Quantize BF16 activation tensor to NVFP4 (cudagraph-safe).

    Unlike quantize_to_nvfp4(), this takes a pre-computed global_scale
    instead of computing it via .max() (which forces CPU-GPU sync).
    All operations are pure GPU with no CPU-GPU syncs.

    Args:
        x_bf16: (..., D) BF16 tensor
        global_scale: float32 scalar (pre-computed, NOT from .max())
        block_size: NVFP4 block size
    
    Returns:
        x_fp4: (..., D//2) float4_e2m1fn_x2
        x_sf: (..., D//16) float8_e4m3fn
    """
    x_f32 = x_bf16.float()
    x_norm = x_f32 / global_scale

    last_dim = x_norm.shape[-1]
    n_blocks = ceil_div(last_dim, block_size)

    if last_dim % block_size != 0:
        pad_size = n_blocks * block_size - last_dim
        x_norm = torch.nn.functional.pad(x_norm, (0, pad_size))

    x_reshaped = x_norm.reshape(*x_norm.shape[:-1], n_blocks, block_size)
    block_amax = x_reshaped.abs().amax(dim=-1)
    # Detect zero blocks and underflow blocks (same threshold as quantize_to_nvfp4).
    zero_block = block_amax < (6.0 * 2.0 ** -9)
    x_reshaped = torch.where(zero_block.unsqueeze(-1),
                              torch.zeros_like(x_reshaped), x_reshaped)
    block_amax = block_amax.clamp(min=1e-8, max=6.0 * 448.0)  # E4M3 max = 448
    block_scale = (block_amax / 6.0).to(torch.float8_e4m3fn)
    block_scale = torch.where(zero_block, torch.zeros_like(block_scale), block_scale)

    block_sf_expanded = block_scale.float().unsqueeze(-1)
    x_scaled = x_reshaped / block_sf_expanded.clamp(min=1e-8)
    signs = torch.sign(x_scaled)
    abs_scaled = x_scaled.abs().clamp(max=6.0)
    
    half_steps = (abs_scaled * 2.0).round().clamp(0, 12).to(torch.int8)
    step_to_idx = _get_step_to_idx_lut(x_bf16.device)
    indices = step_to_idx[half_steps.long()]

    nibbles = torch.where(signs < 0, indices + 8, indices).to(torch.uint8)
    even = nibbles[..., ::2]
    odd = nibbles[..., 1::2]
    packed = (odd << 4) | even

    packed_shape = list(x_bf16.shape)
    packed_shape[-1] = last_dim // 2
    x_fp4 = packed.view(torch.float4_e2m1fn_x2).reshape(packed_shape)

    sf_shape = list(x_bf16.shape[:-1]) + [n_blocks]
    block_scale = block_scale.reshape(sf_shape)

    return x_fp4, block_scale


def quantize_weight_to_nvfp4(w_bf16, block_size=SF_VEC_SIZE):
    """Quantize BF16 weight matrix to NVFP4.
    
    The weight is (K, N) where K is the input dim (packed dimension).
    Block scales are computed along K (dim 0).
    
    Args:
        w_bf16: (K, N) BF16 weight matrix
    
    Returns:
        w_fp4: (K//2, N) float4_e2m1fn_x2 — K is the packed dim
        w_sf: (K//16, N) float8_e4m3fn — block scales along K
        global_scale: float32 scalar
    """
    K, N = w_bf16.shape
    w_f32 = w_bf16.float()
    amax = w_f32.abs().max().clamp(min=1e-8).float()
    global_scale = amax / (6.0 * 448.0)
    w_norm = w_f32 / global_scale

    k_blocks = ceil_div(K, block_size)
    if K % block_size != 0:
        w_norm = torch.nn.functional.pad(w_norm, (0, 0, 0, k_blocks * block_size - K))

    w_reshaped = w_norm.reshape(k_blocks, block_size, N)
    w_block_amax = w_reshaped.abs().amax(dim=1)
    # Detect zero blocks and underflow blocks (same threshold).
    zero_block = w_block_amax < (6.0 * 2.0 ** -9)
    w_reshaped = torch.where(zero_block.unsqueeze(1),
                              torch.zeros_like(w_reshaped), w_reshaped)
    w_block_amax = w_block_amax.clamp(min=1e-8)
    w_sf = (w_block_amax / 6.0).to(torch.float8_e4m3fn)
    w_sf = torch.where(zero_block, torch.zeros_like(w_sf), w_sf)

    w_block_sf = w_sf.float().unsqueeze(1)
    w_scaled = w_reshaped / w_block_sf.clamp(min=1e-8)

    signs = torch.sign(w_scaled)
    abs_scaled = w_scaled.abs().clamp(max=6.0)
    
    half_steps = (abs_scaled * 2.0).round().clamp(0, 12).to(torch.int8)
    step_to_idx = _get_step_to_idx_lut(w_bf16.device)
    indices = step_to_idx[half_steps.long()]
    nibbles = torch.where(signs < 0, indices + 8, indices).to(torch.uint8)

    even = nibbles[:, ::2, :]
    odd = nibbles[:, 1::2, :]
    packed = (odd << 4) | even

    w_fp4 = packed.reshape(K // 2, N).view(torch.float4_e2m1fn_x2)
    return w_fp4, w_sf, global_scale


# ── Scale Factor Assembly ─────────────────────────────────────────────
def deinterleave_quantize_nvfp4_cuda(fused_bf16, intermediate, global_scale, granularity=8):
    """De-interleave + quantize fused SwiGLU output using a custom CUDA kernel.
    
    Single kernel launch, no Python loop. 4x faster than the Python path.
    
    Args:
        fused_bf16: (M, 2*intermediate) BF16 — fused L1 output with interleaved gate/up
        intermediate: intermediate dimension (e.g., 3072)
        global_scale: pre-computed global scale for quantization
        granularity: interleave granularity in BF16 columns (default 8)
    
    Returns:
        x_fp4: (M, intermediate//2) float4_e2m1fn_x2 — quantized SwiGLU
        x_sf: (M, intermediate//16) float8_e4m3fn — block scales
    """
    from dsv4.kernels.cuda.loader import get_cuda_module
    mod = get_cuda_module("deinterleave_quantize_nvfp4", ["deinterleave_quantize.cu"])
    return mod.deinterleave_quantize_nvfp4(fused_bf16, intermediate, granularity, global_scale)


def deinterleave_amax_quantize_nvfp4_fused(fused_bf16, intermediate, divisor=6.0 * 448.0, granularity=8):
    """Fused deinterleave + amax + quantize: zero CPU syncs, two kernel launches.
    
    For the MoE fused_swiglu L2 path. Two-kernel approach (correct):
      Kernel 1: compute_amax_gsa on the de-interleaved values (GPU-only)
      Kernel 2: deinterleave_quantize_from_buffer using gsa from GPU buffer
    
    Args:
        fused_bf16: (M, 2*intermediate) BF16 — fused L1 output
        intermediate: intermediate dimension
        divisor: gsa = amax / divisor. Default 2688.0.
        granularity: interleave granularity (default 8)
    
    Returns:
        x_fp4: (M, intermediate//2) float4_e2m1fn_x2
        x_sf: (M, intermediate//16) float8_e4m3fn
        gsa: (M,) float32 GPU tensor — per-row global scale for L2 GEMM
    """
    from dsv4.kernels.cuda.loader import get_cuda_module
    # Compute gsa from the fused output
    amax_mod = get_cuda_module("amax_gsa", ["amax_gsa.cu"])
    gsa_gpu = amax_mod.compute_amax_gsa(fused_bf16, divisor)
    M = fused_bf16.shape[0]
    if gsa_gpu.dim() == 0:
        gsa_gpu = gsa_gpu.reshape(1).expand(M).contiguous()
    elif gsa_gpu.shape[0] == 1 and M > 1:
        gsa_gpu = gsa_gpu.expand(M).contiguous()
    # Deinterleave + quantize using gsa from GPU buffer
    quant_mod = get_cuda_module("fused_amax_quantize", ["fused_amax_quantize.cu"])
    x_fp4, x_sf = quant_mod.deinterleave_quantize_from_buffer(fused_bf16, intermediate, granularity, gsa_gpu)
    return x_fp4, x_sf, gsa_gpu


def compute_amax_gsa_gpu(x_bf16, divisor=6.0 * 448.0):
    """Compute gsa = max(|x|) / divisor on GPU. No CPU sync.
    
    Returns a scalar GPU tensor (not a Python float!).
    
    NOTE: Prefer quantize_nvfp4_gpu_fused() which does amax+quantize in
    one kernel launch. This function is kept for cases where you need gsa
    without quantization.
    """
    from dsv4.kernels.cuda.loader import get_cuda_module
    mod = get_cuda_module("amax_gsa", ["amax_gsa.cu"])
    return mod.compute_amax_gsa(x_bf16, divisor)


def quantize_nvfp4_gpu_fused(x_bf16, divisor=6.0 * 448.0):
    """Fused amax + gsa + quantize: zero CPU syncs, two kernel launches.
    
    Two-kernel approach (correct cross-CTA reduction):
      Kernel 1: compute_amax_gsa — row-wise amax → gsa on GPU (no .item())
      Kernel 2: quantize_nvfp4_from_buffer — quantize using gsa from GPU buffer
    
    The previous single-kernel approach had a race condition: the cross-CTA
    shared memory reduction used __syncthreads() which only syncs within a
    CTA, not across CTAs in the same grid. CTA 0 could read s_amax[b] before
    CTA b had written it, producing garbage gsa values.
    
    Args:
        x_bf16: (M, N) BF16 tensor. N must be a multiple of 16.
        divisor: gsa = amax / divisor. Default 6.0 * 448.0 = 2688.0.
    
    Returns:
        x_fp4: (M, N//2) float4_e2m1fn_x2
        x_sf: (M, N//16) float8_e4m3fn
        gsa: (M,) float32 GPU tensor — per-row global scale for GEMM
    """
    # CUDA kernels require contiguous input — column slices from deinterleave are non-contiguous
    if not x_bf16.is_contiguous():
        x_bf16 = x_bf16.contiguous()
    from dsv4.kernels.cuda.loader import get_cuda_module
    amax_mod = get_cuda_module("amax_gsa", ["amax_gsa.cu"])
    gsa_gpu = amax_mod.compute_amax_gsa(x_bf16, divisor)  # scalar GPU tensor
    # Broadcast to (M,) for the quantize-from-buffer kernel
    M = x_bf16.shape[0]
    if gsa_gpu.dim() == 0:
        gsa_gpu = gsa_gpu.reshape(1).expand(M).contiguous()  # (M,) all rows same gsa
    elif gsa_gpu.shape[0] == 1 and M > 1:
        gsa_gpu = gsa_gpu.expand(M).contiguous()
    quant_mod = get_cuda_module("fused_amax_quantize", ["fused_amax_quantize.cu"])
    x_fp4, x_sf = quant_mod.quantize_nvfp4_from_buffer(x_bf16, gsa_gpu)
    return x_fp4, x_sf, gsa_gpu


def quantize_nvfp4_gpu(x_bf16, global_scale):
    """Quantize BF16 tensor to NVFP4 using a custom CUDA kernel (GPU-only, no CPU sync).
    
    Replaces quantize_activation_nvfp4() which uses .amax() (CPU sync).
    The global_scale must be pre-computed (from warmup or known value).
    
    NOTE: Prefer quantize_nvfp4_gpu_fused() which also computes gsa on GPU.
    This function is kept for cases where global_scale is already known.
    
    Args:
        x_bf16: (M, N) BF16 tensor. N must be a multiple of 16.
        global_scale: float32 scalar (pre-computed, NOT from .max())
    
    Returns:
        x_fp4: (M, N//2) float4_e2m1fn_x2
        x_sf: (M, N//16) float8_e4m3fn
    """
    from dsv4.kernels.cuda.loader import get_cuda_module
    mod = get_cuda_module("quantize_nvfp4", ["quantize_nvfp4.cu"])
    return mod.quantize_nvfp4(x_bf16, global_scale)


class QuantizedActivation:
    """Pre-quantized NVFP4 activation tensor.
    
    Carries the FP4 data, block scales, and per-row global scale
    so downstream Nvfp4Linear calls can skip quantization and go
    straight to GEMM.
    
    Created by rmsnorm_quantize_nvfp4() or quantize_nvfp4_gpu_fused().
    Consumed by Nvfp4Linear.run_from_quantized().
    """
    __slots__ = ['x_fp4', 'x_sf', 'gsa', 'inv_rms', 'num_tokens']
    
    def __init__(self, x_fp4, x_sf, gsa, inv_rms=None):
        self.x_fp4 = x_fp4  # (M, N//2) FP4
        self.x_sf = x_sf    # (M, N//16) E4M3
        self.gsa = gsa      # (M,) FP32
        self.inv_rms = inv_rms  # (M,) FP32, optional
        self.num_tokens = x_fp4.shape[0]


def dequantize_nvfp4(x_fp4, x_sf, gsa, shape=None):
    """Dequantize NVFP4 → BF16 using the CUDA dequant kernel.
    
    Args:
        x_fp4: (M, N//2) FP4 packed
        x_sf: (M, N//16) E4M3 block scales
        gsa: (M,) or (M, 1) or (1,) FP32 global scale per row
        shape: unused, kept for API compat
    
    Returns:
        (M, N) BF16 tensor
    """
    from dsv4.kernels.cuda.loader import get_cuda_module
    mod = get_cuda_module("dequant_nvfp4", ["dequant_nvfp4.cu"])
    if gsa.dim() == 2:
        gsa = gsa.squeeze(1)  # (M, 1) → (M,)
    # dequant kernel expects uint8 for both fp4 and sf
    if x_fp4.dtype != torch.uint8:
        x_fp4 = x_fp4.view(torch.uint8)
    if x_sf.dtype != torch.uint8:
        x_sf = x_sf.view(torch.uint8)
    return mod.dequant_nvfp4(x_fp4, x_sf, gsa)


def mhc_rmsnorm_quantize_nvfp4(X_l, A_l, norm_weight, eps=1e-6, divisor=6.0 * 448.0):
    """Fused mHC pre_block + RMSNorm + NVFP4 quantize: 2 kernel launches total.
    
    Replaces: bmm (1 launch) + rmsnorm (4+ launches) + quantize (2 launches)
    Total unfused: 7+ launches per site × 122 sites = 854+ launches/token
    Fused: 2 launches per site × 122 sites = 244 launches → 610 launches saved/token.
    
    Args:
        X_l: (M, n_hc, N) BF16 tensor. n_hc must be <= 4, N multiple of 16.
        A_l: (M, n_hc) BF16 tensor. Softmax weights from mHC._dynamic_params.
        norm_weight: (N,) FP32 RMSNorm weight.
        eps: RMSNorm epsilon (default 1e-6).
        divisor: gsa = amax / divisor. Default 6.0 * 448.0 = 2688.0.
    
    Returns:
        QuantizedActivation with x_fp4, x_sf, gsa, inv_rms
    """
    from dsv4.kernels.cuda.loader import get_cuda_module
    mod = get_cuda_module("fused_mhc_rmsnorm_quantize", ["fused_mhc_rmsnorm_quantize.cu"])
    x_fp4, x_sf, gsa, inv_rms = mod.mhc_rmsnorm_quantize_nvfp4(X_l, A_l, norm_weight, eps, divisor)
    return QuantizedActivation(x_fp4, x_sf, gsa, inv_rms)


def rmsnorm_quantize_nvfp4(x_bf16, norm_weight, eps=1e-6, divisor=6.0 * 448.0):
    """Fused RMSNorm + amax + NVFP4 quantize: 2 kernel launches total.
    
    Replaces the unfused path:
      rmsnorm(x, weight) → 4+ BF16 launches
      quantize_nvfp4_gpu_fused(rmsnormed) → 2 kernel launches + amax
    Total unfused: 6+ launches per call × 122 calls/layer-step = 732+ launches/token
    
    Fused: 2 kernel launches per call × 122 calls = 244 launches → 488 launches saved/token.
    
    Two-kernel approach (correct cross-CTA reduction):
      Kernel 1: compute RMS + amax of normalized output → gsa per row (GPU buffer)
      Kernel 2: normalize + quantize using gsa from GPU buffer (no CPU sync)
    
    Args:
        x_bf16: (M, N) BF16 tensor. N must be a multiple of 16.
        norm_weight: (N,) FP32 RMSNorm weight.
        eps: RMSNorm epsilon (default 1e-6).
        divisor: gsa = amax / divisor. Default 6.0 * 448.0 = 2688.0.
    
    Returns:
        x_fp4: (M, N//2) FP4 packed (uint8 view of float4_e2m1fn_x2)
        x_sf: (M, N//16) E4M3 block scales
        gsa: (M,) FP32 per-row global scale for GEMM
        inv_rms: (M,) FP32 per-row 1/RMS (useful for downstream if needed)
    """
    from dsv4.kernels.cuda.loader import get_cuda_module
    mod = get_cuda_module("fused_rmsnorm_quantize", ["fused_rmsnorm_quantize.cu"])
    x_fp4, x_sf, gsa, inv_rms = mod.rmsnorm_quantize_nvfp4(x_bf16, norm_weight, eps, divisor)
    return QuantizedActivation(x_fp4, x_sf, gsa, inv_rms)
