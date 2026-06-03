"""DSV4 Blackwell Attention — Production kernel wrapper.

All attention goes through the 6-warp multi-tile FMHA kernel
(fmha_6warp_tma_multirow_multitile.cuh) via the C-API + ctypes bridge.
No CuTeDSL runtime dependency. No Python KV merge. No cudaDeviceSynchronize.

See README.md and ROADMAP.md for architecture, constraints, and next steps.
"""
import torch
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: 6-warp multi-head multi-tile decode kernel
# ---------------------------------------------------------------------------

def _dsv4_attention_multitile(
    q: torch.Tensor,            # (n_q_heads, T, hd) BF16
    k: torch.Tensor,            # (n_kv_heads, N, hd) or (N, hd) BF16
    v: torch.Tensor,            # same shape as k
    scale: float,
    n_comp: int = 0,
    sink_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-tile decode via TMA-based 6-warp FMHA kernel."""
    from dsv4.kernels.attention.fmha_multitile_op import fmha_multitile_decode_raw

    n_q, T, hd = q.shape
    n_kv = k.shape[0] if k.dim() == 3 else 1
    N = k.shape[-2] if k.dim() == 3 else k.shape[0]

    q_4d = q.unsqueeze(0).contiguous()
    if k.dim() == 2:
        k_4d = k.unsqueeze(0).unsqueeze(0).contiguous()
        v_4d = v.unsqueeze(0).unsqueeze(0).transpose(-1, -2).contiguous()
    else:
        k_4d = k.unsqueeze(0).contiguous()
        v_4d = v.unsqueeze(0).transpose(-1, -2).contiguous()

    o_4d, _lse = fmha_multitile_decode_raw(q_4d, k_4d, v_4d, scale, attn_sink=sink_bias,
                                           skip_gqa_expand=True)
    return o_4d.squeeze(0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dsv4_attention(
    q: torch.Tensor,            # (batch, n_q_heads, T, hd) or (n_q_heads, T, hd)
    k: torch.Tensor,            # (batch, n_kv_heads, N, hd) or (n_kv_heads, N, hd) or (N, hd)
    v: torch.Tensor,            # same shape as k
    scale: Optional[float] = None,
    swa_len: Optional[int] = None,
    is_causal: bool = False,
    n_comp: int = 0,
    sink_bias: Optional[torch.Tensor] = None,  # (n_q_heads,) or (batch, n_q_heads)
) -> torch.Tensor:
    """Production DSV4 attention: MHA / MQA / GQA via 6-warp multi-tile kernel.

    All head dims (64/128/256/512) and all sequence lengths handled by the
    in-kernel multi-tile path. No Python KV merge, no CuTeDSL runtime.

    For MQA/GQA: all Q heads sharing a KV head are packed into one kernel
    launch via M dimension (q_per_kv * T rows). Each KV head is loaded once
    per CTA, not per Q head.

    Args:
        q: (n_q_heads, T, hd) or (batch, n_q_heads, T, hd) BF16
        k: (n_kv_heads, N, hd) or (N, hd) for MQA, or with batch dim BF16
        v: same shape as k
        scale: 1/sqrt(hd) if None
        swa_len: sliding window length (currently unused by kernel — masks applied upstream)
        is_causal: causal mask (currently unused by kernel — masks applied upstream)
        n_comp: compressed KV length for D5c sink bias (reserved for future kernel integration)
        sink_bias: per-head FP32 logit bias (reserved for future kernel integration)

    Returns:
        Same shape as q input (without batch: (n_q_heads, T, hd) BF16)
    """
    # Handle batch dimension
    has_batch = q.dim() == 4
    if has_batch:
        # E5: Batch is handled natively by the kernel grid (blockIdx.z).
        # The C API launch sets dim3 grid(1, n_h, batch) which processes
        # all batch items in a single kernel launch.
        batch_size = q.shape[0]
        n_q, T, hd = q.shape[1], q.shape[2], q.shape[3]
        scale = scale or (1.0 / math.sqrt(hd))

        # Normalize K/V to (batch, n_kv, N, hd)
        if k.dim() == 2:
            k = k.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).contiguous()
            v = v.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).contiguous()
        elif k.dim() == 3:
            k = k.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
            v = v.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()

        n_kv = k.shape[1]
        q_per_kv = n_q // n_kv
        assert n_q % n_kv == 0

        if T == 1 and hd in (64, 128, 256, 512):
            # Direct 4D dispatch — single kernel launch for all batch items
            # GQA: expand K/V to n_h heads (handled inside fmha_multitile_decode_raw)
            from dsv4.kernels.attention.fmha_multitile_op import fmha_multitile_decode_raw
            o_4d, _lse = fmha_multitile_decode_raw(q, k, v, scale, n_comp)
            return o_4d

        # T>1 fallback: still need per-batch dispatch for prefill
        output = torch.zeros(batch_size, n_q, T, hd, dtype=torch.bfloat16, device='cuda')
        for b in range(batch_size):
            out_b = dsv4_attention(q[b], k[b], v[b], scale=scale, swa_len=swa_len,
                                   is_causal=is_causal, n_comp=n_comp, sink_bias=sink_bias)
            output[b] = out_b
        return output

    # 3D case: (n_q_heads, T, hd)
    n_q, T, hd = q.shape
    scale = scale or (1.0 / math.sqrt(hd))

    # Normalize K/V to (n_kv, N, hd)
    if k.dim() == 2:
        k = k.unsqueeze(0)  # (1, N, hd) — MQA
    if v.dim() == 2:
        v = v.unsqueeze(0)
    n_kv, N, _ = k.shape

    # GQA ratio: each KV head serves (n_q // n_kv) Q heads
    q_per_kv = n_q // n_kv
    assert n_q % n_kv == 0, f"n_q_heads ({n_q}) must be divisible by n_kv_heads ({n_kv})"

    # 6-warp multi-tile kernel handles all N, T=1 decode
    if T == 1 and hd in (64, 128, 256, 512):
        return _dsv4_attention_multitile(q, k, v, scale, n_comp, sink_bias)

    # Prefill / T>1: head-packed dispatch per KV group
    # TODO (E8): multi-CTA grid for T>1 prefill
    output = torch.zeros(n_q, T, hd, dtype=torch.bfloat16, device='cuda')
    for kv_idx in range(n_kv):
        q_start = kv_idx * q_per_kv
        q_end = q_start + q_per_kv
        q_group = q[q_start:q_end]  # (q_per_kv, T, hd)
        k_kv = k[kv_idx:kv_idx+1]   # (1, N, hd)
        v_kv = v[kv_idx:kv_idx+1]

        o_group = _dsv4_attention_multitile(q_group, k_kv, v_kv, scale, n_comp, sink_bias)
        output[q_start:q_end] = o_group

    return output


def dsv4_attention_per_head(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    swa_len: Optional[int] = None,
    is_causal: bool = False,
    n_comp: int = 0,
    sink_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-head launch variant — exact per-head sink bias support.

    Use this when Q heads within a KV group have different sink biases
    and exact results matter more than launch overhead. Otherwise prefer
    dsv4_attention (head-packed).
    """
    n_q, T, hd = q.shape
    scale = scale or (1.0 / math.sqrt(hd))

    if k.dim() == 2:
        k = k.unsqueeze(0)
    if v.dim() == 2:
        v = v.unsqueeze(0)
    n_kv, N, _ = k.shape
    q_per_kv = n_q // n_kv

    output = torch.zeros(n_q, T, hd, dtype=torch.bfloat16, device='cuda')

    for kv_idx in range(n_kv):
        k_kv = k[kv_idx:kv_idx+1]  # (1, N, hd)
        v_kv = v[kv_idx:kv_idx+1]

        for qi in range(q_per_kv):
            q_idx = kv_idx * q_per_kv + qi
            q_h = q[q_idx:q_idx+1]  # (1, T, hd)
            sb = sink_bias[q_idx:q_idx+1] if sink_bias is not None else None

            o = _dsv4_attention_multitile(q_h, k_kv, v_kv, scale, n_comp, sb)
            output[q_idx] = o

    return output


# ---------------------------------------------------------------------------
# B1: mixed FP8/BF16 DeepSeek-V4 decode attention
# ---------------------------------------------------------------------------

def dsv4_attention_mixed_fp8_decode(
    q: torch.Tensor,                 # (n_q_heads,T,HD) or (B,n_q_heads,T,HD) BF16
    k_nope_fp8: torch.Tensor,        # (N,NOPE) uint8/float8_e4m3fn
    k_nope_scale: torch.Tensor,      # (N,) FP32
    k_rope_bf16: torch.Tensor,       # (N,ROPE) BF16
    scale: Optional[float] = None,
    sink_bias: Optional[torch.Tensor] = None,
    rope_dim: int = 64,
) -> torch.Tensor:
    """B1 production path: storage-native FP8/BF16 KV decode FMHA.

    This intentionally has no PyTorch/BF16 fallback. It is the decode-only path
    for DeepSeek-V4 attention where noPE KV is already stored as FP8_E4M3 with
    per-row FP32 scales and RoPE KV is BF16.
    """
    from dsv4.kernels.attention.fmha_mixed_fp8_op import fmha_mixed_fp8_decode_raw

    has_batch = q.dim() == 4
    if q.dim() == 3:
        q4 = q.unsqueeze(0).contiguous()
    elif q.dim() == 4:
        q4 = q.contiguous()
    else:
        raise RuntimeError("q must be (H,T,HD) or (B,H,T,HD)")

    hd = q4.shape[-1]
    scale = scale or (1.0 / math.sqrt(hd))
    o4, _lse = fmha_mixed_fp8_decode_raw(
        q4, k_nope_fp8, k_nope_scale, k_rope_bf16,
        scale, attn_sink=sink_bias, rope_dim=rope_dim,
    )
    return o4 if has_batch else o4.squeeze(0)


# ---------------------------------------------------------------------------
# B1: mixed FP8/BF16 DeepSeek-V4 PREFILL attention (T > 1)
# ---------------------------------------------------------------------------

def dsv4_attention_mixed_fp8_prefill(
    q: torch.Tensor,                 # (n_q_heads,T,HD) or (B,n_q_heads,T,HD) BF16
    k_nope_fp8: torch.Tensor,        # (N,NOPE) uint8/float8_e4m3fn
    k_nope_scale: torch.Tensor,      # (N,) FP32
    k_rope_bf16: torch.Tensor,       # (N,ROPE) BF16
    scale: Optional[float] = None,
    sink_bias: Optional[torch.Tensor] = None,
    rope_dim: int = 64,
) -> torch.Tensor:
    """B1 production path: storage-native FP8/BF16 KV prefill FMHA.

    Supports T = 1..128. For T > 128, caller must split into multiple launches.
    Uses the same mixed FP8/BF16 KV format as the decode path.
    """
    from dsv4.kernels.attention.fmha_mixed_fp8_prefill_op import fmha_mixed_fp8_prefill_raw

    has_batch = q.dim() == 4
    if q.dim() == 3:
        q4 = q.unsqueeze(0).contiguous()  # (1, H, T, HD)
    elif q.dim() == 4:
        q4 = q.contiguous()
    else:
        raise RuntimeError("q must be (H,T,HD) or (B,H,T,HD)")

    hd = q4.shape[-1]
    scale = scale or (1.0 / math.sqrt(hd))
    o4, _lse = fmha_mixed_fp8_prefill_raw(
        q4, k_nope_fp8, k_nope_scale, k_rope_bf16,
        scale, attn_sink=sink_bias, rope_dim=rope_dim,
    )
    return o4 if has_batch else o4.squeeze(0)
