"""DSV4 FMHA — Multi-tile TMA kernel loader.

Loads the TMA-based multi-tile FMHA kernel via nvcc precompile + ctypes.
Supports any s_k (multiple KV tiles), T=1 decode + T>1 prefill.
HD in {64, 128, 256, 512}.
"""

import torch
import logging
import os
import subprocess
import ctypes
from typing import Optional

logger = logging.getLogger(__name__)

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(KERNEL_DIR, "..", ".."))
SOURCE = os.path.join(KERNEL_DIR, "fmha_multitile_capi.cu")
BUILD_DIR = os.path.join(REPO_ROOT, "build", "fmha_multitile")
SO_NAME = "libfmha_multitile.so"

_lib = None
_lib_lock = False


def _find_nvcc():
    import shutil
    for c in ["/usr/local/cuda-13.2/bin/nvcc", "/usr/local/cuda/bin/nvcc"]:
        if os.path.isfile(c): return c
    nvcc = shutil.which("nvcc")
    if nvcc: return nvcc
    raise RuntimeError("nvcc not found")


def _ensure_built():
    global _lib
    if _lib is not None: return _lib
    global _lib_lock
    if _lib_lock: raise RuntimeError("Recursive build")
    _lib_lock = True
    try:
        so_path = os.path.join(BUILD_DIR, SO_NAME)
        need_build = True
        if os.path.isfile(so_path):
            src_mtime = os.path.getmtime(SOURCE)
            for dep in ["fmha_common.cuh", "fmha_umma_desc.cuh", "fmha_tma.cuh",
                         "fmha_6warp_tma_multirow_multitile.cuh"]:
                dp = os.path.join(KERNEL_DIR, dep)
                if os.path.isfile(dp): src_mtime = max(src_mtime, os.path.getmtime(dp))
            need_build = src_mtime > os.path.getmtime(so_path)
        if not need_build:
            _lib = ctypes.CDLL(so_path)
            return _lib

        logger.info("Building libfmha_multitile.so (sm_100a)...")
        os.makedirs(BUILD_DIR, exist_ok=True)
        nvcc = _find_nvcc()
        cmd = [nvcc, "-std=c++20", "-shared", "-Xcompiler", "-fPIC",
               "-gencode=arch=compute_100a,code=sm_100a",
               "-gencode=arch=compute_100a,code=compute_100a",
               f"-I{KERNEL_DIR}", f"-I{REPO_ROOT}",
               "-O3", "--expt-relaxed-constexpr",
               SOURCE, "-o", so_path, "-lcudart", "-lcuda"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{result.stderr}")
        _lib = ctypes.CDLL(so_path)
        logger.info(f"Built {so_path}")
        return _lib
    finally:
        _lib_lock = False


def fmha_multitile_decode_raw(
    q: torch.Tensor,          # (batch, n_h, T, hd) BF16
    k: torch.Tensor,          # (batch, n_kv, N, hd) BF16
    v: torch.Tensor,          # (batch, n_kv, hd, N) BF16
    scale: float,
    n_comp: int = 0,
    swa_len: int = 0,
    is_causal: bool = False,
    attn_sink: Optional[torch.Tensor] = None,
    skip_gqa_expand: bool = False,  # Skip K/V repeat_interleave for MQA
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the multi-tile TMA FMHA kernel. Returns (O, LSE)."""
    lib = _ensure_built()

    B = q.shape[0]
    n_h = q.shape[1]
    T = q.shape[2]
    hd = q.shape[3]
    n_kv = k.shape[1]
    N = k.shape[2]

    assert hd in (64, 128, 256, 512), f"Unsupported hd={hd}"
    q_per_kv = n_h // n_kv

    # GQA: expand K/V to n_h heads
    # MQA fast path: skip the expensive repeat_interleave (128× memory copy).
    # Instead, pass stride=0 for the head dimension so all Q heads read the same KV.
    # This saves ~1.15MB allocation + copy per layer per decode step.
    if n_kv < n_h:
        if skip_gqa_expand:
            # Don't expand K/V — pass stride(1)=0 to kernel for MQA
            pass
        else:
            k = k.repeat_interleave(q_per_kv, dim=1)
            v = v.repeat_interleave(q_per_kv, dim=1)

    # Pad N to multiple of 128 (TMA descriptor alignment)
    N_orig = N
    N_padded = ((N + 127) // 128) * 128
    if N < N_padded:
        pad = N_padded - N
        k = torch.cat([k, torch.zeros(B, k.shape[1], pad, hd, dtype=torch.bfloat16, device=k.device)], dim=2)
        v = torch.cat([v, torch.zeros(v.shape[0], v.shape[1], hd, pad, dtype=torch.bfloat16, device=v.device)], dim=3)
        N = N_padded  # N is now the physical size (padded)

    k = k.contiguous()
    v = v.contiguous()
    q = q.contiguous()

    o = torch.zeros(B, n_h, T, hd, dtype=torch.bfloat16, device=q.device)
    lse = torch.zeros(B, n_h, T, dtype=torch.float32, device=q.device)

    # Sink bias: must be contiguous FP32 (n_h,) per batch
    sink_bias_ptr = ctypes.c_void_p(0)
    if attn_sink is not None:
        sb = attn_sink.float().contiguous()
        if sb.dim() == 1:
            sb = sb.unsqueeze(0).expand(B, -1).contiguous()  # (batch, n_h)
        assert sb.shape == (B, n_h), f"sink_bias shape {sb.shape} != ({B}, {n_h})"
        sink_bias_ptr = ctypes.c_void_p(sb.data_ptr())

    # For MQA skip_gqa_expand: pass stride(1)=0 for K and V so all heads
    # read from the same KV head (head 0). The kernel's CTA for head h
    # computes k_ptr + h * k_stride1, so stride1=0 means all heads share
    # the same K/V data without the 128× memory expansion.
    k_stride1 = 0 if (n_kv < n_h and skip_gqa_expand) else k.stride(1)
    v_stride1 = 0 if (n_kv < n_h and skip_gqa_expand) else v.stride(1)

    ret = lib.fmha_multitile_decode_launch(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(o.data_ptr()),
        ctypes.c_void_p(lse.data_ptr()),
        sink_bias_ptr,  # per-head FP32 sink logit
        ctypes.c_int(B), ctypes.c_int(n_h), ctypes.c_int(T),
        ctypes.c_int(N_orig),   # s_k: logical KV length (for softmax masking)
        ctypes.c_int(N_padded), # N_padded: physical KV length (for TMA descriptors)
        ctypes.c_int(hd),
        ctypes.c_int(q.stride(1)), ctypes.c_int(q.stride(0)),
        ctypes.c_int(k_stride1), ctypes.c_int(k.stride(0)),
        ctypes.c_int(v_stride1), ctypes.c_int(v.stride(0)),
        ctypes.c_int(o.stride(1)), ctypes.c_int(o.stride(0)),
        ctypes.c_int(lse.stride(1)), ctypes.c_int(lse.stride(0)),
        ctypes.c_float(scale),
    )
    if ret != 0:
        raise RuntimeError(f"Multi-tile kernel launch failed: return code {ret}")
    return o, lse
