"""DSV4 B1 mixed FP8/BF16 decode FMHA loader.

This path is intentionally hard-error only: it does not fall back to PyTorch or to
BF16 FMHA if the mixed FP8 kernel is requested.
"""
import ctypes
import logging
import os
import subprocess
from typing import Optional

import torch

logger = logging.getLogger(__name__)

KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(KERNEL_DIR, "..", ".."))
SOURCE = os.path.join(KERNEL_DIR, "fmha_mixed_fp8_capi.cu")
BUILD_DIR = os.path.join(REPO_ROOT, "build", "fmha_mixed_fp8")
SO_NAME = "libfmha_mixed_fp8.so"

_lib = None
_lib_lock = False


def _find_nvcc():
    import shutil
    for c in ["/usr/local/cuda-13.2/bin/nvcc", "/usr/local/cuda/bin/nvcc"]:
        if os.path.isfile(c):
            return c
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    raise RuntimeError("nvcc not found")


def _ensure_built():
    global _lib, _lib_lock
    if _lib is not None:
        return _lib
    if _lib_lock:
        raise RuntimeError("Recursive mixed-FP8 FMHA build")
    _lib_lock = True
    try:
        so_path = os.path.join(BUILD_DIR, SO_NAME)
        deps = [
            SOURCE,
            os.path.join(KERNEL_DIR, "fmha_common.cuh"),
            os.path.join(KERNEL_DIR, "fmha_umma_desc.cuh"),
            os.path.join(KERNEL_DIR, "fmha_mixed_fp8_decode.cuh"),
        ]
        src_mtime = max(os.path.getmtime(p) for p in deps if os.path.exists(p))
        need_build = not os.path.isfile(so_path) or src_mtime > os.path.getmtime(so_path)
        if not need_build:
            _lib = ctypes.CDLL(so_path)
            return _lib

        os.makedirs(BUILD_DIR, exist_ok=True)
        nvcc = _find_nvcc()
        cmd = [
            nvcc, "-std=c++20", "-shared", "-Xcompiler", "-fPIC",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-gencode=arch=compute_100a,code=compute_100a",
            f"-I{KERNEL_DIR}", f"-I{REPO_ROOT}",
            "-O3", "--use_fast_math", "--expt-relaxed-constexpr",
            SOURCE, "-o", so_path, "-lcudart", "-lcuda",
        ]
        logger.info("Building libfmha_mixed_fp8.so (sm_100a)...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"mixed FP8 FMHA nvcc failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
        _lib = ctypes.CDLL(so_path)
        return _lib
    finally:
        _lib_lock = False


def _quantize_q_split(q: torch.Tensor, rope_dim: int):
    from dsv4.kernels.cuda.loader import get_cuda_module
    mod = get_cuda_module("fp8_attention_io", ["fp8_attention_io.cu"],
                          extra_cuda_cflags=[
                              "-gencode=arch=compute_100a,code=sm_100a",
                              "-O3", "--use_fast_math", "--expt-relaxed-constexpr",
                          ])
    return mod.quantize_q_fp8_split(q, rope_dim)


def fmha_mixed_fp8_decode_raw(
    q: torch.Tensor,                 # (B,H,1,HD) BF16
    k_nope_fp8: torch.Tensor,        # (N,NOPE) uint8/float8_e4m3fn
    k_nope_scale: torch.Tensor,      # (N,) FP32
    k_rope_bf16: torch.Tensor,       # (N,ROPE) BF16
    scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    rope_dim: int = 64,
):
    if q.dim() != 4:
        raise RuntimeError("q must be (B,H,T,HD)")
    B, H, T, HD = q.shape
    if T != 1:
        raise RuntimeError("mixed FP8 FMHA supports decode T==1 only")
    NOPE = HD - rope_dim
    if HD != 512 or NOPE != 448 or rope_dim != 64:
        raise RuntimeError(f"mixed FP8 FMHA first pass supports HD=512/NOPE=448/ROPE=64, got {HD}/{NOPE}/{rope_dim}")

    q = q.contiguous()
    k_nope_fp8 = k_nope_fp8.contiguous()
    k_nope_scale = k_nope_scale.contiguous()
    k_rope_bf16 = k_rope_bf16.contiguous()
    q_nope_fp8, q_nope_scale, q_rope = _quantize_q_split(q, rope_dim)

    N = k_nope_fp8.shape[0]
    o = torch.empty((B, H, T, HD), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((B, H, T), dtype=torch.float32, device=q.device)

    sink_ptr = ctypes.c_void_p(0)
    sb = None
    if attn_sink is not None:
        sb = attn_sink.float().contiguous()
        if sb.dim() == 1:
            sb = sb.unsqueeze(0).expand(B, -1).contiguous()
        if tuple(sb.shape) != (B, H):
            raise RuntimeError(f"sink bias shape {tuple(sb.shape)} != {(B,H)}")
        sink_ptr = ctypes.c_void_p(sb.data_ptr())

    lib = _ensure_built()
    ret = lib.fmha_mixed_fp8_decode_launch(
        ctypes.c_void_p(q_nope_fp8.data_ptr()),
        ctypes.c_void_p(q_nope_scale.data_ptr()),
        ctypes.c_void_p(q_rope.data_ptr()),
        ctypes.c_void_p(k_nope_fp8.data_ptr()),
        ctypes.c_void_p(k_nope_scale.data_ptr()),
        ctypes.c_void_p(k_rope_bf16.data_ptr()),
        ctypes.c_void_p(o.data_ptr()),
        ctypes.c_void_p(lse.data_ptr()),
        sink_ptr,
        ctypes.c_int(B), ctypes.c_int(H), ctypes.c_int(T), ctypes.c_int(N),
        ctypes.c_int(HD), ctypes.c_int(NOPE), ctypes.c_int(rope_dim),
        ctypes.c_int(q_nope_fp8.stride(1)), ctypes.c_int(q_nope_fp8.stride(0)),
        ctypes.c_int(q_nope_scale.stride(1)), ctypes.c_int(q_nope_scale.stride(0)),
        ctypes.c_int(q_rope.stride(1)), ctypes.c_int(q_rope.stride(0)),
        ctypes.c_int(o.stride(1)), ctypes.c_int(o.stride(0)),
        ctypes.c_int(lse.stride(1)), ctypes.c_int(lse.stride(0)),
        ctypes.c_float(scale),
    )
    if ret != 0:
        raise RuntimeError(f"mixed FP8 FMHA launch failed: return code {ret}")
    return o, lse
