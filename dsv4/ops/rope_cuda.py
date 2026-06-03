"""CUDA RoPE kernel — 1 kernel launch per call instead of 5-6 PyTorch ops.

Uses ctypes to call the compiled kernel directly (no ATen/pybind11).
Same pattern as fmha_multitile_op.py and other production kernels.
"""

import torch
import ctypes
import subprocess
from pathlib import Path

_LIB = None

def _compile_and_load():
    global _LIB
    if _LIB is not None:
        return _LIB
    
    cu_path = Path(__file__).parent.parent / "kernels" / "cuda" / "rope_cuda.cu"
    assert cu_path.exists(), f"rope_cuda.cu not found at {cu_path}"
    
    # Compile to shared library
    build_dir = Path(__file__).parent / "cuda" / "_build_cache"
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / "librope_cuda.so"
    
    if not so_path.exists() or cu_path.stat().st_mtime > so_path.stat().st_mtime:
        nvcc = "/usr/local/cuda/bin/nvcc"
        cmd = [
            nvcc, "-shared", "-o", str(so_path), str(cu_path),
            "-arch=sm_100a",
            "--generate-code=arch=compute_100a,code=[sm_100a,compute_100a]",
            "-use_fast_math", "-O3",
            "-Xcompiler", "-fPIC",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"rope_cuda.cu compilation failed:\n{result.stderr}")
    
    _LIB = ctypes.CDLL(str(so_path))
    return _LIB


def apply_rope(x, positions, cos_cache, sin_cache, rope_dim, inverse=False):
    """Apply forward or inverse RoPE in-place using a single CUDA kernel.
    
    Args:
        x: (T, n_h, hd) BF16 — modified in-place
        positions: (T,) int64 — token positions  
        cos_cache: (max_pos, rope_dim//2) float32
        sin_cache: (max_pos, rope_dim//2) float32
        rope_dim: 64
        inverse: True for inverse RoPE
    
    Returns:
        x (modified in-place)
    """
    lib = _compile_and_load()
    T, n_h, hd = x.shape
    nope_dim = hd - rope_dim
    half_rope = rope_dim // 2
    
    # Ensure types and devices
    pos = positions.to(device=x.device, dtype=torch.int64)
    assert x.dtype == torch.bfloat16
    assert cos_cache.dtype == torch.float32
    assert sin_cache.dtype == torch.float32
    
    # Launch parameters
    total_pairs = T * n_h * half_rope
    threads = 256
    blocks = (total_pairs + threads - 1) // threads
    
    # Get raw CUDA stream
    stream = torch.cuda.current_stream().cuda_stream
    
    # Call the kernel
    lib.apply_rope_launch(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(pos.data_ptr()),
        ctypes.c_void_p(cos_cache.data_ptr()),
        ctypes.c_void_p(sin_cache.data_ptr()),
        ctypes.c_int(T),
        ctypes.c_int(n_h),
        ctypes.c_int(hd),
        ctypes.c_int(nope_dim),
        ctypes.c_int(rope_dim),
        ctypes.c_bool(inverse),
        ctypes.c_int(blocks),
        ctypes.c_int(threads),
        ctypes.c_void_p(stream),
    )
    return x
