"""CUDA kernel loader with compile-once caching.

Compiles .cu kernels on first call, caches the loaded module for subsequent calls.
Eliminates the JIT recompilation overhead from torch.utils.cpp_extension.load
being called on every kernel invocation (was ~100ms per call, called ~500x per token).

Usage:
    from dsv4.kernels.cuda.loader import get_cuda_module
    mod = get_cuda_module("fused_amax_quantize", ["fused_amax_quantize.cu"])
    result = mod.quantize_nvfp4_from_buffer(x, divisor)
"""
import os
import time
import hashlib
import torch
from torch.utils.cpp_extension import load

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_KERNEL_DIR, "_build_cache")
_LOADED_MODULES = {}

# Maximum age of a stale lock file before we remove it (seconds).
# torch.utils.cpp_extension.load creates a lock file during compilation.
# If the process is killed during compilation, the lock remains and the
# next process spins forever polling it. This timeout prevents that.
_STALE_LOCK_TIMEOUT_S = 600  # 10 minutes


def _cleanup_stale_lock():
    """Remove stale lock files from the build cache directory.
    
    torch.utils.cpp_extension.load creates a 'lock' file in the build
    directory during compilation. If the compiling process is killed
    (OOM, timeout, user interrupt), the lock file is never removed and
    subsequent processes spin forever waiting for it.
    
    This function checks if a lock file exists and is older than
    _STALE_LOCK_TIMEOUT_S. If so, it removes it.
    """
    lock_path = os.path.join(_CACHE_DIR, "lock")
    if os.path.exists(lock_path):
        try:
            lock_age = time.time() - os.path.getmtime(lock_path)
            if lock_age > _STALE_LOCK_TIMEOUT_S:
                os.remove(lock_path)
                print(f"[loader] Removed stale lock file (age={lock_age:.0f}s)", flush=True)
        except OSError:
            pass  # Lock was removed between exists() and remove()


def get_cuda_module(name, sources, extra_cuda_cflags=None):
    """Load a CUDA kernel module, compiling once and caching forever.
    
    Args:
        name: Module name (used for caching key).
        sources: List of .cu filenames relative to the kernels/cuda/ directory.
        extra_cuda_cflags: Optional list of extra CUDA compiler flags.
    
    Returns:
        The loaded Python module with the kernel functions.
    """
    if name in _LOADED_MODULES:
        return _LOADED_MODULES[name]
    
    # Clean up stale lock files from crashed previous compilations
    _cleanup_stale_lock()
    
    source_paths = [os.path.join(_KERNEL_DIR, s) for s in sources]
    
    # Build a cache key from source file contents + compile flags
    hasher = hashlib.md5()
    for sp in source_paths:
        hasher.update(open(sp, 'rb').read())
    cflags = extra_cuda_cflags or []
    for cf in cflags:
        hasher.update(cf.encode())
    cache_key = f"{name}_{hasher.hexdigest()}"
    
    # Ensure cache directory exists
    os.makedirs(_CACHE_DIR, exist_ok=True)
    
    cflags = cflags or [
        "-gencode=arch=compute_100a,code=sm_100a",
        "-O3",
        "--use_fast_math",
    ]
    
    mod = load(
        name=cache_key,
        sources=source_paths,
        extra_cuda_cflags=cflags,
        build_directory=_CACHE_DIR,
        verbose=False,
    )
    
    _LOADED_MODULES[name] = mod
    return mod



