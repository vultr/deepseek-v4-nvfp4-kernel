"""Python wrapper for the append_swa CUDA kernel.

Writes raw BF16 KV into the FP8/BF16 split state cache layout.
Quantizes the non-RoPE half BF16 -> FP8 (E4M3 amax-based scaling),
writes the RoPE half as-is, computes per-token inverse scale, and
updates the ring buffer head + position field.

One block per token. Threads cooperatively:
  1. Compute amax over fp8-dim elements (warp reduce).
  2. Quantize BF16 -> FP8 with per-token scale.
  3. Write FP8 entries + BF16 RoPE entries + inv_scale + position.
  4. Atomic increment ring buffer head.
"""
import os
import torch
from torch.utils.cpp_extension import load

_kernel_module = None


def _get_kernel_module():
    global _kernel_module
    if _kernel_module is not None:
        return _kernel_module
    kernel_dir = os.path.join(os.path.dirname(__file__), "..", "cuda")
    _kernel_module = load(
        name="append_swa",
        sources=[os.path.join(kernel_dir, "append_swa.cu")],
        extra_cuda_cflags=["-O3", "--generate-code=arch=compute_100a,code=[sm_100a]"],
        verbose=False,
    )
    return _kernel_module


def append_swa_kernel(
    raw_kv: torch.Tensor,      # (T, head_dim) BF16
    request_slots: torch.Tensor,  # (T,) int32
    positions: torch.Tensor,     # (T,) int32
    swa_fp8: torch.Tensor,      # (max_req, n_win, fp8_dim) uint8
    swa_rope: torch.Tensor,     # (max_req, n_win, rope_dim) BF16
    swa_inv: torch.Tensor,      # (max_req, n_win) FP32
    swa_pos: torch.Tensor,      # (max_req, n_win) int32
    swa_head: torch.Tensor,     # (max_req,) int32
    rope_dim: int,
):
    mod = _get_kernel_module()
    mod.append_swa(
        raw_kv, request_slots, positions,
        swa_fp8, swa_rope, swa_inv, swa_pos, swa_head,
        rope_dim,
    )
