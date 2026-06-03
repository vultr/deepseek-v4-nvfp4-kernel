"""Python wrapper for the fused activation + top-k CUDA kernel.

This module lazy-loads the CUDA extension (same pattern as dsv4/ops/topk.py)
and provides the run_fused_activation_topk() function called by dense_router_dispatch.
"""

import os
import torch

_kernel_module = None


def _get_kernel_module():
    """Lazy-load the fused_activation_topk CUDA extension."""
    global _kernel_module
    if _kernel_module is not None:
        return _kernel_module

    from torch.utils.cpp_extension import load
    kernel_dir = os.path.join(os.path.dirname(__file__), "..", "cuda")
    _kernel_module = load(
        name="fused_activation_topk",
        sources=[os.path.join(kernel_dir, "activation_topk.cu")],
        extra_cuda_cflags=["-O3", "--generate-code=arch=compute_100a,code=[sm_100a]"],
        verbose=False,
    )
    return _kernel_module


def run_fused_activation_topk(
    logits: torch.Tensor,          # [N, E] FP32
    e_bias: torch.Tensor,          # [E] FP32
    routed_scaling_factor: float,
    top_k: int,
    out_weights: torch.Tensor,     # [N, top_k] FP32, pre-allocated
    out_ids: torch.Tensor,         # [N, top_k] int32, pre-allocated
):
    """Run the fused activation + top-k + renormalization kernel.

    Computes:
      act = sqrt(softplus(logits))
      score = act + e_bias
      topk_ids = argtopk(score, k=top_k)   (tie-break: lower index wins)
      raw_w = gather(act, topk_ids)          (unbiased activation)
      topk_w = raw_w / sum(raw_w) * scaling  (renormalized)
    """
    mod = _get_kernel_module()
    return mod.fused_activation_topk(
        logits, e_bias,
        float(routed_scaling_factor),
        top_k,
        out_weights, out_ids,
    )


def run_fused_activation_topk_pre_activated(
    activated_scores: torch.Tensor, # [N, E] FP32, already sqrt(softplus(logits))
    e_bias: torch.Tensor,          # [E] FP32
    routed_scaling_factor: float,
    top_k: int,
    out_weights: torch.Tensor,     # [N, top_k] FP32, pre-allocated
    out_ids: torch.Tensor,         # [N, top_k] int32, pre-allocated
):
    """Run top-k + renormalization on pre-activated scores.

    The CUDA kernel is called with logits=activated_scores.
    Since the kernel computes sqrt(softplus(logits)) + e_bias,
    we pass e_bias=0 and add e_bias ourselves in a pre-step,
    then call the kernel with the scores (which are already activated).

    Actually, simpler approach: just add e_bias to activated_scores,
    then call the standard kernel with e_bias=0. The kernel will
    compute sqrt(softplus(score + 0)) = sqrt(softplus(score)).
    But that double-applies softplus!

    Correct approach: Add a dedicated kernel entry point that
    skips activation and just does top-k + renorm.
    For now, use the existing kernel with a workaround:
    pre-add e_bias to get selection scores, do top-k on those,
    then gather the unbiased activations for weights.
    """
    # Step 1: selection scores = activated + e_bias
    sel_scores = activated_scores + e_bias.unsqueeze(0)  # [N, E]

    # Step 2: top-k on selection scores
    topk_vals, topk_indices = sel_scores.topk(top_k, dim=-1)  # [N, k]

    # Step 3: gather unbiased activations (without e_bias)
    raw_w = activated_scores.gather(1, topk_indices)  # [N, k]

    # Step 4: renormalize
    row_sum = raw_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    out_weights.copy_(raw_w / row_sum * routed_scaling_factor)
    out_ids.copy_(topk_indices.to(torch.int32))
