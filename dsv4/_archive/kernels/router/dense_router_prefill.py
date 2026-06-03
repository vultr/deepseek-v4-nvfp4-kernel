"""DSV4 Dense Router — prefill path.

For prefill with N >= ~256, the gate GEMM has enough work to make DeepGEMM
(or the standard BF16 persistent GEMM) the better choice for the matmul,
with a separate fused activation+top-k kernel on the output.

Currently both decode and prefill go through this path (GEMM + activation_topk).
The CuTeDSL fused decode kernel will replace the small-N path when complete.
"""

from __future__ import annotations
import torch


def dense_router_prefill(
    hidden_states: torch.Tensor,       # [N, hidden_size] BF16
    W_gate: torch.Tensor,              # [hidden_size, num_experts] BF16
    e_bias: torch.Tensor,              # [num_experts] FP32
    routed_scaling_factor: float,
    top_k: int,
    out_weights: torch.Tensor,         # [N, top_k] FP32
    out_ids: torch.Tensor,             # [N, top_k] int32
):
    """Prefill path: BF16 GEMM → FP32 logits → fused activation + top-k.

    Step 1: logits = hidden_states @ W_gate (BF16 GEMM, FP32 output)
    Step 2: fused kernel: act=sqrt(softplus(logits)), score=act+bias,
            top-k, renorm → (out_weights, out_ids)
    """
    # FP32 GEMM for numerical accuracy in the activation.
    # BF16 accumulator would lose too much precision for softplus.
    logits = torch.nn.functional.linear(hidden_states.float(), W_gate.float())

    from dsv4.kernels.router._activation_topk import run_fused_activation_topk
    run_fused_activation_topk(
        logits, e_bias, routed_scaling_factor, top_k,
        out_weights, out_ids,
    )
