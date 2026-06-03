"""torch.library.custom_op wrappers for CuTeDSL NVFP4 kernels.

Dynamo (torch.compile fullgraph) cannot trace through CuTeDSL internals
(JIT compilation, cute.compile, etc.). By wrapping the runner calls in
torch.library.custom_op, Dynamo treats them as opaque black boxes.

This is the correct approach per PyTorch's extensibility model:
- custom_op is the supported way to make Dynamo skip tracing
- autograd.Function does NOT work reliably with fullgraph mode
- The runner's _run_impl is already cudagraph-safe

The registry pattern: custom ops can only take tensor/scalar arguments.
We store runners in a global dict keyed by integer ID, and pass the ID
as an int parameter. During Dynamo tracing, the fake impl returns a
correctly-shaped tensor without touching the runner. During execution,
the real impl looks up the runner and calls _run_impl.
"""

import torch

# ---------------------------------------------------------------------------
# Runner registry — maps integer IDs to runner objects
# ---------------------------------------------------------------------------
_next_runner_id = 0
_runner_registry: dict[int, object] = {}


def register_runner(runner) -> int:
    """Register a CuTeDSL runner and return its integer ID."""
    global _next_runner_id
    rid = _next_runner_id
    _next_runner_id += 1
    _runner_registry[rid] = runner
    return rid


def get_runner(rid: int):
    """Look up a runner by ID."""
    return _runner_registry[rid]


# ---------------------------------------------------------------------------
# NVFP4 Linear GEMM custom op (single linear layer)
# ---------------------------------------------------------------------------
@torch.library.custom_op("nvfp4::linear_gemm", mutates_args=())
def nvfp4_linear_gemm(
    x: torch.Tensor,
    runner_id: int,
    out_features: int,
) -> torch.Tensor:
    """Opaque NVFP4 linear GEMM for torch.compile.

    Args:
        x: (M, K) BF16 input
        runner_id: integer key into the runner registry
        out_features: output dimension (for shape inference)
    Returns:
        (M, out_features) BF16 output
    """
    runner = get_runner(runner_id)
    return runner._run_impl(x)


@nvfp4_linear_gemm.register_fake
def _(x, runner_id, out_features):
    return torch.empty(x.shape[0], out_features, dtype=torch.bfloat16, device=x.device)


# ---------------------------------------------------------------------------
# NVFP4 MoE custom op (L1 + SiLU + L2 grouped GEMM)
# ---------------------------------------------------------------------------
@torch.library.custom_op("nvfp4::moe_gemm", mutates_args=())
def nvfp4_moe_gemm(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    runner_id: int,
    hidden_size: int,
) -> torch.Tensor:
    """Opaque NVFP4 MoE GEMM for torch.compile.

    Args:
        hidden_states: (M, K) BF16 input
        topk_weights: (M, top_k) float32 routing weights
        topk_ids: (M, top_k) int32 expert IDs
        runner_id: integer key into the runner registry
        hidden_size: output dimension (for shape inference)
    Returns:
        (M, hidden_size) BF16 output
    """
    runner = get_runner(runner_id)
    return runner._run_impl(hidden_states, topk_weights, topk_ids)


@nvfp4_moe_gemm.register_fake
def _(hidden_states, topk_weights, topk_ids, runner_id, hidden_size):
    return torch.empty(
        hidden_states.shape[0], hidden_size,
        dtype=torch.bfloat16, device=hidden_states.device,
    )


# ---------------------------------------------------------------------------
# DSV4 Sparse FMHA custom op (attention with SWA + sink bias)
# ---------------------------------------------------------------------------
@torch.library.custom_op("dsv4::sparse_fmha_with_swa", mutates_args=())
def dsv4_sparse_fmha(
    q: torch.Tensor,          # (n_q_heads, T, hd) BF16
    k: torch.Tensor,          # (n_kv_heads, N, hd) or (N, hd) BF16
    v: torch.Tensor,          # same as k
    sink_bias: torch.Tensor,  # (n_q_heads,) FP32 — can be zeros if unused
    scale: float,
    swa_len: int,
    is_causal: bool,
    n_comp: int,
) -> torch.Tensor:
    """Opaque DSV4 attention for torch.compile.

    Delegates to dsv4_attention with the appropriate flags.
    sink_bias is always passed (use zeros when unused) to keep the
    custom_op signature tensor-only for Dynamo compatibility.
    """
    from dsv4.kernels.attention.production import dsv4_attention as _dsv4_attention

    # If sink_bias is all zeros and n_comp == 0, skip sink bias
    has_sink = n_comp > 0 and sink_bias.abs().sum().item() > 0
    return _dsv4_attention(
        q, k, v, scale=scale,
        swa_len=swa_len if swa_len > 0 else None,
        is_causal=is_causal,
        n_comp=n_comp,
        sink_bias=sink_bias if has_sink else None,
    )


@dsv4_sparse_fmha.register_fake
def _(q, k, v, sink_bias, scale, swa_len, is_causal, n_comp):
    return torch.empty_like(q)
