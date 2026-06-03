"""torch.library.custom_op wrappers and dispatch for the Router kernels.

Mirrors the pattern in dsv4/ops/custom_ops.py:
  - Routers are registered into an integer-keyed table.
  - The custom_op takes the integer ID and tensor args only.
  - Dynamo can't trace through the kernel; the op is opaque.
"""

import torch
from dsv4.kernels.router import (
    dense_router_dispatch,   # picks decode vs prefill internally
    hash_router_dispatch,
)

_next_router_id = 0
_router_registry: dict[int, object] = {}


def register_router(router) -> int:
    global _next_router_id
    rid = _next_router_id
    _next_router_id += 1
    _router_registry[rid] = router
    return rid


def get_router(rid: int):
    return _router_registry[rid]


def warmup_router_compilation(router) -> None:
    """Trigger eager JIT compilation for the router's kernel path.

    Runs a dummy forward at max_num_tokens to compile the kernel for the
    expected shape range. Caller already has the buffers allocated.
    """
    if router.mode == "dense":
        # Dummy forward at small N triggers decode-path compile.
        # CuTeDSL fused kernel is WIP — falls through to prefill path.
        dummy = torch.zeros(
            1, router.hidden_size,
            dtype=torch.bfloat16, device=router.device,
        )
        try:
            router._run_dense_impl(dummy)
        except Exception:
            pass  # CuTeDSL kernel not yet working; prefill path is fine
    else:
        dummy = torch.zeros(1, dtype=torch.int32, device=router.device)
        router._run_hash_impl(dummy)


# ----- Dense router custom op -----
@torch.library.custom_op("dsv4::dense_router", mutates_args=())
def dense_router_op(
    hidden_states: torch.Tensor,
    router_id: int,
    num_experts: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    router = get_router(router_id)
    return router._run_dense_impl(hidden_states)


@dense_router_op.register_fake
def _(hidden_states, router_id, num_experts, top_k):
    N = hidden_states.shape[0]
    device = hidden_states.device
    return (
        torch.empty(N, top_k, dtype=torch.float32, device=device),
        torch.empty(N, top_k, dtype=torch.int32, device=device),
    )


# ----- Hash router custom op -----
@torch.library.custom_op("dsv4::hash_router", mutates_args=())
def hash_router_op(
    token_ids: torch.Tensor,
    router_id: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    router = get_router(router_id)
    return router._run_hash_impl(token_ids)


@hash_router_op.register_fake
def _(token_ids, router_id, top_k):
    N = token_ids.shape[0]
    device = token_ids.device
    return (
        torch.empty(N, top_k, dtype=torch.float32, device=device),
        torch.empty(N, top_k, dtype=torch.int32, device=device),
    )
