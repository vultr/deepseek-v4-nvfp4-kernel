"""DSV4 Router — token-to-expert assignment.

Two routing modes that share an output shape:
  - 'dense': sqrt(softplus(X @ W_gate)) + per-expert bias, top-k selection.
             Used by MoE layers 3+ (the bulk of the network).
  - 'hash':  deterministic per-token-ID lookup, uniform weights.
             Used by the first 3 MoE layers per DSV4 §2.1.

Both modes produce (topk_weights, topk_ids) suitable for direct
consumption by Nvfp4MoE.run().

CUDA-graph-compatible: pre-allocated buffers, no CPU-GPU syncs.
Selection between modes is by layer_idx at construction time —
the kernel path is fixed once the Router is built so the dispatch
is constant-folded by torch.compile.
"""

from __future__ import annotations
from typing import Optional, Literal
import torch

from dsv4.ops.router import (
    register_router,
    dense_router_op,
    hash_router_op,
)


RouterMode = Literal["dense", "hash"]


class Router:
    """DSV4 expert router.

    Per the DeepSeek-V4 paper (§2.1):
      - Affinity activation is sqrt(softplus(·)), replacing V3's sigmoid(·).
      - Auxiliary-loss-free strategy: a learned per-expert bias (loaded
        from checkpoint, frozen at inference) is added to the activation
        for SELECTION only. The actual gating weight applied to expert
        outputs uses the UNBIASED activation.
      - First 3 MoE layers use Hash routing (Roller et al. 2021): a
        precomputed [vocab_size, k] LUT mapping token IDs to expert IDs.
        No gate GEMM is performed.
      - Sequence-wise balance loss is training-only; not applied here.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension. Must match W_gate's K dimension.
    num_experts : int
        Total routed experts (Flash: 256, Pro: 384). Shared experts are
        handled separately by Nvfp4SharedExpert.
    top_k : int
        Experts activated per token. DSV4 uses 6.
    routed_scaling_factor : float
        Post-renormalization scale on gating weights. DSV3 used 2.5;
        verify against the V4 checkpoint config — may be per-layer.
    mode : {'dense', 'hash'}
        Routing strategy. Decided at construction; cannot change at runtime.
    vocab_size : int, optional
        Required when mode='hash'. The LUT is [vocab_size, top_k] int32.
    max_num_tokens : int
        Upper bound on N for pre-allocated buffer sizing.
    device : str
        CUDA device.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 6,
        routed_scaling_factor: float = 2.5,
        *,
        mode: RouterMode,
        vocab_size: Optional[int] = None,
        max_num_tokens: int = 8192,
        device: str = "cuda",
    ):
        if mode == "hash" and vocab_size is None:
            raise ValueError("vocab_size is required when mode='hash'")
        if mode not in ("dense", "hash"):
            raise ValueError(f"unknown router mode: {mode!r}")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.mode = mode
        self.vocab_size = vocab_size
        self.max_num_tokens = max_num_tokens
        self.device = device

        # ---- Parameters (filled by load_weights / finalize_weights) ----
        # Dense mode — fused NVFP4 kernel (single-kernel, preferred):
        #   gate_weight: raw NVFP4 gate weight tensor [K_packed, E_packed] uint8
        #   gate_weight_scale: weight scale [K_sf, E_sf] FP8 E4M3
        #   gate_ws2: weight_scale_2 (global scale base)
        #   gate_input_scale: input_scale (activation global scale base)
        # Dense mode — 2-kernel NVFP4 path (fallback):
        #   gate_lin: Nvfp4Linear for the gate projection
        # Dense mode — BF16 fallback:
        #   W_gate: BF16 weight for cuBLAS when NVFP4 scales not available
        # Hash mode:
        #   hash_lut: [vocab_size, top_k] int32 — precomputed expert IDs.
        self.gate_weight = None        # Raw NVFP4 weight for fused kernel
        self.gate_weight_scale = None   # FP8 E4M3 scale for fused kernel
        self.gate_ws2 = None            # weight_scale_2 for fused kernel
        self.gate_input_scale = None    # input_scale for fused kernel
        self.gate_lin = None            # Nvfp4Linear for 2-kernel NVFP4 path
        self.W_gate: Optional[torch.Tensor] = None  # BF16 fallback
        self.e_bias: Optional[torch.Tensor] = None
        self.hash_lut: Optional[torch.Tensor] = None

        # ---- Pre-allocated output buffers (cudagraph-safe) ----
        self._topk_weights_buf: Optional[torch.Tensor] = None
        self._topk_ids_buf: Optional[torch.Tensor] = None

        # Runner ID assigned on first call (see custom_op pattern).
        self._runner_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(
        self,
        W_gate: Optional[torch.Tensor] = None,
        e_bias: Optional[torch.Tensor] = None,
        hash_lut: Optional[torch.Tensor] = None,
    ) -> None:
        """Populate router parameters from a checkpoint shard.

        Dense mode expects (W_gate, e_bias). Hash mode expects (hash_lut).
        Mismatches with self.mode raise immediately — these errors are
        nearly always loader bugs and silent acceptance would mask them.
        """
        if self.mode == "dense":
            if e_bias is None:
                raise ValueError("dense router needs e_bias")
            assert e_bias.shape == (self.num_experts,), \
                f"e_bias shape {tuple(e_bias.shape)} != ({self.num_experts},)"
            self.e_bias = e_bias.to(device=self.device, dtype=torch.float32)
            if W_gate is not None:
                self.W_gate = W_gate.to(device=self.device, dtype=torch.bfloat16)
            # gate_lin is set separately via load_nvfp4_gate()
        else:  # hash
            if hash_lut is None:
                raise ValueError("hash router needs hash_lut")
            assert hash_lut.shape == (self.vocab_size, self.top_k), \
                f"hash_lut shape {tuple(hash_lut.shape)} != " \
                f"{(self.vocab_size, self.top_k)}"
            assert (hash_lut >= 0).all() and (hash_lut < self.num_experts).all(), \
                "hash_lut contains out-of-range expert IDs"
            self.hash_lut = hash_lut.to(device=self.device, dtype=torch.int32)

    def load_nvfp4_gate(self, gate_lin) -> None:
        """Set the NVFP4 gate linear layer (2-kernel path).

        Called by the single_shot after constructing the Nvfp4Linear
        from checkpoint NVFP4 scales. When set, _run_dense_impl uses
        the production NVFP4 GEMM path instead of BF16 cuBLAS.
        """
        self.gate_lin = gate_lin

    def load_nvfp4_fused_gate(self, gate_weight, gate_weight_scale,
                               gate_ws2, gate_input_scale,
                               gate_weight_bf16=None) -> None:
        """Set raw NVFP4 gate tensors and create Nvfp4Linear for production GEMM."""
        self.gate_weight = gate_weight.to(device=self.device)
        self.gate_weight_scale = gate_weight_scale.to(device=self.device)
        self.gate_ws2 = gate_ws2.to(device=self.device) if gate_ws2 is not None else None
        self.gate_input_scale = gate_input_scale.to(self.device)

        # Create Nvfp4Linear from BF16 weight (handles layout correctly)
        if gate_weight_bf16 is not None:
            from dsv4.layers.linear import Nvfp4Linear
            from dsv4.ops.quantize import quantize_to_nvfp4
            E = gate_weight_bf16.shape[0]
            gate_lin = Nvfp4Linear(in_features=self.hidden_size, out_features=E, device=self.device)
            g_fp4, g_sf, g_gs = quantize_to_nvfp4(gate_weight_bf16.bfloat16().to(self.device))
            gate_lin.fp4 = [g_fp4]
            gate_lin.sf = [g_sf]
            gate_lin.gs = [g_gs]
            ws2_val = gate_ws2.float().item() if gate_ws2.numel() == 1 else gate_ws2.float().mean().item()
            gate_lin.ws2 = [torch.tensor([ws2_val], device=self.device, dtype=torch.float32)]
            gate_lin._activation_global_scale = gate_input_scale.float().item() if gate_input_scale.numel() == 1 else gate_input_scale.float().mean().item()
            gate_lin._use_runtime_gsa = True  # compute gsa from actual input to avoid E4M3 overflow
            gate_lin.finalize_weights()
            self.gate_lin = gate_lin

    def finalize_weights(self) -> None:
        """Allocate output buffers and JIT-compile the routing kernel.

        Mirrors the finalize_weights() pattern in Nvfp4Linear: a one-time
        setup step called after all parameters are loaded. Triggers
        kernel compilation so the first forward isn't paying that cost.
        """
        self._topk_weights_buf = torch.empty(
            self.max_num_tokens, self.top_k,
            dtype=torch.float32, device=self.device,
        )
        self._topk_ids_buf = torch.empty(
            self.max_num_tokens, self.top_k,
            dtype=torch.int32, device=self.device,
        )

        # Eager JIT — dispatcher knows our mode and triggers the right
        # kernel's compile path. See dsv4/ops/router.py.
        from dsv4.ops.router import warmup_router_compilation
        warmup_router_compilation(self)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce (topk_weights, topk_ids) for downstream Nvfp4MoE.

        Parameters
        ----------
        hidden_states : Tensor [N, hidden_size] bfloat16
            Required for dense mode. Ignored for hash mode (kept in the
            signature so the call site is mode-agnostic).
        token_ids : Tensor [N] int32, optional
            Required for hash mode. Ignored for dense mode.

        Returns
        -------
        topk_weights : Tensor [N, top_k] float32
        topk_ids     : Tensor [N, top_k] int32

        Notes
        -----
        Both outputs are views into pre-allocated buffers — do not retain
        them across router calls. Nvfp4MoE consumes them immediately,
        which matches its existing contract.
        """
        if self._topk_weights_buf is None:
            raise RuntimeError("Router.finalize_weights() not called")

        if self.mode == "dense":
            if hidden_states is None:
                raise ValueError("dense router requires hidden_states")
            return self._run_dense(hidden_states)
        else:
            if token_ids is None:
                raise ValueError("hash router requires token_ids")
            return self._run_hash(token_ids)

    # ------------------------------------------------------------------
    # Mode-specific dispatch — each routes through a torch.library.custom_op
    # so Dynamo / torch.compile treats the kernel as opaque.
    # ------------------------------------------------------------------
    def _run_dense(self, hidden_states: torch.Tensor):
        if self._runner_id is None:
            self._runner_id = register_router(self)
        return dense_router_op(
            hidden_states,
            self._runner_id,
            self.num_experts,
            self.top_k,
        )

    def _run_hash(self, token_ids: torch.Tensor):
        if self._runner_id is None:
            self._runner_id = register_router(self)
        return hash_router_op(
            token_ids,
            self._runner_id,
            self.top_k,
        )

    # ------------------------------------------------------------------
    # Called by the custom_op dispatch in dsv4/ops/router.py — not by user code.
    # ------------------------------------------------------------------
    def _run_dense_impl(self, hidden_states: torch.Tensor):
        """Hot-path: fused NVFP4, 2-kernel NVFP4, or BF16 fallback.

        Priority:
        1. Fused NVFP4 kernel (single-kernel GEMM + router epilogue)
        2. 2-kernel NVFP4 path (Nvfp4Linear + activation_topk)
        3. BF16 cuBLAS fallback
        """
        N = hidden_states.shape[0]
        out_w = self._topk_weights_buf[:N]
        out_ids = self._topk_ids_buf[:N]
        if self.gate_lin is not None:
            # NVFP4 production GEMM path (proven Nvfp4Linear)
            from dsv4.kernels.router import dense_router_dispatch_nvfp4
            dense_router_dispatch_nvfp4(
                hidden_states=hidden_states,
                gate_lin=self.gate_lin,
                e_bias=self.e_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                top_k=self.top_k,
                out_weights=out_w,
                out_ids=out_ids,
            )
        elif self.gate_weight is not None:
            # Fused NVFP4 path (gate_lin was not created)
            # Fall back to BF16
            from dsv4.kernels.router import dense_router_dispatch
            dense_router_dispatch(
                hidden_states=hidden_states,
                W_gate=self.W_gate,
                e_bias=self.e_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                top_k=self.top_k,
                out_weights=out_w,
                out_ids=out_ids,
            )
        else:
            from dsv4.kernels.router import dense_router_dispatch
            dense_router_dispatch(
                hidden_states=hidden_states,
                W_gate=self.W_gate,
                e_bias=self.e_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                top_k=self.top_k,
                out_weights=out_w,
                out_ids=out_ids,
            )
        return out_w, out_ids

    def _run_hash_impl(self, token_ids: torch.Tensor):
        """Hot-path entry into the hash gather kernel.

        Implementation lives in dsv4/kernels/cuda/hash_router.cu via the
        wrapper in dsv4/ops/router.py.
        """
        from dsv4.kernels.router import hash_router_dispatch
        N = token_ids.shape[0]
        out_w = self._topk_weights_buf[:N]
        out_ids = self._topk_ids_buf[:N]
        hash_router_dispatch(
            token_ids=token_ids,
            hash_lut=self.hash_lut,
            top_k=self.top_k,
            out_weights=out_w,  # filled with 1/k
            out_ids=out_ids,
        )
        return out_w, out_ids
