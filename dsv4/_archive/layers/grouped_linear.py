"""CuTeDSL NVFP4 Grouped Linear for wo_a (o_proj first half).

wo_a in DeepSeek V4 is a grouped matmul (bmm) with n_local_groups=8 groups.
Each group: (tokens, heads_per_group * head_dim) × (heads_per_group * head_dim, o_lora_rank) → (tokens, o_lora_rank)

The vLLM forward does this via DeepGEMM fp8_einsum with equation "bhr,hdr->bhd".
We replace it with our CuTeDSL ScaledGroupedGemm using n_local_groups as num_experts,
where every token goes to every "expert" (group).

wo_a is loaded as BF16 from our NVFP4 checkpoint, then quantized to NVFP4 here.

CUDA-graph-compatible: all buffers pre-allocated, no CPU-GPU syncs.
"""

import torch

from dsv4.ops.quantize import (
    quantize_activation_nvfp4,
    quantize_weight_to_nvfp4,
    quantize_nvfp4_gpu_fused,
)
from dsv4.ops.layouts import (
    make_b_k_major,
    assemble_scales_2d_side,
    assemble_scales_3d_side,
)
from dsv4.ops.gemm_runner import (
    run_nvfp4_grouped_gemm,
)
from dsv4.ops.layouts import (
    ceil_div as cutedsl_ceil_div,
    pad_and_swizzle_single,
)
from dsv4.ops.custom_ops import register_runner, nvfp4_linear_gemm


class Nvfp4GroupedLinear:
    """Grouped NVFP4 linear for wo_a (o-projection first half).

    Handles the "bhr,hdr->bhd" einsum pattern:
    - o: (tokens, n_local_heads, head_dim) → reshape to (tokens, n_local_groups, heads_per_group * head_dim)
    - wo_a: (n_local_groups, heads_per_group * head_dim, o_lora_rank) → NVFP4 per group
    - z: (tokens, n_local_groups, o_lora_rank)

    Uses ScaledGroupedGemm with num_groups=n_local_groups.
    Every token goes to every group (no routing).

    CUDA-graph-compatible: all buffers pre-allocated, no CPU-GPU syncs.
    """

    def __init__(
        self,
        n_local_groups: int,
        heads_per_group: int,
        head_dim: int,
        o_lora_rank: int,
        max_num_tokens: int = 8192,
        device: str = "cuda",
    ):
        self.n_local_groups = n_local_groups
        self.heads_per_group = heads_per_group
        self.head_dim = head_dim
        self.o_lora_rank = o_lora_rank
        self.max_num_tokens = max_num_tokens
        self.device = device

        # Per-group dimensions
        self.group_in_features = heads_per_group * head_dim  # 8192
        self.group_out_features = o_lora_rank  # 1536

        # NVFP4 weight storage: lists of per-group tensors
        self._weight_fp4 = None  # list of (K//2, N) float4_e2m1fn_x2
        self._weight_sf = None   # list of (K//16, N) float8_e4m3fn
        self._weight_gs = None   # list of float32

        # Processed weights (set by finalize_weights)
        self._mat_b = None
        self._scale_b = None
        self._gsb = None

        # Activation global scale
        self._activation_global_scale = 1.0 / (6.0 * 448.0)

        # Pre-allocated buffers
        self._padded_x_fp4_buf = None
        self._gsa_buf = None
        self._expert_offsets_buf = None
        self._buffers_allocated = False

    def set_bf16_weight(self, wo_a_bf16: torch.Tensor):
        """Set wo_a weight from BF16 and quantize to NVFP4.

        Args:
            wo_a_bf16: (n_local_groups * o_lora_rank, heads_per_group * head_dim) BF16
                       OR (n_local_groups, heads_per_group * head_dim, o_lora_rank) if from bmm
        """
        # Quantize each group separately
        fp4_list = []
        sf_list = []
        gs_list = []

        if wo_a_bf16.ndim == 3:
            # bmm format: (n_local_groups, heads_per_group * head_dim, o_lora_rank)
            for g in range(self.n_local_groups):
                w_g = wo_a_bf16[g]  # (in_features, out_features)
                w_fp4, w_sf, w_gs = quantize_weight_to_nvfp4(w_g)
                # quantize_weight_to_nvfp4 returns (K//2, N) with K=in_features
                # Our kernel expects (K_packed, N_packed) where K is the contraction dim
                # For weight (in_features, out_features): K=in_features (contraction)
                # quantize_weight_to_nvfp4 treats dim 0 as K, so result is (K//2, N) ✓
                fp4_list.append(w_fp4)
                sf_list.append(w_sf)
                gs_list.append(w_gs)
        else:
            # Dense format: (n_local_groups * o_lora_rank, heads_per_group * head_dim)
            # Split into per-group blocks
            for g in range(self.n_local_groups):
                start = g * self.o_lora_rank
                end = start + self.o_lora_rank
                w_g = wo_a_bf16[start:end, :]  # (o_lora_rank, in_features)
                # NOTE: This is transposed — weight is (out, in) but quantize_weight_to_nvfp4
                # expects (K, N) where K is the packed/contraction dim.
                # For matmul X @ W^T, the contraction dim of W is dim 1 (in_features).
                # So we need to transpose before quantizing.
                w_g_t = w_g.T  # (in_features, o_lora_rank) = (K, N)
                w_fp4, w_sf, w_gs = quantize_weight_to_nvfp4(w_g_t)
                fp4_list.append(w_fp4)
                sf_list.append(w_sf)
                gs_list.append(w_gs)

        self._weight_fp4 = fp4_list
        self._weight_sf = sf_list
        self._weight_gs = gs_list

    def load_nvfp4_weight(self, weight, weight_scale, weight_scale_2=None, input_scale=None):
        """Load NVFP4 weights directly from checkpoint — no dequant/re-quant.

        The checkpoint stores weights in (out_features, in_features) layout:
          weight: (n_groups * o_rank, group_in_features // 2) uint8
          weight_scale: (n_groups * o_rank, group_in_features // 16) float8_e4m3fn
          weight_scale_2: scalar or (n_groups * o_rank,) float
          input_scale: scalar or (n_groups * o_rank,) float (unused for weight dequant)

        Each group's chunk is (o_rank, K_packed) = (N, K_packed) in row-major.
        Our GEMM expects (K_packed, N) per group, so we transpose each group.
        Block scales follow the same transpose.

        Args:
            weight: (n_groups * o_rank, group_in_features // 2) uint8
            weight_scale: (n_groups * o_rank, group_in_features // 16) float8_e4m3fn
            weight_scale_2: scalar or per-row scale tensor (optional)
            input_scale: scalar or per-row (unused — for activation quantization)
        """
        fp4_list = []
        sf_list = []
        gs_list = []

        K_packed = self.group_in_features // 2
        N = self.o_lora_rank
        K_sf = self.group_in_features // 16  # block scale dim along K

        for g in range(self.n_local_groups):
            # Extract this group's weight: (o_rank, K_packed) = (N, K_packed)
            start = g * N
            end = start + N
            w_g = weight[start:end]  # (N, K_packed) uint8
            ws_g = weight_scale[start:end]  # (N, K_sf) float8_e4m3fn

            # Transpose to (K_packed, N) — the layout quantize_weight_to_nvfp4 produces
            w_g_t = w_g.view(torch.float4_e2m1fn_x2).permute(1, 0).contiguous()
            ws_g_t = ws_g.permute(1, 0).contiguous()

            fp4_list.append(w_g_t)
            sf_list.append(ws_g_t)

            # Global scale: weight_scale_2
            if weight_scale_2 is not None:
                if weight_scale_2.numel() == 1:
                    gs_list.append(weight_scale_2.float().item())
                else:
                    # Per-row: take mean of this group's rows
                    gs_list.append(weight_scale_2[start:end].float().mean().item())
            else:
                gs_list.append(1.0)

        self._weight_fp4 = fp4_list
        self._weight_sf = sf_list
        self._weight_gs = gs_list

    def finalize_weights(self):
        """Process NVFP4 weights for CuTeDSL GEMM."""
        if self._weight_fp4 is None:
            raise RuntimeError("Call set_bf16_weight() before finalize_weights()")

        self._mat_b = make_b_k_major(torch.stack(self._weight_fp4))  # (groups, K_packed, N_packed)
        self._scale_b = assemble_scales_3d_side(self._weight_sf)
        self._gsb = torch.tensor(self._weight_gs, dtype=torch.float32, device=self.device)

        # Free raw weights
        self._weight_fp4 = None
        self._weight_sf = None
        self._weight_gs = None

    def _allocate_buffers(self):
        """Pre-allocate buffers at max size for cudagraph compatibility."""
        max_rows_per_group = cutedsl_ceil_div(self.max_num_tokens, 128) * 128
        total_max_rows = max_rows_per_group * self.n_local_groups

        self._padded_x_fp4_buf = torch.zeros(
            total_max_rows, self.group_in_features // 2, dtype=torch.uint8, device=self.device
        ).view(torch.float4_e2m1fn_x2)

        self._gsa_buf = torch.zeros(self.n_local_groups, dtype=torch.float32, device=self.device)
        self._expert_offsets_buf = torch.zeros(self.n_local_groups, dtype=torch.int32, device=self.device)
        self._buffers_allocated = True

    def _ensure_initialized(self):
        if self._mat_b is None:
            self.finalize_weights()
        if not self._buffers_allocated:
            self._allocate_buffers()

    def _assemble_scales_single_group(self, x_sf):
        """Assemble 2D-side activation scales for num_groups=1."""
        num_rows, num_cols = x_sf.shape
        padded_rows = cutedsl_ceil_div(num_rows, 128) * 128
        padded_cols = cutedsl_ceil_div(num_cols, 4) * 4

        buf = torch.zeros(padded_rows, padded_cols, dtype=torch.float16, device=x_sf.device).to(torch.float8_e4m3fn)
        buf[:num_rows, :num_cols] = x_sf
        swizzled_flat = pad_and_swizzle_single(buf)
        return swizzled_flat.reshape(padded_rows, padded_cols)

    def compute_activation_global_scale(self, o_sample: torch.Tensor):
        """Compute activation global scale from a warmup forward.

        Args:
            o_sample: (tokens, n_local_heads, head_dim) BF16 attention output sample
        """
        self._ensure_initialized()
        # Reshape to grouped format, then flatten to 2D for quantization
        o_grouped = o_sample.reshape(-1, self.n_local_groups, self.group_in_features)
        # We need a single gs for all groups — use the overall amax
        from dsv4.ops.quantize import (
            quantize_to_nvfp4,
        )
        o_flat = o_sample.reshape(-1, o_sample.shape[-1])  # (tokens, n_local_heads * head_dim) — not right
        # Actually, for grouped GEMM, each group's activation is (tokens, group_in_features)
        # The global scale should be computed per-group, but for simplicity use one scale
        # based on the overall amax.
        with torch.no_grad():
            _, _, gs = quantize_to_nvfp4(o_grouped.reshape(-1, self.group_in_features))
            self._activation_global_scale = gs

    def run(self, o: torch.Tensor) -> torch.Tensor:
        """Forward: BF16 attention output → NVFP4 grouped GEMM → BF16 z.

        Args:
            o: (num_tokens, n_local_heads, head_dim) BF16 — attention output
               AFTER inverse RoPE has been applied

        Returns:
            z: (num_tokens, n_local_groups, o_lora_rank) BF16
        """
        if not hasattr(self, '_runner_id'):
            self._runner_id = register_runner(self)
        return nvfp4_linear_gemm(
            o, self._runner_id, self.n_local_groups * self.o_lora_rank,
        )

    def _run_impl(self, o: torch.Tensor) -> torch.Tensor:
        """Actual implementation.

        Input o is (tokens, n_local_heads, head_dim).
        We reshape to (tokens, n_local_groups, heads_per_group * head_dim),
        then treat each group's (tokens, group_in_features) as one "expert"
        in our grouped GEMM. All tokens go to all groups.

        The grouped GEMM layout requires each group's tokens to be
        contiguous at their correct offset:
        - Group 0: rows [0, padded_T)
        - Group 1: rows [padded_T, 2*padded_T)
        - ...
        - Group G: rows [(G-1)*padded_T, G*padded_T)
        """
        self._ensure_initialized()

        num_tokens = o.shape[0]
        padded_rows_per_group = cutedsl_ceil_div(num_tokens, 128) * 128

        # Reshape: (tokens, n_local_heads, head_dim) → (tokens, n_local_groups, group_in_features)
        o_grouped = o.reshape(num_tokens, self.n_local_groups, self.group_in_features)

        # Permute to groups-first: (G, T, D)
        o_grouped = o_grouped.permute(1, 0, 2)

        # Flatten all groups into (G*T, D) for batched fused quantize — single kernel launch
        o_flat = o_grouped.reshape(self.n_local_groups * num_tokens, self.group_in_features)

        # Fused amax + quantize: zero CPU-GPU syncs.
        # Computes gsa on GPU, quantizes to NVFP4, returns GPU tensor.
        # Replaces the old path: .item() sync + Python quantize per group.
        if getattr(self, '_use_runtime_gsa', False):
            x_fp4_flat, x_sf_flat, gsa_gpu = quantize_nvfp4_gpu_fused(o_flat)
            # gsa_gpu is (G*T,) — all rows share same amax (from max over full tensor)
            # For the GEMM's global_scale_a, fill all group slots with the same gsa value
            # Use GPU-only copy: no .item(), no CPU sync
            self._gsa_buf[:1].copy_(gsa_gpu[:1])  # GPU→GPU scalar copy, no sync
            # Broadcast to all groups (all get same gsa)
            if self.n_local_groups > 1:
                self._gsa_buf[1:].copy_(self._gsa_buf[:1].expand(self.n_local_groups - 1))
        else:
            self._gsa_buf.fill_(self._activation_global_scale)
            x_fp4_flat, x_sf_flat = quantize_activation_nvfp4(
                o_flat, self._activation_global_scale
            )

        # Reshape FP4 back to (G, T, D//2) and scatter into padded buffer
        padded_x_fp4 = self._padded_x_fp4_buf
        padded_x_fp4.view(torch.uint8).zero_()

        x_fp4_grouped = x_fp4_flat.reshape(self.n_local_groups, num_tokens, self.group_in_features // 2)

        for g in range(self.n_local_groups):
            offset = g * padded_rows_per_group
            padded_x_fp4.view(torch.uint8)[offset:offset + num_tokens] = x_fp4_grouped[g].view(torch.uint8)

        # Reshape scales back to (G, T, D//16) and assemble
        x_sf_grouped = x_sf_flat.reshape(self.n_local_groups, num_tokens, self.group_in_features // 16)
        all_x_sf = [x_sf_grouped[g] for g in range(self.n_local_groups)]

        # Assemble A-side scales for all groups
        from dsv4.ops.layouts import (
            assemble_scales_2d_side,
        )
        scale_a = assemble_scales_2d_side(all_x_sf)

        # Expert offsets: cumulative [padded_T, 2*padded_T, ..., n_groups*padded_T]
        expert_offsets = self._expert_offsets_buf
        for g in range(self.n_local_groups):
            expert_offsets[g] = (g + 1) * padded_rows_per_group

        # Global scales — GPU-computed gsa already in _gsa_buf (no CPU sync)
        gsa = self._gsa_buf

        # Run grouped GEMM
        out = run_nvfp4_grouped_gemm(
            mat_a=padded_x_fp4,
            mat_b=self._mat_b,
            scale_a=scale_a,
            scale_b=self._scale_b,
            expert_offsets=expert_offsets,
            global_scale_a=gsa,
            global_scale_b=self._gsb,
        )

        # Extract real outputs and reshape
        # GEMM output has the same layout as mat_a: groups-first with padding
        z = torch.empty(num_tokens, self.n_local_groups, self.o_lora_rank,
                        dtype=torch.bfloat16, device=o.device)
        for g in range(self.n_local_groups):
            offset = g * padded_rows_per_group
            z[:, g, :] = out[offset:offset + num_tokens, :]

        return z

    def __call__(self, o: torch.Tensor) -> torch.Tensor:
        return self.run(o)
