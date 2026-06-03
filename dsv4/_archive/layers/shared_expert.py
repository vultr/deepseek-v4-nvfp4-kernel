"""CuTeDSL Shared Expert Pipeline

NVFP4 inference for DeepSeek V4 shared experts.
Uses ScaledGroupedGemmKernel with num_groups=1.

Pipeline:
  1. Quantize activation: BF16 → NVFP4 (using warmup gs)
  2. L1 GEMM: NVFP4_act × NVFP4_weight(gate_up) → BF16
  3. SiLU(gate) * up → BF16
  4. Re-quantize: BF16 → NVFP4 (using warmup gs)
  5. L2 GEMM: NVFP4_act × NVFP4_weight(down) → BF16

Unlike MoE, there's no routing, no scatter, no expert offsets.
All tokens go through the same expert (the shared expert).
Scale assembly is just: quantize activation → pad to 128-row alignment → Blackwell swizzle.

CUDA-graph-compatible: all buffers pre-allocated, no CPU-GPU syncs,
no dynamic shapes. Padding rows are zeros that contribute nothing to GEMM output.
"""

import torch

from dsv4.ops.quantize import (
    quantize_activation_nvfp4,
    quantize_to_nvfp4,
)
from dsv4.ops.layouts import (
    make_b_k_major,
    interleave_l1_weights,
    deinterleave_l1_weights,
)
from dsv4.ops.gemm_runner import (
    run_nvfp4_grouped_gemm,
    run_fused_swiglu_grouped_gemm,
)
from dsv4.ops.quantize import quantize_nvfp4_gpu_fused
from dsv4.kernels.gemm.grouped import (
    ceil_div as cutedsl_ceil_div,
    pad_and_swizzle_single,
)


class _SharedExpertApply(torch.autograd.Function):
    """Custom autograd function to make CuTeDSL runner opaque to torch.compile."""
    @staticmethod
    def forward(ctx, runner, hidden_states):
        return runner._run_impl(hidden_states)


class Nvfp4SharedExpert:
    """NVFP4 shared expert runner using CuTeDSL GEMM (num_groups=1).

    CUDA-graph-compatible: all buffers pre-allocated, no CPU-GPU syncs.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        max_num_tokens: int = 8192,
        device: str = "cuda",
        swiglu_limit: float = 10.0,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = max_num_tokens
        self.device = device
        self.swiglu_limit = swiglu_limit
        self._fused_swiglu = False  # Set via set_fused_swiglu()

        # Weights (set after construction, then call finalize_weights)
        self.l1_fp4 = None
        self.l1_sf = None
        self.l1_gs = None
        self.l2_fp4 = None
        self.l2_sf = None
        self.l2_gs = None
        # weight_scale_2 per layer (scalar, folded into global_scale_b in finalize_weights)
        self.l1_ws2 = None
        self.l2_ws2 = None

        # Processed weights (set by finalize_weights)
        self._l1_mat_b = None
        self._l2_mat_b = None
        self._l1_scale_b = None
        self._l2_scale_b = None
        self._l1_gsb = None
        self._l2_gsb = None

        # Activation global scales (set by compute_activation_global_scales)
        self._l1_activation_global_scale = 1.0 / (6.0 * 448.0)
        self._l2_activation_global_scale = 1.0 / (6.0 * 448.0)

        # Pre-allocated cudagraph buffers (set in _allocate_buffers)
        self._padded_x_fp4_buf_l1 = None
        self._padded_x_sf_buf_l1 = None
        self._padded_x_fp4_buf_l2 = None
        self._padded_x_sf_buf_l2 = None
        self._l1_gsa_buf = None
        self._l2_gsa_buf = None
        self._expert_offsets_buf = None
        self._buffers_allocated = False

    def set_swiglu_limit(self, limit: float):
        self.swiglu_limit = limit

    def set_fused_swiglu(self, enabled: bool):
        """Enable fused L1 GEMM + SwiGLU kernel (1-group variant of MoE fused kernel)."""
        self._fused_swiglu = enabled

    def finalize_weights(self):
        """Process weights for CuTeDSL GEMM. Must be called after setting l1/l2 weights."""
        # Convert uint8 checkpoint weights to float4_e2m1fn_x2 view
        l1_view = [w.view(torch.float4_e2m1fn_x2) if w.dtype == torch.uint8 else w for w in self.l1_fp4]
        l2_view = [w.view(torch.float4_e2m1fn_x2) if w.dtype == torch.uint8 else w for w in self.l2_fp4]
        # Checkpoint weight is (N_packed, K_packed), make_b_k_major expects (E, K_packed, N_packed)
        l1_stacked = torch.stack(l1_view).permute(0, 2, 1).contiguous()
        l2_stacked = torch.stack(l2_view).permute(0, 2, 1).contiguous()
        # P1: Interleave L1 gate/up weights for fused SwiGLU kernel compatibility.
        # The fused kernel's SwiGLU epilogue expects granularity-8 interleaved gate/up.
        # The unfused path (if _fused_swiglu=False) deinterleaves the GEMM output before splitting.
        if self._fused_swiglu:
            l1_stacked = interleave_l1_weights(l1_stacked, granularity_bf16=8)
        # Stack weights and convert to K-major
        self._l1_mat_b = make_b_k_major(l1_stacked)  # (1, K_packed, N_packed)
        self._l2_mat_b = make_b_k_major(l2_stacked)
        # Checkpoint scale is (N_packed, K_sf) — use assemble_raw_scales_2d3d_3d_side
        from dsv4.ops.layouts import assemble_raw_scales_2d3d_3d_side
        self._l1_scale_b = assemble_raw_scales_2d3d_3d_side(self.l1_sf)
        self._l2_scale_b = assemble_raw_scales_2d3d_3d_side(self.l2_sf)
        self._l1_gsb = torch.tensor(self.l1_gs, dtype=torch.float32, device=self.device)
        self._l2_gsb = torch.tensor(self.l2_gs, dtype=torch.float32, device=self.device)

        # Fold weight_scale_2 into global_scale_b
        # gsb = input_scale * weight_scale_2
        if self.l1_ws2 is not None:
            for i, ws2 in enumerate(self.l1_ws2):
                if ws2 is not None:
                    self._l1_gsb[i] *= ws2.float().item()
        if self.l2_ws2 is not None:
            for i, ws2 in enumerate(self.l2_ws2):
                if ws2 is not None:
                    self._l2_gsb[i] *= ws2.float().item()

        # Free raw weights
        self.l1_fp4 = None
        self.l1_sf = None
        self.l1_gs = None
        self.l2_fp4 = None
        self.l2_sf = None
        self.l2_gs = None
        self.l1_ws2 = None
        self.l2_ws2 = None

    def _allocate_buffers(self):
        """Pre-allocate all buffers at max size for cudagraph compatibility."""
        max_rows = cutedsl_ceil_div(self.max_num_tokens, 128) * 128  # pad to 128

        # L1: hidden_size packed, L2: intermediate_size packed
        self._padded_x_fp4_buf_l1 = torch.zeros(
            max_rows, self.hidden_size // 2, dtype=torch.uint8, device=self.device
        ).view(torch.float4_e2m1fn_x2)
        self._padded_x_fp4_buf_l2 = torch.zeros(
            max_rows, self.intermediate_size // 2, dtype=torch.uint8, device=self.device
        ).view(torch.float4_e2m1fn_x2)

        # Padded scale buffers (need same padded dimensions as pad_and_swizzle_single produces)
        K_sf_l1 = cutedsl_ceil_div(self.hidden_size, 16)
        padded_cols_l1 = cutedsl_ceil_div(K_sf_l1, 4) * 4
        K_sf_l2 = cutedsl_ceil_div(self.intermediate_size, 16)
        padded_cols_l2 = cutedsl_ceil_div(K_sf_l2, 4) * 4
        self._padded_x_sf_buf_l1 = torch.zeros(
            max_rows, padded_cols_l1, dtype=torch.float16, device=self.device
        ).to(torch.float8_e4m3fn)
        self._padded_x_sf_buf_l2 = torch.zeros(
            max_rows, padded_cols_l2, dtype=torch.float16, device=self.device
        ).to(torch.float8_e4m3fn)

        # Global scale buffers
        self._l1_gsa_buf = torch.zeros(1, dtype=torch.float32, device=self.device)
        self._l2_gsa_buf = torch.zeros(1, dtype=torch.float32, device=self.device)

        # Expert offsets for num_groups=1: just [num_tokens_padded]
        # The GEMM expects expert_offsets as (num_experts,) cumulative offsets
        # For 1 expert: offsets = [num_tokens] (just one element)
        self._expert_offsets_buf = torch.zeros(1, dtype=torch.int32, device=self.device)

        self._buffers_allocated = True

    def _ensure_initialized(self):
        """Lazily initialize stacked weights and buffers."""
        if self._l1_mat_b is None:
            self.finalize_weights()
        if not self._buffers_allocated:
            self._allocate_buffers()

    def _assemble_scales_single_group(self, x_sf, num_tokens, padded_x_sf_buf):
        """Assemble 2D-side activation scales for num_groups=1.

        For a single group, scale assembly is just:
        1. Copy x_sf into a correctly-sized buffer (padded to 128 rows, 4 cols)
        2. Apply pad_and_swizzle_single (Blackwell swizzle)
        3. Reshape back to 2D (kernel expects 2D scale_a)

        The padded buffer must be sized exactly for 128-aligned num_tokens,
        NOT the max_num_tokens buffer (which would be way too large).
        """
        num_rows, num_cols = x_sf.shape
        padded_rows = cutedsl_ceil_div(num_rows, 128) * 128
        padded_cols = cutedsl_ceil_div(num_cols, 4) * 4

        # Use a temp buffer sized for this exact token count
        buf = torch.zeros(padded_rows, padded_cols, dtype=torch.float16, device=x_sf.device).to(torch.float8_e4m3fn)
        buf[:num_rows, :num_cols] = x_sf
        swizzled_flat = pad_and_swizzle_single(buf)
        return swizzled_flat.reshape(padded_rows, padded_cols)

    def compute_activation_global_scales(self, hidden_states_sample):
        """Compute activation global scales from a warmup forward pass.

        Called BEFORE cudagraph capture. Uses quantize_to_nvfp4 to get
        the exact global_scale from the data, then runs L1 to compute
        L2 gs from actual SiLU(gate)*up output.
        """
        self._ensure_initialized()

        with torch.no_grad():
            # L1: exact gs from quantize_to_nvfp4
            _, _, l1_gs = quantize_to_nvfp4(hidden_states_sample)
            self._l1_activation_global_scale = l1_gs

            # Run L1 GEMM to get intermediate for L2 gs
            num_tokens = hidden_states_sample.shape[0]
            l1_out = self._run_l1(hidden_states_sample)
            if l1_out is not None and not torch.isnan(l1_out).any():
                gate = l1_out[:, :self.intermediate_size]
                up = l1_out[:, self.intermediate_size:]
                if self.swiglu_limit is not None:
                    gate = gate.clamp(max=self.swiglu_limit)
                    up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
                activated = torch.nn.functional.silu(gate) * up
                _, _, l2_gs = quantize_to_nvfp4(activated)
                self._l2_activation_global_scale = l2_gs



    def _run_l1_fused(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fused L1 GEMM + SwiGLU + clamp — single kernel launch (1-group variant of MoE fused kernel)."""
        num_tokens = hidden_states.shape[0]
        x_bf16 = hidden_states.reshape(num_tokens, self.hidden_size)

        # Quantize activation to NVFP4 (fused amax + quantize)
        if getattr(self, '_use_runtime_gsa', False):
            from dsv4.ops.quantize import quantize_nvfp4_gpu_fused
            x_fp4, x_sf, gsa_l1_gpu = quantize_nvfp4_gpu_fused(x_bf16)
            self._l1_gsa_buf.copy_(gsa_l1_gpu[:1].reshape(1))  # GPU → GPU
        else:
            from dsv4.ops.quantize import quantize_activation_nvfp4
            x_fp4, x_sf = quantize_activation_nvfp4(x_bf16, self._l1_activation_global_scale)

        # Padded buffer setup for 1-group GEMM
        padded_rows = cutedsl_ceil_div(num_tokens, 128) * 128
        padded_x_fp4 = self._padded_x_fp4_buf_l1
        padded_x_fp4.view(torch.uint8).zero_()
        padded_x_fp4.view(torch.uint8)[:num_tokens] = x_fp4.view(torch.uint8)

        # Assemble A-side scales
        scale_a = self._assemble_scales_single_group(x_sf, num_tokens, self._padded_x_sf_buf_l1)

        # Expert offsets: [padded_rows] for 1 group (int32, pre-allocated)
        expert_offsets = self._expert_offsets_buf
        expert_offsets.fill_(padded_rows)

        # Global scales — GPU-computed gsa already in _l1_gsa_buf (no CPU sync)
        gsa = self._l1_gsa_buf

        # Run fused GEMM + SwiGLU
        l1_out = run_fused_swiglu_grouped_gemm(
            mat_a=padded_x_fp4,
            mat_b=self._l1_mat_b,
            scale_a=scale_a,
            scale_b=self._l1_scale_b,
            expert_offsets=expert_offsets,
            global_scale_a=gsa,
            global_scale_b=self._l1_gsb,
            swiglu_limit=self.swiglu_limit if self.swiglu_limit is not None else 0.0,
        )
        l1_out_real = l1_out[:num_tokens]  # (num_tokens, 2*intermediate) BF16, interleaved [silu(gate), silu(gate)*up]
        # Deinterleave to separate gate and up, then take up half (SwiGLU result)
        l1_deil = deinterleave_l1_weights(l1_out_real.unsqueeze(0).contiguous())[0]  # (num_tokens, 2*intermediate) deinterleaved
        intermediate = l1_deil[:, self.intermediate_size:]  # up half = silu(gate)*up
        return intermediate  # (num_tokens, intermediate_size) BF16

    def _run_l1(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """L1 GEMM: activation × gate_up_weight → BF16."""
        num_tokens = hidden_states.shape[0]
        padded_rows = cutedsl_ceil_div(num_tokens, 128) * 128

        # Fused amax + quantize: zero CPU syncs.
        if getattr(self, '_use_runtime_gsa', False):
            from dsv4.ops.quantize import quantize_nvfp4_gpu_fused
            x_fp4, x_sf, gsa_l1_gpu = quantize_nvfp4_gpu_fused(hidden_states)
            self._l1_gsa_buf.copy_(gsa_l1_gpu[:1].reshape(1))  # GPU → GPU, no sync
        else:
            x_fp4, x_sf = quantize_activation_nvfp4(
                hidden_states, self._l1_activation_global_scale
            )

        # Scatter x_fp4 into padded buffer
        padded_x_fp4 = self._padded_x_fp4_buf_l1
        padded_x_fp4.view(torch.uint8).zero_()
        padded_x_fp4.view(torch.uint8)[:num_tokens] = x_fp4.view(torch.uint8)

        # Assemble A-side scales
        scale_a = self._assemble_scales_single_group(x_sf, num_tokens, self._padded_x_sf_buf_l1)

        # Expert offsets: [padded_rows] for 1 group
        expert_offsets = self._expert_offsets_buf
        expert_offsets.fill_(padded_rows)

        # Global scales — GPU-computed gsa already in _l1_gsa_buf (no CPU sync)
        gsa = self._l1_gsa_buf

        # Run GEMM
        out = run_nvfp4_grouped_gemm(
            mat_a=padded_x_fp4,
            mat_b=self._l1_mat_b,
            scale_a=scale_a,
            scale_b=self._l1_scale_b,
            expert_offsets=expert_offsets,
            global_scale_a=gsa,
            global_scale_b=self._l1_gsb,
        )

        # Extract real token outputs
        return out[:num_tokens]

    def _run_l2(self, intermediate: torch.Tensor) -> torch.Tensor:
        """L2 GEMM: intermediate × down_weight → BF16."""
        num_tokens = intermediate.shape[0]
        padded_rows = cutedsl_ceil_div(num_tokens, 128) * 128

        # Fused amax + quantize: zero CPU syncs.
        if getattr(self, '_use_runtime_gsa', False):
            from dsv4.ops.quantize import quantize_nvfp4_gpu_fused
            x_fp4, x_sf, gsa_l2_gpu = quantize_nvfp4_gpu_fused(intermediate)
            self._l2_gsa_buf.copy_(gsa_l2_gpu[:1].reshape(1))  # GPU → GPU, no sync
        else:
            x_fp4, x_sf = quantize_activation_nvfp4(
                intermediate, self._l2_activation_global_scale
            )

        # Scatter into padded buffer
        padded_x_fp4 = self._padded_x_fp4_buf_l2
        padded_x_fp4.view(torch.uint8).zero_()
        padded_x_fp4.view(torch.uint8)[:num_tokens] = x_fp4.view(torch.uint8)

        # Assemble A-side scales
        scale_a = self._assemble_scales_single_group(x_sf, num_tokens, self._padded_x_sf_buf_l2)

        # Expert offsets
        expert_offsets = self._expert_offsets_buf
        expert_offsets.fill_(padded_rows)

        # Global scales — GPU-computed gsa already in _l2_gsa_buf (no CPU sync)
        gsa = self._l2_gsa_buf

        # Run GEMM
        out = run_nvfp4_grouped_gemm(
            mat_a=padded_x_fp4,
            mat_b=self._l2_mat_b,
            scale_a=scale_a,
            scale_b=self._l2_scale_b,
            expert_offsets=expert_offsets,
            global_scale_a=gsa,
            global_scale_b=self._l2_gsb,
        )

        return out[:num_tokens]

    def run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Full shared expert forward: L1 → SiLU → L2 → output."""
        return _SharedExpertApply.apply(self, hidden_states)

    def _run_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Actual implementation — called via custom autograd to be torch.compile-safe."""
        self._ensure_initialized()

        if self._fused_swiglu:
            # P1: Fused L1 GEMM + SwiGLU + clamp in one kernel launch
            intermediate = self._run_l1_fused(hidden_states)
        else:
            l1_out = self._run_l1(hidden_states)
            if l1_out.shape[1] < 2 * self.intermediate_size:
                print(f"  WARNING: l1_out shape {l1_out.shape} < expected (N, {2*self.intermediate_size})", flush=True)

            gate = l1_out[:, :self.intermediate_size]
            up = l1_out[:, self.intermediate_size:]
            if torch.isnan(l1_out).any():
                print(f"  SE L1 NaN: l1_out nan at {torch.isnan(l1_out).sum().item()} / {l1_out.numel()} positions, shape={l1_out.shape}", flush=True)
            if torch.isnan(gate).any() or torch.isnan(up).any():
                print(f"  SE gate nan={torch.isnan(gate).any().item()} up nan={torch.isnan(up).any().item()}", flush=True)
            if self.swiglu_limit is not None:
                gate = gate.clamp(max=self.swiglu_limit)
                up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            intermediate = torch.nn.functional.silu(gate) * up

        output = self._run_l2(intermediate)
        return output
