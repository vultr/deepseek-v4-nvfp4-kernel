"""CuTeDSL NVFP4 Linear (single GEMM)

Generic NVFP4 GEMM runner for attention projections and any single
linear layer. Uses ScaledGroupedGemmKernel with num_groups=1.

CUDA-graph-compatible: all buffers pre-allocated, no CPU-GPU syncs.
"""

import torch

from dsv4.ops.quantize import (
    quantize_activation_nvfp4,
    quantize_to_nvfp4,
)
from dsv4.ops.layouts import (
    make_b_k_major,
)
from dsv4.ops.gemm_runner import (
    run_nvfp4_grouped_gemm,
)
from dsv4.kernels.gemm.grouped import (
    ceil_div as cutedsl_ceil_div,
    pad_and_swizzle_single,
)
from dsv4.ops.custom_ops import register_runner, nvfp4_linear_gemm


class Nvfp4Linear:
    """Single NVFP4 GEMM using CuTeDSL (num_groups=1).

    Handles any (K, N) weight matrix in NVFP4 format.
    Simple: quantize activation → GEMM → BF16 output.
    No SiLU, no fusion, no routing.

    CUDA-graph-compatible: all buffers pre-allocated, no CPU-GPU syncs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_num_tokens: int = 8192,
        device: str = "cuda",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.max_num_tokens = max_num_tokens
        self.device = device

        # Weights (set after construction, then call finalize_weights)
        self.fp4 = None  # list of 1 tensor
        self.sf = None   # list of 1 tensor
        self.gs = None   # list of 1 float
        self.ws2 = None  # list of 1 tensor — weight_scale_2 (scalar, folded into global_scale_b)

        # Processed weights
        self._mat_b = None
        self._scale_b = None
        self._gsb = None

        # Activation global scale
        self._activation_global_scale = 1.0 / (6.0 * 448.0)

        # Pre-allocated buffers
        self._padded_x_fp4_buf = None
        self._expert_offsets_buf = None
        self._gsa_buf = None
        self._buffers_allocated = False

    def finalize_weights(self):
        """Process weights for CuTeDSL GEMM."""
        # Convert uint8 checkpoint weights to float4_e2m1fn_x2 view
        fp4_view = [w.view(torch.float4_e2m1fn_x2) if w.dtype == torch.uint8 else w for w in self.fp4]
        # Checkpoint weight is (out_features//2, in_features//2) = (N_packed, K_packed)
        # make_b_k_major expects (E, K_packed, N_packed), so we need to permute
        stacked = torch.stack(fp4_view).permute(0, 2, 1).contiguous()  # (1, K_packed, N_packed)
        self._mat_b = make_b_k_major(stacked)
        # Checkpoint scale is (N_packed, K_sf) — already in the right row order for the
        # kernel's swizzle. Use assemble_raw_scales_2d3d_3d_side (no transpose),
        # NOT assemble_scales_3d_side (which transposes K_sf↔N).
        from dsv4.ops.layouts import assemble_raw_scales_2d3d_3d_side
        self._scale_b = assemble_raw_scales_2d3d_3d_side(self.sf)
        self._gsb = torch.tensor(self.gs, dtype=torch.float32, device=self.device)

        # Fold weight_scale_2 into global_scale_b
        # Dequant formula: w = lut[w_packed] * weight_scale * weight_scale_2
        # Production GEMM: y = (x * scale_a * gsa) @ (w * scale_b * gsb)
        # So gsb = input_scale * weight_scale_2
        if self.ws2 is not None and len(self.ws2) > 0 and self.ws2[0] is not None:
            ws2_val = self.ws2[0].float().item()
            self._gsb = self._gsb * ws2_val

        # Free raw weights
        self.fp4 = None
        self.sf = None
        self.gs = None
        self.ws2 = None

        # Eagerly JIT-compile the GEMM kernel for this (K, N) shape.
        # Uses num_groups=1 since this is a single linear layer.
        K_packed = self.in_features // 2
        N_packed = self.out_features // 2
        # warmup_compilation(1, K_packed, N_packed, self.device)  # Lazy compile on first real forward

    def _ensure_buffer_size(self, num_tokens: int):
        """Ensure the padded buffer is large enough for num_tokens."""
        needed_rows = cutedsl_ceil_div(num_tokens, 128) * 128
        if self._padded_x_fp4_buf is not None and self._padded_x_fp4_buf.shape[0] >= needed_rows:
            return  # Already big enough

        self._padded_x_fp4_buf = torch.zeros(
            needed_rows, self.in_features // 2, dtype=torch.uint8, device=self.device
        ).view(torch.float4_e2m1fn_x2)

        self._expert_offsets_buf = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._gsa_buf = torch.full((1,), self._activation_global_scale, dtype=torch.float32, device=self.device)

    def _ensure_initialized(self):
        if self._mat_b is None:
            self.finalize_weights()

    def _assemble_scales_single_group(self, x_sf):
        """Assemble 2D-side activation scales for num_groups=1."""
        num_rows, num_cols = x_sf.shape
        padded_rows = cutedsl_ceil_div(num_rows, 128) * 128
        padded_cols = cutedsl_ceil_div(num_cols, 4) * 4

        buf = torch.zeros(padded_rows, padded_cols, dtype=torch.float16, device=x_sf.device).to(torch.float8_e4m3fn)
        buf[:num_rows, :num_cols] = x_sf
        swizzled_flat = pad_and_swizzle_single(buf)
        return swizzled_flat.reshape(padded_rows, padded_cols)

    def compute_activation_global_scale(self, hidden_states_sample):
        """Compute activation global scale from a warmup forward."""
        self._ensure_initialized()
        with torch.no_grad():
            _, _, gs = quantize_to_nvfp4(hidden_states_sample)
            self._activation_global_scale = gs


    def run(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward: BF16 input → NVFP4 GEMM → BF16 output.

        Uses torch.library.custom_op (nvfp4::linear_gemm) so torch.compile
        treats this as an opaque op. The custom op calls _run_impl internally.
        """
        if not hasattr(self, '_runner_id'):
            self._runner_id = register_runner(self)
        return nvfp4_linear_gemm(
            hidden_states, self._runner_id, self.out_features,
        )

    def _run_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Actual implementation — called via custom autograd to be torch.compile-safe."""
        self._ensure_initialized()

        num_tokens = hidden_states.shape[0]
        padded_rows = cutedsl_ceil_div(num_tokens, 128) * 128

        # Ensure buffer is large enough
        self._ensure_buffer_size(num_tokens)

        # Fused amax + quantize: single kernel launch, zero CPU-GPU syncs.
        # Computes amax on GPU → derives gsa → quantizes to NVFP4.
        # gsa written to GPU buffer for downstream GEMM global_scale_a.
        #
        # This replaces the two-step path:
        #   compute_amax_gsa_gpu(hidden_states) → .item() sync
        #   quantize_nvfp4_gpu(hidden_states, gsa_float) → another kernel launch
        #
        # Old path: ~2 kernel launches + 1 .item() sync per projection.
        # New path: 1 kernel launch + 0 .item() syncs per projection.
        # Total across 61 layers: ~486 .item() syncs eliminated.
        if getattr(self, '_use_runtime_gsa', False):
            from dsv4.ops.quantize import quantize_nvfp4_gpu_fused
            x_fp4, x_sf, gsa_gpu = quantize_nvfp4_gpu_fused(hidden_states)
            self._gsa_buf.copy_(gsa_gpu[:1].reshape(1))  # GPU → GPU, no sync
        else:
            # P2 FIX: No per-call fill_(). The _gsa_buf already has the correct
            # value — set either during initialization (via _ensure_buffer_size)
            # or by the first GPU compute when _use_runtime_gsa was True.
            # Old path: self._gsa_buf.fill_(self._activation_global_scale)
            #   — H2D transfer every call (~5µs each × 244 calls = ~1.2ms/token).
            # New path: zero H2D transfers on the hot path.
            from dsv4.ops.quantize import quantize_nvfp4_gpu
            x_fp4, x_sf = quantize_nvfp4_gpu(hidden_states, self._activation_global_scale)

        # Scatter x_fp4 into padded buffer
        padded_x_fp4 = self._padded_x_fp4_buf
        padded_x_fp4.view(torch.uint8).zero_()
        padded_x_fp4.view(torch.uint8)[:x_fp4.shape[0]] = x_fp4.view(torch.uint8)

        # Assemble A-side scales
        scale_a = self._assemble_scales_single_group(x_sf)

        # Expert offsets: [padded_rows] for 1 group
        expert_offsets = self._expert_offsets_buf
        expert_offsets.fill_(padded_rows)

        # Global scales — GPU-computed gsa already in _gsa_buf (no CPU sync)
        gsa = self._gsa_buf

        # Run GEMM
        out = run_nvfp4_grouped_gemm(
            mat_a=padded_x_fp4,
            mat_b=self._mat_b,
            scale_a=scale_a,
            scale_b=self._scale_b,
            expert_offsets=expert_offsets,
            global_scale_a=gsa,
            global_scale_b=self._gsb,
        )

        return out[:num_tokens]

    def run_from_quantized(self, quant: 'QuantizedActivation') -> torch.Tensor:
        """Run GEMM with pre-quantized activation (skip quantize step).
        
        Used when the input has already been quantized by a fused
        RMSNorm+quantize kernel. Saves 2 kernel launches per call.
        
        Args:
            quant: QuantizedActivation with x_fp4, x_sf, gsa
        """
        from dsv4.ops.quantize import QuantizedActivation
        assert isinstance(quant, QuantizedActivation)
        
        self._ensure_initialized()
        num_tokens = quant.num_tokens
        padded_rows = cutedsl_ceil_div(num_tokens, 128) * 128
        self._ensure_buffer_size(num_tokens)

        # Scatter pre-quantized x_fp4 into padded buffer
        padded_x_fp4 = self._padded_x_fp4_buf
        padded_x_fp4.view(torch.uint8).zero_()
        padded_x_fp4.view(torch.uint8)[:quant.x_fp4.shape[0]] = quant.x_fp4.view(torch.uint8)

        # Assemble A-side scales from pre-quantized sf
        scale_a = self._assemble_scales_single_group(quant.x_sf)

        # Expert offsets
        expert_offsets = self._expert_offsets_buf
        expert_offsets.fill_(padded_rows)

        # Global scales — use the per-row gsa from the fused kernel
        # Reshape to (1,) if scalar, or use per-row (M,) broadcast
        gsa = quant.gsa[:1].reshape(1) if quant.gsa.shape[0] == 1 else quant.gsa[:num_tokens]
        if gsa.shape != self._gsa_buf.shape:
            self._gsa_buf = gsa.contiguous()
        else:
            self._gsa_buf.copy_(gsa)

        # Run GEMM
        out = run_nvfp4_grouped_gemm(
            mat_a=padded_x_fp4,
            mat_b=self._mat_b,
            scale_a=scale_a,
            scale_b=self._scale_b,
            expert_offsets=expert_offsets,
            global_scale_a=self._gsa_buf,
            global_scale_b=self._gsb,
        )

        return out[:num_tokens]

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.run(hidden_states)
