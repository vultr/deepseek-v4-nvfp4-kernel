"""
vLLM integration for the CuTeDSL NVFP4 MoE kernel.

CUDA-graph-compatible design:
- All intermediate buffers pre-allocated at max_num_tokens * top_k size
- No .item(), .tolist(), .cpu() — zero CPU-GPU syncs
- No dynamic slicing with GPU scalars — always operate on full pre-allocated buffers
- Extra slots (beyond real tokens) are zero and contribute nothing to output
- Fixed-shape tensors throughout the forward pass

vLLM cudagraph captures at fixed token budgets (1,2,4,8,...,8192).
During capture, num_tokens equals the budget — all shapes are fixed.
During replay, inputs are padded to the budget size. Our runner always
processes max_slots = budget * top_k rows; padding rows are zeros.
"""
import torch

from dsv4.ops.quantize import (
    quantize_activation_nvfp4,
    quantize_weight_to_nvfp4,
    quantize_to_nvfp4,
    quantize_nvfp4_gpu,
    deinterleave_quantize_nvfp4_cuda,
)
from dsv4.ops.layouts import (
    make_b_k_major,
    assemble_scales_3d_side,
    interleave_l1_weights,
    deinterleave_l1_weights,
)
from dsv4.ops.gemm_runner import (
    run_nvfp4_grouped_gemm,
    run_fused_swiglu_grouped_gemm,
    warmup_fused_swiglu_compilation,
)
from dsv4.ops.layouts import (
    ceil_div as cutedsl_ceil_div,
    pad_and_swizzle_single,
)
from dsv4.ops.custom_ops import register_runner, nvfp4_moe_gemm


class Nvfp4MoE:
    """Manages NVFP4 MoE execution via the CuTeDSL kernel.
    
    CUDA-graph-compatible: all buffers pre-allocated, no CPU-GPU syncs,
    no dynamic shapes. Always computes at max_num_tokens * top_k capacity.
    """
    
    def __init__(self, num_experts, hidden_size, intermediate_size,
                 max_num_tokens=8192, top_k=8, device="cuda",
                 experts_start_idx=0):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = max_num_tokens
        self.top_k = top_k
        self.device = device
        self.experts_start_idx = experts_start_idx
        self._swiglu_limit = None  # Set via set_swiglu_limit()
        self._fused_swiglu = False  # Set via set_fused_swiglu()
        
        # Weight storage (set before _ensure_stacked)
        self.l1_fp4 = None
        self.l1_sf = None
        self.l1_gs = None
        self.l2_fp4 = None
        self.l2_sf = None
        self.l2_gs = None
        
        # Stacked weight tensors (set in _ensure_stacked)
        self._l1_mat_b = None
        self._l2_mat_b = None
        self._l1_scale_b = None
        self._l2_scale_b = None
        self._l1_gsb = None
        self._l2_gsb = None
        
        # Default: 1/2688 ≈ 0.000372 (amax=1 → gs=1/2688)
        # Overridden in finalize_weights with checkpoint input_scale or warmup value
        self._l1_activation_global_scale = 1.0 / (6.0 * 448.0)
        self._l2_activation_global_scale = 1.0 / (6.0 * 448.0)
        
        # Pre-allocated cudagraph buffers (set in _allocate_buffers)
        self._token_indices = None
        self._expert_offsets_buf = None
        self._per_expert_scale_bufs_l1 = None
        self._per_expert_scale_bufs_l2 = None
        self._padded_x_sf_buf_l1 = None
        self._padded_x_sf_buf_l2 = None
        self._l1_gsa_buf = None
        self._l2_gsa_buf = None
        self._output_buf = None
        self._row_indices_buf = None
        self._padded_hidden_buf = None
        self._padded_activated_buf = None  # unused, using shared
        self._padded_expert_offsets_buf = None
        self._max_chunks_per_expert = cutedsl_ceil_div(
            self.max_num_tokens * self.top_k, self.num_experts * 128
        )
        self._buffers_allocated = False

    def set_swiglu_limit(self, limit: float | None):
        """Set the swiglu_limit for activation clamping."""
        self._swiglu_limit = limit

    def set_fused_swiglu(self, enabled: bool):
        """Enable fused L1 GEMM + SwiGLU kernel (saves 240+ BF16 kernel launches per token)."""
        self._fused_swiglu = enabled

    def _fill_token_indices(self):
        """Fill _token_indices with [0,0,..0, 1,1,..1, ...] (each token repeated top_k times).
        
        Builds on CPU first, then copies to GPU, to ensure correctness
        regardless of CuTeDSL JIT GPU memory corruption.
        """
        src = torch.arange(self.max_num_tokens, dtype=torch.int32)
        cpu_indices = src.unsqueeze(1).expand(-1, self.top_k).contiguous().view(-1)
        self._token_indices.copy_(cpu_indices)

    def _allocate_buffers(self):
        """Pre-allocate scale buffers at max size for cudagraph compatibility."""
        # Per-expert scale buffers: separate L1/L2 since K_sf differs
        K_sf_l1 = cutedsl_ceil_div(self.hidden_size, 16)
        padded_cols_l1 = cutedsl_ceil_div(K_sf_l1, 4) * 4
        K_sf_l2 = cutedsl_ceil_div(self.intermediate_size, 16)
        padded_cols_l2 = cutedsl_ceil_div(K_sf_l2, 4) * 4
        
        self._per_expert_scale_bufs_l1 = [
            torch.zeros(128, padded_cols_l1, dtype=torch.float16, device=self.device).to(torch.float8_e4m3fn)
            for _ in range(self.num_experts)
        ]
        self._per_expert_scale_bufs_l2 = [
            torch.zeros(128, padded_cols_l2, dtype=torch.float16, device=self.device).to(torch.float8_e4m3fn)
            for _ in range(self.num_experts)
        ]
        
        # Initialize shared buffers dict (if not already)
        device_key = str(self.device)
        if not hasattr(Nvfp4MoE, '_shared_padded_bufs'):
            Nvfp4MoE._shared_padded_bufs = {}
        if device_key not in Nvfp4MoE._shared_padded_bufs:
            Nvfp4MoE._shared_padded_bufs[device_key] = {}
        
        # Padded x_sf buffers: SHARED across all runners (not per-layer)
        max_sf_rows = self.num_experts * self._max_chunks_per_expert * 128
        if 'xsf_l1' not in Nvfp4MoE._shared_padded_bufs[device_key]:
            Nvfp4MoE._shared_padded_bufs[device_key].update({
                'xsf_l1': torch.zeros(
                    max_sf_rows, padded_cols_l1, dtype=torch.float16, device=self.device
                ).to(torch.float8_e4m3fn),
                'xsf_l2': torch.zeros(
                    max_sf_rows, padded_cols_l2, dtype=torch.float16, device=self.device
                ).to(torch.float8_e4m3fn),
                'output': torch.zeros(
                    self.max_num_tokens, self.hidden_size, dtype=torch.bfloat16, device=self.device
                ),
            })
        self._padded_x_sf_buf_l1 = Nvfp4MoE._shared_padded_bufs[device_key]['xsf_l1']
        self._padded_x_sf_buf_l2 = Nvfp4MoE._shared_padded_bufs[device_key]['xsf_l2']
        self._output_buf = Nvfp4MoE._shared_padded_bufs[device_key]['output']
        
        # Pre-allocated global_scale_a buffers (filled via .fill_(), no torch.full during capture)
        self._l1_gsa_buf = torch.zeros(self.num_experts, dtype=torch.float32, device=self.device)
        self._l2_gsa_buf = torch.zeros(self.num_experts, dtype=torch.float32, device=self.device)
        
        # Row indices for scale assembly (max_num_tokens * top_k slots)
        self._row_indices_buf = torch.arange(
            self.max_num_tokens * self.top_k, device=self.device
        )
        
        # Padded hidden/activated: SHARED across all runners (not per-layer)
        max_rows_per_expert = self._max_chunks_per_expert * 128
        padded_max_slots = self.num_experts * max_rows_per_expert
        if 'hidden' not in Nvfp4MoE._shared_padded_bufs[device_key]:
            Nvfp4MoE._shared_padded_bufs[device_key].update({
                'hidden': torch.zeros(
                    padded_max_slots, self.hidden_size, dtype=torch.bfloat16, device=self.device
                ),
                'hidden_fp4': torch.zeros(
                    padded_max_slots, self.hidden_size // 2, dtype=torch.uint8, device=self.device
                ).view(torch.float4_e2m1fn_x2),
                'activated': torch.zeros(
                    padded_max_slots, self.intermediate_size, dtype=torch.bfloat16, device=self.device
                ),
                'activated_fp4': torch.zeros(
                    padded_max_slots, self.intermediate_size // 2, dtype=torch.uint8, device=self.device
                ).view(torch.float4_e2m1fn_x2),
            })
        self._shared_bufs = Nvfp4MoE._shared_padded_bufs[device_key]
        
        # Padded expert offsets buffer: [0, max_rows, 2*max_rows, ...] (fixed)
        self._padded_expert_offsets_buf = torch.zeros(
            self.num_experts + 1, dtype=torch.int32, device=self.device
        )
        max_rows_per_expert = self._max_chunks_per_expert * 128
        self._padded_expert_offsets_buf[1:] = torch.arange(
            1, self.num_experts + 1, dtype=torch.int32, device=self.device
        ) * max_rows_per_expert
        
        self._buffers_allocated = True

    def _ensure_stacked(self):
        if self._l1_mat_b is not None:
            return
        
        # Convert weights to kernel format
        if hasattr(self, 'l1_fp4_stacked') and self.l1_fp4_stacked is not None:
            # Fast path: pre-stacked 3D tensors in checkpoint format (E, N, K)
            # Permute to (E, K, N) then make K-major
            l1_fp4_ekn = self.l1_fp4_stacked.permute(0, 2, 1).contiguous()
            l2_fp4_ekn = self.l2_fp4_stacked.permute(0, 2, 1).contiguous()
            # Interleave L1 gate/up weights at granularity 4 BF16.
            # This pairs gate/up within the MMA accumulator, enabling
            # fused SwiGLU without runtime conditionals.
            l1_fp4_ekn = interleave_l1_weights(l1_fp4_ekn)
            # Convert uint8 checkpoint weights to float4_e2m1fn_x2 view
            if l1_fp4_ekn.dtype == torch.uint8:
                l1_fp4_ekn = l1_fp4_ekn.view(torch.float4_e2m1fn_x2)
            if l2_fp4_ekn.dtype == torch.uint8:
                l2_fp4_ekn = l2_fp4_ekn.view(torch.float4_e2m1fn_x2)
            # Free stacked checkpoints before make_b_k_major (saves one copy)
            self.l1_fp4_stacked = None
            self.l2_fp4_stacked = None
            torch.cuda.empty_cache()
            
            self._l1_mat_b = make_b_k_major(l1_fp4_ekn)
            self._l2_mat_b = make_b_k_major(l2_fp4_ekn)
            del l1_fp4_ekn, l2_fp4_ekn
            torch.cuda.empty_cache()
            
            # Scales: checkpoint is (E, N, K_sf) — the kernel expects (N, K_sf)
            # per expert for swizzle. Split into views (no copy), then assemble.
            l1_sf_list = [self.l1_sf_stacked[i] for i in range(self.num_experts)]
            l2_sf_list = [self.l2_sf_stacked[i] for i in range(self.num_experts)]
            self.l1_sf_stacked = None
            self.l2_sf_stacked = None
            torch.cuda.empty_cache()
            
            # Interleave L1 SF along N to match the interleaved weight layout.
            # SF per expert from checkpoint is (N, K_sf). Interleave along N.
            # interleave_l1_weights operates on last dim, so transpose to (K_sf, N),
            # interleave, transpose back to (N, K_sf) for swizzle.
            l1_sf_il = []
            for sf_nk in l1_sf_list:
                sf_kn = sf_nk.T.contiguous().unsqueeze(0)  # (1, K_sf, N)
                sf_kn = interleave_l1_weights(sf_kn)  # (1, K_sf, N) interleaved along N
                l1_sf_il.append(sf_kn[0].T.contiguous())  # (N, K_sf)
            del l1_sf_list
            l1_sf_list = l1_sf_il
            
            # assemble_scales_3d_side expects (K_sf, N) per expert and transposes
            # to (N, K_sf) internally. But our scales are already (N, K_sf) from
            # the checkpoint! Skip the transpose by calling the assembly directly.
            from dsv4.ops.layouts import (
                assemble_raw_scales_2d3d_3d_side,
            )
            self._l1_scale_b = assemble_raw_scales_2d3d_3d_side(l1_sf_list)
            self._l2_scale_b = assemble_raw_scales_2d3d_3d_side(l2_sf_list)
            del l1_sf_list, l2_sf_list
        else:
            # Legacy path: per-expert lists
            l1_stacked = torch.stack(self.l1_fp4)  # (E, K, N)
            l1_stacked = interleave_l1_weights(l1_stacked)  # interleave gate/up
            if l1_stacked.dtype == torch.uint8:
                l1_stacked = l1_stacked.view(torch.float4_e2m1fn_x2)
            l2_stacked = torch.stack(self.l2_fp4)
            if l2_stacked.dtype == torch.uint8:
                l2_stacked = l2_stacked.view(torch.float4_e2m1fn_x2)
            self._l1_mat_b = make_b_k_major(l1_stacked)
            self._l2_mat_b = make_b_k_major(l2_stacked)
            # Interleave L1 SF to match weight interleave
            # SF from quantize_weight_to_nvfp4 is (K_sf, N). Interleave along N,
            # then transpose to (N, K_sf) for swizzle via assemble_scales_3d_side.
            l1_sf_il = []
            for sf in self.l1_sf:
                sf_ekn = sf.unsqueeze(0)  # (1, K_sf, N)
                sf_ekn = interleave_l1_weights(sf_ekn)  # interleaved along N
                l1_sf_il.append(sf_ekn[0])  # (K_sf, N)
            self._l1_scale_b = assemble_scales_3d_side(l1_sf_il)
            self._l2_scale_b = assemble_scales_3d_side(self.l2_sf)
            del l1_stacked, l1_sf_il
            self.l1_fp4 = None
            self.l1_sf = None
            self.l2_fp4 = None
            self.l2_sf = None
        
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

        self.l1_gs = None
        self.l2_gs = None
        self.l1_ws2 = None
        self.l2_ws2 = None
        
        # Allocate buffers and eagerly warmup JIT compilation.
        # cute.compile does NOT corrupt GPU memory (verified 2026-05-20).
        # We warmup eagerly here to ensure compilation happens before
        # the model's first forward pass, not during it.
        self._token_indices = torch.zeros(
            self.max_num_tokens * self.top_k, dtype=torch.int32, device=self.device
        )
        self._fill_token_indices()
        # No _needs_token_refill: cute.compile does NOT corrupt GPU memory.
        # The original corruption was a misdiagnosis (see bridge.py cache docs).
        
        # Eagerly JIT-compile GEMM kernels for L1 and L2 shapes.
        # This triggers cute.compile once per shape, caching the compiled
        # kernel + workspace. Subsequent run() calls hit the cache.
        # MUST happen before model forward pass to avoid OOM from lazy JIT.
        from dsv4.ops.layouts import (
            ceil_div as bridge_ceil_div,
        )
        from dsv4.ops.gemm_runner import (
            warmup_compilation,
            warmup_fused_swiglu_compilation,
        )
        K_packed = self.hidden_size // 2
        N_packed_l1 = (2 * self.intermediate_size) // 2  # gate+up combined
        N_packed_l2 = self.hidden_size // 2  # down
        warmup_compilation(self.num_experts, K_packed, N_packed_l1, self.device)  # L1
        warmup_compilation(self.num_experts, K_packed, N_packed_l2, self.device)  # L2
        if self._fused_swiglu:
            warmup_fused_swiglu_compilation(
                self.num_experts, K_packed, N_packed_l1, self.device,
                swiglu_limit=self._swiglu_limit if self._swiglu_limit is not None else 0.0,
            )  # Fused L1
        
        self._expert_offsets_buf = torch.zeros(
            self.num_experts + 1, dtype=torch.int32, device=self.device
        )
        self._allocate_buffers()

    def prepare_weights_direct(self, l1_fp4, l1_sf, l1_gs, l2_fp4, l2_sf, l2_gs):
        """DEPRECATED: Use prepare_weights_from_stacked() for checkpoint weights.
        
        This path takes pre-quantized per-expert lists. The stacked path is
        more memory-efficient and avoids per-expert list overhead.
        """
        self.l1_fp4 = l1_fp4
        self.l1_sf = l1_sf
        self.l1_gs = l1_gs
        self.l2_fp4 = l2_fp4
        self.l2_sf = l2_sf
        self.l2_gs = l2_gs
        self._l1_mat_b = None

    def prepare_weights_from_stacked(self, l1_fp4_stacked, l1_sf_stacked,
                                      l1_gs, l2_fp4_stacked, l2_sf_stacked,
                                      l2_gs):
        """Prepare weights from pre-stacked 3D tensors (checkpoint format).

        Takes (E, N, K_packed) fp4 and (E, N, K_sf) scale tensors directly
        from the checkpoint, avoiding the per-expert list→stack round-trip.

        The conversion to K-major and swizzled layout happens in _ensure_stacked.
        This just stores the tensors for deferred processing.
        """
        # Store in checkpoint format (E, N, K) — _ensure_stacked will convert
        self.l1_fp4_stacked = l1_fp4_stacked
        self.l1_sf_stacked = l1_sf_stacked
        self.l1_gs = l1_gs
        self.l2_fp4_stacked = l2_fp4_stacked
        self.l2_sf_stacked = l2_sf_stacked
        self.l2_gs = l2_gs
        self._l1_mat_b = None

    def prepare_weights_from_dequantized(self, l1_weights_bf16, l2_weights_bf16):
        """DEPRECATED: Use prepare_weights_from_stacked() instead.
        
        This path dequantizes checkpoint NVFP4 to BF16 then re-quantizes to our FP4.
        While the round-trip is lossless for DeepSeek-V4 (our packing matches
        the checkpoint convention exactly), it wastes memory and compute.
        The direct byte path (prepare_weights_from_stacked) is preferred.
        """
        self.l1_fp4, self.l1_sf, self.l1_gs = [], [], []
        self.l2_fp4, self.l2_sf, self.l2_gs = [], [], []
        for l1_w, l2_w in zip(l1_weights_bf16, l2_weights_bf16):
            l1_w_t = l1_w.T
            w_fp4, w_sf, w_gs = quantize_weight_to_nvfp4(l1_w_t)
            self.l1_fp4.append(w_fp4)
            self.l1_sf.append(w_sf)
            self.l1_gs.append(w_gs)
            l2_w_t = l2_w.T
            w_fp4, w_sf, w_gs = quantize_weight_to_nvfp4(l2_w_t)
            self.l2_fp4.append(w_fp4)
            self.l2_sf.append(w_sf)
            self.l2_gs.append(w_gs)
        self._l1_mat_b = None

    def _assemble_scales_cudagraph_safe(self, x_sf, expert_offsets,
                                          padded_expert_offsets,
                                          padded_x_sf_buf, per_expert_bufs):
        """Assemble 2D-side activation scales (cudagraph-safe, NO CPU syncs).
        
        Phase 1: Scatter x_sf into padded per-expert sections (GPU-only).
        Phase 2: Apply full-buffer Blackwell 32_4_4 swizzle (no Python loops).
        
        The buffer is 128-row aligned per expert (from padded_expert_offsets),
        so the full-buffer swizzle produces the correct layout. The GEMM reads
        scale_a using padded_expert_offsets, matching the scatter layout.
        """
        K_sf = x_sf.shape[1]
        padded_x_sf = padded_x_sf_buf
        padded_x_sf.zero_()
        
        # Phase 1: Scatter x_sf into padded per-expert sections (GPU-only)
        total_rows = x_sf.shape[0]
        row_indices = self._row_indices_buf[:total_rows]
        expert_assign = torch.searchsorted(
            expert_offsets[1:], row_indices, right=True
        ).clamp(max=self.num_experts - 1)
        local_row = row_indices - expert_offsets[expert_assign]
        dst_rows = padded_expert_offsets[expert_assign] + local_row
        padded_x_sf[dst_rows, :K_sf] = x_sf
        
        # Phase 2: Full-buffer swizzle (no CPU sync, no Python loops)
        # padded_x_sf is 128-row aligned per expert and 4-col aligned.
        # to_blocked: (rows, cols) → view(R, 128, C, 4) → permute(0,2,1,3)
        #   → reshape(-1, 4, 32, 4) → transpose(1,2) → reshape(-1, 32, 16) → flatten
        rows = padded_x_sf.shape[0]
        cols = padded_x_sf.shape[1]
        R = rows // 128
        C = cols // 4
        blocks = padded_x_sf.view(R, 128, C, 4).permute(0, 2, 1, 3)
        rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
        swizzled = rearranged.flatten().view(torch.float8_e4m3fn)
        return swizzled.reshape(rows, cols)

    def compute_activation_global_scales(self, hidden_states_sample, topk_weights, topk_ids):
        """Compute activation global scales from a warmup forward pass.
        
        Called BEFORE cudagraph capture. Uses the SAME padded GEMM path as run()
        to ensure kernel JIT happens with the same layout, and L2 gs is computed
        from actual L1 output (not an approximation).
        """
        self._ensure_stacked()
        device = hidden_states_sample.device
        num_tokens = hidden_states_sample.shape[0]
        top_k = topk_ids.shape[1]
        
        with torch.no_grad():
            # Build slot mapping (same as run())
            flat_ids = topk_ids.reshape(-1)
            num_slots = num_tokens * top_k
            token_indices = self._token_indices[:num_slots]
            sort_idx = flat_ids.argsort(stable=True)
            sorted_ids = flat_ids[sort_idx]
            sorted_token_ids = token_indices[sort_idx]
            slot_hidden = hidden_states_sample[sorted_token_ids]
            
            # L1: get exact gs from quantize_to_nvfp4
            _, _, l1_gs = quantize_to_nvfp4(slot_hidden)
            
            # Quantize slot_hidden for GEMM
            slot_x_fp4, slot_x_sf = quantize_activation_nvfp4(slot_hidden, l1_gs)
            
            tokens_per_expert = torch.bincount(sorted_ids, minlength=self.num_experts)[:self.num_experts].int()
            expert_offsets = self._expert_offsets_buf
            expert_offsets.zero_()
            expert_offsets[1:self.num_experts + 1] = tokens_per_expert.cumsum(0)
            
            padded_tokens_per_expert = ((tokens_per_expert + 127) // 128) * 128
            padded_expert_offsets = self._padded_expert_offsets_buf
            padded_expert_offsets.zero_()
            padded_expert_offsets[1:self.num_experts + 1] = padded_tokens_per_expert.cumsum(0)
            
            # Compute padded_dst (same as run())
            row_indices = self._row_indices_buf[:num_slots]
            expert_assign = torch.searchsorted(
                expert_offsets[1:], row_indices, right=True
            ).clamp(max=self.num_experts - 1)
            local_row = row_indices - expert_offsets[expert_assign]
            padded_dst = padded_expert_offsets[expert_assign] + local_row
            
            # Scatter x_fp4 into padded layout
            padded_x_fp4 = self._shared_bufs['hidden_fp4']
            padded_x_fp4.view(torch.uint8).zero_()
            padded_x_fp4.view(torch.uint8)[padded_dst] = slot_x_fp4.view(torch.uint8)
            
            l1_scale_a = self._assemble_scales_cudagraph_safe(
                slot_x_sf, expert_offsets[:self.num_experts + 1],
                padded_expert_offsets,
                self._padded_x_sf_buf_l1, self._per_expert_scale_bufs_l1
            )
            l1_gsa = torch.full((self.num_experts,), l1_gs, dtype=torch.float32, device=device)
            
            l1_out = run_nvfp4_grouped_gemm(
                mat_a=padded_x_fp4, mat_b=self._l1_mat_b,
                scale_a=l1_scale_a, scale_b=self._l1_scale_b,
                expert_offsets=padded_expert_offsets[1:self.num_experts + 1],
                global_scale_a=l1_gsa, global_scale_b=self._l1_gsb,
            )
            
            # Extract real token outputs
            l1_out_real = l1_out[padded_dst]
            
            # L2: get exact gs from SiLU(gate)*up
            # De-interleave L1 output: with interleaved weights, L1 GEMM
            # output has [gate]*4, [up]*4 pattern. De-interleave before splitting.
            l1_deil = deinterleave_l1_weights(l1_out_real.unsqueeze(0).contiguous())[0]
            gate = l1_deil[:, :self.intermediate_size]
            up = l1_deil[:, self.intermediate_size:]
            gate_silu = torch.nn.functional.silu(gate)
            if self._swiglu_limit is not None:
                gate_silu = gate_silu.clamp(max=self._swiglu_limit)
                up = up.clamp(min=-self._swiglu_limit, max=self._swiglu_limit)
            activated = gate_silu * up
            _, _, l2_gs = quantize_to_nvfp4(activated)
        
        self._l1_activation_global_scale = l1_gs
        self._l2_activation_global_scale = l2_gs
        


    def run(self, hidden_states, topk_weights, topk_ids, expert_indices=None):
        """Forward: route tokens to experts, GEMM, combine.

        Uses torch.library.custom_op (nvfp4::moe_gemm) so torch.compile
        treats this as an opaque op. The custom op calls _run_impl internally.
        """
        if not hasattr(self, '_runner_id'):
            self._runner_id = register_runner(self)
        return nvfp4_moe_gemm(
            hidden_states, topk_weights, topk_ids,
            self._runner_id, self.hidden_size,
        )

    def _run_impl(self, hidden_states, topk_weights, topk_ids, expert_indices=None):
        """Run the NVFP4 MoE forward pass.
        
        Handles global→local expert ID remapping for expert parallelism.
        Fully cudagraph-safe: no CPU-GPU syncs, no dynamic shapes.
        
        Each expert's slots are padded to multiples of 128 for the GEMM.
        expert_offsets is [0, padded_e0, padded_e0+padded_e1, ...].
        scale_a is produced at those same offsets.
        """
        num_tokens = hidden_states.shape[0]
        top_k = topk_ids.shape[1]
        device = hidden_states.device
        
        self._ensure_stacked()
        
        # -- Remap global expert IDs to local IDs --
        local_ids = topk_ids - self.experts_start_idx
        local_mask = (local_ids >= 0) & (local_ids < self.num_experts)
        safe_ids = local_ids.clamp(0, self.num_experts - 1)
        safe_weights = topk_weights * local_mask.float()
        
        # -- Build slot mapping --
        flat_ids = safe_ids.reshape(-1)
        flat_weights = safe_weights.reshape(-1)
        num_slots = num_tokens * top_k
        token_indices = self._token_indices[:num_slots]
        
        sort_idx = flat_ids.argsort(stable=True)
        sorted_ids = flat_ids[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        sorted_token_ids = token_indices[sort_idx]
        
        # Expert offsets (real token counts)
        tokens_per_expert = torch.bincount(sorted_ids, minlength=self.num_experts)[:self.num_experts].int()
        expert_offsets = self._expert_offsets_buf
        expert_offsets.zero_()
        expert_offsets[1:self.num_experts + 1] = tokens_per_expert.cumsum(0)
        
        # Pad each expert to 128-row alignment (GPU-only computation)
        padded_tokens_per_expert = ((tokens_per_expert + 127) // 128) * 128
        padded_expert_offsets = self._padded_expert_offsets_buf
        padded_expert_offsets.zero_()
        padded_expert_offsets[1:self.num_experts + 1] = padded_tokens_per_expert.cumsum(0)
        total_padded_slots = padded_expert_offsets[self.num_experts]
        
        # -- Gather hidden states into slot order, compute padded_dst --
        slot_hidden = hidden_states[sorted_token_ids]
        row_indices = self._row_indices_buf[:num_slots]
        expert_assign = torch.searchsorted(
            expert_offsets[1:], row_indices, right=True
        ).clamp(max=self.num_experts - 1)
        local_row = row_indices - expert_offsets[expert_assign]
        padded_dst = padded_expert_offsets[expert_assign] + local_row
        
        # === L1: gate + up ===
        # Fused amax + quantize: single kernel, zero CPU-GPU syncs.
        # Computes amax on GPU → derives gsa → quantizes to NVFP4.
        # gsa written to GPU buffer for GEMM global_scale_a.
        if getattr(self, '_use_runtime_gsa', False):
            from dsv4.ops.quantize import quantize_nvfp4_gpu_fused
            slot_x_fp4, slot_x_sf, gsa_l1_gpu = quantize_nvfp4_gpu_fused(slot_hidden)
            self._l1_gsa_buf.copy_(gsa_l1_gpu[:1].reshape(1))  # GPU → GPU, no sync
        else:
            slot_x_fp4, slot_x_sf = quantize_nvfp4_gpu(
                slot_hidden, self._l1_activation_global_scale
            )
        # Scatter x_fp4 into padded layout for the GEMM
        # Must scatter as uint8 (float4_e2m1fn_x2 doesn't support index_put)
        padded_x_fp4 = self._shared_bufs['hidden_fp4']
        padded_x_fp4.view(torch.uint8).zero_()
        padded_x_fp4.view(torch.uint8)[padded_dst] = slot_x_fp4.view(torch.uint8)
        
        l1_scale_a = self._assemble_scales_cudagraph_safe(
            slot_x_sf, expert_offsets[:self.num_experts + 1],
            padded_expert_offsets,
            self._padded_x_sf_buf_l1, self._per_expert_scale_bufs_l1
        )
        l1_gsa = self._l1_gsa_buf  # already filled by GPU compute (no .fill_ needed)
        
        if self._fused_swiglu:
            # === Fused L1 GEMM + SwiGLU in kernel registers ===
            l1_out = run_fused_swiglu_grouped_gemm(
                mat_a=padded_x_fp4, mat_b=self._l1_mat_b,
                scale_a=l1_scale_a, scale_b=self._l1_scale_b,
                expert_offsets=padded_expert_offsets[1:self.num_experts + 1],
                global_scale_a=l1_gsa, global_scale_b=self._l1_gsb,
                swiglu_limit=self._swiglu_limit if self._swiglu_limit is not None else 0.0,
            )
            l1_out_real = l1_out[padded_dst]
            # Fused deinterleave + amax + quantize: zero CPU syncs.
            # Computes gsa from de-interleaved SwiGLU output on GPU,
            # quantizes in the same kernel. Writes gsa to GPU buffer.
            if getattr(self, '_use_runtime_gsa', False):
                from dsv4.ops.quantize import deinterleave_amax_quantize_nvfp4_fused
                slot_l2_x_fp4, slot_l2_x_sf, gsa_l2_gpu = deinterleave_amax_quantize_nvfp4_fused(
                    l1_out_real, self.intermediate_size)
                self._l2_gsa_buf.copy_(gsa_l2_gpu[:1].reshape(1))  # GPU → GPU, no sync
            else:
                slot_l2_x_fp4, slot_l2_x_sf = deinterleave_quantize_nvfp4_cuda(
                    l1_out_real, self.intermediate_size, self._l2_activation_global_scale
                )
        else:
            # === Non-fused L1 GEMM + PyTorch SiLU(gate)*up ===
            l1_out = run_nvfp4_grouped_gemm(
                mat_a=padded_x_fp4, mat_b=self._l1_mat_b,
                scale_a=l1_scale_a, scale_b=self._l1_scale_b,
                expert_offsets=padded_expert_offsets[1:self.num_experts + 1],
                global_scale_a=l1_gsa, global_scale_b=self._l1_gsb,
            )
            l1_out_real = l1_out[padded_dst]
            l1_deil = deinterleave_l1_weights(l1_out_real.unsqueeze(0).contiguous())[0]
            gate = l1_deil[:, :self.intermediate_size]
            up = l1_deil[:, self.intermediate_size:]
            gate_silu = torch.nn.functional.silu(gate)
            if self._swiglu_limit is not None:
                gate_silu = gate_silu.clamp(max=self._swiglu_limit)
                up = up.clamp(min=-self._swiglu_limit, max=self._swiglu_limit)
            activated = gate_silu * up

        # Compute runtime gsa for L2 from activated output (non-fused path)
        # Fused amax + quantize: zero CPU syncs.
        if not self._fused_swiglu and getattr(self, '_use_runtime_gsa', False):
            from dsv4.ops.quantize import quantize_nvfp4_gpu_fused
            slot_l2_x_fp4, slot_l2_x_sf, gsa_l2_gpu = quantize_nvfp4_gpu_fused(activated)
            self._l2_gsa_buf.copy_(gsa_l2_gpu[:1].reshape(1))  # GPU → GPU, no sync
        elif not self._fused_swiglu:
            slot_l2_x_fp4, slot_l2_x_sf = quantize_nvfp4_gpu(
                activated, self._l2_activation_global_scale
            )
        padded_activated_fp4 = self._shared_bufs['activated_fp4']
        padded_activated_fp4.view(torch.uint8).zero_()
        padded_activated_fp4.view(torch.uint8)[padded_dst] = slot_l2_x_fp4.view(torch.uint8)
        
        l2_scale_a = self._assemble_scales_cudagraph_safe(
            slot_l2_x_sf, expert_offsets[:self.num_experts + 1],
            padded_expert_offsets,
            self._padded_x_sf_buf_l2, self._per_expert_scale_bufs_l2
        )
        l2_gsa = self._l2_gsa_buf  # already filled by GPU compute (no .fill_ needed)
        
        l2_out = run_nvfp4_grouped_gemm(
            mat_a=padded_activated_fp4, mat_b=self._l2_mat_b,
            scale_a=l2_scale_a, scale_b=self._l2_scale_b,
            expert_offsets=padded_expert_offsets[1:self.num_experts + 1],
            global_scale_a=l2_gsa, global_scale_b=self._l2_gsb,
        )
        
        l2_out_real = l2_out[padded_dst]
        
        # === Scatter -> final output ===
        y = self._output_buf[:num_tokens]
        y.zero_()
        weighted_out = l2_out_real * sorted_weights.unsqueeze(1).to(l2_out_real.dtype)
        y.scatter_add_(
            0,
            sorted_token_ids.unsqueeze(1).expand(-1, self.hidden_size),
            weighted_out,
        )
        
        return y
