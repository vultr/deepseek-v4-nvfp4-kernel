"""
Full NVFP4 MoE pipeline using CuTeDSL ScaledGroupedGemmKernel.

Data flow (NVFP4-native, BF16 only where required):
  1. BF16 hidden_states → quantize to NVFP4 (stage_activation)
  2. L1 GEMM: NVFP4 × NVFP4 → BF16 output (gate+up)
  3. SiLU(gate) * up → BF16 activated (nonlinear requires BF16)
  4. Re-quantize activated → NVFP4 (stage_activation)
  5. L2 GEMM: NVFP4 × NVFP4 → BF16 output (down_proj)
  6. Scatter with routing weights → BF16 output

Both GEMMs are fully NVFP4: A in float4_e2m1fn_x2, B in float4_e2m1fn_x2,
block scales in float8_e4m3fn, global scales in float32.
"""
import torch

from dsv4.ops.quantize import (
    quantize_to_nvfp4,
    quantize_weight_to_nvfp4,
)
from dsv4.ops.layouts import (
    assemble_scales_2d_side,
    assemble_scales_3d_side,
    make_b_k_major,
    compute_expert_offsets,
    interleave_l1_weights,
    deinterleave_l1_weights,
)
from dsv4.ops.gemm_runner import (
    run_nvfp4_grouped_gemm,
    run_fused_swiglu_grouped_gemm,
    warmup_fused_swiglu_compilation,
)


def stage_activation(x_bf16):
    """Quantize BF16 activation to NVFP4.
    
    This is the NVFP4-native equivalent of the old stage_activation.
    Keeps data in FP4 as long as possible — only leaves NVFP4 for nonlinear ops.
    
    Returns (x_fp4, x_sf, global_scale) where:
      x_fp4: float4_e2m1fn_x2 (native PyTorch FP4)
      x_sf: float8_e4m3fn block scales
      global_scale: float32 scalar
    """
    return quantize_to_nvfp4(x_bf16)


def quantize_weight(w_bf16):
    """Quantize BF16 weight to NVFP4.
    
    Weight is (K, N) where K is the input/hidden dim (packed dimension).
    
    Returns (w_fp4, w_sf, global_scale).
    """
    return quantize_weight_to_nvfp4(w_bf16)


def prepare_nvfp4_moe_weights(nvfp4_tensors, layer_idx, expert_indices):
    """Load NVFP4 checkpoint weights and prepare for the grouped GEMM.
    
    Dequantizes checkpoint NVFP4 → BF16 → re-quantizes to our native format.
    This round-trip ensures our FP4 packing convention matches the kernel.
    
    Future optimization: load checkpoint FP4 bytes directly into
    float4_e2m1fn_x2 tensors without the BF16 round-trip.
    
    Returns dict with l1 and l2 weight info per expert.
    """
    from tests.layertest import dequantize_nvfp4_weight, DEVICE
    
    l1_weights = []  # gate+up fused, (K, N) = (hidden, intermediate)
    l2_weights = []  # down, (K, N) = (intermediate, hidden)
    
    for e in expert_indices:
        # L1: gate + up
        gate_w_bf16 = dequantize_nvfp4_weight(
            nvfp4_tensors.get(f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight",
                              nvfp4_tensors.get(f"model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight")).to(DEVICE),
            nvfp4_tensors[f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight_scale"].to(DEVICE),
            nvfp4_tensors[f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight_scale_2"].item(),
        )
        up_w_bf16 = dequantize_nvfp4_weight(
            nvfp4_tensors[f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"].to(DEVICE),
            nvfp4_tensors[f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight_scale"].to(DEVICE),
            nvfp4_tensors[f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight_scale_2"].item(),
        )
        
        # Fuse gate+up: (6144, 7168) → transpose to (7168, 6144) for weight quantization
        fused_l1 = torch.cat([gate_w_bf16, up_w_bf16], dim=0)  # (6144, 7168)
        l1_w_bf16 = fused_l1.T  # (7168, 6144) — K=7168, N=6144
        l1_weights.append(l1_w_bf16)
        
        # L2: down
        down_w_key = f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
        if down_w_key in nvfp4_tensors:
            down_w_bf16 = dequantize_nvfp4_weight(
                nvfp4_tensors[down_w_key].to(DEVICE),
                nvfp4_tensors[f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight_scale"].to(DEVICE),
                nvfp4_tensors[f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight_scale_2"].item(),
            )
            # down_proj is (7168, 3072) → transpose to (3072, 7168) for K=intermediate
            l2_w_bf16 = down_w_bf16.T  # (3072, 7168) — K=3072, N=7168
        else:
            # Expert 211 has no down_proj
            l2_w_bf16 = torch.zeros(3072, 7168, dtype=torch.bfloat16, device=DEVICE)
        
        l2_weights.append(l2_w_bf16)
    
    # Quantize all weights to NVFP4
    l1_fp4, l1_sf, l1_gs = [], [], []
    l2_fp4, l2_sf, l2_gs = [], [], []
    
    for l1_w, l2_w in zip(l1_weights, l2_weights):
        w_fp4, w_sf, w_gs = quantize_weight(l1_w)
        l1_fp4.append(w_fp4)
        l1_sf.append(w_sf)
        l1_gs.append(w_gs)
        
        w_fp4, w_sf, w_gs = quantize_weight(l2_w)
        l2_fp4.append(w_fp4)
        l2_sf.append(w_sf)
        l2_gs.append(w_gs)
    
    return {
        'l1_fp4': l1_fp4, 'l1_sf': l1_sf, 'l1_gs': l1_gs,
        'l2_fp4': l2_fp4, 'l2_sf': l2_sf, 'l2_gs': l2_gs,
    }


def run_nvfp4_moe(
    hidden_states,    # (num_tokens, hidden_size) BF16
    expert_ids,       # (num_tokens, top_k) int32
    expert_weights,   # (num_tokens, top_k) float32
    weights,          # dict from prepare_nvfp4_moe_weights
    expert_indices,   # list of expert IDs
    swiglu_limit=None,  # Optional clamp for SiLU output
):
    """Run the full NVFP4 MoE forward pass.
    
    NVFP4-native pipeline:
      1. Quantize activation → NVFP4
      2. L1 GEMM (NVFP4 × NVFP4 → BF16)
      3. SiLU(gate) * up (BF16 — nonlinear requires BF16)
      4. Re-quantize → NVFP4
      5. L2 GEMM (NVFP4 × NVFP4 → BF16)
      6. Scatter with routing weights → BF16
    
    Returns: (num_tokens, hidden_size) BF16
    """
    num_tokens, hidden_size = hidden_states.shape
    top_k = expert_ids.shape[1]
    device = hidden_states.device
    
    # ── Build slot-based routing ──
    expert_token_lists = {e: [] for e in expert_indices}
    for t in range(num_tokens):
        for k in range(top_k):
            e = expert_ids[t, k].item()
            if e in expert_token_lists:
                expert_token_lists[e].append(t)
    
    tokens_per_expert = [len(expert_token_lists[e]) for e in expert_indices]
    num_experts = len(expert_indices)
    
    # Slot-major activation: [expert0_tokens | expert1_tokens | ...]
    slot_hidden = torch.cat([
        hidden_states[expert_token_lists[e]] for e in expert_indices
    ], dim=0) if any(tpe > 0 for tpe in tokens_per_expert) else torch.zeros(0, hidden_size, dtype=torch.bfloat16, device=device)
    
    num_slots = slot_hidden.shape[0]
    if num_slots == 0:
        return torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    
    expert_offsets = compute_expert_offsets(tokens_per_expert, num_experts)
    
    # ════════════════════════════════════════════════════════════════
    # L1: gate + up projection (NVFP4 × NVFP4 → BF16)
    # ════════════════════════════════════════════════════════════════
    
    # Quantize activation to NVFP4
    x_fp4, x_sf, x_igs = stage_activation(slot_hidden)
    
    # Stack L1 weights, interleave gate/up, convert to K-major
    l1_stacked = torch.stack(weights['l1_fp4'])  # (E, K, N)
    l1_stacked = interleave_l1_weights(l1_stacked)  # gate/up at granularity 4 BF16
    l1_mat_b = make_b_k_major(l1_stacked)
    
    # Assemble scales
    x_sf_parts = []
    offset = 0
    for tpe in tokens_per_expert:
        x_sf_parts.append(x_sf[offset:offset+tpe])
        offset += tpe
    l1_scale_a = assemble_scales_2d_side(x_sf_parts)
    # Interleave L1 SF to match the interleaved weight layout.
    # SF is (K_sf, N) from quantize_weight_to_nvfp4. interleave_l1_weights
    # operates on the last dim, which is N. So (1, K_sf, N) is correct.
    # After interleave, transpose to (N, K_sf) for the assembly function.
    l1_sf_il = []
    for sf in weights['l1_sf']:
        sf_ekn = sf.unsqueeze(0)              # (1, K_sf, N)
        sf_ekn = interleave_l1_weights(sf_ekn) # interleaved along N
        l1_sf_il.append(sf_ekn[0].T.contiguous())  # (N, K_sf) for assembly
    from dsv4.kernels.gemm.grouped import assemble_raw_scales_2d3d_3d_side as _assemble_3d
    l1_scale_b = _assemble_3d(l1_sf_il)
    
    # Global scales: alpha = igs * weight_gs for each expert
    l1_global_scale_a = torch.tensor([x_igs] * num_experts, dtype=torch.float32, device=device)
    l1_global_scale_b = torch.tensor(weights['l1_gs'], dtype=torch.float32, device=device)
    
    # Run L1 GEMM
    l1_out = run_nvfp4_grouped_gemm(
        mat_a=x_fp4, mat_b=l1_mat_b,
        scale_a=l1_scale_a, scale_b=l1_scale_b,
        expert_offsets=expert_offsets,
        global_scale_a=l1_global_scale_a, global_scale_b=l1_global_scale_b,
    )  # (num_slots, 2*intermediate) BF16
    
    # ════════════════════════════════════════════════════════════════
    # SiLU(gate) * up (BF16 — nonlinear requires BF16)
    # ════════════════════════════════════════════════════════════════
    # L1 output is (tokens, 2*intermediate) with interleaved gate/up.
    # De-interleave to recover standard [gate | up] layout.
    intermediate_size = l1_out.shape[1] // 2
    l1_deil = deinterleave_l1_weights(l1_out.unsqueeze(0).contiguous())[0]
    gate = l1_deil[:, :intermediate_size]
    up = l1_deil[:, intermediate_size:]
    gate_silu = torch.nn.functional.silu(gate)
    if swiglu_limit is not None:
        gate_silu = gate_silu.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    activated = gate_silu * up  # (num_slots, intermediate) BF16
    
    # ════════════════════════════════════════════════════════════════
    # L2: down projection (NVFP4 × NVFP4 → BF16)
    # ════════════════════════════════════════════════════════════════
    
    # Re-quantize activated → NVFP4
    l2_x_fp4, l2_x_sf, l2_x_igs = stage_activation(activated)
    
    # Stack L2 weights
    l2_mat_b = make_b_k_major(torch.stack(weights['l2_fp4']))
    
    # Assemble L2 scales
    l2_sf_parts = []
    offset = 0
    for tpe in tokens_per_expert:
        l2_sf_parts.append(l2_x_sf[offset:offset+tpe])
        offset += tpe
    l2_scale_a = assemble_scales_2d_side(l2_sf_parts)
    l2_scale_b = assemble_scales_3d_side(weights['l2_sf'])
    
    # Global scales
    l2_global_scale_a = torch.tensor([l2_x_igs] * num_experts, dtype=torch.float32, device=device)
    l2_global_scale_b = torch.tensor(weights['l2_gs'], dtype=torch.float32, device=device)
    
    # Run L2 GEMM
    l2_out = run_nvfp4_grouped_gemm(
        mat_a=l2_x_fp4, mat_b=l2_mat_b,
        scale_a=l2_scale_a, scale_b=l2_scale_b,
        expert_offsets=expert_offsets,
        global_scale_a=l2_global_scale_a, global_scale_b=l2_global_scale_b,
    )  # (num_slots, hidden_size) BF16
    
    # ════════════════════════════════════════════════════════════════
    # Scatter with routing weights → final output
    # ════════════════════════════════════════════════════════════════
    y = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    
    slot_idx = 0
    for e in expert_indices:
        for t in expert_token_lists[e]:
            # Find which top-k slot this is for this token
            for k in range(top_k):
                if expert_ids[t, k].item() == e:
                    w = expert_weights[t, k].item()
                    y[t] += w * l2_out[slot_idx]
                    break
            slot_idx += 1
    
    return y


def run_nvfp4_moe_fused(
    hidden_states,    # (num_tokens, hidden_size) BF16
    expert_ids,       # (num_tokens, top_k) int32
    expert_weights,   # (num_tokens, top_k) float32
    weights,          # dict from prepare_nvfp4_moe_weights
    expert_indices,   # list of expert IDs
    swiglu_limit=0.0,
    l2_activation_gs=None,  # pre-computed L2 activation global scale (avoids amax sync)
):
    """Run the NVFP4 MoE forward pass with fused SwiGLU kernel.

    Fused pipeline (saves BF16 GMEM write+read for gate/up):
      1. Quantize activation -> NVFP4
      2. Fused L1 GEMM + SwiGLU (NVFP4 x NVFP4 -> BF16 with silu(gate)*up in registers)
      3. De-interleave fused output, extract SwiGLU result
      4. Re-quantize -> NVFP4
      5. L2 GEMM (NVFP4 x NVFP4 -> BF16)
      6. Scatter with routing weights -> BF16

    Returns: (num_tokens, hidden_size) BF16
    """
    num_tokens, hidden_size = hidden_states.shape
    top_k = expert_ids.shape[1]
    device = hidden_states.device

    # Build slot-based routing
    expert_token_lists = {e: [] for e in expert_indices}
    for t in range(num_tokens):
        for k in range(top_k):
            e = expert_ids[t, k].item()
            if e in expert_token_lists:
                expert_token_lists[e].append(t)

    tokens_per_expert = [len(expert_token_lists[e]) for e in expert_indices]
    num_experts = len(expert_indices)

    slot_hidden = torch.cat([
        hidden_states[expert_token_lists[e]] for e in expert_indices
    ], dim=0) if any(tpe > 0 for tpe in tokens_per_expert) else torch.zeros(0, hidden_size, dtype=torch.bfloat16, device=device)

    num_slots = slot_hidden.shape[0]
    if num_slots == 0:
        return torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    expert_offsets = compute_expert_offsets(tokens_per_expert, num_experts)

    # === L1: Fused gate+up projection with SwiGLU in registers ===

    # Quantize activation to NVFP4
    x_fp4, x_sf, x_igs = stage_activation(slot_hidden)

    # Stack L1 weights, interleave gate/up, convert to K-major
    l1_stacked = torch.stack(weights['l1_fp4'])
    l1_stacked = interleave_l1_weights(l1_stacked)
    l1_mat_b = make_b_k_major(l1_stacked)

    # Assemble scales (same as non-fused path)
    x_sf_parts = []
    offset = 0
    for tpe in tokens_per_expert:
        x_sf_parts.append(x_sf[offset:offset+tpe])
        offset += tpe
    l1_scale_a = assemble_scales_2d_side(x_sf_parts)

    l1_sf_il = []
    for sf in weights['l1_sf']:
        sf_ekn = sf.unsqueeze(0)
        sf_ekn = interleave_l1_weights(sf_ekn)
        l1_sf_il.append(sf_ekn[0].T.contiguous())
    from dsv4.kernels.gemm.grouped import assemble_raw_scales_2d3d_3d_side as _assemble_3d
    l1_scale_b = _assemble_3d(l1_sf_il)

    l1_global_scale_a = torch.tensor([x_igs] * num_experts, dtype=torch.float32, device=device)
    l1_global_scale_b = torch.tensor(weights['l1_gs'], dtype=torch.float32, device=device)

    # Run fused SwiGLU kernel
    # Output: (num_slots, 2*intermediate) BF16
    # Even 8-col groups = silu(gate), Odd 8-col groups = silu(gate)*up
    l1_fused_out = run_fused_swiglu_grouped_gemm(
        mat_a=x_fp4, mat_b=l1_mat_b,
        scale_a=l1_scale_a, scale_b=l1_scale_b,
        expert_offsets=expert_offsets,
        global_scale_a=l1_global_scale_a, global_scale_b=l1_global_scale_b,
        swiglu_limit=swiglu_limit,
    )

    # De-interleave + quantize using custom CUDA kernel (4x faster)
    intermediate_size = l1_fused_out.shape[1] // 2
    # Use pre-computed L2 activation gs, or compute from amax (fallback)
    l2_gs = l2_activation_gs if l2_activation_gs is not None else l1_fused_out.abs().amax().float().item() / 2688.0
    from dsv4.ops.quantize import (
        deinterleave_quantize_nvfp4_cuda,
        quantize_activation_nvfp4,
    )
    l2_x_fp4, l2_x_sf = deinterleave_quantize_nvfp4_cuda(l1_fused_out, intermediate_size, l2_gs)
    # Skip the separate L2 quantize step below — we already have FP4+SF
    # Set activated to None to signal we already quantized
    activated = None

    # === L2: down projection ===
    if activated is not None:
        l2_x_fp4, l2_x_sf, l2_x_igs = stage_activation(activated)
    else:
        # Already quantized by the custom CUDA kernel
        l2_x_igs = l2_gs
    l2_mat_b = make_b_k_major(torch.stack(weights['l2_fp4']))

    l2_sf_parts = []
    offset = 0
    for tpe in tokens_per_expert:
        l2_sf_parts.append(l2_x_sf[offset:offset+tpe])
        offset += tpe
    l2_scale_a = assemble_scales_2d_side(l2_sf_parts)
    l2_scale_b = assemble_scales_3d_side(weights['l2_sf'])

    l2_global_scale_a = torch.tensor([l2_x_igs] * num_experts, dtype=torch.float32, device=device)
    l2_global_scale_b = torch.tensor(weights['l2_gs'], dtype=torch.float32, device=device)

    l2_out = run_nvfp4_grouped_gemm(
        mat_a=l2_x_fp4, mat_b=l2_mat_b,
        scale_a=l2_scale_a, scale_b=l2_scale_b,
        expert_offsets=expert_offsets,
        global_scale_a=l2_global_scale_a, global_scale_b=l2_global_scale_b,
    )

    # Scatter with routing weights
    y = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    slot_idx = 0
    for e in expert_indices:
        for t in expert_token_lists[e]:
            for k in range(top_k):
                if expert_ids[t, k].item() == e:
                    w = expert_weights[t, k].item()
                    y[t] += w * l2_out[slot_idx]
                    break
            slot_idx += 1

    return y
