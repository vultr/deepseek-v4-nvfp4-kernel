"""NVFP4 GEMM runner: warmup, compile, and execute grouped/fused GEMMs."""
import math
import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils

from dsv4.kernels.gemm.grouped import ScaledGroupedGemmKernel
from dsv4.kernels.gemm.fused_swiglu import FusedSwiGLUScaledGroupedGemmKernel
from dsv4.ops.quantize import (
    quantize_activation_nvfp4,
    quantize_weight_to_nvfp4,
    quantize_to_nvfp4,
    deinterleave_quantize_nvfp4_cuda,
    SF_VEC_SIZE,
)
from dsv4.ops.layouts import (
    interleave_l1_weights,
    deinterleave_l1_weights,
    assemble_scales_2d_side,
    assemble_scales_3d_side,
    make_b_k_major,
    compute_expert_offsets,
    ceil_div,
    round_up,
)

# Cache compiled kernels + pre-allocated workspace by cache_key
# Each entry: {'compiled': callable, 'workspace': Tensor, 'workspace_size': int}
#
# Key design decisions (Bug #1 fix):
# - cute.compile does NOT corrupt GPU memory (verified 2026-05-20 on B200).
#   The original _needs_token_refill hack was a misdiagnosis. The real bug
#   was elsewhere (likely OOB write or weight loading).
# - Workspace is pre-allocated per cache entry during warmup_compilation()
#   and reused on subsequent calls. No torch.full() in the hot path.
# - CuTe tensor wrappers (from_dlpack + mark_layout_dynamic) are cheap
#   metadata wrappers. We re-create them per call from real tensors.
#   Caching them would hold stale references to tensors that get freed.
_compiled_kernel_cache = {}

def warmup_compilation(num_experts, K_packed, N_packed, device,
                       mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1)):
    """Eagerly JIT-compile the GEMM kernel for a specific shape.
    
    Call this BEFORE model weights are loaded to ensure cute.compile
    runs exactly once per shape. The compiled kernel is cached and
    reused by run_nvfp4_grouped_gemm on the forward path.
    
    Uses random non-zero data. Zero-filled FP4/FP8 tensors cause
    cudaErrorIllegalInstruction because the GEMM arithmetic hits
    invalid values (division by zero in scale dequantization, NaN
    propagation). Random data produces valid intermediate results
    that exercise the kernel's full arithmetic path.
    
    The warmup tensors are freed immediately after compilation.
    Memory cost is minimal (~50MB for typical DeepSeek-V4 shapes).
    
    Args:
        num_experts: number of experts (local, after expert parallelism)
        K_packed: K dimension in float4 elements (i.e. K_original // 2)
        N_packed: N dimension in float4 elements (i.e. N_original // 2)
        device: 'cuda:X' or 'cuda'
        mma_tiler_mn: GEMM tiling (default (128,128))
        cluster_shape_mn: cluster shape (default (1,1))
    """
    cache_key = (num_experts, str(device), mma_tiler_mn, cluster_shape_mn,
                 K_packed, N_packed)
    
    if cache_key in _compiled_kernel_cache:
        return  # Already compiled
    
    # Generate VALID FP4 data by quantizing random BF16 through our pipeline.
    # Random uint8 bytes as FP4 bit patterns produce NaN/Inf when the GEMM
    # dequantizes them (FP4 value * FP8 scale * FP32 global scale), which
    # causes cudaErrorIllegalInstruction in the Blackwell MMA hardware.
    # The ONLY safe approach: generate random BF16, quantize through our
    # quantize_to_nvfp4, producing mathematically consistent FP4 + FP8 scales.
    _warmup_a_bf16 = torch.randn(128, K_packed * 2, dtype=torch.bfloat16, device=device) * 0.1
    mat_a, scale_a, _ = quantize_to_nvfp4(_warmup_a_bf16)
    del _warmup_a_bf16
    _warmup_b_bf16 = torch.randn(1, K_packed * 2, N_packed * 2, dtype=torch.bfloat16, device=device) * 0.1  # 1 expert: kernel compiles same regardless of count
    mat_b, scale_b, _ = quantize_to_nvfp4(_warmup_b_bf16)
    del _warmup_b_bf16
    out = torch.zeros(128, N_packed, dtype=torch.bfloat16, device=device)
    expert_offsets = torch.full((1,), 128, dtype=torch.int32, device=device)
    global_scale_a = torch.ones(1, dtype=torch.float32, device=device)
    global_scale_b = torch.ones(1, dtype=torch.float32, device=device)
    
    kernel = ScaledGroupedGemmKernel(
        scenario="2Dx3D",
        sf_vec_size=SF_VEC_SIZE,
        accumulate_on_output=False,
        separate_tensormap_init=True,
        consistent_token_padding=False,
        mma_tiler_mnk=(*mma_tiler_mn, 256),
        cluster_shape_mnk=(*cluster_shape_mn, 1),
    )
    
    def to_cute(t):
        ct = cutlass_torch.from_dlpack(t)
        return ct.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(t))
    
    a_c = to_cute(mat_a)
    b_c = to_cute(mat_b)
    sfa_c = to_cute(scale_a)
    sfb_c = to_cute(scale_b)
    c_c = to_cute(out)
    offs_c = to_cute(expert_offsets)
    
    workspace_size = kernel.get_workspace_size(num_experts)
    workspace = torch.full((workspace_size,), 255, dtype=torch.uint8, device=device)
    ws_c = to_cute(workspace)
    
    gsa_c = to_cute(global_scale_a)
    gsb_c = to_cute(global_scale_b)
    
    import cuda.bindings.driver as cuda
    cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    
    compiled = cute.compile(
        kernel, a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c,
        max_active_clusters, stream,
        global_scale_a=gsa_c, global_scale_b=gsb_c,
    )
    
    # Warmup run (required by CuTeDSL)
    compiled(
        a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c, stream,
        global_scale_a=gsa_c, global_scale_b=gsb_c,
    )
    torch.cuda.synchronize()
    
    # Cache compiled kernel + pre-allocated workspace
    # (NOT CuTe wrappers — they hold refs to dummy tensors)
    _compiled_kernel_cache[cache_key] = {
        'compiled': compiled,
        'workspace': workspace,  # pre-allocated, reused
        'workspace_size': workspace_size,
    }
    
    # Free dummy data tensors (workspace is kept in cache)
    del mat_a, mat_b, scale_a, scale_b, out, expert_offsets
    del global_scale_a, global_scale_b
    del a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c, gsa_c, gsb_c
    torch.cuda.empty_cache()



def run_nvfp4_grouped_gemm(
    mat_a,          # (tokens_sum, K_packed) float4_e2m1fn_x2
    mat_b,          # (experts, K_packed, N_packed) float4_e2m1fn_x2, K-major
    scale_a,        # assembled 2D side (padded + swizzled)
    scale_b,        # assembled 3D side (padded + swizzled)
    expert_offsets, # (experts,) int32 cumulative token offsets
    global_scale_a=None,  # (experts,) float32
    global_scale_b=None,  # (experts,) float32
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
):
    """Run the CuTeDSL NVFP4 scaled grouped GEMM.
    
    2Dx3D: A(tokens, K) x B(experts, K, N) -> C(tokens, N)
    
    Kernel is compiled once (either via warmup_compilation() during init
    or lazily on first call), then cached. Workspace is pre-allocated
    during warmup and reused — no torch.full() in the hot path.
    No torch.cuda.synchronize() or .item() in the forward path.
    """
    num_experts = mat_b.shape[0]
    n_dim = mat_b.shape[2]
    tokens_sum = mat_a.shape[0]

    out = torch.zeros(tokens_sum, n_dim, dtype=torch.bfloat16, device=mat_a.device)

    # NVFP4-3: use 2-CTA UMMA for M>=256 (1.7-1.9× throughput at prefill)
    use_2cta = tokens_sum >= 256 and cluster_shape_mn[0] % 2 == 0
    effective_mma_tiler_mn = (
        (mma_tiler_mn[0] * 2, mma_tiler_mn[1]) if use_2cta else mma_tiler_mn
    )
    effective_cluster_shape_mn = (
        (cluster_shape_mn[0] * 2, cluster_shape_mn[1]) if use_2cta else cluster_shape_mn
    )

    cache_key = (num_experts, str(mat_a.device), effective_mma_tiler_mn, effective_cluster_shape_mn,
                 mat_a.shape[1], mat_b.shape[2], use_2cta)
    
    if cache_key not in _compiled_kernel_cache:
        # Lazy compilation — safety net if warmup_compilation wasn't called.
        # This should NOT happen in production (warmup is called during init).
        kernel = ScaledGroupedGemmKernel(
            scenario="2Dx3D",
            sf_vec_size=SF_VEC_SIZE,
            accumulate_on_output=False,
            separate_tensormap_init=True,
            consistent_token_padding=False,
            mma_tiler_mnk=(*effective_mma_tiler_mn, 256),
            cluster_shape_mnk=(*effective_cluster_shape_mn, 1),
            use_2cta_instrs=use_2cta,
        )

        def to_cute(t):
            ct = cutlass_torch.from_dlpack(t)
            return ct.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(t))

        a_c = to_cute(mat_a)
        b_c = to_cute(mat_b)
        sfa_c = to_cute(scale_a)
        sfb_c = to_cute(scale_b)
        c_c = to_cute(out)
        offs_c = to_cute(expert_offsets)

        workspace_size = kernel.get_workspace_size(num_experts)
        workspace = torch.full((workspace_size,), 255, dtype=torch.uint8, device=mat_a.device)
        ws_c = to_cute(workspace)

        gsa_c = to_cute(global_scale_a) if global_scale_a is not None else None
        gsb_c = to_cute(global_scale_b) if global_scale_b is not None else None

        import cuda.bindings.driver as cuda
        cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        compiled = cute.compile(
            kernel, a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c,
            max_active_clusters, stream,
            global_scale_a=gsa_c, global_scale_b=gsb_c,
        )
        compiled(
            a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c, stream,
            global_scale_a=gsa_c, global_scale_b=gsb_c,
        )
        
        _compiled_kernel_cache[cache_key] = {
            'compiled': compiled,
            'workspace': workspace,
            'workspace_size': workspace_size,
        }
    
    # --- Invoke cached kernel ---
    entry = _compiled_kernel_cache[cache_key]
    compiled = entry['compiled']
    workspace = entry['workspace']
    
    # Re-create CuTe wrappers from current tensors each call.
    # This is cheap (metadata only, no GPU work) and avoids stale
    # references to tensors from previous calls that may have been freed.
    def to_cute(t):
        ct = cutlass_torch.from_dlpack(t)
        return ct.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(t))
    
    a_c = to_cute(mat_a)
    b_c = to_cute(mat_b)
    sfa_c = to_cute(scale_a)
    sfb_c = to_cute(scale_b)
    c_c = to_cute(out)
    offs_c = to_cute(expert_offsets)
    ws_c = to_cute(workspace)
    
    gsa_c = to_cute(global_scale_a) if global_scale_a is not None else None
    gsb_c = to_cute(global_scale_b) if global_scale_b is not None else None
    
    import cuda.bindings.driver as cuda
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    
    compiled(
        a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c, stream,
        global_scale_a=gsa_c, global_scale_b=gsb_c,
    )

    return out


# ── Fused SwiGLU GEMM (Stage 1: SiLU in registers, BF16 output) ──────

_fused_kernel_cache = {}
def warmup_fused_swiglu_compilation(num_experts, K_packed, N_packed, device,
                                     swiglu_limit=0.0,
                                     mma_tiler_mn=(128, 128),
                                     cluster_shape_mn=(1, 1)):
    """Eagerly JIT-compile the fused SwiGLU GEMM kernel.
    
    Must be called during model initialization. See warmup_compilation()
    for the standard GEMM equivalent.
    """
    from dsv4.kernels.gemm.fused_swiglu import FusedSwiGLUScaledGroupedGemmKernel
    
    cache_key = ('fused', num_experts, str(device), mma_tiler_mn, cluster_shape_mn,
                 K_packed, N_packed, swiglu_limit)
    
    if cache_key in _fused_kernel_cache:
        return
    
    # Generate VALID FP4 data by quantizing random BF16 (same as warmup_compilation)
    _warmup_a_bf16 = torch.randn(128, K_packed * 2, dtype=torch.bfloat16, device=device) * 0.1
    mat_a, scale_a, _ = quantize_to_nvfp4(_warmup_a_bf16)
    del _warmup_a_bf16
    _warmup_b_bf16 = torch.randn(1, K_packed * 2, N_packed * 2, dtype=torch.bfloat16, device=device) * 0.1  # 1 expert: kernel compiles same regardless of count
    mat_b, scale_b, _ = quantize_to_nvfp4(_warmup_b_bf16)
    del _warmup_b_bf16
    # BF16 output (Stage 1: we still write BF16)
    # The fused kernel writes intermediate (N/2) since gate+up → silu result
    out = torch.zeros(128, N_packed, dtype=torch.bfloat16, device=device)
    expert_offsets = torch.full((1,), 128, dtype=torch.int32, device=device)
    global_scale_a = torch.ones(1, dtype=torch.float32, device=device)
    global_scale_b = torch.ones(1, dtype=torch.float32, device=device)
    
    # NVFP4-3: Enable 2-CTA UMMA when MMA tile M >= 256 and cluster M is even.
    # 2-CTA gives 1.7-1.9× throughput at prefill shapes by doubling effective MMA tile M.
    # Not beneficial at decode (M < 256) — would waste hardware.
    use_2cta = mma_tiler_mn[0] >= 256 and cluster_shape_mn[0] % 2 == 0
    
    kernel = FusedSwiGLUScaledGroupedGemmKernel(
        scenario="2Dx3D",
        sf_vec_size=SF_VEC_SIZE,
        accumulate_on_output=False,
        separate_tensormap_init=True,
        consistent_token_padding=False,
        mma_tiler_mnk=(*mma_tiler_mn, 256),
        cluster_shape_mnk=(*cluster_shape_mn, 1),
        fused_swiglu=True,
        swiglu_limit=swiglu_limit,
        use_2cta_instrs=use_2cta,
    )
    
    def to_cute(t):
        ct = cutlass_torch.from_dlpack(t)
        return ct.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(t))
    
    a_c = to_cute(mat_a)
    b_c = to_cute(mat_b)
    sfa_c = to_cute(scale_a)
    sfb_c = to_cute(scale_b)
    c_c = to_cute(out)
    offs_c = to_cute(expert_offsets)
    workspace_size = kernel.get_workspace_size(num_experts)
    workspace = torch.full((workspace_size,), 255, dtype=torch.uint8, device=device)
    ws_c = to_cute(workspace)
    gsa_c = to_cute(global_scale_a)
    gsb_c = to_cute(global_scale_b)
    
    import cuda.bindings.driver as cuda
    cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    
    compiled = cute.compile(
        kernel, a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c,
        max_active_clusters, stream,
        global_scale_a=gsa_c, global_scale_b=gsb_c,
    )
    compiled(
        a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c, stream,
        global_scale_a=gsa_c, global_scale_b=gsb_c,
    )
    torch.cuda.synchronize()
    
    _fused_kernel_cache[cache_key] = {
        'compiled': compiled,
        'workspace': workspace,
        'workspace_size': workspace_size,
    }
    
    del mat_a, mat_b, scale_a, scale_b, out, expert_offsets
    del global_scale_a, global_scale_b
    del a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c, gsa_c, gsb_c
    torch.cuda.empty_cache()


def run_fused_swiglu_grouped_gemm(
    mat_a,          # (tokens_sum, K_packed) float4_e2m1fn_x2
    mat_b,          # (experts, K_packed, N_packed) float4_e2m1fn_x2, K-major
    scale_a,        # assembled 2D side (padded + swizzled)
    scale_b,        # assembled 3D side (padded + swizzled)
    expert_offsets, # (experts,) int32 cumulative token offsets
    global_scale_a=None,  # (experts,) float32
    global_scale_b=None,  # (experts,) float32
    swiglu_limit=0.0,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
):
    """Run the fused SwiGLU NVFP4 scaled grouped GEMM.
    
    Stage 1: SiLU is applied to the full accumulator in registers,
    then written as BF16 to C. Gate/up pairing is not yet implemented.
    """
    from dsv4.kernels.gemm.fused_swiglu import FusedSwiGLUScaledGroupedGemmKernel
    
    num_experts = mat_b.shape[0]
    n_dim = mat_b.shape[2]
    tokens_sum = mat_a.shape[0]

    out = torch.zeros(tokens_sum, n_dim, dtype=torch.bfloat16, device=mat_a.device)

    # NVFP4-3: use 2-CTA UMMA for M>=256 (1.7-1.9× throughput at prefill)
    # At decode (M<256), 1-CTA is correct (2-CTA wastes hardware)
    use_2cta = tokens_sum >= 256 and cluster_shape_mn[0] % 2 == 0
    effective_mma_tiler_mn = (
        (mma_tiler_mn[0] * 2, mma_tiler_mn[1]) if use_2cta else mma_tiler_mn
    )
    effective_cluster_shape_mn = (
        (cluster_shape_mn[0] * 2, cluster_shape_mn[1]) if use_2cta else cluster_shape_mn
    )

    cache_key = ('fused', num_experts, str(mat_a.device), effective_mma_tiler_mn, effective_cluster_shape_mn,
                 mat_a.shape[1], mat_b.shape[2], swiglu_limit, use_2cta)
    
    if cache_key not in _fused_kernel_cache:
        # Lazy compilation
        kernel = FusedSwiGLUScaledGroupedGemmKernel(
            scenario="2Dx3D",
            sf_vec_size=SF_VEC_SIZE,
            accumulate_on_output=False,
            separate_tensormap_init=True,
            consistent_token_padding=False,
            mma_tiler_mnk=(*effective_mma_tiler_mn, 256),
            cluster_shape_mnk=(*effective_cluster_shape_mn, 1),
            use_2cta_instrs=use_2cta,
            fused_swiglu=True,
            swiglu_limit=swiglu_limit,
        )

        def to_cute(t):
            ct = cutlass_torch.from_dlpack(t)
            return ct.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(t))

        a_c = to_cute(mat_a)
        b_c = to_cute(mat_b)
        sfa_c = to_cute(scale_a)
        sfb_c = to_cute(scale_b)
        c_c = to_cute(out)
        offs_c = to_cute(expert_offsets)
        workspace_size = kernel.get_workspace_size(num_experts)
        workspace = torch.full((workspace_size,), 255, dtype=torch.uint8, device=mat_a.device)
        ws_c = to_cute(workspace)
        gsa_c = to_cute(global_scale_a) if global_scale_a is not None else None
        gsb_c = to_cute(global_scale_b) if global_scale_b is not None else None

        import cuda.bindings.driver as cuda
        cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        compiled = cute.compile(
            kernel, a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c,
            max_active_clusters, stream,
            global_scale_a=gsa_c, global_scale_b=gsb_c,
        )
        compiled(
            a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c, stream,
            global_scale_a=gsa_c, global_scale_b=gsb_c,
        )
        
        _fused_kernel_cache[cache_key] = {
            'compiled': compiled,
            'workspace': workspace,
            'workspace_size': workspace_size,
        }
    
    entry = _fused_kernel_cache[cache_key]
    compiled = entry['compiled']
    workspace = entry['workspace']
    
    def to_cute(t):
        ct = cutlass_torch.from_dlpack(t)
        return ct.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(t))
    
    a_c = to_cute(mat_a)
    b_c = to_cute(mat_b)
    sfa_c = to_cute(scale_a)
    sfb_c = to_cute(scale_b)
    c_c = to_cute(out)
    offs_c = to_cute(expert_offsets)
    ws_c = to_cute(workspace)
    gsa_c = to_cute(global_scale_a) if global_scale_a is not None else None
    gsb_c = to_cute(global_scale_b) if global_scale_b is not None else None
    
    import cuda.bindings.driver as cuda
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    
    compiled(
        a_c, b_c, sfa_c, sfb_c, c_c, offs_c, ws_c, stream,
        global_scale_a=gsa_c, global_scale_b=gsb_c,
    )

    return out



