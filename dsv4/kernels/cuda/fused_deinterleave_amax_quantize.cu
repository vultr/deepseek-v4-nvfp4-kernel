/**
 * Fused deinterleave + amax + gsa + NVFP4 quantize kernel.
 *
 * Single kernel launch that:
 *   1. De-interleaves fused L1 SwiGLU output (extracts odd groups)
 *   2. Computes row-wise amax of the de-interleaved values (GPU-only)
 *   3. Derives gsa = max(amax) / divisor
 *   4. Quantizes to NVFP4 (FP4 data + FP8 E4M3 block scales)
 *   5. Writes gsa to a GPU buffer for downstream L2 GEMM global_scale_a
 *
 * This replaces the two-step path in Nvfp4MoE's fused_swiglu path:
 *   compute_amax_gsa_gpu(l1_out_real) → .item() sync
 *   deinterleave_quantize_nvfp4_cuda(l1_out_real, ..., gsa) → separate kernel
 *
 * Now: zero CPU-GPU syncs. gsa stays on GPU. Single kernel launch.
 *
 * Grid: (intermediate / 16, M, 1) — each CTA processes one 16-element block.
 * Shared memory: n_blocks * sizeof(float) for cross-CTA amax reduction.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp8.hpp>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cstdint>

__device__ __forceinline__ int half_step_to_e2m1(int hs) {
    if (hs <= 4) return hs;
    if (hs <= 5) return 4;
    if (hs <= 7) return 5;
    if (hs <= 10) return 6;
    return 7;
}

__global__ void fused_deinterleave_amax_quantize_kernel(
    const __nv_bfloat16* __restrict__ fused,
    int M, int N, int intermediate, int granularity,
    float divisor,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_sf,
    float* __restrict__ out_gsa  // (M,) GPU buffer — gsa per row
) {
    int m = blockIdx.y;
    int n_block = blockIdx.x;
    int n_blocks = gridDim.x;
    if (m >= M || n_block * 16 >= intermediate) return;

    extern __shared__ float s_amax[];

    // Step 1: De-interleave and compute local amax
    float vals[16];
    float block_amax = 0.0f;

    for (int i = 0; i < 16; i++) {
        int nd = n_block * 16 + i;
        if (nd >= intermediate) { vals[i] = 0; continue; }
        // Map de-interleaved position to fused position
        int group = 2 * (nd / granularity) + 1;  // odd group = SwiGLU
        int offset = nd % granularity;
        int fc = group * granularity + offset;
        vals[i] = __bfloat162float(fused[m * N + fc]);
        block_amax = fmaxf(block_amax, fabsf(vals[i]));
    }

    // Step 2: Cross-CTA reduction to get row-wide amax
    if (n_block < n_blocks) {
        s_amax[n_block] = block_amax;
    }
    __syncthreads();

    float gsa;
    if (n_block == 0) {
        float row_amax = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            row_amax = fmaxf(row_amax, s_amax[b]);
        }
        gsa = fmaxf(row_amax, 1e-8f) / divisor;
        out_gsa[m] = gsa;
    }
    if (n_block == 0) {
        s_amax[0] = gsa;
    }
    __syncthreads();
    gsa = s_amax[0];

    // Step 3: Quantize — divide by gsa, compute FP8 block scale, quantize to FP4
    for (int i = 0; i < 16; i++) {
        vals[i] = vals[i] / gsa;
    }

    float q_amax = 0.0f;
    for (int i = 0; i < 16; i++) {
        q_amax = fmaxf(q_amax, fabsf(vals[i]));
    }

    float bsf = q_amax / 6.0f;
    if (q_amax < 6.0f * 0.001953125f) {
        bsf = 0;
        for (int i = 0; i < 16; i++) vals[i] = 0;
    }
    __nv_fp8_e4m3 bsf8_obj(bsf);
    float bs = (float)bsf8_obj;
    uint8_t bsf8 = *(uint8_t*)&bsf8_obj;

    uint8_t nibbles[16];
    for (int i = 0; i < 16; i++) {
        if (bs < 1e-8f) { nibbles[i] = 0; continue; }
        float s = vals[i] / bs;
        int hs = __float2int_rn(fminf(fabsf(s), 6.0f) * 2.0f);
        if (hs > 12) hs = 12;
        int idx = half_step_to_e2m1(hs);
        if (s < 0) idx += 8;
        nibbles[i] = idx;
    }

    for (int i = 0; i < 8; i++)
        out_fp4[m * (intermediate / 2) + n_block * 8 + i] = (nibbles[2*i+1] << 4) | nibbles[2*i];

    out_sf[m * (intermediate / 16) + n_block] = bsf8;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_deinterleave_amax_quantize_cuda(
    torch::Tensor fused_bf16, int64_t intermediate, int64_t granularity, double divisor
) {
    int M = fused_bf16.size(0);
    int N = fused_bf16.size(1);
    auto opts = fused_bf16.options();
    auto out_fp4 = torch::zeros({M, (int)intermediate / 2}, opts.dtype(torch::kUInt8));
    auto out_sf = torch::zeros({M, (int)intermediate / 16}, opts.dtype(torch::kUInt8));
    auto out_gsa = torch::zeros({M}, opts.dtype(torch::kFloat32));

    int nb = (int)intermediate / 16;
    dim3 grid(nb, M);
    dim3 block(16);
    int smem_size = nb * sizeof(float);

    fused_deinterleave_amax_quantize_kernel<<<grid, block, smem_size, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(fused_bf16.data_ptr<at::BFloat16>()),
        M, N, (int)intermediate, (int)granularity, (float)divisor,
        out_fp4.data_ptr<uint8_t>(), out_sf.data_ptr<uint8_t>(),
        out_gsa.data_ptr<float>()
    );
    return {out_fp4.view(torch::kFloat4_e2m1fn_x2), out_sf.view(torch::kFloat8_e4m3fn), out_gsa};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_deinterleave_amax_quantize", &fused_deinterleave_amax_quantize_cuda);
}
