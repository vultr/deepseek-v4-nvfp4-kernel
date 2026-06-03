/**
 * Fused amax + gsa + NVFP4 quantization kernel.
 *
 * Two-phase approach:
 *   Phase 1: Each CTA quantizes its 16-element block (independent).
 *   Phase 2: CTA 0 of each row reduces across all CTAs via atomicMax
 *            to get the row-wide amax, then derives gsa.
 *
 * The amax reduction uses global memory atomics (not shared memory)
 * to correctly handle cross-CTA synchronization within the same kernel.
 * Each CTA writes its block_amax to a global memory buffer.
 * After a grid-sync (via cooperative groups or a second launch),
 * CTA 0 computes the row-wide amax from all block amaxes.
 *
 * Since we can't do a proper grid sync in a single kernel without
 * cooperative groups (which requires special launch), we use a two-kernel
 * approach instead:
 *   Kernel 1: Compute per-block amaxes + quantize to NVFP4.
 *   Kernel 2: Reduce per-block amaxes to per-row gsa.
 *
 * Actually, the simplest correct approach is:
 *   - Compute gsa in a separate lightweight kernel (amax_gsa.cu already does this)
 *   - Pass gsa as a GPU buffer to quantize_nvfp4
 *   - quantize_nvfp4 reads gsa from the GPU buffer instead of a kernel param
 *
 * This file implements the SINGLE-CTA-per-row case (N <= 16).
 * For the general case, use the two-kernel approach.
 *
 * UPDATE: Switched to per-CTA-independent quantize with a global amax
 * reduction. Each CTA computes its own amax, writes to a global buffer.
 * A final pass (CTA 0 per row) reads all amaxes and computes gsa.
 * But this requires grid sync which we don't have.
 *
 * SIMPLEST CORRECT APPROACH:
 * Use the existing amax_gsa.cu kernel to compute gsa on GPU,
 * then pass the GPU tensor to quantize_nvfp4 via a modified kernel
 * that reads global_scale from a GPU buffer instead of a kernel parameter.
 *
 * This file is KEPT but the quantize kernel is modified to accept
 * global_scale from a GPU buffer.
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

/**
 * Quantize kernel that reads global_scale from a GPU buffer.
 * Same as quantize_nvfp4.cu but gsa comes from GMEM, not a kernel param.
 * This enables zero-CPU-sync operation: gsa computed on GPU → passed directly.
 */
__global__ void quantize_nvfp4_from_buffer_kernel(
    const __nv_bfloat16* __restrict__ input,
    int M, int N,
    const float* __restrict__ gsa_buffer,  // (M,) GPU buffer with per-row gsa
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_sf
) {
    int m = blockIdx.y;
    int n_block = blockIdx.x;
    if (m >= M || n_block * 16 >= N) return;

    float gsa = gsa_buffer[m];

    float vals[16];
    float block_amax = 0.0f;

    // Step 1: Read 16 BF16 elements and compute amax
    for (int i = 0; i < 16; i++) {
        int col = n_block * 16 + i;
        if (col < N) {
            vals[i] = __bfloat162float(input[m * N + col]) / gsa;
        } else {
            vals[i] = 0;
        }
        block_amax = fmaxf(block_amax, fabsf(vals[i]));
    }

    // Step 2: Compute FP8 E4M3 block scale
    float bsf = block_amax / 6.0f;
    if (block_amax < 6.0f * 0.001953125f) {
        bsf = 0;
        for (int i = 0; i < 16; i++) vals[i] = 0;
    }
    __nv_fp8_e4m3 bsf8_obj(bsf);
    float bs = (float)bsf8_obj;
    uint8_t bsf8 = *(uint8_t*)&bsf8_obj;

    // Step 3: Quantize each value to FP4 E2M1
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

    // Step 4: Pack pairs
    for (int i = 0; i < 8; i++)
        out_fp4[m * (N / 2) + n_block * 8 + i] = (nibbles[2*i+1] << 4) | nibbles[2*i];

    // Step 5: Write FP8 block scale
    out_sf[m * (N / 16) + n_block] = bsf8;
}

/**
 * Deinterleave + quantize kernel that reads global_scale from a GPU buffer.
 * For the MoE fused_swiglu L2 path.
 */
__global__ void deinterleave_quantize_from_buffer_kernel(
    const __nv_bfloat16* __restrict__ fused,
    int M, int N, int intermediate, int granularity,
    const float* __restrict__ gsa_buffer,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_sf
) {
    int m = blockIdx.y;
    int n_block = blockIdx.x;
    if (m >= M || n_block * 16 >= intermediate) return;

    float gsa = gsa_buffer[m];

    float vals[16];
    float block_amax = 0.0f;

    for (int i = 0; i < 16; i++) {
        int nd = n_block * 16 + i;
        if (nd >= intermediate) { vals[i] = 0; continue; }
        int group = 2 * (nd / granularity) + 1;
        int offset = nd % granularity;
        int fc = group * granularity + offset;
        float v = __bfloat162float(fused[m * N + fc]);
        vals[i] = v / gsa;
        block_amax = fmaxf(block_amax, fabsf(vals[i]));
    }

    float bsf = block_amax / 6.0f;
    if (block_amax < 6.0f * 0.001953125f) {
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

// Python API: quantize with gsa from GPU buffer
std::tuple<torch::Tensor, torch::Tensor> quantize_nvfp4_from_buffer_cuda(
    torch::Tensor input_bf16, torch::Tensor gsa_buffer
) {
    int M = input_bf16.size(0);
    int N = input_bf16.size(1);
    TORCH_CHECK(N % 16 == 0, "N must be a multiple of 16");
    TORCH_CHECK(gsa_buffer.size(0) == M, "gsa_buffer size must match M");
    auto opts = input_bf16.options();
    auto out_fp4 = torch::zeros({M, N / 2}, opts.dtype(torch::kUInt8));
    auto out_sf = torch::zeros({M, N / 16}, opts.dtype(torch::kUInt8));
    int nb = N / 16;
    dim3 grid(nb, M);
    dim3 block(16);
    quantize_nvfp4_from_buffer_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(input_bf16.data_ptr<at::BFloat16>()),
        M, N, gsa_buffer.data_ptr<float>(),
        out_fp4.data_ptr<uint8_t>(), out_sf.data_ptr<uint8_t>()
    );
    return {out_fp4.view(torch::kFloat4_e2m1fn_x2), out_sf.view(torch::kFloat8_e4m3fn)};
}

// Python API: deinterleave + quantize with gsa from GPU buffer
std::tuple<torch::Tensor, torch::Tensor> deinterleave_quantize_from_buffer_cuda(
    torch::Tensor fused_bf16, int64_t intermediate, int64_t granularity, torch::Tensor gsa_buffer
) {
    int M = fused_bf16.size(0);
    int N = fused_bf16.size(1);
    auto opts = fused_bf16.options();
    auto out_fp4 = torch::zeros({M, (int)intermediate / 2}, opts.dtype(torch::kUInt8));
    auto out_sf = torch::zeros({M, (int)intermediate / 16}, opts.dtype(torch::kUInt8));
    int nb = (int)intermediate / 16;
    dim3 grid(nb, M);
    dim3 block(16);
    deinterleave_quantize_from_buffer_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(fused_bf16.data_ptr<at::BFloat16>()),
        M, N, (int)intermediate, (int)granularity, gsa_buffer.data_ptr<float>(),
        out_fp4.data_ptr<uint8_t>(), out_sf.data_ptr<uint8_t>()
    );
    return {out_fp4.view(torch::kFloat4_e2m1fn_x2), out_sf.view(torch::kFloat8_e4m3fn)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_nvfp4_from_buffer", &quantize_nvfp4_from_buffer_cuda);
    m.def("deinterleave_quantize_from_buffer", &deinterleave_quantize_from_buffer_cuda);
}
