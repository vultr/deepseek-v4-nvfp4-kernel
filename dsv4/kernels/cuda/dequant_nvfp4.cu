/**
 * NVFP4 → BF16 dequantization kernels.
 *
 * Converts FP4 (E2M1) data + FP8 (E4M3) block scales + FP32 global scales
 * back to BF16. Used for the FMHA gather path: compressed KV is stored as
 * NVFP4, and dequantized on-the-fly when gathering for attention.
 *
 * Two variants:
 *   1. Full dequant: entire FP4 buffer → BF16 (for HCA dense gather)
 *   2. Selective dequant: only selected rows → BF16 (for CSA top-k gather)
 *
 * Grid layout: (N/16, M) — one CTA per (row, 16-element block).
 * Block size: 16 threads (1 thread per element in the 16-wide block).
 *
 * Memory savings: FP4 is 4× smaller than BF16. At hd=512:
 *   BF16: 512 × 2 = 1024 bytes per entry
 *   NVFP4: 256 + 64 + 4 = 324 bytes per entry (fp4 + sf + gsa)
 *   Savings: ~3.2×
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp8.hpp>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cstdint>

// E2M1 magnitudes: index 0-7 → 0, 0.5, 1, 1.5, 2, 3, 4, 6
__device__ __constant__ float E2M1_LUT[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// ===========================================================================
// Full dequant: entire buffer → BF16
// ===========================================================================

__global__ void dequant_nvfp4_kernel(
    const uint8_t* __restrict__ fp4_data,     // (M, N/2) packed E2M1
    const uint8_t* __restrict__ sf_data,      // (M, N/16) E4M3 block scales (stored as uint8)
    const float* __restrict__ gsa_data,       // (M,) FP32 global scale per row
    __nv_bfloat16* __restrict__ output,       // (M, N) BF16 output
    int M, int N
) {
    int m = blockIdx.y;
    int n_block = blockIdx.x;
    if (m >= M || n_block * 16 >= N) return;

    float gsa = gsa_data[m];

    // Read FP8 E4M3 block scale
    uint8_t sf_byte = sf_data[m * (N / 16) + n_block];
    __nv_fp8_e4m3 sf_val;
    memcpy(&sf_val, &sf_byte, 1);
    float bsf = (float)sf_val;

    // Read 8 packed bytes = 16 E2M1 values
    for (int i = 0; i < 8; i++) {
        uint8_t packed = fp4_data[m * (N / 2) + n_block * 8 + i];
        uint8_t lo_nibble = packed & 0x0F;
        uint8_t hi_nibble = (packed >> 4) & 0x0F;

        // Low nibble
        int lo_idx = lo_nibble & 0x07;
        float lo_sign = (lo_nibble & 0x08) ? -1.0f : 1.0f;
        float lo_val = lo_sign * E2M1_LUT[lo_idx] * bsf * gsa;
        int lo_col = n_block * 16 + 2 * i;
        if (lo_col < N) {
            output[m * N + lo_col] = __float2bfloat16(lo_val);
        }

        // High nibble
        int hi_idx = hi_nibble & 0x07;
        float hi_sign = (hi_nibble & 0x08) ? -1.0f : 1.0f;
        float hi_val = hi_sign * E2M1_LUT[hi_idx] * bsf * gsa;
        int hi_col = n_block * 16 + 2 * i + 1;
        if (hi_col < N) {
            output[m * N + hi_col] = __float2bfloat16(hi_val);
        }
    }
}

// ===========================================================================
// Selective dequant: only dequant selected rows from a larger FP4 buffer
// This is the CSA gather path — dequant only the top-k entries needed by FMHA
// ===========================================================================

__global__ void dequant_nvfp4_selective_kernel(
    const uint8_t* __restrict__ fp4_data,     // (max_comp, N/2) packed E2M1
    const uint8_t* __restrict__ sf_data,      // (max_comp, N/16) E4M3 block scales
    const float* __restrict__ gsa_data,       // (max_comp,) FP32 global scale per row
    const int32_t* __restrict__ indices,      // (K,) int32 — which rows to dequant
    __nv_bfloat16* __restrict__ output,       // (K, N) BF16 output
    int K, int N
) {
    int k = blockIdx.y;       // which selected entry
    int n_block = blockIdx.x; // which 16-element block
    if (k >= K || n_block * 16 >= N) return;

    int src_row = indices[k];
    float gsa = gsa_data[src_row];

    int N_half = N / 2;
    int N_sf = N / 16;

    // Read FP8 E4M3 block scale for this row and block
    uint8_t sf_byte = sf_data[src_row * N_sf + n_block];
    __nv_fp8_e4m3 sf_val;
    memcpy(&sf_val, &sf_byte, 1);
    float bsf = (float)sf_val;

    for (int i = 0; i < 8; i++) {
        uint8_t packed = fp4_data[src_row * N_half + n_block * 8 + i];
        uint8_t lo_nibble = packed & 0x0F;
        uint8_t hi_nibble = (packed >> 4) & 0x0F;

        int lo_idx = lo_nibble & 0x07;
        float lo_sign = (lo_nibble & 0x08) ? -1.0f : 1.0f;
        float lo_val = lo_sign * E2M1_LUT[lo_idx] * bsf * gsa;
        int lo_col = n_block * 16 + 2 * i;
        if (lo_col < N) {
            output[k * N + lo_col] = __float2bfloat16(lo_val);
        }

        int hi_idx = hi_nibble & 0x07;
        float hi_sign = (hi_nibble & 0x08) ? -1.0f : 1.0f;
        float hi_val = hi_sign * E2M1_LUT[hi_idx] * bsf * gsa;
        int hi_col = n_block * 16 + 2 * i + 1;
        if (hi_col < N) {
            output[k * N + hi_col] = __float2bfloat16(hi_val);
        }
    }
}

// ===========================================================================
// PyTorch bindings
// ===========================================================================

torch::Tensor dequant_nvfp4_cuda(
    torch::Tensor fp4_data,    // (M, N/2) uint8 packed E2M1
    torch::Tensor sf_data,     // (M, N/16) uint8 (viewed as E4M3)
    torch::Tensor gsa_data     // (M,) float32 global scale
) {
    int M = fp4_data.size(0);
    int N = fp4_data.size(1) * 2;  // N/2 packed → N actual
    TORCH_CHECK(sf_data.size(0) == M, "sf_data row count must match fp4_data");
    TORCH_CHECK(gsa_data.size(0) == M, "gsa_data row count must match fp4_data");

    auto output = torch::zeros({M, N}, fp4_data.options().dtype(torch::kBFloat16));
    int nb = N / 16;
    dim3 grid(nb, M);
    dim3 block(16);

    dequant_nvfp4_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        fp4_data.data_ptr<uint8_t>(),
        sf_data.data_ptr<uint8_t>(),
        gsa_data.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        M, N
    );
    return output;
}

torch::Tensor dequant_nvfp4_selective_cuda(
    torch::Tensor fp4_data,    // (max_comp, N/2) uint8 packed E2M1
    torch::Tensor sf_data,     // (max_comp, N/16) uint8 (viewed as E4M3)
    torch::Tensor gsa_data,    // (max_comp,) float32 global scale
    torch::Tensor indices      // (K,) int32
) {
    int K = indices.size(0);
    int N = fp4_data.size(1) * 2;  // N/2 packed → N actual
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");

    auto output = torch::zeros({K, N}, fp4_data.options().dtype(torch::kBFloat16));
    int nb = N / 16;
    dim3 grid(nb, K);
    dim3 block(16);

    dequant_nvfp4_selective_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        fp4_data.data_ptr<uint8_t>(),
        sf_data.data_ptr<uint8_t>(),
        gsa_data.data_ptr<float>(),
        indices.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        K, N
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_nvfp4", &dequant_nvfp4_cuda, "NVFP4 → BF16 dequant");
    m.def("dequant_nvfp4_selective", &dequant_nvfp4_selective_cuda, "Selective NVFP4 → BF16 dequant for CSA gather");
}
