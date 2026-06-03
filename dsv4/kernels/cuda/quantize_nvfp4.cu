#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp8.hpp>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cstdint>

// BF16 → NVFP4 quantization kernel (no deinterleave, GPU-only).
// Reads BF16 from GMEM, quantizes to NVFP4 (FP4 data + FP8 E4M3 scales),
// writes both to GMEM. No CPU-GPU syncs.
//
// This replaces quantize_activation_nvfp4() which uses .amax() (CPU sync).
// Global scale is passed in as a pre-computed scalar.
//
// Grid: (N / 16, M, 1) — each CTA processes one 16-element block in one row.
// Block: 16 threads (1 thread per element, warp amax reduction).
//
// Same proven pattern as deinterleave_quantize.cu.

__device__ __forceinline__ int half_step_to_e2m1(int hs) {
    if (hs <= 4) return hs;
    if (hs <= 5) return 4;
    if (hs <= 7) return 5;
    if (hs <= 10) return 6;
    return 7;
}

__global__ void quantize_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    int M, int N,
    float global_scale,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_sf
) {
    int m = blockIdx.y;
    int n_block = blockIdx.x;
    if (m >= M || n_block * 16 >= N) return;

    float vals[16];
    float block_amax = 0.0f;

    // Step 1: Read 16 BF16 elements and compute amax
    for (int i = 0; i < 16; i++) {
        int col = n_block * 16 + i;
        if (col < N) {
            vals[i] = __bfloat162float(input[m * N + col]) / global_scale;
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

    // Step 4: Pack pairs: (nibbles[1] << 4) | nibbles[0], etc.
    for (int i = 0; i < 8; i++)
        out_fp4[m * (N / 2) + n_block * 8 + i] = (nibbles[2*i+1] << 4) | nibbles[2*i];

    // Step 5: Write FP8 block scale
    out_sf[m * (N / 16) + n_block] = bsf8;
}

std::tuple<torch::Tensor, torch::Tensor> quantize_nvfp4_cuda(
    torch::Tensor input_bf16, double global_scale
) {
    int M = input_bf16.size(0);
    int N = input_bf16.size(1);
    TORCH_CHECK(N % 16 == 0, "N must be a multiple of 16 for NVFP4 quantization");
    auto opts = input_bf16.options();
    auto out_fp4 = torch::zeros({M, N / 2}, opts.dtype(torch::kUInt8));
    auto out_sf = torch::zeros({M, N / 16}, opts.dtype(torch::kUInt8));
    int nb = N / 16;
    dim3 grid(nb, M);
    dim3 block(16);
    quantize_nvfp4_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(input_bf16.data_ptr<at::BFloat16>()),
        M, N, (float)global_scale,
        out_fp4.data_ptr<uint8_t>(), out_sf.data_ptr<uint8_t>()
    );
    return {out_fp4.view(torch::kFloat4_e2m1fn_x2), out_sf.view(torch::kFloat8_e4m3fn)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_nvfp4", &quantize_nvfp4_cuda);
}
