#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp8.hpp>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cstdint>

// De-interleave + NVFP4 quantize kernel for fused SwiGLU output.
// Fused L1 output has [silu(gate)*8, swiglu*8, ...] interleaved at granularity 8.
// This kernel extracts odd 8-col groups (SwiGLU result) and quantizes to NVFP4.
// Single kernel launch, no Python loop, no CPU-GPU sync.

__device__ __forceinline__ int half_step_to_e2m1(int hs) {
    // Matches Python step_to_idx LUT:
    // 0→0, 1→1, 2→2, 3→3, 4→4, 5→4, 6→5, 7→5, 8→6, 9→6, 10→6, 11→7, 12→7
    if (hs <= 4) return hs;
    if (hs <= 5) return 4;
    if (hs <= 7) return 5;
    if (hs <= 10) return 6;
    return 7;
}

__global__ void deinterleave_quantize_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ fused,
    int M, int N, int intermediate, int granularity,
    float global_scale,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_sf
) {
    int m = blockIdx.y;
    int n_block = blockIdx.x;
    if (m >= M || n_block * 16 >= intermediate) return;

    float vals[16];
    float block_amax = 0.0f;

    for (int i = 0; i < 16; i++) {
        int nd = n_block * 16 + i;
        if (nd >= intermediate) { vals[i] = 0; continue; }
        // Map de-interleaved position to fused position
        int group = 2 * (nd / granularity) + 1;  // odd group = SwiGLU
        int offset = nd % granularity;
        int fc = group * granularity + offset;
        float v = __bfloat162float(fused[m * N + fc]);
        vals[i] = v / global_scale;
        block_amax = fmaxf(block_amax, fabsf(vals[i]));
    }

    // Block scale: amax/6 → FP8 E4M3 → float (round-trip)
    float bsf = block_amax / 6.0f;
    if (block_amax < 6.0f * 0.001953125f) {  // underflow threshold
        bsf = 0;
        for (int i = 0; i < 16; i++) vals[i] = 0;
    }
    __nv_fp8_e4m3 bsf8_obj(bsf);
    float bs = (float)bsf8_obj;
    uint8_t bsf8 = *(uint8_t*)&bsf8_obj;

    // Quantize each value to FP4 E2M1
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

    // Pack pairs: (nibbles[1] << 4) | nibbles[0], etc.
    for (int i = 0; i < 8; i++)
        out_fp4[m * (intermediate / 2) + n_block * 8 + i] = (nibbles[2*i+1] << 4) | nibbles[2*i];

    out_sf[m * (intermediate / 16) + n_block] = bsf8;
}

std::tuple<torch::Tensor, torch::Tensor> deinterleave_quantize_nvfp4_cuda(
    torch::Tensor fused_bf16, int64_t intermediate, int64_t granularity, double global_scale
) {
    int M = fused_bf16.size(0);
    int N = fused_bf16.size(1);
    auto opts = fused_bf16.options();
    auto out_fp4 = torch::zeros({M, intermediate / 2}, opts.dtype(torch::kUInt8));
    auto out_sf = torch::zeros({M, intermediate / 16}, opts.dtype(torch::kUInt8));
    int nb = intermediate / 16;
    dim3 grid(nb, M);
    dim3 block(16);
    deinterleave_quantize_nvfp4_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(fused_bf16.data_ptr<at::BFloat16>()),
        M, N, (int)intermediate, (int)granularity, (float)global_scale,
        out_fp4.data_ptr<uint8_t>(), out_sf.data_ptr<uint8_t>()
    );
    return {out_fp4.view(torch::kFloat4_e2m1fn_x2), out_sf.view(torch::kFloat8_e4m3fn)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deinterleave_quantize_nvfp4", &deinterleave_quantize_nvfp4_cuda);
}
