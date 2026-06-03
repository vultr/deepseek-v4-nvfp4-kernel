/**
 * Quantize FP32 tensor to NVFP4.
 *
 * Same proven pattern as quantize_nvfp4.cu (which reads BF16),
 * but takes FP32 input directly — avoids BF16 intermediate.
 *
 * This is the correct path for compressor output → NVFP4:
 *   Compressor produces FP32 → this kernel → NVFP4 stored in KV cache
 *   No BF16 anywhere in the pipeline.
 *
 * Two-kernel approach (proven correct in fused_amax_quantize.cu):
 *   Kernel 1: amax_gsa_fp32 — compute per-row gsa from FP32 input (GPU-only)
 *   Kernel 2: quantize_nvfp4_from_fp32 — quantize FP32 → NVFP4 using GPU gsa buffer
 *
 * Grid: (N/16, M, 1) — each CTA processes one 16-element block in one row.
 * Block: 16 threads (1 thread per element, warp amax reduction).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp8.hpp>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cstdint>
#include <cfloat>

__device__ __forceinline__ int half_step_to_e2m1(int hs) {
    if (hs <= 4) return hs;
    if (hs <= 5) return 4;
    if (hs <= 7) return 5;
    if (hs <= 10) return 6;
    return 7;
}

// ===========================================================================
// Kernel 1: Compute per-row amax → gsa from FP32 input
// Same pattern as amax_gsa.cu but for FP32 (not BF16) input
// ===========================================================================

__global__ void compute_amax_gsa_fp32_kernel(
    const float* __restrict__ input,
    int M, int N,
    float divisor,
    float* __restrict__ out_gsa
) {
    int m = blockIdx.x;
    if (m >= M) return;

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < N; i += 256) {
        float v = fabsf(input[m * N + i]);
        local_max = fmaxf(local_max, v);
    }

    // Warp-level reduction
    for (int offset = 128; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    // Block-level reduction using shared memory
    __shared__ float s_max[8];
    if (threadIdx.x % 32 == 0)
        s_max[threadIdx.x / 32] = local_max;
    __syncthreads();

    if (threadIdx.x < 32) {
        float v = (threadIdx.x < 8) ? s_max[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
        if (threadIdx.x == 0)
            out_gsa[m] = v / divisor;
    }
}

// ===========================================================================
// Kernel 2: Quantize FP32 → NVFP4 using gsa from GPU buffer
// Same proven pattern as quantize_nvfp4_from_buffer_kernel (fused_amax_quantize.cu)
// but reads FP32 instead of BF16
// ===========================================================================

__global__ void quantize_nvfp4_from_fp32_kernel(
    const float* __restrict__ input,
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

    // Step 1: Read 16 FP32 elements and compute block amax
    for (int i = 0; i < 16; i++) {
        int col = n_block * 16 + i;
        if (col < N) {
            vals[i] = input[m * N + col] / gsa;
        } else {
            vals[i] = 0;
        }
        block_amax = fmaxf(block_amax, fabsf(vals[i]));
    }

    // Step 2: Compute FP8 E4M3 block scale (with FP8 round-trip)
    float bsf = block_amax / 6.0f;
    if (block_amax < 6.0f * 0.001953125f) {
        // Zero/underflow block
        bsf = 0;
        for (int i = 0; i < 16; i++) vals[i] = 0;
    }
    __nv_fp8_e4m3 bsf8_obj(bsf);
    float bs = (float)bsf8_obj;  // FP8 round-trip — matches dequant
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

// ===========================================================================
// FP32 GPT-J interleaved RoPE (for compressed KV — no BF16 intermediate)
// Same math as rope_cuda.cu but operates on FP32 directly.
// ===========================================================================

__global__ void rope_fp32_kernel(
    float* __restrict__ x,           // (M, 1, N) FP32 — modified in-place
    const float* __restrict__ cos_c, // (max_pos, rope_dim/2) FP32
    const float* __restrict__ sin_c, // (max_pos, rope_dim/2) FP32
    const int64_t* __restrict__ pos, // (M,) positions
    int N, int rope_dim, bool inverse
) {
    int m = blockIdx.x;
    if (m >= gridDim.x) return;
    int64_t p = pos[m];
    int nope = N - rope_dim;
    for (int i = threadIdx.x; i < rope_dim / 2; i += 256) {
        float c = cos_c[p * (rope_dim / 2) + i];
        float s = sin_c[p * (rope_dim / 2) + i];
        int ev_idx = m * N + nope + 2 * i;
        int od_idx = m * N + nope + 2 * i + 1;
        float ev = x[ev_idx];
        float od = x[od_idx];
        if (inverse) {
            x[ev_idx] = ev * c + od * s;
            x[od_idx] = -ev * s + od * c;
        } else {
            x[ev_idx] = ev * c - od * s;
            x[od_idx] = ev * s + od * c;
        }
    }
}

// ===========================================================================
// FP8 E4M3 quantize FP32 → FP8 (for indexer keys — higher precision)
// ===========================================================================

__global__ void quantize_fp8_e4m3_from_fp32_kernel(
    const float* __restrict__ input,
    int M, int N,
    float* __restrict__ out_scale,  // (M,) per-row scale
    uint8_t* __restrict__ out_fp8   // (M, N) packed FP8 E4M3
) {
    int m = blockIdx.x;
    if (m >= M) return;

    // Per-row amax → scale = amax / 448.0 (E4M3 max = 448)
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < N; i += 256) {
        float v = fabsf(input[m * N + i]);
        local_max = fmaxf(local_max, v);
    }
    for (int offset = 128; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    __shared__ float s_max[8];
    if (threadIdx.x % 32 == 0) s_max[threadIdx.x / 32] = local_max;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < 8) ? s_max[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
        if (threadIdx.x == 0) {
            float scale = v / 448.0f;
            if (scale < 1e-8f) scale = 1e-8f;
            out_scale[m] = scale;
        }
    }
    __syncthreads();

    // Quantize each element
    float scale = out_scale[m];
    float inv_scale = 1.0f / scale;
    for (int i = threadIdx.x; i < N; i += 256) {
        float v = input[m * N + i] * inv_scale;
        v = fmaxf(v, -448.0f);
        v = fminf(v, 448.0f);
        __nv_fp8_e4m3 obj(v);
        out_fp8[m * N + i] = *(uint8_t*)&obj;
    }
}

// ===========================================================================
// FP8 E4M3 dequant → BF16 (for indexer key gather)
// ===========================================================================

__global__ void dequant_fp8_e4m3_kernel(
    const uint8_t* __restrict__ fp8_data,
    const float* __restrict__ scale_data,
    int M, int N,
    __nv_bfloat16* __restrict__ output
) {
    int m = blockIdx.x;
    if (m >= M) return;
    float scale = scale_data[m];
    for (int i = threadIdx.x; i < N; i += 256) {
        uint8_t byte = fp8_data[m * N + i];
        __nv_fp8_e4m3 val;
        memcpy(&val, &byte, 1);
        float v = (float)val * scale;
        output[m * N + i] = __float2bfloat16(v);
    }
}

__global__ void dequant_fp8_e4m3_selective_kernel(
    const uint8_t* __restrict__ fp8_data,
    const float* __restrict__ scale_data,
    const int32_t* __restrict__ indices,
    int K, int N,
    __nv_bfloat16* __restrict__ output
) {
    int k = blockIdx.x;
    if (k >= K) return;
    int src_row = indices[k];
    float scale = scale_data[src_row];
    for (int i = threadIdx.x; i < N; i += 256) {
        uint8_t byte = fp8_data[src_row * N + i];
        __nv_fp8_e4m3 val;
        memcpy(&val, &byte, 1);
        float v = (float)val * scale;
        output[k * N + i] = __float2bfloat16(v);
    }
}

// ===========================================================================
// PyTorch bindings
// ===========================================================================

torch::Tensor compute_amax_gsa_fp32_cuda(torch::Tensor input, double divisor) {
    int M = input.size(0);
    int N = input.size(1);
    auto out_gsa = torch::zeros({M}, input.options().dtype(torch::kFloat32));
    compute_amax_gsa_fp32_kernel<<<M, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), M, N, (float)divisor, out_gsa.data_ptr<float>());
    return out_gsa;
}

std::tuple<torch::Tensor, torch::Tensor> quantize_nvfp4_from_fp32_cuda(
    torch::Tensor input, torch::Tensor gsa_buffer
) {
    int M = input.size(0);
    int N = input.size(1);
    TORCH_CHECK(N % 16 == 0, "N must be a multiple of 16 for NVFP4 quantization");
    TORCH_CHECK(gsa_buffer.size(0) == M, "gsa_buffer size must match M");
    auto opts = input.options();
    auto out_fp4 = torch::zeros({M, N / 2}, opts.dtype(torch::kUInt8));
    auto out_sf = torch::zeros({M, N / 16}, opts.dtype(torch::kUInt8));
    int nb = N / 16;
    dim3 grid(nb, M);
    dim3 block(16);
    quantize_nvfp4_from_fp32_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), M, N, gsa_buffer.data_ptr<float>(),
        out_fp4.data_ptr<uint8_t>(), out_sf.data_ptr<uint8_t>()
    );
    return {out_fp4.view(torch::kFloat4_e2m1fn_x2), out_sf.view(torch::kFloat8_e4m3fn)};
}

std::tuple<torch::Tensor, torch::Tensor> quantize_fp8_e4m3_from_fp32_cuda(
    torch::Tensor input
) {
    int M = input.size(0);
    int N = input.size(1);
    auto opts = input.options();
    auto out_scale = torch::zeros({M}, opts.dtype(torch::kFloat32));
    auto out_fp8 = torch::zeros({M, N}, opts.dtype(torch::kUInt8));
    quantize_fp8_e4m3_from_fp32_kernel<<<M, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), M, N,
        out_scale.data_ptr<float>(), out_fp8.data_ptr<uint8_t>()
    );
    return {out_fp8.view(torch::kFloat8_e4m3fn), out_scale};
}

torch::Tensor dequant_fp8_e4m3_cuda(
    torch::Tensor fp8_data, torch::Tensor scale_data
) {
    int M = fp8_data.size(0);
    int N = fp8_data.size(1);
    auto output = torch::zeros({M, N}, fp8_data.options().dtype(torch::kBFloat16));
    dequant_fp8_e4m3_kernel<<<M, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        fp8_data.data_ptr<uint8_t>(), scale_data.data_ptr<float>(), M, N,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>())
    );
    return output;
}

torch::Tensor dequant_fp8_e4m3_selective_cuda(
    torch::Tensor fp8_data, torch::Tensor scale_data, torch::Tensor indices
) {
    int K = indices.size(0);
    int N = fp8_data.size(1);
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
    auto output = torch::zeros({K, N}, fp8_data.options().dtype(torch::kBFloat16));
    dequant_fp8_e4m3_selective_kernel<<<K, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        fp8_data.data_ptr<uint8_t>(), scale_data.data_ptr<float>(),
        indices.data_ptr<int32_t>(), K, N,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>())
    );
    return output;
}

void rope_fp32_cuda(
    torch::Tensor x,           // (M, N) FP32 — modified in-place
    torch::Tensor positions,   // (M,) int64
    torch::Tensor cos_cache,   // (max_pos, rope_dim/2) FP32
    torch::Tensor sin_cache,   // (max_pos, rope_dim/2) FP32
    int64_t rope_dim,
    bool inverse
) {
    int M = x.size(0);
    int N = x.size(1);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    rope_fp32_kernel<<<M, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        cos_cache.data_ptr<float>(),
        sin_cache.data_ptr<float>(),
        positions.data_ptr<int64_t>(),
        N, (int)rope_dim, inverse
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_amax_gsa_fp32", &compute_amax_gsa_fp32_cuda,
          "Compute per-row gsa from FP32 input (GPU-only, no CPU sync)");
    m.def("quantize_nvfp4_from_fp32", &quantize_nvfp4_from_fp32_cuda,
          "Quantize FP32 → NVFP4 using gsa from GPU buffer");
    m.def("quantize_fp8_e4m3_from_fp32", &quantize_fp8_e4m3_from_fp32_cuda,
          "Quantize FP32 → FP8 E4M3 (for indexer keys)");
    m.def("dequant_fp8_e4m3", &dequant_fp8_e4m3_cuda,
          "Dequant FP8 E4M3 → BF16");
    m.def("dequant_fp8_e4m3_selective", &dequant_fp8_e4m3_selective_cuda,
          "Selective dequant FP8 E4M3 → BF16 (for CSA indexer gather)");
    m.def("rope_fp32", &rope_fp32_cuda,
          "FP32 GPT-J interleaved RoPE (for compressed KV)");
}
