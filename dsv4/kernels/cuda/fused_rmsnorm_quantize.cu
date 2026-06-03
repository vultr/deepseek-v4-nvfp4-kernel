/**
 * fused_rmsnorm_quantize.cu
 *
 * Fused RMSNorm + amax + NVFP4 quantize.
 * Replaces: rmsnorm (4+ BF16 launches) + amax (1 launch) + quantize (1 launch)
 * with just 2 kernel launches.
 *
 * Kernel 1: rmsnorm_amax_gsa_kernel
 *   - Compute RMS of each row: rms = sqrt(mean(x^2) + eps)
 *   - Compute row-wise amax of (x / rms * weight) — the normalized output
 *   - Derive gsa = amax / divisor for each row
 *   - Write gsa (per-row) and inv_rms (per-row) to GPU buffers
 *
 * Kernel 2: rmsnorm_quantize_nvfp4_kernel
 *   - Read gsa + inv_rms from GPU buffers (no CPU sync)
 *   - Normalize: val = x * inv_rms * weight
 *   - Quantize to NVFP4 using the same proven path as quantize_nvfp4.cu
 *   - Write FP4 data + E4M3 block scales
 *
 * Quantization is bit-identical to quantize_nvfp4.cu:
 *   - half_step_to_e2m1 for E2M1 encoding
 *   - __nv_fp8_e4m3 for block scale
 *   - (nibbles[2*i+1] << 4) | nibbles[2*i] packing
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp8.hpp>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <cstring>

// FP4 E2M1 half-step → index mapping (same as quantize_nvfp4.cu)
__device__ __forceinline__ int half_step_to_e2m1(int hs) {
    if (hs <= 4) return hs;
    if (hs <= 5) return 4;
    if (hs <= 7) return 5;
    if (hs <= 10) return 6;
    return 7;
}

// ============================================================================
// Kernel 1: Compute RMS + amax of normalized output → gsa per row
// ============================================================================
// Each block processes one row of (M, N).
// Threadblock: blockDim.x threads per row (must be multiple of warpSize).

__global__ void rmsnorm_amax_gsa_kernel(
    const __nv_bfloat16* __restrict__ x,     // (M, N) BF16 row-major
    const float* __restrict__ norm_weight,    // (N,) FP32
    float* __restrict__ gsa_out,              // (M,) FP32 — per-row gsa
    float* __restrict__ inv_rms_out,          // (M,) FP32 — per-row 1/rms (for kernel 2)
    const int M,
    const int N,
    const float eps,
    const float divisor                        // gsa = amax / divisor
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const __nv_bfloat16* x_row = x + (size_t)row * N;

    // Sub-pass 1: compute sum(x^2) for RMS
    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float val = __bfloat162float(x_row[col]);
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Block-level reduction via shared memory
    const int num_warps = blockDim.x / warpSize;
    __shared__ float s_sum_sq[32];  // max 32 warps
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        s_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    // First warp reduces across warps
    float row_sum_sq = 0.0f;
    if (warp_id == 0) {
        row_sum_sq = (lane < num_warps) ? s_sum_sq[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            row_sum_sq += __shfl_down_sync(0xFFFFFFFF, row_sum_sq, offset);
        }
    }

    // Broadcast inv_rms to all threads
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        float rms = sqrtf(row_sum_sq / N + eps);
        s_inv_rms = 1.0f / fmaxf(rms, 1e-8f);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // Sub-pass 2: amax of normalized output (x * inv_rms * weight)
    float row_amax = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float val = __bfloat162float(x_row[col]) * inv_rms * norm_weight[col];
        float abs_val = fabsf(val);
        if (abs_val > row_amax) row_amax = abs_val;
    }

    // Warp-level reduce max
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        row_amax = fmaxf(row_amax, __shfl_down_sync(0xFFFFFFFF, row_amax, offset));
    }

    __shared__ float s_amax[32];
    if (lane == 0) {
        s_amax[warp_id] = row_amax;
    }
    __syncthreads();

    if (warp_id == 0) {
        float global_amax = 0.0f;
        if (lane < num_warps) global_amax = s_amax[lane];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            global_amax = fmaxf(global_amax, __shfl_down_sync(0xFFFFFFFF, global_amax, offset));
        }
        if (lane == 0) {
            gsa_out[row] = fmaxf(global_amax, 1e-8f) / divisor;
            inv_rms_out[row] = inv_rms;
        }
    }
}

// ============================================================================
// Kernel 2: RMSNorm + quantize using gsa from GPU buffer
// ============================================================================
// Same grid as quantize_nvfp4_kernel: (N/16, M, 1)
// Each CTA processes one 16-element microblock in one row.
// Bit-identical quantization to quantize_nvfp4.cu.

__global__ void rmsnorm_quantize_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ x,     // (M, N) BF16 row-major
    const float* __restrict__ norm_weight,    // (N,) FP32
    const float* __restrict__ gsa,            // (M,) FP32 — per-row global scale
    const float* __restrict__ inv_rms,        // (M,) FP32 — per-row 1/rms
    uint8_t* __restrict__ out_fp4,            // (M, N//2) FP4 packed
    uint8_t* __restrict__ out_sf,             // (M, N//16) E4M3 block scales (uint8 view)
    const int M,
    const int N
) {
    const int row = blockIdx.y;
    const int n_block = blockIdx.x;
    if (row >= M) return;
    if (n_block * 16 >= N) return;

    const __nv_bfloat16* x_row = x + (size_t)row * N;
    float row_gsa = gsa[row];
    float row_inv_rms = inv_rms[row];

    // Step 1: Load 16 BF16 elements, normalize (RMSNorm), compute block amax
    float vals[16];
    float block_amax = 0.0f;
    const int col_base = n_block * 16;

    for (int i = 0; i < 16; i++) {
        int col = col_base + i;
        if (col < N) {
            float v = __bfloat162float(x_row[col]);
            v = v * row_inv_rms * norm_weight[col];  // RMSNorm
            vals[i] = v;
            float av = fabsf(v);
            if (av > block_amax) block_amax = av;
        } else {
            vals[i] = 0.0f;
        }
    }

    // Step 2: Compute FP8 E4M3 block scale (same as quantize_nvfp4.cu)
    // block_scale = block_amax / (gsa * 6.0)
    float bsf = block_amax / (row_gsa * 6.0f);
    if (block_amax < row_gsa * 6.0f * 0.001953125f) {
        bsf = 0.0f;
        for (int i = 0; i < 16; i++) vals[i] = 0.0f;
    }
    __nv_fp8_e4m3 bsf8_obj(bsf);
    float bs = (float)bsf8_obj;  // dequantized block scale for FP4 computation
    uint8_t bsf8;
    memcpy(&bsf8, &bsf8_obj, 1);

    // Step 3: Quantize each value to FP4 E2M1 (same as quantize_nvfp4.cu)
    uint8_t nibbles[16];
    for (int i = 0; i < 16; i++) {
        if (bs < 1e-8f) { nibbles[i] = 0; continue; }
        float s = vals[i] / (row_gsa * bs);  // scale by gsa * block_scale
        int hs = __float2int_rn(fminf(fabsf(s), 6.0f) * 2.0f);
        if (hs > 12) hs = 12;
        int idx = half_step_to_e2m1(hs);
        if (s < 0) idx += 8;
        nibbles[i] = idx;
    }

    // Step 4: Pack pairs: (nibbles[2*i+1] << 4) | nibbles[2*i] (same as quantize_nvfp4.cu)
    for (int i = 0; i < 8; i++) {
        out_fp4[(size_t)row * (N / 2) + n_block * 8 + i] =
            (nibbles[2 * i + 1] << 4) | nibbles[2 * i];
    }

    // Step 5: Write FP8 block scale (uint8 view, same as quantize_nvfp4.cu)
    out_sf[(size_t)row * (N / 16) + n_block] = bsf8;
}

// ============================================================================
// PyTorch bridge
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rmsnorm_quantize_nvfp4_cuda(
    torch::Tensor x,               // (M, N) BF16
    torch::Tensor norm_weight,     // (N,) FP32
    double eps,
    double divisor
) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat32, "norm_weight must be FP32");

    const int M = x.size(0);
    const int N = x.size(1);
    TORCH_CHECK(N % 16 == 0, "N must be multiple of 16");

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto options = x.options();

    // Output buffers (uint8, then .view() to FP4/FP8 dtypes)
    auto gsa = torch::empty({M}, options.dtype(torch::kFloat32));
    auto inv_rms = torch::empty({M}, options.dtype(torch::kFloat32));
    auto x_fp4 = torch::empty({M, N / 2}, options.dtype(torch::kUInt8));
    auto x_sf = torch::empty({M, N / 16}, options.dtype(torch::kUInt8));

    // Kernel 1: RMSNorm + amax → gsa (1 block per row)
    const int threads1 = 256;  // 8 warps, handles up to N=8192
    rmsnorm_amax_gsa_kernel<<<M, threads1, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        norm_weight.data_ptr<float>(),
        gsa.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        M, N, (float)eps, (float)divisor
    );

    // Kernel 2: Normalize + quantize (1 block per (row, microblock))
    const int n_blocks = N / 16;
    dim3 grid2(n_blocks, M);
    const int threads2 = 16;  // 1 thread per element in the 16-elem microblock
    rmsnorm_quantize_nvfp4_kernel<<<grid2, threads2, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        norm_weight.data_ptr<float>(),
        gsa.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        x_fp4.data_ptr<uint8_t>(),
        x_sf.data_ptr<uint8_t>(),
        M, N
    );

    // View as proper dtypes (same as quantize_nvfp4.cu)
    return std::make_tuple(
        x_fp4.view(torch::kFloat4_e2m1fn_x2),
        x_sf.view(torch::kFloat8_e4m3fn),
        gsa,
        inv_rms
    );
}

// Standalone kernel 1 entry point (for testing / when only gsa needed)
torch::Tensor rmsnorm_amax_gsa_cuda(
    torch::Tensor x,
    torch::Tensor norm_weight,
    double eps,
    double divisor
) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16");

    const int M = x.size(0);
    const int N = x.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto gsa = torch::empty({M}, x.options().dtype(torch::kFloat32));
    auto inv_rms = torch::empty({M}, x.options().dtype(torch::kFloat32));

    const int threads = 256;
    rmsnorm_amax_gsa_kernel<<<M, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        norm_weight.data_ptr<float>(),
        gsa.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        M, N, (float)eps, (float)divisor
    );

    return gsa;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_quantize_nvfp4", &rmsnorm_quantize_nvfp4_cuda,
          "Fused RMSNorm + amax + quantize to NVFP4");
    m.def("rmsnorm_amax_gsa", &rmsnorm_amax_gsa_cuda,
          "RMSNorm + amax → gsa (kernel 1 only)");
}
