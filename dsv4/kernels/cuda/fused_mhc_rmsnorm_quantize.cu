/**
 * fused_mhc_rmsnorm_quantize.cu
 *
 * Fused mHC pre_block + RMSNorm + NVFP4 quantize.
 * Replaces: bmm (1 launch) + rmsnorm (4+ launches) + quantize (2 launches)
 * with just 2 kernel launches.
 *
 * For decode (T=1): x_in = sum_j A[j] * X[j, :] — weighted sum of n_hc streams
 * Then: RMSNorm(x_in, weight) → quantize to NVFP4
 *
 * Two-kernel approach (same pattern as fused_rmsnorm_quantize.cu):
 *   Kernel 1: mhc_rmsnorm_amax_gsa — compute x_in via bmm, then RMS + amax → gsa
 *   Kernel 2: mhc_rmsnorm_quantize_nvfp4 — normalize + quantize using GPU-computed gsa
 *
 * Usage: 2 sites per layer (attn + ffn) × 61 layers = 122 calls/step
 * Each site saves ~5 launches → ~610 launches/token eliminated
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

// E2M1 half-step → index (same as quantize_nvfp4.cu)
__device__ __forceinline__ int half_step_to_e2m1(int hs) {
    if (hs <= 4) return hs;
    if (hs <= 5) return 4;
    if (hs <= 7) return 5;
    if (hs <= 10) return 6;
    return 7;
}

// ============================================================================
// Kernel 1: mHC bmm + RMS + amax → gsa + inv_rms
// ============================================================================
// Input: X_l (M, n_hc, N) BF16, A_l (M, n_hc) BF16, norm_weight (N,) FP32
// For T=1 decode: M=1, n_hc=4, N=7168
//
// Each block handles one row (one token).
// The bmm: x_in = sum_j A[j] * X[j, :] is a weighted sum of n_hc streams.
// For n_hc=4: x_in = A[0]*X[0,:] + A[1]*X[1,:] + A[2]*X[2,:] + A[3]*X[3,:]

__global__ void mhc_rmsnorm_amax_gsa_kernel(
    const __nv_bfloat16* __restrict__ X_l,    // (M, n_hc, N) BF16
    const __nv_bfloat16* __restrict__ A_l,    // (M, n_hc) BF16
    const float* __restrict__ norm_weight,     // (N,) FP32
    float* __restrict__ gsa_out,               // (M,) FP32
    float* __restrict__ inv_rms_out,           // (M,) FP32
    const int M,
    const int n_hc,
    const int N,
    const float eps,
    const float divisor
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const __nv_bfloat16* X_row = X_l + (size_t)row * n_hc * N;
    const __nv_bfloat16* A_row = A_l + (size_t)row * n_hc;

    // Load A coefficients (n_hc=4 typically, always small)
    float a_coeff[4];  // n_hc max = 4
    for (int j = 0; j < n_hc && j < 4; j++) {
        a_coeff[j] = __bfloat162float(A_row[j]);
    }

    // Sub-pass 1: compute x_in = sum_j A[j] * X[j, :] and sum(x_in^2)
    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float x_in_val = 0.0f;
        for (int j = 0; j < n_hc && j < 4; j++) {
            x_in_val += a_coeff[j] * __bfloat162float(X_row[(size_t)j * N + col]);
        }
        sum_sq += x_in_val * x_in_val;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    const int num_warps = blockDim.x / warpSize;
    __shared__ float s_sum_sq[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) s_sum_sq[warp_id] = sum_sq;
    __syncthreads();

    float row_sum_sq = 0.0f;
    if (warp_id == 0) {
        row_sum_sq = (lane < num_warps) ? s_sum_sq[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            row_sum_sq += __shfl_down_sync(0xFFFFFFFF, row_sum_sq, offset);
        }
    }

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        float rms = sqrtf(row_sum_sq / N + eps);
        s_inv_rms = 1.0f / fmaxf(rms, 1e-8f);
    }
    __syncthreads();
    float inv_rms = s_inv_rms;

    // Sub-pass 2: amax of (x_in * inv_rms * weight)
    float row_amax = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float x_in_val = 0.0f;
        for (int j = 0; j < n_hc && j < 4; j++) {
            x_in_val += a_coeff[j] * __bfloat162float(X_row[(size_t)j * N + col]);
        }
        float normalized = x_in_val * inv_rms * norm_weight[col];
        float abs_val = fabsf(normalized);
        if (abs_val > row_amax) row_amax = abs_val;
    }

    // Warp-level reduce max
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        row_amax = fmaxf(row_amax, __shfl_down_sync(0xFFFFFFFF, row_amax, offset));
    }

    __shared__ float s_amax[32];
    if (lane == 0) s_amax[warp_id] = row_amax;
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
// Kernel 2: mHC bmm + normalize + quantize using GPU-computed gsa
// ============================================================================

__global__ void mhc_rmsnorm_quantize_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ X_l,    // (M, n_hc, N) BF16
    const __nv_bfloat16* __restrict__ A_l,    // (M, n_hc) BF16
    const float* __restrict__ norm_weight,     // (N,) FP32
    const float* __restrict__ gsa,             // (M,) FP32
    const float* __restrict__ inv_rms,         // (M,) FP32
    uint8_t* __restrict__ out_fp4,             // (M, N//2) FP4 packed
    uint8_t* __restrict__ out_sf,              // (M, N//16) E4M3 block scales
    const int M,
    const int n_hc,
    const int N
) {
    const int row = blockIdx.y;
    const int n_block = blockIdx.x;
    if (row >= M) return;
    if (n_block * 16 >= N) return;

    const __nv_bfloat16* X_row = X_l + (size_t)row * n_hc * N;
    const __nv_bfloat16* A_row = A_l + (size_t)row * n_hc;
    float row_gsa = gsa[row];
    float row_inv_rms = inv_rms[row];

    // Load A coefficients
    float a_coeff[4];
    for (int j = 0; j < n_hc && j < 4; j++) {
        a_coeff[j] = __bfloat162float(A_row[j]);
    }

    // Step 1: Compute x_in for 16 elements, normalize, compute block amax
    float vals[16];
    float block_amax = 0.0f;
    const int col_base = n_block * 16;

    for (int i = 0; i < 16; i++) {
        int col = col_base + i;
        if (col < N) {
            float x_in_val = 0.0f;
            for (int j = 0; j < n_hc && j < 4; j++) {
                x_in_val += a_coeff[j] * __bfloat162float(X_row[(size_t)j * N + col]);
            }
            float normalized = x_in_val * row_inv_rms * norm_weight[col];  // RMSNorm
            vals[i] = normalized;
            float av = fabsf(normalized);
            if (av > block_amax) block_amax = av;
        } else {
            vals[i] = 0.0f;
        }
    }

    // Step 2: Compute FP8 E4M3 block scale (same as quantize_nvfp4.cu)
    float bsf = block_amax / (row_gsa * 6.0f);
    if (block_amax < row_gsa * 6.0f * 0.001953125f) {
        bsf = 0.0f;
        for (int i = 0; i < 16; i++) vals[i] = 0.0f;
    }
    __nv_fp8_e4m3 bsf8_obj(bsf);
    float bs = (float)bsf8_obj;
    uint8_t bsf8;
    memcpy(&bsf8, &bsf8_obj, 1);

    // Step 3: Quantize to FP4 E2M1 (same as quantize_nvfp4.cu)
    uint8_t nibbles[16];
    for (int i = 0; i < 16; i++) {
        if (bs < 1e-8f) { nibbles[i] = 0; continue; }
        float s = vals[i] / (row_gsa * bs);
        int hs = __float2int_rn(fminf(fabsf(s), 6.0f) * 2.0f);
        if (hs > 12) hs = 12;
        int idx = half_step_to_e2m1(hs);
        if (s < 0) idx += 8;
        nibbles[i] = idx;
    }

    // Step 4: Pack pairs (same as quantize_nvfp4.cu)
    for (int i = 0; i < 8; i++) {
        out_fp4[(size_t)row * (N / 2) + n_block * 8 + i] =
            (nibbles[2 * i + 1] << 4) | nibbles[2 * i];
    }

    // Step 5: Write FP8 block scale
    out_sf[(size_t)row * (N / 16) + n_block] = bsf8;
}

// ============================================================================
// PyTorch bridge
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mhc_rmsnorm_quantize_nvfp4_cuda(
    torch::Tensor X_l,               // (M, n_hc, N) BF16
    torch::Tensor A_l,               // (M, n_hc) BF16
    torch::Tensor norm_weight,       // (N,) FP32
    double eps,
    double divisor
) {
    TORCH_CHECK(X_l.is_contiguous(), "X_l must be contiguous");
    TORCH_CHECK(X_l.scalar_type() == torch::kBFloat16, "X_l must be BF16");
    TORCH_CHECK(A_l.scalar_type() == torch::kBFloat16, "A_l must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat32, "norm_weight must be FP32");

    const int M = X_l.size(0);
    const int n_hc = X_l.size(1);
    const int N = X_l.size(2);
    TORCH_CHECK(N % 16 == 0, "N must be multiple of 16");
    TORCH_CHECK(n_hc <= 4, "n_hc must be <= 4");

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto options = X_l.options();

    auto gsa = torch::empty({M}, options.dtype(torch::kFloat32));
    auto inv_rms = torch::empty({M}, options.dtype(torch::kFloat32));
    auto x_fp4 = torch::empty({M, N / 2}, options.dtype(torch::kUInt8));
    auto x_sf = torch::empty({M, N / 16}, options.dtype(torch::kUInt8));

    // Kernel 1: mHC bmm + RMS + amax → gsa (1 block per row)
    const int threads1 = 256;
    mhc_rmsnorm_amax_gsa_kernel<<<M, threads1, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(X_l.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(A_l.data_ptr<at::BFloat16>()),
        norm_weight.data_ptr<float>(),
        gsa.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        M, n_hc, N, (float)eps, (float)divisor
    );

    // Kernel 2: bmm + normalize + quantize
    const int n_blocks = N / 16;
    dim3 grid2(n_blocks, M);
    const int threads2 = 16;
    mhc_rmsnorm_quantize_nvfp4_kernel<<<grid2, threads2, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(X_l.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(A_l.data_ptr<at::BFloat16>()),
        norm_weight.data_ptr<float>(),
        gsa.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        x_fp4.data_ptr<uint8_t>(),
        x_sf.data_ptr<uint8_t>(),
        M, n_hc, N
    );

    return std::make_tuple(
        x_fp4.view(torch::kFloat4_e2m1fn_x2),
        x_sf.view(torch::kFloat8_e4m3fn),
        gsa,
        inv_rms
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhc_rmsnorm_quantize_nvfp4", &mhc_rmsnorm_quantize_nvfp4_cuda,
          "Fused mHC pre_block + RMSNorm + NVFP4 quantize");
}
