/**
 * DSV4 B1 — FP8 attention input/output preparation kernels.
 *
 * These are deliberately tiny launch-count reducers for the mixed-precision
 * FMHA path:
 *   - quantize Q noPE dims BF16 -> FP8_E4M3 with a per-(batch,head,row) scale
 *   - keep Q RoPE dims BF16
 *   - gather compressed KV noPE bytes/scales and RoPE BF16 without global dequant
 *   - quantize the SWA noPE tail BF16 -> FP8_E4M3 in the same gather kernel
 *
 * No PyTorch fallback and no FP8->BF16 global staging for noPE KV.
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

static constexpr float E4M3_MAX = 448.0f;

__device__ __forceinline__ float bf16_load(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

__device__ __forceinline__ uint8_t fp8_e4m3_from_f32(float x) {
    x = fminf(fmaxf(x, -E4M3_MAX), E4M3_MAX);
    __nv_fp8_e4m3 v(x);
    return *reinterpret_cast<uint8_t*>(&v);
}

__global__ void quantize_q_fp8_split_kernel(
    const __nv_bfloat16* __restrict__ q,   // (B,H,T,HD)
    uint8_t* __restrict__ q_nope_fp8,      // (B,H,T,NOPE)
    float* __restrict__ q_nope_scale,      // (B,H,T)
    __nv_bfloat16* __restrict__ q_rope,    // (B,H,T,ROPE)
    int rows, int hd, int nope, int rope
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* q_row = q + (int64_t)row * hd;
    uint8_t* out8 = q_nope_fp8 + (int64_t)row * nope;
    __nv_bfloat16* outrope = q_rope + (int64_t)row * rope;

    float local_max = 0.0f;
    for (int c = threadIdx.x; c < nope; c += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(bf16_load(q_row + c)));
    }

    // block reduction over 256 threads
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    __shared__ float warp_max[8];
    if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = local_max;
    __syncthreads();
    float amax = 0.0f;
    if (threadIdx.x < 32) {
        amax = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_max[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, off));
        if (threadIdx.x == 0) {
            float scale = amax / E4M3_MAX;
            if (scale < 1e-8f) scale = 1e-8f;
            q_nope_scale[row] = scale;
        }
    }
    __syncthreads();

    float scale = q_nope_scale[row];
    float inv_scale = 1.0f / scale;
    for (int c = threadIdx.x; c < nope; c += blockDim.x) {
        out8[c] = fp8_e4m3_from_f32(bf16_load(q_row + c) * inv_scale);
    }
    for (int c = threadIdx.x; c < rope; c += blockDim.x) {
        outrope[c] = q_row[nope + c];
    }
}

__global__ void copy_comp_rows_kernel(
    const uint8_t* __restrict__ comp_nope_fp8,
    const float* __restrict__ comp_nope_scale,
    const __nv_bfloat16* __restrict__ comp_rope,
    const int32_t* __restrict__ indices,  // optional; nullptr => row i
    uint8_t* __restrict__ out_nope_fp8,
    float* __restrict__ out_nope_scale,
    __nv_bfloat16* __restrict__ out_rope,
    int K, int nope, int rope
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= K) return;
    int src = indices ? indices[row] : row;
    if (col < nope) out_nope_fp8[(int64_t)row * nope + col] = comp_nope_fp8[(int64_t)src * nope + col];
    if (col < rope) out_rope[(int64_t)row * rope + col] = comp_rope[(int64_t)src * rope + col];
    if (blockIdx.x == 0 && threadIdx.x == 0) out_nope_scale[row] = comp_nope_scale[src];
}

__global__ void quantize_swa_tail_kernel(
    const __nv_bfloat16* __restrict__ swa, // (S, HD), BF16
    uint8_t* __restrict__ out_nope_fp8,    // (K+S, NOPE)
    float* __restrict__ out_nope_scale,    // (K+S)
    __nv_bfloat16* __restrict__ out_rope,  // (K+S, ROPE)
    int K, int S, int hd, int nope, int rope
) {
    int s = blockIdx.x;
    if (s >= S) return;
    int out_row = K + s;
    const __nv_bfloat16* src = swa + (int64_t)s * hd;
    uint8_t* out8 = out_nope_fp8 + (int64_t)out_row * nope;
    __nv_bfloat16* outrope = out_rope + (int64_t)out_row * rope;

    float local_max = 0.0f;
    for (int c = threadIdx.x; c < nope; c += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(bf16_load(src + c)));
    }
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    __shared__ float warp_max[8];
    if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = local_max;
    __syncthreads();
    float amax = 0.0f;
    if (threadIdx.x < 32) {
        amax = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_max[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, off));
        if (threadIdx.x == 0) {
            float scale = amax / E4M3_MAX;
            if (scale < 1e-8f) scale = 1e-8f;
            out_nope_scale[out_row] = scale;
        }
    }
    __syncthreads();

    float inv_scale = 1.0f / out_nope_scale[out_row];
    for (int c = threadIdx.x; c < nope; c += blockDim.x) {
        out8[c] = fp8_e4m3_from_f32(bf16_load(src + c) * inv_scale);
    }
    for (int c = threadIdx.x; c < rope; c += blockDim.x) {
        outrope[c] = src[nope + c];
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_q_fp8_split_cuda(
    torch::Tensor q, int64_t rope_dim
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be BF16");
    TORCH_CHECK(q.dim() == 4, "q must be (B,H,T,HD)");
    q = q.contiguous();
    int B = q.size(0), H = q.size(1), T = q.size(2), HD = q.size(3);
    int rope = (int)rope_dim;
    int nope = HD - rope;
    TORCH_CHECK(nope > 0 && rope > 0, "invalid rope_dim");
    auto q8 = torch::empty({B, H, T, nope}, q.options().dtype(torch::kUInt8));
    auto qs = torch::empty({B, H, T}, q.options().dtype(torch::kFloat32));
    auto qr = torch::empty({B, H, T, rope}, q.options().dtype(torch::kBFloat16));
    int rows = B * H * T;
    quantize_q_fp8_split_kernel<<<rows, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        q8.data_ptr<uint8_t>(), qs.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(qr.data_ptr<at::BFloat16>()),
        rows, HD, nope, rope);
    return {q8.view(torch::kFloat8_e4m3fn), qs, qr};
}

void gather_mixed_selective_cuda(
    torch::Tensor comp_nope_fp8, torch::Tensor comp_nope_scale, torch::Tensor comp_rope,
    torch::Tensor swa, torch::Tensor indices,
    torch::Tensor out_nope_fp8, torch::Tensor out_nope_scale, torch::Tensor out_rope
) {
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
    int K = indices.size(0);
    int S = swa.size(0);
    int nope = comp_nope_fp8.size(1);
    int rope = comp_rope.size(1);
    int hd = nope + rope;
    if (K > 0) {
        dim3 grid(((nope > rope ? nope : rope) + 255) / 256, K);
        copy_comp_rows_kernel<<<grid, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
            comp_nope_fp8.data_ptr<uint8_t>(), comp_nope_scale.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(comp_rope.data_ptr<at::BFloat16>()),
            indices.data_ptr<int32_t>(),
            out_nope_fp8.data_ptr<uint8_t>(), out_nope_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out_rope.data_ptr<at::BFloat16>()),
            K, nope, rope);
    }
    if (S > 0) {
        quantize_swa_tail_kernel<<<S, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(swa.data_ptr<at::BFloat16>()),
            out_nope_fp8.data_ptr<uint8_t>(), out_nope_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out_rope.data_ptr<at::BFloat16>()),
            K, S, hd, nope, rope);
    }
}

void gather_mixed_all_cuda(
    torch::Tensor comp_nope_fp8, torch::Tensor comp_nope_scale, torch::Tensor comp_rope,
    torch::Tensor swa, torch::Tensor out_nope_fp8, torch::Tensor out_nope_scale, torch::Tensor out_rope
) {
    int K = comp_nope_fp8.size(0);
    int S = swa.size(0);
    int nope = comp_nope_fp8.size(1);
    int rope = comp_rope.size(1);
    int hd = nope + rope;
    if (K > 0) {
        dim3 grid(((nope > rope ? nope : rope) + 255) / 256, K);
        copy_comp_rows_kernel<<<grid, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
            comp_nope_fp8.data_ptr<uint8_t>(), comp_nope_scale.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(comp_rope.data_ptr<at::BFloat16>()),
            nullptr,
            out_nope_fp8.data_ptr<uint8_t>(), out_nope_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out_rope.data_ptr<at::BFloat16>()),
            K, nope, rope);
    }
    if (S > 0) {
        quantize_swa_tail_kernel<<<S, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(swa.data_ptr<at::BFloat16>()),
            out_nope_fp8.data_ptr<uint8_t>(), out_nope_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out_rope.data_ptr<at::BFloat16>()),
            K, S, hd, nope, rope);
    }
}

void gather_mixed_swa_only_cuda(torch::Tensor swa, torch::Tensor out_nope_fp8,
                                torch::Tensor out_nope_scale, torch::Tensor out_rope,
                                int64_t rope_dim) {
    int S = swa.size(0);
    int hd = swa.size(1);
    int rope = (int)rope_dim;
    int nope = hd - rope;
    if (S > 0) {
        quantize_swa_tail_kernel<<<S, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(swa.data_ptr<at::BFloat16>()),
            out_nope_fp8.data_ptr<uint8_t>(), out_nope_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out_rope.data_ptr<at::BFloat16>()),
            0, S, hd, nope, rope);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_q_fp8_split", &quantize_q_fp8_split_cuda,
          "Split Q into FP8_E4M3 noPE + BF16 RoPE");
    m.def("gather_mixed_selective_", &gather_mixed_selective_cuda,
          "In-place mixed KV gather for selected compressed rows + SWA tail");
    m.def("gather_mixed_all_", &gather_mixed_all_cuda,
          "In-place mixed KV gather for all compressed rows + SWA tail");
    m.def("gather_mixed_swa_only_", &gather_mixed_swa_only_cuda,
          "In-place mixed KV gather for SWA-only attention");
}
