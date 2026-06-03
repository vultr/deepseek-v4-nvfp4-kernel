/**
 * Compressor reduce kernels for DSV4 CSA and HCA.
 *
 * Takes the OUTPUT of the NVFP4 GEMM projections (kv_proj, gate_proj)
 * and performs the token-level softmax + weighted sum reduction.
 *
 * CSA (paper eq. 11-12):
 *   kv_proj output: (T, 2*hd) — Ca (first hd) and Cb (second hd)
 *   gate_proj output: (T, 2*hd) — Ga (first hd) and Gb (second hd)
 *   For block i: if i > 0, concat Ca[i-1] + Cb[i] and Ga[i-1] + Gb[i]
 *                else just Cb[0] and Gb[0]
 *   compressed[i] = softmax(gate_block, dim=0) * kv_block summed over tokens
 *
 * HCA (paper eq. 9-10):
 *   kv_proj output: (T, hd)
 *   gate_proj output: (T, hd)
 *   For block i: kv_block = kv[i*m : (i+1)*m], gate_block = gate[i*m : (i+1)*m]
 *   compressed[i] = softmax(gate_block, dim=0) * kv_block summed over tokens
 *
 * Both kernels also apply kv_norm (unweighted RMSNorm) if weight is provided.
 *
 * One block per compressed output entry. 128 threads per block.
 * Each thread processes a strided subset of columns.
 * FP32 accumulation throughout. No extern shared memory needed.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

// Block-level sum reduction (for kv_norm)
__device__ __forceinline__ float block_reduce_sum(float val, float* smem, int n_warps) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (threadIdx.x % 32 == 0) {
        smem[threadIdx.x / 32] = val;
    }
    __syncthreads();
    float result = 0.0f;
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        result = v;
    }
    __syncthreads();
    return result;
}

// ===========================================================================
// CSA compressor reduce kernel
// ===========================================================================

__global__ void csa_compress_reduce_kernel(
    const float* __restrict__ kv_proj,      // [T, 2*hd] FP32 (Ca | Cb)
    const float* __restrict__ gate_proj,    // [T, 2*hd] FP32 (Ga | Gb)
    const float* __restrict__ position_bias, // [m, 2*hd] FP32 or nullptr
    const float* __restrict__ kv_norm_weight, // [hd] FP32 or nullptr (unused here, applied separately)
    float* __restrict__ compressed,          // [n_blocks, hd] FP32
    int T, int hd, int m, int n_blocks
) {
    int block_i = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;
    int kv_dim = 2 * hd;

    if (block_i >= n_blocks) return;

    int n_tokens = (block_i > 0) ? 2 * m : m;
    int prev_start = (block_i - 1) * m;
    int cur_start = block_i * m;

    // Each thread processes columns [tid, tid+n_threads, tid+2*n_threads, ...]
    // Max cols per thread for hd=512, 128 threads = 4
    int cols_per_thread = (hd + n_threads - 1) / n_threads;

    float local_max[4];
    float local_denom[4];
    float local_acc[4];

    for (int ci = 0; ci < cols_per_thread; ci++) {
        int c = tid + ci * n_threads;
        if (c >= hd) break;
        local_max[ci] = -FLT_MAX;
        local_denom[ci] = 0.0f;
        local_acc[ci] = 0.0f;

        // Pass 1: find max gate value
        for (int t = 0; t < n_tokens; t++) {
            int token_idx, gate_offset;
            if (block_i > 0) {
                if (t < m) { token_idx = prev_start + t; gate_offset = 0; }
                else { token_idx = cur_start + (t - m); gate_offset = hd; }
            } else {
                token_idx = t; gate_offset = hd;
            }
            if (token_idx < 0 || token_idx >= T) continue;

            float g = gate_proj[token_idx * kv_dim + gate_offset + c];
            // Position bias: same (m, 2*hd) bias added to every block
            if (position_bias != nullptr) {
                int pos_bias_row = (block_i > 0 && t < m) ? t : (block_i > 0 ? (t - m) : t);
                if (pos_bias_row >= 0 && pos_bias_row < m) {
                    g += position_bias[pos_bias_row * kv_dim + gate_offset + c];
                }
            }
            local_max[ci] = fmaxf(local_max[ci], g);
        }

        // Pass 2: exp sum + weighted sum
        for (int t = 0; t < n_tokens; t++) {
            int token_idx, kv_offset, gate_offset;
            if (block_i > 0) {
                if (t < m) { token_idx = prev_start + t; kv_offset = 0; gate_offset = 0; }
                else { token_idx = cur_start + (t - m); kv_offset = hd; gate_offset = hd; }
            } else {
                token_idx = t; kv_offset = hd; gate_offset = hd;
            }
            if (token_idx < 0 || token_idx >= T) continue;

            float g = gate_proj[token_idx * kv_dim + gate_offset + c];
            float kv_val = kv_proj[token_idx * kv_dim + kv_offset + c];
            // Position bias: added to gate logits (softmax Z + B) only.
            // The paper defines compression as softmax(Z + B) then weighted sum of C.
            // The bias must NOT be added to kv_val — that poisons compressed content.
            if (position_bias != nullptr) {
                int pos_bias_row = (block_i > 0 && t < m) ? t : (block_i > 0 ? (t - m) : t);
                if (pos_bias_row >= 0 && pos_bias_row < m) {
                    float pb = position_bias[pos_bias_row * kv_dim + gate_offset + c];
                    g += pb;
                }
            }
            float e = expf(g - local_max[ci]);
            local_denom[ci] += e;
            local_acc[ci] += e * kv_val;
        }

        float val = (local_denom[ci] > 0.0f) ? (local_acc[ci] / local_denom[ci]) : 0.0f;
        compressed[block_i * hd + c] = val;
    }
}

// ===========================================================================
// HCA compressor reduce kernel (no overlap, single stream)
// ===========================================================================

__global__ void hca_compress_reduce_kernel(
    const float* __restrict__ kv_proj,      // [T, hd] FP32
    const float* __restrict__ gate_proj,    // [T, hd] FP32
    const float* __restrict__ position_bias, // [m, hd] FP32 or nullptr
    const float* __restrict__ kv_norm_weight, // [hd] FP32 or nullptr (unused here)
    float* __restrict__ compressed,          // [n_blocks, hd] FP32
    int T, int hd, int m, int n_blocks
) {
    int block_i = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    if (block_i >= n_blocks) return;

    int cols_per_thread = (hd + n_threads - 1) / n_threads;

    for (int ci = 0; ci < cols_per_thread; ci++) {
        int c = tid + ci * n_threads;
        if (c >= hd) break;

        float local_max = -FLT_MAX;
        float local_denom = 0.0f;
        float local_acc = 0.0f;

        int start = block_i * m;

        // Pass 1: max
        for (int t = 0; t < m; t++) {
            int token_idx = start + t;
            if (token_idx >= T) break;
            float g = gate_proj[token_idx * hd + c];
            if (position_bias != nullptr && t < m) {
                g += position_bias[t * hd + c];
            }
            local_max = fmaxf(local_max, g);
        }

        // Pass 2: exp + weighted sum
        for (int t = 0; t < m; t++) {
            int token_idx = start + t;
            if (token_idx >= T) break;
            float g = gate_proj[token_idx * hd + c];
            float kv_val = kv_proj[token_idx * hd + c];
            // Position bias: added to gate logits (softmax Z + B) only.
            // The paper defines compression as softmax(Z + B) then weighted sum of C.
            // The bias must NOT be added to kv_val — that poisons compressed content.
            if (position_bias != nullptr && t < m) {
                float pb = position_bias[t * hd + c];
                g += pb;
            }
            float e = expf(g - local_max);
            local_denom += e;
            local_acc += e * kv_val;
        }

        float val = (local_denom > 0.0f) ? (local_acc / local_denom) : 0.0f;
        compressed[block_i * hd + c] = val;
    }
}

// ===========================================================================
// Unweighted RMSNorm kernel (applied after compress reduce)
// ===========================================================================

__global__ void apply_kv_norm_kernel(
    const float* __restrict__ input,         // [n_blocks, hd] FP32
    const float* __restrict__ norm_weight,   // [hd] FP32
    float* __restrict__ output,               // [n_blocks, hd] FP32 (can be same as input)
    int n_blocks, int hd
) {
    int block_i = blockIdx.x;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;
    int n_warps = n_threads / 32;

    if (block_i >= n_blocks) return;

    // Compute sum of squares for this block
    float local_sq = 0.0f;
    for (int c = tid; c < hd; c += n_threads) {
        float v = input[block_i * hd + c];
        local_sq += v * v;
    }

    __shared__ float s_sum;
    float total_sq = block_reduce_sum(local_sq, &s_sum, n_warps);
    __shared__ float s_inv_rms;
    if (tid == 0) {
        float mean_sq = total_sq / hd;
        s_inv_rms = rsqrtf(mean_sq + 1e-6f);
    }
    __syncthreads();

    for (int c = tid; c < hd; c += n_threads) {
        output[block_i * hd + c] = input[block_i * hd + c] * s_inv_rms * norm_weight[c];
    }
}

// ===========================================================================
// PyTorch bindings
// ===========================================================================

void csa_compress_reduce_cuda(
    torch::Tensor kv_proj,       // [T, 2*hd] FP32
    torch::Tensor gate_proj,     // [T, 2*hd] FP32
    torch::Tensor position_bias, // [m, 2*hd] FP32 or empty
    torch::Tensor kv_norm_weight, // [hd] FP32 or empty
    torch::Tensor compressed,    // [n_blocks, hd] FP32
    int64_t m, int64_t n_blocks
) {
    int T = kv_proj.size(0);
    int hd = compressed.size(1);
    int threads = 128;

    TORCH_CHECK(kv_proj.scalar_type() == torch::kFloat32, "kv_proj must be float32");
    TORCH_CHECK(gate_proj.scalar_type() == torch::kFloat32, "gate_proj must be float32");

    const float* pos_bias_ptr = nullptr;
    if (position_bias.numel() > 0) {
        pos_bias_ptr = position_bias.data_ptr<float>();
    }
    const float* norm_ptr = nullptr;
    if (kv_norm_weight.numel() > 0) {
        norm_ptr = kv_norm_weight.data_ptr<float>();
    }

    csa_compress_reduce_kernel<<<n_blocks, threads>>>(
        kv_proj.data_ptr<float>(),
        gate_proj.data_ptr<float>(),
        pos_bias_ptr,
        norm_ptr,
        compressed.data_ptr<float>(),
        T, hd, (int)m, (int)n_blocks
    );
    C10_CUDA_CHECK(cudaGetLastError());

    // Apply kv_norm if provided
    if (norm_ptr != nullptr) {
        apply_kv_norm_kernel<<<n_blocks, threads>>>(
            compressed.data_ptr<float>(),
            norm_ptr,
            compressed.data_ptr<float>(),
            (int)n_blocks, hd
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }
}

void hca_compress_reduce_cuda(
    torch::Tensor kv_proj,       // [T, hd] FP32
    torch::Tensor gate_proj,     // [T, hd] FP32
    torch::Tensor position_bias, // [m, hd] FP32 or empty
    torch::Tensor kv_norm_weight, // [hd] FP32 or empty
    torch::Tensor compressed,    // [n_blocks, hd] FP32
    int64_t m, int64_t n_blocks
) {
    int T = kv_proj.size(0);
    int hd = compressed.size(1);
    int threads = 128;

    TORCH_CHECK(kv_proj.scalar_type() == torch::kFloat32, "kv_proj must be float32");
    TORCH_CHECK(gate_proj.scalar_type() == torch::kFloat32, "gate_proj must be float32");

    const float* pos_bias_ptr = nullptr;
    if (position_bias.numel() > 0) {
        pos_bias_ptr = position_bias.data_ptr<float>();
    }
    const float* norm_ptr = nullptr;
    if (kv_norm_weight.numel() > 0) {
        norm_ptr = kv_norm_weight.data_ptr<float>();
    }

    hca_compress_reduce_kernel<<<n_blocks, threads>>>(
        kv_proj.data_ptr<float>(),
        gate_proj.data_ptr<float>(),
        pos_bias_ptr,
        norm_ptr,
        compressed.data_ptr<float>(),
        T, hd, (int)m, (int)n_blocks
    );
    C10_CUDA_CHECK(cudaGetLastError());

    if (norm_ptr != nullptr) {
        apply_kv_norm_kernel<<<n_blocks, threads>>>(
            compressed.data_ptr<float>(),
            norm_ptr,
            compressed.data_ptr<float>(),
            (int)n_blocks, hd
        );
        C10_CUDA_CHECK(cudaGetLastError());
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("csa_compress_reduce", &csa_compress_reduce_cuda, "CSA compress reduce kernel");
    m.def("hca_compress_reduce", &hca_compress_reduce_cuda, "HCA compress reduce kernel");
}
