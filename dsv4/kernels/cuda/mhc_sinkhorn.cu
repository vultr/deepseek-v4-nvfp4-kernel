/**
 * Fused mHC Sinkhorn-Knopp projection kernel.
 *
 * Operates on (T, n, n) matrices. For DSV4-Pro: T=1, n=4.
 * 20 iterations of alternating row/col normalization.
 *
 * Replaces 38 Python kernel launches with 1 CUDA kernel launch.
 * At 61 layers × 2 mHC calls = 122 calls/step, saves ~4,600 kernel launches.
 *
 * Matches HuggingFace DeepseekV4HyperConnection exactly:
 *   1. softmax(logits, dim=-1) + eps
 *   2. column normalize
 *   3. (t_max - 1) alternating row/col normalize
 *
 * NVFP4 PATH: This kernel operates on the B_l (comb) matrix which must be
 * doubly-stochastic for residual bounding. The residual |X| growth to 500-700
 * at L60 indicates B was NOT properly doubly-stochastic at runtime. This kernel
 * ensures it. No fallback to Python. If this kernel fails, the pipeline fails.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cmath>

// Max supported n — DSV4 uses n=4. Increase if needed.
#define MHC_MAX_N 16

// One block per batch element. n*n threads per block (for n=4: 16 threads).
// Shared memory holds the (n, n) matrix + row/col sums.
// All loops use fixed-size arrays (no VLA — CUDA requirement).

__global__ void mhc_sinkhorn_kernel(
    const float* __restrict__ logits,  // (T, n, n)
    float* __restrict__ out,           // (T, n, n)
    int T, int n, int t_max, float eps
) {
    int t = blockIdx.x;
    if (t >= T) return;

    // Shared memory layout: M (n, n) | row_max (MHC_MAX_N) | row_sum (MHC_MAX_N) | col_sum (MHC_MAX_N)
    extern __shared__ float smem[];
    float* M = smem;                                       // n*n floats
    float* row_max = smem + n * n;                         // MHC_MAX_N floats
    float* row_sum_arr = row_max + MHC_MAX_N;              // MHC_MAX_N floats
    float* col_sum_arr = row_sum_arr + MHC_MAX_N;          // MHC_MAX_N floats

    int i = threadIdx.x / n;
    int j = threadIdx.x % n;

    // Step 1: softmax(logits, dim=-1) + eps
    if (i < n && j < n) {
        M[i * n + j] = logits[t * n * n + i * n + j];
    }
    __syncthreads();

    // Compute row max for numerical stability
    // Thread 0 does all the work (n is tiny — 4)
    if (threadIdx.x == 0) {
        for (int ri = 0; ri < n; ri++) {
            float mx = -INFINITY;
            for (int rj = 0; rj < n; rj++) {
                mx = fmaxf(mx, M[ri * n + rj]);
            }
            row_max[ri] = mx;
        }

        // Apply softmax + eps
        for (int ri = 0; ri < n; ri++) {
            float exp_sum = 0.0f;
            for (int rj = 0; rj < n; rj++) {
                M[ri * n + rj] = expf(M[ri * n + rj] - row_max[ri]);
                exp_sum += M[ri * n + rj];
            }
            for (int rj = 0; rj < n; rj++) {
                M[ri * n + rj] = M[ri * n + rj] / exp_sum + eps;
            }
        }

        // Step 2: column normalize
        for (int cj = 0; cj < n; cj++) {
            float cs = 0.0f;
            for (int ci = 0; ci < n; ci++) cs += M[ci * n + cj];
            for (int ci = 0; ci < n; ci++) M[ci * n + cj] = M[ci * n + cj] / (cs + eps);
        }

        // Step 3: (t_max - 1) alternating row/col normalize
        for (int iter = 0; iter < t_max - 1; iter++) {
            // Row normalize
            for (int ri = 0; ri < n; ri++) {
                float rs = 0.0f;
                for (int rj = 0; rj < n; rj++) rs += M[ri * n + rj];
                for (int rj = 0; rj < n; rj++) M[ri * n + rj] = M[ri * n + rj] / (rs + eps);
            }
            // Column normalize
            for (int cj = 0; cj < n; cj++) {
                float cs = 0.0f;
                for (int ci = 0; ci < n; ci++) cs += M[ci * n + cj];
                for (int ci = 0; ci < n; ci++) M[ci * n + cj] = M[ci * n + cj] / (cs + eps);
            }
        }
    }
    __syncthreads();

    // Write output
    if (i < n && j < n) {
        out[t * n * n + i * n + j] = M[i * n + j];
    }
}

torch::Tensor mhc_sinkhorn_cuda(
    torch::Tensor logits,  // (T, n, n) FP32
    int64_t t_max,
    double eps
) {
    TORCH_CHECK(logits.dim() == 3, "logits must be 3D (T, n, n)");
    int T = logits.size(0);
    int n = logits.size(1);
    TORCH_CHECK(logits.size(2) == n, "logits must be square");
    TORCH_CHECK(n <= MHC_MAX_N, "n must be <= MHC_MAX_N (16)");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be FP32");

    auto out = torch::empty_like(logits);

    // One block per batch element, n*n threads per block
    int threads = n * n;
    // Shared memory: M (n*n) + row_max + row_sum + col_sum (3 * MHC_MAX_N)
    int smem_size = (n * n + 3 * MHC_MAX_N) * sizeof(float);

    mhc_sinkhorn_kernel<<<T, threads, smem_size, c10::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(),
        out.data_ptr<float>(),
        T, n, t_max, (float)eps
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mhc_sinkhorn", &mhc_sinkhorn_cuda, "Fused mHC Sinkhorn-Knopp projection (NO FALLBACK)");
}
