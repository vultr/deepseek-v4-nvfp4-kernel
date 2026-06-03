/**
 * Hash router kernel for DeepSeek-V4.
 *
 * Deterministic per-token-ID expert assignment via precomputed lookup table.
 * Used by the first 3 MoE layers (§2.1) — no gate GEMM, no hidden-state input.
 *
 * One thread per token. Each thread gathers k expert IDs from the LUT and
 * writes uniform 1/k weights. Bandwidth-bound: 3 MB LUT fits in L2 for
 * repeated decode calls.
 *
 * Launch: grid(N), block(k) — k threads per token cooperate on the gather.
 *   - Thread h in each block handles the h-th expert slot for that token.
 *   - No shared memory needed.
 *
 * Tie-breaking: not applicable (LUT is deterministic). If two tokens map to
 * the same expert ID for different h slots, that's the checkpoint's problem.
 *
 * Bounds: vocab_size <= 256K, k <= 16, num_experts <= 384. All fit int32.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cstdint>

__global__ void hash_router_kernel(
    // Inputs
    const int32_t* __restrict__ token_ids,    // [N] — token indices into LUT
    const int32_t* __restrict__ hash_lut,     // [vocab_size, k] — expert IDs
    int64_t lut_stride,                        // stride in elements (= k)
    int32_t k,                                 // experts per token (6)
    int32_t vocab_size,                        // LUT row count
    // Outputs
    float* __restrict__ out_weights,           // [N, k] — 1/k uniform
    int64_t out_weights_stride,                // stride in elements
    int32_t* __restrict__ out_ids,             // [N, k] — expert IDs
    int64_t out_ids_stride                     // stride in elements
) {
    int64_t n = blockIdx.x;  // one block per token
    int32_t h = threadIdx.x; // one thread per expert slot

    if (n >= gridDim.x || h >= k) return;

    int32_t tid = token_ids[n];

    // Bounds check — invalid token IDs get expert 0, weight 0.
    // This should never happen with correct tokenization, but we don't
    // want silent OOB reads. Log a sentinel rather than crash.
    if (tid < 0 || tid >= vocab_size) {
        out_ids[n * out_ids_stride + h] = 0;
        out_weights[n * out_weights_stride + h] = 0.0f;
        return;
    }

    // Gather expert ID from LUT
    int32_t expert_id = hash_lut[static_cast<int64_t>(tid) * lut_stride + h];

    out_ids[n * out_ids_stride + h] = expert_id;

    // Uniform weight: 1/k. This is FP32 — the Router contract specifies
    // topk_weights as float32. No renormalization needed since it's
    // already uniform and sums to 1.0 before routed_scaling_factor
    // (scaling is applied in the Router layer, not here).
    out_weights[n * out_weights_stride + h] = 1.0f / static_cast<float>(k);
}


std::tuple<torch::Tensor, torch::Tensor> hash_router_cuda(
    torch::Tensor token_ids,       // [N] int32
    torch::Tensor hash_lut,        // [vocab_size, k] int32
    int64_t k,
    torch::Tensor out_weights,     // [N, k] float32, pre-allocated
    torch::Tensor out_ids          // [N, k] int32, pre-allocated
) {
    int64_t N = token_ids.size(0);
    int64_t vocab_size = hash_lut.size(0);
    int64_t lut_stride = hash_lut.stride(0);

    TORCH_CHECK(token_ids.scalar_type() == torch::kInt32, "token_ids must be int32");
    TORCH_CHECK(hash_lut.scalar_type() == torch::kInt32, "hash_lut must be int32");
    TORCH_CHECK(out_weights.scalar_type() == torch::kFloat32, "out_weights must be float32");
    TORCH_CHECK(out_ids.scalar_type() == torch::kInt32, "out_ids must be int32");
    TORCH_CHECK(out_weights.size(0) >= N, "out_weights too small");
    TORCH_CHECK(out_ids.size(0) >= N, "out_ids too small");

    if (N == 0) return std::make_tuple(out_weights, out_ids);

    // Launch: one block per token, k threads per block.
    // k=6 → 6 threads/block. Occupancy is fine — the kernel is bandwidth-bound
    // and each thread does one gather + one store. No shared memory, no smem bank
    // conflicts. The LUT (3 MB) stays hot in L2 across decode iterations.
    dim3 grid(static_cast<uint32_t>(N));
    dim3 block(static_cast<uint32_t>(k));

    hash_router_kernel<<<grid, block>>>(
        token_ids.data_ptr<int32_t>(),
        hash_lut.data_ptr<int32_t>(),
        lut_stride,
        static_cast<int32_t>(k),
        static_cast<int32_t>(vocab_size),
        out_weights.data_ptr<float>(),
        out_weights.stride(0),
        out_ids.data_ptr<int32_t>(),
        out_ids.stride(0)
    );

    return std::make_tuple(out_weights, out_ids);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hash_router", &hash_router_cuda);
}
