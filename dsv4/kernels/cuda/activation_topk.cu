/**
 * Fused activation + top-k + renormalization kernel for DSV4 router.
 *
 * This kernel implements the full router computation for the prefill path
 * (and as a fallback for the decode path when the fused GEMM kernel isn't
 * yet compiled):
 *
 *   1. act[n, e] = sqrt(softplus(logits[n, e]))     in FP32
 *   2. score[n, e] = act[n, e] + e_bias[e]          in FP32
 *   3. topk_ids[n, :] = argtopk(score[n, :], k=6)   min-heap in registers
 *   4. raw_w[n, h] = act[n, topk_ids[n, h]]          gather unbiased
 *   5. topk_w[n, h] = raw_w / sum(raw_w) * scaling  renormalize
 *
 * Single kernel launch, no CPU-GPU sync, no intermediate buffers.
 * One block per row (token), 64 threads per block.
 *
 * Numerical details:
 *   softplus(x) = max(x, 0) + log1p(exp(-|x|))
 *   This is the numerically stable form. Do NOT use log(1 + exp(x)).
 *
 * Tie-breaking: lower index wins on equal scores.
 * Same logic as topk_select.cu: (score, -index) as heap comparison key.
 *
 * This kernel is the reference implementation. The decode path (CuTeDSL
 * fused GEMM + epilogue) should produce identical results. If it doesn't,
 * the CuTeDSL kernel has a bug.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cstdint>
#include <cfloat>

// Same HeapEntry and heap logic as topk_select.cu — duplicated here
// because this kernel is standalone (no dependency on topk_select.cu).
// If we could link multiple .cu files into one extension, we'd share.
// For now, the duplication is intentional and correct.

struct HeapEntry {
    float score;
    int32_t index;
};

__device__ __forceinline__ void heap_sift_down_router(
    HeapEntry* heap, int32_t k, int32_t root
) {
    while (true) {
        int32_t left = 2 * root + 1;
        int32_t right = 2 * root + 2;
        int32_t smallest = root;

        if (left < k) {
            if (heap[left].score < heap[smallest].score ||
                (heap[left].score == heap[smallest].score &&
                 heap[left].index > heap[smallest].index)) {
                smallest = left;
            }
        }
        if (right < k) {
            if (heap[right].score < heap[smallest].score ||
                (heap[right].score == heap[smallest].score &&
                 heap[right].index > heap[smallest].index)) {
                smallest = right;
            }
        }
        if (smallest == root) break;
        HeapEntry tmp = heap[root];
        heap[root] = heap[smallest];
        heap[smallest] = tmp;
        root = smallest;
    }
}

__device__ __forceinline__ void heap_push_router(
    HeapEntry* heap, int32_t k, float score, int32_t index
) {
    if (score < heap[0].score) return;
    if (score == heap[0].score && index >= heap[0].index) return;

    heap[0].score = score;
    heap[0].index = index;
    heap_sift_down_router(heap, k, 0);
}

// ---------------------------------------------------------------------------
// Numerically stable softplus
// ---------------------------------------------------------------------------

__device__ __forceinline__ float stable_softplus(float x) {
    // softplus(x) = max(x, 0) + log1p(exp(-|x|))
    float pos = fmaxf(x, 0.0f);
    float neg_abs = -fabsf(x);
    float exp_val = expf(neg_abs);
    return pos + log1pf(exp_val);
}

// ---------------------------------------------------------------------------
// Fused activation + top-k + renorm kernel
// ---------------------------------------------------------------------------

// K=6 as template parameter for compile-time unrolling of the heap.
// THREADS_PER_ROW = 64. For E <= 384, each thread processes <= 6 elements.
template <int K, int THREADS_PER_ROW>
__global__ void fused_activation_topk_kernel(
    // Inputs
    const float* __restrict__ logits,       // [num_rows, E] FP32
    int64_t logits_stride,                   // stride in elements
    const float* __restrict__ e_bias,        // [E] FP32
    int32_t E,                               // number of experts
    float routed_scaling_factor,             // post-renorm scale
    // Outputs
    float* __restrict__ out_weights,         // [num_rows, K] FP32
    int64_t out_weights_stride,
    int32_t* __restrict__ out_ids,           // [num_rows, K] int32
    int64_t out_ids_stride
) {
    extern __shared__ char smem[];
    HeapEntry* shared_heaps = reinterpret_cast<HeapEntry*>(smem);

    // Also store unbiased activation values for each heap entry so we can
    // gather them during the renormalization step without re-computing
    // softplus. 6 floats per thread, 64 threads = 384 floats = 1.5 KB.
    float* shared_acts = reinterpret_cast<float*>(
        smem + THREADS_PER_ROW * K * sizeof(HeapEntry)
    );

    int64_t row = blockIdx.x;
    int32_t tid = threadIdx.x;

    // Per-thread local top-k heap in registers
    HeapEntry local_heap[K];
    float local_acts[K];  // unbiased activation values for heap entries

    #pragma unroll
    for (int i = 0; i < K; i++) {
        local_heap[i].score = -FLT_MAX;
        local_heap[i].index = -1;
        local_acts[i] = 0.0f;
    }

    // Scan this thread's stripe of E
    const float* row_logits = logits + row * logits_stride;
    int32_t elements_per_thread = (E + THREADS_PER_ROW - 1) / THREADS_PER_ROW;
    int32_t e_start = tid * elements_per_thread;
    int32_t e_end = min(e_start + elements_per_thread, E);

    for (int32_t e = e_start; e < e_end; e++) {
        float logit = row_logits[e];

        // Step 1: act = sqrt(softplus(logit))
        float sp = stable_softplus(logit);
        float act = sqrtf(sp);

        // Step 2: score = act + e_bias[e]
        float score = act + e_bias[e];

        // Step 3: push to heap (using score for selection, storing act for later)
        if (score > local_heap[0].score ||
            (score == local_heap[0].score && e < local_heap[0].index)) {
            // Replace root
            local_heap[0].score = score;
            local_heap[0].index = e;
            local_acts[0] = act;  // store unbiased activation

            // Sift down (fully unrolled for K=6)
            #pragma unroll
            for (int root = 0; root < K; ) {
                int left = 2 * root + 1;
                int right = 2 * root + 2;
                int smallest = root;

                if (left < K) {
                    if (local_heap[left].score < local_heap[smallest].score ||
                        (local_heap[left].score == local_heap[smallest].score &&
                         local_heap[left].index > local_heap[smallest].index)) {
                        smallest = left;
                    }
                }
                if (right < K) {
                    if (local_heap[right].score < local_heap[smallest].score ||
                        (local_heap[right].score == local_heap[smallest].score &&
                         local_heap[right].index > local_heap[smallest].index)) {
                        smallest = right;
                    }
                }
                if (smallest == root) break;

                HeapEntry tmp_h = local_heap[root];
                float tmp_a = local_acts[root];
                local_heap[root] = local_heap[smallest];
                local_acts[root] = local_acts[smallest];
                local_heap[smallest] = tmp_h;
                local_acts[smallest] = tmp_a;
                root = smallest;
            }
        }
    }

    // Write local heap to shared memory
    int32_t heap_base = tid * K;
    int32_t act_base = tid * K;
    #pragma unroll
    for (int i = 0; i < K; i++) {
        shared_heaps[heap_base + i] = local_heap[i];
        shared_acts[act_base + i] = local_acts[i];
    }
    __syncthreads();

    // Thread 0 merges all local heaps into final top-k
    if (tid == 0) {
        HeapEntry final_heap[K];
        float final_acts[K];

        #pragma unroll
        for (int i = 0; i < K; i++) {
            final_heap[i] = shared_heaps[i];
            final_acts[i] = shared_acts[i];
        }

        // Merge remaining threads' heaps
        for (int t = 1; t < THREADS_PER_ROW; t++) {
            int32_t tbase = t * K;
            #pragma unroll
            for (int i = 0; i < K; i++) {
                HeapEntry cand = shared_heaps[tbase + i];
                float cand_act = shared_acts[tbase + i];

                if (cand.index < 0) continue;  // sentinel

                if (cand.score > final_heap[0].score ||
                    (cand.score == final_heap[0].score &&
                     cand.index < final_heap[0].index)) {
                    final_heap[0] = cand;
                    final_acts[0] = cand_act;

                    // Sift down
                    #pragma unroll
                    for (int root = 0; root < K; ) {
                        int left = 2 * root + 1;
                        int right = 2 * root + 2;
                        int smallest = root;
                        if (left < K) {
                            if (final_heap[left].score < final_heap[smallest].score ||
                                (final_heap[left].score == final_heap[smallest].score &&
                                 final_heap[left].index > final_heap[smallest].index)) {
                                smallest = left;
                            }
                        }
                        if (right < K) {
                            if (final_heap[right].score < final_heap[smallest].score ||
                                (final_heap[right].score == final_heap[smallest].score &&
                                 final_heap[right].index > final_heap[smallest].index)) {
                                smallest = right;
                            }
                        }
                        if (smallest == root) break;
                        HeapEntry tmp_h = final_heap[root];
                        float tmp_a = final_acts[root];
                        final_heap[root] = final_heap[smallest];
                        final_acts[root] = final_acts[smallest];
                        final_heap[smallest] = tmp_h;
                        final_acts[smallest] = tmp_a;
                        root = smallest;
                    }
                }
            }
        }

        // Step 4-5: Gather unbiased activations and renormalize
        // final_acts already contains the unbiased activation at the top-k positions.
        // Renormalize: w = (act / sum(act)) * scaling
        float act_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < K; i++) {
            act_sum += final_acts[i];
        }

        float inv_sum = (act_sum > 0.0f) ? (1.0f / act_sum) : 0.0f;

        // Sort descending by score (selection sort, k=6 is trivial)
        // We need sorted output for deterministic behavior matching torch.topk
        HeapEntry sorted_heap[K];
        float sorted_acts[K];
        bool taken[K];
        #pragma unroll
        for (int i = 0; i < K; i++) taken[i] = false;

        #pragma unroll
        for (int i = 0; i < K; i++) {
            int best = -1;
            #pragma unroll
            for (int j = 0; j < K; j++) {
                if (taken[j]) continue;
                if (best < 0 ||
                    final_heap[j].score > final_heap[best].score ||
                    (final_heap[j].score == final_heap[best].score &&
                     final_heap[j].index < final_heap[best].index)) {
                    best = j;
                }
            }
            sorted_heap[i] = final_heap[best];
            sorted_acts[i] = final_acts[best];
            taken[best] = true;
        }

        // Step 6: Write outputs
        int64_t w_base = row * out_weights_stride;
        int64_t id_base = row * out_ids_stride;

        #pragma unroll
        for (int i = 0; i < K; i++) {
            out_ids[id_base + i] = sorted_heap[i].index;
            out_weights[w_base + i] = sorted_acts[i] * inv_sum * routed_scaling_factor;
        }
    }
}


// ---------------------------------------------------------------------------
// Host launch function
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> fused_activation_topk_cuda(
    torch::Tensor logits,              // [N, E] FP32
    torch::Tensor e_bias,              // [E] FP32
    double routed_scaling_factor,
    int64_t k,
    torch::Tensor out_weights,         // [N, k] FP32, pre-allocated
    torch::Tensor out_ids              // [N, k] int32, pre-allocated
) {
    int64_t N = logits.size(0);
    int64_t E = logits.size(1);
    int32_t k_int = static_cast<int32_t>(k);

    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(e_bias.scalar_type() == torch::kFloat32, "e_bias must be float32");
    TORCH_CHECK(e_bias.size(0) == E, "e_bias size mismatch");
    TORCH_CHECK(k_int == 6, "only k=6 is currently supported");

    if (N == 0) return std::make_tuple(out_weights, out_ids);

    // Thread config: 64 threads per row (covers E <= 384 with ~6 elements/thread)
    const int THREADS_PER_ROW = 64;

    // Shared memory: heaps (THREADS_PER_ROW * K * sizeof(HeapEntry))
    //              + acts (THREADS_PER_ROW * K * sizeof(float))
    int64_t smem = THREADS_PER_ROW * k_int * (sizeof(HeapEntry) + sizeof(float));

    dim3 grid(static_cast<uint32_t>(N));
    dim3 block(THREADS_PER_ROW);

    fused_activation_topk_kernel<6, THREADS_PER_ROW><<<grid, block, smem>>>(
        logits.data_ptr<float>(),
        logits.stride(0),
        e_bias.data_ptr<float>(),
        static_cast<int32_t>(E),
        static_cast<float>(routed_scaling_factor),
        out_weights.data_ptr<float>(),
        out_weights.stride(0),
        out_ids.data_ptr<int32_t>(),
        out_ids.stride(0)
    );

    return std::make_tuple(out_weights, out_ids);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_activation_topk", &fused_activation_topk_cuda);
}
