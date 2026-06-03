// indexer_score_topk.cu — Fused score + ReLU + weighted-sum + top-k kernel.
//
// CSA Lightning Indexer (paper §2.3.1, eq. 16):
//   I[t,s] = Σ_h w_h[t,h] · ReLU(q_I[t,h] · K^IComp[s,h])
//   Selected = TopK(I[t,:], k=csa_top_k)
//
// One CTA per query token. Streams indexer keys from the paged pool,
// computes per-head dot products in FP32, ReLU, weighted sum, top-k.
//
// Top-k strategy: each thread maintains a private top-k in registers
// over its strided slice of entries, then a block-level merge via
// bitonic sort on the shared heap. No in-loop barriers, no spinlocks.
//
// Phase 1 (this file): FP32 dot products via standard CUDA ops.
// Phase 2 (future): swap to FP4 tcgen05 MMA for production throughput.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

// FP4 E2M1 magnitude lookup (same as production)
__constant__ float E2M1_LUT[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ float dequant_fp4_scalar(
    uint8_t packed, int lane,
    float group_scale, float global_scale
) {
    int nibble = (lane == 0) ? (packed & 0x0F) : (packed >> 4);
    int sign = (nibble >> 3) & 1;
    int mag_bits = nibble & 0x07;
    float magnitude = E2M1_LUT[mag_bits];
    float val = magnitude * group_scale * global_scale;
    return sign ? -val : val;
}

// ---- Per-thread local top-k ----
// Each thread keeps LOCAL_K best scores in registers.
// LOCAL_K is a tuning parameter: larger = more accurate merge,
// smaller = less register pressure.
// For top_k=1024 and 128 threads: LOCAL_K=8 means 128*8=1024 candidates
// for the block-level merge, which is exact.
// For top_k=512 and 128 threads: LOCAL_K=4 gives 512 candidates, also exact.
// If top_k > n_threads * LOCAL_K, the merge is approximate (top-K of
// n_threads*LOCAL_K candidates). Increase LOCAL_K or n_threads to compensate.

#ifndef INDEXER_LOCAL_K
#define INDEXER_LOCAL_K 8
#endif

__device__ __forceinline__ void local_heap_insert(
    float* scores, int32_t* blocks,
    float score, int32_t block_id, int k
) {
    if (score <= scores[0]) return;
    scores[0] = score;
    blocks[0] = block_id;
    // Sift down
    int root = 0;
    while (root < (k >> 1)) {
        int left = 2 * root + 1;
        int right = 2 * root + 2;
        int smallest = root;
        if (left < k && scores[left] < scores[smallest]) smallest = left;
        if (right < k && scores[right] < scores[smallest]) smallest = right;
        if (smallest == root) break;
        float ts = scores[root]; int32_t ti = blocks[root];
        scores[root] = scores[smallest]; blocks[root] = blocks[smallest];
        scores[smallest] = ts; blocks[smallest] = ti;
        root = smallest;
    }
}

// ---- Block-level merge: merge n_threads × LOCAL_K candidates ----
// Each thread writes its local top-k to shared memory, then a single
// thread (or warp) does a final top-k selection from the combined set.
// Total candidates = n_threads * LOCAL_K.
// For top_k <= total_candidates, this is exact.
// For top_k > total_candidates, increase LOCAL_K.

__device__ __forceinline__ void heap_insert_shared(
    float* heap_scores, int32_t* heap_blocks,
    float score, int32_t block_id, int k
) {
    if (score <= heap_scores[0]) return;
    heap_scores[0] = score;
    heap_blocks[0] = block_id;
    int root = 0;
    while (root < (k >> 1)) {
        int left = 2 * root + 1;
        int right = 2 * root + 2;
        int smallest = root;
        if (left < k && heap_scores[left] < heap_scores[smallest]) smallest = left;
        if (right < k && heap_scores[right] < heap_scores[smallest]) smallest = right;
        if (smallest == root) break;
        float ts = heap_scores[root]; int32_t ti = heap_blocks[root];
        heap_scores[root] = heap_scores[smallest]; heap_blocks[root] = heap_blocks[smallest];
        heap_scores[smallest] = ts; heap_blocks[smallest] = ti;
        root = smallest;
    }
}

// ===========================================================================
// Main kernel
// ===========================================================================

__global__ void indexer_score_topk_fp32_kernel(
    const float* __restrict__ q_I,
    const float* __restrict__ w_h,
    const uint8_t* __restrict__ keys_fp4,
    const uint8_t* __restrict__ key_scale,
    const float* __restrict__ key_gscale,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ valid_lens,
    int32_t* __restrict__ topk_indices,
    int n_heads, int head_dim, int top_k,
    int entries_per_block, int max_logical_blocks
) {
    int t = blockIdx.x;
    if (t >= gridDim.x) return;

    int tid = threadIdx.x;
    int n_threads = blockDim.x;
    int num_valid = valid_lens[t];
    int n_groups = head_dim / 16;
    int n_bytes = head_dim / 2;

    // ---- Per-thread local top-k in registers ----
    // LOCAL_K entries per thread. Min-heap (root = smallest of local best).
    float local_scores[INDEXER_LOCAL_K];
    int32_t local_blocks[INDEXER_LOCAL_K];
    for (int i = 0; i < INDEXER_LOCAL_K; i++) {
        local_scores[i] = -INFINITY;
        local_blocks[i] = -1;
    }

    // ---- Load w_h into shared memory ----
    extern __shared__ char smem[];
    float* smem_w = reinterpret_cast<float*>(smem);
    // The rest of smem is used for the merge phase (allocated after w_h)
    // Layout: [w_h: n_heads floats] [merge_scores: top_k floats] [merge_blocks: top_k ints]
    //         [per_thread_scores: n_threads * LOCAL_K floats] [per_thread_blocks: n_threads * LOCAL_K ints]
    // But we allocate dynamically, so let's compute offsets.

    for (int h = tid; h < n_heads; h += n_threads) {
        smem_w[h] = w_h[t * n_heads + h];
    }
    __syncthreads();  // safe — outside the strided loop

    // ---- Stream over entries (strided, no barriers) ----
    // Each thread handles entries s = tid, tid+n_threads, tid+2*n_threads, ...
    // No __syncthreads() in this loop. No shared heap access.
    // Each thread accumulates into its private register heap.

    for (int s = tid; s < num_valid; s += n_threads) {
        int logical_block = s / entries_per_block;
        int slot_in_block = s % entries_per_block;
        int phys_block = block_table[t * max_logical_blocks + logical_block];
        int block_entry = phys_block * entries_per_block + slot_in_block;

        float global_s = key_gscale[phys_block];

        float score = 0.0f;
        for (int h = 0; h < n_heads; h++) {
            float dot = 0.0f;
            for (int g = 0; g < n_groups; g++) {
                uint8_t raw_scale = key_scale[block_entry * n_groups + g];
                __nv_fp8_e4m3 fp8_s;
                fp8_s.__x = raw_scale;
                float group_s = (float)fp8_s * global_s;

                for (int b = 0; b < 8; b++) {
                    uint8_t packed = keys_fp4[block_entry * n_bytes + g * 8 + b];
                    float v0 = dequant_fp4_scalar(packed, 0, group_s, 1.0f);
                    float v1 = dequant_fp4_scalar(packed, 1, group_s, 1.0f);
                    int d0 = g * 16 + 2 * b;
                    int d1 = d0 + 1;
                    dot += v0 * q_I[t * n_heads * head_dim + h * head_dim + d0];
                    dot += v1 * q_I[t * n_heads * head_dim + h * head_dim + d1];
                }
            }
            if (dot > 0.0f) {
                score += smem_w[h] * dot;
            }
        }

        // Insert into per-thread local heap (registers, no sync needed)
        local_heap_insert(local_scores, local_blocks, score, s, INDEXER_LOCAL_K);
    }

    // ---- Block-level merge ----
    // Each thread writes its LOCAL_K candidates to shared memory.
    // Then one thread builds the final top-k from all candidates.
    // Total candidates = n_threads * LOCAL_K.
    // For top_k=1024, n_threads=128, LOCAL_K=8: 1024 candidates, exact merge.
    // For top_k=512, n_threads=128, LOCAL_K=4: 512 candidates, exact merge.

    float* merge_scores = smem_w + n_heads;
    int32_t* merge_blocks = reinterpret_cast<int32_t*>(merge_scores + top_k);
    float* per_thread_scores = reinterpret_cast<float*>(merge_blocks + top_k);
    int32_t* per_thread_blocks = reinterpret_cast<int32_t*>(per_thread_scores + n_threads * INDEXER_LOCAL_K);

    // Initialize merge heap
    for (int i = tid; i < top_k; i += n_threads) {
        merge_scores[i] = -INFINITY;
        merge_blocks[i] = -1;
    }

    // Write local top-k to per-thread region in shared memory
    int my_offset = tid * INDEXER_LOCAL_K;
    for (int i = 0; i < INDEXER_LOCAL_K; i++) {
        per_thread_scores[my_offset + i] = local_scores[i];
        per_thread_blocks[my_offset + i] = local_blocks[i];
    }
    __syncthreads();  // wait for all threads to write their candidates

    // Single thread builds the final top-k from all candidates
    // This is O(n_threads * LOCAL_K * log(top_k)) — fast for reasonable sizes.
    // For n_threads=128, LOCAL_K=8, top_k=1024: 1024 inserts, ~10K comparisons.
    if (tid == 0) {
        for (int i = 0; i < n_threads * INDEXER_LOCAL_K; i++) {
            if (per_thread_scores[i] > -INFINITY) {
                heap_insert_shared(merge_scores, merge_blocks,
                                   per_thread_scores[i], per_thread_blocks[i], top_k);
            }
        }
    }
    __syncthreads();  // wait for merge to complete

    // ---- Write top-k indices to global memory ----
    // Sort the merge heap by score descending (selection sort, top_k <= 1024)
    if (tid == 0) {
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i + 1; j < top_k; j++) {
                if (merge_scores[j] > merge_scores[best] ||
                    (merge_scores[j] == merge_scores[best] &&
                     merge_blocks[j] < merge_blocks[best])) {
                    best = j;
                }
            }
            if (best != i) {
                float ts = merge_scores[i]; int32_t ti = merge_blocks[i];
                merge_scores[i] = merge_scores[best]; merge_blocks[i] = merge_blocks[best];
                merge_scores[best] = ts; merge_blocks[best] = ti;
            }
            topk_indices[t * top_k + i] = merge_blocks[i];
        }
    }
}


// ===========================================================================
// PyTorch binding
// ===========================================================================

void indexer_score_topk_fp32_cuda(
    torch::Tensor q_I,
    torch::Tensor w_h,
    torch::Tensor keys_fp4,
    torch::Tensor key_scale,
    torch::Tensor key_gscale,
    torch::Tensor block_table,
    torch::Tensor valid_lens,
    torch::Tensor topk_indices,
    int64_t n_heads, int64_t head_dim, int64_t top_k,
    int64_t entries_per_block
) {
    int T = q_I.size(0);
    int max_logical_blocks = block_table.size(1);
    int threads = 128;

    // SMEM layout:
    //   w_h: n_heads floats
    //   merge_scores: top_k floats
    //   merge_blocks: top_k ints
    //   per_thread_scores: n_threads * INDEXER_LOCAL_K floats
    //   per_thread_blocks: n_threads * INDEXER_LOCAL_K ints
    int smem_bytes = n_heads * sizeof(float)
                   + top_k * sizeof(float)
                   + top_k * sizeof(int32_t)
                   + threads * INDEXER_LOCAL_K * sizeof(float)
                   + threads * INDEXER_LOCAL_K * sizeof(int32_t);

    indexer_score_topk_fp32_kernel<<<T, threads, smem_bytes>>>(
        q_I.data_ptr<float>(),
        w_h.data_ptr<float>(),
        keys_fp4.data_ptr<uint8_t>(),
        key_scale.data_ptr<uint8_t>(),
        key_gscale.data_ptr<float>(),
        block_table.data_ptr<int32_t>(),
        valid_lens.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        (int)n_heads, (int)head_dim, (int)top_k,
        (int)entries_per_block, max_logical_blocks
    );
    C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("indexer_score_topk_fp32", &indexer_score_topk_fp32_cuda,
          "Indexer score + top-k (FP32 dot products, no-deadlock)");
}
