/**
 * DSV4 B2 — FP8 tensor-core indexer scoring + weighted ReLU + top-k.
 *
 * CSA Lightning Indexer (paper §2.3.1, eq. 16):
 *   I[t,s] = Σ_h w_h[t,h] · ReLU(q_I[t,h] · K^IComp[s])
 *
 * Decode-specialized Blackwell FP8 tensor-core path (T=1):
 *   1. Quantize Q (n_ih=64, ihd=128) BF16 → FP8_E4M3 with per-row FP32 scale.
 *   2. Run Q (128x128 padded) × K^T (128x128 tile) with tcgen05.mma.kind::f8f6f4.
 *   3. Read accumulator rows from TMEM with tcgen05.ld.32x32b.x8.
 *   4. Dequant logits in registers, apply ReLU, weighted sum across indexer heads.
 *   5. Block-local top-k selection.
 *
 * Important TMEM rule for M=128, cta_group::1:
 *   tcgen05.ld.32x32b.x8 does NOT use a row offset in the address.  The warp id in
 *   the first warpgroup selects the row/lane slice:
 *     warp 0 -> TMEM lanes/rows  0..31
 *     warp 1 -> TMEM lanes/rows 32..63
 *     warp 2 -> TMEM lanes/rows 64..95
 *     warp 3 -> TMEM lanes/rows 96..127
 *   All those warps use the same taddr for the same column group.
 *
 * No PyTorch fallback here.  No FP32 einsum.  The only FP32 CUDA-core work is the
 * unavoidable post-MMA dequant/ReLU/weighted-reduction/top-k epilogue.
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
#include <cmath>

static constexpr float E4M3_MAX = 448.0f;
static constexpr int NTHREADS = 192;
static constexpr int NWARPS = 6;
typedef unsigned short bf16_t;

// ---- PTX helpers ----
__device__ __forceinline__ float bf16_to_f32_ptx(bf16_t h) {
    float f; asm("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ uint8_t fp8_e4m3_from_f32(float x) {
    x = fminf(fmaxf(x, -E4M3_MAX), E4M3_MAX);
    __nv_fp8_e4m3 v(x);
    return *reinterpret_cast<uint8_t*>(&v);
}

// ---- UMMA helpers (mirrors the B1 FMHA helpers) ----
__device__ __forceinline__ uint64_t desc_encode(uint64_t byte_val) { return byte_val >> 4; }

__device__ __forceinline__ uint64_t make_umma_desc_kmajor_none(uint32_t smem_addr, int block_mn) {
    const uint64_t LBO = block_mn * 16;
    const uint64_t SBO = 128;
    uint64_t desc = 0;
    desc |= desc_encode(smem_addr) & 0x3FFF;
    desc |= (desc_encode(LBO) & 0x3FFF) << 16;
    desc |= (desc_encode(SBO) & 0x3FFF) << 32;
    desc |= 1ULL << 46;
    return desc;
}

__device__ __forceinline__ uint32_t make_idesc_f8_e4m3(int block_m, int block_n) {
    return (1U << 4) | ((uint32_t)(block_n >> 3) << 17) | ((uint32_t)(block_m >> 4) << 24);
}

__device__ __forceinline__ void umma_ss_f8f6f4(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
                                               uint32_t i_desc, bool accumulate) {
    uint32_t scaleC_bits = accumulate ? 0x3F800000u : 0u;
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(i_desc), "r"(scaleC_bits)
        : "memory");
}

__device__ __forceinline__ void tmem_alloc(uint32_t smem_ptr, int num_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_ptr), "r"(num_cols) : "memory");
}
__device__ __forceinline__ void tmem_dealloc(uint32_t tmem_ptr, int num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(tmem_ptr), "r"(num_cols) : "memory");
}

__device__ __forceinline__ void mbarrier_init_cta(uint32_t smem_mbar, uint32_t arrival_count = 1) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(smem_mbar), "r"(arrival_count) : "memory");
}

__device__ __forceinline__ void tcgen05_commit_mma(uint32_t smem_mbar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(smem_mbar) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cta(uint32_t smem_mbar, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "B2_WAIT_MMA:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, %2;\n\t"
        "@p bra.uni B2_DONE_MMA;\n\t"
        "bra.uni B2_WAIT_MMA;\n\t"
        "B2_DONE_MMA:\n\t"
        "}\n"
        :: "r"(smem_mbar), "r"(phase), "r"(0x989680)
        : "memory");
}

// ---- FP8 canonical SMEM layout for tcgen05.mma.kind::f8f6f4 ----
__device__ __forceinline__ int canon_idx_fp8_128x32(int r, int c) {
    int core_mn = r >> 3;
    int core_k = c >> 4;
    int local_r = r & 7;
    int local_c = c & 15;
    return core_k * 16 * 128 + core_mn * 128 + local_r * 16 + local_c;
}

// ---- Top-k helpers ----
#ifndef INDEXER_LOCAL_K
#define INDEXER_LOCAL_K 8
#endif

__device__ __forceinline__ void local_heap_insert(float* scores, int32_t* blocks,
                                                   float score, int32_t block_id, int k) {
    if (score <= scores[0]) return;
    scores[0] = score; blocks[0] = block_id;
    int root = 0;
    while (root < (k >> 1)) {
        int left = 2 * root + 1, right = 2 * root + 2, smallest = root;
        if (left < k && scores[left] < scores[smallest]) smallest = left;
        if (right < k && scores[right] < scores[smallest]) smallest = right;
        if (smallest == root) break;
        float ts = scores[root]; int32_t ti = blocks[root];
        scores[root] = scores[smallest]; blocks[root] = blocks[smallest];
        scores[smallest] = ts; blocks[smallest] = ti;
        root = smallest;
    }
}

__device__ __forceinline__ void heap_insert_shared(float* heap_scores, int32_t* heap_blocks,
                                                    float score, int32_t block_id, int k) {
    if (score <= heap_scores[0]) return;
    heap_scores[0] = score; heap_blocks[0] = block_id;
    int root = 0;
    while (root < (k >> 1)) {
        int left = 2 * root + 1, right = 2 * root + 2, smallest = root;
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
// Kernel
// ===========================================================================

template<int SK_TILE=128>
__global__ void __launch_bounds__(192)
indexer_fp8_score_topk_kernel(
    const bf16_t* __restrict__ q_bf16,     // (n_ih, ihd) BF16 row-major
    const uint8_t* __restrict__ k_fp8,     // (n_comp, ihd) FP8_E4M3 bytes
    const float* __restrict__ k_scale,     // (n_comp,) FP32 dequant scales
    const bf16_t* __restrict__ w_h_bf16,   // (n_ih,) BF16 weights
    int32_t* __restrict__ topk_indices,    // (top_k,) int32 output
    int n_comp, int n_ih, int ihd, int top_k
) {
    constexpr int MMA_K_F8 = 32;
    constexpr int NKT = 4;               // ihd=128 / 32
    constexpr int TILE_F8 = 128 * 32;    // bytes per canonical FP8 tile
    constexpr int TMEM_COLS = 512;       // full 128 lanes x 512 columns allocation

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const bool is_mma_warp = (wid == 4);

    __shared__ float sQ_amax_warp[NWARPS];

    // ---- SMEM layout ----
    extern __shared__ __align__(128) char sbuf[];
    size_t off = 0;
    uint32_t* sTmemBase = (uint32_t*)(sbuf + off); off += 4;
    off = (off + 15) & ~(size_t)15;
    uint64_t* sMbar = (uint64_t*)(sbuf + off); off += 8;
    off = (off + 127) & ~(size_t)127;

    uint8_t* sQ8 = (uint8_t*)(sbuf + off); off += TILE_F8;
    off = (off + 127) & ~(size_t)127;
    uint8_t* sK8 = (uint8_t*)(sbuf + off); off += TILE_F8;
    off = (off + 127) & ~(size_t)127;

    float* sQ_scale = (float*)(sbuf + off); off += 128 * sizeof(float);
    off = (off + 127) & ~(size_t)127;
    float* sW_h = (float*)(sbuf + off); off += 128 * sizeof(float);
    off = (off + 127) & ~(size_t)127;

    // Two warp partial sums: warp 0 covers heads 0..31, warp 1 covers 32..63.
    float* sWarpScores = (float*)(sbuf + off); off += 2 * SK_TILE * sizeof(float);
    off = (off + 127) & ~(size_t)127;

    float* sMergeScores = (float*)(sbuf + off); off += top_k * sizeof(float);
    int32_t* sMergeBlocks = (int32_t*)(sbuf + off); off += top_k * sizeof(int32_t);
    float* sCandScores = (float*)(sbuf + off); off += NTHREADS * INDEXER_LOCAL_K * sizeof(float);
    int32_t* sCandBlocks = (int32_t*)(sbuf + off); off += NTHREADS * INDEXER_LOCAL_K * sizeof(int32_t);

    float local_scores[INDEXER_LOCAL_K];
    int32_t local_blocks[INDEXER_LOCAL_K];
    #pragma unroll
    for (int i = 0; i < INDEXER_LOCAL_K; i++) {
        local_scores[i] = -INFINITY;
        local_blocks[i] = -1;
    }

    for (int i = tid; i < 128; i += NTHREADS) {
        sQ_scale[i] = 0.0f;
        sW_h[i] = 0.0f;
    }
    for (int i = tid; i < n_ih; i += NTHREADS) sW_h[i] = bf16_to_f32_ptx(w_h_bf16[i]);
    __syncthreads();

    // ---- Phase 0: Q per-row amax + scale ----
    for (int h = 0; h < n_ih; h++) {
        float local_max = 0.0f;
        for (int d = tid; d < ihd; d += NTHREADS) {
            local_max = fmaxf(local_max, fabsf(bf16_to_f32_ptx(q_bf16[h * ihd + d])));
        }
        for (int o = 16; o > 0; o >>= 1)
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, o));
        if (lane == 0) sQ_amax_warp[wid] = local_max;
        __syncthreads();

        float amax = 0.0f;
        if (tid < 32) {
            amax = (tid < NWARPS) ? sQ_amax_warp[tid] : 0.0f;
            for (int o = 16; o > 0; o >>= 1)
                amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, o));
        }
        if (tid == 0) {
            float scale = amax / E4M3_MAX;
            sQ_scale[h] = (scale < 1e-8f) ? 1e-8f : scale;
        }
        __syncthreads();
    }

    // ---- TMEM + mbarrier init ----
    const uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(sMbar);
    if (tid == 0) {
        mbarrier_init_cta(mbar_addr, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (is_mma_warp) tmem_alloc((uint32_t)__cvta_generic_to_shared(sTmemBase), TMEM_COLS);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    uint32_t tb = *sTmemBase;

    const int n_k_tiles = (n_comp + SK_TILE - 1) / SK_TILE;
    const uint32_t idesc_f8 = make_idesc_f8_e4m3(128, 128);
    int mma_phase = 0;

    for (int kv_tile = 0; kv_tile < n_k_tiles; kv_tile++) {
        const int kv_start = kv_tile * SK_TILE;
        const int kv_len = min(SK_TILE, n_comp - kv_start);

        for (int i = tid; i < 2 * SK_TILE; i += NTHREADS) sWarpScores[i] = 0.0f;
        __syncthreads();

        // ---- FP8 QK GEMM over ihd=128 in four K-slices ----
        for (int kt = 0; kt < NKT; kt++) {
            for (int i = tid; i < TILE_F8; i += NTHREADS) { sQ8[i] = 0; sK8[i] = 0; }
            __syncthreads();

            for (int i = tid; i < n_ih * MMA_K_F8; i += NTHREADS) {
                int row = i / MMA_K_F8;
                int col = i % MMA_K_F8;
                int d = kt * MMA_K_F8 + col;
                float val = bf16_to_f32_ptx(q_bf16[row * ihd + d]);
                sQ8[canon_idx_fp8_128x32(row, col)] = fp8_e4m3_from_f32(val / sQ_scale[row]);
            }
            for (int i = tid; i < kv_len * MMA_K_F8; i += NTHREADS) {
                int row = i / MMA_K_F8;
                int col = i % MMA_K_F8;
                int d = kt * MMA_K_F8 + col;
                int g_row = kv_start + row;
                sK8[canon_idx_fp8_128x32(row, col)] = k_fp8[(int64_t)g_row * ihd + d];
            }
            __syncthreads();
            // Generic-proxy SMEM writes above must be visible to the tcgen05 async proxy.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();

            if (is_mma_warp && lane == 0) {
                uint64_t dq = make_umma_desc_kmajor_none((uint32_t)__cvta_generic_to_shared(sQ8), 128);
                uint64_t dk = make_umma_desc_kmajor_none((uint32_t)__cvta_generic_to_shared(sK8), 128);
                umma_ss_f8f6f4(tb, dq, dk, idesc_f8, kt > 0);
            }
            __syncthreads();
        }

        // Track completion of all prior tcgen05.mma operations before TMEM reads.
        if (is_mma_warp && lane == 0) tcgen05_commit_mma(mbar_addr);
        mbarrier_wait_cta(mbar_addr, mma_phase);
        mma_phase ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        __syncthreads();

        // ---- Read TMEM and reduce across indexer heads ----
        // warps 0/1 read the same taddr; hardware maps them to lanes 0..31 / 32..63.
        if (wid < 2) {
            const int h = wid * 32 + lane;
            const bool h_valid = h < n_ih;
            const float q_s = h_valid ? sQ_scale[h] : 0.0f;
            const float wh = h_valid ? sW_h[h] : 0.0f;

            #pragma unroll
            for (int n = 0; n < SK_TILE / 8; n++) {
                int col_base = n * 8;
                float tmp[8];
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                    : "=f"(tmp[0]),"=f"(tmp[1]),"=f"(tmp[2]),"=f"(tmp[3]),
                      "=f"(tmp[4]),"=f"(tmp[5]),"=f"(tmp[6]),"=f"(tmp[7])
                    : "r"(tb + col_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

                float contrib[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int c = col_base + j;
                    if (h_valid && c < kv_len) {
                        float logit = tmp[j] * q_s * k_scale[kv_start + c];
                        contrib[j] = wh * fmaxf(logit, 0.0f);
                    } else {
                        contrib[j] = 0.0f;
                    }
                }

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float v = contrib[j];
                    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
                    if (lane == 0 && (col_base + j) < kv_len) {
                        sWarpScores[wid * SK_TILE + col_base + j] = v;
                    }
                }
            }
        }
        __syncthreads();

        // ---- Merge per-column scores into per-thread local top-k heaps ----
        for (int c = tid; c < kv_len; c += NTHREADS) {
            float score = sWarpScores[c] + sWarpScores[SK_TILE + c];
            local_heap_insert(local_scores, local_blocks, score, kv_start + c, INDEXER_LOCAL_K);
        }
        __syncthreads();
    }

    if (is_mma_warp) tmem_dealloc(tb, TMEM_COLS);
    __syncthreads();

    // ---- Block-level top-k merge ----
    for (int i = tid; i < top_k; i += NTHREADS) {
        sMergeScores[i] = -INFINITY;
        sMergeBlocks[i] = -1;
    }
    int my_offset = tid * INDEXER_LOCAL_K;
    #pragma unroll
    for (int i = 0; i < INDEXER_LOCAL_K; i++) {
        sCandScores[my_offset + i] = local_scores[i];
        sCandBlocks[my_offset + i] = local_blocks[i];
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < NTHREADS * INDEXER_LOCAL_K; i++) {
            if (sCandBlocks[i] >= 0) {
                heap_insert_shared(sMergeScores, sMergeBlocks,
                                   sCandScores[i], sCandBlocks[i], top_k);
            }
        }

        // Sort descending for deterministic torch.topk-like output order.
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i + 1; j < top_k; j++) {
                if (sMergeScores[j] > sMergeScores[best]) best = j;
            }
            if (best != i) {
                float ts = sMergeScores[i]; int32_t ti = sMergeBlocks[i];
                sMergeScores[i] = sMergeScores[best]; sMergeBlocks[i] = sMergeBlocks[best];
                sMergeScores[best] = ts; sMergeBlocks[best] = ti;
            }
            topk_indices[i] = sMergeBlocks[i];
        }
    }
}

// ===========================================================================
// PyTorch binding
// ===========================================================================

static size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

void indexer_fp8_score_topk_cuda(
    torch::Tensor q_bf16,         // (n_ih, ihd) BF16
    torch::Tensor k_fp8,          // (n_comp, ihd) uint8/float8_e4m3fn
    torch::Tensor k_scale,        // (n_comp,) FP32
    torch::Tensor w_h,            // (n_ih,) BF16
    torch::Tensor topk_indices,   // (top_k,) int32 output
    int64_t n_ih, int64_t ihd, int64_t top_k
) {
    TORCH_CHECK(q_bf16.is_cuda() && q_bf16.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(k_fp8.is_cuda());
    TORCH_CHECK(k_scale.is_cuda() && k_scale.scalar_type() == torch::kFloat32);
    TORCH_CHECK(w_h.is_cuda() && w_h.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(topk_indices.is_cuda() && topk_indices.scalar_type() == torch::kInt32);
    TORCH_CHECK(n_ih == 64 && ihd == 128, "B2 first pass is specialized to n_ih=64, ihd=128");
    TORCH_CHECK(top_k > 0, "top_k must be positive");

    int n_comp = k_fp8.size(0);
    TORCH_CHECK(n_comp > 0, "n_comp must be positive");
    TORCH_CHECK(k_fp8.size(1) == ihd, "k_fp8 must have shape (n_comp, ihd)");
    TORCH_CHECK(k_scale.numel() >= n_comp, "k_scale must contain at least n_comp scales");
    TORCH_CHECK(topk_indices.numel() >= top_k, "topk_indices is smaller than top_k");

    auto k8 = k_fp8.dtype() == torch::kUInt8 ? k_fp8 : k_fp8.view(torch::kUInt8);

    // Must exactly mirror kernel SMEM layout.  The previous B2 missed the score
    // scratch allocation, which can corrupt following SMEM and manifest as a hang.
    size_t smem = 0;
    smem += 4;                                // sTmemBase
    smem = align_up(smem, 16);
    smem += 8;                                // sMbar
    smem = align_up(smem, 128);
    smem += 128 * 32; smem = align_up(smem, 128);  // sQ8
    smem += 128 * 32; smem = align_up(smem, 128);  // sK8
    smem += 128 * 4;  smem = align_up(smem, 128);  // sQ_scale
    smem += 128 * 4;  smem = align_up(smem, 128);  // sW_h
    smem += 2 * 128 * 4; smem = align_up(smem, 128); // sWarpScores
    smem += (size_t)top_k * 4;                       // sMergeScores
    smem += (size_t)top_k * 4;                       // sMergeBlocks
    smem += 192 * INDEXER_LOCAL_K * 4;               // sCandScores
    smem += 192 * INDEXER_LOCAL_K * 4;               // sCandBlocks

    cudaFuncSetAttribute(indexer_fp8_score_topk_kernel<128>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    indexer_fp8_score_topk_kernel<128><<<1, 192, smem, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16_t*>(q_bf16.data_ptr<at::BFloat16>()),
        k8.data_ptr<uint8_t>(),
        k_scale.data_ptr<float>(),
        reinterpret_cast<const bf16_t*>(w_h.data_ptr<at::BFloat16>()),
        topk_indices.data_ptr<int32_t>(),
        n_comp, (int)n_ih, (int)ihd, (int)top_k);

    C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("indexer_fp8_score_topk", &indexer_fp8_score_topk_cuda,
          "B2 FP8 tensor-core indexer scoring + weighted ReLU + top-k");
}
