/**
 * DSV4 B1 — mixed FP8/BF16 prefill FMHA for DeepSeek-V4 attention KV.
 *
 * Extension of the decode kernel (fmha_mixed_fp8_decode.cuh) to support T > 1.
 * Same storage-native DSV4 layout as decode:
 *   Q noPE: FP8_E4M3 + per-row FP32 scale, Q RoPE: BF16
 *   KV noPE: FP8_E4M3 + per-row FP32 scale, KV RoPE: BF16
 *
 * Architecture:
 *   - noPE QK: f8f6f4 E4M3 x E4M3 -> FP32  (same MMA as decode)
 *   - RoPE QK: f16 BF16 x BF16 -> FP32      (same MMA as decode)
 *   - Multi-row softmax: T independent per-row softmax in SMEM (online algorithm)
 *   - PV: per query row (one PV MMA per row; correctness first, batched PV is TODO)
 *   - Sink bias: denominator-only logit per head
 *   - Output: normalized (BF16)
 *
 * SMEM budget: process in T_BATCH sub-batches to fit in 232KB.
 * T_BATCH=32: sOacc=64KB, sLogits=16KB, sP=16KB, rest=40KB → ~136KB ✓
 * T_BATCH=64: sOacc=128KB, sLogits=32KB, sP=32KB, rest=40KB → ~232KB (tight)
 *
 * Supports T=1..128. For T>128, caller must split into multiple launches.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp8.hpp>
#include <cstdint>
#include <cmath>
#include "fmha_common.cuh"
#include "fmha_umma_desc.cuh"

namespace dsv4::kernels::attention {

struct FmhaMixedFp8PrefillParams {
    const uint8_t* __restrict__ q_nope_fp8;   // (B,H,T,NOPE)
    const float* __restrict__ q_nope_scale;   // (B,H,T)
    const bf16_t* __restrict__ q_rope_bf16;   // (B,H,T,ROPE)

    const uint8_t* __restrict__ k_nope_fp8;   // (N,NOPE), MQA shared
    const float* __restrict__ k_nope_scale;   // (N,)
    const bf16_t* __restrict__ k_rope_bf16;   // (N,ROPE)

    bf16_t* __restrict__ o;                   // (B,H,T,HD)
    float* __restrict__ lse;                  // (B,H,T), optional
    const float* __restrict__ sink_bias;      // (B,H), optional

    int B, H, T, N, HD, NOPE, ROPE;
    int q_nope_t_stride, q_nope_head_stride, q_nope_batch_stride;
    int q_scale_t_stride, q_scale_head_stride, q_scale_batch_stride;
    int q_rope_t_stride, q_rope_head_stride, q_rope_batch_stride;
    int o_head_stride, o_batch_stride, o_t_stride;
    int lse_head_stride, lse_batch_stride, lse_t_stride;
    float scale;
};

// ---- Reuse helpers from decode kernel ----

__device__ __forceinline__ float _prefill_fp8_to_f32(uint8_t byte) {
    __nv_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v) = byte;
    return static_cast<float>(v);
}

__device__ __forceinline__ int _pfill_cidx_f8(int r, int c) {
    int cm = r >> 3, ck = c >> 4, lr = r & 7, lc = c & 15;
    return ck * 16 * 128 + cm * 128 + lr * 16 + lc;
}

__device__ __forceinline__ int _pfill_cidx_bf16_128(int r, int c) {
    int cm = r >> 3, ck = c >> 3, lr = r & 7, lc = c & 7;
    return ck * 16 * 64 + cm * 64 + lr * 8 + lc;
}

__device__ __forceinline__ int _pfill_cidx_bf16_16(int r, int c) {
    int cm = r >> 3, ck = c >> 3, lr = r & 7, lc = c & 7;
    return ck * 2 * 64 + cm * 64 + lr * 8 + lc;
}

/**
 * Read T_ACT rows of QK TMEM result into sLogits (T_ACT × SK_TILE).
 *
 * tcgen05.ld.32x32b.x8 reads 32 rows × 8 columns per call.
 * Warp 0 → rows 0-31, Warp 1 → rows 32-63 (from SAME TMEM address).
 * Rows 64-127 require TMEM base offset +256.
 *
 * Only warps 0 and 1 participate.
 */
template<int SK_TILE=128>
__device__ void prefill_read_qk_rows(uint32_t tb, float* sLogits,
                                      int T_ACT, int kv_len) {
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= 2) return;

    // 2 super-groups: rows 0-63 (tb+0), rows 64-127 (tb+256)
    for (int sg = 0; sg < 2; sg++) {
        int row_base = sg * 64;
        if (row_base >= T_ACT) break;

        uint32_t sg_off = sg * 256;
        int warp_row = row_base + (wid == 0 ? 0 : 32);
        if (warp_row >= T_ACT) continue;

        for (int n = 0; n < SK_TILE / 8; n++) {
            float tmp[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                : "=f"(tmp[0]),"=f"(tmp[1]),"=f"(tmp[2]),"=f"(tmp[3]),
                  "=f"(tmp[4]),"=f"(tmp[5]),"=f"(tmp[6]),"=f"(tmp[7])
                : "r"(tb + sg_off + n * 8));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

            int row = warp_row + lane;
            if (row < T_ACT) {
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    int col = n * 8 + c;
                    sLogits[row * SK_TILE + col] = (col < kv_len) ? tmp[c] : -INFINITY;
                }
            }
        }
    }
}

/**
 * Read a single row (query row qr) from PV TMEM result.
 * The PV MMA result has 128 rows, but only row qr has valid data.
 * Using tcgen05.ld.32x32b.x8, lane (qr % 32) holds row qr's data.
 * For qr >= 64, offset TMEM base by 256.
 *
 * Writes 16 values (one n_sub PV output) to sOacc[qr*HD + d_base + 0..15].
 */
/**
 * Read a single row (query row qr) from ALL PV TMEM results.
 * Uses the SAME approach as the decode kernel PV read, but extracts
 * from the lane corresponding to row qr instead of always lane 0.
 *
 * For qr < 32: warp 0, lane qr
 * For qr 32-63: warp 1, lane (qr-32) -- same TMEM address, different rows
 * For qr 64-95: same but TMEM offset +256
 * For qr 96-127: same but TMEM offset +256
 *
 * This mirrors the proven decode kernel read pattern exactly.
 */
template<int HD=512, int N_SUB=32>
__device__ void prefill_read_pv_all_subs(uint32_t tb, int qr,
                                          float* sOacc, float rescale) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    int local_lane = qr % 32;
    int target_wid = (qr < 32) ? 0 : 1;
    uint32_t rg_off = (qr >= 64) ? 256 : 0;

    for (int n = 0; n < HD / 8; n++) {
        float tmp[8];
        if (wid == target_wid) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                : "=f"(tmp[0]),"=f"(tmp[1]),"=f"(tmp[2]),"=f"(tmp[3]),
                  "=f"(tmp[4]),"=f"(tmp[5]),"=f"(tmp[6]),"=f"(tmp[7])
                : "r"(tb + rg_off + n * 8));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        }

        if (wid == target_wid && lane == local_lane) {
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                int d = n * 8 + c;
                sOacc[qr * HD + d] += tmp[c] * rescale;
            }
        }
    }
}

/**
 * Prefill kernel: T query rows, processing in T_BATCH sub-batches.
 *
 * T_BATCH controls the SMEM usage. T_BATCH=32 uses ~136KB. T_BATCH=64 uses ~232KB.
 * For each sub-batch of T_BATCH rows, we iterate over all KV tiles, computing
 * QK → softmax → PV for those rows.
 */
template<int HD=512, int NOPE=448, int ROPE=64, int SK_TILE=128, int T_BATCH=32>
__global__ void __launch_bounds__(192)
fmha_mixed_fp8_prefill_kernel(FmhaMixedFp8PrefillParams p) {
    static_assert(HD == 512 && NOPE == 448 && ROPE == 64,
        "B1 prefill kernel specialized for DSV4 HD=512/NOPE=448/ROPE=64");

    constexpr int MMA_K_F8 = 32;
    constexpr int MMA_K_F16 = 16;
    constexpr int NKT_NOPE = NOPE / MMA_K_F8;
    constexpr int NKT_ROPE = ROPE / MMA_K_F16;
    constexpr int NKT_PV = SK_TILE / MMA_K_F16;
    constexpr int N_SUB = HD / 16;
    constexpr int TILE_F8 = 128 * MMA_K_F8;
    constexpr int TILE_F16 = 128 * MMA_K_F16;
    constexpr int V_SUB_SZ = 16 * MMA_K_F16;
    constexpr int TMEM_COLS = 512;

    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const bool is_mma_warp = (wid == 4);
    const int n_kv_tiles = (p.N + SK_TILE - 1) / SK_TILE;

    const uint8_t* q8 = p.q_nope_fp8 + batch_idx * p.q_nope_batch_stride + head_idx * p.q_nope_head_stride;
    const float* q8_scale = p.q_nope_scale + batch_idx * p.q_scale_batch_stride + head_idx * p.q_scale_head_stride;
    const bf16_t* qrope = p.q_rope_bf16 + batch_idx * p.q_rope_batch_stride + head_idx * p.q_rope_head_stride;

    // SMEM layout — sized for T_BATCH rows
    extern __shared__ __align__(128) char sbuf[];
    size_t off = 0;
    uint32_t* sTmemBase = (uint32_t*)(sbuf + off); off += 4;
    off = (off + 127) & ~(size_t)127;
    uint8_t* sQ8 = (uint8_t*)(sbuf + off); off += TILE_F8;
    off = (off + 127) & ~(size_t)127;
    uint8_t* sK8 = (uint8_t*)(sbuf + off); off += TILE_F8;
    off = (off + 127) & ~(size_t)127;
    bf16_t* sQ16 = (bf16_t*)(sbuf + off); off += TILE_F16 * sizeof(bf16_t);
    off = (off + 127) & ~(size_t)127;
    bf16_t* sK16 = (bf16_t*)(sbuf + off); off += TILE_F16 * sizeof(bf16_t);
    off = (off + 127) & ~(size_t)127;
    bf16_t* sPk = (bf16_t*)(sbuf + off); off += TILE_F16 * sizeof(bf16_t);
    off = (off + 127) & ~(size_t)127;
    bf16_t* sV = (bf16_t*)(sbuf + off); off += V_SUB_SZ * sizeof(bf16_t);
    off = (off + 127) & ~(size_t)127;
    // Per-sub-batch SMEM
    float* sLogits = (float*)(sbuf + off); off += T_BATCH * SK_TILE * sizeof(float);
    float* sP = (float*)(sbuf + off); off += T_BATCH * SK_TILE * sizeof(float);
    float* sOacc = (float*)(sbuf + off); off += T_BATCH * HD * sizeof(float);
    float* sRunningMax = (float*)(sbuf + off); off += T_BATCH * sizeof(float);
    float* sRunningSum = (float*)(sbuf + off); off += T_BATCH * sizeof(float);
    bf16_t* sOepi = (bf16_t*)(sbuf + off); off += T_BATCH * HD * sizeof(bf16_t);

    // TMEM alloc
    if (is_mma_warp) tmem_alloc((uint32_t)__cvta_generic_to_shared(sTmemBase), TMEM_COLS);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    uint32_t tb = *sTmemBase;

    const uint32_t idesc_f8_qk = make_idesc_f8_e4m3(128, 128);
    const uint32_t idesc_f16_qk = make_idesc(128, 128);
    const uint32_t idesc_pv = make_idesc(128, 16);

    // ================================================================
    // Outer loop: process T_BATCH rows at a time
    // ================================================================
    for (int t_start = 0; t_start < p.T; t_start += T_BATCH) {
        int T_ACT = min(T_BATCH, p.T - t_start);

        // Initialize accumulators for this sub-batch
        for (int i = tid; i < T_ACT * HD; i += blockDim.x) sOacc[i] = 0.0f;
        for (int t = tid; t < T_ACT; t += blockDim.x) {
            sRunningMax[t] = -INFINITY;
            sRunningSum[t] = 0.0f;
        }
        __syncthreads();

        // ============================================================
        // KV-tile loop (shared across all sub-batch rows)
        // ============================================================
        for (int kv_tile = 0; kv_tile < n_kv_tiles; kv_tile++) {
            const int kv_start = kv_tile * SK_TILE;
            const int kv_len = min(SK_TILE, p.N - kv_start);

            // --------------------------------------------------------
            // QK noPE: FP8 tensor cores
            // Write T_ACT rows of Q (not just row 0)
            // --------------------------------------------------------
            for (int kt = 0; kt < NKT_NOPE; kt++) {
                for (int i = tid; i < TILE_F8; i += blockDim.x) { sQ8[i] = 0; sK8[i] = 0; }
                __syncthreads();
                // T_ACT rows of Q
                for (int r = tid; r < T_ACT; r += blockDim.x) {
                    int qr = t_start + r;
                    for (int c = 0; c < MMA_K_F8; c++) {
                        int d = kt * MMA_K_F8 + c;
                        sQ8[_pfill_cidx_f8(r, c)] = q8[qr * p.q_nope_t_stride + d];
                    }
                }
                // K: same as decode
                for (int i = tid; i < kv_len * MMA_K_F8; i += blockDim.x) {
                    int r = i / MMA_K_F8, c = i % MMA_K_F8;
                    int d = kt * MMA_K_F8 + c;
                    sK8[_pfill_cidx_f8(r, c)] = p.k_nope_fp8[(int64_t)(kv_start + r) * NOPE + d];
                }
                __syncthreads();
                if (is_mma_warp && lane == 0) {
                    uint64_t dq = make_umma_desc_kmajor_none((uint32_t)__cvta_generic_to_shared(sQ8), 128);
                    uint64_t dk = make_umma_desc_kmajor_none((uint32_t)__cvta_generic_to_shared(sK8), 128);
                    umma_ss_f8f6f4(tb, dq, dk, idesc_f8_qk, kt > 0);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                }
                __syncthreads();
            }
            asm volatile("fence.sc.gpu;" ::: "memory");
            __syncthreads();

            // Read all T_ACT rows of QK noPE result
            prefill_read_qk_rows<SK_TILE>(tb, sLogits, T_ACT, kv_len);
            __syncthreads();

            // Apply Q and K scales
            for (int r = tid; r < T_ACT; r += blockDim.x) {
                int qr = t_start + r;
                float q_s = q8_scale[qr * p.q_scale_t_stride];
                for (int c = 0; c < kv_len; c++) {
                    float ks = p.k_nope_scale[kv_start + c];
                    sLogits[r * SK_TILE + c] *= q_s * ks;
                }
            }
            __syncthreads();

            // --------------------------------------------------------
            // QK RoPE: BF16 tensor cores
            // --------------------------------------------------------
            for (int kt = 0; kt < NKT_ROPE; kt++) {
                for (int i = tid; i < TILE_F16; i += blockDim.x) { sQ16[i] = 0; sK16[i] = 0; }
                __syncthreads();
                for (int r = tid; r < T_ACT; r += blockDim.x) {
                    int qr = t_start + r;
                    for (int c = 0; c < MMA_K_F16; c++) {
                        int d = kt * MMA_K_F16 + c;
                        sQ16[_pfill_cidx_bf16_128(r, c)] = qrope[qr * p.q_rope_t_stride + d];
                    }
                }
                for (int i = tid; i < kv_len * MMA_K_F16; i += blockDim.x) {
                    int r = i / MMA_K_F16, c = i % MMA_K_F16;
                    int d = kt * MMA_K_F16 + c;
                    sK16[_pfill_cidx_bf16_128(r, c)] = p.k_rope_bf16[(int64_t)(kv_start + r) * ROPE + d];
                }
                __syncthreads();
                if (is_mma_warp && lane == 0) {
                    uint64_t dq = make_umma_desc_kmajor_none((uint32_t)__cvta_generic_to_shared(sQ16), 128);
                    uint64_t dk = make_umma_desc_kmajor_none((uint32_t)__cvta_generic_to_shared(sK16), 128);
                    umma_ss_f16(tb, dq, dk, idesc_f16_qk, kt > 0);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                }
                __syncthreads();
            }
            asm volatile("fence.sc.gpu;" ::: "memory");
            __syncthreads();

            // Add RoPE logits to noPE logits (reuse sP as temp buffer)
            prefill_read_qk_rows<SK_TILE>(tb, sP, T_ACT, kv_len);
            __syncthreads();
            for (int i = tid; i < T_ACT * kv_len; i += blockDim.x) {
                sLogits[i] += sP[i];
            }
            __syncthreads();

            // --------------------------------------------------------
            // Per-row softmax (online algorithm)
            // Each thread handles a few rows
            // --------------------------------------------------------
            for (int r = tid; r < T_ACT; r += blockDim.x) {
                float tile_max = -INFINITY;
                for (int c = 0; c < kv_len; c++)
                    tile_max = fmaxf(tile_max, sLogits[r * SK_TILE + c] * p.scale);

                float tile_sum = 0.0f;
                for (int c = 0; c < kv_len; c++) {
                    float pv = expf(sLogits[r * SK_TILE + c] * p.scale - tile_max);
                    sP[r * SK_TILE + c] = pv;
                    tile_sum += pv;
                }
                for (int c = kv_len; c < SK_TILE; c++) sP[r * SK_TILE + c] = 0.0f;

                float old_max = sRunningMax[r];
                float new_max = fmaxf(old_max, tile_max);
                float rescale_old = (old_max > -INFINITY) ? expf(old_max - new_max) : 0.0f;
                for (int d = 0; d < HD; d++) sOacc[r * HD + d] *= rescale_old;
                float rescale_new = expf(tile_max - new_max);
                sRunningSum[r] = sRunningSum[r] * rescale_old + tile_sum * rescale_new;
                sRunningMax[r] = new_max;

                // Store rescale_new for PV (reuse sLogits first column)
                sLogits[r * SK_TILE] = rescale_new;
            }
            __syncthreads();

            // --------------------------------------------------------
            // PV: per query row (one PV MMA per row)
            // TODO: batch all T_ACT rows into one PV MMA for performance
            // --------------------------------------------------------
            for (int qr = 0; qr < T_ACT; qr++) {
                float p_rescale = sLogits[qr * SK_TILE];

                for (int n_sub = 0; n_sub < N_SUB; n_sub++) {
                    int d_base = n_sub * 16;
                    for (int pv_kt = 0; pv_kt < NKT_PV; pv_kt++) {
                        const int col_start = pv_kt * MMA_K_F16;
                        for (int i = tid; i < TILE_F16; i += blockDim.x) sPk[i] = 0;
                        for (int i = tid; i < V_SUB_SZ; i += blockDim.x) sV[i] = 0;
                        __syncthreads();

                        // P matrix: only row qr is active
                        for (int c = tid; c < MMA_K_F16; c += blockDim.x) {
                            int gc = col_start + c;
                            sPk[_pfill_cidx_bf16_128(qr, c)] = f32_to_bf16(sP[qr * SK_TILE + gc]);
                        }

                        // V matrix (same as decode)
                        for (int i = tid; i < 16 * MMA_K_F16; i += blockDim.x) {
                            int dd = i / MMA_K_F16, kk = i % MMA_K_F16;
                            int row = col_start + kk;
                            int g_row = kv_start + row;
                            int d = d_base + dd;
                            bf16_t vbits = 0;
                            if (row < kv_len) {
                                if (d < NOPE) {
                                    uint8_t b = p.k_nope_fp8[(int64_t)g_row * NOPE + d];
                                    float v = _prefill_fp8_to_f32(b) * p.k_nope_scale[g_row];
                                    vbits = f32_to_bf16(v);
                                } else {
                                    vbits = p.k_rope_bf16[(int64_t)g_row * ROPE + (d - NOPE)];
                                }
                            }
                            sV[_pfill_cidx_bf16_16(dd, kk)] = vbits;
                        }
                        __syncthreads();

                        bool first = (pv_kt == 0);  // Fresh for each query row's PV
                        if (is_mma_warp && lane == 0) {
                            uint64_t dp = make_umma_desc_kmajor_none((uint32_t)__cvta_generic_to_shared(sPk), 128);
                            uint64_t dv = make_umma_desc_kmajor_none((uint32_t)__cvta_generic_to_shared(sV), 16);
                            umma_ss_f16(tb + n_sub * 16, dp, dv, idesc_pv, !first);
                            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                        }
                        __syncthreads();
                    }  // pv_kt
                }  // n_sub

                // Read PV result for row qr from TMEM
                asm volatile("fence.sc.gpu;" ::: "memory");
                __syncthreads();
                prefill_read_pv_all_subs<HD, N_SUB>(tb, qr, sOacc, p_rescale);
                __syncthreads();
            }  // qr
        }  // kv_tile

        // --------------------------------------------------------
        // Attention sink
        // --------------------------------------------------------
        if (p.sink_bias != nullptr) {
            float sb = p.sink_bias[batch_idx * p.H + head_idx];
            for (int r = tid; r < T_ACT; r += blockDim.x) {
                float old_max = sRunningMax[r];
                float new_max = fmaxf(old_max, sb);
                float rescale_old = (old_max > -INFINITY) ? expf(old_max - new_max) : 0.0f;
                for (int d = 0; d < HD; d++) sOacc[r * HD + d] *= rescale_old;
                sRunningSum[r] = sRunningSum[r] * rescale_old + expf(sb - new_max);
                sRunningMax[r] = new_max;
            }
            __syncthreads();
        }

        // --------------------------------------------------------
        // Normalize and write output
        // --------------------------------------------------------
        bf16_t* out = p.o + batch_idx * p.o_batch_stride + head_idx * p.o_head_stride;
        float* lse = p.lse ? p.lse + batch_idx * p.lse_batch_stride + head_idx * p.lse_head_stride : nullptr;

        for (int r = tid; r < T_ACT; r += blockDim.x) {
            float inv_sum = 1.0f / sRunningSum[r];
            int qr = t_start + r;
            for (int d = 0; d < HD; d++) {
                bf16_t val = f32_to_bf16(sOacc[r * HD + d] * inv_sum);
                sOepi[r * HD + d] = val;
            }
            if (lse) lse[qr * p.lse_t_stride] = logf(sRunningSum[r]) + sRunningMax[r];
        }
        __syncthreads();

        // Write to GMEM
        for (int r = 0; r < T_ACT; r++) {
            int qr = t_start + r;
            bf16_t* out_row = out + qr * p.o_t_stride;
            for (int d = tid; d < HD; d += blockDim.x) out_row[d] = sOepi[r * HD + d];
        }
        __syncthreads();

    }  // t_start sub-batch loop

    if (is_mma_warp) tmem_dealloc(tb, TMEM_COLS);
}

} // namespace dsv4::kernels::attention
