/**
 * DSV4 B1 — mixed FP8/BF16 decode FMHA for DeepSeek-V4 attention KV.
 *
 * Inputs are the storage-native DSV4 layout:
 *   Q noPE: FP8_E4M3 + per-row FP32 scale, Q RoPE: BF16
 *   KV noPE: FP8_E4M3 + per-row FP32 scale, KV RoPE: BF16
 *
 * This first B1 kernel targets the decode hot path (T == 1) and HD=512,
 * NOPE=448, ROPE=64. It removes the global FP8->BF16 KV dequant/gather and
 * uses Blackwell tcgen05 tensor cores for:
 *   - noPE QK: f8f6f4 E4M3 x E4M3 -> FP32
 *   - RoPE QK: f16 BF16 x BF16 -> FP32
 *   - PV:      f16 BF16 x BF16 -> FP32, with noPE V dequantized only into SMEM
 *
 * The noPE KV is never materialized as a global BF16 buffer.
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

struct FmhaMixedFp8DecodeParams {
    const uint8_t* __restrict__ q_nope_fp8;   // (B,H,1,NOPE)
    const float* __restrict__ q_nope_scale;   // (B,H,1)
    const bf16_t* __restrict__ q_rope_bf16;   // (B,H,1,ROPE)

    const uint8_t* __restrict__ k_nope_fp8;   // (N,NOPE), MQA shared
    const float* __restrict__ k_nope_scale;   // (N,)
    const bf16_t* __restrict__ k_rope_bf16;   // (N,ROPE)

    bf16_t* __restrict__ o;                   // (B,H,1,HD)
    float* __restrict__ lse;                  // (B,H,1), optional
    const float* __restrict__ sink_bias;      // (B,H), optional

    int B, H, N, HD, NOPE, ROPE;
    int q_nope_head_stride, q_nope_batch_stride;
    int q_scale_head_stride, q_scale_batch_stride;
    int q_rope_head_stride, q_rope_batch_stride;
    int o_head_stride, o_batch_stride;
    int lse_head_stride, lse_batch_stride;
    float scale;
};

__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t byte) {
    __nv_fp8_e4m3 v;
    *reinterpret_cast<uint8_t*>(&v) = byte;
    return static_cast<float>(v);
}

// FP8 canonical K-major layout for tcgen05.mma.kind::f8f6f4.
// Logical matrix shape is (128, 32): 8 row groups x 16 FP8 columns per 128B atom.
__device__ __forceinline__ int canon_idx_fp8_128x32(int r, int c) {
    constexpr int CORES_MN = 16;  // 128 / 8
    int core_mn = r >> 3;
    int core_k = c >> 4;          // 16 FP8 values = 16B atom width
    int local_r = r & 7;
    int local_c = c & 15;
    return core_k * CORES_MN * 128 + core_mn * 128 + local_r * 16 + local_c;
}

__device__ __forceinline__ int canon_idx_bf16_128x16(int r, int c) {
    constexpr int CORES_MN = 16;
    int core_mn = r >> 3;
    int core_k = c >> 3;
    int local_r = r & 7;
    int local_c = c & 7;
    return core_k * CORES_MN * 64 + core_mn * 64 + local_r * 8 + local_c;
}

__device__ __forceinline__ int canon_idx_bf16_16x16(int r, int c) {
    constexpr int CORES_MN = 2;   // 16 / 8
    int core_mn = r >> 3;
    int core_k = c >> 3;
    int local_r = r & 7;
    int local_c = c & 7;
    return core_k * CORES_MN * 64 + core_mn * 64 + local_r * 8 + local_c;
}

__device__ __forceinline__ bf16_t f32_to_bf16_bits(float x) { return f32_to_bf16(x); }

// Read row 0 of a 128-wide TMEM result. Must be called by a full warp;
// lane 0 receives row 0, lanes 1..31 receive rows 1..31 and are ignored.
__device__ __forceinline__ void read_tmem_row0_128(uint32_t tb, float* out128, bool lane0) {
    for (int n = 0; n < 16; n++) {
        float tmp[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
            : "=f"(tmp[0]),"=f"(tmp[1]),"=f"(tmp[2]),"=f"(tmp[3]),
              "=f"(tmp[4]),"=f"(tmp[5]),"=f"(tmp[6]),"=f"(tmp[7])
            : "r"(tb + n * 8));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (lane0) {
            #pragma unroll
            for (int c = 0; c < 8; c++) out128[n * 8 + c] = tmp[c];
        }
    }
}

template<int HD=512, int NOPE=448, int ROPE=64, int SK_TILE=128>
__global__ void __launch_bounds__(192)
fmha_mixed_fp8_decode_kernel(FmhaMixedFp8DecodeParams p) {
    static_assert(HD == 512 && NOPE == 448 && ROPE == 64, "B1 first pass is specialized for DSV4 HD=512/NOPE=448/ROPE=64");
    constexpr int MMA_K_F8 = 32;
    constexpr int MMA_K_F16 = 16;
    constexpr int NKT_NOPE = NOPE / MMA_K_F8;
    constexpr int NKT_ROPE = ROPE / MMA_K_F16;
    constexpr int NKT_PV = SK_TILE / MMA_K_F16;
    constexpr int N_SUB = HD / 16;
    constexpr int TILE_F8 = 128 * MMA_K_F8;    // bytes
    constexpr int TILE_F16 = 128 * MMA_K_F16;  // bf16 elements
    constexpr int V_SUB_SZ = 16 * MMA_K_F16;   // bf16 elements
    constexpr int TMEM_COLS = 512;

    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const bool is_mma_warp = (wid == 4);
    const bool is_lane0 = (wid == 0 && lane == 0);
    const int n_kv_tiles = (p.N + SK_TILE - 1) / SK_TILE;

    const uint8_t* q8 = p.q_nope_fp8 + batch_idx * p.q_nope_batch_stride + head_idx * p.q_nope_head_stride;
    const float q8_scale = p.q_nope_scale[batch_idx * p.q_scale_batch_stride + head_idx * p.q_scale_head_stride];
    const bf16_t* qrope = p.q_rope_bf16 + batch_idx * p.q_rope_batch_stride + head_idx * p.q_rope_head_stride;
    bf16_t* out = p.o + batch_idx * p.o_batch_stride + head_idx * p.o_head_stride;
    float* lse = p.lse ? p.lse + batch_idx * p.lse_batch_stride + head_idx * p.lse_head_stride : nullptr;

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
    float* sLogits = (float*)(sbuf + off); off += SK_TILE * sizeof(float);
    float* sP = (float*)(sbuf + off); off += SK_TILE * sizeof(float);
    float* sOacc = (float*)(sbuf + off); off += HD * sizeof(float);
    bf16_t* sOepi = (bf16_t*)(sbuf + off); off += HD * sizeof(bf16_t);

    if (is_mma_warp) tmem_alloc((uint32_t)__cvta_generic_to_shared(sTmemBase), TMEM_COLS);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    uint32_t tb = *sTmemBase;

    if (tid < HD) sOacc[tid] = 0.0f;
    if (tid < SK_TILE) { sLogits[tid] = -INFINITY; sP[tid] = 0.0f; }
    __syncthreads();

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const uint32_t idesc_f8_qk = make_idesc_f8_e4m3(128, 128);
    const uint32_t idesc_f16_qk = make_idesc(128, 128);
    const uint32_t idesc_pv = make_idesc(128, 16);

    for (int kv_tile = 0; kv_tile < n_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * SK_TILE;
        const int kv_len = min(SK_TILE, p.N - kv_start);

        // ------------------------------------------------------------
        // QK noPE: FP8 tensor cores, raw logits in TMEM.
        // ------------------------------------------------------------
        for (int kt = 0; kt < NKT_NOPE; kt++) {
            for (int i = tid; i < TILE_F8; i += blockDim.x) { sQ8[i] = 0; sK8[i] = 0; }
            __syncthreads();
            for (int c = tid; c < MMA_K_F8; c += blockDim.x) {
                int d = kt * MMA_K_F8 + c;
                sQ8[canon_idx_fp8_128x32(0, c)] = q8[d];
            }
            for (int i = tid; i < kv_len * MMA_K_F8; i += blockDim.x) {
                int r = i / MMA_K_F8, c = i % MMA_K_F8;
                int d = kt * MMA_K_F8 + c;
                sK8[canon_idx_fp8_128x32(r, c)] = p.k_nope_fp8[(int64_t)(kv_start + r) * NOPE + d];
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

        if (wid == 0) read_tmem_row0_128(tb, sLogits, lane == 0);
        __syncthreads();
        if (is_lane0) {
            #pragma unroll
            for (int c = 0; c < SK_TILE; c++) {
                if (c < kv_len) {
                    float ks = p.k_nope_scale[kv_start + c];
                    sLogits[c] = sLogits[c] * q8_scale * ks;
                } else {
                    sLogits[c] = -INFINITY;
                }
            }
        }
        __syncthreads();

        // ------------------------------------------------------------
        // QK RoPE: BF16 tensor cores, then add to scaled noPE logits.
        // ------------------------------------------------------------
        for (int kt = 0; kt < NKT_ROPE; kt++) {
            for (int i = tid; i < TILE_F16; i += blockDim.x) { sQ16[i] = 0; sK16[i] = 0; }
            __syncthreads();
            for (int c = tid; c < MMA_K_F16; c += blockDim.x) {
                int d = kt * MMA_K_F16 + c;
                sQ16[canon_idx_bf16_128x16(0, c)] = qrope[d];
            }
            for (int i = tid; i < kv_len * MMA_K_F16; i += blockDim.x) {
                int r = i / MMA_K_F16, c = i % MMA_K_F16;
                int d = kt * MMA_K_F16 + c;
                sK16[canon_idx_bf16_128x16(r, c)] = p.k_rope_bf16[(int64_t)(kv_start + r) * ROPE + d];
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

        // Use sP as a temporary row buffer here; probabilities are formed later.
        if (wid == 0) read_tmem_row0_128(tb, sP, lane == 0);
        __syncthreads();
        if (is_lane0) {
            for (int c = 0; c < kv_len; c++) sLogits[c] += sP[c];
        }
        __syncthreads();

        // ------------------------------------------------------------
        // Softmax tile probabilities for row 0.
        // ------------------------------------------------------------
        float tile_max = -INFINITY;
        if (is_lane0) {
            for (int c = 0; c < kv_len; c++) tile_max = fmaxf(tile_max, sLogits[c] * p.scale);
            float tile_sum = 0.0f;
            for (int c = 0; c < kv_len; c++) {
                float pv = expf(sLogits[c] * p.scale - tile_max);
                sP[c] = pv;
                tile_sum += pv;
            }
            for (int c = kv_len; c < SK_TILE; c++) sP[c] = 0.0f;

            float new_max = fmaxf(running_max, tile_max);
            float rescale_old = (running_max > -INFINITY) ? expf(running_max - new_max) : 0.0f;
            for (int d = 0; d < HD; d++) sOacc[d] *= rescale_old;
            running_sum = running_sum * rescale_old + tile_sum * expf(tile_max - new_max);
            running_max = new_max;
        }
        __syncthreads();

        // ------------------------------------------------------------
        // PV: probabilities BF16 x V BF16. noPE V is dequantized into SMEM only.
        // ------------------------------------------------------------
        for (int n_sub = 0; n_sub < N_SUB; n_sub++) {
            int d_base = n_sub * 16;
            for (int pv_kt = 0; pv_kt < NKT_PV; pv_kt++) {
                const int col_start = pv_kt * MMA_K_F16;
                for (int i = tid; i < TILE_F16; i += blockDim.x) sPk[i] = 0;
                for (int i = tid; i < V_SUB_SZ; i += blockDim.x) sV[i] = 0;
                __syncthreads();

                // P matrix: only row 0 non-zero.
                for (int c = tid; c < MMA_K_F16; c += blockDim.x) {
                    int gc = col_start + c;
                    sPk[canon_idx_bf16_128x16(0, c)] = f32_to_bf16_bits(sP[gc]);
                }

                // V matrix B: logical (16 K rows, 16 N cols) in BF16 canonical layout.
                for (int i = tid; i < 16 * MMA_K_F16; i += blockDim.x) {
                    int dd = i / MMA_K_F16;
                    int kk = i % MMA_K_F16;
                    int row = col_start + kk;
                    int g_row = kv_start + row;
                    int d = d_base + dd;
                    bf16_t vbits = 0;
                    if (row < kv_len) {
                        if (d < NOPE) {
                            uint8_t b = p.k_nope_fp8[(int64_t)g_row * NOPE + d];
                            float v = fp8_e4m3_to_f32(b) * p.k_nope_scale[g_row];
                            vbits = f32_to_bf16_bits(v);
                        } else {
                            vbits = p.k_rope_bf16[(int64_t)g_row * ROPE + (d - NOPE)];
                        }
                    }
                    // B is (K=16 rows, N=16 cols). Reuse BF16 canonical with rows=16
                    // by embedding into the first 16 rows of a 128-row tile; MMA_N=16.
                    sV[canon_idx_bf16_16x16(dd, kk)] = vbits;
                }
                __syncthreads();

                if (is_mma_warp && lane == 0) {
                    uint64_t dp = make_umma_desc_kmajor_none((uint32_t)__cvta_generic_to_shared(sPk), 128);
                    uint64_t dv = make_umma_desc_kmajor_none((uint32_t)__cvta_generic_to_shared(sV), 16);
                    umma_ss_f16(tb + n_sub * 16, dp, dv, idesc_pv, pv_kt > 0);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                }
                __syncthreads();
            }
        }
        asm volatile("fence.sc.gpu;" ::: "memory");
        __syncthreads();

        // Accumulate PV tile contribution after applying exp(tile_max-new_max).
        if (wid == 0) {
            float rescale_new = 0.0f;
            if (lane == 0) {
                // running_max is already the post-tile max. Recompute tile scale.
                float tile_max2 = -INFINITY;
                for (int c = 0; c < kv_len; c++) tile_max2 = fmaxf(tile_max2, sLogits[c] * p.scale);
                rescale_new = expf(tile_max2 - running_max);
            }
            for (int n = 0; n < HD / 8; n++) {
                float tmp[8];
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                    : "=f"(tmp[0]),"=f"(tmp[1]),"=f"(tmp[2]),"=f"(tmp[3]),
                      "=f"(tmp[4]),"=f"(tmp[5]),"=f"(tmp[6]),"=f"(tmp[7])
                    : "r"(tb + n * 8));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (lane == 0) {
                    #pragma unroll
                    for (int c = 0; c < 8; c++) sOacc[n * 8 + c] += tmp[c] * rescale_new;
                }
            }
        }
        __syncthreads();
    }

    // Attention sink: denominator-only logit.
    if (is_lane0 && p.sink_bias != nullptr) {
        float sb = p.sink_bias[batch_idx * p.H + head_idx];
        float new_max = fmaxf(running_max, sb);
        float rescale_old = (running_max > -INFINITY) ? expf(running_max - new_max) : 0.0f;
        for (int d = 0; d < HD; d++) sOacc[d] *= rescale_old;
        running_sum = running_sum * rescale_old + expf(sb - new_max);
        running_max = new_max;
    }
    __syncthreads();

    if (is_lane0) {
        float inv_sum = 1.0f / running_sum;
        for (int d = 0; d < HD; d++) sOepi[d] = f32_to_bf16_bits(sOacc[d] * inv_sum);
        if (lse) lse[0] = logf(running_sum) + running_max;
    }
    __syncthreads();
    for (int d = tid; d < HD; d += blockDim.x) out[d] = sOepi[d];
    __syncthreads();

    if (is_mma_warp) tmem_dealloc(tb, TMEM_COLS);
}

} // namespace dsv4::kernels::attention
