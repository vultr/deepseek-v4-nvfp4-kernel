/**
 * DSV4 FMHA — 6-warp TMA, multi-row softmax, multi-tile KV, in-kernel rescale.
 *
 * HD tiling: processes HD in chunks of HD_CHUNK (256).
 * For HD≤256, single chunk (same as before). For HD=512, two chunks.
 *
 * Design:
 *   For each hd_chunk (outer):
 *     Zero sOacc_chunk[T][HD_CHUNK]
 *     For each kv_tile (inner):
 *       1. QK GEMM → softmax → P in registers
 *       2. PV GEMM (only n_sub in hd_chunk) → O in TMEM
 *       3. Read O from TMEM → rescale → accumulate in sOacc_chunk
 *       4. Update running_max, running_sum
 *     Normalize sOacc_chunk → BF16 → GMEM
 *
 * Running_max/running_sum are recomputed per hd_chunk (identical result,
 * negligible cost — just a few float ops per KV tile).
 *
 * SMEM budget (HD=512, HD_CHUNK=256): ~128KB sOacc + ~26KB pipeline = ~154KB
 */

#pragma once

#include "fmha_common.cuh"
#include "fmha_umma_desc.cuh"
#include "fmha_tma.cuh"

namespace dsv4::kernels::attention {

struct FmhaTmaMultiRowMultiTileParams {
    const bf16_t* __restrict__ q;
    CUtensorMap* __restrict__ tma_k;
    CUtensorMap* __restrict__ tma_v;
    bf16_t* __restrict__ o;
    float* __restrict__ lse;
    const float* __restrict__ sink_bias;  // per-head FP32 sink logit (n_h,), NULL if unused
    int s_k, T, n_h;
    float scale;
    int q_head_stride, q_batch_stride;
    int o_head_stride, o_batch_stride;
    int lse_head_stride, lse_batch_stride;
};

template<int HD, int SK_TILE = 128, int HD_CHUNK = (HD <= 256 ? HD : 256)>
__global__ void __launch_bounds__(192)
fmha_6warp_tma_multirow_multitile_kernel(FmhaTmaMultiRowMultiTileParams params) {
    static constexpr int N_HD_CHUNKS = HD / HD_CHUNK;
    static constexpr int NKT_QK = HD / MMA_K_BF16;
    static constexpr int NKT_PV = SK_TILE / MMA_K_BF16;
    static constexpr int N_NSUB_TOTAL = HD / 16;         // total PV sub-tiles for full HD
    static constexpr int N_NSUB_CHUNK = HD_CHUNK / 16;   // PV sub-tiles per hd_chunk
    static constexpr int MAX_ROWS = 128;
    static constexpr int TILE_SZ = 128 * MMA_K_BF16;
    static constexpr int V_SUB_SZ = 16 * MMA_K_BF16;
    static constexpr int TMEM_N = (HD <= 128) ? 128 : (HD <= 256) ? 256 : 512;
    static constexpr int CORES_MN = 128 / 8;
    static constexpr int NUM_READS = SK_TILE / 8;
    static constexpr int TMA_TILE_BYTES = TILE_SZ * sizeof(bf16_t);
    // TMEM reads: N_NSUB_CHUNK sub-tiles × 2 (each sub-tile = 16 cols, 2 reads of 8 cols)
    static constexpr int TMEM_READS_PER_CHUNK = N_NSUB_CHUNK * 2;

    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    const bool is_softmax_warp = (wid < 4);
    const bool is_mma_warp = (wid == 4);
    const bool is_load_warp = (wid == 5);
    const int T = params.T;
    const int s_k = params.s_k;
    const int n_kv_tiles = (s_k + SK_TILE - 1) / SK_TILE;
    const float scale = params.scale;

    const bf16_t* __restrict__ q_head = params.q + head_idx * params.q_head_stride + batch_idx * params.q_batch_stride;
    bf16_t* __restrict__ o_head = params.o + head_idx * params.o_head_stride + batch_idx * params.o_batch_stride;
    float* __restrict__ lse_head = params.lse ? params.lse + head_idx * params.lse_head_stride + batch_idx * params.lse_batch_stride : nullptr;
    CUtensorMap* __restrict__ my_tma_k = params.tma_k + batch_idx * params.n_h + head_idx;
    CUtensorMap* __restrict__ my_tma_v = params.tma_v + batch_idx * params.n_h + head_idx;

    // ================================================================
    // SMEM allocation (sized for HD_CHUNK, not full HD)
    // ================================================================
    extern __shared__ __align__(128) char sbuf[];
    size_t off = 0;
    uint32_t* sTmemBase = (uint32_t*)(sbuf + off); off += 4;
    off = (off + 127) & ~(size_t)127;
    uint64_t* sMbar = (uint64_t*)(sbuf + off); off += 16;
    off = (off + 127) & ~(size_t)127;
    bf16_t* sTmaBuf = (bf16_t*)(sbuf + off); off += TILE_SZ * sizeof(bf16_t);
    off = (off + 127) & ~(size_t)127;
    bf16_t* sQ0 = (bf16_t*)(sbuf + off); off += TILE_SZ * sizeof(bf16_t);
    off = (off + 127) & ~(size_t)127;
    bf16_t* sK0 = (bf16_t*)(sbuf + off); off += TILE_SZ * sizeof(bf16_t);
    off = (off + 127) & ~(size_t)127;
    bf16_t* sPk = (bf16_t*)(sbuf + off); off += TILE_SZ * sizeof(bf16_t);
    off = (off + 127) & ~(size_t)127;
    bf16_t* sV = (bf16_t*)(sbuf + off); off += V_SUB_SZ * sizeof(bf16_t);
    off = (off + 127) & ~(size_t)127;
    // SMEM accumulator (chunk-sized)
    float* sOacc = (float*)(sbuf + off); off += MAX_ROWS * HD_CHUNK * sizeof(float);
    float* sRunningMax = (float*)(sbuf + off); off += MAX_ROWS * sizeof(float);
    float* sRunningSum = (float*)(sbuf + off); off += MAX_ROWS * sizeof(float);
    float* sTileRowMax = (float*)(sbuf + off); off += MAX_ROWS * sizeof(float);
    float* sTileRowSum = (float*)(sbuf + off); off += MAX_ROWS * sizeof(float);
    // P6: Row-major O epilogue buffer + TMA store mbarrier
    off = (off + 127) & ~(size_t)127;
    bf16_t* sO_epi_rowmajor = (bf16_t*)(sbuf + off); off += MAX_ROWS * HD_CHUNK * sizeof(bf16_t);

    // Init TMEM + mbarrier (once, shared across hd_chunks)
    if (is_mma_warp) tmem_alloc(__cvta_generic_to_shared(sTmemBase), TMEM_N);
    if (tid == 0) {
        tma_mbarrier_init((uint32_t)__cvta_generic_to_shared(sMbar), 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    uint32_t tb = *sTmemBase;
    const uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(sMbar);

    const bool my_warp_active = (T <= 32) ? (wid == 0) : is_softmax_warp;
    const int my_row = my_warp_active ? (wid * 32 + lane) : 0;
    const bool my_row_active = my_warp_active && (my_row < T);

    // LSE written only once (first hd_chunk)
    bool lse_written = false;

    // ================================================================
    // HD chunk loop
    // ================================================================
    for (int hd_chunk = 0; hd_chunk < N_HD_CHUNKS; hd_chunk++) {
        const int hd_chunk_start = hd_chunk * HD_CHUNK;
        const int n_sub_start = hd_chunk_start / 16;
        const int n_sub_end = n_sub_start + N_NSUB_CHUNK;
        int phase = 0;  // reset mbarrier phase per hd_chunk

        // Zero accumulator
        for (int i = tid; i < MAX_ROWS * HD_CHUNK; i += 192) sOacc[i] = 0.0f;
        for (int i = tid; i < MAX_ROWS; i += 192) {
            sRunningMax[i] = -INFINITY;
            sRunningSum[i] = 0.0f;
        }
        __syncthreads();

        // ---- KV tile loop ----
        for (int kv_tile = 0; kv_tile < n_kv_tiles; kv_tile++) {
            int kv_start = kv_tile * SK_TILE;
            int kv_len = min(SK_TILE, s_k - kv_start);

            // ---- QK GEMM ----
            for (int kt = 0; kt < NKT_QK; kt++) {
                if (is_load_warp) {
                    for (int i = lane; i < TILE_SZ; i += 32) sQ0[i] = 0;
                    for (int r = 0; r < T; r++) {
                        for (int d = lane; d < MMA_K_BF16; d += 32) {
                            int full_d = kt * MMA_K_BF16 + d;
                            if (full_d < HD) {
                                int ck = d/8, lc = d%8, cm = r/8, lr = r%8;
                                sQ0[ck*CORES_MN*64 + cm*64 + lr*8 + lc] = q_head[r * HD + full_d];
                            }
                        }
                    }
                }

                if (is_load_warp && lane == 0) {
                    tma_load_2d((uint32_t)__cvta_generic_to_shared(sTmaBuf), (uint64_t)my_tma_k,
                                mbar_addr, kt * MMA_K_BF16, kv_start);
                    tma_mbarrier_arrive_expect_tx(mbar_addr, TMA_TILE_BYTES);
                }
                tma_mbarrier_wait(mbar_addr, phase); phase ^= 1;
                __syncthreads();

                for (int i = tid; i < TILE_SZ; i += 192) sK0[i] = 0;
                for (int i = tid; i < kv_len * MMA_K_BF16; i += 192) {
                    int r = i / MMA_K_BF16, c = i % MMA_K_BF16;
                    int ck = c/8, lc = c%8, tmn = r/8, lr = r%8;
                    sK0[ck*CORES_MN*64 + tmn*64 + lr*8 + lc] = sTmaBuf[i];
                }
                __syncthreads();

                if (is_mma_warp) {
                    uint32_t idesc = make_idesc(128, 128);
                    uint64_t dq = make_umma_desc_kmajor_none(__cvta_generic_to_shared(sQ0), 128);
                    uint64_t dk = make_umma_desc_kmajor_none(__cvta_generic_to_shared(sK0), 128);
                    if (tid == 128) umma_ss_f16(tb, dq, dk, idesc, kt > 0);
                    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                }
                __syncthreads();
            }
            asm volatile("fence.sc.gpu;" ::: "memory");
            __syncthreads();

            // ---- Softmax (2-pass, un-normalized P) ----
            float my_row_max = -INFINITY;
            if (my_warp_active) {
                for (int n = 0; n < NUM_READS; n++) {
                    float tmp[8];
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                        : "=f"(tmp[0]),"=f"(tmp[1]),"=f"(tmp[2]),"=f"(tmp[3]),
                          "=f"(tmp[4]),"=f"(tmp[5]),"=f"(tmp[6]),"=f"(tmp[7])
                        : "r"(tb + n * 8));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (my_row_active) {
                        for (int c = 0; c < 8; c++) {
                            int col = n * 8 + c;
                            if (col < kv_len) my_row_max = fmaxf(my_row_max, tmp[c] * scale);
                        }
                    }
                }
            }
            if (my_row_active) sTileRowMax[my_row] = my_row_max;
            __syncthreads();

            float my_p_vals[SK_TILE] = {};  // Zero-init: padded positions contribute 0 to PV
            float my_row_sum = 0.0f;
            if (my_warp_active) {
                float rm = my_row_active ? sTileRowMax[my_row] : 0.0f;
                for (int n = 0; n < NUM_READS; n++) {
                    float tmp[8];
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                        : "=f"(tmp[0]),"=f"(tmp[1]),"=f"(tmp[2]),"=f"(tmp[3]),
                          "=f"(tmp[4]),"=f"(tmp[5]),"=f"(tmp[6]),"=f"(tmp[7])
                        : "r"(tb + n * 8));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (my_row_active) {
                        for (int c = 0; c < 8; c++) {
                            int col = n * 8 + c;
                            if (col < kv_len) {
                                float p = expf(tmp[c] * scale - rm);
                                my_p_vals[col] = p;
                                my_row_sum += p;
                            }
                        }
                    }
                }
            }
            if (my_row_active) sTileRowSum[my_row] = my_row_sum;
            __syncthreads();

            // ---- PV GEMM (only hd_chunk's n_sub range) ----
            for (int n_sub = n_sub_start; n_sub < n_sub_end; n_sub++) {
                int d_base = n_sub * 16;
                for (int pv_kt = 0; pv_kt < NKT_PV; pv_kt++) {
                    const int col_start = pv_kt * MMA_K_BF16;

                    if (is_load_warp) {
                        for (int i = lane; i < TILE_SZ; i += 32) sPk[i] = 0;
                    }
                    __syncthreads();

                    if (my_row_active) {
                        for (int c = 0; c < MMA_K_BF16; c++) {
                            int gc = col_start + c;
                            int ck = c/8, lc = c%8;
                            int core_mn = my_row/8, local_r = my_row%8;
                            sPk[ck*CORES_MN*64 + core_mn*64 + local_r*8 + lc] = f32_to_bf16(my_p_vals[gc]);
                        }
                    }
                    __syncthreads();

                    if (is_load_warp && lane == 0) {
                        tma_load_2d((uint32_t)__cvta_generic_to_shared(sTmaBuf), (uint64_t)my_tma_v,
                                    mbar_addr, kv_start + col_start, d_base);
                        tma_mbarrier_arrive_expect_tx(mbar_addr, V_SUB_SZ * sizeof(bf16_t));
                    }
                    tma_mbarrier_wait(mbar_addr, phase); phase ^= 1;
                    __syncthreads();

                    for (int i = tid; i < V_SUB_SZ; i += 192) sV[i] = 0;
                    for (int i = tid; i < 16 * MMA_K_BF16; i += 192) {
                        int dd = i / MMA_K_BF16, lr = i % MMA_K_BF16;
                        int g_mn = dd/8, g_k = lr/8, llr = dd%8, lc = lr%8;
                        sV[g_k*2*64 + g_mn*64 + llr*8 + lc] = sTmaBuf[i];
                    }
                    __syncthreads();

                    if (is_mma_warp) {
                        // PV GEMM writes to TMEM at n_sub*16, but we offset by hd_chunk_start
                        // so the TMEM columns for this chunk start at 0 (relative)
                        uint32_t idesc_pv = make_idesc(128, 16);
                        uint64_t dp = make_umma_desc_kmajor_none(__cvta_generic_to_shared(sPk), 128);
                        uint64_t dv = make_umma_desc_kmajor_none(__cvta_generic_to_shared(sV), 16);
                        // TMEM column offset: (n_sub - n_sub_start) * 16
                        // PV output at NATIVE TMEM column (not remapped)
                        if (tid == 128) umma_ss_f16(tb + n_sub * 16, dp, dv, idesc_pv, pv_kt > 0);
                        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
                    }
                    __syncthreads();
                }
            }
            asm volatile("fence.sc.gpu;" ::: "memory");
            __syncthreads();

            // ---- Read O from TMEM, rescale sOacc, accumulate ----
            if (my_warp_active) {
                float tile_max = my_row_active ? sTileRowMax[my_row] : 0.0f;
                float tile_sum = my_row_active ? sTileRowSum[my_row] : 0.0f;

                // Read O_chunk from TMEM at NATIVE column offsets (warp-collective)
                float o_tile_buf[HD_CHUNK];
                for (int n = 0; n < TMEM_READS_PER_CHUNK; n++) {
                    float tmp[8];
                    // Read from hd_chunk's native TMEM columns
                    int tmem_read_addr = tb + (n_sub_start * 16) + n * 8;
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
                        : "=f"(tmp[0]),"=f"(tmp[1]),"=f"(tmp[2]),"=f"(tmp[3]),
                          "=f"(tmp[4]),"=f"(tmp[5]),"=f"(tmp[6]),"=f"(tmp[7])
                        : "r"(tmem_read_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (my_row_active) {
                        for (int c = 0; c < 8; c++) {
                            int d = n * 8 + c;
                            if (d < HD_CHUNK) o_tile_buf[d] = tmp[c];
                        }
                    }
                }

                // Online softmax rescale + accumulate
                if (my_row_active) {
                    float old_max = sRunningMax[my_row];
                    float new_max = fmaxf(old_max, tile_max);
                    float rescale_old = expf(old_max - new_max);
                    float rescale_new = expf(tile_max - new_max);

                    for (int d = 0; d < HD_CHUNK; d++) {
                        sOacc[my_row * HD_CHUNK + d] = sOacc[my_row * HD_CHUNK + d] * rescale_old + o_tile_buf[d] * rescale_new;
                    }

                    sRunningSum[my_row] = sRunningSum[my_row] * rescale_old + tile_sum * rescale_new;
                    sRunningMax[my_row] = new_max;
                }
            }
            __syncthreads();
        } // kv_tile loop

        // ---- Sink bias correction (D5c: single softmax over [S_comp, S_swa + sink]) ----
        // The attention sink is a per-head logit bias. It adds one extra
        // "position" to the softmax that contributes to the denominator
        // but NOT the numerator (no corresponding V row). This is the
        // key insight: sink merge = single softmax, not two-branch merge.
        //
        // Math: after all KV tiles, we have (running_max, running_sum, O_unnorm).
        // Sink adds: sink_weight = exp(sink_bias * scale - new_max)
        //   new_max = max(running_max, sink_bias * scale)
        //   rescale O_unnorm and running_sum by exp(old_max - new_max)
        //   running_sum += sink_weight
        // The sink does NOT produce a PV contribution — O_unnorm unchanged.
        if (params.sink_bias != nullptr && my_warp_active) {
            // Load per-head sink bias (same for all rows in this head)
            float sb = params.sink_bias[head_idx + batch_idx * params.n_h];
            if (my_row_active) {
                // sink_bias is already in the scaled domain (added to QK*scale in softmax)
                // Do NOT multiply by scale again — the kernel's softmax already applies
                // scale to QK values, and running_max is in the scaled domain.
                float sink_logit = sb;
                float old_max = sRunningMax[my_row];
                float new_max = fmaxf(old_max, sink_logit);
                float rescale_old = (old_max > -INFINITY) ? expf(old_max - new_max) : 0.0f;
                float sink_weight = expf(sink_logit - new_max);

                // Rescale existing accumulator and running sum
                for (int d = 0; d < HD_CHUNK; d++) {
                    sOacc[my_row * HD_CHUNK + d] *= rescale_old;
                }
                sRunningSum[my_row] = sRunningSum[my_row] * rescale_old + sink_weight;
                sRunningMax[my_row] = new_max;
            }
        }
        __syncthreads();

        // ---- Write chunk to SMEM row-major, then TMA store to GMEM ----
        // P6: One-way epilogue pattern — normalize in registers,
        // write to SMEM row-major, then TMA store to GMEM.
        // This enables multi-CTA grids and FP4 output fusion.
        //
        // Each active row writes its normalized output to sO_epi_rowmajor
        // in row-major layout, then writes directly to GMEM.
        if (my_row_active) {
            float inv_rs = 1.0f / sRunningSum[my_row];
            for (int d = 0; d < HD_CHUNK; d++) {
                sO_epi_rowmajor[my_row * HD_CHUNK + d] = f32_to_bf16(sOacc[my_row * HD_CHUNK + d] * inv_rs);
            }
        }
        if (hd_chunk == 0) lse_written = true;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();

        // Write SMEM → GMEM (direct)
        if (my_row_active) {
            for (int d = 0; d < HD_CHUNK; d++) {
                o_head[my_row * HD + hd_chunk_start + d] = sO_epi_rowmajor[my_row * HD_CHUNK + d];
            }
            // LSE: same for all chunks, write once
            if (lse_head && !lse_written) {
                lse_head[my_row] = logf(sRunningSum[my_row]) + sRunningMax[my_row];
            }
        }
        __syncthreads();
    } // hd_chunk loop

    __syncthreads();
    if (is_mma_warp) tmem_dealloc(tb, TMEM_N);
}

} // namespace
