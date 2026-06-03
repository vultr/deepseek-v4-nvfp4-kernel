/**
 * DSV4 FMHA — Multi-tile kernel C API (TMA-based).
 *
 * Wraps fmha_6warp_tma_multirow_multitile_kernel with TMA descriptor
 * creation and launch. Uses create_tma_desc_2d_bf16 from fmha_tma.cuh
 * for correct descriptor format.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "fmha_common.cuh"
#include "fmha_umma_desc.cuh"
#include "fmha_tma.cuh"
#include "fmha_6warp_tma_multirow_multitile.cuh"

using namespace dsv4::kernels::attention;

extern "C" {

int fmha_multitile_decode_launch(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    void* o_ptr,
    void* lse_ptr,
    const float* sink_bias_ptr,
    int batch, int n_h, int T, int N_orig, int N_padded, int hd,
    int q_head_stride, int q_batch_stride,
    int k_head_stride, int k_batch_stride,
    int v_head_stride, int v_batch_stride,
    int o_head_stride, int o_batch_stride,
    int lse_head_stride, int lse_batch_stride,
    float scale
) {
    // N_orig:  logical KV length (used for softmax masking in kernel)
    // N_padded: physical KV length (used for TMA descriptor creation)
    // When N_orig < N_padded, the extra rows are zero-padded and
    // correctly excluded from softmax by the kernel's col < kv_len guard.
    size_t desc_count = n_h * batch;

    CUtensorMap* d_tma_k;
    CUtensorMap* d_tma_v;
    cudaMalloc(&d_tma_k, desc_count * sizeof(CUtensorMap));
    cudaMalloc(&d_tma_v, desc_count * sizeof(CUtensorMap));

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_h; h++) {
            const bf16_t* k_head = (const bf16_t*)k_ptr + h * k_head_stride + b * k_batch_stride;
            const bf16_t* v_head = (const bf16_t*)v_ptr + h * v_head_stride + b * v_batch_stride;
            int idx = b * n_h + h;

            // K: (N_padded, hd), TMA tile (128, 16) — use physical size for TMA
            CUtensorMap h_desc;
            if (!create_tma_desc_2d_bf16(&h_desc, k_head, N_padded, hd, 128, 16)) {
                cudaFree(d_tma_k); cudaFree(d_tma_v);
                return -1;
            }
            cudaMemcpy(d_tma_k + idx, &h_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

            // V: (hd, N_padded), TMA tile (16, 16) — use physical size for TMA
            if (!create_tma_desc_2d_bf16(&h_desc, v_head, hd, N_padded, 16, 16)) {
                cudaFree(d_tma_k); cudaFree(d_tma_v);
                return -1;
            }
            cudaMemcpy(d_tma_v + idx, &h_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
        }
    }

    FmhaTmaMultiRowMultiTileParams params;
    params.q = (const bf16_t*)q_ptr;
    params.tma_k = d_tma_k;
    params.tma_v = d_tma_v;
    params.o = (bf16_t*)o_ptr;
    params.lse = (float*)lse_ptr;
    params.s_k = N_orig;  // Logical KV length — kernel uses this for softmax masking
    params.T = T;
    params.n_h = n_h;
    params.scale = scale;
    params.q_head_stride = q_head_stride;
    params.q_batch_stride = q_batch_stride;
    params.o_head_stride = o_head_stride;
    params.o_batch_stride = o_batch_stride;
    params.lse_head_stride = lse_head_stride;
    params.lse_batch_stride = lse_batch_stride;
    params.sink_bias = sink_bias_ptr;  // per-head FP32 sink logit, NULL if unused

    // SMEM size (match kernel layout)
    constexpr int HD_CHUNK = 256;
    constexpr int TILE_SZ = 128 * MMA_K_BF16;
    constexpr int V_SUB_SZ = 16 * MMA_K_BF16;
    int hc = (hd <= 256) ? hd : HD_CHUNK;
    size_t off = 4;
    off = (off+127)&~(size_t)127;  // sTmemBase
    off += 16;                      // sMbar
    off = (off+127)&~(size_t)127;
    off += TILE_SZ * 2;            // sTmaBuf
    off = (off+127)&~(size_t)127;
    off += TILE_SZ * 2;            // sQ0
    off = (off+127)&~(size_t)127;
    off += TILE_SZ * 2;            // sK0
    off = (off+127)&~(size_t)127;
    off += TILE_SZ * 2;            // sPk
    off = (off+127)&~(size_t)127;
    off += V_SUB_SZ * 2;          // sV
    off = (off+127)&~(size_t)127;
    off += 128 * hc * 4;          // sOacc
    off += 128 * 4;               // sRunningMax
    off += 128 * 4;               // sRunningSum
    off += 128 * 4;               // sTileRowMax
    off += 128 * 4;               // sTileRowSum
    // P6: sO_epi_rowmajor + sMbarStore
    off = (off+127)&~(size_t)127;
    off += 128 * hc * 2;          // sO_epi_rowmajor (MAX_ROWS * HD_CHUNK BF16)
    off += 256;                    // slack
    int smem = (int)((off + 127) & ~(size_t)127);

    dim3 grid(1, n_h, batch);
    dim3 block(NTHREADS);

    if (smem > 48 * 1024) {
        if (hd == 64) cudaFuncSetAttribute(fmha_6warp_tma_multirow_multitile_kernel<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        else if (hd == 128) cudaFuncSetAttribute(fmha_6warp_tma_multirow_multitile_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        else if (hd == 256) cudaFuncSetAttribute(fmha_6warp_tma_multirow_multitile_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        else if (hd == 512) cudaFuncSetAttribute(fmha_6warp_tma_multirow_multitile_kernel<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }

    cudaError_t err;
    if (hd == 64) fmha_6warp_tma_multirow_multitile_kernel<64><<<grid, block, smem>>>(params);
    else if (hd == 128) fmha_6warp_tma_multirow_multitile_kernel<128><<<grid, block, smem>>>(params);
    else if (hd == 256) fmha_6warp_tma_multirow_multitile_kernel<256><<<grid, block, smem>>>(params);
    else if (hd == 512) fmha_6warp_tma_multirow_multitile_kernel<512><<<grid, block, smem>>>(params);
    else { cudaFree(d_tma_k); cudaFree(d_tma_v); return -1; }

    err = cudaGetLastError();
    cudaFree(d_tma_k);
    cudaFree(d_tma_v);
    if (err != cudaSuccess) return (int)err;
    return 0;
}

} // extern "C"
