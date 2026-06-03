#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "fmha_common.cuh"
#include "fmha_umma_desc.cuh"
#include "fmha_mixed_fp8_prefill.cuh"

using namespace dsv4::kernels::attention;

extern "C" {

int fmha_mixed_fp8_prefill_launch(
    const void* q_nope_fp8,
    const float* q_nope_scale,
    const void* q_rope_bf16,
    const void* k_nope_fp8,
    const float* k_nope_scale,
    const void* k_rope_bf16,
    void* o_ptr,
    void* lse_ptr,
    const float* sink_bias_ptr,
    int B, int H, int T, int N, int HD, int NOPE, int ROPE,
    int q_nope_t_stride, int q_nope_head_stride, int q_nope_batch_stride,
    int q_scale_t_stride, int q_scale_head_stride, int q_scale_batch_stride,
    int q_rope_t_stride, int q_rope_head_stride, int q_rope_batch_stride,
    int o_head_stride, int o_batch_stride, int o_t_stride,
    int lse_head_stride, int lse_batch_stride, int lse_t_stride,
    float scale
) {
    if (HD != 512 || NOPE != 448 || ROPE != 64) return -2;
    if (T < 1 || T > 128) return -3;

    FmhaMixedFp8PrefillParams p;
    p.q_nope_fp8 = (const uint8_t*)q_nope_fp8;
    p.q_nope_scale = q_nope_scale;
    p.q_rope_bf16 = (const bf16_t*)q_rope_bf16;
    p.k_nope_fp8 = (const uint8_t*)k_nope_fp8;
    p.k_nope_scale = k_nope_scale;
    p.k_rope_bf16 = (const bf16_t*)k_rope_bf16;
    p.o = (bf16_t*)o_ptr;
    p.lse = (float*)lse_ptr;
    p.sink_bias = sink_bias_ptr;
    p.B = B; p.H = H; p.T = T; p.N = N;
    p.HD = HD; p.NOPE = NOPE; p.ROPE = ROPE;
    p.q_nope_t_stride = q_nope_t_stride;
    p.q_nope_head_stride = q_nope_head_stride;
    p.q_nope_batch_stride = q_nope_batch_stride;
    p.q_scale_t_stride = q_scale_t_stride;
    p.q_scale_head_stride = q_scale_head_stride;
    p.q_scale_batch_stride = q_scale_batch_stride;
    p.q_rope_t_stride = q_rope_t_stride;
    p.q_rope_head_stride = q_rope_head_stride;
    p.q_rope_batch_stride = q_rope_batch_stride;
    p.o_head_stride = o_head_stride;
    p.o_batch_stride = o_batch_stride;
    p.o_t_stride = o_t_stride;
    p.lse_head_stride = lse_head_stride;
    p.lse_batch_stride = lse_batch_stride;
    p.lse_t_stride = lse_t_stride;
    p.scale = scale;

    // SMEM size for T_BATCH=32
    constexpr int T_BATCH = 32;
    constexpr int SK_TILE = 128;
    constexpr int TILE_F8 = 128 * 32;
    constexpr int TILE_F16 = 128 * 16;
    constexpr int V_SUB_SZ = 16 * 16;
    int smem = 0;
    smem += 4; smem = (smem + 127) & ~127;
    smem += TILE_F8; smem = (smem + 127) & ~127;           // sQ8
    smem += TILE_F8; smem = (smem + 127) & ~127;           // sK8
    smem += TILE_F16 * 2; smem = (smem + 127) & ~127;      // sQ16
    smem += TILE_F16 * 2; smem = (smem + 127) & ~127;      // sK16
    smem += TILE_F16 * 2; smem = (smem + 127) & ~127;      // sPk
    smem += V_SUB_SZ * 2; smem = (smem + 127) & ~127;      // sV
    smem += T_BATCH * SK_TILE * 4;                           // sLogits
    smem += T_BATCH * SK_TILE * 4;                           // sP
    smem += T_BATCH * 512 * 4;                               // sOacc
    smem += T_BATCH * 4;                                     // sRunningMax
    smem += T_BATCH * 4;                                     // sRunningSum
    smem += T_BATCH * 512 * 2;                               // sOepi
    smem = (smem + 127) & ~127;

    cudaFuncSetAttribute(
        fmha_mixed_fp8_prefill_kernel<512,448,64,128,32>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(1, H, B);
    dim3 block(192);
    fmha_mixed_fp8_prefill_kernel<512,448,64,128,32>
        <<<grid, block, smem>>>(p);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : (int)err;
}

} // extern C
