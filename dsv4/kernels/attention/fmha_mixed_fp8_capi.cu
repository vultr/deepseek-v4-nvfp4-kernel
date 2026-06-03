#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "fmha_common.cuh"
#include "fmha_umma_desc.cuh"
#include "fmha_mixed_fp8_decode.cuh"

using namespace dsv4::kernels::attention;

extern "C" {

int fmha_mixed_fp8_decode_launch(
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
    int q_nope_head_stride, int q_nope_batch_stride,
    int q_scale_head_stride, int q_scale_batch_stride,
    int q_rope_head_stride, int q_rope_batch_stride,
    int o_head_stride, int o_batch_stride,
    int lse_head_stride, int lse_batch_stride,
    float scale
) {
    if (T != 1 || HD != 512 || NOPE != 448 || ROPE != 64) return -2;

    FmhaMixedFp8DecodeParams p;
    p.q_nope_fp8 = (const uint8_t*)q_nope_fp8;
    p.q_nope_scale = q_nope_scale;
    p.q_rope_bf16 = (const bf16_t*)q_rope_bf16;
    p.k_nope_fp8 = (const uint8_t*)k_nope_fp8;
    p.k_nope_scale = k_nope_scale;
    p.k_rope_bf16 = (const bf16_t*)k_rope_bf16;
    p.o = (bf16_t*)o_ptr;
    p.lse = (float*)lse_ptr;
    p.sink_bias = sink_bias_ptr;
    p.B = B; p.H = H; p.N = N; p.HD = HD; p.NOPE = NOPE; p.ROPE = ROPE;
    p.q_nope_head_stride = q_nope_head_stride;
    p.q_nope_batch_stride = q_nope_batch_stride;
    p.q_scale_head_stride = q_scale_head_stride;
    p.q_scale_batch_stride = q_scale_batch_stride;
    p.q_rope_head_stride = q_rope_head_stride;
    p.q_rope_batch_stride = q_rope_batch_stride;
    p.o_head_stride = o_head_stride;
    p.o_batch_stride = o_batch_stride;
    p.lse_head_stride = lse_head_stride;
    p.lse_batch_stride = lse_batch_stride;
    p.scale = scale;

    // Static shared memory size for fmha_mixed_fp8_decode_kernel<512,448,64>.
    // Keep this mirrored with the header layout and aligned up generously.
    int smem = 0;
    smem += 4; smem = (smem + 127) & ~127;
    smem += 128 * 32; smem = (smem + 127) & ~127;           // sQ8
    smem += 128 * 32; smem = (smem + 127) & ~127;           // sK8
    smem += 128 * 16 * 2; smem = (smem + 127) & ~127;       // sQ16
    smem += 128 * 16 * 2; smem = (smem + 127) & ~127;       // sK16
    smem += 128 * 16 * 2; smem = (smem + 127) & ~127;       // sPk
    smem += 16 * 16 * 2; smem = (smem + 127) & ~127;        // sV
    smem += 128 * 4;                                        // sLogits
    smem += 128 * 4;                                        // sP
    smem += 512 * 4;                                        // sOacc
    smem += 512 * 2;                                        // sOepi
    smem = (smem + 127) & ~127;

    cudaFuncSetAttribute(fmha_mixed_fp8_decode_kernel<512,448,64>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid(1, H, B);
    dim3 block(192);
    fmha_mixed_fp8_decode_kernel<512,448,64><<<grid, block, smem>>>(p);
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : (int)err;
}

} // extern C
