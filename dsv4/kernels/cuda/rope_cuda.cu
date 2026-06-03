/*
 * rope_cuda.cu
 *
 * Fused forward/inverse partial RoPE kernel for DeepSeek V4.
 * GPT-J style (interleaved) RoPE on last rope_dim=64 dims of each head.
 *
 * Replaces 5-6 PyTorch kernel launches per RoPE call with 1 CUDA kernel.
 * Total savings: ~1000 launches/token → 183 launches/token (~0.8ms at 2µs/launch).
 *
 * C API for ctypes loading (no ATen/pybind11).
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

__global__ void apply_rope_kernel(
    __nv_bfloat16* __restrict__ x,           // (T, n_h, hd) — modified in-place
    const int64_t* __restrict__ positions,    // (T,) — token positions
    const float* __restrict__ cos_cache,      // (max_pos, rope_dim//2)
    const float* __restrict__ sin_cache,      // (max_pos, rope_dim//2)
    const int T,
    const int n_h,
    const int hd,
    const int nope_dim,                       // hd - rope_dim = 448
    const int rope_dim,                       // 64
    const bool inverse                        // true = inverse RoPE
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_rope = rope_dim / 2;
    const int total_pairs = T * n_h * half_rope;
    
    if (idx >= total_pairs) return;
    
    const int pair_idx = idx % half_rope;
    const int head_idx = (idx / half_rope) % n_h;
    const int token_idx = idx / (half_rope * n_h);
    
    // Get position and cos/sin values
    int64_t pos = positions[token_idx];
    float c = cos_cache[pos * half_rope + pair_idx];
    float s = sin_cache[pos * half_rope + pair_idx];
    
    // Compute pointer to the two elements of the pair
    const int even_offset = token_idx * n_h * hd + head_idx * hd + nope_dim + 2 * pair_idx;
    const int odd_offset = even_offset + 1;
    
    // Load BF16 values, convert to FP32
    float x_even = __bfloat162float(x[even_offset]);
    float x_odd = __bfloat162float(x[odd_offset]);
    
    // Apply rotation
    float rot_even, rot_odd;
    if (inverse) {
        rot_even = x_even * c + x_odd * s;
        rot_odd = -x_even * s + x_odd * c;
    } else {
        rot_even = x_even * c - x_odd * s;
        rot_odd = x_even * s + x_odd * c;
    }
    
    // Store back as BF16
    x[even_offset] = __float2bfloat16(rot_even);
    x[odd_offset] = __float2bfloat16(rot_odd);
}

// C API for ctypes
extern "C" {

void apply_rope_launch(
    void* x_ptr,
    const int64_t* positions_ptr,
    const float* cos_ptr,
    const float* sin_ptr,
    int T, int n_h, int hd,
    int nope_dim, int rope_dim,
    bool inverse,
    int grid_size, int block_size,
    void* stream_ptr
) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    apply_rope_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x_ptr),
        positions_ptr,
        cos_ptr,
        sin_ptr,
        T, n_h, hd, nope_dim, rope_dim, inverse
    );
}

}  // extern "C"
