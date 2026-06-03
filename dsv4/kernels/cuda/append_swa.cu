// append_swa.cu — write raw BF16 KV into the SWA ring buffer.
//
// One block per token. Threads cooperatively:
//   1. Compute amax over fp8-dim elements (warp reduce).
//   2. Quantize BF16 -> FP8 E4M3 with per-token scale.
//   3. Write FP8 entries + BF16 RoPE entries + inv_scale + position.
//   4. Atomic increment ring buffer head.
//
// Paper §2.3.4: BF16 for RoPE'd dims, FP8 for the rest.
// Per-token inverse scale stored for dequant in the attention kernel.

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

// Warp-level amax reduction
__device__ __forceinline__ float warp_reduce_amax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, fabsf(other));
    }
    return val;
}

// Warp-level sum for counting valid entries
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void append_swa_kernel(
    const __nv_bfloat16* __restrict__ raw_kv,   // [T, head_dim]
    const int32_t* __restrict__ request_slots,    // [T] -> slot in state pool
    const int32_t* __restrict__ positions,         // [T] -> absolute position
    // State cache pool — written in place.
    uint8_t* __restrict__ swa_fp8,                // [max_req, n_win, fp8_dim]
    __nv_bfloat16* __restrict__ swa_rope,         // [max_req, n_win, rope_dim]
    float* __restrict__ swa_inv,                  // [max_req, n_win]
    int32_t* __restrict__ swa_pos,                // [max_req, n_win]
    int32_t* __restrict__ swa_head,               // [max_req]
    int T, int n_win, int head_dim, int rope_dim
) {
    int t = blockIdx.x;
    if (t >= T) return;

    int lane = threadIdx.x;
    int warp_size = blockDim.x;  // expect 128 threads per block

    int slot = request_slots[t];
    int pos = positions[t];
    int fp8_dim = head_dim - rope_dim;

    // ---- Step 1: Compute amax over fp8_dim elements ----
    // Each thread processes strided elements of the fp8 half.
    float local_amax = 0.0f;
    for (int i = lane; i < fp8_dim; i += warp_size) {
        float val = __bfloat162float(raw_kv[t * head_dim + i]);
        local_amax = fmaxf(local_amax, fabsf(val));
    }

    // Warp-level amax reduction (works for warp_size <= 32).
    // For 128 threads, we need to reduce across 4 warps.
    float block_amax = 0.0f;
    // Intra-warp reduce
    float warp_amax = warp_reduce_amax(local_amax);
    // Lane 0 of each warp writes to shared memory
    __shared__ float smem_amax[4];  // max 4 warps for 128 threads
    if (lane % 32 == 0) {
        smem_amax[lane / 32] = warp_amax;
    }
    __syncthreads();
    if (lane < 32) {
        float v = (lane < (warp_size + 31) / 32) ? smem_amax[lane] : 0.0f;
        block_amax = warp_reduce_amax(v);
    }
    __syncthreads();
    // Broadcast block_amax to all threads
    __shared__ float s_inv_scale;
    if (lane == 0) {
        float scale = block_amax / 448.0f;  // FP8 E4M3 max = 448
        if (scale < 1e-12f) scale = 1e-12f;  // avoid div-by-zero
        s_inv_scale = scale;
    }
    __syncthreads();
    float inv_scale_val = s_inv_scale;

    // ---- Step 2: Atomic increment ring buffer head ----
    // Only one thread per block does the atomic
    __shared__ int slot_in_window;
    if (lane == 0) {
        slot_in_window = atomicAdd(&swa_head[slot], 1) % n_win;
    }
    __syncthreads();

    // ---- Step 3: Write FP8 entries ----
    for (int i = lane; i < fp8_dim; i += warp_size) {
        float val = __bfloat162float(raw_kv[t * head_dim + i]);
        float quantized = val / inv_scale_val;
        // Clamp to FP8 E4M3 range [-448, 448]
        quantized = fmaxf(-448.0f, fminf(448.0f, quantized));
        // Convert to FP8 E4M3
        __nv_fp8_e4m3 fp8_val;
        fp8_val.__x = __nv_fp8_e4m3(quantized).__x;
        swa_fp8[slot * n_win * fp8_dim + slot_in_window * fp8_dim + i] = fp8_val.__x;
    }

    // ---- Step 4: Write BF16 RoPE entries ----
    for (int i = lane; i < rope_dim; i += warp_size) {
        __nv_bfloat16 val = raw_kv[t * head_dim + fp8_dim + i];
        swa_rope[slot * n_win * rope_dim + slot_in_window * rope_dim + i] = val;
    }

    // ---- Step 5: Write metadata (single thread) ----
    if (lane == 0) {
        swa_inv[slot * n_win + slot_in_window] = inv_scale_val;
        swa_pos[slot * n_win + slot_in_window] = pos;
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
append_swa_cuda(
    torch::Tensor raw_kv,         // [T, head_dim] BF16
    torch::Tensor request_slots,  // [T] int32
    torch::Tensor positions,       // [T] int32
    torch::Tensor swa_fp8,        // [max_req, n_win, fp8_dim] uint8
    torch::Tensor swa_rope,       // [max_req, n_win, rope_dim] BF16
    torch::Tensor swa_inv,        // [max_req, n_win] FP32
    torch::Tensor swa_pos,        // [max_req, n_win] int32
    torch::Tensor swa_head,       // [max_req] int32
    int64_t rope_dim
) {
    int T = raw_kv.size(0);
    int head_dim = raw_kv.size(1);
    int n_win = swa_fp8.size(1);

    int threads = 128;
    int blocks = T;

    append_swa_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(raw_kv.data_ptr<at::BFloat16>()),
        request_slots.data_ptr<int32_t>(),
        positions.data_ptr<int32_t>(),
        swa_fp8.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(swa_rope.data_ptr<at::BFloat16>()),
        swa_inv.data_ptr<float>(),
        swa_pos.data_ptr<int32_t>(),
        swa_head.data_ptr<int32_t>(),
        T, n_win, head_dim, static_cast<int>(rope_dim)
    );

    C10_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(swa_fp8, swa_rope, swa_inv, swa_pos, swa_head);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("append_swa", &append_swa_cuda, "Append SWA kernel");
}
