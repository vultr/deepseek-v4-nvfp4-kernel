// flush_write.cu — Quantize and scatter compressed entries into paged KV pool.
//
// Two kernel variants:
//   flush_write_csa_kernel: writes compressed entry + FP4 indexer key
//   flush_write_hca_kernel: writes compressed entry only (no indexer)
//
// Both do BF16 → FP8 (E4M3) quantization with per-token amax for the
// non-RoPE half, and write the RoPE half as-is BF16.
//
// One block per request. Each block handles writing ONE compressed entry
// per flush. At decode (B small, 1 entry/flush) this is 1-16 CTAs.
// At prefill (B up to 128), this is up to 128 CTAs — good occupancy.
//
// Blackwell SM100: 128 threads per block for the FP8 quantize loop
// covers head_dim=512 with 4 elements per thread. The FP4 indexer
// quantize uses 64 threads (indexer_head_dim=128, 2 elements/thread).

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

// ---- Warp-level reductions ----

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, fabsf(other));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---- Block-level amax (128 threads = 4 warps) ----

__device__ __forceinline__ float block_reduce_amax(float val, int n_warps) {
    float warp_amax = warp_reduce_max(val);
    __shared__ float smem[4];
    if (threadIdx.x % 32 == 0) {
        smem[threadIdx.x / 32] = warp_amax;
    }
    __syncthreads();
    float result = 0.0f;
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.0f;
        result = warp_reduce_max(v);
    }
    __syncthreads();
    return result;
}

// ---- NVFP4 quantization for indexer keys ----
// 16-element groups, one E4M3 scale per group.
// FP4 E2M1 has 6 possible values: 0, 2, 4, 6, 8, 10, 12, 14 (shifted).
// We use a simplified approach: group amax / 6.0 -> scale,
// quantize each element to nearest of {0,1,2,3,4,5,6} * scale.

__device__ __forceinline__ void quantize_fp4_group(
    const __nv_bfloat16* __restrict__ input,  // 16 elements
    uint8_t* __restrict__ output,              // 8 bytes (2 FP4 per byte)
    uint8_t* __restrict__ scale_out            // 1 FP8 E4M3 scale
) {
    // Compute group amax
    float amax = 0.0f;
    for (int i = 0; i < 16; i++) {
        amax = fmaxf(amax, fabsf(__bfloat162float(input[i])));
    }
    // FP4 E2M1 has max representable = 6.0 (before scaling)
    float scale = amax / 6.0f;
    if (scale < 1e-12f) scale = 1e-12f;
    float inv_scale = scale;

    // Write scale as FP8 E4M3
    __nv_fp8_e4m3 fp8_scale;
    fp8_scale = __nv_fp8_e4m3(scale);
    *scale_out = fp8_scale.__x;

    // Quantize 16 elements to FP4 E2M1, pack 2 per byte
    for (int i = 0; i < 8; i++) {
        float v0 = __bfloat162float(input[2 * i]) / inv_scale;
        float v1 = __bfloat162float(input[2 * i + 1]) / inv_scale;
        // Clamp to [0, 6] and round to nearest int
        int q0 = (int)roundf(fmaxf(0.0f, fminf(6.0f, v0)));
        int q1 = (int)roundf(fmaxf(0.0f, fminf(6.0f, v1)));
        // Pack: low nibble = element 0, high nibble = element 1
        output[i] = (uint8_t)((q1 << 4) | q0);
    }
}

// ===========================================================================
// CSA flush write kernel
// ===========================================================================

__global__ void flush_write_csa_kernel(
    // Inputs
    const __nv_bfloat16* __restrict__ entry,          // [B, head_dim] BF16
    const __nv_bfloat16* __restrict__ indexer_key,    // [B, indexer_head_dim] BF16
    const bool* __restrict__ valid_mask,               // [B]
    const int32_t* __restrict__ request_slots,          // [B]
    const int32_t* __restrict__ positions,               // [B]
    const int32_t* __restrict__ block_table,             // [B, max_logical_blocks]
    // Outputs — paged pool tensors, mutated in place
    uint8_t* __restrict__ entries_fp8,                   // [num_blocks, epb, fp8_dim]
    __nv_bfloat16* __restrict__ entries_rope,            // [num_blocks, epb, rope_dim]
    float* __restrict__ inv_scale,                       // [num_blocks, epb]
    uint8_t* __restrict__ indexer_keys_fp4,              // [num_blocks, epb, ihd/2]
    uint8_t* __restrict__ indexer_scale,                 // [num_blocks, epb, ihd/16]
    // Geometry
    int entries_per_block, int m, int rope_dim,
    int head_dim, int indexer_head_dim, int max_logical_blocks
) {
    int b = blockIdx.x;
    if (!valid_mask[b]) return;  // Early exit for no-op requests.

    // Resolve destination slot in the paged pool.
    int pos = positions[b];
    int entry_idx = pos / m;  // which compressed entry index
    int logical_block = entry_idx / entries_per_block;
    int slot_in_block = entry_idx % entries_per_block;
    int phys_block = block_table[b * max_logical_blocks + logical_block];

    int fp8_dim = head_dim - rope_dim;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;  // 128
    int n_warps = n_threads / 32;

    // ---- Step 1: Compute amax over non-RoPE half ----
    float local_amax = 0.0f;
    for (int i = tid; i < fp8_dim; i += n_threads) {
        float v = fabsf(__bfloat162float(entry[b * head_dim + i]));
        local_amax = fmaxf(local_amax, v);
    }
    float block_amax = block_reduce_amax(local_amax, n_warps);

    // ---- Step 2: Write inv_scale ----
    __shared__ float s_inv_scale;
    if (tid == 0) {
        float scale = (block_amax > 1e-12f) ? (block_amax / 448.0f) : 1e-12f;
        s_inv_scale = scale;
        inv_scale[phys_block * entries_per_block + slot_in_block] = scale;
    }
    __syncthreads();

    // ---- Step 3: Quantize and write FP8 half ----
    float inv_s = s_inv_scale;
    for (int i = tid; i < fp8_dim; i += n_threads) {
        float v = __bfloat162float(entry[b * head_dim + i]);
        float quantized = v / inv_s;
        quantized = fmaxf(-448.0f, fminf(448.0f, quantized));
        __nv_fp8_e4m3 fp8_val;
        fp8_val = __nv_fp8_e4m3(quantized);
        entries_fp8[(phys_block * entries_per_block + slot_in_block) * fp8_dim + i] = fp8_val.__x;
    }

    // ---- Step 4: Write BF16 RoPE half ----
    for (int i = tid; i < rope_dim; i += n_threads) {
        entries_rope[(phys_block * entries_per_block + slot_in_block) * rope_dim + i]
            = entry[b * head_dim + fp8_dim + i];
    }

    // ---- Step 5: FP4 quantize and write indexer key ----
    // 16 elements per group, one FP8 E4M3 scale per group.
    // Process groups in parallel across threads.
    int n_groups = indexer_head_dim / 16;
    int n_bytes = indexer_head_dim / 2;  // 2 FP4 per byte
    int n_scales = n_groups;

    for (int g = tid; g < n_groups; g += n_threads) {
        // Gather 16 BF16 values for this group
        __nv_bfloat16 group_in[16];
        for (int j = 0; j < 16; j++) {
            group_in[j] = indexer_key[b * indexer_head_dim + g * 16 + j];
        }
        uint8_t group_out[8];
        uint8_t group_scale;
        quantize_fp4_group(group_in, group_out, &group_scale);

        // Write 8 packed bytes
        int byte_offset = (phys_block * entries_per_block + slot_in_block) * n_bytes + g * 8;
        for (int j = 0; j < 8; j++) {
            indexer_keys_fp4[byte_offset + j] = group_out[j];
        }
        // Write scale
        int scale_offset = (phys_block * entries_per_block + slot_in_block) * n_scales + g;
        indexer_scale[scale_offset] = group_scale;
    }
}

// ===========================================================================
// HCA flush write kernel (no indexer)
// ===========================================================================

__global__ void flush_write_hca_kernel(
    const __nv_bfloat16* __restrict__ entry,
    const bool* __restrict__ valid_mask,
    const int32_t* __restrict__ request_slots,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ block_table,
    uint8_t* __restrict__ entries_fp8,
    __nv_bfloat16* __restrict__ entries_rope,
    float* __restrict__ inv_scale,
    int entries_per_block, int m, int rope_dim,
    int head_dim, int max_logical_blocks
) {
    int b = blockIdx.x;
    if (!valid_mask[b]) return;

    int pos = positions[b];
    int entry_idx = pos / m;
    int logical_block = entry_idx / entries_per_block;
    int slot_in_block = entry_idx % entries_per_block;
    int phys_block = block_table[b * max_logical_blocks + logical_block];

    int fp8_dim = head_dim - rope_dim;
    int tid = threadIdx.x;
    int n_threads = blockDim.x;
    int n_warps = n_threads / 32;

    // Amax reduction
    float local_amax = 0.0f;
    for (int i = tid; i < fp8_dim; i += n_threads) {
        float v = fabsf(__bfloat162float(entry[b * head_dim + i]));
        local_amax = fmaxf(local_amax, v);
    }
    float block_amax = block_reduce_amax(local_amax, n_warps);

    __shared__ float s_inv_scale;
    if (tid == 0) {
        float scale = (block_amax > 1e-12f) ? (block_amax / 448.0f) : 1e-12f;
        s_inv_scale = scale;
        inv_scale[phys_block * entries_per_block + slot_in_block] = scale;
    }
    __syncthreads();

    // FP8 quantize + write
    float inv_s = s_inv_scale;
    for (int i = tid; i < fp8_dim; i += n_threads) {
        float v = __bfloat162float(entry[b * head_dim + i]);
        float quantized = v / inv_s;
        quantized = fmaxf(-448.0f, fminf(448.0f, quantized));
        __nv_fp8_e4m3 fp8_val;
        fp8_val = __nv_fp8_e4m3(quantized);
        entries_fp8[(phys_block * entries_per_block + slot_in_block) * fp8_dim + i] = fp8_val.__x;
    }

    // BF16 RoPE half
    for (int i = tid; i < rope_dim; i += n_threads) {
        entries_rope[(phys_block * entries_per_block + slot_in_block) * rope_dim + i]
            = entry[b * head_dim + fp8_dim + i];
    }
}

// ===========================================================================
// State rotation kernels (in-place, single-kernel launches)
// ===========================================================================

// CSA: after flush, rotate a-stream -> b-stream, clear a-stream
__global__ void csa_rotate_state_kernel(
    const bool* __restrict__ valid_mask,  // [B]
    const int32_t* __restrict__ request_slots,  // [B]
    // State cache tensors — mutated in place
    __nv_bfloat16* __restrict__ tail_ka,  // [max_req, m, head_dim]
    __nv_bfloat16* __restrict__ tail_za,
    __nv_bfloat16* __restrict__ tail_kb,
    __nv_bfloat16* __restrict__ tail_zb,
    int32_t* __restrict__ tail_len,        // [max_req]
    int m, int head_dim, int max_requests
) {
    int b = blockIdx.x;
    if (!valid_mask[b]) return;

    int slot = request_slots[b];
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    // Rotate: kb <- ka, zb <- za (current a-stream becomes next b-stream)
    int total = m * head_dim;
    for (int i = tid; i < total; i += n_threads) {
        tail_kb[slot * total + i] = tail_ka[slot * total + i];
        tail_zb[slot * total + i] = tail_za[slot * total + i];
    }

    // Clear a-stream (zero out) and reset tail_len
    if (tid == 0) {
        tail_len[slot] = 0;
    }
    for (int i = tid; i < total; i += n_threads) {
        tail_ka[slot * total + i] = __float2bfloat16(0.0f);
        tail_za[slot * total + i] = __float2bfloat16(0.0f);
    }
}

// HCA: after flush, just clear a-stream and reset tail_len
__global__ void hca_reset_state_kernel(
    const bool* __restrict__ valid_mask,
    const int32_t* __restrict__ request_slots,
    __nv_bfloat16* __restrict__ tail_ka,
    __nv_bfloat16* __restrict__ tail_za,
    int32_t* __restrict__ tail_len,
    int m, int head_dim, int max_requests
) {
    int b = blockIdx.x;
    if (!valid_mask[b]) return;

    int slot = request_slots[b];
    int tid = threadIdx.x;
    int n_threads = blockDim.x;

    int total = m * head_dim;
    if (tid == 0) {
        tail_len[slot] = 0;
    }
    for (int i = tid; i < total; i += n_threads) {
        tail_ka[slot * total + i] = __float2bfloat16(0.0f);
        tail_za[slot * total + i] = __float2bfloat16(0.0f);
    }
}


// ===========================================================================
// PyTorch bindings
// ===========================================================================

void flush_write_csa_cuda(
    torch::Tensor entry,
    torch::Tensor indexer_key,
    torch::Tensor valid_mask,
    torch::Tensor request_slots,
    torch::Tensor positions,
    torch::Tensor block_table,
    torch::Tensor entries_fp8,
    torch::Tensor entries_rope,
    torch::Tensor inv_scale,
    torch::Tensor indexer_keys_fp4,
    torch::Tensor indexer_scale,
    int64_t entries_per_block, int64_t m, int64_t rope_dim,
    int64_t head_dim, int64_t indexer_head_dim
) {
    int B = entry.size(0);
    int max_logical_blocks = block_table.size(1);
    int threads = 128;
    flush_write_csa_kernel<<<B, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(entry.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(indexer_key.data_ptr<at::BFloat16>()),
        valid_mask.data_ptr<bool>(),
        request_slots.data_ptr<int32_t>(),
        positions.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        entries_fp8.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(entries_rope.data_ptr<at::BFloat16>()),
        inv_scale.data_ptr<float>(),
        indexer_keys_fp4.data_ptr<uint8_t>(),
        indexer_scale.data_ptr<uint8_t>(),
        (int)entries_per_block, (int)m, (int)rope_dim,
        (int)head_dim, (int)indexer_head_dim, max_logical_blocks
    );
    C10_CUDA_CHECK(cudaGetLastError());
}

void flush_write_hca_cuda(
    torch::Tensor entry,
    torch::Tensor valid_mask,
    torch::Tensor request_slots,
    torch::Tensor positions,
    torch::Tensor block_table,
    torch::Tensor entries_fp8,
    torch::Tensor entries_rope,
    torch::Tensor inv_scale,
    int64_t entries_per_block, int64_t m, int64_t rope_dim,
    int64_t head_dim
) {
    int B = entry.size(0);
    int max_logical_blocks = block_table.size(1);
    int threads = 128;
    flush_write_hca_kernel<<<B, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(entry.data_ptr<at::BFloat16>()),
        valid_mask.data_ptr<bool>(),
        request_slots.data_ptr<int32_t>(),
        positions.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        entries_fp8.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(entries_rope.data_ptr<at::BFloat16>()),
        inv_scale.data_ptr<float>(),
        (int)entries_per_block, (int)m, (int)rope_dim,
        (int)head_dim, max_logical_blocks
    );
    C10_CUDA_CHECK(cudaGetLastError());
}

void csa_rotate_state_cuda(
    torch::Tensor valid_mask,
    torch::Tensor request_slots,
    torch::Tensor tail_ka,
    torch::Tensor tail_za,
    torch::Tensor tail_kb,
    torch::Tensor tail_zb,
    torch::Tensor tail_len,
    int64_t m, int64_t head_dim
) {
    int B = valid_mask.size(0);
    int threads = 128;
    csa_rotate_state_kernel<<<B, threads>>>(
        valid_mask.data_ptr<bool>(),
        request_slots.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(tail_ka.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(tail_za.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(tail_kb.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(tail_zb.data_ptr<at::BFloat16>()),
        tail_len.data_ptr<int32_t>(),
        (int)m, (int)head_dim, 0  // max_requests unused in kernel
    );
    C10_CUDA_CHECK(cudaGetLastError());
}

void hca_reset_state_cuda(
    torch::Tensor valid_mask,
    torch::Tensor request_slots,
    torch::Tensor tail_ka,
    torch::Tensor tail_za,
    torch::Tensor tail_len,
    int64_t m, int64_t head_dim
) {
    int B = valid_mask.size(0);
    int threads = 128;
    hca_reset_state_kernel<<<B, threads>>>(
        valid_mask.data_ptr<bool>(),
        request_slots.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(tail_ka.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(tail_za.data_ptr<at::BFloat16>()),
        tail_len.data_ptr<int32_t>(),
        (int)m, (int)head_dim, 0
    );
    C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flush_write_csa", &flush_write_csa_cuda, "CSA flush write kernel");
    m.def("flush_write_hca", &flush_write_hca_cuda, "HCA flush write kernel");
    m.def("csa_rotate_state", &csa_rotate_state_cuda, "CSA state rotation kernel");
    m.def("hca_reset_state", &hca_reset_state_cuda, "HCA state reset kernel");
}
