// gather_swa.cu — Gather SWA window entries into a dense BF16 tile.
//
// Reads from the state cache's SWA ring buffer (FP8 + BF16 split layout).
// One CTA per request. Each CTA iterates over the n_win positions in the
// ring buffer, dequantizes FP8 → BF16, concatenates the RoPE half, and
// writes to the dense output tensor.
//
// Output shape: [batch, n_win, head_dim] BF16 — consumed by FMHA.
// Positions with swa_pos == -1 (unused slots) are zero-filled.

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>


__global__ void gather_swa_kernel(
    // Inputs
    const uint8_t* __restrict__ swa_fp8,      // [max_req, n_win, fp8_dim]
    const __nv_bfloat16* __restrict__ swa_rope, // [max_req, n_win, rope_dim]
    const float* __restrict__ swa_inv,         // [max_req, n_win]
    const int32_t* __restrict__ swa_pos,       // [max_req, n_win]
    const int32_t* __restrict__ request_slots, // [batch] — state cache slot per request
    // Output
    __nv_bfloat16* __restrict__ output,        // [batch, n_win, head_dim] BF16
    // Geometry
    int batch, int n_win, int head_dim, int rope_dim, int max_requests
) {
    int fp8_dim = head_dim - rope_dim;
    int b = blockIdx.x;
    if (b >= batch) return;

    int slot = request_slots[b];

    for (int w = 0; w < n_win; w++) {
        int pos = swa_pos[slot * n_win + w];
        int out_row = b * n_win * head_dim + w * head_dim;

        if (pos < 0) {
            // Unused slot — zero fill
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                output[out_row + d] = __float2bfloat16(0.0f);
            }
            continue;
        }

        float s = swa_inv[slot * n_win + w];

        // Dequantize FP8 half
        for (int d = threadIdx.x; d < fp8_dim; d += blockDim.x) {
            uint8_t raw = swa_fp8[(slot * n_win + w) * fp8_dim + d];
            __nv_fp8_e4m3 fp8_val;
            fp8_val.__x = raw;
            float dequant = (float)fp8_val * s;
            output[out_row + d] = __float2bfloat16(dequant);
        }

        // Copy BF16 RoPE half
        for (int d = threadIdx.x; d < rope_dim; d += blockDim.x) {
            output[out_row + fp8_dim + d] = swa_rope[(slot * n_win + w) * rope_dim + d];
        }
    }
}


void gather_swa_cuda(
    torch::Tensor swa_fp8,
    torch::Tensor swa_rope,
    torch::Tensor swa_inv,
    torch::Tensor swa_pos,
    torch::Tensor request_slots,
    torch::Tensor output,
    int64_t head_dim, int64_t rope_dim
) {
    int batch = request_slots.size(0);
    int n_win = swa_fp8.size(1);
    int max_requests = swa_fp8.size(0);

    int threads = 128;
    gather_swa_kernel<<<batch, threads>>>(
        swa_fp8.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(swa_rope.data_ptr<at::BFloat16>()),
        swa_inv.data_ptr<float>(),
        swa_pos.data_ptr<int32_t>(),
        request_slots.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        batch, n_win, (int)head_dim, (int)rope_dim, max_requests
    );
    C10_CUDA_CHECK(cudaGetLastError());
}


// gather_all_compressed_kv_kernel — Gather ALL compressed entries for HCA
// (no top-k, dense attention over the short compressed sequence).
//
// One CTA per request. Iterates over all valid blocks in the block_table,
// dequantizes all entries, writes to a dense output.
// Output: [batch, total_entries, head_dim] BF16

__global__ void gather_all_compressed_kernel(
    const uint8_t* __restrict__ entries_fp8,
    const __nv_bfloat16* __restrict__ entries_rope,
    const float* __restrict__ inv_scale,
    const int32_t* __restrict__ block_table,       // [batch, max_logical_blocks]
    const int32_t* __restrict__ block_lens,        // [batch] — valid blocks per request
    __nv_bfloat16* __restrict__ output,            // [batch, total_entries, head_dim]
    int batch, int entries_per_block, int head_dim,
    int rope_dim, int max_logical_blocks
) {
    int fp8_dim = head_dim - rope_dim;
    int b = blockIdx.x;
    if (b >= batch) return;

    int n_blocks = block_lens[b];
    int out_idx = 0;

    for (int lb = 0; lb < n_blocks; lb++) {
        int phys_block = block_table[b * max_logical_blocks + lb];
        if (phys_block < 0) continue;

        for (int epb = 0; epb < entries_per_block; epb++) {
            int block_entry = phys_block * entries_per_block + epb;
            float s = inv_scale[block_entry];
            int out_row = (b * n_blocks * entries_per_block + out_idx) * head_dim;

            // Dequantize FP8 half
            for (int d = threadIdx.x; d < fp8_dim; d += blockDim.x) {
                uint8_t raw = entries_fp8[block_entry * fp8_dim + d];
                __nv_fp8_e4m3 fp8_val;
                fp8_val.__x = raw;
                float dequant = (float)fp8_val * s;
                output[out_row + d] = __float2bfloat16(dequant);
            }

            // Copy BF16 RoPE half
            for (int d = threadIdx.x; d < rope_dim; d += blockDim.x) {
                output[out_row + fp8_dim + d] = entries_rope[block_entry * rope_dim + d];
            }
            out_idx++;
        }
    }
}


void gather_all_compressed_cuda(
    torch::Tensor entries_fp8,
    torch::Tensor entries_rope,
    torch::Tensor inv_scale,
    torch::Tensor block_table,
    torch::Tensor block_lens,
    torch::Tensor output,
    int64_t entries_per_block, int64_t rope_dim
) {
    int batch = block_table.size(0);
    int head_dim = entries_fp8.size(2) + entries_rope.size(2);
    int max_logical_blocks = block_table.size(1);

    int threads = 128;
    gather_all_compressed_kernel<<<batch, threads>>>(
        entries_fp8.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(entries_rope.data_ptr<at::BFloat16>()),
        inv_scale.data_ptr<float>(),
        block_table.data_ptr<int32_t>(),
        block_lens.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        batch, (int)entries_per_block, (int)head_dim,
        (int)rope_dim, max_logical_blocks
    );
    C10_CUDA_CHECK(cudaGetLastError());
}


// gather_swa functions are registered in gather_kv.cu's PYBIND11_MODULE
// (both files are compiled together into a single cache_gather.so)
