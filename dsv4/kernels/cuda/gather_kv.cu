// gather_kv.cu — Gather selected compressed entries into a dense BF16 tile.
//
// One CTA per (query token, key_group). Each CTA handles a contiguous
// group of top-k entries for one query token. Reads from the FP8/BF16
// split paged pool via block_table resolution, dequantizes FP8 → BF16,
// concatenates the RoPE half, writes to the dense output.
//
// Pure bandwidth-bound kernel — no MMA, just load-multiply-store.
// The output [T, top_k, head_dim] BF16 tile is what the FMHA kernel
// consumes. Sparsity is hidden in the gather; FMHA sees dense tiles.

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>


__global__ void gather_kv_kernel(
    // Inputs
    const uint8_t* __restrict__ entries_fp8,        // [num_blocks, epb, fp8_dim]
    const __nv_bfloat16* __restrict__ entries_rope, // [num_blocks, epb, rope_dim]
    const float* __restrict__ inv_scale,            // [num_blocks, epb]
    const int32_t* __restrict__ topk_indices,       // [T, top_k] — compressed entry indices
    const int32_t* __restrict__ block_table,         // [T, max_logical_blocks]
    // Output
    __nv_bfloat16* __restrict__ output,              // [T, top_k, head_dim] BF16
    // Geometry
    int T, int top_k, int entries_per_block,
    int head_dim, int rope_dim, int max_logical_blocks
) {
    int fp8_dim = head_dim - rope_dim;

    // Each CTA handles one (query_token, topk_entry) pair.
    int flat_idx = blockIdx.x;
    int t = flat_idx / top_k;
    int k = flat_idx % top_k;
    if (t >= T) return;

    // Resolve which compressed entry to gather.
    int comp_idx = topk_indices[t * top_k + k];
    if (comp_idx < 0) {
        // Invalid entry — zero fill.
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            output[t * top_k * head_dim + k * head_dim + d] = __float2bfloat16(0.0f);
        }
        return;
    }

    int logical_block = comp_idx / entries_per_block;
    int slot_in_block = comp_idx % entries_per_block;
    int phys_block = block_table[t * max_logical_blocks + logical_block];

    int block_entry = phys_block * entries_per_block + slot_in_block;

    // Dequantize and write FP8 half.
    float s = inv_scale[block_entry];
    for (int d = threadIdx.x; d < fp8_dim; d += blockDim.x) {
        uint8_t raw = entries_fp8[block_entry * fp8_dim + d];
        __nv_fp8_e4m3 fp8_val;
        fp8_val.__x = raw;
        float dequant = (float)fp8_val * s;
        output[t * top_k * head_dim + k * head_dim + d] = __float2bfloat16(dequant);
    }

    // Copy BF16 RoPE half.
    for (int d = threadIdx.x; d < rope_dim; d += blockDim.x) {
        output[t * top_k * head_dim + k * head_dim + fp8_dim + d]
            = entries_rope[block_entry * rope_dim + d];
    }
}


void gather_kv_cuda(
    torch::Tensor entries_fp8,
    torch::Tensor entries_rope,
    torch::Tensor inv_scale,
    torch::Tensor topk_indices,
    torch::Tensor block_table,
    torch::Tensor output,
    int64_t entries_per_block, int64_t rope_dim
) {
    int T = topk_indices.size(0);
    int top_k = topk_indices.size(1);
    int head_dim = entries_fp8.size(2) + entries_rope.size(2);
    int max_logical_blocks = block_table.size(1);

    int total_entries = T * top_k;
    int threads = 128;
    gather_kv_kernel<<<total_entries, threads>>>(
        entries_fp8.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(entries_rope.data_ptr<at::BFloat16>()),
        inv_scale.data_ptr<float>(),
        topk_indices.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        T, top_k, (int)entries_per_block,
        (int)head_dim, (int)rope_dim, max_logical_blocks
    );
    C10_CUDA_CHECK(cudaGetLastError());
}


// Forward declarations for gather_swa.cu functions
// (both files compiled together into one .so)
extern void gather_swa_cuda(
    torch::Tensor swa_fp8, torch::Tensor swa_rope, torch::Tensor swa_inv,
    torch::Tensor swa_pos, torch::Tensor request_slots, torch::Tensor output,
    int64_t head_dim, int64_t rope_dim);
extern void gather_all_compressed_cuda(
    torch::Tensor entries_fp8, torch::Tensor entries_rope, torch::Tensor inv_scale,
    torch::Tensor block_table, torch::Tensor block_lens, torch::Tensor output,
    int64_t entries_per_block, int64_t rope_dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_kv", &gather_kv_cuda, "Gather KV entries into dense tile");
    // gather_swa + gather_all_compressed are defined in gather_swa.cu
    // (compiled together into the same .so)
    m.def("gather_swa", &gather_swa_cuda, "Gather SWA window into dense BF16 tile");
    m.def("gather_all_compressed", &gather_all_compressed_cuda, "Gather all compressed KV for HCA");
}
