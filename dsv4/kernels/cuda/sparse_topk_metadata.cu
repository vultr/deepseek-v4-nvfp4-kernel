#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cstdint>

// ============================================================================
// C128A topk metadata kernel
// ============================================================================
// For C128A (compress_ratio=128) decode tokens:
//   - position -> num_compressed = (position + 1) / compress_ratio
//   - For each compressed KV slot [0, num_compressed):
//     block_index = i / block_size
//     block_offset = i % block_size
//     global_slot = block_table[req_idx, block_index] * block_size + block_offset
//   - Output: global_slot IDs in out_indices, count in out_lens
//   - Invalid tokens (slot_mapping < 0) get length 0
//
// For prefill tokens:
//   - Output: local indices [0, 1, ..., num_compressed-1, -1, -1, ...]
// ============================================================================

__global__ void build_c128a_topk_metadata_kernel(
    // Decode outputs
    int32_t* __restrict__ global_decode_ptr,     // [num_decode_tokens, max_compressed]
    int64_t global_decode_stride,                 // stride in elements
    int32_t* __restrict__ decode_lens_ptr,        // [num_decode_tokens]
    // Prefill output
    int32_t* __restrict__ prefill_local_ptr,      // [num_prefill_tokens, max_compressed]
    int64_t prefill_local_stride,
    // Inputs
    const int64_t* __restrict__ positions_ptr,    // [num_tokens]
    int32_t compress_ratio,
    int32_t max_compressed_tokens,
    int32_t num_decode_tokens,
    const int32_t* __restrict__ token_to_req_ptr, // [num_tokens]
    const int32_t* __restrict__ block_table_ptr,  // [num_reqs, max_blocks_per_seq]
    int64_t block_table_stride,                   // stride in elements
    int32_t block_size,
    const int64_t* __restrict__ slot_mapping_ptr  // [num_tokens]
) {
    int token_idx = blockIdx.x;
    int64_t position = positions_ptr[token_idx];
    int32_t num_compressed = static_cast<int32_t>((position + 1) / compress_ratio);
    if (num_compressed > max_compressed_tokens)
        num_compressed = max_compressed_tokens;

    bool is_decode = token_idx < num_decode_tokens;

    if (is_decode) {
        // Decode: block-table lookup -> global slot ids + count
        int64_t slot = slot_mapping_ptr[token_idx];
        bool is_valid = slot >= 0;
        int32_t req_idx = token_to_req_ptr[token_idx];
        int32_t count = 0;

        for (int32_t i = 0; i < max_compressed_tokens; i++) {
            int64_t out_offset = static_cast<int64_t>(token_idx) * global_decode_stride + i;
            if (i < num_compressed) {
                int32_t block_index = i / block_size;
                int32_t block_offset = i % block_size;
                int64_t bt_offset = static_cast<int64_t>(req_idx) * block_table_stride + block_index;
                int32_t block_number = block_table_ptr[bt_offset];
                int32_t slot_id = block_number * block_size + block_offset;
                global_decode_ptr[out_offset] = slot_id;
                count++;
            } else {
                global_decode_ptr[out_offset] = -1;
            }
        }
        decode_lens_ptr[token_idx] = is_valid ? count : 0;
    } else {
        // Prefill: write local indices [0, 1, ..., n-1, -1, ...]
        int32_t pfx_idx = token_idx - num_decode_tokens;
        for (int32_t i = 0; i < max_compressed_tokens; i++) {
            int64_t out_offset = static_cast<int64_t>(pfx_idx) * prefill_local_stride + i;
            prefill_local_ptr[out_offset] = (i < num_compressed) ? i : -1;
        }
    }
}

// ============================================================================
// C4A topk metadata kernel
// ============================================================================
// For C4A (compress_ratio=4) decode tokens:
//   - topk_indices: local compressed indices from the indexer
//   - Map each local index to a global KV cache slot via block table lookup
//   - Count valid entries (local_idx >= 0)
//   - Invalid tokens get length 0
// ============================================================================

__global__ void compute_c4a_global_topk_kernel(
    // Outputs
    int32_t* __restrict__ global_topk_ptr,        // [num_tokens, topk_dim]
    int64_t global_topk_stride,                    // stride in elements
    int32_t* __restrict__ topk_lens_ptr,           // [num_tokens]
    // Inputs
    const int32_t* __restrict__ local_topk_ptr,    // [num_tokens, topk_dim]
    int64_t local_topk_stride,                      // stride in elements
    int32_t topk_dim,
    const int32_t* __restrict__ token_to_req_ptr,   // [num_tokens]
    const int32_t* __restrict__ block_table_ptr,    // [num_reqs, max_blocks_per_seq]
    int64_t block_table_stride,
    int32_t block_size,
    const int32_t* __restrict__ is_valid_token_ptr  // [num_tokens] boolean as int32
) {
    int token_idx = blockIdx.x;
    int32_t is_valid = is_valid_token_ptr[token_idx];
    int32_t req_idx = token_to_req_ptr[token_idx];
    int32_t count = 0;

    for (int32_t i = 0; i < topk_dim; i++) {
        int64_t in_offset = static_cast<int64_t>(token_idx) * local_topk_stride + i;
        int32_t local_idx = local_topk_ptr[in_offset];
        int64_t out_offset = static_cast<int64_t>(token_idx) * global_topk_stride + i;

        if (local_idx >= 0) {
            int32_t block_index = local_idx / block_size;
            int32_t block_offset = local_idx % block_size;
            int64_t bt_offset = static_cast<int64_t>(req_idx) * block_table_stride + block_index;
            int32_t block_number = block_table_ptr[bt_offset];
            int32_t slot_id = block_number * block_size + block_offset;
            global_topk_ptr[out_offset] = slot_id;
            count++;
        } else {
            global_topk_ptr[out_offset] = -1;
        }
    }
    topk_lens_ptr[token_idx] = is_valid ? count : 0;
}

// ============================================================================
// Python bindings
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> build_c128a_topk_metadata_cuda(
    torch::Tensor positions,            // [num_tokens] int64
    int32_t compress_ratio,
    int32_t num_decode_tokens,
    torch::Tensor token_to_req,         // [num_tokens] int32
    torch::Tensor block_table,          // [num_reqs, max_blocks] int32
    int32_t block_size,
    torch::Tensor slot_mapping,         // [num_tokens] int64
    torch::Tensor global_decode_buffer, // [max_tokens, max_compressed] int32
    torch::Tensor decode_lens_buffer,   // [max_tokens] int32
    torch::Tensor prefill_buffer,       // [max_tokens, max_compressed] int32
    int32_t max_compressed_tokens
) {
    int32_t num_tokens = positions.size(0);
    int32_t num_prefill_tokens = num_tokens - num_decode_tokens;

    auto global_decode = global_decode_buffer.narrow(0, 0, num_decode_tokens);
    auto decode_lens = decode_lens_buffer.narrow(0, 0, num_decode_tokens);
    auto prefill_local = prefill_buffer.narrow(0, 0, num_prefill_tokens);

    if (num_tokens == 0) {
        return std::make_tuple(global_decode, decode_lens, prefill_local);
    }

    build_c128a_topk_metadata_kernel<<<num_tokens, 1>>>(
        global_decode_buffer.data_ptr<int32_t>(),
        global_decode_buffer.stride(0),
        decode_lens_buffer.data_ptr<int32_t>(),
        prefill_buffer.data_ptr<int32_t>(),
        prefill_buffer.stride(0),
        positions.data_ptr<int64_t>(),
        compress_ratio,
        max_compressed_tokens,
        num_decode_tokens,
        token_to_req.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        block_table.stride(0),
        block_size,
        slot_mapping.data_ptr<int64_t>()
    );

    return std::make_tuple(global_decode, decode_lens, prefill_local);
}

std::tuple<torch::Tensor, torch::Tensor> compute_c4a_global_topk_cuda(
    torch::Tensor local_topk,           // [num_tokens, topk_dim] int32
    torch::Tensor token_to_req,         // [num_tokens] int32
    torch::Tensor block_table,          // [num_reqs, max_blocks] int32
    int32_t block_size,
    torch::Tensor is_valid_token        // [num_tokens] bool (stored as int32)
) {
    int32_t num_tokens = local_topk.size(0);
    int32_t topk_dim = local_topk.size(1);

    auto global_topk = torch::empty_like(local_topk);
    auto topk_lens = torch::empty(num_tokens, local_topk.options().dtype(torch::kInt32));

    compute_c4a_global_topk_kernel<<<num_tokens, 1>>>(
        global_topk.data_ptr<int32_t>(),
        global_topk.stride(0),
        topk_lens.data_ptr<int32_t>(),
        local_topk.data_ptr<int32_t>(),
        local_topk.stride(0),
        topk_dim,
        token_to_req.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        block_table.stride(0),
        block_size,
        is_valid_token.data_ptr<int32_t>()
    );

    return std::make_tuple(global_topk, topk_lens);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_c128a_topk_metadata", &build_c128a_topk_metadata_cuda,
          "Build C128A topk metadata (global slot IDs + lengths)");
    m.def("compute_c4a_global_topk", &compute_c4a_global_topk_cuda,
          "Compute C4A global topk indices and lengths from local indices");
}
