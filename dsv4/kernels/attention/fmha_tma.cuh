/**
 * DSV4 FMHA — TMA async load infrastructure for Blackwell SM100.
 *
 * ==================================================================
 * CUDA 13 CRITICAL NOTES (verified on B200, driver 580.126.20)
 * ==================================================================
 *
 * 1. globalStrides are in BYTES, not elements.
 *    Old (CUDA 12): gs[] = {1, cols}       — element strides
 *    New (CUDA 13): gs[] = {cols*2, ...}   — byte strides
 *    This was the root cause of cuTensorMapEncodeTiled returning
 *    INVALID_VALUE for 2D+ descriptors.
 *
 * 2. Use BFLOAT16 data type (CU_TENSOR_MAP_DATA_TYPE_BFLOAT16)
 *    instead of UINT16 for BF16 tensors.
 *
 * 3. mbarrier: MUST use mbarrier.arrive.expect_tx after issuing TMA.
 *    Without it, the tx_count underflows (TMA decrements by bytes but no one
 *    incremented it) and the phase never completes. This was the root cause
 *    of the multi-warp TMA hang — NOT the @p bra pattern (which works fine).
 *    See gau-nernst tcgen05 tutorial for the correct TMA + mbarrier pattern.
 *
 * 4. CUTLASS driver workaround: for driver <= 13.1 and total_bytes
 *    < 128KB, clear bit 21 of descriptor word[1]. See
 *    docs/cuda13_tma_notes.md for details.
 * ==================================================================
 */

#pragma once

#include "fmha_common.cuh"
#include <cstdint>

namespace dsv4::kernels::attention {

// ==================================================================
// TMA descriptor helpers (host-side)
// ==================================================================

/**
 * Create a 2D TMA descriptor for a BF16 tensor of shape (rows, cols).
 *
 * IMPORTANT: CUDA 13 requires byte strides, not element strides.
 * globalStrides[0] = cols * 2 (bytes, BF16 = 2 bytes)
 * For rank=2, only 1 stride is needed (rank-1).
 */
inline bool create_tma_desc_2d_bf16(
    CUtensorMap* out,
    const void* gmem_ptr,
    uint64_t rows,
    uint64_t cols,
    uint32_t tile_rows,
    uint32_t tile_cols,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE
) {
    // 2D: (cols, rows) innermost-first
    // Byte strides: stride[0] = cols * sizeof(bf16) = cols * 2
    uint64_t global_dim[] = {cols, rows};
    uint64_t global_str[] = {cols * 2};  // byte stride (CUDA 13)
    uint32_t tile_dim[] = {tile_cols, tile_rows};
    uint32_t tile_str[] = {1, 1};  // element strides within tile

    CUresult res = cuTensorMapEncodeTiled(
        out,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        const_cast<void*>(gmem_ptr),
        global_dim, global_str, tile_dim, tile_str,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled failed: error=%d, gdim=[%lu,%lu], gstr=[%lu], tdim=[%u,%u]\n",
            (int)res, global_dim[0], global_dim[1], global_str[0],
            tile_dim[0], tile_dim[1]);
        return false;
    }

    // CUTLASS driver workaround: clear bit 21 of desc[1] for driver <= 13.1 + small tensors
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    size_t total_bytes = rows * cols * 2;  // BF16
    if (driver_version <= 13010 && total_bytes < 131072) {
        reinterpret_cast<uint64_t*>(out)[1] &= ~(1ULL << 21);
    }

    return true;
}

// ==================================================================
// TMA kernel-side operations
// ==================================================================

/**
 * Initialize an mbarrier in SMEM with expected arrival count.
 * For TMA, use count=1 (single producer thread issues the TMA).
 */
__device__ __forceinline__ void tma_mbarrier_init(uint32_t smem_mbar, uint32_t arrival_count = 1) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(smem_mbar), "r"(arrival_count));
}

/**
 * Arrive at mbarrier and set expected transaction bytes.
 *
 * CRITICAL: This MUST be called after issuing TMA but before waiting.
 * The TMA hardware decrements tx_count by the bytes transferred (complete_tx::bytes).
 * Without this arrive.expect_tx, tx_count starts at 0 and underflows when TMA
 * completes, so the mbarrier phase never completes and the wait hangs forever.
 *
 * This also decrements the arrival count (the TMA-issuing thread "arrives").
 * The phase completes when BOTH arrival count AND tx_count reach 0.
 *
 * @param smem_mbar  SMEM mbarrier address (via __cvta_generic_to_shared)
 * @param tx_bytes   Total bytes the TMA will transfer (must match exactly)
 */
__device__ __forceinline__ void tma_mbarrier_arrive_expect_tx(uint32_t smem_mbar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(smem_mbar), "r"(tx_bytes) : "memory");
}

/**
 * Issue a 2D TMA async copy from GMEM to SMEM.
 *
 * @param smem_dst  SMEM destination address (via __cvta_generic_to_shared)
 * @param tma_desc  Pointer to CUtensorMap in device memory (uint64_t cast)
 * @param smem_mbar SMEM mbarrier address (via __cvta_generic_to_shared)
 * @param coord_x   Column coordinate (innermost dimension)
 * @param coord_y   Row coordinate (outermost dimension)
 */
__device__ __forceinline__ void tma_load_2d(
    uint32_t smem_dst,
    uint64_t tma_desc,
    uint32_t smem_mbar,
    int coord_x,
    int coord_y
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];"
        :: "r"(smem_dst),
           "l"(tma_desc),
           "r"(smem_mbar),
           "r"(coord_x),
           "r"(coord_y)
        : "memory"
    );
}

/**
 * Wait for mbarrier phase completion using try_wait.parity.
 *
 * Uses the CUTLASS-style @P1 bra DONE pattern (from gau-nernst tutorial).
 * Phase parity must be tracked externally and flipped after each wait.
 *
 * IMPORTANT: The mbarrier must be properly initialized with
 * tma_mbarrier_init + tma_mbarrier_arrive_expect_tx before calling wait.
 * Without expect_tx, the tx_count underflows and this function hangs forever.
 */
__device__ __forceinline__ void tma_mbarrier_wait(uint32_t smem_mbar, int phase) {
    // CUTLASS-style wait loop (gau-nernst tutorial)
    // 0x989680 ticks ≈ 10us timeout per attempt
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(smem_mbar), "r"(phase), "r"(0x989680)
        : "memory"
    );
}

// ==================================================================
// TMA parameter structure
// ==================================================================

struct FmhaTmaDescriptors {
    CUtensorMap* __restrict__ tma_q;   // Q descriptor: (T, HD)
    CUtensorMap* __restrict__ tma_k;   // K descriptor: (s_k, HD)
    CUtensorMap* __restrict__ tma_v;   // V descriptor: (HD, s_k)
};

// ==================================================================
// TMA store (GMEM write) operations
// ==================================================================
// The one-way epilogue writes O from SMEM to GMEM via TMA store.
// This replaces direct GMEM writes and enables multi-CTA grids.
// ==================================================================

/**
 * Issue a 2D TMA async copy from SMEM to GMEM (store).
 *
 * PTX syntax: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
 *   [tensorMap, tensorCoords], [srcMem];
 *
 * IMPORTANT: TMA store uses bulk_group (not mbarrier) for completion tracking.
 * Completion is tracked via cp.async.bulk.commit_group + cp.async.bulk.wait_group.
 *
 * @param smem_src  SMEM source address (via __cvta_generic_to_shared)
 * @param tma_desc  Pointer to CUtensorMap in device memory (uint64_t cast)
 * @param coord_x   Column coordinate (innermost dimension)
 * @param coord_y   Row coordinate (outermost dimension)
 */
__device__ __forceinline__ void tma_store_2d(
    uint32_t smem_src,
    uint64_t tma_desc,
    uint32_t smem_mbar,
    int coord_x,
    int coord_y
) {
    // cp.async.bulk.tensor.2d.global.shared::cluster.mbarrier::complete_tx::bytes
    // For store: dest=global, src=shared::cluster
    // mbarrier on SMEM side tracks completion
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cluster.mbarrier::complete_tx::bytes "
        "[%1, {%2, %3}], [%0], [%4];"
        :: "r"(smem_src),
           "l"(tma_desc),
           "r"(coord_x),
           "r"(coord_y),
           "r"(smem_mbar)
        : "memory"
    );
}

/**
 * Commit all pending TMA store operations to a group.
 * Must be called after issuing TMA stores, before waiting.
 */
__device__ __forceinline__ void tma_store_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

/**
 * Wait for N most recent TMA store groups to complete.
 * N=0 waits for all groups. N=1 waits for the most recent group.
 */
__device__ __forceinline__ void tma_store_wait_group(int n) {
    asm volatile("cp.async.bulk.wait_group %0;" :: "r"(n) : "memory");
}

/**
 * Wait for all pending TMA store groups to complete.
 */
__device__ __forceinline__ void tma_store_wait_all() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

} // namespace dsv4::kernels::attention
