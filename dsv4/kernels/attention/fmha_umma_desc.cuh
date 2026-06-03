/**
 * DSV4 FMHA — UMMA SMEM layout and descriptor construction.
 *
 * ==================================================================
 * SMEM LAYOUT FOR tcgen05.mma (from PTX spec + gau-nernst tutorial)
 * ==================================================================
 *
 * tcgen05.mma expects SMEM data in a specific "canonical" layout
 * described by the descriptor's LBO and SBO strides.
 *
 * For K-major, SWIZZLE_NONE, BF16:
 *   LBO = BLOCK_MN * 16 (in bytes, NOT in 16B units)
 *     → desc field = BLOCK_MN (after >> 4)
 *     This is the stride from the "first column" to the "second column"
 *     of the 8×2 core matrix tile. Each column is 16 bytes wide and
 *     BLOCK_MN rows tall.
 *
 *   SBO = 8 * 16 = 128 (in bytes, NOT in 16B units)
 *     → desc field = 8 (after >> 4)
 *     This is the stride between 8-row groups.
 *
 * The hardware walks the SMEM as:
 *   For 8-row group g (0..BLOCK_MN/8-1):
 *     For column c (0 or 1):  (2 columns per MMA, 16B each)
 *       Read 8×16B at address: start + g * SBO + c * LBO
 *
 * This means the SMEM layout is:
 *   [core(0,0), core(1,0), ..., core(BLOCK_MN/8-1,0),
 *    core(0,1), core(1,1), ..., core(BLOCK_MN/8-1,1)]
 *
 * Where core(g, c) is an 8×8 BF16 tile (128 bytes) representing
 * rows [8g, 8g+8) and columns [8c, 8c+8) of the logical matrix.
 *
 * ==================================================================
 * DESCRIPTOR BIT LAYOUT (64 bits)
 * ==================================================================
 * [0,14)    start_address    SMEM address >> 4 (in 16-byte units)
 * [14,16)   (unused)
 * [16,30)   leading_byte_offset (LBO) in 16-byte units
 * [30,32)   (unused)
 * [32,46)   stride_byte_offset (SBO) in 16-byte units
 * [46,48)   version          = 1
 * [48,49)   (unused)
 * [49,52)   base_offset
 * [52,53)   lbo_mode         0=LBO in 16B, 1=LBO in 128B
 * [53,61)   (unused)
 * [61,64)   layout_type      0=NONE, 1=SW128, 2=SW64, 3=SW32
 *
 * For SWIZZLE_NONE:
 *   LBO = BLOCK_MN * 16 (bytes) → field = BLOCK_MN (16B units)
 *   SBO = 128 (bytes) → field = 8 (16B units)
 *   lbo_mode = 0, layout_type = 0
 *
 * For SWIZZLE_128B:
 *   SBO = 8 * 128 = 1024 (bytes) → field = 64 (16B units)
 *   lbo_mode = 1, layout_type = 1
 *   (LBO not explicitly set — hardware assumes LBO=1 in 128B units)
 *
 * ==================================================================
 * MMA K-TILING
 * ==================================================================
 * tcgen05.mma.kind::f16 operates on K=16 BF16 per MMA call.
 * The descriptor describes a single MMA K-slice (16 BF16 = 32 bytes).
 * For BLOCK_K > 16, we call MMA multiple times, each with a different
 * descriptor pointing to the next K-slice in SMEM.
 *
 * For NO swizzle, each 16B slice is BLOCK_MN * 16B bytes in SMEM.
 * The k-th MMA uses descriptor with start_address pointing to
 * the k-th (BLOCK_MN, 16B) slice.
 * But wait — MMA needs 2 columns (32B). The descriptor covers 2 slices.
 * The first slice is at start, the second at start + LBO.
 * So for k-th MMA (k even), the start is at the k/2-th slice base.
 *
 * Actually from gau-nernst: for NO swizzle, each MMA call gets a
 * descriptor pointing to the start of the (BLOCK_MN, 32B) region.
 * The 32B = 2 × 16B columns. LBO = BLOCK_MN * 16B gives the stride
 * between the 2 columns.
 *
 * The k-th MMA reads columns [16k, 16k+16) in SMEM.
 * In the core-matrix layout, columns are BLOCK_MN * 16B bytes apart
 * (each column is BLOCK_MN/8 core matrices × 128B = BLOCK_MN * 16B).
 * Wait, each core matrix is 128B. BLOCK_MN/8 core matrices per column = BLOCK_MN/8 * 128 = BLOCK_MN * 16B. ✓
 * So the k-th 16B column starts at offset k * BLOCK_MN * 16B.
 * MMA needs columns 2k and 2k+1, so the descriptor for the k-th MMA
 * starts at column 2k: offset = 2k * BLOCK_MN * 16B = k * BLOCK_MN * 32B.
 * This matches gau-nernst: A_smem + k * BLOCK_M * 32.
 */
#pragma once

#include "fmha_common.cuh"
#include <cstdint>

namespace dsv4::kernels::attention {

// MMA K-tile size for BF16: 16 elements (32 bytes)
constexpr int MMA_K_BF16 = 16;

// ==================================================================
// SMEM layout: row-major → canonical K-major core-matrix layout
// ==================================================================
// The canonical layout for NO swizzle interleaves core matrices:
//   [core(0,0), core(1,0), ..., core(MN/8-1,0), core(0,1), ...]
// where core(g, c) = 8×8 BF16 tile at rows [8g, 8g+8), cols [8c, 8c+8)
//
// Each core matrix is 128 bytes (8 rows × 8 BF16).
// The offset of core(g, c) in BF16 elements:
//   c * (BLOCK_MN/8) * 64 + g * 64 + local_r * 8 + local_c
// = c * BLOCK_MN * 8 + g * 64 + local_r * 8 + local_c
// (since BLOCK_MN/8 * 64 = BLOCK_MN * 8)
//
// And in bytes: (c * BLOCK_MN * 8 + g * 64 + ...) * 2
// = c * BLOCK_MN * 16 + g * 128 + ...
//
// This matches: g * SBO + c * LBO + local_r * 16 + local_c * 2
// where SBO = 128, LBO = BLOCK_MN * 16 ✓
// ==================================================================

/**
 * Write a (ROWS, COLS) BF16 matrix from row-major to canonical
 * K-major core-matrix layout for tcgen05.mma (NO swizzle).
 *
 * ROWS must be a multiple of 8 (ideally 128 for full MMA tile).
 * COLS must be a multiple of 8.
 * STRIDE is the thread stride (default NTHREADS=192 for full CTA).
 */
template<int ROWS, int COLS, int STRIDE = NTHREADS>
__device__ void write_smem_canonical(bf16_t* __restrict__ dst,
                                     const bf16_t* __restrict__ src) {
    constexpr int CORES_MN = ROWS / 8;
    constexpr int CORES_K = COLS / 8;
    constexpr int TOTAL = ROWS * COLS;
    for (int i = threadIdx.x; i < TOTAL; i += STRIDE) {
        int r = i / COLS;
        int c = i % COLS;
        int core_mn = r / 8;
        int core_k = c / 8;
        int local_r = r % 8;
        int local_c = c % 8;
        // Canonical layout: column-major in (core_mn, core_k)
        // All core_mn values for core_k=0 first, then core_k=1, etc.
        int dst_idx = core_k * CORES_MN * 64 + core_mn * 64 + local_r * 8 + local_c;
        dst[dst_idx] = src[i];
    }
}

/**
 * Write Q (1, HD) to SMEM in canonical layout, padded to (128, HD).
 * Only row 0 has actual data; rows 1-127 are zero.
 */
template<int HD>
__device__ void write_q_to_smem(bf16_t* __restrict__ dst,
                                const bf16_t* __restrict__ q) {
    constexpr int CORES_MN = 128 / 8;  // 16
    constexpr int CORES_K = HD / 8;
    // Zero all first
    for (int i = threadIdx.x; i < 128 * HD; i += NTHREADS) dst[i] = 0;
    // Write row 0 only: row 0 is in core_mn=0, local_r=0
    for (int c = threadIdx.x; c < HD; c += NTHREADS) {
        int core_k = c / 8;
        int local_c = c % 8;
        int dst_idx = core_k * CORES_MN * 64 + 0 * 64 + 0 * 8 + local_c;
        dst[dst_idx] = q[c];
    }
}

/**
 * Write K (SK, HD) to SMEM in canonical layout, padded to (128, HD).
 */
template<int SK, int HD>
__device__ void write_k_to_smem(bf16_t* __restrict__ dst,
                                const bf16_t* __restrict__ k) {
    constexpr int CORES_MN = 128 / 8;  // 16
    constexpr int CORES_K = HD / 8;
    // Zero all first
    for (int i = threadIdx.x; i < 128 * HD; i += NTHREADS) dst[i] = 0;
    // Write actual rows
    for (int i = threadIdx.x; i < SK * HD; i += NTHREADS) {
        int r = i / HD;
        int c = i % HD;
        int core_mn = r / 8;
        int core_k = c / 8;
        int local_r = r % 8;
        int local_c = c % 8;
        int dst_idx = core_k * CORES_MN * 64 + core_mn * 64 + local_r * 8 + local_c;
        dst[dst_idx] = k[i];
    }
}

/**
 * Write V (HD, SK) to SMEM in canonical layout, padded to (HD, 128).
 * V is stored as (HD, SK) in GMEM. For PV GEMM, V is the B matrix
 * with shape (BLOCK_N=128, BLOCK_K=HD) K-major in SMEM.
 * But V in GMEM is (HD, SK) — we need to transpose it.
 * V's column c (position c in the sequence) maps to row c in B.
 * B's row c has columns d=0..HD-1, which is V[d, c] = V[d * SK + c].
 * So B[r, d] = V[d * SK + r] for r < SK, d < HD.
 */
template<int SK, int HD>
__device__ void write_v_to_smem(bf16_t* __restrict__ dst,
                                const bf16_t* __restrict__ v) {
    // dst is (128, HD) in canonical layout
    constexpr int CORES_MN = 128 / 8;
    constexpr int CORES_K = HD / 8;
    for (int i = threadIdx.x; i < 128 * HD; i += NTHREADS) dst[i] = 0;
    // B = (128, HD) where B[r, d] = v[d * SK + r] for r < SK
    for (int d = threadIdx.x; d < HD; d += NTHREADS) {
        for (int r = 0; r < SK; r += NTHREADS) {
            if (r >= SK) continue;
            int core_mn = r / 8;
            int core_k = d / 8;
            int local_r = r % 8;
            int local_c = d % 8;
            int dst_idx = core_k * CORES_MN * 64 + core_mn * 64 + local_r * 8 + local_c;
            dst[dst_idx] = v[d * SK + r];
        }
    }
}

// ==================================================================
// UMMA descriptor construction
// ==================================================================

/**
 * desc_encode: convert byte value to 16B-unit descriptor field.
 * Same as value >> 4 (divide by 16).
 */
__device__ __forceinline__ uint64_t desc_encode(uint64_t byte_val) {
    return byte_val >> 4;
}

/**
 * Construct a 64-bit SMEM descriptor for tcgen05.mma with
 * K-major, SWIZZLE_NONE, BF16.
 *
 * From gau-nernst's tutorial:
 *   LBO = BLOCK_MN * 16 (bytes)  → desc field = BLOCK_MN
 *   SBO = 128 (bytes)            → desc field = 8
 *
 * @param smem_addr  SMEM address (from __cvta_generic_to_shared)
 * @param block_mn   The MN dimension of the MMA tile (e.g., 128 for M=128)
 */
__device__ __forceinline__ uint64_t make_umma_desc_kmajor_none(
    uint32_t smem_addr, int block_mn
) {
    const uint64_t LBO = block_mn * 16;  // bytes
    const uint64_t SBO = 128;            // bytes (8 * 16B)
    uint64_t desc = 0;
    desc |= desc_encode(smem_addr) & 0x3FFF;         // start_address
    desc |= (desc_encode(LBO) & 0x3FFF) << 16;        // LBO
    desc |= (desc_encode(SBO) & 0x3FFF) << 32;          // SBO
    desc |= 1ULL << 46;                                 // version = 1
    return desc;
}

/**
 * Construct descriptor for K-major, SWIZZLE_128B, BF16.
 *
 * From gau-nernst's tutorial:
 *   SBO = 8 * 128 = 1024 (bytes) → desc field = 64
 *   layout_type = 2 (SW128 in PTX encoding — wait, gau-nernst uses 2ULL << 61)
 *   lbo_mode = 1 (128B units)
 *
 * @param smem_addr  SMEM address (from __cvta_generic_to_shared), must be 128B-aligned
 * @param block_mn   The MN dimension of the MMA tile
 */
__device__ __forceinline__ uint64_t make_umma_desc_kmajor_sw128(
    uint32_t smem_addr, int block_mn
) {
    const uint64_t SBO = 8 * 128;  // 1024 bytes
    uint64_t desc = 0;
    desc |= desc_encode(smem_addr) & 0x3FFF;
    desc |= (desc_encode(SBO) & 0x3FFF) << 32;
    desc |= 1ULL << 46;     // version
    desc |= 1ULL << 52;     // lbo_mode = 1 (128B units)
    desc |= 2ULL << 61;     // layout_type = 2 (SW128)
    return desc;
}

// ==================================================================
// tcgen05.mma PTX wrappers
// ==================================================================

/**
 * tcgen05.mma SS: both operands from SMEM, result to TMEM.
 * A (SMEM) × B (SMEM) → D (TMEM).
 * Only ONE thread per CTA calls this.
 */
__device__ void umma_ss_f16(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t i_desc, bool accumulate = false
) {
    uint32_t scaleC_bits = accumulate ? 0x3F800000u : 0u;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
        "}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
           "r"(i_desc), "r"(scaleC_bits),
           "r"(0), "r"(0), "r"(0), "r"(0)
    );
}

/**
 * tcgen05.mma TS: A from TMEM, B from SMEM, result to TMEM.
 * Used for PV GEMM.
 */
__device__ void umma_ts_f16(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t i_desc, bool accumulate = true
) {
    uint32_t scaleC_bits = accumulate ? 0x3F800000u : 0u;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p;\n\t"
        "}"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b),
           "r"(i_desc), "r"(scaleC_bits),
           "r"(0), "r"(0), "r"(0), "r"(0)
    );
}

/**
 * Construct the instruction descriptor (idesc) for tcgen05.mma.kind::f16.
 * From gau-nernst:
 *   bit 4: dtype=FP32 (1)
 *   bit 7: atype=BF16 (1)
 *   bit 10: btype=BF16 (1)
 *   bits 17-23: MMA_N (BLOCK_N >> 3)
 *   bits 24-28: MMA_M (BLOCK_M >> 4)
 */
__device__ __forceinline__ uint32_t make_idesc(int block_m, int block_n) {
    return (1U << 4)    // dtype=FP32
         | (1U << 7)    // atype=BF16
         | (1U << 10)   // btype=BF16
         | ((uint32_t)(block_n >> 3) << 17)  // MMA_N
         | ((uint32_t)(block_m >> 4) << 24); // MMA_M
}

/**
 * tcgen05.mma SS for .kind::f8f6f4 with E4M3xE4M3 -> FP32.
 * A and B element types are encoded in idesc. For B1 we use E4M3/E4M3.
 */
__device__ void umma_ss_f8f6f4(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t i_desc, bool accumulate = false
) {
    uint32_t scaleC_bits = accumulate ? 0x3F800000u : 0u;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
           "r"(i_desc), "r"(scaleC_bits)
    );
}

/** Instruction descriptor for .kind::f8f6f4 E4M3 x E4M3 -> FP32. */
__device__ __forceinline__ uint32_t make_idesc_f8_e4m3(int block_m, int block_n) {
    return (1U << 4)                         // dtype = F32
         | ((uint32_t)(block_n >> 3) << 17)  // MMA_N
         | ((uint32_t)(block_m >> 4) << 24); // MMA_M
}

} // namespace dsv4::kernels::attention
