/**
 * DSV4 FMHA shared definitions — base header for raw CUDA C++ kernels.
 *
 * ==================================================================
 * WHY RAW CUDA C++ INSTEAD OF CUTEDSL
 * ==================================================================
 *
 * CuTeDSL (the Python DSL for CUTLASS) has FOUR fundamental gaps on
 * Blackwell SM100 that made it impossible to ship a production FMHA
 * kernel. We spent weeks hitting these walls before switching to raw
 * CUDA C++ with inline PTX. The reference kernel you see here was
 * written and verified correct in ~2 hours.
 *
 * GAP 1: TMEM round-trip is BROKEN
 * ---------------------------------
 * CuTeDSL's Ld32x32bOp and St32x32bOp atoms have different column
 * mappings — they address DIFFERENT TMEM columns even when constructed
 * with the same layout. A NO-OP round-trip (load → store unchanged)
 * corrupts data by ~3% (cos ~0.97). This makes in-kernel O rescale
 * (the D1.5 fix) impossible in CuTeDSL — you can't read the TMEM
 * accumulator, multiply by acc_scale, and write it back. The MoE
 * kernel avoids this because it only does a ONE-WAY TMEM → regs →
 * SMEM → GMEM epilogue. FMHA needs a round-trip for per-KV-tile
 * rescaling. Raw CUDA with tcgen05.ld/st PTX gives us full control
 * over TMEM addressing.
 *
 * GAP 2: Float-to-int conversion is IMPOSSIBLE
 * ---------------------------------------------
 * The CuTeDSL MLIR pipeline CANNOT lower ANY float→int operation.
 * arith.fptosi, nvvm.inline_ptx with cvt.rni.s32.f32, llvm.bitcast
 * Float32→Int32 — ALL fail with "LLVM ERROR: unsupported operation."
 * This makes quantize-in-epilogue fusion (NVFP4-1.1) impossible in
 * CuTeDSL. Every quantization kernel needs f32→i32. Raw CUDA has
 * __float2int_rn which works perfectly.
 *
 * GAP 3: epilogue_tma_store BLOCKS multi-CTA
 * -------------------------------------------
 * The epilogue_tma_store helper cannot accept flat_divide-based GMEM
 * coordinates, which are required for multi-CTA grid launches. This
 * means FMHA is stuck at per-head Python launch (128 launches per Pro
 * decode step) instead of a single GPU launch. The MoE kernel avoids
 * this by using the explicit epilogue_tmem_copy + epilogue_smem_copy
 + tma_partition pipeline. Raw CUDA gives us full control over TMA
 * descriptors and GMEM writes.
 *
 * GAP 4: hd=512 MLIR backend HANGS
 * ---------------------------------
 * The CuTeDSL MLIR optimizer cannot process the hd=512 kernel in
 * reasonable time. Tracer completes in 0.8s (kernel is structurally
 * correct), but the MLIR optimizer runs for 3+ hours before we kill
 * it. This is a CuTeDSL/MLIR toolchain limitation, not a kernel bug.
 * Raw CUDA compiles hd=512 in seconds.
 *
 * ==================================================================
 * WHAT WORKS (tested on B200, CUDA 13.2, SM100)
 * ==================================================================
 * - BF16 conversion via inline PTX cvt.rn.bf16.f32 / cvt.f32.bf16
 * - Warp reductions (fmax, sum) via shfl_xor
 * - TMEM alloc/dealloc via tcgen05.alloc/dealloc PTX
 * - TMEM load/store via tcgen05.ld/st PTX (uint32_t b32 registers)
 * - TMEM fence via tcgen05.wait::st/ld.sync.aligned
 * - One-way TMEM epilogue: TMEM → regs → normalize → BF16 → GMEM
 *   (verified cos 0.999999 at hd=64, cos 0.999998 at hd=128)
 *
 * ==================================================================
 * TMEM LANE MAPPING (verified on B200 via test_tmem_lane_mapping.cu)
 * ==================================================================
 * tcgen05.st/ld 16x256b.x1.b32 are warp-collective operations:
 *   - ALL 32 lanes in a warp MUST execute them (or HANG)
 *   - Each lane i reads/writes positions i*4+0..i*4+3 within the column
 *   - 32 lanes × 4 FP32 = 128 FP32 per column
 *   - For row 0: lane 0 = positions 0-3, lane 1 = 4-7, ..., lane 31 = 124-127
 *   - HD values need ceil(HD/128) TMEM columns
 *   - Column address = tmem_base + column_index
 *   - tmem_base is WRITTEN to SMEM by tcgen05.alloc (read it back after alloc)
 *
 * CRITICAL: If fewer than 32 lanes call tmem_store/tmem_load, the
 * warp is divergent on a collective operation and the GPU HANGS.
 * Always loop over enough columns that all 32 lanes participate,
 * even if some write don't-care data to unused columns.
 *
 * ==================================================================
 * TMEM ALLOC/DEALLOC
 * ==================================================================
 * - tcgen05.alloc: pass SMEM pointer (via __cvta_generic_to_shared)
 *   and num_columns (power of 2, minimum 32). Must be called by an
 *   ENTIRE fully active warp (all 32 lanes). The alloc WRITES the
 *   tmem_base value to the SMEM location.
 * - tcgen05.dealloc: pass tmem_base (the VALUE from SMEM, not the
 *   SMEM pointer) and num_columns. Must also be warp-collective.
 *
 * ==================================================================
 * KEY INSIGHT FOR NVIDIA
 * ==================================================================
 * The CuTeDSL float→int gap is the single most impactful limitation.
 * It blocks ALL quantization-epilogue fusion (NVFP4-1.1, NVFP4-1.2)
 * and forces a separate kernel launch for every quantize step. The
 * TMEM round-trip issue is the second most impactful — it blocks
 * in-kernel O rescaling, forcing 5-9 Python kernel launches per
 * decode step instead of 1 GPU launch. Both of these work trivially
 * in raw CUDA C++ with inline PTX.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace dsv4::kernels::attention {

typedef unsigned short bf16_t;

__device__ __forceinline__ bf16_t f32_to_bf16(float f) {
    bf16_t h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
__device__ __forceinline__ float bf16_to_f32(bf16_t h) {
    float f; asm("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}

// Float ↔ uint32_t bitcast (for TMEM operations)
__device__ __forceinline__ uint32_t f32_to_u32(float f) { uint32_t u; memcpy(&u,&f,4); return u; }
__device__ __forceinline__ float u32_to_f32(uint32_t u) { float f; memcpy(&f,&u,4); return f; }

constexpr int WARP = 32;
constexpr int NTHREADS = 192;
constexpr int NWARPS = 6;

__device__ __forceinline__ float wmax(float v) {
    for(int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_xor_sync(0xFFFFFFFF,v,o)); return v;
}
__device__ __forceinline__ float wsum(float v) {
    for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(0xFFFFFFFF,v,o); return v;
}

// ==================================================================
// TMEM operations — inline PTX for Blackwell SM100
// ==================================================================
// These wrap the tcgen05 PTX instructions for TMEM management.
// All TMEM load/store ops are WARP-COLLECTIVE: all 32 lanes must
// execute them. See the header comment for the lane mapping.
// ==================================================================

__device__ void tmem_alloc(uint32_t smem_ptr, int num_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_ptr), "r"(num_cols));
}

__device__ void tmem_dealloc(uint32_t tmem_ptr, int num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(tmem_ptr), "r"(num_cols));
}

/** Load 16 rows × 256 bits from TMEM column. Warp-collective. 4 uint32_t per lane. */
__device__ void tmem_load(uint32_t col_addr,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(col_addr));
}

/** Store 16 rows × 256 bits to TMEM column. Warp-collective. 4 uint32_t per lane. */
__device__ void tmem_store(uint32_t col_addr,
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.16x256b.x1.b32 [%0], {%1, %2, %3, %4};"
        :: "r"(col_addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

/** Wait for prior TMEM stores to be visible. Call after tmem_store. */
__device__ void tmem_fence_store() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

/** Wait for prior TMEM loads to complete. Call after tmem_load. */
__device__ void tmem_fence_load() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

} // namespace
