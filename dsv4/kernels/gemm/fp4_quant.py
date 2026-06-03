"""
NVFP4 quantization primitives — TOOLCHAIN BLOCKED in CuTeDSL.

CuTeDSL's MLIR lowering pipeline CANNOT lower any float→int operation:
  - arith.fptosi → LLVM ERROR: unsupported operation
  - llvm.inline_asm with cvt.rni.s32.f32 → LLVM ERROR: unsupported operation
  - nvvm.inline_ptx with cvt.rni.s32.f32 → LLVM ERROR: unsupported operation
  - llvm.bitcast Float32→Int32 → LLVM ERROR: unsupported operation

The pipeline has no path from Float32 MLIR types to Int32 MLIR types.
This is a fundamental toolchain limitation, not an implementation issue.

Production path: Use dsv4/kernels/cuda/quantize_nvfp4.cu instead.
That kernel uses __float2int_rn() and raw CUDA intrinsics — works perfectly.

For NVFP4-1.1 (fusing FP4 quant into MoE SwiGLU epilogue), the approach
will be a post-epilogue CUDA kernel that reads BF16 from GMEM and quantizes
to FP4. See ROADMAP.md Priority 3.

This file is kept for documentation of the toolchain limitation.
If CuTeDSL gains float→int support in the future, these primitives can be
reimplemented here.
"""

# All functions removed. Use dsv4/kernels/cuda/quantize_nvfp4.cu instead.
#
# Attempted approaches (all failed with "LLVM ERROR: unsupported operation"):
# 1. arith.fptosi (cutlass.Int32(float_val))
# 2. llvm.inline_asm with cvt.rni.s32.f32, cvt.rzi.s32.f32, cvt.rmi.s32.f32
# 3. nvvm.inline_ptx with cvt.rni.s32.f32
# 4. llvm.bitcast Float32 → Int32
# 5. Threshold rounding (Float32 comparisons selecting Int32 constants) —
#    this SHOULD work since it never generates fptosi, but even a trivial
#    Int32 GMEM store kernel fails on the B200 as of 2026-05-28.
#    Possible GPU state corruption from prior LLVM ERROR crashes.
#    TODO: Re-test threshold approach after B200 GPU state is clean.
