"""
DeepSeek-V4 CSA/HCA compressor kernels for CUTLASS CuTe DSL / Blackwell.

This is a production-oriented fusion boundary for the PyTorch reference in
`Pasted markdown.md`:

  1. Run the projection stage as one or two packed Blackwell GEMMs:
       CSA main:    H @ [W_a_KV | W_a_Z | W_b_KV | W_b_Z]       -> (N, 4*C)
       CSA indexer: H @ [W_I_a_KV | W_I_a_Z | W_I_b_KV | W_I_b_Z] -> (N, 4*C_I)
       HCA:         H @ [W_KV | W_Z]                            -> (N, 2*C)

     Use your tcgen05 / NVFP4 blockscaled GEMM here.  Keep the packed projection
     outputs in BF16/FP32 as you prefer; the compressor reads them as tensor
     elements and accumulates the softmax reduction in FP32.

  2. These native CuTe DSL kernels fuse:
       bias add + column-wise softmax over token positions + weighted C sum
       + partial RoPE for KV output.

The full projection+compression single-kernel tcgen05 variant is possible, but it
needs your exact NVFP4 weight/scale layout and preferred tile shape.  This file is
therefore the safe fusion seam: the expensive D x C math stays in your Blackwell
GEMM, while the small-position softmax/reduction/rope path avoids PyTorch ops and
extra materialization after projection.

Target dimensions used by DeepSeek-V4-Pro reference:
  D=7168, C=512, C_I=128, CSA_M=4, HCA_M=128, NOPE=448, ROPE=64.

Assumptions:
  * Tensors are contiguous row-major from PyTorch/DLPack.
  * Projection buffers are laid out as described above.
  * One sequence at a time.  State/tail management remains on the caller side.
  * For CSA continuation across calls, provide external previous-block B-side
    projections for the first committed block.  For fresh prefill, set
    has_external_prev=False.

NOTE: I cannot compile this in this sandbox because CUTLASS CuTe DSL is not
installed here.  It follows the CuTe DSL @jit/@kernel launch style, but you may
need tiny API-name edits if you are pinned to a specific CUTLASS 4.x commit.
"""

from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

LOG2_E = 1.44269504088896340736


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@cute.jit
def _expf(x: cutlass.Float32) -> cutlass.Float32:
    # CuTe DSL exposes exp2 in cute.math; exp(x) = exp2(x * log2(e)).
    return cute.math.exp2(x * cutlass.Float32(LOG2_E))


@cute.jit
def _read_csa_current_packed(
    proj: cute.Tensor,
    token_idx: cutlass.Int64,
    col: cutlass.Int32,
    kind: cutlass.Constexpr,  # 0 Ca, 1 Za, 2 Cb, 3 Zb
    OUT: cutlass.Constexpr,
) -> cutlass.Float32:
    return proj[token_idx, kind * OUT + col].to(cutlass.Float32)


@cute.jit
def _read_hca_packed(
    proj: cute.Tensor,
    token_idx: cutlass.Int64,
    col: cutlass.Int32,
    kind: cutlass.Constexpr,  # 0 C, 1 Z
    C: cutlass.Constexpr,
) -> cutlass.Float32:
    return proj[token_idx, kind * C + col].to(cutlass.Float32)


# -----------------------------------------------------------------------------
# CSA fused compressor from packed projections
# -----------------------------------------------------------------------------

@cute.jit
def _csa_raw_reduce_one_col(
    proj: cute.Tensor,                 # (N_tokens, 4*OUT): Ca, Za, Cb, Zb
    prev_b_proj: cute.Tensor,           # (M, 2*OUT): Cb_prev, Zb_prev for first block, may be dummy
    B_a: cute.Tensor,                   # (M, OUT)
    B_b: cute.Tensor,                   # (M, OUT)
    block_i: cutlass.Int64,
    col: cutlass.Int32,
    start_token_in_proj: cutlass.Int64,
    has_external_prev: cutlass.Constexpr,
    M: cutlass.Constexpr,
    OUT: cutlass.Constexpr,
) -> cutlass.Float32:
    """Column-wise CSA softmax+weighted-sum for either main KV or indexer.

    This implements exactly:
      softmax([Z_a_cur + B_a ; Z_b_prev + B_b], dim=position)
      sum S_a*C_a + sum S_b*C_b
    with the block-0 no-prev case reducing over M current positions only.
    """
    # First pass: max logit.
    max_logit = cutlass.Float32(-3.4028234663852886e38)

    # Current/current-a side always exists.
    for p in cutlass.range_constexpr(M):
        tok = start_token_in_proj + block_i * M + p
        za = _read_csa_current_packed(proj, tok, col, 1, OUT) + B_a[p, col].to(cutlass.Float32)
        max_logit = cute.math.fmax(max_logit, za)

    # Previous/b side exists if this is not the first fresh block.
    use_prev = (block_i > 0) or has_external_prev
    if use_prev:
        for p in cutlass.range_constexpr(M):
            if block_i > 0:
                tok_prev = start_token_in_proj + (block_i - 1) * M + p
                zb = _read_csa_current_packed(proj, tok_prev, col, 3, OUT)
            else:
                # External previous block is packed as [Cb_prev | Zb_prev]
                zb = prev_b_proj[p, OUT + col].to(cutlass.Float32)
            zb = zb + B_b[p, col].to(cutlass.Float32)
            max_logit = cute.math.fmax(max_logit, zb)

    # Second pass: exp denominator and weighted value.
    denom = cutlass.Float32(0.0)
    acc = cutlass.Float32(0.0)

    for p in cutlass.range_constexpr(M):
        tok = start_token_in_proj + block_i * M + p
        za = _read_csa_current_packed(proj, tok, col, 1, OUT) + B_a[p, col].to(cutlass.Float32)
        ca = _read_csa_current_packed(proj, tok, col, 0, OUT)
        e = _expf(za - max_logit)
        denom += e
        acc += e * ca

    if use_prev:
        for p in cutlass.range_constexpr(M):
            if block_i > 0:
                tok_prev = start_token_in_proj + (block_i - 1) * M + p
                cb = _read_csa_current_packed(proj, tok_prev, col, 2, OUT)
                zb = _read_csa_current_packed(proj, tok_prev, col, 3, OUT)
            else:
                cb = prev_b_proj[p, col].to(cutlass.Float32)
                zb = prev_b_proj[p, OUT + col].to(cutlass.Float32)
            zb = zb + B_b[p, col].to(cutlass.Float32)
            e = _expf(zb - max_logit)
            denom += e
            acc += e * cb

    return acc / denom


@cute.kernel
def csa_compress_projected_kernel(
    proj_main: cute.Tensor,             # (N_tokens, 4*C)
    proj_indexer: cute.Tensor,          # (N_tokens, 4*C_I)
    prev_main_b: cute.Tensor,           # (M, 2*C), Cb_prev|Zb_prev; dummy if no ext prev
    prev_indexer_b: cute.Tensor,        # (M, 2*C_I), Cb_prev|Zb_prev; dummy if no ext prev
    B_a: cute.Tensor,                   # (M, C)
    B_b: cute.Tensor,                   # (M, C)
    B_I_a: cute.Tensor,                 # (M, C_I)
    B_I_b: cute.Tensor,                 # (M, C_I)
    cos_sin_cache: cute.Tensor,         # (max_pos, ROPE), cos first half, sin second half
    kv_out: cute.Tensor,                # (n_blocks, C)
    indexer_out: cute.Tensor,           # (n_blocks, C_I)
    n_blocks: cutlass.Int64,
    start_token_in_proj: cutlass.Int64,
    start_abs_pos: cutlass.Int64,
    has_external_prev: cutlass.Constexpr,
    M: cutlass.Constexpr,
    C: cutlass.Constexpr,
    C_I: cutlass.Constexpr,
    NOPE: cutlass.Constexpr,
    ROPE: cutlass.Constexpr,
    COLS_PER_CTA: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()

    block_i = bx.to(cutlass.Int64)
    base_col = by * COLS_PER_CTA
    col = base_col + tx

    # bz == 0: main KV output with RoPE.  bz == 1: indexer output, no RoPE.
    if bz == 0:
        if block_i < n_blocks and col < C:
            # RoPE dims need even/odd pair.  Let even lane compute/store both.
            if col < NOPE:
                val = _csa_raw_reduce_one_col(
                    proj_main, prev_main_b, B_a, B_b,
                    block_i, col, start_token_in_proj, has_external_prev, M, C,
                )
                kv_out[block_i, col] = val.to(kv_out.element_type)
            else:
                rope_col = col - NOPE
                if (rope_col % 2) == 0:
                    # Compute pair and rotate by block end position.
                    x0 = _csa_raw_reduce_one_col(
                        proj_main, prev_main_b, B_a, B_b,
                        block_i, col, start_token_in_proj, has_external_prev, M, C,
                    )
                    x1 = _csa_raw_reduce_one_col(
                        proj_main, prev_main_b, B_a, B_b,
                        block_i, col + 1, start_token_in_proj, has_external_prev, M, C,
                    )
                    block_end_pos = start_abs_pos + block_i * M + (M - 1)
                    half_idx = rope_col // 2
                    cosv = cos_sin_cache[block_end_pos, half_idx].to(cutlass.Float32)
                    sinv = cos_sin_cache[block_end_pos, half_idx + ROPE // 2].to(cutlass.Float32)
                    kv_out[block_i, col] = (x0 * cosv - x1 * sinv).to(kv_out.element_type)
                    kv_out[block_i, col + 1] = (x0 * sinv + x1 * cosv).to(kv_out.element_type)
    else:
        if block_i < n_blocks and col < C_I:
            val_i = _csa_raw_reduce_one_col(
                proj_indexer, prev_indexer_b, B_I_a, B_I_b,
                block_i, col, start_token_in_proj, has_external_prev, M, C_I,
            )
            indexer_out[block_i, col] = val_i.to(indexer_out.element_type)


@cute.jit
def launch_csa_compress_projected(
    proj_main: cute.Tensor,
    proj_indexer: cute.Tensor,
    prev_main_b: cute.Tensor,
    prev_indexer_b: cute.Tensor,
    B_a: cute.Tensor,
    B_b: cute.Tensor,
    B_I_a: cute.Tensor,
    B_I_b: cute.Tensor,
    cos_sin_cache: cute.Tensor,
    kv_out: cute.Tensor,
    indexer_out: cute.Tensor,
    n_blocks: int,
    start_token_in_proj: int,
    start_abs_pos: int,
    has_external_prev: cutlass.Constexpr,
    stream: cuda.CUstream,
    M: cutlass.Constexpr = 4,
    C: cutlass.Constexpr = 512,
    C_I: cutlass.Constexpr = 128,
    NOPE: cutlass.Constexpr = 448,
    ROPE: cutlass.Constexpr = 64,
    COLS_PER_CTA: cutlass.Constexpr = 128,
):
    grid_y = _ceil_div(C, COLS_PER_CTA)  # enough for main; indexer just masks col < C_I
    csa_compress_projected_kernel(
        proj_main,
        proj_indexer,
        prev_main_b,
        prev_indexer_b,
        B_a,
        B_b,
        B_I_a,
        B_I_b,
        cos_sin_cache,
        kv_out,
        indexer_out,
        n_blocks,
        start_token_in_proj,
        start_abs_pos,
        has_external_prev,
        M,
        C,
        C_I,
        NOPE,
        ROPE,
        COLS_PER_CTA,
    ).launch(
        grid=[n_blocks, grid_y, 2],
        block=[COLS_PER_CTA, 1, 1],
        stream=stream,
    )


# -----------------------------------------------------------------------------
# HCA fused compressor from packed projections
# -----------------------------------------------------------------------------

@cute.jit
def _hca_raw_reduce_one_col(
    proj: cute.Tensor,                  # (N_tokens, 2*C): C, Z
    B: cute.Tensor,                     # (M, C)
    block_i: cutlass.Int64,
    col: cutlass.Int32,
    start_token_in_proj: cutlass.Int64,
    M: cutlass.Constexpr,
    C: cutlass.Constexpr,
) -> cutlass.Float32:
    max_logit = cutlass.Float32(-3.4028234663852886e38)

    for p in cutlass.range_constexpr(M):
        tok = start_token_in_proj + block_i * M + p
        z = _read_hca_packed(proj, tok, col, 1, C) + B[p, col].to(cutlass.Float32)
        max_logit = cute.math.fmax(max_logit, z)

    denom = cutlass.Float32(0.0)
    acc = cutlass.Float32(0.0)

    for p in cutlass.range_constexpr(M):
        tok = start_token_in_proj + block_i * M + p
        z = _read_hca_packed(proj, tok, col, 1, C) + B[p, col].to(cutlass.Float32)
        c = _read_hca_packed(proj, tok, col, 0, C)
        e = _expf(z - max_logit)
        denom += e
        acc += e * c

    return acc / denom


@cute.kernel
def hca_compress_projected_kernel(
    proj: cute.Tensor,                  # (N_tokens, 2*C)
    B: cute.Tensor,                     # (M, C)
    cos_sin_cache: cute.Tensor,         # (max_pos, ROPE)
    kv_out: cute.Tensor,                # (n_blocks, C)
    n_blocks: cutlass.Int64,
    start_token_in_proj: cutlass.Int64,
    start_abs_pos: cutlass.Int64,
    M: cutlass.Constexpr,
    C: cutlass.Constexpr,
    NOPE: cutlass.Constexpr,
    ROPE: cutlass.Constexpr,
    COLS_PER_CTA: cutlass.Constexpr,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    block_i = bx.to(cutlass.Int64)
    col = by * COLS_PER_CTA + tx

    if block_i < n_blocks and col < C:
        if col < NOPE:
            val = _hca_raw_reduce_one_col(proj, B, block_i, col, start_token_in_proj, M, C)
            kv_out[block_i, col] = val.to(kv_out.element_type)
        else:
            rope_col = col - NOPE
            if (rope_col % 2) == 0:
                x0 = _hca_raw_reduce_one_col(proj, B, block_i, col, start_token_in_proj, M, C)
                x1 = _hca_raw_reduce_one_col(proj, B, block_i, col + 1, start_token_in_proj, M, C)
                block_end_pos = start_abs_pos + block_i * M + (M - 1)
                half_idx = rope_col // 2
                cosv = cos_sin_cache[block_end_pos, half_idx].to(cutlass.Float32)
                sinv = cos_sin_cache[block_end_pos, half_idx + ROPE // 2].to(cutlass.Float32)
                kv_out[block_i, col] = (x0 * cosv - x1 * sinv).to(kv_out.element_type)
                kv_out[block_i, col + 1] = (x0 * sinv + x1 * cosv).to(kv_out.element_type)


@cute.jit
def launch_hca_compress_projected(
    proj: cute.Tensor,
    B: cute.Tensor,
    cos_sin_cache: cute.Tensor,
    kv_out: cute.Tensor,
    n_blocks: int,
    start_token_in_proj: int,
    start_abs_pos: int,
    stream: cuda.CUstream,
    M: cutlass.Constexpr = 128,
    C: cutlass.Constexpr = 512,
    NOPE: cutlass.Constexpr = 448,
    ROPE: cutlass.Constexpr = 64,
    COLS_PER_CTA: cutlass.Constexpr = 128,
):
    grid_y = _ceil_div(C, COLS_PER_CTA)
    hca_compress_projected_kernel(
        proj,
        B,
        cos_sin_cache,
        kv_out,
        n_blocks,
        start_token_in_proj,
        start_abs_pos,
        M,
        C,
        NOPE,
        ROPE,
        COLS_PER_CTA,
    ).launch(
        grid=[n_blocks, grid_y, 1],
        block=[COLS_PER_CTA, 1, 1],
        stream=stream,
    )


# -----------------------------------------------------------------------------
# PyTorch-side packing helpers
# -----------------------------------------------------------------------------


def pack_csa_main_weights(W_a_KV, W_a_Z, W_b_KV, W_b_Z):
    """Return W packed as [W_a_KV | W_a_Z | W_b_KV | W_b_Z]."""
    return torch.cat([W_a_KV, W_a_Z, W_b_KV, W_b_Z], dim=1).contiguous()


def pack_csa_indexer_weights(W_I_a_KV, W_I_a_Z, W_I_b_KV, W_I_b_Z):
    """Return W_I packed as [W_I_a_KV | W_I_a_Z | W_I_b_KV | W_I_b_Z]."""
    return torch.cat([W_I_a_KV, W_I_a_Z, W_I_b_KV, W_I_b_Z], dim=1).contiguous()


def pack_hca_weights(W_KV, W_Z):
    """Return W packed as [W_KV | W_Z]."""
    return torch.cat([W_KV, W_Z], dim=1).contiguous()


def make_dummy_prev_b(M: int, OUT: int, *, device, dtype):
    """Dummy external previous-block projection for fresh prefill."""
    return torch.empty((M, 2 * OUT), device=device, dtype=dtype)
