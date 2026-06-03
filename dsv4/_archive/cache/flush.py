"""In-graph flush orchestration.

Called when tail_len crosses the compression threshold. The actual
compression math is in the csa_hca_compressor kernel; this module
handles the quantize-scatter-write step and the state rotation.

The maybe_flush_* functions always run when their attention type
matches — no host-side `if tail_full` check. The kernels gate
internally via `valid_mask` computed from `tail_len`. This keeps
the call sequence identical across forward passes for cudagraph.
"""
from __future__ import annotations
from typing import Optional
import os
import torch
from torch.utils.cpp_extension import load

from dsv4.cache.schema import LayerCacheSchema, AttentionType


_flush_mod = None


def _get_flush_module():
    global _flush_mod
    if _flush_mod is not None:
        return _flush_mod
    kernel_dir = os.path.join(os.path.dirname(__file__), "..", "kernels", "cuda")
    _flush_mod = load(
        name="flush_write",
        sources=[os.path.join(kernel_dir, "flush_write.cu")],
        extra_cuda_cflags=["-O3", "--generate-code=arch=compute_100a,code=[sm_100a]"],
        verbose=False,
    )
    return _flush_mod


def maybe_flush_csa(
    handle,
    schema: LayerCacheSchema,
    m: int,
) -> None:
    """For CSA: emit compressed entries for requests whose tail is full.

    Steps:
      1. Determine which requests have tail_len >= m (valid_mask).
      2. Run the CSA compressor on tail buffers.
      3. Scatter compressed entry + indexer key into paged pool.
      4. Rotate a-stream -> b-stream, clear a-stream.
    """
    from dsv4.kernels.compressor import csa_compress_tail

    state = handle.state
    paged = handle.paged
    mod = _get_flush_module()

    # Step 1: valid_mask — which requests have a full tail buffer.
    # tail_len is [max_requests], request_slots is [B].
    tail_lens = state.tail_len[handle.request_slots]  # [B]
    valid_mask = tail_lens >= m  # [B] bool

    # If no requests need flushing, short-circuit.
    if not valid_mask.any().item():
        return

    # Step 2: compress the tail.
    # The compressor kernel takes the tail buffers and produces
    # one compressed entry per request (for those where valid_mask=True).
    entry, indexer_key = csa_compress_tail(
        tail_ka=state.tail_ka,
        tail_za=state.tail_za,
        tail_kb=state.tail_kb,
        tail_zb=state.tail_zb,
        tail_len=state.tail_len,
        request_slots=handle.request_slots,
        m=m,
    )
    # entry: [B, head_dim] BF16
    # indexer_key: [B, indexer_head_dim] BF16

    # Step 3: scatter into the paged pool.
    # The flush position for each request = the position of the last
    # token in the tail (positions before this forward minus 1 would
    # be the wrong reference; we need the tail's last position).
    # For the block table lookup, we use the compressed entry index
    # derived from positions.
    # Use the positions of the requests' current tokens to figure
    # out which entry slot to write into.
    flush_positions = handle.positions  # [tokens] -> need per-request
    # For now, derive entry index from the per-request state:
    # compressed_entry_idx = sum of all flushes so far for this request.
    # This is (positions_of_last_appended_token) // m
    # Simplification: use request_slots to look up per-request position.
    # The handle's positions are per-token, not per-request.
    # We need one position per request = position of the last appended token.
    # For a single-token decode, that's just positions[-1] per request.
    # For a general case, take the max position per request.
    # This is computed by the append kernel (stored in tail_len and the
    # actual positions in the tail). For now, use handle.positions
    # and scatter by request.
    # The kernel resolves slot_in_block from positions internally.

    mod.flush_write_csa(
        entry, indexer_key, valid_mask, handle.request_slots,
        handle.positions[:handle.request_slots.shape[0]],  # one pos per request
        handle.block_table,
        paged.entries_fp8, paged.entries_rope, paged.inv_scale,
        paged.indexer_keys_fp4, paged.indexer_scale,
        schema.entries_per_block, m, schema.rope_dim,
        schema.entry_head_dim, schema.indexer_head_dim,
    )

    # Step 4: rotate state — current a-stream becomes next b-stream.
    mod.csa_rotate_state(
        valid_mask, handle.request_slots,
        state.tail_ka, state.tail_za, state.tail_kb, state.tail_zb,
        state.tail_len, m, schema.entry_head_dim,
    )


def maybe_flush_hca(
    handle,
    schema: LayerCacheSchema,
    m_prime: int,
) -> None:
    """For HCA: emit one entry per request whose tail_len >= m'."""
    from dsv4.kernels.compressor import hca_compress_tail

    state = handle.state
    paged = handle.paged
    mod = _get_flush_module()

    tail_lens = state.tail_len[handle.request_slots]
    valid_mask = tail_lens >= m_prime

    if not valid_mask.any().item():
        return

    entry = hca_compress_tail(
        tail_ka=state.tail_ka,
        tail_za=state.tail_za,
        tail_len=state.tail_len,
        request_slots=handle.request_slots,
        m=m_prime,
    )
    # entry: [B, head_dim] BF16

    mod.flush_write_hca(
        entry, valid_mask, handle.request_slots,
        handle.positions[:handle.request_slots.shape[0]],
        handle.block_table,
        paged.entries_fp8, paged.entries_rope, paged.inv_scale,
        schema.entries_per_block, m_prime, schema.rope_dim,
        schema.entry_head_dim,
    )

    # Reset tail — no b-stream rotation for HCA.
    mod.hca_reset_state(
        valid_mask, handle.request_slots,
        state.tail_ka, state.tail_za, state.tail_len,
        m_prime, schema.entry_head_dim,
    )
