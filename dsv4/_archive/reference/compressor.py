"""
CSA / HCA Token-Level Compressor for DeepSeek-V4.

Implements Section 2.3 of the DeepSeek-V4 paper exactly:
  - CSA (m=4):  overlapping weighted sum over 2m hidden states per block
  - HCA (m'=128): non-overlapping weighted sum over m' hidden states per block

Both produce compressed KV entries C^Comp ∈ R^{n/m × c} where each entry
is a weighted sum of hidden states using softmax-normalised gate weights.

CSA additionally produces compressed indexer keys K^IComp ∈ R^{n/m × c_I}
for the Lightning Indexer (top-k sparse selection).

V4-Pro reference dimensions (Section 4.2.1):
  d  = 7168   hidden dim
  c  = 512    head dim (CSA and HCA both)
  m  = 4      CSA compression ratio
  m' = 128    HCA compression ratio
  c_I = 128   indexer head dim
  n_I_h = 64  num indexer query heads
  n_win = 128 sliding window size (separate, not handled here)
  rope_dim = 64  partial RoPE on last 64 dims of each head

Design notes
------------
* BF16 matmuls throughout — swap the _proj() calls for your NVFP4 GEMMs.
* No batch dimension: one sequence at a time (matching decode latency path).
* CompressorState carries the incomplete-block tail and the previous block's
  raw projections needed for the CSA overlap.
* Partial RoPE is applied to the last rope_dim=64 dims of C^Comp before
  the entry is stored in the compressed KV cache, using the representative
  position = last token of the block.  The inverse-RoPE on attention outputs
  is handled by your attention kernel (already in blackwell_attention.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class CompressorState:
    """
    Per-sequence mutable state for the compressor.

    tail_hidden  — hidden states that have arrived but don't yet fill a
                   complete compression block.  Shape (tail_len, d),
                   0 <= tail_len < m.
    prev_hidden  — the m hidden states from the previous complete block.
                   Needed for the CSA overlap (C^b / Z^b projections).
                   None before the first block is committed.
                   Not used by HCA (no overlap).
    compressed_kv        — accumulated C^Comp entries, shape (n_blocks, c).
    compressed_indexer_kv — accumulated K^IComp entries, shape (n_blocks, c_I).
                             None for HCA layers.
    """
    tail_hidden: Optional[torch.Tensor] = None           # (tail_len, d)
    prev_hidden: Optional[torch.Tensor] = None           # (m, d)  CSA only
    compressed_kv: Optional[torch.Tensor] = None         # (n_blocks, c)
    compressed_indexer_kv: Optional[torch.Tensor] = None # (n_blocks, c_I)
    num_blocks: int = 0

    def reset(self):
        self.tail_hidden = None
        self.prev_hidden = None
        self.compressed_kv = None
        self.compressed_indexer_kv = None
        self.num_blocks = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_partial_rope(
    x: torch.Tensor,          # (..., c)
    positions: torch.Tensor,  # (...,) int64
    cos_sin_cache: torch.Tensor,  # (max_pos, rope_dim)
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """GPT-J style RoPE on the last rope_dim dimensions only."""
    if rope_dim == 0:
        return x
    half = rope_dim // 2
    cos = cos_sin_cache[positions, :half].to(x.dtype)   # (..., half)
    sin = cos_sin_cache[positions, half:].to(x.dtype)   # (..., half)
    out = x.clone()
    rope_part = out[..., nope_dim:]                      # (..., rope_dim)
    even = rope_part[..., 0::2]
    odd  = rope_part[..., 1::2]
    out[..., nope_dim:][..., 0::2] = even * cos - odd * sin
    out[..., nope_dim:][..., 1::2] = even * sin + odd * cos
    return out


def _proj(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Linear projection: x @ W.
    x: (..., d)   W: (d, out_dim)  →  (..., out_dim)

    *** SWAP THIS FOR YOUR NVFP4 GEMM. ***
    The W tensor would become your NVFP4 weight + scale_b + gsb triple,
    and x would be quantised to NVFP4 activation before the call.
    """
    return x.to(W.dtype) @ W


# ---------------------------------------------------------------------------
# CSA compressor
# ---------------------------------------------------------------------------

class CSACompressor:
    """
    Compressed Sparse Attention token-level compressor.

    Paper equations (11) and (12), Section 2.3.1:

      C^a = H · W^a_KV        Z^a = H · W^a_Z          (current block)
      C^b = H · W^b_KV        Z^b = H · W^b_Z          (prev-block overlap)

    For compressed block i  (0-indexed):
      [S^a ; S^b] = softmax_row( [Z^a_{cur} + B^a ; Z^b_{prev} + B^b] )
      C^Comp_i   = Σ S^a_j ⊙ C^a_j   +   Σ S^b_j ⊙ C^b_j

    When i=0: Z^b / C^b are padded with -inf / 0 so only Z^a contributes.

    The same compression is applied independently to produce indexer keys,
    using separate projections W^I_KV, W^I_Z, B^I_a, B^I_b.
    """

    def __init__(
        self,
        hidden_dim: int  = 7168,   # d
        head_dim: int    = 512,    # c
        compress_ratio: int = 4,   # m
        indexer_head_dim: int = 128,   # c_I
        num_indexer_heads: int = 64,   # n_I_h  (not used in compressor itself)
        nope_dim: int = 448,       # c - rope_dim = 512 - 64
        rope_dim: int = 64,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.d      = hidden_dim
        self.c      = head_dim
        self.m      = compress_ratio
        self.c_I    = indexer_head_dim
        self.n_I_h  = num_indexer_heads
        self.nope   = nope_dim
        self.rope   = rope_dim
        self.device = device
        self.dtype  = dtype

        # ── Main KV projection weights ──────────────────────────────
        # W^a_{KV}, W^b_{KV}: (d, c)
        # W^a_Z,    W^b_Z:    (d, c)
        self.W_a_KV = self._param(hidden_dim, head_dim)
        self.W_b_KV = self._param(hidden_dim, head_dim)
        self.W_a_Z  = self._param(hidden_dim, head_dim)
        self.W_b_Z  = self._param(hidden_dim, head_dim)

        # Positional biases B^a, B^b: (m, c)  — learnable per-position offsets
        # added to the gate logits before softmax.
        self.B_a = self._param(compress_ratio, head_dim)
        self.B_b = self._param(compress_ratio, head_dim)

        # ── Indexer key projection weights ──────────────────────────
        # Same overlap structure, separate projections, output dim c_I.
        self.W_I_a_KV = self._param(hidden_dim, indexer_head_dim)
        self.W_I_b_KV = self._param(hidden_dim, indexer_head_dim)
        self.W_I_a_Z  = self._param(hidden_dim, indexer_head_dim)
        self.W_I_b_Z  = self._param(hidden_dim, indexer_head_dim)
        self.B_I_a = self._param(compress_ratio, indexer_head_dim)
        self.B_I_b = self._param(compress_ratio, indexer_head_dim)

    def _param(self, *shape) -> torch.Tensor:
        """Uninitialised placeholder — replace with checkpoint-loaded tensor."""
        return torch.empty(*shape, dtype=self.dtype, device=self.device)

    def load_weights(
        self,
        W_a_KV, W_b_KV, W_a_Z, W_b_Z, B_a, B_b,
        W_I_a_KV, W_I_b_KV, W_I_a_Z, W_I_b_Z, B_I_a, B_I_b,
    ):
        """Assign weights from checkpoint. All tensors moved to device/dtype."""
        def _cvt(t): return t.to(device=self.device, dtype=self.dtype)
        self.W_a_KV  = _cvt(W_a_KV);  self.W_b_KV  = _cvt(W_b_KV)
        self.W_a_Z   = _cvt(W_a_Z);   self.W_b_Z   = _cvt(W_b_Z)
        self.B_a     = _cvt(B_a);     self.B_b     = _cvt(B_b)
        self.W_I_a_KV = _cvt(W_I_a_KV); self.W_I_b_KV = _cvt(W_I_b_KV)
        self.W_I_a_Z  = _cvt(W_I_a_Z);  self.W_I_b_Z  = _cvt(W_I_b_Z)
        self.B_I_a   = _cvt(B_I_a);   self.B_I_b   = _cvt(B_I_b)

    # ----------------------------------------------------------------
    # Core: compress one block of m hidden states
    # ----------------------------------------------------------------

    def _compress_block(
        self,
        cur_hidden: torch.Tensor,          # (m, d)  current block
        prev_hidden: Optional[torch.Tensor],  # (m, d) or None if block 0
        cos_sin_cache: Optional[torch.Tensor],
        block_end_pos: int,                # position of last token in block
        for_indexer: bool = False,
    ) -> torch.Tensor:
        """
        Compress one block into a single C^Comp entry.

        Returns: (c,) or (c_I,) compressed entry.

        The overlap computation (equations 11-12):
          Z_cat = [Z^a + B^a ; Z^b + B^b]   shape (2m, c) or (m, c) when block 0
          S     = softmax(Z_cat, dim=0)       normalise over the 2m position axis
          C_out = (S[:m] * C^a).sum(0) + (S[m:] * C^b).sum(0)
        """
        m = self.m
        assert cur_hidden.shape[0] == m

        if for_indexer:
            W_a_KV, W_b_KV = self.W_I_a_KV, self.W_I_b_KV
            W_a_Z,  W_b_Z  = self.W_I_a_Z,  self.W_I_b_Z
            B_a, B_b       = self.B_I_a,     self.B_I_b
        else:
            W_a_KV, W_b_KV = self.W_a_KV, self.W_b_KV
            W_a_Z,  W_b_Z  = self.W_a_Z,  self.W_b_Z
            B_a, B_b       = self.B_a,    self.B_b

        # Current block projections: (m, c)
        C_a = _proj(cur_hidden, W_a_KV)   # KV candidates
        Z_a = _proj(cur_hidden, W_a_Z)    # gate logits

        if prev_hidden is None:
            # Block 0: no previous block → softmax over m entries only
            # (paper pads Z^b with -inf, C^b with 0)
            Z_cat = Z_a + B_a                # (m, c)
            S     = F.softmax(Z_cat.float(), dim=0).to(self.dtype)  # (m, c)
            C_out = (S * C_a).sum(dim=0)     # (c,)
        else:
            # Blocks 1..: overlap with previous block
            C_b = _proj(prev_hidden, W_b_KV)  # (m, c)
            Z_b = _proj(prev_hidden, W_b_Z)   # (m, c)

            # Concatenate along the position axis → (2m, c)
            Z_cat = torch.cat([Z_a + B_a, Z_b + B_b], dim=0)
            S     = F.softmax(Z_cat.float(), dim=0).to(self.dtype)  # (2m, c)

            S_a, S_b = S[:m], S[m:]           # each (m, c)
            C_out = (S_a * C_a).sum(dim=0) + (S_b * C_b).sum(dim=0)  # (c,)

        # Partial RoPE on the last rope_dim dims using the block's end position
        if cos_sin_cache is not None and self.rope > 0 and not for_indexer:
            pos_t = torch.tensor([block_end_pos], dtype=torch.long,
                                 device=cur_hidden.device)
            C_out = _apply_partial_rope(
                C_out.unsqueeze(0), pos_t, cos_sin_cache,
                self.nope, self.rope
            ).squeeze(0)

        return C_out  # (c,) or (c_I,)

    # ----------------------------------------------------------------
    # Prefill: all n tokens at once
    # ----------------------------------------------------------------

    def prefill(
        self,
        hidden: torch.Tensor,              # (n, d)
        cos_sin_cache: Optional[torch.Tensor],  # (max_pos, rope_dim)
        start_pos: int = 0,
        state: Optional[CompressorState] = None,
    ) -> CompressorState:
        """
        Process all n tokens in one shot (prefill / context ingestion).

        Tokens that don't fill a complete block of m are stored in
        state.tail_hidden for future incremental decode steps.

        Returns an updated CompressorState.
        """
        if state is None:
            state = CompressorState()

        n, d = hidden.shape
        m    = self.m

        # If there are tail tokens from a previous call, prepend them
        if state.tail_hidden is not None and state.tail_hidden.shape[0] > 0:
            hidden = torch.cat([state.tail_hidden, hidden], dim=0)
            # The positions need to be adjusted accordingly
            n = hidden.shape[0]

        n_complete_blocks = n // m
        n_tail            = n % m

        kv_list       = []
        indexer_kv_list = []

        prev_hidden = state.prev_hidden  # None on first ever call

        for i in range(n_complete_blocks):
            cur   = hidden[i * m : (i + 1) * m]       # (m, d)
            # Absolute position of the last token in this block
            block_end = start_pos + i * m + (m - 1)

            c_kv = self._compress_block(
                cur, prev_hidden, cos_sin_cache, block_end, for_indexer=False
            )
            c_I = self._compress_block(
                cur, prev_hidden, None, block_end, for_indexer=True
            )

            kv_list.append(c_kv)
            indexer_kv_list.append(c_I)
            prev_hidden = cur   # this block becomes the "previous" for the next

        # Accumulate into state
        new_kv = torch.stack(kv_list, dim=0) if kv_list else None  # (n_blocks, c)
        new_I  = torch.stack(indexer_kv_list, dim=0) if indexer_kv_list else None

        if state.compressed_kv is None:
            state.compressed_kv         = new_kv
            state.compressed_indexer_kv = new_I
        elif new_kv is not None:
            state.compressed_kv         = torch.cat([state.compressed_kv, new_kv], dim=0)
            state.compressed_indexer_kv = torch.cat([state.compressed_indexer_kv, new_I], dim=0)

        state.num_blocks   += n_complete_blocks
        state.prev_hidden   = prev_hidden
        state.tail_hidden   = hidden[n_complete_blocks * m :] if n_tail > 0 else None

        return state

    # ----------------------------------------------------------------
    # Decode: single new token, incremental
    # ----------------------------------------------------------------

    def decode_step(
        self,
        hidden_new: torch.Tensor,          # (1, d)  or (d,)
        cos_sin_cache: Optional[torch.Tensor],
        current_pos: int,
        state: CompressorState,
    ) -> tuple[CompressorState, bool]:
        """
        Ingest one new token into the state.

        Returns (updated_state, new_block_committed).

        new_block_committed is True when the new token completes a block
        and a new compressed entry has been appended to state.compressed_kv.
        The caller only needs to re-run the Lightning Indexer when True.
        """
        h = hidden_new.reshape(1, self.d)

        # Append to tail
        if state.tail_hidden is None:
            state.tail_hidden = h
        else:
            state.tail_hidden = torch.cat([state.tail_hidden, h], dim=0)

        tail_len = state.tail_hidden.shape[0]

        if tail_len < self.m:
            # Block not yet complete — nothing to compress
            return state, False

        # Tail is exactly m tokens — compress
        assert tail_len == self.m, f"tail_len={tail_len} should equal m={self.m}"
        cur         = state.tail_hidden          # (m, d)
        block_end   = current_pos               # last token in block

        c_kv = self._compress_block(
            cur, state.prev_hidden, cos_sin_cache, block_end, for_indexer=False
        )
        c_I = self._compress_block(
            cur, state.prev_hidden, None, block_end, for_indexer=True
        )

        # Append to accumulated KV
        c_kv_2d = c_kv.unsqueeze(0)  # (1, c)
        c_I_2d  = c_I.unsqueeze(0)   # (1, c_I)

        if state.compressed_kv is None:
            state.compressed_kv         = c_kv_2d
            state.compressed_indexer_kv = c_I_2d
        else:
            state.compressed_kv         = torch.cat([state.compressed_kv, c_kv_2d], dim=0)
            state.compressed_indexer_kv = torch.cat([state.compressed_indexer_kv, c_I_2d], dim=0)

        state.num_blocks  += 1
        state.prev_hidden  = cur          # save for next block's overlap
        state.tail_hidden  = None         # clear tail

        return state, True


# ---------------------------------------------------------------------------
# HCA compressor
# ---------------------------------------------------------------------------

class HCACompressor:
    """
    Heavily Compressed Attention token-level compressor.

    Paper Section 2.3.2.  Simpler than CSA:
      - No overlap: each block of m' tokens is self-contained.
      - No indexer: HCA uses dense MQA over all compressed entries,
        so no top-k selection is needed and there are no indexer keys.

    For compressed block i:
      C    = H · W_KV                      (n, c)
      Z    = H · W_Z                       (n, c)
      S_i  = softmax_row(Z[m'i:m'(i+1)] + B)   (m', c)
      C^Comp_i = (S_i * C[m'i:m'(i+1)]).sum(0)  (c,)
    """

    def __init__(
        self,
        hidden_dim: int  = 7168,
        head_dim: int    = 512,
        compress_ratio: int = 128,   # m'
        nope_dim: int = 448,
        rope_dim: int = 64,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.d    = hidden_dim
        self.c    = head_dim
        self.m    = compress_ratio      # m' in the paper
        self.nope = nope_dim
        self.rope = rope_dim
        self.device = device
        self.dtype  = dtype

        # W_KV: (d, c)   W_Z: (d, c)
        self.W_KV = torch.empty(hidden_dim, head_dim, dtype=dtype, device=device)
        self.W_Z  = torch.empty(hidden_dim, head_dim, dtype=dtype, device=device)
        # Positional bias B: (m', c)
        self.B    = torch.empty(compress_ratio, head_dim, dtype=dtype, device=device)

    def load_weights(self, W_KV, W_Z, B):
        def _cvt(t): return t.to(device=self.device, dtype=self.dtype)
        self.W_KV = _cvt(W_KV)
        self.W_Z  = _cvt(W_Z)
        self.B    = _cvt(B)

    def _compress_block(
        self,
        block_hidden: torch.Tensor,        # (m', d)
        cos_sin_cache: Optional[torch.Tensor],
        block_end_pos: int,
    ) -> torch.Tensor:
        """Compress one block of m' tokens → (c,)."""
        m = self.m
        assert block_hidden.shape[0] == m

        C = _proj(block_hidden, self.W_KV)    # (m', c)
        Z = _proj(block_hidden, self.W_Z)     # (m', c)

        S = F.softmax((Z + self.B).float(), dim=0).to(self.dtype)  # (m', c)
        C_out = (S * C).sum(dim=0)            # (c,)

        if cos_sin_cache is not None and self.rope > 0:
            pos_t = torch.tensor([block_end_pos], dtype=torch.long,
                                 device=block_hidden.device)
            C_out = _apply_partial_rope(
                C_out.unsqueeze(0), pos_t, cos_sin_cache,
                self.nope, self.rope
            ).squeeze(0)

        return C_out

    def prefill(
        self,
        hidden: torch.Tensor,              # (n, d)
        cos_sin_cache: Optional[torch.Tensor],
        start_pos: int = 0,
        state: Optional[CompressorState] = None,
    ) -> CompressorState:
        """Process all n tokens (prefill). Tail stored for later decode."""
        if state is None:
            state = CompressorState()

        n = hidden.shape[0]

        if state.tail_hidden is not None and state.tail_hidden.shape[0] > 0:
            hidden = torch.cat([state.tail_hidden, hidden], dim=0)
            n = hidden.shape[0]

        m          = self.m
        n_complete = n // m
        n_tail     = n % m

        kv_list = []
        for i in range(n_complete):
            block     = hidden[i * m : (i + 1) * m]
            block_end = start_pos + i * m + (m - 1)
            kv_list.append(self._compress_block(block, cos_sin_cache, block_end))

        new_kv = torch.stack(kv_list, dim=0) if kv_list else None

        if state.compressed_kv is None:
            state.compressed_kv = new_kv
        elif new_kv is not None:
            state.compressed_kv = torch.cat([state.compressed_kv, new_kv], dim=0)

        state.num_blocks  += n_complete
        state.tail_hidden  = hidden[n_complete * m :] if n_tail > 0 else None
        # HCA has no prev_hidden needed (no overlap)

        return state

    def decode_step(
        self,
        hidden_new: torch.Tensor,          # (1, d)
        cos_sin_cache: Optional[torch.Tensor],
        current_pos: int,
        state: CompressorState,
    ) -> tuple[CompressorState, bool]:
        """Ingest one token. Returns (state, new_block_committed)."""
        h = hidden_new.reshape(1, self.d)

        state.tail_hidden = h if state.tail_hidden is None else \
                            torch.cat([state.tail_hidden, h], dim=0)

        if state.tail_hidden.shape[0] < self.m:
            return state, False

        # Full block ready
        block   = state.tail_hidden               # (m', d)
        c_kv    = self._compress_block(block, cos_sin_cache, current_pos)
        c_kv_2d = c_kv.unsqueeze(0)

        state.compressed_kv = c_kv_2d if state.compressed_kv is None else \
                               torch.cat([state.compressed_kv, c_kv_2d], dim=0)

        state.num_blocks  += 1
        state.tail_hidden  = None

        return state, True


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16

    # ── V4-Pro dims ─────────────────────────────────────────────────
    D, C, M, C_I   = 7168, 512, 4, 128
    M_HCA          = 128
    NOPE, ROPE     = 448, 64
    MAX_POS        = 4096

    cos_sin_cache = torch.randn(MAX_POS, ROPE, dtype=dtype, device=device)

    # ── CSA ─────────────────────────────────────────────────────────
    print("=== CSA compressor ===")
    csa = CSACompressor(D, C, M, C_I, num_indexer_heads=64,
                        nope_dim=NOPE, rope_dim=ROPE, device=device, dtype=dtype)

    # Prefill 20 tokens (5 complete blocks, 0 tail)
    n_prefill = 20
    h_prefill = torch.randn(n_prefill, D, dtype=dtype, device=device)
    state = csa.prefill(h_prefill, cos_sin_cache, start_pos=0)

    print(f"  After prefill {n_prefill} tokens:")
    print(f"    num_blocks:          {state.num_blocks}")           # 5
    print(f"    compressed_kv:       {state.compressed_kv.shape}")  # (5, 512)
    print(f"    indexer_kv:          {state.compressed_indexer_kv.shape}")  # (5, 128)
    print(f"    tail_len:            {0 if state.tail_hidden is None else state.tail_hidden.shape[0]}")

    # Prefill 6 more (1 complete block + 2 tail)
    h2 = torch.randn(6, D, dtype=dtype, device=device)
    state = csa.prefill(h2, cos_sin_cache, start_pos=n_prefill, state=state)
    print(f"\n  After prefill 6 more:")
    print(f"    num_blocks:          {state.num_blocks}")           # 6
    print(f"    compressed_kv:       {state.compressed_kv.shape}")  # (6, 512)
    print(f"    tail_len:            {state.tail_hidden.shape[0]}") # 2

    # Decode 2 tokens (fills tail → 1 new block)
    for tok_i in range(2):
        h_tok = torch.randn(1, D, dtype=dtype, device=device)
        pos   = n_prefill + 6 + tok_i
        state, committed = csa.decode_step(h_tok, cos_sin_cache, pos, state)
        print(f"    decode tok {tok_i}: committed={committed}")
    print(f"    num_blocks now:      {state.num_blocks}")           # 7

    # ── HCA ─────────────────────────────────────────────────────────
    print("\n=== HCA compressor ===")
    hca = HCACompressor(D, C, M_HCA, nope_dim=NOPE, rope_dim=ROPE,
                        device=device, dtype=dtype)

    n_hca = 384                        # 3 complete blocks + 0 tail
    h_hca = torch.randn(n_hca, D, dtype=dtype, device=device)
    hca_state = hca.prefill(h_hca, cos_sin_cache, start_pos=0)
    print(f"  After prefill {n_hca} tokens:")
    print(f"    num_blocks:          {hca_state.num_blocks}")        # 3
    print(f"    compressed_kv:       {hca_state.compressed_kv.shape}")  # (3, 512)

    # Decode 128 tokens → exactly 1 new HCA block
    for tok_i in range(M_HCA):
        h_tok = torch.randn(1, D, dtype=dtype, device=device)
        pos   = n_hca + tok_i
        hca_state, committed = hca.decode_step(h_tok, cos_sin_cache, pos, hca_state)

    print(f"  After decode {M_HCA} tokens:")
    print(f"    num_blocks:          {hca_state.num_blocks}")        # 4
    print(f"    compressed_kv:       {hca_state.compressed_kv.shape}")  # (4, 512)

    # ── Correctness sanity: prefill == incremental decode ───────────
    print("\n=== Equivalence check: prefill vs incremental decode ===")
    csa2 = CSACompressor(D, C, M, C_I, nope_dim=NOPE, rope_dim=ROPE,
                         device=device, dtype=dtype)
    # Copy weights
    for attr in ("W_a_KV","W_b_KV","W_a_Z","W_b_Z","B_a","B_b",
                 "W_I_a_KV","W_I_b_KV","W_I_a_Z","W_I_b_Z","B_I_a","B_I_b"):
        setattr(csa2, attr, getattr(csa, attr))

    n_check = 8
    h_check = torch.randn(n_check, D, dtype=dtype, device=device)

    # Batch prefill
    s_batch = csa2.prefill(h_check, cos_sin_cache, start_pos=0)

    # Token-by-token decode
    s_incr = CompressorState()
    for i in range(n_check):
        s_incr, _ = csa2.decode_step(h_check[i:i+1], cos_sin_cache, i, s_incr)

    if s_batch.compressed_kv is not None and s_incr.compressed_kv is not None:
        max_diff = (s_batch.compressed_kv - s_incr.compressed_kv).abs().max().item()
        print(f"  max |prefill - decode| on compressed_kv: {max_diff:.6f}")
        assert max_diff < 1e-3, "Mismatch between prefill and decode paths!"
        print("  PASSED")
    else:
        print("  (no complete blocks produced in 8 tokens with m=4 — increase n_check)")

    print("\nAll checks done.")
