"""State cache: SWA window + uncompressed tail buffer.

One slot per active request. Slot index is fixed for a request's
lifetime — the manager hands out slot indices at request admission
and reclaims them at completion.

Per paper §3.5.1: SWA and tail tokens are state-space-like — they
depend only on the current position, not on a paged history. No
block table; a flat [max_requests, ...] tensor.

CSA b-stream lifecycle (paper eq.11-12):
  After a CSA flush, the current a-stream (tail_ka/tail_za) becomes
  the next flush's b-stream input (tail_kb/tail_zb). Both are sized
  at m entries, not m-1. On first flush, tail_zb is filled with -1e9
  so the softmax in the compressor naturally masks out the b-stream
  (exp(-inf) = 0).
"""
from __future__ import annotations
import torch

from dsv4.cache.schema import LayerCacheSchema, AttentionType


class StateCachePool:
    """Per-layer state cache (SWA window + uncompressed tail).

    Storage layout per slot:
      swa_fp8:   [n_win, head_dim - rope_dim] FP8 raw KV in window
      swa_rope:  [n_win, rope_dim]             BF16 RoPE'd half
      swa_inv:   [n_win]                       FP32 per-token inv scale
      swa_pos:   [n_win]                       int32 — absolute position
      swa_head:  scalar int32 — ring buffer write head

      tail_ka:   [m_a, head_dim]   BF16 — current a-stream tokens
      tail_za:   [m_a, head_dim]   BF16 — current a-stream Z weights
      tail_kb:   [m_b, head_dim]   BF16 — previous a-stream kept as b-input (CSA only)
      tail_zb:   [m_b, head_dim]   BF16 — previous Z b-stream (CSA only, init to -1e9)
      tail_len:  scalar int32 — how many entries in a-stream are valid
    """

    def __init__(
        self,
        schema: LayerCacheSchema,
        max_requests: int,
        device: str = "cuda",
    ):
        self.schema = schema
        self.max_requests = max_requests
        self.device = device

        mr = max_requests
        nw = schema.swa_window_size
        hd = schema.entry_head_dim
        rd = schema.rope_dim
        fp8 = hd - rd

        # SWA window — circular within each slot.
        self.swa_fp8 = torch.zeros((mr, nw, fp8), dtype=torch.uint8, device=device)
        self.swa_rope = torch.zeros((mr, nw, rd), dtype=torch.bfloat16, device=device)
        self.swa_inv = torch.ones((mr, nw), dtype=torch.float32, device=device)
        self.swa_pos = torch.full((mr, nw), -1, dtype=torch.int32, device=device)
        self.swa_head = torch.zeros((mr,), dtype=torch.int32, device=device)

        # Tail buffer — only for compressed layers.
        m_a = schema.tail_buffer_size_a  # m (CSA) or m' (HCA)
        m_b = schema.tail_buffer_size_b  # m (CSA only)
        if m_a > 0:
            self.tail_ka = torch.zeros((mr, m_a, hd), dtype=torch.bfloat16, device=device)
            self.tail_za = torch.zeros((mr, m_a, hd), dtype=torch.bfloat16, device=device)
            self.tail_len = torch.zeros((mr,), dtype=torch.int32, device=device)
            if m_b > 0:  # CSA: need b-stream
                self.tail_kb = torch.zeros((mr, m_b, hd), dtype=torch.bfloat16, device=device)
                # Paper §3.5.1: Z^b padded with -inf at first flush.
                # Init to -1e9 so softmax naturally masks b-stream on first flush.
                self.tail_zb = torch.full((mr, m_b, hd), -1e9, dtype=torch.bfloat16, device=device)
            else:
                self.tail_kb = None
                self.tail_zb = None
        else:
            self.tail_ka = self.tail_za = None
            self.tail_kb = self.tail_zb = None
            self.tail_len = None

    def reset_slot(self, slot: int) -> None:
        """Clear a request's state after completion."""
        self.swa_pos[slot].fill_(-1)
        self.swa_head[slot] = 0
        if self.tail_len is not None:
            self.tail_len[slot] = 0
        # Re-init tail_zb to -1e9 for CSA (paper §3.5.1 first-flush mask)
        if self.tail_zb is not None:
            self.tail_zb[slot].fill_(-1e9)

    def memory_bytes(self) -> int:
        """Total GPU memory used by this pool."""
        total = 0
        for name in ("swa_fp8", "swa_rope", "swa_inv", "swa_pos", "swa_head",
                      "tail_ka", "tail_za", "tail_kb", "tail_zb", "tail_len"):
            t = getattr(self, name)
            if t is not None:
                total += t.numel() * t.element_size()
        return total
