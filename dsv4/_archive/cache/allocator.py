"""Fixed-size block allocator for the classical paged KV cache.

One BlockAllocator per layer per "pool kind" (classical / indexer).
Total blocks are sized at engine startup. Blocks are recycled on
request completion.

Cudagraph-safety: allocation can't happen inside a captured graph
(allocation rate is per-request not per-token). The contract is:
  - acquire() called between graph captures.
  - release() called between graph captures.
  - read access (via block table) happens INSIDE captured graphs.
"""
from __future__ import annotations
import torch


class BlockAllocator:
    def __init__(
        self,
        num_total_blocks: int,
        device: str = "cuda",
    ):
        self.num_total_blocks = num_total_blocks
        self.device = device

        # Free-list as a GPU stack: ids[0..top-1] holds free block IDs.
        # `top` lives in pinned host memory so we can read it without a
        # device sync (it's modified only between graph captures).
        self.free_ids = torch.arange(
            num_total_blocks, dtype=torch.int32, device=device,
        )
        self.top_cpu = torch.tensor([num_total_blocks], dtype=torch.int32, pin_memory=True)

    @property
    def num_free(self) -> int:
        return int(self.top_cpu[0])

    def acquire(self, n: int) -> torch.Tensor:
        """Return a tensor of `n` block IDs. Called between captures."""
        top = int(self.top_cpu[0])
        if n > top:
            raise RuntimeError(
                f"KV cache OOM: requested {n} blocks, {top} available "
                f"(of {self.num_total_blocks} total)"
            )
        new_top = top - n
        ids = self.free_ids[new_top:top].clone()  # snapshot
        self.top_cpu[0] = new_top
        return ids

    def release(self, ids: torch.Tensor) -> None:
        """Return blocks to the free list. Called between captures."""
        n = ids.numel()
        top = int(self.top_cpu[0])
        self.free_ids[top:top + n] = ids.to(device=self.device)
        self.top_cpu[0] = top + n
