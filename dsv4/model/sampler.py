"""Production token sampler — fused CUDA kernel wrapper.

Implements temperature scaling, repetition penalty, top-k, top-p (nucleus) sampling.
All computation on GPU, zero CPU syncs, CUDA-graph-compatible.

Usage:
    sampler = CUDASampler(device='cuda:0')
    token_id = sampler(logits, temperature=0.6, top_k=50, top_p=0.95,
                       repetition_penalty=1.1, recent_tokens=token_history)
"""

from __future__ import annotations
import os
import torch
from typing import Optional, List

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    from dsv4.kernels.cuda.loader import get_cuda_module
    _kernel = get_cuda_module("sampler", ["sampler.cu"])
    return _kernel


class CUDASampler:
    """Production sampler with fused CUDA kernel.

    All sampling happens on GPU. No .item() calls, no CPU tensors.
    The output is a GPU int64 tensor — the caller can .item() once
    at the end of the decode loop, or keep it on GPU for further processing.
    """

    def __init__(self, device: str = 'cuda:0', max_penalty_tokens: int = 256):
        self.device = device
        self.max_penalty_tokens = max_penalty_tokens
        self._penalty_ids_buf = torch.zeros(1, max_penalty_tokens, dtype=torch.int64, device=device)
        self._penalty_vals_buf = torch.ones(1, max_penalty_tokens, dtype=torch.float32, device=device)
        self._step = 0

    def __call__(
        self,
        logits: torch.Tensor,             # (1, vocab_size) or (batch, vocab_size) BF16 or FP32
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        min_tokens_to_keep: int = 1,
        recent_tokens: Optional[List[int]] = None,  # token IDs for repetition penalty
        seed: Optional[int] = None,
    ) -> torch.Tensor:  # (batch,) int64 on GPU
        """Sample tokens from logits using fused CUDA kernel.

        Returns int64 tensor on GPU. Use .item() to get Python int if needed.
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        assert logits.dim() == 2

        # Convert to FP32 for the sampler kernel
        logits_f32 = logits.float()

        batch = logits_f32.shape[0]
        if seed is None:
            seed = 42
        offset = self._step
        self._step += 1

        # Build repetition penalty buffers
        pen_ids = None
        pen_vals = None
        if repetition_penalty != 1.0 and recent_tokens:
            # Deduplicate and limit
            unique_tokens = list(dict.fromkeys(recent_tokens[-self.max_penalty_tokens:]))
            n_pen = len(unique_tokens)
            if n_pen > 0 and batch <= self._penalty_ids_buf.shape[0]:
                if batch > self._penalty_ids_buf.shape[0]:
                    self._penalty_ids_buf = torch.zeros(batch, self.max_penalty_tokens, dtype=torch.int64, device=self.device)
                    self._penalty_vals_buf = torch.ones(batch, self.max_penalty_tokens, dtype=torch.float32, device=self.device)
                self._penalty_ids_buf.zero_()
                self._penalty_vals_buf.fill_(1.0)
                for i, tid in enumerate(unique_tokens):
                    self._penalty_ids_buf[0, i] = tid
                    self._penalty_vals_buf[0, i] = repetition_penalty
                pen_ids = self._penalty_ids_buf[:batch, :n_pen]
                pen_vals = self._penalty_vals_buf[:batch, :n_pen]

        k = _get_kernel()
        result = k.sample(
            logits_f32,
            pen_ids,
            pen_vals,
            float(temperature),
            int(top_k),
            float(top_p),
            int(min_tokens_to_keep),
            int(seed),
            int(offset),
        )
        return result  # (batch,) int64 on GPU


class PyTorchSampler:
    """Reference sampler using pure PyTorch ops (for correctness verification).

    Same API as CUDASampler. Used to verify the CUDA kernel produces
    the same distribution.
    """

    def __init__(self, device: str = 'cuda:0'):
        self.device = device

    def __call__(
        self,
        logits: torch.Tensor,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        min_tokens_to_keep: int = 1,
        recent_tokens: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        logits = logits.float().clone()

        # Repetition penalty
        if repetition_penalty != 1.0 and recent_tokens:
            for tid in set(recent_tokens):
                if 0 <= tid < logits.shape[-1]:
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

        # Temperature
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
            sorted_indices_to_remove[..., :min_tokens_to_keep] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')

        # Sample
        probs = torch.softmax(logits, dim=-1)
        if seed is not None:
            torch.manual_seed(seed)
        return torch.multinomial(probs, 1).squeeze(-1).to(torch.int64)
