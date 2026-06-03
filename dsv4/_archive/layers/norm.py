"""RMSNorm — PyTorch reference implementation.

Swap to fused kernel (CuTeDSL) in Phase 6. API won't change.
"""
import torch


class RMSNorm:
    """Root Mean Square Layer Normalization.

    y = x / sqrt(mean(x^2) + eps) * weight

    CUDA-graph-compatible: weight is a buffer, no CPU syncs.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, device: str = "cuda"):
        self.hidden_size = hidden_size
        self.eps = eps
        self.device = device
        self.weight: torch.Tensor | None = None  # (hidden_size,) FP32, set by load_weights

    def load_weights(self, weight: torch.Tensor) -> None:
        assert weight.shape == (self.hidden_size,), f"weight shape {weight.shape} != ({self.hidden_size},)"
        self.weight = weight.to(device=self.device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, hidden_size) BF16 -> (T, hidden_size) BF16"""
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * rms * self.weight).to(torch.bfloat16)
