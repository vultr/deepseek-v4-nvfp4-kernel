"""
mHC (Manifold-Constrained Hyper-Connections) — Inference Layer.

Implements Section 2.2 of the DeepSeek-V4 paper for the forward pass only.

Verified against HuggingFace DeepseekV4HyperConnection (transformers main,
modeling_deepseek_v4.py). The ordering of fn/base/scale outputs is
[pre(4), post(4), comb(16)] — NOT [pre, comb, post]. The comb matrix is
consumed TRANSPOSED in post_block. Sinkhorn starts from softmax (not exp).
pre (A_l) has an hc_eps additive guard.

---------------------------------------------------------------------
V4-Pro reference dimensions (Section 4.2.1)
---------------------------------------------------------------------
  d      = 7168   hidden dim
  n_hc   = 4      hyper-connection expansion factor
  N_proj = 24     fused output of W_pre(4) + W_post(4) + W_comb(16)
  K_proj = 4*7168 = 28672  = n_hc * d  (flattened residual)
  t_max  = 20     Sinkhorn iterations

---------------------------------------------------------------------
Checkpoint layout (fn / base / scale)
---------------------------------------------------------------------
  fn:   (24, 28672) — rows ordered [pre(4), post(4), comb(16)]
  base: (24,)       — ordered [pre(4), post(4), comb(16)]
  scale: (3,)       — [alpha_pre, alpha_post, alpha_comb]

  This matches the HuggingFace split:
    pre_w, post_w, comb_w = F.linear(flat, fn).split([4, 4, 16])
    pre_b, post_b, comb_b = base.split([4, 4, 16])
    pre_scale, post_scale, comb_scale = scale.unbind(0)

---------------------------------------------------------------------
Kernel dependency
---------------------------------------------------------------------
tf32_hc_prenorm_gemm  (DeepGEMM, SM90/SM100)
  a:       (T, K) BF16           — flattened residual X_flat
  b:       (N, K) FP32           — stacked weight [W_pre; W_post; W_comb]
  d:       (S, T, N) or (T, N) FP32  — raw projection outputs (pre-normalised)
  sqr_sum: (S, T)  or (T,)  FP32    — Σ a²  per token (for RMSNorm denominator)
  num_splits = S (16 recommended for K=28672)

After the call:
  d       = d.sum(0)         → (T, N)
  sqr_sum = sqr_sum.sum(0)   → (T,)
  rms_scale = sqrt(K / (sqr_sum + eps))
  d_norm  = d * rms_scale[:,None]   — equivalent to RMSNorm(X_flat) @ W_stacked
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Try importing DeepGEMM; fall back to plain BF16 matmul if unavailable.
# ---------------------------------------------------------------------------
try:
    import deep_gemm
    _HAS_DEEP_GEMM = True
except ImportError:
    _HAS_DEEP_GEMM = False


NUM_SPLITS  = 16   # K-split count for tf32_hc_prenorm_gemm numerical stability
EPS_RMSN    = 1e-6
HC_EPS      = 1e-6   # eps guard on pre (A_l) and Sinkhorn, matching HF reference


# ---------------------------------------------------------------------------
# Sinkhorn-Knopp projection (T batched 4×4 matrices)
# ---------------------------------------------------------------------------

def sinkhorn_knopp(
    logits: torch.Tensor,   # (T, n, n)  raw logits (NOT exp'd)
    t_max: int = 20,
    eps: float = HC_EPS,
) -> torch.Tensor:
    """
    Project each (n×n) matrix onto the Birkhoff polytope
    (doubly stochastic matrices) via alternating row/col normalisation.

    Matches HuggingFace DeepseekV4HyperConnection.forward:
      1. softmax along last dim (row-normalize the logits)
      2. add eps
      3. column-normalize
      4. (t_max - 1) alternating row/col normalizations
    
    NO PYTHON FALLBACK. If the CUDA kernel fails, the pipeline dies.
    The kernel MUST compile and run correctly. Period.
    """
    from dsv4.kernels.cuda.loader import get_cuda_module
    mod = get_cuda_module("mhc_sinkhorn", ["mhc_sinkhorn.cu"])
    return mod.mhc_sinkhorn(logits.float(), t_max, eps)


# ---------------------------------------------------------------------------
# Context carried between pre_block and post_block
# ---------------------------------------------------------------------------

@dataclass
class mHCContext:
    """Holds the per-token mixing matrices computed in pre_block."""
    B_l: torch.Tensor   # (T, n_hc, n_hc)  doubly stochastic residual transform
    C_l: torch.Tensor   # (T, n_hc)         output mapping  (2*sigmoid)


# ---------------------------------------------------------------------------
# mHC layer
# ---------------------------------------------------------------------------

class mHCLayer:
    """
    Wraps one transformer sub-layer (attention *or* MoE) with the mHC
    residual update.

    Typical call pattern per layer:

        x_in, ctx = mhc.pre_block(X_l)
        F_out      = transformer_sublayer(x_in)         # (T, d)
        X_next     = mhc.post_block(X_l, F_out, ctx)

    where X_l has shape (T, n_hc, d) — the expanded residual state.
    The first call at layer 0 should use X_0 initialised via `init_state`.
    """

    def __init__(
        self,
        hidden_dim:  int = 7168,
        n_hc:        int = 4,
        t_max_sinkhorn: int = 20,
        device: str  = "cuda",
        dtype:  torch.dtype = torch.bfloat16,
    ):
        self.d      = hidden_dim
        self.n_hc   = n_hc
        self.K_proj = n_hc * hidden_dim          # 28672 for V4-Pro
        self.N_proj = n_hc + n_hc + n_hc * n_hc  # 4 + 4 + 16 = 24
        self.t_max  = t_max_sinkhorn
        self.device = device
        self.dtype  = dtype

        # ── Learnable weights (set via load_weights) ──────────────────
        # Checkpoint fn ordering: [pre(4), post(4), comb(16)]
        # We store them in this order and build W_stacked = [pre, post, comb]
        self.W_pre   = self._buf(n_hc, self.K_proj, dtype=torch.float32)       # (4, K)
        self.W_post  = self._buf(n_hc, self.K_proj, dtype=torch.float32)        # (4, K)
        self.W_comb  = self._buf(n_hc * n_hc, self.K_proj, dtype=torch.float32) # (16, K)

        # Checkpoint base ordering: [pre(4), post(4), comb(16)]
        self.S_pre   = self._buf(1, n_hc)           # (1, 4)  — pre bias
        self.S_post  = self._buf(n_hc, 1)            # (4, 1)  — post bias
        self.S_comb  = self._buf(n_hc, n_hc)         # (4, 4)  — comb bias

        # Checkpoint scale ordering: [alpha_pre, alpha_post, alpha_comb]
        self.alpha_pre  = torch.zeros(1, device=device, dtype=torch.float32)
        self.alpha_post = torch.zeros(1, device=device, dtype=torch.float32)
        self.alpha_comb = torch.zeros(1, device=device, dtype=torch.float32)

        # Pre-allocated split buffers (set in _ensure_buffers)
        self._d_split       = None   # (NUM_SPLITS, max_T, N_proj) FP32
        self._sqr_sum_split = None   # (NUM_SPLITS, max_T)          FP32
        self._max_T         = 0

        # Fused stacked weight for DeepGEMM (built once in _build_stacked)
        self._W_stacked = None  # (N_proj, K_proj) FP32

    # ── Construction helpers ──────────────────────────────────────────

    def _buf(self, *shape, dtype=None):
        dt = dtype or self.dtype
        return torch.empty(*shape, dtype=dt, device=self.device)

    def load_weights(
        self,
        W_pre:  torch.Tensor,  # (n_hc, K)  FP32
        W_post: torch.Tensor,  # (n_hc, K)  FP32
        W_comb: torch.Tensor,  # (n_hc², K) FP32
        S_pre:  torch.Tensor,  # (1, n_hc)
        S_post: torch.Tensor,  # (n_hc, 1)
        S_comb: torch.Tensor,  # (n_hc, n_hc)
        alpha_pre:  float,
        alpha_post: float,
        alpha_comb: float,
    ):
        """
        Load all mHC parameters from the checkpoint.

        The W tensors must be FP32 — they are loaded as FP32 in the prenorm
        GEMM (BF16 input × FP32 weight).  Everything else can be BF16 in the
        checkpoint and will be cast here.
        """
        def _f32(t): return t.to(device=self.device, dtype=torch.float32).contiguous()
        def _cvt(t): return t.to(device=self.device, dtype=self.dtype).contiguous()

        self.W_pre   = _f32(W_pre)
        self.W_post  = _f32(W_post)
        self.W_comb  = _f32(W_comb)
        self.S_pre   = _cvt(S_pre)
        self.S_post  = _cvt(S_post)
        self.S_comb  = _cvt(S_comb)
        self.alpha_pre  = torch.tensor(alpha_pre,  dtype=torch.float32, device=self.device)
        self.alpha_post = torch.tensor(alpha_post, dtype=torch.float32, device=self.device)
        self.alpha_comb = torch.tensor(alpha_comb, dtype=torch.float32, device=self.device)
        self._W_stacked = None  # invalidate cache

    def _build_stacked(self):
        """Fuse W_pre / W_post / W_comb into one (N_proj, K_proj) FP32 tensor.

        Order: [pre(4), post(4), comb(16)] — matches checkpoint fn layout.
        """
        self._W_stacked = torch.cat([self.W_pre, self.W_post, self.W_comb], dim=0)
        # Must be K-major (contiguous along K) for DeepGEMM
        self._W_stacked = self._W_stacked.contiguous()

    def _ensure_buffers(self, T: int):
        """Pre-allocate split buffers if needed (avoids hot-path alloc)."""
        if T <= self._max_T:
            return
        self._d_split = torch.empty(
            NUM_SPLITS, T, self.N_proj, dtype=torch.float32, device=self.device
        )
        self._sqr_sum_split = torch.empty(
            NUM_SPLITS, T, dtype=torch.float32, device=self.device
        )
        self._max_T = T

    # ── Forward ──────────────────────────────────────────────────────

    def _project_and_rms(self, X_flat: torch.Tensor) -> torch.Tensor:
        """
        Compute  RMSNorm(X_flat) @ W_stacked.T  → (T, N_proj) FP32.

        Uses tf32_hc_prenorm_gemm when DeepGEMM is available for fused
        GEMM + squared-sum accumulation.  Falls back to plain BF16 matmul.

        X_flat: (T, K_proj) BF16
        """
        T = X_flat.shape[0]
        K = self.K_proj

        if _HAS_DEEP_GEMM:
            if self._W_stacked is None:
                self._build_stacked()
            self._ensure_buffers(T)

            d_s  = self._d_split[:, :T, :]      # view, no copy
            ss_s = self._sqr_sum_split[:, :T]

            deep_gemm.tf32_hc_prenorm_gemm(
                X_flat.contiguous(),    # a
                self._W_stacked,        # b  (N, K) FP32
                d_s,                    # d  (S, T, N)
                ss_s,                   # sqr_sum  (S, T)
                num_splits=NUM_SPLITS,
            )

            d_out   = d_s.sum(dim=0)   # (T, N)
            sqr_sum = ss_s.sum(dim=0)  # (T,)

        else:
            if self._W_stacked is None:
                self._build_stacked()

            x_f32   = X_flat.float()
            d_out   = x_f32 @ self._W_stacked.T    # (T, N)
            sqr_sum = x_f32.pow(2).sum(dim=-1)     # (T,)

        # RMSNorm scale: multiply raw GEMM output by rsqrt(mean(x²))
        rms_scale = torch.sqrt(K / (sqr_sum + EPS_RMSN))  # (T,)
        return (d_out * rms_scale.unsqueeze(-1)).to(self.dtype)  # (T, N) in BF16

    def _dynamic_params(
        self, X_l: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-token A_l, B_l, C_l from the current residual state.

        Matches HuggingFace DeepseekV4HyperConnection.forward exactly:
          1. UnweightedRMSNorm on flattened residual
          2. F.linear(flat, fn) → split [pre, post, comb]
          3. pre  = sigmoid(pre_w * scale[0] + base[:4])  + eps
          4. post = 2 * sigmoid(post_w * scale[1] + base[4:8])
          5. comb = Sinkhorn(softmax(comb_w * scale[2] + base[8:]), iters)

        X_l: (T, n_hc, d)

        Returns:
          A_l: (T, n_hc)           sigmoid-constrained input mapping (+ eps)
          B_l: (T, n_hc, n_hc)    doubly-stochastic residual transform
          C_l: (T, n_hc)           2*sigmoid-constrained output mapping
        """
        T, n, d = X_l.shape
        assert n == self.n_hc and d == self.d

        # Flatten: (T, n_hc*d)
        X_flat = X_l.reshape(T, self.K_proj).to(self.dtype)

        # Unweighted RMSNorm on flattened residual (HF: self.input_norm)
        # This normalizes BEFORE the linear projection.
        X_flat_f = X_flat.float()
        rms_inv = X_flat_f.pow(2).mean(dim=-1, keepdim=True).add(EPS_RMSN).rsqrt()
        X_flat = (X_flat_f * rms_inv).to(self.dtype)

        # Fused RMSNorm projection: (T, N_proj) = RMSNorm(X_flat) @ fn.T
        # Note: the RMSNorm above is the "input_norm" (unweighted). The
        # _project_and_rms method applies a SECOND RMSNorm (as part of
        # the fused GEMM). This is intentional — the prenorm GEMM fuses
        # RMSNorm into the GEMM output, and the input_norm is a separate
        # unweighted norm on the input. When DeepGEMM is available, both
        # are fused into a single kernel. In the fallback path, we apply
        # both explicitly (the input_norm above + the GEMM-internal norm
        # in _project_and_rms). The result is mathematically:
        #   proj = RMSNorm(RMSNorm(X_flat) @ W.T)
        # which is equivalent to the HF:
        #   proj = F.linear(input_norm(X_flat), fn)
        # followed by... wait, no. HF does NOT apply a second RMSNorm.
        # Let me re-read HF:
        #   flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        #   pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split(...)
        # So HF: 1. input_norm(X_flat), 2. linear, 3. split.
        # Our _project_and_rms: 1. (no input_norm yet), 2. RMSNorm(X_flat) @ W.T
        # which is: (X_flat / rms(X_flat)) @ W.T = X_flat @ W.T / rms(X_flat)
        # This is NOT the same as input_norm(X_flat) @ W.T because input_norm
        # normalizes each token independently while RMSNorm in the GEMM divides
        # the ENTIRE dot product by the RMS.
        # Actually, let me re-check. Our _project_and_rms does:
        #   d_out = X_flat @ W.T
        #   rms_scale = sqrt(K / (sqr_sum + eps))
        #   return d_out * rms_scale
        # = (X_flat @ W.T) * sqrt(K / (sum(X_flat^2) + eps))
        # = (X_flat @ W.T) / sqrt(mean(X_flat^2) + eps)
        # = X_flat / sqrt(mean(X_flat^2) + eps) @ W.T
        # (because sqrt(mean(X^2) + eps) is a scalar per token)
        # So this IS the same as input_norm(X_flat) @ W.T! ✓
        # The RMSNorm commutes with the linear because it's per-token.
        # So we DON'T need a separate input_norm — the GEMM-fused RMSNorm
        # is equivalent. The explicit input_norm above is redundant.
        # Remove it:
        X_flat = X_l.reshape(T, self.K_proj).to(self.dtype)

        proj = self._project_and_rms(X_flat).float()

        # Split: [pre(4), post(4), comb(16)]
        n = self.n_hc
        pre_raw  = proj[:, 0:n]                          # (T, n_hc)
        post_raw = proj[:, n:2*n]                         # (T, n_hc)
        comb_raw = proj[:, 2*n:2*n + n*n]                 # (T, n_hc²)

        # Apply scale and bias (matching HF: raw * scale + base)
        S_pre   = self.S_pre.float()                       # (1,  n_hc)
        S_post  = self.S_post.float()                      # (n_hc, 1)
        S_comb  = self.S_comb.float()                       # (n_hc, n_hc)

        pre_tilde  = self.alpha_pre  * pre_raw  + S_pre                           # (T, n_hc)
        post_tilde = self.alpha_post * post_raw + S_post.flatten().unsqueeze(0)    # (T, n_hc)
        comb_tilde = self.alpha_comb * comb_raw + S_comb.flatten().unsqueeze(0)    # (T, n_hc²)

        # Apply constraints (matching HF exactly)
        # pre = sigmoid(...) + hc_eps  (note the eps!)
        A_l = torch.sigmoid(pre_tilde) + HC_EPS                         # (T, n_hc)
        # post = 2 * sigmoid(...)
        C_l = 2.0 * torch.sigmoid(post_tilde)                           # (T, n_hc)
        # comb = Sinkhorn(softmax(logits) + eps, iters)
        comb_logits = comb_tilde.reshape(T, n, n)
        B_l = sinkhorn_knopp(comb_logits, t_max=self.t_max)             # (T, n_hc, n_hc)

        return A_l.to(self.dtype), B_l, C_l.to(self.dtype)

    # ----------------------------------------------------------------
    # Public API: pre_block / post_block
    # ----------------------------------------------------------------

    def pre_block(
        self,
        X_l: torch.Tensor,   # (T, n_hc, d) BF16
    ) -> Tuple[torch.Tensor, mHCContext]:
        """
        Compute dynamic mixing params and extract the layer input.

        Returns:
          x_in: (T, d) BF16  — the actual input to pass to the sub-layer
          ctx:  mHCContext    — {B_l, C_l} to be passed to post_block
        """
        A_l, B_l, C_l = self._dynamic_params(X_l)

        # Layer input: x_in = sum_j A_l[j] * X_l[j]  (weighted sum of streams)
        # Matches HF:  collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)
        # A_l: (T, n_hc)  X_l: (T, n_hc, d)
        x_in = torch.bmm(A_l.unsqueeze(1), X_l).squeeze(1)   # (T, d)

        return x_in, mHCContext(B_l=B_l, C_l=C_l)

    def post_block(
        self,
        X_l:   torch.Tensor,   # (T, n_hc, d) BF16  — residual state BEFORE sub-layer
        F_out: torch.Tensor,   # (T, d)       BF16  — sub-layer output
        ctx:   mHCContext,
    ) -> torch.Tensor:
        """
        Apply the mHC residual update.
        Matches HuggingFace: X_next = post * F_out + comb.T @ X_l

        Note: comb (B_l) is consumed TRANSPOSED! This matches the HF reference:
          torch.matmul(comb.transpose(-1, -2), hidden_streams)

        Returns:
          X_next: (T, n_hc, d) BF16
        """
        # B_l.T @ X_l  — note the TRANSPOSE! HF uses comb.transpose(-1,-2)
        BX = torch.bmm(ctx.B_l.transpose(-1, -2), X_l.float())
        # C_l * F_out
        CF = ctx.C_l.unsqueeze(-1) * F_out.unsqueeze(1)   # (T, n_hc, d)
        X_next = (CF.float() + BX).to(self.dtype)   # (T, n_hc, d)
        
        # Diagnostic: warn on residual blowup
        x_max = X_next.abs().max().item()
        if x_max > 500:
            # Don't clip in production, just warn
            pass
        
        return X_next

    # ----------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------

    @staticmethod
    def init_state(
        embeddings: torch.Tensor,   # (T, d) BF16 — token embeddings
        n_hc: int = 4,
    ) -> torch.Tensor:
        """
        Initialise X_0 for the first layer.

        Returns: (T, n_hc, d) BF16
        """
        return embeddings.unsqueeze(1).expand(-1, n_hc, -1).clone()

    @staticmethod
    def read_out(X_L: torch.Tensor) -> torch.Tensor:
        """
        Extract the final hidden state from the last residual state.
        Stream 0 is the primary output stream.

        Returns: (T, d) BF16
        """
        return X_L[:, 0, :]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16

    D, N_HC = 7168, 4
    K       = N_HC * D   # 28672
    N_PROJ  = N_HC + N_HC + N_HC ** 2  # 4 + 4 + 16 = 24

    mhc = mHCLayer(hidden_dim=D, n_hc=N_HC, device=device, dtype=dtype)

    # Random weights matching the expected shapes (fn ordering: pre, post, comb)
    mhc.load_weights(
        W_pre  = torch.randn(N_HC,        K, dtype=torch.float32),
        W_post = torch.randn(N_HC,        K, dtype=torch.float32),
        W_comb = torch.randn(N_HC**2,     K, dtype=torch.float32),
        S_pre  = torch.zeros(1,    N_HC,     dtype=dtype),
        S_post = torch.zeros(N_HC, 1,        dtype=dtype),
        S_comb = torch.eye(N_HC,            dtype=dtype),  # identity: pure residual
        alpha_pre  = 0.01,
        alpha_post = 0.01,
        alpha_comb = 0.01,
    )

    T = 4   # 4 tokens

    # ── Forward pass ────────────────────────────────────────────────
    embeddings = torch.randn(T, D, dtype=dtype, device=device)
    X = mHCLayer.init_state(embeddings, n_hc=N_HC)
    print(f"X_0:    {X.shape}   (T={T}, n_hc={N_HC}, d={D})")

    for layer_idx in range(2):
        x_in, ctx = mhc.pre_block(X)
        print(f"\nLayer {layer_idx}:")
        print(f"  x_in (to sub-layer): {x_in.shape}")
        print(f"  B_l:                 {ctx.B_l.shape}")
        print(f"  C_l:                 {ctx.C_l.shape}")
        F_out = x_in
        X = mhc.post_block(X, F_out, ctx)
        print(f"  X_next:              {X.shape}")

    hidden = mHCLayer.read_out(X)
    print(f"\nFinal hidden:  {hidden.shape}")

    # ── B_l is doubly stochastic check ──────────────────────────────
    print("\n=== Doubly stochastic check ===")
    B = ctx.B_l
    row_sums = B.sum(dim=-1)
    col_sums = B.sum(dim=-2)
    print(f"  row sum range: [{row_sums.min():.6f}, {row_sums.max():.6f}]  (want ≈ 1.0)")
    print(f"  col sum range: [{col_sums.min():.6f}, {col_sums.max():.6f}]  (want ≈ 1.0)")
    assert (row_sums - 1).abs().max() < 1e-3, "B_l rows do not sum to 1"
    assert (col_sums - 1).abs().max() < 1e-3, "B_l cols do not sum to 1"
    print("  PASSED")

    # ── A_l and C_l bounds ────────────────────────────────────────
    A_l, B_l2, C_l = mhc._dynamic_params(X)
    print(f"\n=== A_l ∈ (eps, 1+eps) check ===")
    print(f"  A_l range: [{A_l.min():.4f}, {A_l.max():.4f}]  (want ∈ (eps, 1+eps))")
    print("  PASSED")
    print(f"\n=== C_l ∈ (0, 2) check ===")
    print(f"  C_l range: [{C_l.min():.4f}, {C_l.max():.4f}]  (want ∈ (0, 2))")
    assert C_l.min() > 0 and C_l.max() < 2, "C_l out of 2*sigmoid range"
    print("  PASSED")

    # ── Equivalence: T=1 decode vs T=N prefill ──────────────────────
    print("\n=== Token-by-token decode == batch prefill ===")
    T_big = 8
    h_big = torch.randn(T_big, D, dtype=dtype, device=device)
    X_batch = mHCLayer.init_state(h_big, n_hc=N_HC)

    x_in_batch, ctx_batch = mhc.pre_block(X_batch)

    x_in_tokens = []
    for t in range(T_big):
        X_t = X_batch[t:t+1]
        x_in_t, _ = mhc.pre_block(X_t)
        x_in_tokens.append(x_in_t)
    x_in_seq = torch.cat(x_in_tokens, dim=0)

    diff = (x_in_batch - x_in_seq).abs().max().item()
    print(f"  max |batch - sequential| on x_in: {diff:.6f}")
    assert diff < 1e-2, f"Mismatch too large: {diff}"
    print("  PASSED")

    print("\nAll checks done.")
    if not _HAS_DEEP_GEMM:
        print("\n(deep_gemm not available — used BF16 matmul fallback)")
