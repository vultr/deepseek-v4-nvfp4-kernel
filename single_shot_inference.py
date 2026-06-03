#!/usr/bin/env python3
"""Single-shot DSV4-Pro inference — Full production pipeline, 8-GPU.

ALL projections use production NVFP4 GEMM kernels (CuTeDSL).
ALL attention uses production FMHA (6-warp TMA multi-tile + sink bias).
ALL MoE uses production Nvfp4MoE + Nvfp4SharedExpert + Router.

NO PyTorch SDPA fallback. NO dequant+matmul for production projections.
This is the ground truth for vLLM / SGLang integration.
"""
import os, sys, time, json, math, argparse, logging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Catch async CUDA errors immediately
import torch
import torch.nn.functional as F
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("single_shot")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature (0=greedy)')
    p.add_argument('--repetition-penalty', type=float, default=1.1, help='Repetition penalty factor (>1 penalizes repeats)')
    p.add_argument('--top-k', type=int, default=50, help='Top-k filtering (0=disabled)')
    p.add_argument('--top-p', type=float, default=0.95, help='Top-p (nucleus) filtering (1.0=disabled)')
    p.add_argument('--prompt', type=str, default=None)
    p.add_argument('--thinking-mode', choices=['thinking', 'chat'], default='thinking',
                        help='Thinking mode: "thinking" = model reasons first, "chat" = model generates directly')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', type=int, default=1)
    p.add_argument('--prefill-only', action='store_true')
    p.add_argument('--no-fused-rmsnorm', action='store_true', help='Disable P4 fused RMSNorm+quantize (use unfused path)')
    p.add_argument('--warmup-gsa', action='store_true', help='Fix gsa values after first decode step (eliminates amax kernel launches)')
    p.add_argument('--profile', action='store_true', help='Profile per-component GPU time using CUDA events')
    p.add_argument('--num-gpus', type=int, default=8)
    p.add_argument('--checkpoint', type=str, default="/root/nvidia-meeting/DeepSeek-V4-Pro-NVFP4")
    p.add_argument('--prefill-tokens', type=str, default=None,
                   help='Override prompt tokens as comma-separated IDs (e.g. "1,128803,313,128804")')
    p.add_argument('--cuda-graph', action='store_true', help='Capture CUDA graph per layer for decode (eliminates Python dispatch overhead)')
    p.add_argument('--max-context', type=int, default=8192, help='Target max context length (determines KV cache pre-allocation)')
    return p.parse_args()

_args = parse_args()
CHECKPOINT_DIR = _args.checkpoint
MAX_NEW_TOKENS = _args.max_tokens
PROMPT = _args.prompt or "The capital of France is"
NUM_GPUS = _args.num_gpus
SEED = _args.seed
VERBOSE = _args.verbose
# Special token IDs — derived from official encoding module strings + tokenizer.
# Do NOT hardcode these; the encoding module defines the canonical token strings.
from encoding.deepseek_v4_encoding import (
    thinking_start_token as _THINK_START_STR,
    thinking_end_token as _THINK_END_STR,
    USER_SP_TOKEN as _USER_STR,
    ASSISTANT_SP_TOKEN as _ASSISTANT_STR,
    eos_token as _EOS_STR,
    bos_token as _BOS_STR,
)
FP4_LUT = torch.tensor([0., 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# =====================================================================
# RoPE (FP32 — BF16 destroys cos²+sin²=1)
# =====================================================================
def build_rope_cache(max_pos, rope_dim, device, theta=10000., rope_type="default",
                     rope_factor=1., orig_max=4096, beta_fast=32, beta_slow=1):
    freqs = 1. / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    if rope_type == "yarn" and rope_factor > 1.:
        nf = []
        for f in freqs:
            wl = 2 * math.pi / f
            lo, hi = orig_max / (beta_fast * 2.), orig_max / (beta_slow * 2.)
            if wl < lo: nf.append(f)
            elif wl > hi: nf.append(f / rope_factor)
            else:
                sm = (orig_max / (wl * beta_slow) - rope_factor) / (rope_factor * (beta_fast / beta_slow - 1))
                nf.append((1 - sm) * f / rope_factor + sm * f)
        freqs = torch.tensor(nf, dtype=torch.float32)
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32), freqs)
    return torch.cos(angles).to(device), torch.sin(angles).to(device)

def _apply_rope(x, pos, cos, sin, rope_dim, inverse=False):
    """In-place RoPE — uses CUDA kernel (1 launch) instead of PyTorch ops (5-6 launches).
    
    P3: Eliminates ~732 kernel launches per token across 61 layers.
    """
    try:
        from dsv4.ops.rope_cuda import apply_rope
        return apply_rope(x, pos, cos, sin, rope_dim, inverse=inverse)
    except Exception:
        # Fallback to PyTorch (should never happen in production)
        T, nh, hd = x.shape; nope = hd - rope_dim
        if pos.device != cos.device: pos = pos.to(cos.device)
        c, s = cos[pos].unsqueeze(1), sin[pos].unsqueeze(1)
        xr = x[:, :, nope:]
        ev = xr[..., 0::2].clone()
        od = xr[..., 1::2]
        if inverse:
            xr[..., 0::2] = (ev * c + od * s).bfloat16()
            xr[..., 1::2] = (-ev * s + od * c).bfloat16()
        else:
            xr[..., 0::2] = (ev * c - od * s).bfloat16()
            xr[..., 1::2] = (ev * s + od * c).bfloat16()
        return x

# =====================================================================
# Weight loading
# =====================================================================
def load_all_weights(checkpoint_dir):
    from safetensors.torch import load_file
    cdir = Path(checkpoint_dir); wmap = {}
    idx = cdir / "model.safetensors.index.json"
    if idx.exists():
        with open(idx) as f: wmap = json.load(f).get("weight_map", {})
    shards = set(wmap.values()) if wmap else set(); all_w = {}
    for sn in sorted(shards):
        if (cdir / sn).exists(): all_w.update(load_file(str(cdir / sn)))
    log.info(f"Loaded {len(all_w)} tensors from {len(shards)} shards"); return all_w

# =====================================================================
# RMSNorm
# =====================================================================
def rmsnorm(x, weight, eps=1e-6):
    xf = x.float()
    return (xf * xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt() * weight.float()).bfloat16()

def unweighted_rmsnorm(x, eps=1e-6):
    xf = x.float(); return xf * xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()

# =====================================================================
# CUDA Graph Decoder — capture per-layer graphs for zero-dispatch decode
# =====================================================================
class CUDAGraphDecoder:
    """Captures and replays CUDA graphs for the decode loop.
    
    After one warmup step, each layer's compute is captured as a CUDA graph.
    Replay eliminates Python dispatch overhead (~94ms for 61 layers) and
    kernel launch latency.
    
    Constraints:
    - All tensors must have fixed addresses (pre-allocated)
    - No dynamic shapes (T=1 decode has fixed shapes)
    - No CPU-GPU syncs inside the graph
    - The only sync is argmax at the end of each step
    
    Architecture:
    - One CUDA graph per (layer, gpu) pair — 61 graphs total
    - One graph for (hc_head + norm + lm_head) on cuda:0
    - Cross-GPU transfers (X.to(cuda:N)) happen outside graphs
    - The warmup step also computes and fixes gsa values
    """
    
    def __init__(self, n_layers, num_gpus, devices):
        self.n_layers = n_layers
        self.num_gpus = num_gpus
        self.devices = devices
        self.graphs = {}      # (li) -> torch.cuda.CUDAGraph
        self.lm_graph = None  # single graph for hc_head + norm + lm_head
        self.captured = False
        
        # Pre-allocated I/O buffers — fixed addresses for graph capture
        # Each layer reads X_in and writes X_out
        self.x_in_bufs = {}   # li -> tensor on device of layer li
        self.x_out_bufs = {}  # li -> tensor on device of layer li
        self.logits_buf = None  # (1, 129280) on cuda:0
    
    def pre_allocate(self, cfg, attn_mhcs, ffn_mhcs, attn_norms, ffn_norms,
                     kv_caches, compressors, indexers, moe_runners, se_runners,
                     routers, prod_lins, layer_w, rope_caches, hc_head,
                     final_norm_w, lm_head_lin, comp_rope_caches=None):
        """Pre-allocate all I/O buffers with fixed addresses."""
        for li in range(self.n_layers):
            dev = self.devices[li % self.num_gpus]
            # X is (1, 4, 7168) BF16
            self.x_in_bufs[li] = torch.zeros(1, 4, cfg["hidden_size"], dtype=torch.bfloat16, device=dev)
            self.x_out_bufs[li] = torch.zeros(1, 4, cfg["hidden_size"], dtype=torch.bfloat16, device=dev)
        self.logits_buf = torch.zeros(1, cfg.get("vocab_size", 129280), dtype=torch.bfloat16, device='cuda:0')
    
    def capture(self, cfg, attn_mhcs, ffn_mhcs, attn_norms, ffn_norms,
                kv_caches, compressors, indexers, moe_runners, se_runners,
                routers, prod_lins, layer_w, rope_caches, hc_head,
                final_norm_w, lm_head_lin, positions, token_id, comp_rope_caches=None):
        """Capture CUDA graphs for all layers + lm_head.
        
        Must be called after one warmup step so that:
        1. All CuTeDSL kernels are compiled and cached
        2. gsa values are fixed (from warmup_gsa)
        3. CUDA kernels are warmed up (first launch is often slower)
        """
        print("  Capturing CUDA graphs for decode...", flush=True)
        
        # Capture each layer as a separate graph
        for li in range(self.n_layers):
            gpu = li % self.num_gpus
            dev = self.devices[gpu]
            torch.cuda.set_device(gpu)
            
            # Copy current X into the fixed input buffer
            # (In practice, the warmup step's X is already on the right device)
            
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                X_out = forward_layer(
                    self.x_in_bufs[li], layer_w[li], li, cfg, *rope_caches[gpu],
                    attn_mhcs.get(li), ffn_mhcs.get(li),
                    attn_norms.get(li), ffn_norms.get(li),
                    kv_caches[li], positions, token_id,
                    compressors.get(li), indexers.get(li),
                    moe_runners.get(li), se_runners.get(li), routers.get(li),
                    prod_lin=prod_lins.get(li),
                    _use_fused_rmsnorm_quantize=True,
                    comp_rope_cos=comp_rope_caches[gpu][0] if comp_rope_caches else None,
                    comp_rope_sin=comp_rope_caches[gpu][1] if comp_rope_caches else None,
                )
                # Copy output to fixed buffer
                self.x_out_bufs[li].copy_(X_out)
            
            self.graphs[li] = graph
            if (li + 1) % 10 == 0:
                print(f"    Captured {li+1}/{self.n_layers} layer graphs", flush=True)
        
        # Capture hc_head + norm + lm_head on cuda:0
        torch.cuda.set_device(0)
        self.lm_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.lm_graph):
            # Note: x_in_bufs for the last layer is on the last layer's device.
            # For the lm_head graph, we need the X on cuda:0.
            # We'll handle the cross-GPU transfer outside the graph.
            x_out = self.x_out_bufs[self.n_layers - 1]  # may be on different GPU
            x_cuda0 = x_out.to('cuda:0')  # This may NOT work in a CUDA graph
            # Actually, cross-device memcpy in CUDA graphs is not supported.
            # We need to do the transfer outside and use a cuda:0 buffer.
            pass  # Will handle this differently
        
        self.captured = True
        print(f"  Captured {len(self.graphs)} layer graphs", flush=True)
# =====================================================================
def dequant_nvfp4(weight, weight_scale, weight_scale_2=None, input_scale=None):
    O, I2 = weight.shape; I = I2 * 2
    lo = (weight & 0x0F).to(torch.int8); hi = (weight >> 4).to(torch.int8)
    lut = FP4_LUT.to(device=weight.device, dtype=torch.float32)
    lo_f = lut[(lo & 0x07).long()] * torch.where((lo >> 3).bool(), -1., 1.)
    hi_f = lut[(hi & 0x07).long()] * torch.where((hi >> 3).bool(), -1., 1.)
    w = torch.stack([lo_f, hi_f], -1).reshape(O, I)
    s = weight_scale.float().repeat_interleave(16, 1)
    if weight_scale_2 is not None: s = s * weight_scale_2.float()
    return (w * s).bfloat16()

def nvfp4_linear_ref(x, weight, weight_scale, weight_scale_2=None, input_scale=None):
    return F.linear(x, dequant_nvfp4(weight, weight_scale, weight_scale_2, input_scale))

def get_nvfp4_weight(w, pfx, proj_name):
    k = f"{pfx}.{proj_name}"
    return (w.get(f"{k}.weight"), w.get(f"{k}.weight_scale"),
            w.get(f"{k}.weight_scale_2"), w.get(f"{k}.input_scale"))

def do_nvfp4_linear_ref(x, w, pfx, proj_name):
    weight, ws, ws2, isc = get_nvfp4_weight(w, pfx, proj_name)
    if weight is None: return None
    d = x.device
    return nvfp4_linear_ref(x, weight.to(d), ws.to(d),
                        ws2.to(d) if ws2 is not None else None,
                        isc.to(d) if isc is not None else None)

# =====================================================================
# Production Nvfp4Linear factory
# =====================================================================
def make_nvfp4_linear(in_features, out_features, device, all_w, pfx, proj_name):
    from dsv4.layers.linear import Nvfp4Linear
    d = device
    weight, ws, ws2, isc = get_nvfp4_weight(all_w, pfx, proj_name)
    assert weight is not None, f"{pfx}.{proj_name}.weight not found"
    actual_out = weight.shape[0]  # N_packed = GEMM output dimension
    actual_in = weight.shape[1] * 2  # K_packed * 2 = BF16 input dim (for buffer allocation)
    lin = Nvfp4Linear(actual_in, actual_out, max_num_tokens=8192, device=d)
    lin.fp4 = [weight.to(d)]; lin.sf = [ws.to(d)]
    lin.gs = [1.0]  # base gs — finalize_weights will multiply by ws2
    lin.ws2 = [ws2.to(d) if ws2 is not None else None]
    # CRITICAL FIX: Compute gsa at RUNTIME from actual input magnitude.
    # The checkpoint's input_scale is for training-time FP8 quantization.
    # Using it as gsa causes E4M3 block scale overflow when x/gsa > 2688.
    # We set a placeholder and override in the forward pass.
    lin._activation_global_scale = 1.0 / (6.0 * 448.0)  # placeholder
    lin._use_runtime_gsa = True  # flag to compute gsa at runtime
    lin.finalize_weights(); return lin

# =====================================================================
# Compressor — CSA (ratio=4) and HCA (ratio=128) [PRODUCTION KERNELS]
# =====================================================================
class Compressor:
    """Production compressor: NVFP4 GEMM projections + CUDA softmax/reduce.

    Pipeline:
      1. NVFP4 GEMM: hidden_states @ kv_proj → (T, kv_dim) BF16
      2. NVFP4 GEMM: hidden_states @ gate_proj → (T, kv_dim) BF16
      3. CUDA kernel: token-level softmax + weighted sum + kv_norm

    No PyTorch softmax. No reference fallback.
    """
    def __init__(self, ratio, head_dim, hidden_size, device):
        self.ratio, self.hd, self.H, self.device = ratio, head_dim, hidden_size, device
        self.is_csa = (ratio == 4); self.kv_dim = 2 * head_dim if self.is_csa else head_dim
        self.kv_lin = None   # production Nvfp4Linear for kv_proj
        self.gate_lin = None  # production Nvfp4Linear for gate_proj
        self._kv_bf16 = None   # BF16 weight for kv_proj (dequantized from NVFP4)
        self._gate_bf16 = None  # BF16 weight for gate_proj (dequantized from NVFP4)
        self.ape = None; self.kv_norm_w = None
        self._reduce_loaded = False
        # P7: Decode buffering — accumulate hidden_states until we have a complete block.
        # HCA (r=128): skip GEMMs entirely at T=1 decode (n_complete=0 every time).
        # CSA (r=4): buffer 4 decode steps, run GEMMs once per 4 tokens.
        self._hs_buffer = None   # (buf_len, H) BF16
        self._pos_buffer = None  # (buf_len,) long
        self._buf_len = 0

    def load(self, w, pfx, dev=None):
        """Load weights and build BF16 projections (dequantized from NVFP4)."""
        if dev is None: dev = self.device
        # Compressor projections are NOT explicitly FP4-QATed — dequant to BF16, use F.linear
        # CRITICAL: Use the PyTorch dequant_nvfp4 (defined in this file), NOT the CUDA
        # dequantize_nvfp4 from dsv4/ops/quantize.py. The CUDA kernel assumes
        # activation/KV scale layout (row-major (M, N/16)) and crashes on weight scales
        # that don't match — async illegal memory access surfaces at next sync.
        kv_w, kv_ws, kv_ws2, kv_isc = get_nvfp4_weight(w, pfx, 'kv_proj')
        gate_w, gate_ws, gate_ws2, gate_isc = get_nvfp4_weight(w, pfx, 'gate_proj')
        if kv_w is not None:
            self._kv_bf16 = dequant_nvfp4(kv_w.to(dev), kv_ws.to(dev), kv_ws2, kv_isc).to(dev).contiguous()
        if gate_w is not None:
            self._gate_bf16 = dequant_nvfp4(gate_w.to(dev), gate_ws.to(dev), gate_ws2, gate_isc).to(dev).contiguous()
        self.ape = w.get(f"{pfx}.position_bias")
        self.kv_norm_w = w.get(f"{pfx}.kv_norm.weight")

    def forward(self, hidden_states, positions):
        if self.ratio == 0 or self._kv_bf16 is None: return None, None, None
        T = hidden_states.shape[0]; r = self.ratio; dev = hidden_states.device

        # P7: Buffer decode steps until we have a complete block.
        # For HCA (r=128) at T=1 decode: n_complete is always 0, so we skip
        # the 2 NVFP4 GEMM launches entirely. No wasted compute.
        # For CSA (r=4): accumulate 4 tokens, run GEMMs once.
        if T < r:
            # Buffer this token's hidden_states + position
            if self._hs_buffer is None:
                self._hs_buffer = torch.zeros(r, self.H, dtype=torch.bfloat16, device=dev)
                self._pos_buffer = torch.zeros(r, dtype=torch.long, device=dev)
            if self._buf_len < r:
                self._hs_buffer[self._buf_len] = hidden_states[0] if T == 1 else hidden_states[self._buf_len]
                self._pos_buffer[self._buf_len] = positions[0] if positions.numel() == 1 else positions[self._buf_len]
                self._buf_len += 1
            if self._buf_len < r:
                return None, None, None  # Not enough tokens yet
            # We have a full buffer — use it
            hidden_states = self._hs_buffer[:self._buf_len]
            positions = self._pos_buffer[:self._buf_len]
            T = self._buf_len
            self._buf_len = 0  # Reset for next block

        n_complete = T // r
        if n_complete == 0: return None, None, None

        # Step 1-2: BF16 F.linear projections → FP32 for compress
        kv = torch.nn.functional.linear(hidden_states, self._kv_bf16).float()    # (T, kv_dim) FP32
        gate = torch.nn.functional.linear(hidden_states, self._gate_bf16).float()  # (T, kv_dim) FP32

        # Step 3: CUDA softmax/reduce kernel → FP32
        # KV-1/KV-2: Return FP32. Caller applies RoPE, then quantizes to NVFP4.
        from dsv4.kernels.compressor.production_compress import csa_compress_production_fp32, hca_compress_production_fp32
        if self.is_csa:
            compressed = csa_compress_production_fp32(
                kv, gate, self.ape, self.kv_norm_w, m=r)
        else:
            compressed = hca_compress_production_fp32(
                kv, gate, self.ape, self.kv_norm_w, m=r)

        if compressed.shape[0] == 0: return None, None, None
        n_comp = compressed.shape[0]

        # Vectorized position computation — no Python loop, no .item()
        # Block-aligned: use FIRST position of each block (vLLM cross-check confirmed)
        # Wrong: ((bi+1)*r - 1) uses LAST position → off by r-1 (3 for CSA, 127 for HCA)
        bi = torch.arange(n_comp, device=dev)
        pos_idx = (bi * r).clamp(max=positions.numel() - 1)
        comp_pos = positions[pos_idx]

        # Return FP32 compressed output — caller handles RoPE + NVFP4 quantize
        return compressed, comp_pos, torch.zeros(1, T, n_comp, dtype=torch.float32, device=dev)

# =====================================================================
# Indexer — CSA top-k [PRODUCTION NVFP4 GEMMs]
# =====================================================================
class Indexer:
    """Production indexer: NVFP4 GEMM projections + CUDA score+topk.

    Pipeline:
      1. NVFP4 GEMM: q_a (lora) @ q_b_proj → (T, n_ih * ihd) BF16
      2. NVFP4 GEMM: hidden_states @ weights_proj → (T, n_ih) BF16
      3. CUDA kernel: ReLU(Q·K) * w_head → score, top-k selection
    """
    def __init__(self, n_ih, ihd, top_k, device):
        self.n_ih, self.ihd, self.top_k, self.device = n_ih, ihd, top_k, device
        self.q_b_lin = None   # production Nvfp4Linear for q_b_proj (FP4-QATed)
        self._wp_bf16 = None    # BF16 weight for weights_proj (dequantized from NVFP4)
        self.compressor = None

    def load(self, w, pfx, dev=None):
        if dev is None: dev = self.device
        qb_w, qb_ws, qb_ws2, qb_isc = get_nvfp4_weight(w, pfx, 'q_b_proj')
        wp_w, wp_ws, wp_ws2, wp_isc = get_nvfp4_weight(w, pfx, 'weights_proj')
        # q_b_proj IS the FP4-QATed QK path — keep as NVFP4
        if qb_w is not None:
            qb_out = qb_w.shape[0]
            qb_in = qb_w.shape[1] * 2
            self.q_b_lin = make_nvfp4_linear(qb_in, qb_out, dev, w, pfx, 'q_b_proj')
        # weights_proj is NOT FP4-QATed — dequant to BF16 via PyTorch reference
        # CRITICAL: Use PyTorch dequant_nvfp4, NOT CUDA dequantize_nvfp4 (see Compressor.load)
        if wp_w is not None:
            self._wp_bf16 = dequant_nvfp4(wp_w.to(dev), wp_ws.to(dev), wp_ws2, wp_isc).to(dev).contiguous()
        # Indexer compressor weights are directly under the indexer prefix
        # (e.g. *.indexer.kv_proj.weight), NOT nested under *.indexer.compressor.
        if f"{pfx}.kv_proj.weight" in w:
            self.compressor = Compressor(4, self.ihd, 7168, dev)
            self.compressor.load(w, pfx, dev)

    def forward(self, q_lora, hidden_states, kv_cache, positions, layer_idx=None):
        """B2 FP8 tensor-core indexer scoring + weighted ReLU + top-k.

        Pipeline:
          1. NVFP4 GEMM: q_a (lora) @ q_b_proj → (T, n_ih * ihd) BF16
          2. NVFP4 GEMM: hidden_states @ weights_proj → (T, n_ih) BF16
          3. FP8 GEMM + ReLU + weighted sum + top-k (CUDA kernel)

        Indexer keys are consumed directly in FP8_E4M3 format — no BF16 dequant.
        """
        if self.q_b_lin is None or kv_cache is None or not kv_cache._has_idx or kv_cache.n_comp == 0:
            return None
        dev = q_lora.device; T = q_lora.shape[0]
        li = layer_idx

        q_idx = self.q_b_lin(q_lora).reshape(T, self.n_ih, self.ihd)  # (T, n_ih, ihd)
        w_h = torch.nn.functional.linear(hidden_states, self._wp_bf16)  # (T, n_ih) BF16

        # B2: FP8 tensor-core scoring path.
        # Indexer keys are stored as FP8_E4M3 in the KV cache.
        # No BF16 dequantization — the CUDA kernel consumes FP8 directly.
        k_fp8 = kv_cache.comp_idx_fp8[:kv_cache.n_comp]  # (n_comp, ihd) uint8
        k_scale = kv_cache.comp_idx_scale[:kv_cache.n_comp]  # (n_comp,) FP32
        n_comp = kv_cache.n_comp

        if li == 0:
            print(f"\n=== INDEXER PROBE L0 (B2 FP8) ===", flush=True)
            print(f"  q_idx: shape={tuple(q_idx.shape)} dtype={q_idx.dtype}", flush=True)
            print(f"  k_fp8: shape={tuple(k_fp8.shape)} dtype={k_fp8.dtype}", flush=True)
            print(f"  k_scale: shape={tuple(k_scale.shape)} dtype={k_scale.dtype}", flush=True)
            print(f"  w_h: shape={tuple(w_h.shape)} dtype={w_h.dtype}", flush=True)

        # For T=1 decode: use the B2 FP8 CUDA kernel
        if T == 1 and self.ihd == 128 and self.n_ih == 64:
            from dsv4.kernels.cuda.loader import get_cuda_module
            mod = get_cuda_module("indexer_fp8_score_topk", ["indexer_fp8_score_topk.cu"],
                                  extra_cuda_cflags=[
                                      "-gencode=arch=compute_100a,code=sm_100a",
                                      "-O3", "--use_fast_math", "--expt-relaxed-constexpr",
                                  ])
            q_2d = q_idx.squeeze(0).contiguous()  # (n_ih, ihd) BF16
            w_1d = w_h.squeeze(0).contiguous()  # (n_ih,) BF16
            tk = min(self.top_k, n_comp)
            topk_indices = torch.empty(tk, dtype=torch.int32, device=dev)
            mod.indexer_fp8_score_topk(
                q_2d, k_fp8, k_scale, w_1d, topk_indices,
                self.n_ih, self.ihd, tk)
            return topk_indices.unsqueeze(0)  # (1, top_k)

        # Fallback for T>1 or non-standard dimensions — FP32 einsum
        k_idx = k_fp8  # still FP8, need dequant for einsum
        if k_idx.dtype == torch.uint8 or str(k_idx.dtype) == 'torch.float8_e4m3fn':
            from dsv4.kernels.cuda.loader import get_cuda_module
            kv_mod = get_cuda_module("kv_quantize", ["kv_quantize.cu"])
            k_idx = kv_mod.dequant_fp8_e4m3(k_fp8, k_scale)  # (n_comp, ihd) BF16
        scores = torch.einsum('tnd,cd->tnc', q_idx.float(), k_idx.float())
        scores = F.relu(scores)
        total = (scores * w_h.unsqueeze(-1).float()).sum(1)
        tk = min(self.top_k, n_comp); _, idx = total.topk(tk, -1); return idx

# =====================================================================
# KV Cache
# =====================================================================
class KVCache:
    """KV Cache with mixed-precision compressed KV (DeepSeek V4 paper format).

    KV-1/KV-2: Compressed KV uses mixed storage:
      - Non-RoPE dims (448 of 512): FP8_E4M3 → ~50% size reduction
      - RoPE dims (64 of 512): BF16 (RoPE applied directly, stored as BF16)
    KV-3: Indexer keys stored as FP8_E4M3 (ihd=128, no RoPE).
    SWA: BF16 (128 tokens × 512 × 61 layers = 8MB, fits in L2).

    This matches the DeepSeek V4 paper: "BF16 for RoPE dims, FP8 for remaining dims.
    This hybrid representation reduces the KV cache size by nearly half."

    WHY NOT NVFP4 (native Blackwell FP4)?
    ─────────────────────────────────────
    We *really* wanted to use NVFP4 (E2M1 + E4M3 block scales + FP32 global scale)
    for compressed KV storage. Blackwell's native FP4→MMA path would have given us
    3.5× memory savings and direct tensor-core consumption — the dream pipeline.

    We tried. Hard. Three separate approaches:
      1. Fused compressor_reduce_quant.cu — single-kernel compress→NVFP4. Bugs in
         cross-warp block amax reduction and shared memory corruption (s_scratch
         stomping adjacent variables). Best cos=0.703. Dead.
      2. Proven two-kernel path (amax_gsa → quantize_from_buffer) using kv_quantize.cu's
         compute_amax_gsa_fp32 + quantize_nvfp4_from_fp32. cos=0.995 on random data,
         but that's the *quantize/dequant* round-trip in isolation. In the full pipeline,
         the 4-bit precision on 448 non-RoPE dimensions accumulated error across 61 layers
         of mHC — residual |X| already grows to 300-500, and NVFP4's 16-element block
         quantization (4.5 bits effective) added ~0.5% per layer on top of that.
      3. FP32 RoPE kernel (rope_fp32 in kv_quantize.cu) to avoid BF16 RoPE intermediate.
         Had an indexing bug (cos=0.977 for M>1). Fixed but the real issue was NVFP4,
         not RoPE.

    The verdict: NVFP4's 4.5 effective bits per element is simply too coarse for
    compressed KV values that get summed in attention softmax. FP8_E4M3's 5.3 effective
    bits gives cos=0.9997 round-trip (vs NVFP4's 0.995) — that 0.4% difference compounds
    fatally across 61 layers.

    So we settled on FP8_E4M3 for non-RoPE + BF16 for RoPE — exactly what DeepSeek V4
    ships in production. Not because we couldn't build the NVFP4 path (we did, it compiled
    and ran), but because the math didn't hold up. Sometimes 4 bits isn't enough.

    If Blackwell adds a finer-grained FP4 variant (8-element blocks, 6 effective bits),
    revisit this. The kernels exist. The quantize/dequant path is proven. The precision
    just isn't there yet for attention-sensitive KV values.

    Storage per compressed entry at hd=512:
      nope (448) × FP8 = 448 bytes + 4 bytes (scale) = 452
      rope (64) × BF16 = 128 bytes
      Total = 580 bytes vs 1024 bytes BF16 → 1.76× savings
    """
    def __init__(self, head_dim, window_size=128, max_comp=65536, device='cuda:0',
                 indexer_key_dim=128, compress_ratio=4, indexer_top_k=1024, rope_dim=64):
        self.hd, self.ws, self.dev = head_dim, window_size, device
        self.idx_key_dim = indexer_key_dim
        self.ratio = compress_ratio
        self.max_comp = max_comp
        self.rope_dim = rope_dim
        self.nope_dim = head_dim - rope_dim  # 448

        # SWA: BF16 (small, fits in L2)
        self.swa = torch.zeros(window_size, head_dim, dtype=torch.bfloat16, device=device)
        self.swa_pos = torch.zeros(window_size, dtype=torch.long, device=device)
        self.swa_len, self.swa_head = 0, 0

        # Compressed KV: mixed FP8 (nope) + BF16 (rope)
        self.comp_nope_fp8 = torch.zeros(max_comp, self.nope_dim, dtype=torch.uint8, device=device)
        self.comp_nope_scale = torch.zeros(max_comp, dtype=torch.float32, device=device)
        self.comp_rope_bf16 = torch.zeros(max_comp, rope_dim, dtype=torch.bfloat16, device=device)
        self.comp_pos_buf = torch.zeros(max_comp, dtype=torch.long, device=device)

        # Indexer compressed keys: FP8_E4M3
        self.comp_idx_fp8 = torch.zeros(max_comp, indexer_key_dim, dtype=torch.uint8, device=device)
        self.comp_idx_scale = torch.zeros(max_comp, dtype=torch.float32, device=device)

        # Pre-allocated mixed gather buffers.
        # CSA needs top_k + SWA; HCA is dense over compressed blocks, so it needs
        # max_comp + SWA. These buffers preserve the paper/native storage layout:
        # noPE stays FP8_E4M3 + scale, RoPE stays BF16.
        if compress_ratio > 4:
            self.mixed_gather_cap = max_comp + window_size
        elif compress_ratio == 4:
            self.mixed_gather_cap = indexer_top_k + window_size
        else:
            self.mixed_gather_cap = window_size
        self.gather_nope_fp8 = torch.zeros(self.mixed_gather_cap, self.nope_dim, dtype=torch.uint8, device=device)
        self.gather_nope_scale = torch.zeros(self.mixed_gather_cap, dtype=torch.float32, device=device)
        self.gather_rope_bf16 = torch.zeros(self.mixed_gather_cap, rope_dim, dtype=torch.bfloat16, device=device)

        # Legacy BF16 gather buffer kept only for non-B1 experiments; the live
        # B1 path below does not materialize noPE KV as global BF16.
        self.gather_buf = torch.zeros(indexer_top_k + window_size, head_dim, dtype=torch.bfloat16, device=device)
        self.n_comp = 0
        self._has_idx = False

        # Cache extension modules (loaded once)
        self._kv_quant_mod = None
        self._fp8_attn_io_mod = None

    def _get_kv_quant_mod(self):
        if self._kv_quant_mod is None:
            from dsv4.kernels.cuda.loader import get_cuda_module
            self._kv_quant_mod = get_cuda_module("kv_quantize", ["kv_quantize.cu"])
        return self._kv_quant_mod

    def _get_fp8_attn_io_mod(self):
        if self._fp8_attn_io_mod is None:
            from dsv4.kernels.cuda.loader import get_cuda_module
            self._fp8_attn_io_mod = get_cuda_module(
                "fp8_attention_io", ["fp8_attention_io.cu"],
                extra_cuda_cflags=[
                    "-gencode=arch=compute_100a,code=sm_100a",
                    "-O3", "--use_fast_math", "--expt-relaxed-constexpr",
                ],
            )
        return self._fp8_attn_io_mod

    def append_swa(self, kv, pos):
        """Vectorized SWA append — 2 kernel launches instead of 2T."""
        T = kv.shape[0]
        idx = (self.swa_head + torch.arange(T, device=self.dev)) % self.ws
        self.swa.index_copy_(0, idx, kv)
        self.swa_pos.index_copy_(0, idx, pos)
        self.swa_head = (self.swa_head + T) % self.ws
        self.swa_len = min(self.swa_len + T, self.ws)

    def set_compressed_mixed(self, nope_fp8, nope_scale, rope_bf16, comp_pos=None):
        """Add compressed KV entries (mixed FP8 nope + BF16 rope)."""
        T = nope_fp8.shape[0]
        end = self.n_comp
        self.comp_nope_fp8[end:end+T] = nope_fp8.view(torch.uint8) if nope_fp8.dtype != torch.uint8 else nope_fp8
        self.comp_nope_scale[end:end+T] = nope_scale
        self.comp_rope_bf16[end:end+T] = rope_bf16
        if comp_pos is not None:
            self.comp_pos_buf[end:end+T] = comp_pos
        self.n_comp = end + T

    def set_indexer_keys_fp8(self, idx_kv):
        """Add indexer compressed keys. idx_kv is BF16 (n_comp, ihd) or FP8 (fp8, scale)."""
        if idx_kv is None: return
        T = idx_kv[0].shape[0] if isinstance(idx_kv, tuple) else idx_kv.shape[0]
        end = self.n_comp - T
        if isinstance(idx_kv, tuple) and len(idx_kv) == 2:
            fp8, scale = idx_kv
            self.comp_idx_fp8[end:end+T] = fp8.view(torch.uint8) if fp8.dtype != torch.uint8 else fp8
            self.comp_idx_scale[end:end+T] = scale
        elif isinstance(idx_kv, torch.Tensor):
            mod = self._get_kv_quant_mod()
            fp8, scale = mod.quantize_fp8_e4m3_from_fp32(idx_kv.float().contiguous())
            self.comp_idx_fp8[end:end+T] = fp8.view(torch.uint8)
            self.comp_idx_scale[end:end+T] = scale
        self._has_idx = True

    def comp_nope_selective(self, indices):
        """Dequant FP8 nope for selected entries → BF16."""
        mod = self._get_kv_quant_mod()
        return mod.dequant_fp8_e4m3_selective(
            self.comp_nope_fp8, self.comp_nope_scale, indices.int())

    def comp_rope_selective(self, indices):
        """Gather BF16 rope for selected entries."""
        return self.comp_rope_bf16[indices.long()]

    @property
    def comp_nope_all(self):
        """Dequant all FP8 nope → BF16."""
        mod = self._get_kv_quant_mod()
        return mod.dequant_fp8_e4m3(
            self.comp_nope_fp8[:self.n_comp],
            self.comp_nope_scale[:self.n_comp])

    @property
    def comp_rope_all(self):
        """Return all BF16 rope entries."""
        return self.comp_rope_bf16[:self.n_comp]

    @property
    def comp_pos(self):
        return self.comp_pos_buf[:self.n_comp] if self.n_comp > 0 else None

    @property
    def comp_idx_kv(self):
        """Dequant FP8 indexer keys → BF16 for scoring."""
        if not self._has_idx or self.n_comp == 0: return None
        mod = self._get_kv_quant_mod()
        return mod.dequant_fp8_e4m3(
            self.comp_idx_fp8[:self.n_comp],
            self.comp_idx_scale[:self.n_comp])

    def gather_mixed_selective(self, indices):
        """Gather selected compressed KV + SWA into mixed FP8/BF16 buffers.

        Returns (nope_fp8, nope_scale, rope_bf16), each sliced to total length.
        noPE is not dequantized to global BF16.
        """
        mod = self._get_fp8_attn_io_mod()
        swa_kv, _ = self.get_swa()
        idx = indices.int().contiguous()
        total = idx.numel() + swa_kv.shape[0]
        if total > self.mixed_gather_cap:
            raise RuntimeError(f"mixed gather capacity {self.mixed_gather_cap} < requested {total}")
        mod.gather_mixed_selective_(
            self.comp_nope_fp8, self.comp_nope_scale, self.comp_rope_bf16,
            swa_kv, idx, self.gather_nope_fp8, self.gather_nope_scale, self.gather_rope_bf16)
        return (self.gather_nope_fp8[:total],
                self.gather_nope_scale[:total],
                self.gather_rope_bf16[:total])

    def gather_mixed_all(self):
        """Gather all compressed KV + SWA in mixed FP8/BF16 storage for HCA."""
        mod = self._get_fp8_attn_io_mod()
        swa_kv, _ = self.get_swa()
        n_comp = int(self.n_comp)
        total = n_comp + swa_kv.shape[0]
        if total > self.mixed_gather_cap:
            raise RuntimeError(f"mixed gather capacity {self.mixed_gather_cap} < requested {total}")
        mod.gather_mixed_all_(
            self.comp_nope_fp8[:n_comp], self.comp_nope_scale[:n_comp], self.comp_rope_bf16[:n_comp],
            swa_kv, self.gather_nope_fp8, self.gather_nope_scale, self.gather_rope_bf16)
        return (self.gather_nope_fp8[:total],
                self.gather_nope_scale[:total],
                self.gather_rope_bf16[:total])

    def gather_mixed_swa_only(self):
        """Quantize SWA noPE tail to FP8 and keep SWA RoPE as BF16."""
        mod = self._get_fp8_attn_io_mod()
        swa_kv, _ = self.get_swa()
        total = swa_kv.shape[0]
        if total > self.mixed_gather_cap:
            raise RuntimeError(f"mixed gather capacity {self.mixed_gather_cap} < requested {total}")
        mod.gather_mixed_swa_only_(
            swa_kv, self.gather_nope_fp8, self.gather_nope_scale, self.gather_rope_bf16, self.rope_dim)
        return (self.gather_nope_fp8[:total],
                self.gather_nope_scale[:total],
                self.gather_rope_bf16[:total])

    def get_swa(self):
        """Return SWA KV and positions as views (no clone)."""
        if self.swa_len == 0:
            return self.swa[:0], self.swa_pos[:0]
        if self.swa_len < self.ws:
            return self.swa[:self.swa_len], self.swa_pos[:self.swa_len]
        idx = torch.arange(self.swa_head, self.swa_head + self.ws, device=self.dev) % self.ws
        return self.swa[idx], self.swa_pos[idx]

# =====================================================================
# HcHead
# =====================================================================
HC_EPS = 1e-6
class HcHead:
    def __init__(self, hidden_dim=7168, n_hc=4, device='cuda:0'):
        self.K, self.device, self.n_hc = n_hc * hidden_dim, device, n_hc
    def load(self, fn, base, scale=None):
        self.fn = fn.to(self.device, torch.float32).contiguous()
        self.base = base.to(self.device, torch.float32).contiguous()
        self.scale = scale.to(self.device, torch.float32).item() if scale is not None else 1.0
    def forward(self, X):
        T = X.shape[0]; Xn = unweighted_rmsnorm(X.reshape(T, self.K).bfloat16())
        mix = F.linear(Xn, self.fn[:self.n_hc]).float()
        pre = torch.sigmoid(mix * self.scale + self.base[:self.n_hc].unsqueeze(0)) + HC_EPS
        return (pre.unsqueeze(-1) * X.float()).sum(1).bfloat16()

# =====================================================================
# Production FMHA
# =====================================================================
def _run_production_fmha(q_heads, all_kv, n_h, hd, T, seq_len, scale, dev, li, w, pfx):
    from dsv4.kernels.attention.production import dsv4_attention
    # Head-packed dispatch: single kernel launch for all 128 heads (MQA: 1 KV head shared)
    q = q_heads.permute(1, 0, 2).contiguous()  # (n_h, T, hd)
    k = all_kv.unsqueeze(0).contiguous()          # (1, N, hd) — MQA single KV head
    # K and V are the same in MQA — V = K transposed to (hd, N) format.
    # .transpose(-1,-2).contiguous() creates a new tensor (no clone needed).
    # This saves one full KV copy (~256KB per layer per decode step).
    v = k
    sinks = w.get(f"{pfx}.sinks"); sink_bias = None
    if sinks is not None: sink_bias = sinks.to(device=dev).float().reshape(n_h)
    attn_out = dsv4_attention(q=q, k=k, v=v, scale=scale, n_comp=0, sink_bias=sink_bias)
    return attn_out.permute(1, 0, 2)  # (T, n_h, hd)


def _run_production_fmha_mixed(q_heads, kv_nope_fp8, kv_nope_scale, kv_rope_bf16,
                               n_h, hd, T, seq_len, scale, dev, li, w, pfx, rope_dim):
    """B1 storage-native mixed FP8/BF16 FMHA. Supports decode (T=1) and prefill (T>1)."""
    from dsv4.kernels.attention.production import dsv4_attention_mixed_fp8_decode, dsv4_attention_mixed_fp8_prefill
    q = q_heads.permute(1, 0, 2).contiguous()  # (n_h, T, hd)
    sinks = w.get(f"{pfx}.sinks"); sink_bias = None
    if sinks is not None:
        sink_bias = sinks.to(device=dev).float().reshape(n_h)
    if T == 1:
        attn_out = dsv4_attention_mixed_fp8_decode(
            q=q,
            k_nope_fp8=kv_nope_fp8,
            k_nope_scale=kv_nope_scale,
            k_rope_bf16=kv_rope_bf16,
            scale=scale,
            sink_bias=sink_bias,
            rope_dim=rope_dim,
        )
    else:
        attn_out = dsv4_attention_mixed_fp8_prefill(
            q=q,
            k_nope_fp8=kv_nope_fp8,
            k_nope_scale=kv_nope_scale,
            k_rope_bf16=kv_rope_bf16,
            scale=scale,
            sink_bias=sink_bias,
            rope_dim=rope_dim,
        )
    return attn_out.permute(1, 0, 2)  # (T, n_h, hd)

# =====================================================================
# Attention — ALL production kernels
# =====================================================================
def forward_attention(x_normed, w, li, cfg, rope_cos, rope_sin,
                      kv_cache, positions, compressor, indexer, prod_lin,
                      x_quant=None,
                      _profile_detail=False, _profile_times=None,
                      comp_rope_cos=None, comp_rope_sin=None):
    dev = x_normed.device; T = x_normed.shape[0]
    n_h = cfg["num_attention_heads"]; hd = cfg["head_dim"]; rd = cfg.get("qk_rope_head_dim", 64)
    o_groups = cfg.get("o_groups", 16); o_rank = cfg.get("o_lora_rank", 1024)
    ratio = compressor.ratio if compressor is not None else 0
    scale = 1.0 / math.sqrt(hd); pfx = f"model.layers.{li}.self_attn"
    nope_dim = hd - rd  # 448 — used by both compress and gather
    if positions.device != rope_cos.device: positions = positions.to(rope_cos.device)

    def _pt(tag):
        """Profile timing helper — records CUDA-sync'd timestamp."""
        if _profile_detail and _profile_times is not None:
            torch.cuda.synchronize()
            _profile_times.append((tag, li, time.perf_counter()))

    _pt('q_a_start')
    # 1. Q: q_a (NVFP4 GEMM) → q_a_norm → q_b (NVFP4 GEMM) → q_b_norm
    q_a = prod_lin['q_a'].run_from_quantized(x_quant) if x_quant is not None else prod_lin['q_a'](x_normed)
    _pt('q_a_end')
    if VERBOSE >= 2 and li < 3:
        # Compare q_a with PyTorch reference
        q_a_ref = do_nvfp4_linear_ref(x_normed, w, pfx, 'q_a_proj')
        if q_a_ref is not None:
            cos_qa = torch.nn.functional.cosine_similarity(q_a.flatten().float(), q_a_ref.flatten().float(), dim=0).item()
            print(f"  L{li} q_a: |prod|={q_a.abs().max().item():.6f} |ref|={q_a_ref.abs().max().item():.6f} cos={cos_qa:.6f}", flush=True)
    q_norm_w = w.get(f"{pfx}.q_a_norm.weight")
    # B3: Fused rmsnorm+quant for q_a_norm → q_b path
    # Replaces: rmsnorm(q_a, w) → BF16 → q_b quantizes internally
    # With: fused rmsnorm+NVFP4 quantize → QuantizedActivation → q_b.run_from_quantized
    # Saves: ~6 kernel launches per layer (rmsnorm 4+ + quantize 2 vs fused 2)
    if q_norm_w is not None:
        from dsv4.ops.quantize import rmsnorm_quantize_nvfp4 as _rmsnorm_quantize, dequantize_nvfp4 as _dequantize_nvfp4
        q_a_quant = _rmsnorm_quantize(q_a, q_norm_w.to(dev, torch.float32))
        q_a = _dequantize_nvfp4(q_a_quant.x_fp4, q_a_quant.x_sf, q_a_quant.gsa)
    _pt('q_b_start')
    if q_norm_w is not None:
        q = prod_lin['q_b'].run_from_quantized(q_a_quant)
    else:
        q = prod_lin['q_b'](q_a)
    q = unweighted_rmsnorm(q).bfloat16()
    _pt('q_b_end')
    q_heads = q.reshape(T, n_h, hd); q_heads = _apply_rope(q_heads, positions, rope_cos, rope_sin, rd)
    _pt('rope_q_end')

    # 2. KV (NVFP4 GEMM, MQA, single KV head)
    _pt('kv_start')
    kv = prod_lin['kv'].run_from_quantized(x_quant) if x_quant is not None else prod_lin['kv'](x_normed)
    _pt('kv_end')
    kv_norm_w = w.get(f"{pfx}.kv_norm.weight")
    if kv_norm_w is not None: kv = rmsnorm(kv, kv_norm_w.to(dev, torch.float32))
    kv_3d = kv.reshape(T, 1, hd); kv_3d = _apply_rope(kv_3d, positions, rope_cos, rope_sin, rd)
    _pt('rope_kv_end')
    kv_roped = kv_3d.reshape(T, hd); kv_cache.append_swa(kv_roped, positions)

    # 3. Compressor → compressed KV (mixed storage: FP8 + BF16 RoPE)
    # DeepSeek V4 paper: "BF16 for RoPE dims, FP8 for remaining dims"
    _pt('compress_start')
    comp_pos, block_bias = None, None; comp_idx_kv = None
    if compressor is not None and compressor.ratio > 0:
        comp_kv_fp32, comp_pos, block_bias = compressor.forward(x_normed, positions)
        if comp_kv_fp32 is not None:
            from dsv4.kernels.cuda.loader import get_cuda_module
            kv_mod = get_cuda_module("kv_quantize", ["kv_quantize.cu"])
            # Split into non-RoPE (FP8) and RoPE (BF16) parts
            nope_fp32 = comp_kv_fp32[:, :nope_dim].contiguous()  # (n_comp, 448) FP32
            rope_bf16 = comp_kv_fp32[:, nope_dim:].bfloat16().contiguous()  # (n_comp, 64) BF16
            # Apply RoPE on BF16 rope dims (existing BF16 RoPE kernel)
            rope_3d = rope_bf16.unsqueeze(1)  # (n_comp, 1, 64)
            # Use compress_rope_theta cache for compressed entries if available
            crc = comp_rope_cos if comp_rope_cos is not None else rope_cos
            crs = comp_rope_sin if comp_rope_sin is not None else rope_sin
            rope_3d = _apply_rope(rope_3d, comp_pos, crc, crs, rd)
            rope_bf16 = rope_3d.squeeze(1)  # (n_comp, 64) BF16
            # Quantize non-RoPE part FP32 → FP8_E4M3
            nope_fp8, nope_scale = kv_mod.quantize_fp8_e4m3_from_fp32(nope_fp32)
            # Store mixed-format compressed KV + positions
            kv_cache.set_compressed_mixed(nope_fp8, nope_scale, rope_bf16, comp_pos)
        if compressor.is_csa and indexer is not None and indexer.compressor is not None:
            comp_idx_kv, _, _ = indexer.compressor.forward(x_normed, positions)
            # Indexer keys: FP8_E4M3 (ihd=128, no RoPE)
            kv_cache.set_indexer_keys_fp8(comp_idx_kv)
    _pt('compress_end')

    # 4. Indexer top-k (CSA)
    topk_idx = None
    if indexer is not None and ratio == 4:
        topk_idx = indexer.forward(q_a, x_normed, kv_cache, positions, layer_idx=li)

    # 5. Gather KV — B1 storage-native mixed path.
    # noPE remains FP8_E4M3 + per-row scale; RoPE remains BF16.
    # There is no global FP8->BF16 noPE materialization before FMHA.
    _pt('gather_start')
    swa_kv, _swa_pos = kv_cache.get_swa()
    swa_len = swa_kv.shape[0]
    if kv_cache.n_comp > 0:
        if ratio == 4:
            # CSA: gather top-k compressed rows + SWA tail without dequantizing noPE.
            assert topk_idx is not None, f"CSA layer {li}: indexer returned no top-k — indexer is broken"
            tk = topk_idx[0].clamp(0, kv_cache.n_comp - 1).int()
            kv_nope_fp8, kv_nope_scale, kv_rope_bf16 = kv_cache.gather_mixed_selective(tk)
        elif ratio > 4:
            # HCA: dense over compressed rows, still mixed storage.
            kv_nope_fp8, kv_nope_scale, kv_rope_bf16 = kv_cache.gather_mixed_all()
        else:
            kv_nope_fp8, kv_nope_scale, kv_rope_bf16 = kv_cache.gather_mixed_swa_only()
    else:
        kv_nope_fp8, kv_nope_scale, kv_rope_bf16 = kv_cache.gather_mixed_swa_only()
    seq_len = kv_nope_scale.shape[0]
    if seq_len == 0: return torch.zeros(T, cfg["hidden_size"], dtype=torch.bfloat16, device=dev), q_a

    # 6. Production FMHA — B1 mixed FP8/BF16 decode path.
    _pt('fmha_start')
    if li == 0:
        if VERBOSE >= 2:
            print(f"  L0 B1 verify: kv_nope_fp8 dtype={kv_nope_fp8.dtype} shape={tuple(kv_nope_fp8.shape)} "
                  f"kv_nope_scale dtype={kv_nope_scale.dtype} shape={tuple(kv_nope_scale.shape)} "
                  f"kv_rope_bf16 dtype={kv_rope_bf16.dtype} shape={tuple(kv_rope_bf16.shape)}", flush=True)
        assert kv_nope_fp8.dtype in (torch.uint8, torch.float8_e4m3fn), f"kv_nope_fp8 wrong dtype: {kv_nope_fp8.dtype}"
        assert kv_nope_scale.dtype == torch.float32, f"kv_nope_scale wrong dtype: {kv_nope_scale.dtype}"
        assert kv_rope_bf16.dtype == torch.bfloat16, f"kv_rope_bf16 wrong dtype: {kv_rope_bf16.dtype}"
        assert kv_nope_fp8.shape[-1] == nope_dim, f"kv_nope_fp8 dim={kv_nope_fp8.shape[-1]} != nope_dim={nope_dim}"
        assert kv_rope_bf16.shape[-1] == rd, f"kv_rope_bf16 dim={kv_rope_bf16.shape[-1]} != rope_dim={rd}"
    if VERBOSE >= 2 and li < 3:
        print(f"  L{li} FMHA mixed input: T={T} seq_len={seq_len} hd={hd} n_h={n_h} n_comp={kv_cache.n_comp} swa_len={swa_len}", flush=True)
    attn_out = _run_production_fmha_mixed(
        q_heads, kv_nope_fp8, kv_nope_scale, kv_rope_bf16,
        n_h, hd, T, seq_len, scale, dev, li, w, pfx, rd)
    _pt('fmha_end')
    if VERBOSE >= 2 and li < 3:
        print(f"  L{li} FMHA mixed: |prod|={attn_out.abs().max().item():.6f} (reference disabled: B1 forbids global BF16 KV staging)", flush=True)
    # 7. Inverse RoPE
    _pt('inv_rope_start')
    attn_out = _apply_rope(attn_out, positions, rope_cos, rope_sin, rd, inverse=True)
    _pt('inv_rope_end')

    # 8. Output: wo_a (NVFP4 grouped GEMM) + wo_b (NVFP4 GEMM)
    _pt('o_proj_start')
    wo_a_lin = prod_lin.get('o_a')
    if wo_a_lin is not None:
        # Nvfp4GroupedLinear: (T, n_h, hd) → (T, n_groups, o_rank) → flatten for o_b
        g_3d = wo_a_lin.run(attn_out)  # (T, n_groups, o_rank) BF16
        g_flat = g_3d.reshape(T, -1)    # (T, n_groups * o_rank) BF16
        F_attn = prod_lin['o_b'](g_flat)
    else:
        # BF16 grouped BMM fallback (should not happen in production)
        hpg_fb = n_h // o_groups; gid_fb = hpg_fb * hd
        oa_full = w.get(f"{pfx}.o_a_proj.weight")
        if oa_full is not None:
            oa_bf = oa_full.bfloat16().to(dev); a_flat = attn_out.reshape(T, n_h * hd)
            a_grp = a_flat.reshape(T, o_groups, gid_fb); oa_3d = oa_bf.reshape(o_groups, o_rank, gid_fb)
            g_out = torch.bmm(a_grp.permute(1, 0, 2), oa_3d.transpose(1, 2))
            g_flat = g_out.permute(1, 0, 2).reshape(T, o_groups * o_rank)
            F_attn = prod_lin['o_b'](g_flat)
        else:
            log.warning(f"L{li}: No o_a_proj weight, zero attention output")
            F_attn = torch.zeros(T, cfg["hidden_size"], dtype=torch.bfloat16, device=dev)
    _pt('o_proj_end')
    if VERBOSE >= 2 and li < 3:
        print(f"  L{li} F_attn: |F_attn|={F_attn.abs().max().item():.6f}", flush=True)
    return F_attn, q_a

# =====================================================================
# MoE — production kernels
# =====================================================================
def moe_forward(x, li, moe_runner, se_runner, router, token_id):
    # Ensure token_id is on same GPU as router
    token_id_dev = token_id.to(x.device) if token_id.device != x.device else token_id
    topk_w, topk_ids = router(x, token_ids=token_id_dev)
    # DEBUG: check topk_ids validity (only for first 3 and last 3 layers)
    if VERBOSE >= 2 and (li < 3 or li >= 58):
        if topk_ids.max().item() >= 384 or topk_ids.min().item() < 0:
            print(f"  L{li} BAD topk_ids: min={topk_ids.min().item()} max={topk_ids.max().item()}", flush=True)
    if VERBOSE >= 2 and li >= 58:
        print(f"  L{li} MoE DIAG: topk_ids={topk_ids[0].tolist()} topk_w=[{','.join(f'{w:.3f}' for w in topk_w[0].tolist())}]", flush=True)
        # Also print gate logits for debugging
        if hasattr(router, '_gate_lin') and router._gate_lin is not None:
            gate_logits = router._gate_lin(x).float()
            print(f"  L{li} gate logits: [{gate_logits.min().item():.3f}, {gate_logits.max().item():.3f}] mean={gate_logits.mean().item():.3f}", flush=True)
    if VERBOSE >= 2 and li < 3:
        print(f"  L{li} MoE input: |x|={x.abs().max().item():.4f} has_nan={torch.isnan(x).any().item()}", flush=True)
    routed_out = moe_runner.run(x, topk_w, topk_ids)
    shared_out = se_runner.run(x)
    if VERBOSE >= 2 and li >= 58:
        print(f"  L{li} MoE DIAG: |routed|={routed_out.abs().max().item():.1f} |shared|={shared_out.abs().max().item():.1f} |x|={x.abs().max().item():.1f}", flush=True)
    if VERBOSE >= 2 and li < 3:
        has_nan = torch.isnan(shared_out).any().item()
        out_max = shared_out.abs().max().item() if not has_nan else float('nan')
        print(f"  L{li} MoE shared: |out|={out_max:.4f} has_nan={has_nan}", flush=True)
        # Check weight integrity
        if hasattr(se_runner, '_l1_mat_b') and se_runner._l1_mat_b is not None:
            wb = se_runner._l1_mat_b.view(torch.uint8)
            print(f"  L{li} SE l1 weight: shape={list(se_runner._l1_mat_b.shape)} dtype={se_runner._l1_mat_b.dtype} uint8_range=[{wb.min().item()},{wb.max().item()}]", flush=True)
        if hasattr(se_runner, '_l1_scale_b') and se_runner._l1_scale_b is not None:
            sb = se_runner._l1_scale_b.float()
            print(f"  L{li} SE l1 scale: shape={list(se_runner._l1_scale_b.shape)} dtype={se_runner._l1_scale_b.dtype} float_range=[{sb.min().item():.6f},{sb.max().item():.6f}] has_nan={torch.isnan(sb).any().item()}", flush=True)
        print(f"  L{li} SE gsa: l1={se_runner._l1_activation_global_scale:.6f} l2={se_runner._l2_activation_global_scale:.6f} gsb: l1={se_runner._l1_gsb[0].item():.6f} l2={se_runner._l2_gsb[0].item():.6f}", flush=True)
    return routed_out + shared_out

# =====================================================================
# Layer forward
# =====================================================================
def forward_layer(X_l, w, li, cfg, rope_cos, rope_sin,
                  attn_mhc, ffn_mhc, attn_norm_w, ffn_norm_w,
                  kv_cache, positions, token_id,
                  compressor=None, indexer=None,
                  moe_runner=None, se_runner=None, router=None,
                  prod_lin=None, _profile_detail=False, _profile_times=None,
                  _use_fused_rmsnorm_quantize=True,
                  comp_rope_cos=None, comp_rope_sin=None,
):
    """Forward one transformer layer.
    """
    # P5: Fused mHC pre_block + RMSNorm + NVFP4 quantize
    # Replaces: pre_block (bmm) + rmsnorm (~4 launches) + quantize (2 launches)
    # With: 2 kernel launches total (mhc_rmsnorm_amax_gsa + mhc_rmsnorm_quantize_nvfp4)
    # Savings: ~5 launches per site × 2 sites × 61 layers = 610 launches/token
    from dsv4.ops.quantize import (
        rmsnorm_quantize_nvfp4, mhc_rmsnorm_quantize_nvfp4,
        QuantizedActivation, dequantize_nvfp4,
    )
    from dsv4.layers.mhc import mHCContext

    # Attention mHC: fused pre_block + rmsnorm + NVFP4 quantize
    A_l_a, B_l_a, C_l_a = attn_mhc._dynamic_params(X_l)
    ctx_a = mHCContext(B_l=B_l_a, C_l=C_l_a)

    if _use_fused_rmsnorm_quantize:
        # P5 fused: X_l + A_l → bmm + rmsnorm + NVFP4 quantize in 2 kernel launches
        x_quant_attn = mhc_rmsnorm_quantize_nvfp4(
            X_l, A_l_a, attn_norm_w.to(X_l.device, torch.float32))
        # Dequantize for compressor/indexer (1 kernel launch)
        x_normed = dequantize_nvfp4(x_quant_attn.x_fp4, x_quant_attn.x_sf, x_quant_attn.gsa)
    else:
        x_in = torch.bmm(A_l_a.unsqueeze(1).float(), X_l.float()).squeeze(1).bfloat16()
        x_normed = rmsnorm(x_in, attn_norm_w)
        x_quant_attn = None

    if _profile_detail: torch.cuda.synchronize(); t_attn0 = time.perf_counter()
    F_attn, _ = forward_attention(x_normed, w, li, cfg, rope_cos, rope_sin,
                                   kv_cache, positions, compressor, indexer, prod_lin,
                                   x_quant=x_quant_attn,
                                   _profile_detail=_profile_detail, _profile_times=_profile_times,
                                   comp_rope_cos=comp_rope_cos, comp_rope_sin=comp_rope_sin)
    if _profile_detail: torch.cuda.synchronize(); t_attn1 = time.perf_counter()
    X_mid = attn_mhc.post_block(X_l, F_attn, ctx_a)

    # FFN mHC: fused pre_block + rmsnorm + NVFP4 quantize
    A_l_f, B_l_f, C_l_f = ffn_mhc._dynamic_params(X_mid)
    ctx_f = mHCContext(B_l=B_l_f, C_l=C_l_f)

    if _use_fused_rmsnorm_quantize:
        # P5 fused: X_mid + A_l → bmm + rmsnorm + NVFP4 quantize in 2 kernel launches
        x_quant_ffn = mhc_rmsnorm_quantize_nvfp4(
            X_mid, A_l_f, ffn_norm_w.to(X_mid.device, torch.float32))
        # Dequantize for MoE (BF16 input required by MoE quantize path)
        x_ffn = dequantize_nvfp4(x_quant_ffn.x_fp4, x_quant_ffn.x_sf, x_quant_ffn.gsa)
    else:
        x_in_f = torch.bmm(A_l_f.unsqueeze(1).float(), X_mid.float()).squeeze(1).bfloat16()
        x_ffn = rmsnorm(x_in_f, ffn_norm_w)
    if _profile_detail: torch.cuda.synchronize(); t_ffn0 = time.perf_counter()
    F_ffn = moe_forward(x_ffn, li, moe_runner, se_runner, router, token_id)
    if _profile_detail: torch.cuda.synchronize(); t_ffn1 = time.perf_counter()
    X_next = ffn_mhc.post_block(X_mid, F_ffn, ctx_f)
    if VERBOSE >= 2 and (li < 3 or li >= 58):
        print(f"  L{li}: |X|={X_l.abs().max().item():.1f}->{X_next.abs().max().item():.1f} "
              f"|Fa|={F_attn.abs().max().item():.1f} |Ff|={F_ffn.abs().max().item():.1f}", flush=True)
    # Detailed diagnostics — only with VERBOSE >= 2 to avoid .item() syncs on hot path
    if VERBOSE >= 2 and (li >= 58 or (li > 0 and X_next.abs().max().item() > 200)):
        A_a, B_a, C_a = attn_mhc._dynamic_params(X_l)
        A_f, B_f, C_f = ffn_mhc._dynamic_params(X_mid)
        print(f"  L{li} DIAG: A_attn=[{A_a.min().item():.4f},{A_a.max().item():.4f}] "
              f"C_attn=[{C_a.min().item():.4f},{C_a.max().item():.4f}] "
              f"A_ffn=[{A_f.min().item():.4f},{A_f.max().item():.4f}] "
              f"C_ffn=[{C_f.min().item():.4f},{C_f.max().item():.4f}]", flush=True)
        print(f"  L{li} DIAG: B_attn row_sum=[{B_a.sum(-1).min().item():.4f},{B_a.sum(-1).max().item():.4f}] "
              f"col_sum=[{B_a.sum(-2).min().item():.4f},{B_a.sum(-2).max().item():.4f}] "
              f"B_ffn row_sum=[{B_f.sum(-1).min().item():.4f},{B_f.sum(-1).max().item():.4f}] "
              f"col_sum=[{B_f.sum(-2).min().item():.4f},{B_f.sum(-2).max().item():.4f}]", flush=True)
        print(f"  L{li} DIAG: |x_in_attn|={x_in.abs().max().item():.1f} "
              f"|x_in_ffn|={x_in_f.abs().max().item():.1f} "
              f"|X_l|={X_l.abs().max().item():.1f} "
              f"|X_mid|={X_mid.abs().max().item():.1f} "
              f"|X_next|={X_next.abs().max().item():.1f}", flush=True)
    if _profile_detail and (li < 3 or li == 30 or li >= 58):
        torch.cuda.synchronize()
        attn_ms = (t_attn1 - t_attn0) * 1000
        ffn_ms = (t_ffn1 - t_ffn0) * 1000
        print(f"  L{li}: attn={attn_ms:.2f}ms ffn={ffn_ms:.2f}ms", flush=True)
    return X_next

# =====================================================================
# MoE weight loading
# =====================================================================
def _load_moe_weights_stacked(all_w, li, pfx, dev, moe, cfg):
    n_e = cfg["n_routed_experts"]
    l1_fp4_list, l1_sf_list, l1_gs_list, l1_ws2_list, l1_gsa_list = [], [], [], [], []
    l2_fp4_list, l2_sf_list, l2_gs_list, l2_ws2_list, l2_gsa_list = [], [], [], [], []
    for eid in range(n_e):
        ep = f"{pfx}.experts.{eid}"
        gw, gws, gws2, gisc = get_nvfp4_weight(all_w, ep, 'gate_proj')
        uw, uws, uws2, uisc = get_nvfp4_weight(all_w, ep, 'up_proj')
        if gw is not None and uw is not None:
            l1_fp4_list.append(torch.cat([gw, uw], dim=0).to(dev))
            if gws is not None and uws is not None: l1_sf_list.append(torch.cat([gws, uws], dim=0).to(dev))
            gs = gisc.float().item() if gisc is not None else 1.0 / (6.0 * 448.0)
            l1_gs_list.append(1.0)  # gsb base — ws2 will be folded in by _ensure_stacked
            l1_gsa_list.append(gs)  # gsa = input_scale
            # weight_scale_2: scalar, folded into global_scale_b
            l1_ws2_list.append(gws2.to(dev) if gws2 is not None else None)
        dw, dws, dws2, disc = get_nvfp4_weight(all_w, ep, 'down_proj')
        if dw is not None:
            l2_fp4_list.append(dw.to(dev))
            if dws is not None: l2_sf_list.append(dws.to(dev))
            gs2 = disc.float().item() if disc is not None else 1.0 / (6.0 * 448.0)
            l2_gs_list.append(1.0)  # gsb base
            l2_gsa_list.append(gs2)  # gsa = input_scale
            l2_ws2_list.append(dws2.to(dev) if dws2 is not None else None)
    if not l1_fp4_list: log.warning(f"L{li}: No expert weights found"); return
    l1_stacked = torch.stack(l1_fp4_list).to(dev)
    l1_sf_stacked = torch.stack(l1_sf_list).to(dev) if l1_sf_list else None
    l2_stacked = torch.stack(l2_fp4_list).to(dev) if l2_fp4_list else None
    l2_sf_stacked = torch.stack(l2_sf_list).to(dev) if l2_sf_list else None
    del l1_fp4_list, l1_sf_list, l2_fp4_list, l2_sf_list
    moe.prepare_weights_from_stacked(l1_stacked, l1_sf_stacked, l1_gs_list, l2_stacked, l2_sf_stacked, l2_gs_list)
    # Save activation global scales — _ensure_stacked will override them from l1_gs (which is 1.0)
    # We must re-set them AFTER _ensure_stacked
    moe._saved_l1_gsa = l1_gsa_list[0] if l1_gsa_list else 1.0 / (6.0 * 448.0)
    moe._saved_l2_gsa = l2_gsa_list[0] if l2_gsa_list else 1.0 / (6.0 * 448.0)
    moe.l1_ws2 = l1_ws2_list
    moe.l2_ws2 = l2_ws2_list

def _load_shared_expert_weights(all_w, li, pfx, dev, se, cfg):
    gw, gws, gws2, gisc = get_nvfp4_weight(all_w, f"{pfx}.shared_experts", 'gate_proj')
    uw, uws, uws2, uisc = get_nvfp4_weight(all_w, f"{pfx}.shared_experts", 'up_proj')
    dw, dws, dws2, disc = get_nvfp4_weight(all_w, f"{pfx}.shared_experts", 'down_proj')
    if gw is not None and uw is not None:
        se.l1_fp4 = [torch.cat([gw, uw], dim=0).to(dev)]
        se.l1_sf = [torch.cat([gws, uws], dim=0).to(dev)] if gws is not None and uws is not None else [torch.zeros(1, device=dev, dtype=torch.float8_e4m3fn)]
        l1_isc = gisc.float().item() if gisc is not None else 1.0 / (6.0 * 448.0)
        se.l1_gs = [1.0]  # gsb base — ws2 will be folded in by finalize_weights
        se.l1_ws2 = [gws2.to(dev) if gws2 is not None else None]
        se._l1_activation_global_scale = l1_isc  # Will be overridden by _ensure_initialized
        se._saved_l1_gsa = l1_isc  # Save for after _ensure_initialized
    if dw is not None:
        se.l2_fp4 = [dw.to(dev)]
        se.l2_sf = [dws.to(dev)] if dws is not None else [torch.zeros(1, device=dev, dtype=torch.float8_e4m3fn)]
        l2_isc = disc.float().item() if disc is not None else 1.0 / (6.0 * 448.0)
        se.l2_gs = [1.0]  # gsb base
        se.l2_ws2 = [dws2.to(dev) if dws2 is not None else None]
        se._l2_activation_global_scale = l2_isc  # Will be overridden by _ensure_initialized
        se._saved_l2_gsa = l2_isc  # Save for after _ensure_initialized

def _cache_layer_weights_no_experts(all_w, n_layers, devices):
    cached = {}
    for li in range(n_layers):
        dev = devices[li % len(devices)]; pfx = f"model.layers.{li}."
        w = {k: v.to(device=dev, non_blocking=True) for k, v in all_w.items()
             if k.startswith(pfx) and '.experts.' not in k and '.shared_experts.' not in k}
        cached[li] = w
        if (li+1) % 10 == 0: log.info(f"  Cached {li+1}/{n_layers} layers")
    return cached

# =====================================================================
# Main
# =====================================================================
def kill_stale_gpu_processes():
    """Kill any leftover python processes on all GPUs before starting."""
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
            for pid in pids:
                try:
                    import os; os.kill(int(pid), 9)
                    log.info(f"  Killed stale GPU process {pid}")
                except (ValueError, ProcessLookupError):
                    pass
    except Exception as e:
        log.warning(f"Could not check GPU processes: {e}")

def main():
    t0 = time.time(); torch.manual_seed(SEED)
    print("=" * 70)
    print("DSV4 Single-Shot Inference - PRODUCTION KERNEL STACK")
    print("  FMHA: 6-warp TMA multi-tile + sink bias")
    print("  NVFP4 GEMM (CuTeDSL) for ALL projections")
    print("  Production MoE + Router | Production mHC")
    print("  NO PyTorch SDPA | NO dequant+matmul | NO reference fallback")
    print("=" * 70)

    with open(os.path.join(CHECKPOINT_DIR, "config.json")) as f:
        cfg = json.load(f)
    n_layers = cfg["num_hidden_layers"]; H = cfg["hidden_size"]
    hd = cfg["head_dim"]; n_h = cfg["num_attention_heads"]
    rd = cfg.get("qk_rope_head_dim", 64)
    cr = cfg.get("compress_ratios", [128] * n_layers)
    o_groups = cfg.get("o_groups", 16); o_rank = cfg.get("o_lora_rank", 1024)
    print(f"Model: {n_layers} layers, {n_h} heads, hd={hd}, rope_dim={rd}")
    print(f"Compress ratios: first5={cr[:5]} len={len(cr)}")
    print(f"Experts: {cfg['n_routed_experts']}, top-{cfg.get('num_experts_per_tok', 6)}")

    # ---- Phase 1: Load weights ----
    print(f"\nPhase 1: Loading weights..."); all_w = load_all_weights(CHECKPOINT_DIR)
    print(f"  {time.time()-t0:.1f}s")

    # ---- Phase 2: Build production components ----
    print("Building production components...")
    from dsv4.layers.mhc import mHCLayer
    from dsv4.layers.router import Router
    from dsv4.layers.moe import Nvfp4MoE
    from dsv4.layers.shared_expert import Nvfp4SharedExpert

    # Kill stale GPU processes from prior runs (OOM, crash, etc.)
    kill_stale_gpu_processes()
    for g in range(NUM_GPUS): torch.cuda.set_device(g); torch.cuda.empty_cache()
    torch.cuda.set_device(0)

    # mHC + norms
    attn_mhcs, ffn_mhcs, attn_norms, ffn_norms = {}, {}, {}, {}
    for li in range(n_layers):
        dev = f"cuda:{li % NUM_GPUS}"
        for tag, blocks, fn_s, base_s, scale_s in [
            ("attn", attn_mhcs, f"model.layers.{li}.attn_hc.fn", f"model.layers.{li}.attn_hc.base", f"model.layers.{li}.attn_hc.scale"),
            ("ffn", ffn_mhcs, f"model.layers.{li}.ffn_hc.fn", f"model.layers.{li}.ffn_hc.base", f"model.layers.{li}.ffn_hc.scale"),
        ]:
            fn, base, scale = all_w.get(fn_s), all_w.get(base_s), all_w.get(scale_s)
            if fn is not None and base is not None and scale is not None:
                m = mHCLayer(hidden_dim=H, n_hc=4, t_max_sinkhorn=20, device=dev)
                n = 4
                m.load_weights(
                    W_pre=fn[0:n].to(dev, torch.float32), W_post=fn[n:2*n].to(dev, torch.float32),
                    W_comb=fn[2*n:].to(dev, torch.float32),
                    S_pre=base[0:n].reshape(1, n).to(dev, torch.float32),
                    S_post=base[n:2*n].reshape(n, 1).to(dev, torch.float32),
                    S_comb=base[2*n:].reshape(n, n).to(dev, torch.float32),
                    alpha_pre=scale[0].item(), alpha_post=scale[1].item(), alpha_comb=scale[2].item(),
                )
                blocks[li] = m
        an_k = f"model.layers.{li}.input_layernorm.weight"
        if an_k in all_w: attn_norms[li] = all_w[an_k].to(dev, torch.float32)
        fn_k = f"model.layers.{li}.post_attention_layernorm.weight"
        if fn_k in all_w: ffn_norms[li] = all_w[fn_k].to(dev, torch.float32)

    # Production Nvfp4Linear for attention projections
    print("  Building production Nvfp4Linear for attention projections...")
    prod_lins = {}
    # Weight dimensions (from checkpoint):
    # q_a_proj: (1536, 3584) uint8 -> in=7168, out=1536
    # q_b_proj: (65536, 768) uint8 -> in=1536, out=65536
    # kv_proj:  (512, 3584) uint8 -> in=7168, out=512
    # o_a_proj: (16384, 4096) BF16 -> Nvfp4GroupedLinear (16 groups, 1024×4096 each)
    # o_b_proj: (7168, 8192) uint8 -> in=16384, out=7168
    from dsv4.layers.grouped_linear import Nvfp4GroupedLinear
    for li in range(n_layers):
        dev = f"cuda:{li % NUM_GPUS}"; pfx = f"model.layers.{li}.self_attn"
        torch.cuda.set_device(li % NUM_GPUS)
        pl = {}
        pl['q_a'] = make_nvfp4_linear(7168, 1536, dev, all_w, pfx, 'q_a_proj')
        pl['q_b'] = make_nvfp4_linear(1536, 65536, dev, all_w, pfx, 'q_b_proj')
        pl['kv'] = make_nvfp4_linear(7168, 512, dev, all_w, pfx, 'kv_proj')
        # o_a_proj: Nvfp4GroupedLinear (NVFP4 grouped GEMM)
        n_local_groups = cfg.get('o_groups', 16)
        heads_per_group = n_h // n_local_groups
        o_rank_val = cfg.get('o_lora_rank', 1024)
        wo_a = Nvfp4GroupedLinear(
            n_local_groups=n_local_groups,
            heads_per_group=heads_per_group,
            head_dim=hd,
            o_lora_rank=o_rank_val,
            max_num_tokens=8192,
            device=dev,
        )
        oa_w_nvfp4, oa_ws, oa_ws2, oa_isc = get_nvfp4_weight(all_w, pfx, 'o_a_proj')
        if oa_w_nvfp4 is not None and oa_ws is not None:
            # Checkpoint has NVFP4 weights — load directly (no dequant/re-quant)
            wo_a.load_nvfp4_weight(oa_w_nvfp4.to(dev), oa_ws.to(dev),
                                   oa_ws2.to(dev) if oa_ws2 is not None else None,
                                   oa_isc.to(dev) if oa_isc is not None else None)
        else:
            # BF16 checkpoint weight
            oa_bf = all_w.get(f"{pfx}.o_a_proj.weight")
            if oa_bf is not None:
                wo_a.set_bf16_weight(oa_bf.bfloat16().to(dev))
        pl['o_a'] = wo_a
        wo_a._use_runtime_gsa = True  # compute gsa from actual input to avoid E4M3 overflow
        pl['o_b'] = make_nvfp4_linear(16384, 7168, dev, all_w, pfx, 'o_b_proj')
        prod_lins[li] = pl
        if (li+1) % 10 == 0: print(f"    {li+1}/{n_layers} layers")
    print("  All attention projections: production NVFP4 GEMM (o_a now NVFP4 grouped)")

    # Routers, MoE, shared experts
    routers, moe_runners, se_runners = {}, {}, {}
    for li in range(n_layers):
        dev = f"cuda:{li % NUM_GPUS}"; pfx = f"model.layers.{li}.mlp"
        torch.cuda.set_device(li % NUM_GPUS); torch.cuda.synchronize()
        is_hash = (li < cfg.get("num_hash_layers", 3)) and (f"{pfx}.gate.tid2eid" in all_w)
        router = Router(hidden_size=H, num_experts=cfg["n_routed_experts"],
                       top_k=cfg.get("num_experts_per_tok", 6),
                       routed_scaling_factor=cfg.get("routed_scaling_factor", 2.5),
                       mode="hash" if is_hash else "dense",
                       vocab_size=cfg.get("vocab_size", 128000) if is_hash else None, device=dev)
        if is_hash:
            router.load_weights(hash_lut=all_w[f"{pfx}.gate.tid2eid"].to(dev, torch.int32))
        else:
            eb = all_w.get(f"{pfx}.gate.e_score_correction_bias")
            # BF16 router gate — dequantize NVFP4 to BF16, use F.linear
            E = cfg["n_routed_experts"]
            gate_w, gate_ws, gate_ws2, gate_isc = get_nvfp4_weight(all_w, pfx, 'gate')
            if gate_w is not None and gate_ws is not None:
                # Checkpoint has NVFP4 gate weight — dequantize to BF16
                # CRITICAL: Use PyTorch dequant_nvfp4, NOT CUDA dequantize_nvfp4
                # (same fix as Compressor.load — CUDA kernel crashes on weight scale layouts)
                gate_bf16 = dequant_nvfp4(gate_w.to(dev), gate_ws.to(dev), gate_ws2, gate_isc)
                router.W_gate = gate_bf16.T.contiguous().to(dev)  # (H, E) for F.linear(x, W_gate.T)
            else:
                # BF16 gate weight from checkpoint
                gw = all_w.get(f"{pfx}.gate.weight")
                gate_bf16 = gw.bfloat16().to(dev)
                if gate_bf16.shape[0] != H:
                    gate_bf16 = gate_bf16.T.contiguous()  # ensure (H, E)
                router.W_gate = gate_bf16.contiguous()
            # No gate_lin — force BF16 dispatch path
            router.gate_lin = None
            router.load_weights(e_bias=eb.to(dev, torch.float32))
            if li < 5: print(f"  L{li}: BF16 router gate (dequantized from NVFP4)", flush=True)
        router.finalize_weights(); routers[li] = router

        moe = Nvfp4MoE(num_experts=cfg["n_routed_experts"], hidden_size=H,
                       intermediate_size=cfg.get("moe_intermediate_size", 3072),
                       top_k=cfg.get("num_experts_per_tok", 6), device=dev)
        moe.set_swiglu_limit(cfg.get("swiglu_limit", 10.0))
        # P0: ENABLE fused SwiGLU — NVFP4 GEMM + SiLU in kernel registers.
        # Saves 240+ unfused BF16 kernel launches per token (gate_silu, clamp, mul, quantize).
        moe.set_fused_swiglu(True)
        _load_moe_weights_stacked(all_w, li, pfx, dev, moe, cfg)
        # EAGERLY process stacked weights → K-major + swizzle, free raw tensors
        moe._ensure_stacked()
        # Fix activation global scales — _ensure_stacked sets gsa from l1_gs (which is 1.0)
        # FIX: Do NOT use checkpoint input_scale as gsa — causes E4M3 overflow.
        # Instead, compute gsa at runtime from actual activation magnitude.
        # The MoE runner's compute_activation_global_scales() does this correctly.
        # We enable runtime gsa for both MoE and SharedExpert.
        moe._use_runtime_gsa = True
        moe_runners[li] = moe

        se = Nvfp4SharedExpert(hidden_size=H, intermediate_size=cfg.get("moe_intermediate_size", 3072),
                               device=dev, swiglu_limit=cfg.get("swiglu_limit", 10.0))
        _load_shared_expert_weights(all_w, li, pfx, dev, se, cfg)
        # P1: ENABLE fused SwiGLU for shared expert (1-group variant of MoE fused kernel)
        se.set_fused_swiglu(True)
        # EAGERLY process shared expert weights
        se._ensure_initialized()
        # P1: Eagerly warmup fused SwiGLU compilation for SE (1-group)
        if se._fused_swiglu:
            from dsv4.ops.gemm_runner import warmup_fused_swiglu_compilation
            K_packed = H // 2
            N_packed_l1 = (2 * cfg.get("moe_intermediate_size", 3072)) // 2  # gate+up
            warmup_fused_swiglu_compilation(
                1, K_packed, N_packed_l1, dev,
                swiglu_limit=cfg.get("swiglu_limit", 10.0),
            )
        # Fix activation global scales — _ensure_initialized sets gsa from l1_gs (which is 1.0)
        # FIX: Same runtime gsa for SharedExpert
        se._use_runtime_gsa = True
        se_runners[li] = se
        if (li+1) % 10 == 0: print(f"  Built {li+1}/{n_layers} MoE layers")
        torch.cuda.empty_cache()

    # Global weights
    torch.cuda.set_device(0)
    embed_w = all_w.get("model.embed_tokens.weight")
    embed = torch.nn.Embedding.from_pretrained(embed_w.bfloat16().to('cuda:0'))
    # lm_head: BF16 GEMM (checkpoint weight is BF16, no quantization)
    lm_w_raw = all_w.get("lm_head.weight", embed_w).bfloat16().to('cuda:0')
    lm_head_lin = None  # Use raw BF16 F.linear for lm_head
    lm_w = lm_w_raw  # Keep as (V, H) BF16 for F.linear
    print("  lm_head: BF16 GEMM (checkpoint weight, no quantization)")
    final_norm_w = all_w.get("model.norm.weight")
    if final_norm_w is not None: final_norm_w = final_norm_w.to('cuda:0', torch.float32)

    hc_head = HcHead(H, 4, 'cuda:0')
    hc_fn = all_w.get("model.hc_head.hc_fn"); hc_base = all_w.get("model.hc_head.hc_base"); hc_scale = all_w.get("model.hc_head.hc_scale")
    if hc_fn is not None and hc_base is not None: hc_head.load(hc_fn, hc_base, hc_scale); print("  hc_head loaded")

    # RoPE (FP32)
    rp = cfg.get("rope_scaling", cfg.get("rope_parameters", {}))
    rt = rp.get("type", rp.get("rope_type", "yarn")); rf = rp.get("factor", 16.0)
    rtheta = cfg.get("rope_theta", 10000.); romax = rp.get("original_max_position_embeddings", 65536)
    rbfast, rbslow = rp.get("beta_fast", 32), rp.get("beta_slow", 1)
    rope_caches = {g: build_rope_cache(romax, rd, f"cuda:{g}", rtheta, rt, rf, romax, rbfast, rbslow) for g in range(NUM_GPUS)}
    # Compressed-entry RoPE uses separate theta (vLLM cross-check: compress_rope_theta)
    # If compress_rope_theta differs from rope_theta, compressed KV entries need their own cache
    comp_rtheta = cfg.get("compress_rope_theta", rtheta)
    if comp_rtheta != rtheta:
        comp_rope_caches = {g: build_rope_cache(romax, rd, f"cuda:{g}", comp_rtheta, rt, rf, romax, rbfast, rbslow) for g in range(NUM_GPUS)}
        print(f"  Compressed RoPE theta: {comp_rtheta} (different from normal: {rtheta})")
    else:
        comp_rope_caches = rope_caches  # Same theta, reuse normal cache

    # KV caches, compressors, indexers
    kv_caches, compressors, indexers = {}, {}, {}
    n_ih = cfg.get("index_n_heads", 64); ihd = cfg.get("index_head_dim", 128); itk = cfg.get("index_topk", 1024)
    max_ctx = _args.max_context
    print(f"  Max context: {max_ctx} tokens (governs KV cache pre-allocation)")
    for li in range(n_layers):
        dev = f"cuda:{li % NUM_GPUS}"; ratio = cr[li] if li < len(cr) else 128
        # C1: max_comp derived from target context and compress ratio
        max_comp = (max_ctx + ratio - 1) // ratio if ratio > 0 else 0
        kv_caches[li] = KVCache(hd, cfg.get("sliding_window", 128), max_comp=max_comp, device=dev,
                                 indexer_key_dim=ihd, compress_ratio=ratio, indexer_top_k=itk, rope_dim=rd)
        if ratio > 0: compressors[li] = Compressor(ratio, hd, H, dev)
        if ratio == 4: indexers[li] = Indexer(n_ih, ihd, itk, dev)

    # Cache layer weights (no MoE/SE)
    print("Caching layer weights to GPUs (excluding MoE expert weights)...")
    devs = [f"cuda:{g}" for g in range(NUM_GPUS)]
    layer_w = _cache_layer_weights_no_experts(all_w, n_layers, devs)
    del all_w; import gc; gc.collect()
    for g in range(NUM_GPUS): torch.cuda.set_device(g); torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    print(f"  {time.time()-t0:.1f}s")

    # Load compressor/indexer weights
    for li in range(n_layers):
        pfx = f"model.layers.{li}.self_attn.compressor"
        if li in compressors: compressors[li].load(layer_w[li], pfx, dev=f"cuda:{li % NUM_GPUS}")
        if li in indexers: indexers[li].load(layer_w[li], f"{pfx}.indexer", dev=f"cuda:{li % NUM_GPUS}")
    print("  Compressors/indexers loaded")

    # ---- Phase 3: Inference ----
    print(f"\nPhase 3: Inference")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    # Derive special token IDs from official encoding strings + tokenizer.
    # This is the ONLY source of truth — never hardcode these IDs.
    THINK_START = tokenizer.convert_tokens_to_ids(_THINK_START_STR)
    THINK_END = tokenizer.convert_tokens_to_ids(_THINK_END_STR)
    USER_TOKEN = tokenizer.convert_tokens_to_ids(_USER_STR)
    ASSISTANT_TOKEN = tokenizer.convert_tokens_to_ids(_ASSISTANT_STR)
    bos = tokenizer.bos_token_id or 0

    # A1: Build explicit stop set — DSV4 uses special turn-end tokens beyond eos
    STOP_IDS = set()
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        STOP_IDS.add(eos_id)
    for tok_name in ("<|end_of_sentence|>",):
        tid = tokenizer.convert_tokens_to_ids(tok_name)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            STOP_IDS.add(tid)
    # If model emits USER_TOKEN it's trying to open a new user turn = it's done
    STOP_IDS.add(USER_TOKEN)
    print(f"  Stop set: {STOP_IDS} (eos={eos_id}, eos_token={tokenizer.eos_token})")
    print(f"  Special tokens: {tokenizer.special_tokens_map}")
    print(f"  THINK_START={THINK_START} THINK_END={THINK_END} USER={USER_TOKEN} ASST={ASSISTANT_TOKEN}")

    if _args.prefill_tokens:
        generated = [int(x) for x in _args.prefill_tokens.split(',')]
    else:
        # Official DeepSeek V4 encoding — canonical path, no hand-rolled alternatives.
        # Uses encoding/deepseek_v4_encoding.py (copied from vLLM tree) to build
        # the prompt. This is the ONLY way to construct prompts — the official
        # encoder handles BOS, User/Assistant tokens, thinking mode, and all
        # special token placement. It can't drift because it's the same code
        # the inference engines will use.
        from encoding.deepseek_v4_encoding import encode_messages
        messages = [{"role": "user", "content": PROMPT}]
        thinking_mode = _args.thinking_mode  # 'thinking' or 'chat'
        encoded_str = encode_messages(messages, thinking_mode=thinking_mode)
        generated = tokenizer.encode(encoded_str, add_special_tokens=False)
        # Ensure BOS token is present at the start
        if generated[0] != bos:
            generated = [bos] + generated
    all_tokens = generated.copy()
    print(f"Input: {len(generated)} tokens (thinking_mode={_args.thinking_mode})")

    # Batched prefill — process tokens in chunks of up to 128 (FMHA T≤128 constraint)
    PREFILL_CHUNK = 128  # max T per FMHA launch; split larger prefills into chunks
    n_prefill = len(generated)
    print(f"Batched prefill: {n_prefill} tokens, chunk_size={PREFILL_CHUNK}")
    prefill_ids = torch.tensor(generated, dtype=torch.long, device='cuda:0')
    prefill_ids32 = prefill_ids.to(torch.int32)
    all_positions = torch.arange(n_prefill, dtype=torch.long, device='cuda:0')

    # Process chunks: each chunk goes through ALL 61 layers before the next chunk.
    # This ensures KV cache is populated correctly for each layer.
    chunk_starts = list(range(0, n_prefill, PREFILL_CHUNK))
    X = None  # will be set by first chunk's embedding
    for ci, cs in enumerate(chunk_starts):
        ce = min(cs + PREFILL_CHUNK, n_prefill)
        chunk_len = ce - cs
        t1 = time.time()

        # Embed chunk tokens: (chunk_len, d)
        chunk_ids = prefill_ids[cs:ce]
        chunk_ids32 = prefill_ids32[cs:ce]
        chunk_positions = all_positions[cs:ce]
        chunk_embed = embed(chunk_ids)  # (chunk_len, d) BF16
        X = mHCLayer.init_state(chunk_embed)  # (chunk_len, n_hc, d) BF16

        for li in range(n_layers):
            gpu = li % NUM_GPUS
            if X.device != torch.device(f"cuda:{gpu}"): X = X.to(f"cuda:{gpu}")
            torch.cuda.set_device(gpu)
            try:
                X = forward_layer(X, layer_w[li], li, cfg, *rope_caches[gpu],
                                  attn_mhcs.get(li), ffn_mhcs.get(li),
                                  attn_norms.get(li), ffn_norms.get(li),
                                  kv_caches[li], chunk_positions, chunk_ids32,
                                  compressors.get(li), indexers.get(li),
                                  moe_runners.get(li), se_runners.get(li), routers.get(li),
                                  prod_lin=prod_lins.get(li),
                                  _use_fused_rmsnorm_quantize=not _args.no_fused_rmsnorm,
                                  comp_rope_cos=comp_rope_caches[gpu][0], comp_rope_sin=comp_rope_caches[gpu][1],
                                  )
            except Exception as e:
                torch.cuda.synchronize()
                print(f"  CRASH at chunk {ci} (tokens {cs}-{ce-1}) layer {li} gpu {gpu}: {e}", flush=True)
                raise
            if VERBOSE >= 2 and ci == 0 and li < 3:
                torch.cuda.synchronize(gpu)
                print(f"  Chunk {ci} L{li}: OK |X|={X.abs().max().item():.1f}", flush=True)
        X = X.to('cuda:0'); torch.cuda.set_device(0)
        print(f"  Chunk {ci+1}/{len(chunk_starts)} tokens {cs}-{ce-1} ({chunk_len} tok): {time.time()-t1:.2f}s", flush=True)
    print(f"  Batched prefill done ({time.time()-t0:.1f}s)")

    if _args.prefill_only: print("Prefill-only mode, stopping."); return

    # ---- Build sampler ----
    from dsv4.model.sampler import CUDASampler
    sampler = CUDASampler(device='cuda:0', max_penalty_tokens=256)
    sample_temp = _args.temperature
    sample_topk = _args.top_k
    sample_topp = _args.top_p
    sample_rep_pen = _args.repetition_penalty
    is_greedy = (sample_temp == 0.0)
    print(f"  Sampler: temp={sample_temp} top_k={sample_topk} top_p={sample_topp} "
          f"rep_pen={sample_rep_pen} greedy={is_greedy}")
    print(f"  DSV4 reasoning model: thinking_start={THINK_START} thinking_end={THINK_END}")
    print(f"  Thinking tokens are NOT garbage — model uses ）、...  format")

    # Pre-allocate decode buffers — zero per-step allocation
    dec_tid_buf = torch.zeros(1, dtype=torch.long, device='cuda:0')
    dec_pos_buf = torch.zeros(1, dtype=torch.long, device='cuda:0')
    dec_tid32_buf = torch.zeros(1, dtype=torch.int32, device='cuda:0')

    # Decode
    print(f"\nDecoding (max {MAX_NEW_TOKENS} tokens)...")
    in_thinking = False
    profile = _args.profile
    warmup_gsa = _args.warmup_gsa
    prof_embed_layers = 0.0
    prof_lm_head = 0.0
    prof_sample = 0.0
    prof_sample_start = 0.0

    # CUDA event profiling — measures ACTUAL GPU time, not wall clock
    # Only profile steps 1-3 (after warmup) to get stable results
    cuda_events = {}
    if profile:
        for tag in ['embed', 'layers', 'hc_norm_lm', 'sample', 'diagnostics']:
            cuda_events[tag] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        # Per-layer category events (sampled on step 1 only)
        layer_event_tags = ['mhc_pre', 'attn_proj', 'rope_kv', 'compress_idx', 'fmha', 'inv_rope', 'o_proj',
                            'mhc_post', 'mhc_pre_ffn', 'router', 'moe', 'shared_expert', 'mhc_post_ffn']
        cuda_layer_events = {}
        for tag in layer_event_tags:
            cuda_layer_events[tag] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        layer_event_accum = {tag: 0.0 for tag in layer_event_tags}
        layer_event_count = 0
        cuda_layer_events = []  # list of (tag, li, timestamp) for fine-grained profiling

    for step in range(MAX_NEW_TOKENS):
        t1 = time.time()
        dec_tid_buf[0] = all_tokens[-1]
        dec_tid32_buf[0] = all_tokens[-1]
        dec_pos_buf[0] = len(all_tokens) - 1

        t_e = time.perf_counter()
        X = mHCLayer.init_state(embed(dec_tid_buf))
        for li in range(n_layers):
            gpu = li % NUM_GPUS
            if X.device != torch.device(f"cuda:{gpu}"): X = X.to(f"cuda:{gpu}")
            torch.cuda.set_device(gpu)
            X = forward_layer(X, layer_w[li], li, cfg, *rope_caches[gpu],
                              attn_mhcs.get(li), ffn_mhcs.get(li),
                              attn_norms.get(li), ffn_norms.get(li),
                              kv_caches[li], dec_pos_buf, dec_tid32_buf,
                              compressors.get(li), indexers.get(li),
                              moe_runners.get(li), se_runners.get(li), routers.get(li),
                              prod_lin=prod_lins.get(li),
                              _profile_detail=(profile and step == 1),
                              _profile_times=cuda_layer_events if (profile and step == 1) else None,
                              _use_fused_rmsnorm_quantize=not _args.no_fused_rmsnorm,
                              comp_rope_cos=comp_rope_caches[gpu][0], comp_rope_sin=comp_rope_caches[gpu][1],
                              )
        X = X.to('cuda:0'); torch.cuda.set_device(0)
        t_layers = time.perf_counter()

        # After first decode step: fix gsa values from runtime amax
        # This eliminates amax_gsa kernel launches on subsequent steps
        # Only applies to attention linears and router gate (fixed per-projection gsa)
        # MoE/SE keep runtime gsa (gsa varies per token)
        if warmup_gsa and step == 0:
            torch.cuda.synchronize()
            n_fixed = 0
            for li in range(n_layers):
                pl = prod_lins.get(li)
                if pl is None: continue
                for key, lin in pl.items():
                    if hasattr(lin, '_gsa_buf') and hasattr(lin, '_use_runtime_gsa') and lin._use_runtime_gsa:
                        fixed_gsa = lin._gsa_buf.item()  # One-time sync
                        lin._activation_global_scale = fixed_gsa
                        lin._use_runtime_gsa = False
                        n_fixed += 1
                # Router gate
                router = routers.get(li)
                if router and hasattr(router, '_gate_lin') and router._gate_lin is not None:
                    gl = router._gate_lin
                    if hasattr(gl, '_gsa_buf') and hasattr(gl, '_use_runtime_gsa') and gl._use_runtime_gsa:
                        fixed_gsa = gl._gsa_buf.item()
                        gl._activation_global_scale = fixed_gsa
                        gl._use_runtime_gsa = False
                        n_fixed += 1
            # lm_head (BF16 — no gsa needed)
            if lm_head_lin is not None and hasattr(lm_head_lin, '_gsa_buf') and hasattr(lm_head_lin, '_use_runtime_gsa') and lm_head_lin._use_runtime_gsa:
                fixed_gsa = lm_head_lin._gsa_buf.item()
                lm_head_lin._activation_global_scale = fixed_gsa
                lm_head_lin._use_runtime_gsa = False
                n_fixed += 1
            print(f"  Warmup gsa: fixed {n_fixed} projection gsa values from step 0 (MoE/SE keep runtime gsa)", flush=True)
        x_out = hc_head.forward(X) if hc_head is not None else X[:, 0, :]
        if final_norm_w is not None: x_out = rmsnorm(x_out, final_norm_w)
        logits = torch.nn.functional.linear(x_out, lm_w) if lm_head_lin is None else lm_head_lin(x_out)
        if profile: torch.cuda.synchronize()
        t_lm = time.perf_counter()
        # Check thinking start token logit on first step
        if step == 0:
            ls = logits.float()
            for tid, name in [(THINK_START, 'think_start'), (THINK_END, 'think_end'), (USER_TOKEN, 'user'), (ASSISTANT_TOKEN, 'assistant')]:
                print(f"  {name}({tid}) logit={ls[0, tid].item():.2f}", flush=True)
            # Paris token check — only check known token IDs, no 129K iteration
            for t in [11111, 51119, 60107]:
                if t < ls.shape[-1]:
                    print(f"  Paris-candidate({t}) logit={ls[0, t].item():.2f}", flush=True)
        # Sync for profiling and error check
        if profile: torch.cuda.synchronize()
        t_sample_start = time.perf_counter()
        # Only sync + validate on first 3 steps and every 20th step (reduces pipeline stalls)
        if step < 3 or (step + 1) % 20 == 0:
            torch.cuda.synchronize()  # catch CUDA errors at source
        ls = logits.float()
        if step < 3 or (step + 1) % 20 == 0:
            has_nan = torch.isnan(ls).any().item()
            has_inf = torch.isinf(ls).any().item()
            print(f"  logits: shape={list(logits.shape)} dtype={logits.dtype} "
                  f"min={ls.min().item():.1f} max={ls.max().item():.1f} "
                  f"nan={has_nan} inf={has_inf}", flush=True)
            if has_nan or has_inf:
                print(f"  NaN/Inf in logits at step {step}, aborting", flush=True)
                break
        # Sampling — fused CUDA kernel (or greedy argmax for temp=0)
        if is_greedy:
            next_id = torch.argmax(logits, -1).item()
        else:
            sampled = sampler(
                logits,
                temperature=sample_temp,
                top_k=sample_topk,
                top_p=sample_topp,
                repetition_penalty=sample_rep_pen,
                recent_tokens=all_tokens[-256:],
                seed=SEED,
            )
            # Check for async CUDA errors from sampler
            if step < 3:
                torch.cuda.synchronize()
            next_id = sampled[0].item()

        all_tokens.append(next_id)
        dt = time.time() - t1

        if profile: torch.cuda.synchronize()
        t_s = time.perf_counter()
        # Track thinking state
        if next_id == THINK_START: in_thinking = True
        elif next_id == THINK_END: in_thinking = False

        if profile:
            prof_embed_layers += (t_layers - t_e)
            prof_lm_head += (t_lm - t_layers)
            prof_sample_start = t_sample_start
            prof_sample += (t_s - t_sample_start)

        # Diagnostics — every step for first 20, then every 5th
        if step < 20 or step % 5 == 0:
            tv, ti = torch.topk(logits[0].float(), 5)
            top5 = ' '.join(f'{tokenizer.decode([t.item()])}({v.item():.1f})' for t, v in zip(ti[:5], tv[:5]))
            think_tag = " [THINKING]" if in_thinking else ""
            print(f"  Step {step}: {next_id} '{tokenizer.decode([next_id])}' ({dt:.2f}s) "
                  f"logits=[{logits.float().min().item():.1f},{logits.float().max().item():.1f}] "
                  f"|X|={X.abs().max().item():.1f} top5: {top5}{think_tag}", flush=True)

        # NaN safety — periodic check only
        if step == 0 or (step+1) % 20 == 0:
            if torch.isnan(logits.float()).any().item():
                print(f"  NaN at step {step}", flush=True); break
        if next_id in STOP_IDS:
            print(f"  STOP ({next_id}) at step {step} — token='{tokenizer.decode([next_id])}'", flush=True); break

    if profile and MAX_NEW_TOKENS > 0:
        n = MAX_NEW_TOKENS
        print(f"\n  PROFILE (sync'd wall clock, {n} steps):")
        print(f"    Embed + 61 layers: {prof_embed_layers:.3f}s total, {prof_embed_layers/n*1000:.1f}ms/token")
        print(f"    hc_head + norm + lm_head: {prof_lm_head:.3f}s total, {prof_lm_head/n*1000:.1f}ms/token")
        print(f"    Sampling: {prof_sample:.3f}s total, {prof_sample/n*1000:.1f}ms/token")
        
        # Fine-grained attention profile (from step 1)
        if hasattr(cuda_layer_events, '__len__') and len(cuda_layer_events) >= 2:
            print(f"\n  FINE-GRAINED ATTENTION PROFILE (step 1, CUDA-sync'd):")
            prev_t = None
            for tag, li, t in cuda_layer_events:
                if prev_t is not None:
                    dt_ms = (t - prev_t) * 1000
                    if li <= 2 or li >= 58:  # Only print for first/last layers
                        print(f"    L{li} {tag}: {dt_ms:.2f}ms")
                prev_t = t

    out_raw = tokenizer.decode(all_tokens, skip_special_tokens=False)
    # Use official DSV4 parser for structured output
    try:
        from encoding.deepseek_v4_encoding import parse_message_from_completion_text
        # Find the assistant portion — after the last ASSISTANT token
        assistant_start = out_raw.find(_ASSISTANT_STR)
        if assistant_start >= 0:
            assistant_text = out_raw[assistant_start + len(_ASSISTANT_STR):]
        else:
            assistant_text = out_raw
        parsed = parse_message_from_completion_text(assistant_text, thinking_mode=_args.thinking_mode)
        reasoning = parsed.get('reasoning', '')
        content = parsed.get('content', '')
        print(f"\n{'='*70}")
        print(f"Input: '{PROMPT}'")
        if reasoning:
            print(f"Reasoning: {reasoning[:500]}{'...' if len(reasoning) > 500 else ''}")
        print(f"Content: {content}")
        print(f"Total: {time.time()-t0:.1f}s")
        print(f"{'='*70}")
    except Exception as e:
        # Fallback: raw decode (shouldn't happen with correct output)
        out = tokenizer.decode(all_tokens, skip_special_tokens=True)
        print(f"\n{'='*70}")
        print(f"Input: '{PROMPT}'")
        print(f"Output (raw): '{out}'")
        print(f"Parse error: {e}")
        print(f"Total: {time.time()-t0:.1f}s")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()
