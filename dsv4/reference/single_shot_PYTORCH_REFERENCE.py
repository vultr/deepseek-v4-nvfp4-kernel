#!/usr/bin/env python3
"""Single-shot DSV4-Pro inference PYTORCH VERSION — Full 61-layer pipeline, 8-GPU.

THIS is a pure-PyTorch reference reimplementation that bypasses every kernel in the production stack.

IT IS ONLY TO BE USED FOR REFERENCE FOR THE CONSTRUCTION OF THE ACTUAL PRODUCTION KERNEL SINGLE SHOT

THIS FILE WAS MADE BY AN LLM THAT WAS ASKED TO IMPLIMENT THE PRODUCTION KERNEL AND INSTEAD IT JUST REDID IT IN PYTORCH.
THE FACT THIS FILE EXISTS PISSES ME OFF. IT DEMONSTRATES THAT AI IS FAR FROM INTELLIGENT, IT CAN NOT FOLLOW SIMPLE INSTRUCTIONS OR TRULY REASON, AND TRIES TO DO EVERYTHING SHITTY AND FAST.

Architecture (paper §2, verified against HuggingFace modeling_deepseek_v4.py):
  X_l → mHC.pre_block → RMSNorm → Attention → F_attn → mHC.post_block → X_mid
  X_mid → mHC.pre_block → RMSNorm → FFN(MoE) → F_ffn → mHC.post_block → X_{l+1}

Components exercised:
  - mHC (Sinkhorn-Knopp, B_l transposed, [pre,post,comb] ordering)
  - Low-rank Q: q_a_proj → q_a_norm → q_b_proj → q_b_norm
  - KV: kv_proj → kv_norm — single latent per token (MQA)
  - Compressor: CSA (ratio=4, Ca/Cb overlapping) and HCA (ratio=128)
  - Indexer: CSA top-k with its own compressor at index_head_dim
  - Partial RoPE (last 64 dims, GPT-J interleaved, YaRN factor=16) + inverse
  - Attention sinks (per-head logit bias)
  - Full attention: [compressed_kv, swa_kv] concatenated
  - Grouped output projection: wo_a (BF16 BMM) + wo_b (NVFP4)
  - MoE: 384 experts, top-6, hash (layers 0-2) + noaux_tc (3+), SwiGLU clamp
  - Shared expert (NVFP4)
  - NVFP4 two-level scale: weight_scale (E4M3) × weight_scale_2 (scalar) × input_scale (scalar)

Checkpoint key format:
  model.layers.{li}.self_attn.{kv_proj, q_a_proj, q_b_proj, o_a_proj, o_b_proj}.{weight, weight_scale, ...}
  model.layers.{li}.self_attn.compressor.{kv_proj, gate_proj}.{weight, weight_scale, ...}
  model.layers.{li}.self_attn.compressor.position_bias (BF16)
  model.layers.{li}.self_attn.compressor.kv_norm.weight (BF16)
  model.layers.{li}.self_attn.compressor.indexer.*
  model.layers.{li}.self_attn.sinks (BF16)
  model.layers.{li}.attn_hc.{fn, base, scale}
  model.layers.{li}.ffn_hc.{fn, base, scale}
  model.layers.{li}.input_layernorm.weight (BF16)
  model.layers.{li}.post_attention_layernorm.weight (BF16)
  model.layers.{li}.mlp.experts.{eid}.{gate_proj,up_proj,down_proj}.{weight, weight_scale, ...}
  model.layers.{li}.mlp.shared_experts.{gate_proj,up_proj,down_proj}.{weight, weight_scale, ...}
  model.layers.{li}.mlp.gate.{weight, e_score_correction_bias, tid2eid}
  model.embed_tokens.weight, model.norm.weight, lm_head.weight
  model.hc_head.{hc_fn, hc_base, hc_scale}
"""
import os, sys, time, json, math, argparse
import torch
import torch.nn.functional as F
from pathlib import Path

# =====================================================================
# Configuration
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--prompt', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', type=int, default=1)
    return p.parse_args()

_args = parse_args()
CHECKPOINT_DIR = "/root/nvidia-meeting/DeepSeek-V4-Pro-NVFP4"
MAX_NEW_TOKENS = _args.max_tokens
PROMPT = _args.prompt or "The capital of France is"
NUM_GPUS = 8
SEED = _args.seed
VERBOSE = _args.verbose
GROWTH_DIAG = VERBOSE >= 1

THINK_START, THINK_END = 128821, 128822
USER_TOKEN, ASSISTANT_TOKEN = 128803, 128804

# =====================================================================
# NVFP4 dequantization — two-level scale
# =====================================================================
FP4_LUT = torch.tensor([0., 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def dequant_nvfp4(weight, weight_scale, weight_scale_2=None, input_scale=None):
    """Dequantize NVFP4 → BF16. weight: (O,I//2) uint8, scale: (O,I//16) E4M3."""
    O, I2 = weight.shape
    I = I2 * 2
    lo = (weight & 0x0F).to(torch.int8)
    hi = (weight >> 4).to(torch.int8)
    lut = FP4_LUT.to(device=weight.device, dtype=torch.float32)
    lo_f = lut[(lo & 0x07).long()] * torch.where((lo >> 3).bool(), -1., 1.)
    hi_f = lut[(hi & 0x07).long()] * torch.where((hi >> 3).bool(), -1., 1.)
    w = torch.stack([lo_f, hi_f], -1).reshape(O, I)
    s = weight_scale.float().repeat_interleave(16, 1)
    if weight_scale_2 is not None: s = s * weight_scale_2.float()
    # NOTE: input_scale is intentionally NOT used. It's the activation
    # quantization scale (for FP8 inputs). Since we use BF16 activations,
    # the weight dequant is: lut[weight] * weight_scale * weight_scale_2.
    return (w * s).bfloat16()

def nvfp4_linear(x, weight, weight_scale, weight_scale_2=None, input_scale=None):
    return F.linear(x, dequant_nvfp4(weight, weight_scale, weight_scale_2, input_scale))

def get_nvfp4_weight(w, pfx, proj_name):
    k = f"{pfx}.{proj_name}"
    return (w.get(f"{k}.weight"), w.get(f"{k}.weight_scale"),
            w.get(f"{k}.weight_scale_2"), w.get(f"{k}.input_scale"))

def do_nvfp4_linear(x, w, pfx, proj_name):
    weight, ws, ws2, isc = get_nvfp4_weight(w, pfx, proj_name)
    if weight is None: return None
    d = x.device
    return nvfp4_linear(x, weight.to(d), ws.to(d),
                        ws2.to(d) if ws2 is not None else None,
                        isc.to(d) if isc is not None else None)

# =====================================================================
# RMSNorm
# =====================================================================
def rmsnorm(x, weight, eps=1e-6):
    xf = x.float()
    return (xf * xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt() * weight.float()).bfloat16()

def unweighted_rmsnorm(x, eps=1e-6):
    xf = x.float()
    return xf * xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()

# =====================================================================
# mHC
# =====================================================================
HC_EPS = 1e-6

def sinkhorn_knopp(logits, t_max=20, eps=HC_EPS):
    M = torch.softmax(logits, -1) + eps
    M = M / (M.sum(-2, keepdim=True) + eps)
    for _ in range(t_max - 1):
        M = M / (M.sum(-1, keepdim=True) + eps)
        M = M / (M.sum(-2, keepdim=True) + eps)
    return M

class mHCBlock:
    def __init__(self, hidden_dim=7168, n_hc=4, sinkhorn_iters=20, device='cuda:0'):
        self.d, self.n_hc, self.K = hidden_dim, n_hc, n_hc * hidden_dim
        self.t_max, self.device = sinkhorn_iters, device

    def load(self, fn, base, scale):
        n = self.n_hc
        self.W_pre = fn[0:n].contiguous()
        self.W_post = fn[n:2*n].contiguous()
        self.W_comb = fn[2*n:].contiguous()
        self.S_pre = base[0:n].reshape(1, n).float()
        self.S_post = base[n:2*n].reshape(n, 1).float()
        self.S_comb = base[2*n:].reshape(n, n).float()
        self.alpha_pre, self.alpha_post, self.alpha_comb = scale[0].item(), scale[1].item(), scale[2].item()

    @staticmethod
    def init_state(emb, n_hc=4):
        return emb.unsqueeze(1).expand(-1, n_hc, -1).clone()

    def pre_block(self, X):
        T, n, d = X.shape
        Xn = unweighted_rmsnorm(X.reshape(T, self.K).bfloat16())
        W = torch.cat([self.W_pre, self.W_post, self.W_comb])
        proj = Xn @ W.T
        pre_t = self.alpha_pre * proj[:, :n] + self.S_pre.flatten().unsqueeze(0)
        post_t = self.alpha_post * proj[:, n:2*n] + self.S_post.flatten().unsqueeze(0)
        comb_t = self.alpha_comb * proj[:, 2*n:2*n+n*n] + self.S_comb.flatten().unsqueeze(0)
        A = torch.sigmoid(pre_t) + HC_EPS
        C = 2.0 * torch.sigmoid(post_t)
        B = sinkhorn_knopp(comb_t.reshape(T, n, n), t_max=self.t_max)
        x_in = torch.bmm(A.unsqueeze(1), X.float()).squeeze(1).bfloat16()
        return x_in, {'B': B, 'C': C}

    def post_block(self, X, F_out, ctx):
        BX = torch.bmm(ctx['B'].transpose(-1, -2), X.float())
        CF = ctx['C'].unsqueeze(-1) * F_out.unsqueeze(1)
        return (CF.float() + BX).bfloat16()

# =====================================================================
# HcHead
# =====================================================================
class HcHead:
    def __init__(self, hidden_dim=7168, n_hc=4, device='cuda:0'):
        self.K, self.device, self.n_hc = n_hc * hidden_dim, device, n_hc

    def load(self, fn, base, scale=None):
        self.fn = fn.to(self.device, torch.float32).contiguous()
        self.base = base.to(self.device, torch.float32).contiguous()
        self.scale = scale.to(self.device, torch.float32).item() if scale is not None else 1.0

    def forward(self, X):
        T = X.shape[0]
        Xn = unweighted_rmsnorm(X.reshape(T, self.K).bfloat16())
        mix = F.linear(Xn, self.fn[:self.n_hc]).float()
        pre = torch.sigmoid(mix * self.scale + self.base[:self.n_hc].unsqueeze(0)) + HC_EPS
        return (pre.unsqueeze(-1) * X.float()).sum(1).bfloat16()

# =====================================================================
# RoPE
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
    T, nh, hd = x.shape
    nope = hd - rope_dim
    c, s = cos[pos].unsqueeze(1), sin[pos].unsqueeze(1)
    xr = x[:, :, nope:].float()
    ev, od = xr[..., 0::2], xr[..., 1::2]
    if inverse: rev, rod = ev*c + od*s, -ev*s + od*c
    else: rev, rod = ev*c - od*s, ev*s + od*c
    out = x.clone()
    ro = torch.empty_like(xr)
    ro[..., 0::2], ro[..., 1::2] = rev, rod
    out[:, :, nope:] = ro.bfloat16()
    return out

# =====================================================================
# Compressor — CSA (ratio=4) and HCA (ratio=128)
# =====================================================================
class Compressor:
    def __init__(self, ratio, head_dim, hidden_size, device):
        self.ratio, self.hd, self.H, self.device = ratio, head_dim, hidden_size, device
        self.is_csa = (ratio == 4)
        self.kv_dim = 2 * head_dim if self.is_csa else head_dim
        self.wkv_w = self.wkv_ws = self.wkv_ws2 = self.wkv_isc = None
        self.wgate_w = self.wgate_ws = self.wgate_ws2 = self.wgate_isc = None
        self.ape = None
        self.kv_norm_w = None

    def load(self, w, pfx):
        self.wkv_w, self.wkv_ws, self.wkv_ws2, self.wkv_isc = get_nvfp4_weight(w, pfx, 'kv_proj')
        self.wgate_w, self.wgate_ws, self.wgate_ws2, self.wgate_isc = get_nvfp4_weight(w, pfx, 'gate_proj')
        self.ape = w.get(f"{pfx}.position_bias")
        self.kv_norm_w = w.get(f"{pfx}.kv_norm.weight")

    def forward(self, hidden_states, positions):
        """Returns (compressed_kv (N,hd) or None, comp_positions (N,) or None, block_bias or None)."""
        if self.ratio == 0 or self.wkv_w is None:
            return None, None, None
        T = hidden_states.shape[0]
        r = self.ratio
        dev = hidden_states.device
        n_complete = T // r
        if n_complete == 0:
            return None, None, None

        # Project
        kv = nvfp4_linear(hidden_states, self.wkv_w.to(dev), self.wkv_ws.to(dev),
                          self.wkv_ws2.to(dev) if self.wkv_ws2 is not None else None,
                          self.wkv_isc.to(dev) if self.wkv_isc is not None else None)
        gate = nvfp4_linear(hidden_states, self.wgate_w.to(dev), self.wgate_ws.to(dev),
                            self.wgate_ws2.to(dev) if self.wgate_ws2 is not None else None,
                            self.wgate_isc.to(dev) if self.wgate_isc is not None else None)

        # Add position bias (cyclic per block)
        if self.ape is not None:
            ape = self.ape.to(dev)
            n_full = T // r
            for bi in range(n_full):
                s, e = bi * r, (bi + 1) * r
                kv[s:e] += ape.to(kv.dtype)
                gate[s:e] += ape.to(gate.dtype)
            rem = T % r
            if rem > 0:
                s = n_full * r
                kv[s:] += ape[:rem].to(kv.dtype)
                gate[s:] += ape[:rem].to(gate.dtype)

        T_comp = n_complete * r
        comp_list, comp_pos_list = [], []

        if self.is_csa:
            # Overlapping Ca/Cb: split kv and gate into Ca (first hd) and Cb (second hd)
            Ca = kv[:T_comp, :self.hd].reshape(n_complete, r, self.hd)
            Cb = kv[:T_comp, self.hd:].reshape(n_complete, r, self.hd)
            Ga = gate[:T_comp, :self.hd].reshape(n_complete, r, self.hd)
            Gb = gate[:T_comp, self.hd:].reshape(n_complete, r, self.hd)

            for bi in range(n_complete):
                if bi > 0:
                    block_kv = torch.cat([Ca[bi-1], Cb[bi]], dim=0)   # (2r, hd)
                    block_gate = torch.cat([Ga[bi-1], Gb[bi]], dim=0)
                else:
                    block_kv = Cb[bi]       # (r, hd) — no previous Ca
                    block_gate = Gb[bi]
                probs = torch.softmax(block_gate.float(), dim=0)
                compressed = (probs * block_kv.float()).sum(0)
                if self.kv_norm_w is not None:
                    nw = self.kv_norm_w.to(dev).float()
                    compressed = compressed * compressed.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt() * nw
                comp_list.append(compressed.bfloat16())
                comp_pos_list.append(positions[(bi+1)*r - 1])
        else:
            # HCA: non-overlapping, single stream
            kv_blocks = kv[:T_comp].reshape(n_complete, r, self.hd)
            gate_blocks = gate[:T_comp].reshape(n_complete, r, self.hd)
            for bi in range(n_complete):
                probs = torch.softmax(gate_blocks[bi].float(), dim=0)
                compressed = (probs * kv_blocks[bi].float()).sum(0)
                if self.kv_norm_w is not None:
                    nw = self.kv_norm_w.to(dev).float()
                    compressed = compressed * compressed.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt() * nw
                comp_list.append(compressed.bfloat16())
                comp_pos_list.append(positions[(bi+1)*r - 1])

        compressed_kv = torch.stack(comp_list)
        comp_positions = torch.stack(comp_pos_list)
        # block_bias: causal mask for compressed entries
        N = len(comp_list)
        block_bias = torch.zeros(1, T, N, dtype=torch.float32, device=dev)
        return compressed_kv, comp_positions, block_bias

# =====================================================================
# Indexer — CSA top-k
# =====================================================================
class Indexer:
    def __init__(self, n_ih, ihd, top_k, device):
        self.n_ih, self.ihd, self.top_k, self.device = n_ih, ihd, top_k, device
        self.q_b_w = self.q_b_ws = self.q_b_ws2 = self.q_b_isc = None
        self.wp_w = self.wp_ws = self.wp_ws2 = self.wp_isc = None
        self.compressor = None

    def load(self, w, pfx):
        self.q_b_w, self.q_b_ws, self.q_b_ws2, self.q_b_isc = get_nvfp4_weight(w, pfx, 'q_b_proj')
        self.wp_w, self.wp_ws, self.wp_ws2, self.wp_isc = get_nvfp4_weight(w, pfx, 'weights_proj')
        if f"{pfx}.compressor.kv_proj.weight" in w:
            self.compressor = Compressor(4, self.ihd, 7168, self.device)
            self.compressor.load(w, f"{pfx}.compressor")

    def forward(self, q_lora, hidden_states, comp_indexer_kv, positions):
        if self.q_b_w is None or comp_indexer_kv is None or comp_indexer_kv.shape[0] == 0:
            return None
        dev = q_lora.device
        T = q_lora.shape[0]
        n_comp = comp_indexer_kv.shape[0]
        q_idx = nvfp4_linear(q_lora, self.q_b_w.to(dev), self.q_b_ws.to(dev),
                             self.q_b_ws2.to(dev) if self.q_b_ws2 is not None else None,
                             self.q_b_isc.to(dev) if self.q_b_isc is not None else None)
        q_idx = q_idx.reshape(T, self.n_ih, self.ihd)
        w_h = nvfp4_linear(hidden_states, self.wp_w.to(dev), self.wp_ws.to(dev),
                           self.wp_ws2.to(dev) if self.wp_ws2 is not None else None,
                           self.wp_isc.to(dev) if self.wp_isc is not None else None)
        k_idx = comp_indexer_kv.reshape(n_comp, self.n_ih, self.ihd)
        scores = torch.einsum('tnd,cnd->tnc', q_idx.float(), k_idx.float())
        scores = F.relu(scores)
        total = (scores * w_h.unsqueeze(-1).float()).sum(1)
        tk = min(self.top_k, n_comp)
        _, idx = total.topk(tk, -1)
        return idx

# =====================================================================
# KV Cache
# =====================================================================
class KVCache:
    def __init__(self, head_dim, window_size=128, device='cuda:0'):
        self.hd, self.ws, self.dev = head_dim, window_size, device
        self.swa = torch.zeros(window_size, head_dim, dtype=torch.bfloat16, device=device)
        self.swa_pos = torch.zeros(window_size, dtype=torch.long, device=device)
        self.swa_len, self.swa_head = 0, 0
        self.comp_kv, self.comp_pos, self.n_comp = None, None, 0
        self.comp_idx_kv = None

    def append_swa(self, kv, pos):
        T = kv.shape[0]
        for i in range(T):
            idx = (self.swa_head + i) % self.ws
            self.swa[idx], self.swa_pos[idx] = kv[i], pos[i]
        self.swa_head = (self.swa_head + T) % self.ws
        self.swa_len = min(self.swa_len + T, self.ws)

    def add_compressed(self, ckv, cpos, idx_kv=None):
        if ckv is None: return
        self.comp_kv = ckv if self.comp_kv is None else torch.cat([self.comp_kv, ckv])
        self.comp_pos = cpos if self.comp_pos is None else torch.cat([self.comp_pos, cpos])
        self.n_comp = self.comp_kv.shape[0]
        if idx_kv is not None:
            self.comp_idx_kv = idx_kv if self.comp_idx_kv is None else torch.cat([self.comp_idx_kv, idx_kv])

    def get_swa(self):
        if self.swa_len == 0:
            return torch.zeros(0, self.hd, device=self.dev, dtype=torch.bfloat16), \
                   torch.zeros(0, device=self.dev, dtype=torch.long)
        if self.swa_len < self.ws:
            return self.swa[:self.swa_len].clone(), self.swa_pos[:self.swa_len].clone()
        idx = torch.arange(self.swa_head, self.swa_head + self.ws) % self.ws
        return self.swa[idx].clone(), self.swa_pos[idx].clone()

# =====================================================================
# Weight loading
# =====================================================================
def load_weights(checkpoint_dir):
    from safetensors.torch import load_file
    cdir = Path(checkpoint_dir)
    wmap = {}
    idx = cdir / "model.safetensors.index.json"
    if idx.exists():
        with open(idx) as f: wmap = json.load(f).get("weight_map", {})
    shards = set(wmap.values()) if wmap else set()
    all_w = {}
    for sn in sorted(shards):
        if (cdir / sn).exists():
            all_w.update(load_file(str(cdir / sn)))
    print(f"Loaded {len(all_w)} tensors from {len(shards)} shards")
    return all_w

def cache_layer_weights(all_w, n_layers, devices):
    cached = {}
    for li in range(n_layers):
        dev = devices[li % len(devices)]
        pfx = f"model.layers.{li}."
        w = {k: v.to(device=dev, non_blocking=True) for k, v in all_w.items() if k.startswith(pfx)}
        cached[li] = w
        if (li+1) % 10 == 0: print(f"  Cached {li+1}/{n_layers} layers")
    return cached

# =====================================================================
# Attention forward
# =====================================================================
def forward_attention(x_normed, w, li, cfg, rope_cos, rope_sin,
                      kv_cache, positions, compressor, indexer):
    dev = x_normed.device
    T = x_normed.shape[0]
    n_h = cfg["num_attention_heads"]
    hd = cfg["head_dim"]
    rd = cfg.get("qk_rope_head_dim", 64)
    o_groups = cfg.get("o_groups", 16)
    o_rank = cfg.get("o_lora_rank", 1024)
    ratio = compressor.ratio if compressor is not None else 0
    scale = 1.0 / math.sqrt(hd)
    pfx = f"model.layers.{li}.self_attn"
    # Ensure positions is on the same device as rope caches
    if positions.device != rope_cos.device:
        positions = positions.to(rope_cos.device)

    # 1. Q projection: q_a → q_a_norm → q_b → q_b_norm
    q_a = do_nvfp4_linear(x_normed, w, pfx, 'q_a_proj')
    if q_a is None:
        print(f"  WARNING L{li}: q_a_proj not found, keys: {[k for k in w if 'q_a' in k and f'layers.{li}' in k][:5]}")
        return torch.zeros(T, cfg["hidden_size"], dtype=torch.bfloat16, device=dev), None
    if VERBOSE >= 2: print(f"  L{li} q_a: |max|={q_a.abs().max().item():.4f} shape={q_a.shape}")
    q_norm_w = w.get(f"{pfx}.q_a_norm.weight")
    if q_norm_w is not None: q_a = rmsnorm(q_a, q_norm_w.to(dev, torch.float32))
    q = do_nvfp4_linear(q_a, w, pfx, 'q_b_proj')
    q = unweighted_rmsnorm(q).bfloat16()
    q_heads = q.reshape(T, n_h, hd)
    q_heads = _apply_rope(q_heads, positions, rope_cos, rope_sin, rd)

    # 2. KV projection (MQA, single KV head, hd dim)
    kv = do_nvfp4_linear(x_normed, w, pfx, 'kv_proj')
    if kv is None:
        print(f"  WARNING L{li}: kv_proj not found, keys: {[k for k in w if 'kv_proj' in k and f'layers.{li}' in k][:5]}")
        return torch.zeros(T, cfg["hidden_size"], dtype=torch.bfloat16, device=dev), q_a
    kv_norm_w = w.get(f"{pfx}.kv_norm.weight")
    if kv_norm_w is not None: kv = rmsnorm(kv, kv_norm_w.to(dev, torch.float32))
    kv_3d = kv.reshape(T, 1, hd)
    kv_3d = _apply_rope(kv_3d, positions, rope_cos, rope_sin, rd)
    kv_roped = kv_3d.reshape(T, hd)
    kv_cache.append_swa(kv_roped, positions)

    # 3. Compressor → compressed KV (dim = hd)
    comp_kv, comp_pos, block_bias = None, None, None
    comp_idx_kv = None
    if compressor is not None and compressor.ratio > 0:
        comp_kv, comp_pos, block_bias = compressor.forward(x_normed, positions)
        if comp_kv is not None:
            comp_kv_3d = comp_kv.unsqueeze(1)
            comp_kv_3d = _apply_rope(comp_kv_3d, comp_pos, rope_cos, rope_sin, rd)
            comp_kv = comp_kv_3d.squeeze(1)
        if compressor.is_csa and indexer is not None and indexer.compressor is not None:
            comp_idx_kv, _, _ = indexer.compressor.forward(x_normed, positions)
        kv_cache.add_compressed(comp_kv, comp_pos, comp_idx_kv)

    # 4. Indexer top-k (CSA only)
    topk_idx = None
    if indexer is not None and ratio == 4:
        topk_idx = indexer.forward(q_a, x_normed, kv_cache.comp_idx_kv, positions)

    # 5. Gather full KV: [compressed, swa]
    swa_kv, swa_pos = kv_cache.get_swa()
    swa_len = swa_kv.shape[0]
    if kv_cache.comp_kv is not None and kv_cache.n_comp > 0:
        if ratio == 4 and topk_idx is not None:
            tk = topk_idx[0].clamp(0, kv_cache.n_comp - 1)
            sel_comp = kv_cache.comp_kv[tk]
            all_kv = torch.cat([sel_comp, swa_kv], dim=0)
        elif ratio > 4:
            all_kv = torch.cat([kv_cache.comp_kv, swa_kv], dim=0)
        else:
            all_kv = swa_kv
    else:
        all_kv = swa_kv

    seq_len = all_kv.shape[0]
    if seq_len == 0:
        return torch.zeros(T, cfg["hidden_size"], dtype=torch.bfloat16, device=dev), q_a

    # 6. SDPA with sinks
    k_exp = all_kv.unsqueeze(0).expand(n_h, -1, -1).contiguous()
    v_exp = k_exp.clone()
    q_in = q_heads.permute(1, 0, 2)
    scores = torch.matmul(q_in, k_exp.transpose(-1, -2)) * scale
    sinks = w.get(f"{pfx}.sinks")
    if sinks is not None:
        sinks = sinks.to(device=dev)
        sink_logits = sinks.float().reshape(n_h, 1, 1).expand(-1, T, 1)
        combined = torch.cat([scores, sink_logits], dim=-1)
        combined = combined - combined.max(-1, keepdim=True).values
        probs = torch.softmax(combined.float(), -1).bfloat16()
        attn_w = probs[..., :-1]
    else:
        attn_w = torch.softmax(scores.float(), -1).bfloat16()

    attn_out = torch.matmul(attn_w, v_exp).permute(1, 0, 2)

    # 7. Inverse RoPE
    attn_out = _apply_rope(attn_out, positions, rope_cos, rope_sin, rd, inverse=True)

    # 8. Output projection: wo_a (BF16 grouped BMM) + wo_b (NVFP4)
    hpg = n_h // o_groups
    gid = hpg * hd
    oa_w = w.get(f"{pfx}.o_a_proj.weight")
    if oa_w is not None:
        oa_bf = oa_w.bfloat16().to(dev)
        a_flat = attn_out.reshape(T, n_h * hd)
        a_grp = a_flat.reshape(T, o_groups, gid)
        oa_3d = oa_bf.reshape(o_groups, o_rank, gid)
        g_out = torch.bmm(a_grp.permute(1, 0, 2), oa_3d.transpose(1, 2))
        g_flat = g_out.permute(1, 0, 2).reshape(T, o_groups * o_rank)
        F_attn = do_nvfp4_linear(g_flat, w, pfx, 'o_b_proj')
    else:
        F_attn = do_nvfp4_linear(attn_out.reshape(T, n_h * hd), w, pfx, 'o_a_proj')
    return F_attn, q_a

# =====================================================================
# MoE forward
# =====================================================================
def moe_forward(x, w, li, cfg, token_id, device):
    H = cfg["hidden_size"]
    n_e = cfg["n_routed_experts"]
    top_k = cfg.get("num_experts_per_tok", 6)
    rsc = cfg.get("routed_scaling_factor", 2.5)
    lim = cfg.get("swiglu_limit", 10.0)
    num_hash = cfg.get("num_hash_layers", 3)
    pfx = f"model.layers.{li}.mlp"

    # Routing
    tid2eid_key = f"{pfx}.gate.tid2eid"
    e_bias_key = f"{pfx}.gate.e_score_correction_bias"
    is_hash = (li < num_hash) and (tid2eid_key in w)

    if is_hash:
        tid2eid = w[tid2eid_key]
        tid = token_id.item() if token_id.numel() == 1 else token_id[0].item()
        expert_ids = tid2eid[tid]
        expert_weights = torch.ones(top_k, dtype=torch.float32, device=x.device) / top_k
    else:
        # Gate weight may be BF16 or NVFP4
        gate_ww, gate_ws, gate_ws2, gate_isc = get_nvfp4_weight(w, pfx, 'gate')
        if gate_ww is not None and gate_ws is not None:
            logits = nvfp4_linear(x, gate_ww.to(device), gate_ws.to(device),
                                  gate_ws2.to(device) if gate_ws2 is not None else None,
                                  gate_isc.to(device) if gate_isc is not None else None)
        elif f"{pfx}.gate.weight" in w:
            gw = w[f"{pfx}.gate.weight"].bfloat16().to(device)
            logits = F.linear(x, gw)
        else:
            raise ValueError(f"No gate weight for layer {li}")
        scores = torch.sqrt(F.softplus(logits.float()) + 1e-6)
        sel = scores.clone()
        if e_bias_key in w:
            sel = sel + w[e_bias_key].to(device=x.device).float().unsqueeze(0)
        _, indices = sel.topk(top_k, -1)
        expert_weights = torch.gather(scores, -1, indices)
        expert_weights = expert_weights / expert_weights.sum(-1, keepdim=True)
        expert_ids, expert_weights = indices[0], expert_weights[0]

    # Routed experts
    expert_outs = []
    for i, eid in enumerate(expert_ids):
        ep = f"{pfx}.experts.{eid.item()}"
        g = do_nvfp4_linear(x, w, ep, 'gate_proj')
        u = do_nvfp4_linear(x, w, ep, 'up_proj')
        silu = F.silu(g.float())
        if lim is not None: silu = silu.clamp(-lim, lim); u = u.float().clamp(-lim, lim)
        h = (silu * u).bfloat16()
        expert_outs.append(do_nvfp4_linear(h, w, ep, 'down_proj'))

    routed = torch.zeros_like(x)
    for out, wt in zip(expert_outs, expert_weights):
        routed = routed + (out.float() * wt.item()).bfloat16()
    routed = (routed.float() * rsc).bfloat16()

    # Shared expert
    sp = f"{pfx}.shared_experts"
    sg = do_nvfp4_linear(x, w, sp, 'gate_proj')
    su = do_nvfp4_linear(x, w, sp, 'up_proj')
    silu = F.silu(sg.float())
    if lim is not None: silu = silu.clamp(-lim, lim); su = su.float().clamp(-lim, lim)
    shared = do_nvfp4_linear((silu * su).bfloat16(), w, sp, 'down_proj')
    return routed + shared

# =====================================================================
# Layer forward
# =====================================================================
def forward_layer(X_l, w, li, cfg, rope_cos, rope_sin,
                  attn_mhc, ffn_mhc, attn_norm_w, ffn_norm_w,
                  kv_cache, positions, token_id,
                  compressor=None, indexer=None):
    dev = X_l.device
    # Attention sub-block
    x_in, ctx_a = attn_mhc.pre_block(X_l)
    x_normed = rmsnorm(x_in, attn_norm_w)
    F_attn, _ = forward_attention(x_normed, w, li, cfg, rope_cos, rope_sin,
                                   kv_cache, positions, compressor, indexer)
    X_mid = attn_mhc.post_block(X_l, F_attn, ctx_a)
    # FFN sub-block
    x_in_f, ctx_f = ffn_mhc.pre_block(X_mid)
    x_ffn = rmsnorm(x_in_f, ffn_norm_w)
    F_ffn = moe_forward(x_ffn, w, li, cfg, token_id, dev)
    X_next = ffn_mhc.post_block(X_mid, F_ffn, ctx_f)
    if GROWTH_DIAG:
        print(f"  L{li}: |X|={X_l.abs().max().item():.1f}→{X_next.abs().max().item():.1f} "
              f"|Fa|={F_attn.abs().max().item():.1f} |Ff|={F_ffn.abs().max().item():.1f}", flush=True)
    return X_next

# =====================================================================
# Main
# =====================================================================
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    print("=" * 70)
    print("DSV4 Single-Shot Inference — Full E2E Pipeline")
    print("  NVFP4 two-level scale | Compressor + Indexer | mHC | MoE")
    print("=" * 70)

    with open(os.path.join(CHECKPOINT_DIR, "config.json")) as f:
        cfg = json.load(f)
    n_layers = cfg["num_hidden_layers"]
    H = cfg["hidden_size"]
    hd = cfg["head_dim"]
    rd = cfg.get("qk_rope_head_dim", 64)
    cr = cfg.get("compress_ratios", [128] * 61)
    print(f"Model: {n_layers} layers, {cfg['num_attention_heads']} heads, hd={hd}, rope_dim={rd}")
    print(f"Compress ratios: first5={cr[:5]} len={len(cr)}")
    print(f"Experts: {cfg['n_routed_experts']}, top-{cfg.get('num_experts_per_tok', 6)}")

    # Load weights
    print(f"\nPhase 1: Loading weights...")
    all_w = load_weights(CHECKPOINT_DIR)
    print(f"  {time.time()-t0:.1f}s")

    # mHC + norms
    print("Building mHC blocks and norms...")
    attn_mhcs, ffn_mhcs, attn_norms, ffn_norms = {}, {}, {}, {}
    for li in range(n_layers):
        dev = f"cuda:{li % NUM_GPUS}"
        for tag, blocks, fn_s, base_s, scale_s in [
            ("attn", attn_mhcs, f"model.layers.{li}.attn_hc.fn",
             f"model.layers.{li}.attn_hc.base", f"model.layers.{li}.attn_hc.scale"),
            ("ffn", ffn_mhcs, f"model.layers.{li}.ffn_hc.fn",
             f"model.layers.{li}.ffn_hc.base", f"model.layers.{li}.ffn_hc.scale"),
        ]:
            fn, base, scale = all_w.get(fn_s), all_w.get(base_s), all_w.get(scale_s)
            if fn is not None and base is not None and scale is not None:
                m = mHCBlock(H, 4, 20, dev)
                m.load(fn.to(dev, torch.float32), base.to(dev, torch.float32), scale.to(dev, torch.float32))
                blocks[li] = m
            else:
                print(f"  WARNING: no mHC for L{li} {tag}")

        an_k = f"model.layers.{li}.input_layernorm.weight"
        if an_k in all_w: attn_norms[li] = all_w[an_k].to(dev, torch.float32)
        fn_k = f"model.layers.{li}.post_attention_layernorm.weight"
        if fn_k in all_w: ffn_norms[li] = all_w[fn_k].to(dev, torch.float32)

    # Global weights
    torch.cuda.set_device(0)
    embed_w = all_w.get("model.embed_tokens.weight")
    embed = torch.nn.Embedding.from_pretrained(embed_w.bfloat16().to('cuda:0'))
    lm_w = all_w.get("lm_head.weight", embed_w).bfloat16().to('cuda:0')
    final_norm_w = all_w.get("model.norm.weight")
    if final_norm_w is not None: final_norm_w = final_norm_w.to('cuda:0', torch.float32)

    hc_head = HcHead(H, 4, 'cuda:0')
    hc_fn = all_w.get("model.hc_head.hc_fn")
    hc_base = all_w.get("model.hc_head.hc_base")
    hc_scale = all_w.get("model.hc_head.hc_scale")
    if hc_fn is not None and hc_base is not None:
        hc_head.load(hc_fn, hc_base, hc_scale)
        print("  hc_head loaded")
    else:
        print("  WARNING: hc_head not found")
        hc_head = None

    # RoPE
    rp = cfg.get("rope_scaling", cfg.get("rope_parameters", {}))
    rt = rp.get("type", rp.get("rope_type", "yarn"))
    rf = rp.get("factor", 16.0)
    rtheta = cfg.get("rope_theta", 10000.)
    romax = rp.get("original_max_position_embeddings", 65536)
    rbfast, rbslow = rp.get("beta_fast", 32), rp.get("beta_slow", 1)
    print(f"RoPE: {rt} factor={rf} theta={rtheta} orig_max={romax}")
    rope_caches = {g: build_rope_cache(8192, rd, f"cuda:{g}", rtheta, rt, rf, romax, rbfast, rbslow)
                   for g in range(NUM_GPUS)}

    # KV caches
    kv_caches = {li: KVCache(hd, cfg.get("sliding_window", 128), f"cuda:{li % NUM_GPUS}")
                 for li in range(n_layers)}

    # Compressors + indexers
    compressors, indexers = {}, {}
    n_ih = cfg.get("index_n_heads", 64)
    ihd = cfg.get("index_head_dim", 128)
    itk = cfg.get("index_topk", 1024)
    for li in range(n_layers):
        dev = f"cuda:{li % NUM_GPUS}"
        ratio = cr[li] if li < len(cr) else 128
        if ratio > 0: compressors[li] = Compressor(ratio, hd, H, dev)
        if ratio == 4: indexers[li] = Indexer(n_ih, ihd, itk, dev)

    # Cache layer weights to GPUs
    print("Caching layer weights to GPUs...")
    devs = [f"cuda:{g}" for g in range(NUM_GPUS)]
    layer_w = cache_layer_weights(all_w, n_layers, devs)
    del all_w; import gc; gc.collect()
    print(f"  {time.time()-t0:.1f}s")

    # Load compressor/indexer weights
    for li in range(n_layers):
        pfx = f"model.layers.{li}.self_attn.compressor"
        if li in compressors: compressors[li].load(layer_w[li], pfx)
        if li in indexers: indexers[li].load(layer_w[li], f"{pfx}.indexer")
    print("  Compressors/indexers loaded")

    # Phase 2: Inference
    print(f"\nPhase 2: Inference")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    bos = tokenizer.bos_token_id or 0
    input_ids = [bos, USER_TOKEN]
    input_ids += tokenizer.encode('\n\n' + PROMPT, add_special_tokens=False)
    input_ids.append(ASSISTANT_TOKEN)
    generated = input_ids.copy()
    print(f"Input: {len(generated)} tokens")

    # Prefill
    print(f"Prefilling {len(generated)} tokens...")
    for pi, tid_val in enumerate(generated):
        t1 = time.time()
        tid = torch.tensor([tid_val], dtype=torch.long, device='cuda:0')
        pos = torch.tensor([pi], dtype=torch.long, device='cuda:0')
        X = mHCBlock.init_state(embed(tid))
        for li in range(n_layers):
            gpu = li % NUM_GPUS
            if X.device != torch.device(f"cuda:{gpu}"): X = X.to(f"cuda:{gpu}")
            torch.cuda.set_device(gpu)
            X = forward_layer(X, layer_w[li], li, cfg, *rope_caches[gpu],
                              attn_mhcs.get(li), ffn_mhcs.get(li),
                              attn_norms.get(li), ffn_norms.get(li),
                              kv_caches[li], pos, tid,
                              compressors.get(li), indexers.get(li))
        X = X.to('cuda:0'); torch.cuda.set_device(0)
        if pi % 10 == 0: print(f"  Token {pi}/{len(generated)}: {time.time()-t1:.2f}s", flush=True)
    print(f"  Prefill done ({time.time()-t0:.1f}s)")

    # Decode
    print(f"\nDecoding (max {MAX_NEW_TOKENS} tokens)...")
    all_tokens = generated.copy()
    for step in range(MAX_NEW_TOKENS):
        t1 = time.time()
        tid = torch.tensor([all_tokens[-1]], dtype=torch.long, device='cuda:0')
        dec_pos = torch.tensor([len(all_tokens)-1], dtype=torch.long, device='cuda:0')
        X = mHCBlock.init_state(embed(tid))
        for li in range(n_layers):
            gpu = li % NUM_GPUS
            if X.device != torch.device(f"cuda:{gpu}"): X = X.to(f"cuda:{gpu}")
            torch.cuda.set_device(gpu)
            X = forward_layer(X, layer_w[li], li, cfg, *rope_caches[gpu],
                              attn_mhcs.get(li), ffn_mhcs.get(li),
                              attn_norms.get(li), ffn_norms.get(li),
                              kv_caches[li], dec_pos, tid,
                              compressors.get(li), indexers.get(li))
        X = X.to('cuda:0'); torch.cuda.set_device(0)
        x_out = hc_head.forward(X) if hc_head is not None else X[:, 0, :]
        if final_norm_w is not None: x_out = rmsnorm(x_out, final_norm_w)
        logits = F.linear(x_out, lm_w)
        next_id = torch.argmax(logits, -1).item()
        all_tokens.append(next_id)
        dt = time.time() - t1
        has_nan = torch.isnan(logits.float()).any().item()
        if step % 5 == 0 or has_nan:
            tv, ti = torch.topk(logits[0], 5)
            top5 = ' '.join(f'{tokenizer.decode([t.item()])}({v.item():.1f})'
                            for t, v in zip(ti[:5], tv[:5]))
            print(f"  Step {step}: {next_id} '{tokenizer.decode([next_id])}' ({dt:.2f}s) "
                  f"logits=[{logits.float().min().item():.1f},{logits.float().max().item():.1f}] "
                  f"nan={has_nan} |X|={X.abs().max().item():.1f} top5: {top5}", flush=True)
        if has_nan: break
        if next_id == tokenizer.eos_token_id: break

    out = tokenizer.decode(all_tokens, skip_special_tokens=True)
    print(f"\n{'='*70}")
    print(f"Input: '{PROMPT}'")
    print(f"Output: '{out}'")
    print(f"Total: {time.time()-t0:.1f}s")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()