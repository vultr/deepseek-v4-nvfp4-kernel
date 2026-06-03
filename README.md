# deepseek-v4-nvfp4-blackwell

A from-scratch, single-node **NVFP4 inference stack for the full DeepSeek-V4 model** on NVIDIA Blackwell (B200). Not just the MoE — the entire model: Compressed Sparse Attention (CSA), Heavily Compressed Attention (HCA), the lightning indexer, Manifold-Constrained Hyper-Connections (mHC), and the Mixture-of-Experts, all running in NVFP4 wherever the math allows.

> **Status:** research / reference implementation. It runs the full model end-to-end on a single B200 node and is the basis for our vLLM and SGLang integrations. See [Precision strategy](#precision-strategy) for exactly what is and isn't quantized, and [Status & results](#status--results) before deploying.

---

## Why this exists


Our goal was simply: **serve more concurrent requests at full (1M-token) context from a single B200 node.** That means (1) push as much of the model as possible to native NVFP4 to shrink the weight footprint and free VRAM for KV cache, and (2) eliminate tensor-parallel communication by fitting the whole model on one node. To do that we had to hand-write the entire attention/indexer/compressor stack in CuTeDSL/CUDA, because no existing kernel does full-model NVFP4 for this architecture.

## Approach: faithful to the paper, native to Blackwell

We implemented DeepSeek-V4 as described in the paper rather than mapping it onto an existing backend:

- **Hybrid attention** — true CSA (m=4, overlapped 2m compression) and HCA (m'=128, dense), with shared-KV MQA, grouped output projection, the sliding-window branch, attention sink, and partial (last-64-dim) RoPE. We did **not** reuse Multi-head Latent Attention as a stop-gap; the attention is built to the paper.
- **Lightning indexer** — the FP4 weighted-ReLU MQA scorer + top-k selection that drives CSA's sparse attention.
- **mHC** — the expanded residual stream with Sinkhorn-Knopp doubly-stochastic mixing (fixed 20 iterations), kept non-expansive per the paper.
- **MoE** — NVFP4 routed and shared experts with Sqrt(Softplus) affinity and hash routing on the early layers.
- **Blackwell-native math** — FP4 (`f8f6f4`) tensor-core GEMMs for the dense and expert paths, a mixed-precision FP8 FMHA (FP8 noPE + BF16 RoPE), and FP8 KV cache, so the hot path runs on Blackwell's low-precision tensor cores rather than falling back to BF16/FP32.

## Precision strategy

This kernel is designed **exclusively** for the [`vultr/DeepSeek-V4-Pro-NVFP4`](https://huggingface.co/vultr/DeepSeek-V4-Pro-NVFP4) checkpoint, which quantizes **every weight in the model to NVFP4**. That's the starting point — flatten everything to 4-bit, then walk back the components that can't handle it. The loader dequantizes the parts that need higher precision (the router gate, compressor helpers, and LM head) back to BF16 or FP8 at load time. This is intentional: rather than maintaining a mixed-precision checkpoint, we keep the on-disk format uniformly NVFP4 and handle the exceptions in the loading path.

We started by quantizing **everything** to NVFP4 — the most aggressive possible target — to find out how far 4-bit could actually go for this architecture. The answer: very far, but not all the way. With the whole model flattened to NVFP4, generation degenerated: a handful of components are quantization-sensitive in ways 4-bit can't absorb. So we walked those back, one at a time, to find the empirical **precision floor** — the minimal set of components that must stay higher-precision for the model to stay coherent.

The result is a model that is **natively NVFP4 except for three deliberate exceptions**, each of which maps to a known sensitivity class:

| Component | Precision | Why |
|---|---|---|
| Attention projections (Q/K/V/O) | **NVFP4** | Tolerates 4-bit |
| MoE routed + shared experts | **NVFP4** | The dominant weight cost; tolerates 4-bit |
| Most dense/linear layers + activations | **NVFP4** | Tolerates 4-bit |
| **CSA lightning-indexer QK path** | **NVFP4** | Explicitly FP4-QATed by the base model; the top-k selection is error-tolerant |
| **LM head** | **BF16** | Output projection — its error isn't normalized away and decides the token distribution directly |
| **Router gate** | **BF16** | Discrete top-k expert selection — a small error flips which experts fire, a catastrophic (not smooth) error |
| **Compressor / indexer helper projections** (non-QATed) | **BF16** | PTQ to 4-bit on transforms the model never learned to absorb in 4-bit |
| KV cache | **FP8 (E4M3) noPE + BF16 RoPE** | 4-bit KV is too coarse and compounds over 61 layers; FP8 is the right floor |

The walk-back costs almost nothing on VRAM — the gate is tiny, the compressor helpers are small, and the LM head is the only sizeable exception (~1–2 GB) — so the model keeps essentially the entire NVFP4 weight-and-KV saving while regaining coherence. The generalizable finding: for this architecture class, keep the **output unembedding, discrete routing, and non-QATed transforms** out of FP4; everything else, including QATed selection paths, tolerates it.

## Repository layout

The implementation is organized by model component. The clearest entry point for understanding how the whole model is wired is the monolithic reference.

| Path | What it is |
|---|---|
| `single_shot_inference.py` | The monolithic end-to-end reference: loads the quantized checkpoint and runs the full forward (prefill + decode) for a single sequence. Start here to see how every component connects. |
| `dsv4/kernels/attention/` | The mixed-precision FP8 FMHA (decode + prefill), using Blackwell `f8f6f4` tensor cores for the noPE QK path and BF16 for RoPE/PV, with attention sink and online softmax. |
| `dsv4/kernels/compressor/` | CSA (overlapped, m=4) and HCA (dense, m'=128) compression → RMSNorm → RoPE → FP8 quant → KV-cache write. |
| `dsv4/kernels/indexer/` | The lightning indexer: FP4 weighted-ReLU MQA scoring + top-k selection for CSA. |
| `dsv4/layers/mhc.py`, `dsv4/kernels/cuda/mhc_sinkhorn.cu` | Manifold-Constrained Hyper-Connections and the fixed-iteration Sinkhorn-Knopp mixing. |
| `dsv4/layers/moe.py`, `dsv4/layers/shared_expert.py` | NVFP4 routed and shared experts (fused SwiGLU FP4 GEMM). |
| `dsv4/layers/router.py` | Hash routing (early layers) and dense routing (BF16 gate → top-k). |
| `dsv4/layers/linear.py`, `dsv4/ops/quantize.py` | NVFP4 linear and the NVFP4 quantize/dequantize ops. |
| `dsv4/kernels/cuda/` | The CUDA kernels (FMHA C-API, compression, dequant, Sinkhorn, sampler, etc.). |
| `dsv4/reference/` | Pure-PyTorch reference oracles used to validate the kernels per-layer. |

## Requirements

- **Hardware:** NVIDIA Blackwell (B200). NVFP4 is Blackwell-native; this will not run on earlier architectures.
- **Weights:** the matching NVFP4 quantization of DeepSeek-V4-Pro — see **[the model card](https://huggingface.co/vultr/DeepSeek-V4-Pro-NVFP4)**. This is a more aggressively-NVFP4 quant than the stock checkpoint and requires this stack to run; it will not load on unmodified vLLM/SGLang.

## Status & results

> ⚠️ **Experimental.** This is a research reference implementation under active development. Expect breaking changes, rough edges, and incomplete features. Do not use in production.

## Disclaimer

This software is provided by Vultr as-is, without warranty of any kind. It is an independent inference implementation for a third-party model (DeepSeek-V4, owned and developed by DeepSeek); Vultr does not own or develop the underlying model. See the model card for model ownership, license, and limitations.

## License

This inference stack is released under the [MIT License](LICENSE). The DeepSeek-V4 model and its weights are governed by their own [license](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/LICENSE).