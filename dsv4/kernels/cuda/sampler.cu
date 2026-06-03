/**
 * Production fused sampler kernel for DSV4 inference.
 *
 * Fused: repetition penalty → temperature → top-k → top-p (nucleus) → sample.
 * Single kernel launch, zero CPU syncs, CUDA-graph-compatible.
 *
 * Architecture:
 *   - 1 CUDA block per batch item
 *   - 256 threads per block
 *   - Each thread scans its slice of the vocab, applies penalty + temperature,
 *     and tracks the top-k candidates using a sorted array in registers
 *   - Thread 0 merges all 256 per-thread top-k lists into a global top-k
 *   - Thread 0 computes softmax over top-k, applies top-p, and samples
 *
 * SMEM: 256 * LOCAL_K * 8 bytes (scores + indices)
 *        = 256 * 32 * 8 = 64KB for LOCAL_K=32
 *        Each thread tracks top-32; the merge considers 256*32=8192 candidates,
 *        yielding an effective top-k of up to 256 (more than enough for any
 *        practical use case).
 *
 * Repetition penalty: passed as (max_penalty, batch, 2) where [:, :, 0] = token_id
 * and [:, :, 1] = penalty_value (multiplicative: >1.0 penalizes, <1.0 boosts).
 * The penalty is applied as: if logit > 0, logit /= penalty; else logit *= penalty.
 * This matches the HuggingFace generate() convention.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cstdint>
#include <cfloat>
#include <curand_kernel.h>

static constexpr int BDIM = 256;
static constexpr int LK = 24;  // per-thread local top-k (SMEM budget: 256*24*8=48KB fits default)

// ---------------------------------------------------------------------------
// Insert into sorted descending array (register-resident, k small)
// ---------------------------------------------------------------------------
__device__ void sorted_insert(float* sc, int* idx, int k, int& n, float s, int i) {
    if (n < k) {
        int p = n;
        while (p > 0 && s > sc[p-1]) { sc[p] = sc[p-1]; idx[p] = idx[p-1]; p--; }
        sc[p] = s; idx[p] = i; n++;
    } else if (s > sc[k-1]) {
        int p = k-1; sc[p] = s; idx[p] = i;
        while (p > 0 && sc[p] > sc[p-1]) {
            float ts=sc[p]; int ti=idx[p]; sc[p]=sc[p-1]; idx[p]=idx[p-1]; sc[p-1]=ts; idx[p-1]=ti; p--;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void fused_sampler_kernel(
    const float*  __restrict__ logits,       // (B, V) stride=vs
    const int64_t* __restrict__ pen_ids,     // (B, max_pen) or nullptr
    const float*  __restrict__ pen_vals,     // (B, max_pen) or nullptr
    int B, int V, int vs, int max_pen,
    float temp, int top_k, float top_p, int min_keep,
    uint64_t seed, uint64_t offset,
    int64_t* __restrict__ out_ids            // (B,)
) {
    int b = blockIdx.x;
    if (b >= B) return;
    int tid = threadIdx.x;
    const float* row = logits + b * vs;

    // ---------- Phase 1: per-thread top-LK ----------
    float lsc[LK]; int lid[LK]; int ln = 0;

    for (int v = tid; v < V; v += BDIM) {
        float val = row[v];
        // Repetition penalty
        if (pen_ids) {
            auto brow = pen_ids + b * max_pen;
            auto vrow = pen_vals + b * max_pen;
            for (int p = 0; p < max_pen; p++) {
                if (brow[p] == v) {
                    val = (val > 0.0f) ? val / vrow[p] : val * vrow[p];
                    break;
                }
            }
        }
        val /= temp;
        sorted_insert(lsc, lid, LK, ln, val, v);
    }

    // ---------- Phase 2: write to SMEM, thread 0 merges ----------
    extern __shared__ char smem[];
    float* s_sc  = reinterpret_cast<float*>(smem);
    int*   s_idx = reinterpret_cast<int*>(smem + BDIM * LK * sizeof(float));

    for (int i = 0; i < ln; i++)  { s_sc[tid*LK+i]  = lsc[i];  s_idx[tid*LK+i]  = lid[i]; }
    for (int i = ln; i < LK; i++) { s_sc[tid*LK+i]  = -FLT_MAX; s_idx[tid*LK+i] = 0; }
    __syncthreads();

    if (tid == 0) {
        // Merge: find global top-k from BDIM * LK = 8192 candidates
        int eff_k = min(top_k, 128);  // kernel max (stack limit: 128 * 8 = 1KB)
        if (eff_k <= 0) eff_k = 128;

        float gsc[128]; int gid[128]; int gn = 0;
        for (int t = 0; t < BDIM; t++) {
            for (int i = 0; i < LK; i++) {
                float s = s_sc[t*LK+i];
                if (s <= -FLT_MAX + 1.0f) continue;
                sorted_insert(gsc, gid, eff_k, gn, s, s_idx[t*LK+i]);
            }
        }

        if (gn == 0) { out_ids[b] = 0; return; }

        // ---------- Phase 3: softmax + top-p + sample ----------
        float mx = gsc[0];  // sorted desc, first is max
        float probs[128]; float total = 0.0f;
        for (int i = 0; i < gn; i++) {
            probs[i] = expf(gsc[i] - mx);
            total += probs[i];
        }

        // Top-p
        int nk = gn;
        if (top_p < 1.0f) {
            float cs = 0.0f;
            for (int i = 0; i < gn; i++) {
                cs += probs[i];
                if (cs / total >= top_p) { nk = max(i+1, min_keep); break; }
            }
        }

        // Renormalize
        float kt = 0.0f;
        for (int i = 0; i < nk; i++) kt += probs[i];

        // Sample
        curandState rng;
        curand_init(seed, b, offset, &rng);
        float r = curand_uniform(&rng) * kt;
        float acc = 0.0f;
        int sel = nk - 1;
        for (int i = 0; i < nk; i++) {
            acc += probs[i];
            if (acc >= r) { sel = i; break; }
        }
        out_ids[b] = gid[sel];
    }
}

// ---------------------------------------------------------------------------
// Binding
// ---------------------------------------------------------------------------
torch::Tensor sample_cuda(
    torch::Tensor logits,
    std::optional<torch::Tensor> pen_ids,
    std::optional<torch::Tensor> pen_vals,
    double temperature,
    int64_t top_k,
    double top_p,
    int64_t min_keep,
    int64_t seed,
    int64_t offset
) {
    TORCH_CHECK(logits.is_contiguous() && logits.dim() == 2 && logits.scalar_type() == torch::kFloat32);
    int B = logits.size(0), V = logits.size(1);
    int mp = 0; const int64_t* pi = nullptr; const float* pv = nullptr;
    if (pen_ids && pen_ids->numel()) { mp = pen_ids->size(1); pi = pen_ids->data_ptr<int64_t>(); pv = pen_vals->data_ptr<float>(); }

    auto options = logits.options().dtype(torch::kInt64);
    auto out = torch::empty({B}, options);
    int smem = BDIM * LK * (sizeof(float) + sizeof(int));

    // Request enough shared memory for 48KB+ per block
    cudaFuncSetAttribute(
        fused_sampler_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem
    );
    // Carveout: prefer more shared memory over L1
    cudaFuncSetAttribute(
        fused_sampler_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    );

    fused_sampler_kernel<<<B, BDIM, smem, c10::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(), pi, pv,
        B, V, logits.stride(0), mp,
        (float)temperature, (int)top_k, (float)top_p, (int)min_keep,
        (uint64_t)seed, (uint64_t)offset,
        out.data_ptr<int64_t>()
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample", &sample_cuda, "Fused top-k/top-p sampler");
}
