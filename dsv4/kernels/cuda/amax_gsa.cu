/**
 * GPU-only amax → gsa computation.
 * Output: scalar GPU tensor containing gsa = max(|x|) / divisor.
 *
 * No CPU-GPU sync. The output tensor stays on GPU and can be passed
 * directly to CuTeDSL GEMM's global_scale_a parameter via to_cute().
 *
 * This eliminates ~915 CPU-GPU syncs per decode step from Nvfp4Linear,
 * Nvfp4MoE, and Nvfp4SharedExpert.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cstdint>
#include <cfloat>

__global__ void compute_amax_gsa_kernel(
    const __nv_bfloat16* __restrict__ input,
    int n,
    float divisor,
    float* __restrict__ out_gsa
) {
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) {
        float v = fabsf(__bfloat162float(input[i]));
        local_max = fmaxf(local_max, v);
    }

    // Warp reduce max
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, mask));
    }

    __shared__ float s_max[8];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) s_max[warp_id] = local_max;
    __syncthreads();

    if (threadIdx.x == 0) {
        float gmax = 0.0f;
        for (int w = 0; w < 8; w++) gmax = fmaxf(gmax, s_max[w]);
        *out_gsa = fmaxf(gmax, 1e-8f) / divisor;
    }
}

torch::Tensor compute_amax_gsa_cuda(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be BF16");

    int n = x.numel();
    auto options = x.options().dtype(torch::kFloat32);
    auto out = torch::zeros({}, options);

    compute_amax_gsa_kernel<<<1, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        n, (float)divisor,
        out.data_ptr<float>()
    );
    return out;  // scalar GPU tensor — no .item() needed!
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_amax_gsa", &compute_amax_gsa_cuda, "GPU-only amax -> gsa");
}
