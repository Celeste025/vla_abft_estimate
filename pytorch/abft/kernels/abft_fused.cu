#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float v, float* shm) {
  int tid = threadIdx.x;
  shm[tid] = v;
  __syncthreads();
  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shm[tid] += shm[tid + s];
    }
    __syncthreads();
  }
  return shm[0];
}

template <typename scalar_t>
__global__ void abft_corner_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    int64_t m,
    int64_t k,
    int64_t n,
    float* __restrict__ out_abft) {
  constexpr int BLOCK_SIZE = 256;
  __shared__ float shm_a[BLOCK_SIZE];
  __shared__ float shm_b[BLOCK_SIZE];
  int64_t p = blockIdx.x;
  if (p >= k) {
    return;
  }
  int tid = threadIdx.x;
  float s_a_local = 0.0f;
  for (int64_t i = tid; i < m; i += BLOCK_SIZE) {
    s_a_local += static_cast<float>(a[i * k + p]);
  }
  float s_b_local = 0.0f;
  for (int64_t j = tid; j < n; j += BLOCK_SIZE) {
    s_b_local += static_cast<float>(b[p * n + j]);
  }
  float s_a = block_reduce_sum<BLOCK_SIZE>(s_a_local, shm_a);
  float s_b = block_reduce_sum<BLOCK_SIZE>(s_b_local, shm_b);
  if (tid == 0) {
    atomicAdd(out_abft, s_a * s_b);
  }
}

template <typename scalar_t>
__global__ void sum_c_bias_kernel(
    const scalar_t* __restrict__ c,
    const scalar_t* __restrict__ bias,
    bool has_bias,
    int64_t m,
    int64_t n,
    float* __restrict__ out_abft,
    float* __restrict__ out_sumc) {
  constexpr int BLOCK_SIZE = 256;
  __shared__ float shm_sumc[BLOCK_SIZE];
  __shared__ float shm_bias[BLOCK_SIZE];
  int tid = threadIdx.x;
  int64_t gtid = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  int64_t total = m * n;

  float local_sumc = 0.0f;
  for (int64_t idx = gtid; idx < total; idx += stride) {
    local_sumc += static_cast<float>(c[idx]);
  }
  float block_sumc = block_reduce_sum<BLOCK_SIZE>(local_sumc, shm_sumc);
  if (tid == 0) {
    atomicAdd(out_sumc, block_sumc);
  }

  if (has_bias) {
    float local_bias = 0.0f;
    for (int64_t j = gtid; j < n; j += stride) {
      local_bias += static_cast<float>(bias[j]);
    }
    float block_bias = block_reduce_sum<BLOCK_SIZE>(local_bias, shm_bias);
    if (tid == 0) {
      atomicAdd(out_abft, static_cast<float>(m) * block_bias);
    }
  }
}

std::vector<torch::Tensor> abft_fused_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    c10::optional<torch::Tensor> bias) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
  auto out_abft = torch::zeros({1}, opts);
  auto out_sumc = torch::zeros({1}, opts);

  int64_t m = a.size(0);
  int64_t k = a.size(1);
  int64_t n = b.size(1);
  constexpr int threads = 256;
  int corner_blocks = static_cast<int>(k);
  if (corner_blocks < 1) corner_blocks = 1;
  if (corner_blocks > 4096) corner_blocks = 4096;
  int sum_blocks = static_cast<int>((m * n + threads - 1) / threads);
  if (sum_blocks < 1) sum_blocks = 1;
  if (sum_blocks > 1024) sum_blocks = 1024;

  const bool has_bias = bias.has_value();
  auto bias_t = has_bias ? bias.value() : torch::Tensor();

  auto stream = at::cuda::getDefaultCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      a.scalar_type(),
      "abft_fused_kernel",
      ([&] {
        const scalar_t* bias_ptr = has_bias ? bias_t.data_ptr<scalar_t>() : nullptr;
        abft_corner_kernel<scalar_t><<<corner_blocks, threads, 0, stream>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            m,
            k,
            n,
            out_abft.data_ptr<float>());
        sum_c_bias_kernel<scalar_t><<<sum_blocks, threads, 0, stream>>>(
            c.data_ptr<scalar_t>(),
            bias_ptr,
            has_bias,
            m,
            n,
            out_abft.data_ptr<float>(),
            out_sumc.data_ptr<float>());
      }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_abft, out_sumc};
}
