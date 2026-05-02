#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#include <stdexcept>

#define WARP_SIZE 32
#define LDST128BITS(value) (reinterpret_cast<int4*>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<int*>(&(value))[0])

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                                   \
  asm volatile(                                                                             \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"                  \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                              \
      : "r"(addr))

#define LDMATRIX_X2_T(R0, R1, addr)                                                         \
  asm volatile(                                                                             \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"                    \
      : "=r"(R0), "=r"(R1)                                                                  \
      : "r"(addr))

#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                         \
  asm volatile(                                                                             \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, "      \
      "{%6, %7}, {%8, %9};\n"                                                               \
      : "=r"(RD0), "=r"(RD1)                                                                \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

namespace {

__host__ __device__ __forceinline__ int div_ceil_int(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

}  // namespace

// ABFT fused into the same K-loop without extra block-level barriers (reuse existing __syncthreads).
// Partial buffers are non-atomic; assignment uses a v2c-like round-robin per kt.
template <int A_PAD, int B_PAD>
__global__ void __launch_bounds__(256) hgemm_128x128x16_abft_kernel(
    half* __restrict__ A,
    half* __restrict__ B,
    half* __restrict__ C,
    float* __restrict__ sum_a_partial,   // (num_pid_m, K)
    float* __restrict__ sum_b_partial,   // (num_pid_n, K)
    float* __restrict__ sum_c_partial,   // (num_pid_m, num_pid_n)
    int M,
    int N,
    int K) {
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int MMA_TILE_M = 2;
  constexpr int MMA_TILE_N = 4;
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;  // 128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;  // 128
  constexpr int BK = MMA_K;                              // 16

  const int bx = int(blockIdx.x);
  const int by = int(blockIdx.y);
  const int num_pid_n = int(gridDim.x);
  const int num_pid_m = int(gridDim.y);
  const int NUM_K_TILES = div_ceil_int(K, BK);

  __shared__ half s_a[BM][BK + A_PAD];
  __shared__ half s_b[BK][BN + B_PAD];

  const int tid = int(threadIdx.x);
  const int warp_id = tid / WARP_SIZE;   // 0..7
  const int lane_id = tid % WARP_SIZE;   // 0..31
  const int warp_m = warp_id % 2;        // 0,1
  const int warp_n = warp_id / 2;        // 0..3

  int load_smem_a_m = tid / 2;                  // 0..127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;   // 0,8
  int load_smem_b_k = tid / 16;                 // 0..15
  int load_smem_b_n = (tid % 16) * 8;           // 0..120

  int load_gmem_a_m = by * BM + load_smem_a_m;
  int load_gmem_b_n = bx * BN + load_smem_b_n;
  if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  // Accumulate sum_c for this CTA (fp32).
  float sum_c_acc = 0.0f;

#pragma unroll
  for (int kt = 0; kt < NUM_K_TILES; ++kt) {
    int k_base = kt * BK;

    int load_gmem_a_k = k_base + load_smem_a_k;
    int load_gmem_b_k = k_base + load_smem_b_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST128BITS(A[load_gmem_a_addr]);
    LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST128BITS(B[load_gmem_b_addr]);
    __syncthreads();

    // v2c-like round-robin: only one pid_n per kt writes sum_a for this pid_m.
    bool do_colsum = ((bx + kt) % num_pid_n) == 0;
    // only one pid_m per kt writes sum_b for this pid_n.
    bool do_rowsum = ((by + kt) % num_pid_m) == 0;

    // Compute sum_a_partial[by, k_base + kk] using warp 0.
    if (do_colsum && warp_id == 0) {
      // Each lane sums 4 rows (BM=128) for each kk in 0..15.
#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        float v = 0.0f;
        int row0 = lane_id * 4 + 0;
        int row1 = lane_id * 4 + 1;
        int row2 = lane_id * 4 + 2;
        int row3 = lane_id * 4 + 3;
        v += __half2float(s_a[row0][kk]);
        v += __half2float(s_a[row1][kk]);
        v += __half2float(s_a[row2][kk]);
        v += __half2float(s_a[row3][kk]);
        v = warp_reduce_sum(v);
        if (lane_id == 0) {
          sum_a_partial[by * K + (k_base + kk)] = v;
        }
      }
    }

    // Compute sum_b_partial[bx, k_base + kk] using warp 1.
    if (do_rowsum && warp_id == 1) {
#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        float v = 0.0f;
        int col0 = lane_id * 4 + 0;
        int col1 = lane_id * 4 + 1;
        int col2 = lane_id * 4 + 2;
        int col3 = lane_id * 4 + 3;
        v += __half2float(s_b[kk][col0]);
        v += __half2float(s_b[kk][col1]);
        v += __half2float(s_b[kk][col2]);
        v += __half2float(s_b[kk][col3]);
        v = warp_reduce_sum(v);
        if (lane_id == 0) {
          sum_b_partial[bx * K + (k_base + kk)] = v;
        }
      }
    }

    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + (lane_id % 16);
      int lane_smem_a_k = (lane_id / 16) * 8;
      uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(&s_a[lane_smem_a_m][lane_smem_a_k]);
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;
      int lane_smem_b_n = warp_smem_b_n;
      uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(&s_b[lane_smem_b_k][lane_smem_b_n]);
      LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2], RA[i][3], RB[j][0],
                  RB[j][1], RC[i][j][0], RC[i][j][1]);
      }
    }
    __syncthreads();
  }

  // Store C and accumulate sum_c for this CTA (fp32) from stored halfs.
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;

      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
      int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
      int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;

      // RC packs 2 halfs in lower 32 bits; convert to float and sum.
      uint32_t r0 = RC[i][j][0];
      uint32_t r1 = RC[i][j][1];
      half2 h0 = *reinterpret_cast<half2*>(&r0);
      half2 h1 = *reinterpret_cast<half2*>(&r1);
      sum_c_acc += __half2float(h0.x) + __half2float(h0.y);
      sum_c_acc += __half2float(h1.x) + __half2float(h1.y);

      LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(r0);
      LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(r1);
    }
  }

  // Reduce sum_c within warp then across warps via atomic add.
  float v = warp_reduce_sum(sum_c_acc);
  if (lane_id == 0) {
    atomicAdd(&sum_c_partial[by * num_pid_n + bx], v);
  }
}

torch::Tensor gemm_tc_sm89_abft(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor sum_a_partial,
    torch::Tensor sum_b_partial,
    torch::Tensor sum_c_partial) {
  if (!a.is_cuda() || !b.is_cuda()) throw std::invalid_argument("a/b must be CUDA");
  if (a.scalar_type() != torch::kFloat16 || b.scalar_type() != torch::kFloat16)
    throw std::invalid_argument("a/b must be fp16");
  if (!a.is_contiguous() || !b.is_contiguous()) throw std::invalid_argument("a/b must be contiguous");
  int M = int(a.size(0));
  int K = int(a.size(1));
  int N = int(b.size(1));
  if (M != N || N != K) throw std::invalid_argument("only supports square M=N=K");
  if (M % 256 != 0) throw std::invalid_argument("only supports M divisible by 256");
  if (sum_a_partial.scalar_type() != torch::kFloat32 || sum_b_partial.scalar_type() != torch::kFloat32 ||
      sum_c_partial.scalar_type() != torch::kFloat32)
    throw std::invalid_argument("partial buffers must be fp32");

  auto c = torch::empty({M, N}, a.options());
  c10::cuda::CUDAGuard device_guard(a.device());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  // grid based on 128x128 tiles
  dim3 block(256);
  dim3 grid(div_ceil_int(N, 128), div_ceil_int(M, 128));

  // Zero sum_c_partial (sum_a/b are fully written by round-robin; sum_c uses atomicAdd).
  // Caller is expected to pass zero-initialized buffers.
  constexpr int A_PAD = 8;
  constexpr int B_PAD = 8;
  hgemm_128x128x16_abft_kernel<A_PAD, B_PAD><<<grid, block, 0, stream>>>(
      reinterpret_cast<half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<half*>(b.data_ptr<at::Half>()),
      reinterpret_cast<half*>(c.data_ptr<at::Half>()),
      sum_a_partial.data_ptr<float>(),
      sum_b_partial.data_ptr<float>(),
      sum_c_partial.data_ptr<float>(),
      M,
      N,
      K);

  return c;
}

