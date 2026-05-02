#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#include <stdexcept>

// Handwritten TensorCore HGEMM (ldmatrix + mma.sync) baseline.
// Scope: only supports square M=N=K=256*i, row-major A/B/C, contiguous.

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

inline void check_cuda(torch::Tensor t, const char* name) {
  if (!t.is_cuda()) throw std::invalid_argument(std::string(name) + " must be CUDA tensor");
  if (t.scalar_type() != torch::kFloat16)
    throw std::invalid_argument(std::string(name) + " must be float16");
  if (t.dim() != 2) throw std::invalid_argument(std::string(name) + " must be 2D");
  if (!t.is_contiguous()) throw std::invalid_argument(std::string(name) + " must be contiguous");
}

}  // namespace

__host__ __device__ __forceinline__ int div_ceil_int(int a, int b) { return (a + b - 1) / b; }

// 128x128 CTA tile, mma2x4, warp2x4 (8 warps, 256 threads)
template <int A_PAD, int B_PAD>
__global__ void __launch_bounds__(256) hgemm_128x128x16_kernel(
    half* __restrict__ A,
    half* __restrict__ B,
    half* __restrict__ C,
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

#pragma unroll
  for (int kt = 0; kt < NUM_K_TILES; ++kt) {
    int load_gmem_a_k = kt * BK + load_smem_a_k;
    int load_gmem_b_k = kt * BK + load_smem_b_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST128BITS(A[load_gmem_a_addr]);
    LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST128BITS(B[load_gmem_b_addr]);
    __syncthreads();

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

      LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]);
      LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]);
    }
  }
}

// 64x64 CTA tile, mma2x2, warp2x2 (4 warps, 128 threads)
template <int A_PAD, int B_PAD>
__global__ void __launch_bounds__(128) hgemm_64x64x16_kernel(
    half* __restrict__ A,
    half* __restrict__ B,
    half* __restrict__ C,
    int M,
    int N,
    int K) {
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int MMA_TILE_M = 2;
  constexpr int MMA_TILE_N = 2;
  constexpr int WARP_TILE_M = 2;
  constexpr int WARP_TILE_N = 4;
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;  // 64
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;  // 64
  constexpr int BK = MMA_K;                              // 16

  const int bx = int(blockIdx.x);
  const int by = int(blockIdx.y);
  const int NUM_K_TILES = div_ceil_int(K, BK);

  __shared__ half s_a[BM][BK + A_PAD];
  __shared__ half s_b[BK][BN + B_PAD];

  const int tid = int(threadIdx.x);      // 0..127
  const int warp_id = tid / WARP_SIZE;   // 0..3
  const int lane_id = tid % WARP_SIZE;
  const int warp_m = warp_id % 2;        // 0,1
  const int warp_n = warp_id / 2;        // 0,1

  int load_smem_a_m = tid / 2;                 // 0..63
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // 0,8
  int load_smem_b_k = tid / 8;                 // 0..15
  int load_smem_b_n = (tid % 8) * 8;           // 0..56

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

#pragma unroll
  for (int kt = 0; kt < NUM_K_TILES; ++kt) {
    int load_gmem_a_k = kt * BK + load_smem_a_k;
    int load_gmem_b_k = kt * BK + load_smem_b_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST128BITS(A[load_gmem_a_addr]);
    LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST128BITS(B[load_gmem_b_addr]);
    __syncthreads();

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

      LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]);
      LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]);
    }
  }
}

torch::Tensor gemm_tc_sm89(torch::Tensor a, torch::Tensor b) {
  check_cuda(a, "a");
  check_cuda(b, "b");
  if (a.size(1) != b.size(0)) throw std::invalid_argument("incompatible shapes");

  int M = int(a.size(0));
  int K = int(a.size(1));
  int N = int(b.size(1));
  if (M != N || N != K) throw std::invalid_argument("only supports square M=N=K");
  if (M % 256 != 0) throw std::invalid_argument("only supports M divisible by 256");

  auto c = torch::empty({M, N}, a.options());

  c10::cuda::CUDAGuard device_guard(a.device());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int A_PAD = 8;
  constexpr int B_PAD = 8;

  if (M <= 256) {
    dim3 block(128);
    dim3 grid(div_ceil_int(N, 64), div_ceil_int(M, 64));
    hgemm_64x64x16_kernel<A_PAD, B_PAD><<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        M,
        N,
        K);
  } else {
    dim3 block(256);
    dim3 grid(div_ceil_int(N, 128), div_ceil_int(M, 128));
    hgemm_128x128x16_kernel<A_PAD, B_PAD><<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        M,
        N,
        K);
  }

  return c;
}

