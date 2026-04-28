import torch
import triton
import triton.language as tl

import reproduce_tutorial_matmul_bench as bench


@triton.jit
def checksum_a_partial_store_kernel(
    a_ptr,
    partial_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    partial = tl.sum(a.to(tl.float32), axis=0)
    out_ptrs = partial_ptr + pid_m * K + offs_k
    tl.store(out_ptrs, partial, mask=offs_k < K)


@triton.jit
def checksum_b_partial_store_kernel(
    b_ptr,
    partial_ptr,
    K,
    N,
    stride_bk,
    stride_bn,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    pid_n = pid // num_pid_k
    pid_k = pid % num_pid_k

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    partial = tl.sum(b.to(tl.float32), axis=1)
    out_ptrs = partial_ptr + pid_n * K + offs_k
    tl.store(out_ptrs, partial, mask=offs_k < K)


def bench_case(s):
    m = n = k = s
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    sum_a = torch.zeros((k,), device="cuda", dtype=torch.float32)
    sum_b = torch.zeros((k,), device="cuda", dtype=torch.float32)
    part_a = torch.empty((triton.cdiv(m, 128), k), device="cuda", dtype=torch.float32)
    part_b = torch.empty((triton.cdiv(n, 128), k), device="cuda", dtype=torch.float32)

    grid_a = (triton.cdiv(m, 128) * triton.cdiv(k, 128),)
    grid_b = (triton.cdiv(n, 128) * triton.cdiv(k, 128),)
    quantiles = [0.5, 0.2, 0.8]

    ms_atomic_kernels, _, _ = triton.testing.do_bench(
        lambda: (
            sum_a.zero_(),
            sum_b.zero_(),
            bench.checksum_a_kernel[grid_a](
                a, sum_a, m, k, a.stride(0), a.stride(1), BLOCK_SIZE_M=128, BLOCK_SIZE_K=128
            ),
            bench.checksum_b_kernel[grid_b](
                b, sum_b, k, n, b.stride(0), b.stride(1), BLOCK_SIZE_N=128, BLOCK_SIZE_K=128
            ),
        ),
        quantiles=quantiles,
    )

    ms_partial_store_kernels, _, _ = triton.testing.do_bench(
        lambda: (
            checksum_a_partial_store_kernel[grid_a](
                a, part_a, m, k, a.stride(0), a.stride(1), BLOCK_SIZE_M=128, BLOCK_SIZE_K=128
            ),
            checksum_b_partial_store_kernel[grid_b](
                b, part_b, k, n, b.stride(0), b.stride(1), BLOCK_SIZE_N=128, BLOCK_SIZE_K=128
            ),
        ),
        quantiles=quantiles,
    )

    ms_partial_store_full, _, _ = triton.testing.do_bench(
        lambda: (
            checksum_a_partial_store_kernel[grid_a](
                a, part_a, m, k, a.stride(0), a.stride(1), BLOCK_SIZE_M=128, BLOCK_SIZE_K=128
            ),
            checksum_b_partial_store_kernel[grid_b](
                b, part_b, k, n, b.stride(0), b.stride(1), BLOCK_SIZE_N=128, BLOCK_SIZE_K=128
            ),
            part_a.sum(dim=0),
            part_b.sum(dim=0),
        ),
        quantiles=quantiles,
    )

    print(
        f"shape={s}^3,"
        f"atomic_kernels_ms={ms_atomic_kernels:.4f},"
        f"partial_store_kernels_ms={ms_partial_store_kernels:.4f},"
        f"partial_store_plus_reduce_ms={ms_partial_store_full:.4f}"
    )


def main():
    for s in [1024, 2048, 4096]:
        bench_case(s)


if __name__ == "__main__":
    main()
