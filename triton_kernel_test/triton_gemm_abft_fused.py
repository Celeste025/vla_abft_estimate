from typing import Dict

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],
    reset_to_zero=["sum_a_ptr", "sum_b_ptr", "sum_c_ptr"],
)
@triton.jit
def matmul_abft_fused_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sum_a_ptr,
    sum_b_ptr,
    sum_c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block_k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = block_k_idx * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)

        partial_a = tl.sum(a.to(tl.float32), axis=0)
        tl.atomic_add(sum_a_ptr + k_offsets, partial_a, mask=(k_offsets < K) & (pid_n == 0))
        partial_b = tl.sum(b.to(tl.float32), axis=1)
        tl.atomic_add(sum_b_ptr + k_offsets, partial_b, mask=(k_offsets < K) & (pid_m == 0))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)
    sum_c_tile = tl.sum(tl.sum(acc, axis=1), axis=0)
    tl.atomic_add(sum_c_ptr, sum_c_tile)


def matmul_abft_fused(a: torch.Tensor, b: torch.Tensor) -> Dict[str, torch.Tensor]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected 2D tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Input tensors must be CUDA tensors.")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Supported dtypes: float16, bfloat16, float32.")
    if a.dtype != b.dtype:
        raise ValueError("Input dtypes must match.")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    sum_a = torch.zeros((K,), device=a.device, dtype=torch.float32)
    sum_b = torch.zeros((K,), device=a.device, dtype=torch.float32)
    sum_c = torch.zeros((1,), device=a.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    matmul_abft_fused_kernel[grid](
        a,
        b,
        c,
        sum_a,
        sum_b,
        sum_c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )

    dot_sum = torch.dot(sum_a, sum_b)
    return {
        "c": c,
        "sum_a": sum_a,
        "sum_b": sum_b,
        "sum_c": sum_c,
        "dot_sum": dot_sum,
        "abft_abs_error": (dot_sum - sum_c[0]).abs(),
        "abft_rel_error": (dot_sum - sum_c[0]).abs() / torch.clamp(sum_c[0].abs(), min=1e-8),
    }
