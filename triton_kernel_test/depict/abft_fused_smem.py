"""
Fused ABFT with K-direction staging and batched global writes.

Triton 3.1 does not expose user-controlled shared memory from Python; staging
buffers are implemented as block-local tensors (typically register-backed). The
compiler may still place some intermediates in SMEM for tl.dot. The performance
goal matches the plan: fewer fragmented global stores by flushing FLUSH_STEPS
K-tiles in one vector store per sum_a / sum_b.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from reproduce_tutorial_matmul_bench import get_abft_partial_shape_bounds


def get_smem_staging_autotune_config():
    """Conservative configs; FLUSH_STEPS>1 increases register pressure."""
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "FLUSH_STEPS": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "FLUSH_STEPS": 2},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "FLUSH_STEPS": 2},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "FLUSH_STEPS": 2},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "FLUSH_STEPS": 4},
            num_stages=3,
            num_warps=4,
        ),
    ]


@triton.autotune(
    configs=get_smem_staging_autotune_config(),
    key=["M", "N", "K"],
    reset_to_zero=["sum_a_partial_ptr", "sum_b_partial_ptr", "sum_c_partial_ptr"],
)
@triton.jit
def matmul_abft_fused_staging_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sum_a_partial_ptr,
    sum_b_partial_ptr,
    sum_c_partial_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FLUSH_STEPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_batches = (num_k + FLUSH_STEPS - 1) // FLUSH_STEPS

    for bi in range(num_batches):
        k_base_step = bi * FLUSH_STEPS
        partial_a_buf = tl.zeros((FLUSH_STEPS, BLOCK_SIZE_K), dtype=tl.float32)
        partial_b_buf = tl.zeros((FLUSH_STEPS, BLOCK_SIZE_K), dtype=tl.float32)

        for j in tl.static_range(0, FLUSH_STEPS):
            k = k_base_step + j
            in_k = k < num_k
            km = k * BLOCK_SIZE_K
            a_ptrs = a_ptr + offs_am[:, None] * stride_am + (km + offs_k)[None, :] * stride_ak
            b_ptrs = b_ptr + (km + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            a = tl.load(
                a_ptrs,
                mask=in_k & (offs_k[None, :] < K - km),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=in_k & (offs_k[:, None] < K - km),
                other=0.0,
            )
            accumulator = tl.dot(a, b, accumulator)

            row_a = tl.sum(a.to(tl.float32), axis=0)
            row_b = tl.sum(b.to(tl.float32), axis=1)
            pick = (tl.arange(0, FLUSH_STEPS) == j)[:, None]
            partial_a_buf = tl.where(pick, row_a[None, :], partial_a_buf)
            partial_b_buf = tl.where(pick, row_b[None, :], partial_b_buf)

        flat_a = tl.reshape(partial_a_buf, (FLUSH_STEPS * BLOCK_SIZE_K,))
        flat_b = tl.reshape(partial_b_buf, (FLUSH_STEPS * BLOCK_SIZE_K,))
        k0 = k_base_step * BLOCK_SIZE_K
        k_off = k0 + tl.arange(0, FLUSH_STEPS * BLOCK_SIZE_K)
        mask_k = k_off < K
        tl.store(
            sum_a_partial_ptr + pid_m * K + k_off,
            flat_a,
            mask=mask_k & (pid_n == 0),
        )
        tl.store(
            sum_b_partial_ptr + pid_n * K + k_off,
            flat_b,
            mask=mask_k & (pid_m == 0),
        )

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

    num_pid_n_c = tl.cdiv(N, BLOCK_SIZE_N)
    sum_c_slot = sum_c_partial_ptr + pid_m * num_pid_n_c + pid_n
    sum_c_tile = tl.sum(tl.sum(accumulator, axis=1), axis=0)
    tl.store(sum_c_slot, sum_c_tile)


def launch_matmul_abft_fused_staging_only(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    sum_a_partial: torch.Tensor,
    sum_b_partial: torch.Tensor,
    sum_c_partial: torch.Tensor,
) -> None:
    M, K = a.shape
    _, N = b.shape
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_abft_fused_staging_kernel[grid](
        a,
        b,
        c,
        sum_a_partial,
        sum_b_partial,
        sum_c_partial,
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


def matmul_abft_fused_staging(a: torch.Tensor, b: torch.Tensor) -> dict:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    num_pid_m, num_pid_n = get_abft_partial_shape_bounds(M, N)
    sum_a_partial = torch.zeros((num_pid_m, K), device=a.device, dtype=torch.float32)
    sum_b_partial = torch.zeros((num_pid_n, K), device=a.device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
    launch_matmul_abft_fused_staging_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial)
    sum_a = sum_a_partial.sum(dim=0)
    sum_b = sum_b_partial.sum(dim=0)
    sum_c = sum_c_partial.sum().reshape(1)
    dot_sum = torch.dot(sum_a, sum_b)
    abft_abs = (dot_sum - sum_c[0]).abs()
    abft_rel = abft_abs / torch.clamp(sum_c[0].abs(), min=1e-8)
    return {
        "c": c,
        "sum_a": sum_a,
        "sum_b": sum_b,
        "sum_c": sum_c,
        "dot_sum": dot_sum,
        "abft_abs_error": abft_abs,
        "abft_rel_error": abft_rel,
    }
