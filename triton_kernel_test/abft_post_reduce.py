from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _abft_reduce_sumab_dotpartial_kernel(
    sum_a_partial_ptr,
    sum_b_partial_ptr,
    sum_a_ptr,
    sum_b_ptr,
    dot_partial_ptr,
    PM: tl.constexpr,
    PN: tl.constexpr,
    K,
    stride_sum_a_p0,
    stride_sum_a_p1,
    stride_sum_b_p0,
    stride_sum_b_p1,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    acc_a = tl.zeros((BLOCK_K,), dtype=tl.float32)
    # sum_a_partial is (PM, K)
    for p0 in range(0, PM, BLOCK_P):
        p_offsets = p0 + tl.arange(0, BLOCK_P)
        p_mask = p_offsets < PM
        ptrs = sum_a_partial_ptr + p_offsets[:, None] * stride_sum_a_p0 + k_offsets[None, :] * stride_sum_a_p1
        vals = tl.load(ptrs, mask=p_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc_a += tl.sum(vals, axis=0)

    acc_b = tl.zeros((BLOCK_K,), dtype=tl.float32)
    # sum_b_partial is (PN, K)
    for p0 in range(0, PN, BLOCK_P):
        p_offsets = p0 + tl.arange(0, BLOCK_P)
        p_mask = p_offsets < PN
        ptrs = sum_b_partial_ptr + p_offsets[:, None] * stride_sum_b_p0 + k_offsets[None, :] * stride_sum_b_p1
        vals = tl.load(ptrs, mask=p_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        acc_b += tl.sum(vals, axis=0)

    tl.store(sum_a_ptr + k_offsets, acc_a, mask=k_mask)
    tl.store(sum_b_ptr + k_offsets, acc_b, mask=k_mask)

    dot_partial = tl.sum(acc_a * acc_b, axis=0)
    tl.store(dot_partial_ptr + pid_k, dot_partial)


@triton.jit
def _abft_finalize_scalars_kernel(
    dot_partial_ptr,
    sum_c_partial_ptr,
    dot_sum_ptr,
    sum_c_ptr,
    abft_abs_ptr,
    abft_rel_ptr,
    num_dot_partials,
    num_sum_c_partials,
    EPS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # One program is enough for our sizes; keep multi-program safe by only letting pid==0 write.
    if pid != 0:
        return

    dot_acc = tl.zeros((), dtype=tl.float32)
    for o in range(0, num_dot_partials, BLOCK_R):
        offs = o + tl.arange(0, BLOCK_R)
        m = offs < num_dot_partials
        dot_acc += tl.sum(tl.load(dot_partial_ptr + offs, mask=m, other=0.0).to(tl.float32), axis=0)

    sum_c_acc = tl.zeros((), dtype=tl.float32)
    for o in range(0, num_sum_c_partials, BLOCK_R):
        offs = o + tl.arange(0, BLOCK_R)
        m = offs < num_sum_c_partials
        sum_c_acc += tl.sum(tl.load(sum_c_partial_ptr + offs, mask=m, other=0.0).to(tl.float32), axis=0)

    tl.store(dot_sum_ptr, dot_acc)
    tl.store(sum_c_ptr, sum_c_acc)
    abs_err = tl.abs(dot_acc - sum_c_acc)
    tl.store(abft_abs_ptr, abs_err)
    denom = tl.maximum(tl.abs(sum_c_acc), EPS)
    tl.store(abft_rel_ptr, abs_err / denom)


@dataclass(frozen=True)
class AbftPostReduceConfig:
    block_k: int = 256
    block_p: int = 128
    block_r: int = 1024
    eps: float = 1e-8
    num_warps: int = 4
    num_stages: int = 1


def abft_post_reduce_triton(
    sum_a_partial: torch.Tensor,
    sum_b_partial: torch.Tensor,
    sum_c_partial_2d: torch.Tensor,
    *,
    config: AbftPostReduceConfig | None = None,
):
    """Post-reduce ABFT partial buffers on GPU.

    Inputs:
      - sum_a_partial: (PM, K) fp32
      - sum_b_partial: (PN, K) fp32
      - sum_c_partial_2d: (PM, PN) fp32
    Outputs (all on GPU):
      - sum_a: (K,) fp32
      - sum_b: (K,) fp32
      - sum_c: (1,) fp32
      - dot_sum: (1,) fp32
      - abft_abs_error: (1,) fp32
      - abft_rel_error: (1,) fp32
    """
    if config is None:
        config = AbftPostReduceConfig()

    assert sum_a_partial.ndim == 2 and sum_b_partial.ndim == 2 and sum_c_partial_2d.ndim == 2
    assert sum_a_partial.dtype == torch.float32 and sum_b_partial.dtype == torch.float32
    assert sum_c_partial_2d.dtype == torch.float32
    PM, K = sum_a_partial.shape
    PN, K2 = sum_b_partial.shape
    assert K2 == K
    PM2, PN2 = sum_c_partial_2d.shape
    assert PM2 == PM and PN2 == PN
    assert sum_a_partial.is_cuda and sum_b_partial.is_cuda and sum_c_partial_2d.is_cuda

    sum_a = torch.empty((K,), device=sum_a_partial.device, dtype=torch.float32)
    sum_b = torch.empty((K,), device=sum_a_partial.device, dtype=torch.float32)
    num_dot_partials = triton.cdiv(K, config.block_k)
    dot_partial = torch.empty((num_dot_partials,), device=sum_a_partial.device, dtype=torch.float32)

    grid = (num_dot_partials,)
    _abft_reduce_sumab_dotpartial_kernel[grid](
        sum_a_partial,
        sum_b_partial,
        sum_a,
        sum_b,
        dot_partial,
        PM=PM,
        PN=PN,
        K=K,
        stride_sum_a_p0=sum_a_partial.stride(0),
        stride_sum_a_p1=sum_a_partial.stride(1),
        stride_sum_b_p0=sum_b_partial.stride(0),
        stride_sum_b_p1=sum_b_partial.stride(1),
        BLOCK_K=config.block_k,
        BLOCK_P=config.block_p,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    # Flatten sum_c_partial for a simple 1D reduction.
    sum_c_partial_1d = sum_c_partial_2d.reshape((-1,))
    sum_c = torch.empty((1,), device=sum_a_partial.device, dtype=torch.float32)
    dot_sum = torch.empty((1,), device=sum_a_partial.device, dtype=torch.float32)
    abft_abs = torch.empty((1,), device=sum_a_partial.device, dtype=torch.float32)
    abft_rel = torch.empty((1,), device=sum_a_partial.device, dtype=torch.float32)

    _abft_finalize_scalars_kernel[(1,)](
        dot_partial,
        sum_c_partial_1d,
        dot_sum,
        sum_c,
        abft_abs,
        abft_rel,
        num_dot_partials=num_dot_partials,
        num_sum_c_partials=sum_c_partial_1d.numel(),
        EPS=config.eps,
        BLOCK_R=config.block_r,
        num_warps=1,
        num_stages=1,
    )

    return {
        "sum_a": sum_a,
        "sum_b": sum_b,
        "sum_c": sum_c,
        "dot_sum": dot_sum[0],
        "abft_abs_error": abft_abs[0],
        "abft_rel_error": abft_rel[0],
    }

