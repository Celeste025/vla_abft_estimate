"""Triton 2D GEMM + bias + 输出侧 |c|>threshold 清零（展平 x 与 F.linear 一致）。

decode M=1 与 tutorial matmul 共用同一 kernel（BLOCK_SIZE_M 最小 16），效率低于 cuBLAS GEMV。
第一版不实现 o_proj 输入侧保护（与 inject 双站点略有差异）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _cuda_autotune_configs():
    """Tutorial matmul configs + 小 M（decode）覆盖。"""
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        # decode / 小 batch：BLOCK_SIZE_M 取 tl.dot 允许的最小值附近
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
        ),
    ]


@triton.autotune(configs=_cuda_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def fused_linear_outlier_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    threshold,
    HAS_BIAS: tl.constexpr,
    DO_OUTLIER: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
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
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - kk * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - kk * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        bias_vals = bias_vals.to(tl.float32)
        accumulator = accumulator + bias_vals[None, :]
    if DO_OUTLIER:
        accumulator = tl.where(tl.abs(accumulator) > threshold, 0.0, accumulator)

    if IS_BF16:
        c = accumulator.to(tl.bfloat16)
    else:
        c = accumulator.to(tl.float16)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def fused_linear_outlier(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    threshold: float,
    *,
    do_outlier: bool = True,
) -> torch.Tensor:
    """
    x: [..., K]，weight: [N, K]（HF nn.Linear.weight），与 F.linear 一致。
    B 逻辑布局为 (K,N)：stride_bk = weight.stride(1), stride_bn = weight.stride(0)。
    """
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"fused_linear_outlier expects bf16/fp16 x, got {x.dtype}")
    if not weight.is_contiguous():
        weight = weight.contiguous()
    orig = x.shape
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    M, Kdim = x2.shape
    N, Kin = weight.shape
    if Kin != Kdim:
        raise ValueError(f"K mismatch: x has {Kdim}, weight has {Kin}")

    c2 = torch.empty((M, N), device=x.device, dtype=x.dtype)
    is_bf16 = x.dtype == torch.bfloat16
    bias_arg = bias if bias is not None else x2
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    fused_linear_outlier_kernel[grid](
        x2,
        weight,
        c2,
        bias_arg,
        M,
        N,
        Kdim,
        x2.stride(0),
        x2.stride(1),
        weight.stride(1),
        weight.stride(0),
        c2.stride(0),
        c2.stride(1),
        float(threshold),
        HAS_BIAS=(bias is not None),
        DO_OUTLIER=bool(do_outlier),
        IS_BF16=is_bf16,
    )
    return c2.view(*orig[:-1], N)


class TritonProtectLinear(nn.Module):
    """与 ProtectLinear 参数形状一致；forward 走同一 Triton GEMM kernel。

    - ``do_outlier=True``：matmul+bias 后在 kernel 内做 |c|>threshold 清零（与 protect_triton 对应）。
    - ``do_outlier=False``：仅 matmul+bias，用于与上者对照（普通 Triton linear）。

    ``protect_input`` 保留字段以便对齐 inject 站点枚举，当前版本**不生效**。
    """

    def __init__(
        self,
        linear: nn.Module,
        threshold: float,
        *,
        protect_input: bool = False,
        do_outlier: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(linear)}")
        self.threshold = float(threshold)
        self.protect_input = bool(protect_input)
        self.do_outlier = bool(do_outlier)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_outlier(
            x, self.weight, self.bias, self.threshold, do_outlier=self.do_outlier
        )


def _sanity():
    if not torch.cuda.is_available():
        print("skip sanity: no CUDA")
        return
    torch.manual_seed(0)
    device = "cuda"
    for dtype in (torch.bfloat16, torch.float16):
        K, N = 256, 512
        x = torch.randn(3, 17, K, device=device, dtype=dtype)
        linear = nn.Linear(K, N, bias=True, device=device, dtype=dtype)
        thr = 2.0
        ref = F.linear(x, linear.weight, linear.bias)
        plain = fused_linear_outlier(x, linear.weight, linear.bias, thr, do_outlier=False)
        assert torch.allclose(plain, ref, atol=1e-2, rtol=1e-2), "plain vs F.linear"
        fused = fused_linear_outlier(x, linear.weight, linear.bias, thr, do_outlier=True)
        ref_clip = torch.where(ref.abs() > thr, torch.zeros_like(ref), ref)
        assert torch.allclose(fused, ref_clip, atol=1e-2, rtol=1e-2), "fused vs where-ref"
    print("triton_protect_linear sanity: OK")


if __name__ == "__main__":
    _sanity()
