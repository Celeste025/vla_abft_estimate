"""Single-entry NCU launcher with fixed Triton config (autotune-disabled).

Usage examples:
    python run_ncu_compare.py --variant triton --dim 1024
    python run_ncu_compare.py --variant abft_v1 --dim 1024
    ncu --clock-control none --import-source yes --set full \\
        --target-processes all -o ncu_reports/stage0_1024_abft_v1 \\
        python run_ncu_compare.py --variant abft_v1 --dim 1024

Fixed config (BM=128, BN=128, BK=32, group_m=8, warps=4, stages=4) matches the
historical clean_std_* runs so per-stage NCU diffs are apples-to-apples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import triton

from matmul_abft_kernels import (
    get_abft_partial_shape_bounds,
    matmul_abft_kernel,
    matmul_kernel,
)


def _build_inputs(dim: int, device: str = "cuda"):
    m = n = k = dim
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float16)
    c = torch.empty((m, n), device=device, dtype=torch.float16)
    num_pid_m, num_pid_n = get_abft_partial_shape_bounds(m, n)
    sum_a_partial = torch.zeros((num_pid_m, k), device=device, dtype=torch.float32)
    sum_b_partial = torch.zeros((num_pid_n, k), device=device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=device, dtype=torch.float32)
    return a, b, c, sum_a_partial, sum_b_partial, sum_c_partial


def _launch_triton_fixed(a, b, c, args):
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, args.block_m) * triton.cdiv(n, args.block_n),)
    matmul_kernel.fn[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=args.block_m,
        BLOCK_SIZE_N=args.block_n,
        BLOCK_SIZE_K=args.block_k,
        GROUP_SIZE_M=args.group_m,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def _launch_abft_v1_fixed(a, b, c, sum_a, sum_b, sum_c, args):
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, args.block_m) * triton.cdiv(n, args.block_n),)
    matmul_abft_kernel.fn[grid](
        a, b, c, sum_a, sum_b, sum_c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=args.block_m,
        BLOCK_SIZE_N=args.block_n,
        BLOCK_SIZE_K=args.block_k,
        GROUP_SIZE_M=args.group_m,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


VARIANT_DISPATCH = {
    "triton": "_launch_triton_fixed",
    "abft_v1": "_launch_abft_v1_fixed",
}


def parse_args():
    p = argparse.ArgumentParser(description="Fixed-config NCU launcher for ABFT matmul.")
    p.add_argument("--variant", required=True, choices=sorted(VARIANT_DISPATCH.keys()))
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-m", type=int, default=128)
    p.add_argument("--block-n", type=int, default=128)
    p.add_argument("--block-k", type=int, default=32)
    p.add_argument("--group-m", type=int, default=8)
    p.add_argument("--num-warps", type=int, default=4)
    p.add_argument("--num-stages", type=int, default=4)
    return p.parse_args()


def run_variant(args):
    a, b, c, sum_a, sum_b, sum_c = _build_inputs(args.dim)
    if args.variant == "triton":
        for _ in range(args.warmup):
            _launch_triton_fixed(a, b, c, args)
        torch.cuda.synchronize()
        for _ in range(args.iters):
            _launch_triton_fixed(a, b, c, args)
    elif args.variant == "abft_v1":
        for _ in range(args.warmup):
            _launch_abft_v1_fixed(a, b, c, sum_a, sum_b, sum_c, args)
        torch.cuda.synchronize()
        for _ in range(args.iters):
            _launch_abft_v1_fixed(a, b, c, sum_a, sum_b, sum_c, args)
    else:
        raise ValueError(args.variant)
    torch.cuda.synchronize()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend required.")
    run_variant(args)
    print(
        f"profile_done variant={args.variant} dim={args.dim} "
        f"BM={args.block_m} BN={args.block_n} BK={args.block_k} "
        f"GM={args.group_m} warps={args.num_warps} stages={args.num_stages}"
    )


if __name__ == "__main__":
    main()
