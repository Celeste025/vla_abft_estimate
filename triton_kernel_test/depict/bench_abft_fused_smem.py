"""
Benchmark fused ABFT staging kernel only (does not modify reproduce_tutorial_matmul_bench).

Compares: cuBLAS (torch.matmul), Triton matmul baseline, fused staging ABFT (kernel / full).
"""

from __future__ import annotations

import argparse

import torch
import triton

from abft_fused_smem import launch_matmul_abft_fused_staging_only, matmul_abft_fused_staging
from reproduce_tutorial_matmul_bench import get_abft_partial_shape_bounds, matmul


def _tflops(m: int, n: int, k: int, ms: float) -> float:
    return 2.0 * m * n * k * 1e-12 / (ms * 1e-3)


def _parse_args():
    p = argparse.ArgumentParser(description="Benchmark abft_fused_smem vs baselines.")
    p.add_argument("--m-min", type=int, default=8, help="Start i for M=N=K=128*i.")
    p.add_argument("--m-max", type=int, default=17, help="End i (exclusive).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(args.seed)

    print(
        "provider,M,N,K,TFLOPS,kernel_overhead_pct,full_overhead_pct,"
        "abft_abs_error,abft_rel_error"
    )
    for i in range(args.m_min, args.m_max):
        m = n = k = 128 * i
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]

        ms_ref, _, _ = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        ms_tri, _, _ = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)

        num_pid_m, num_pid_n = get_abft_partial_shape_bounds(m, n)
        c = torch.empty((m, n), device=a.device, dtype=torch.float16)
        sum_a_p = torch.zeros((num_pid_m, k), device=a.device, dtype=torch.float32)
        sum_b_p = torch.zeros((num_pid_n, k), device=a.device, dtype=torch.float32)
        sum_c_p = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)

        ms_stg_k, _, _ = triton.testing.do_bench(
            lambda: launch_matmul_abft_fused_staging_only(a, b, c, sum_a_p, sum_b_p, sum_c_p),
            quantiles=quantiles,
        )
        ms_stg_full, _, _ = triton.testing.do_bench(
            lambda: matmul_abft_fused_staging(a, b)["c"],
            quantiles=quantiles,
        )
        stg_out = matmul_abft_fused_staging(a, b)

        oh_k = (ms_stg_k - ms_tri) / ms_tri * 100.0
        oh_f = (ms_stg_full - ms_tri) / ms_tri * 100.0

        print(f"cublas,{m},{n},{k},{_tflops(m,n,k,ms_ref):.4f},,,,")
        print(f"triton,{m},{n},{k},{_tflops(m,n,k,ms_tri):.4f},,,,")
        print(
            f"abft_staging_kernel,{m},{n},{k},{_tflops(m,n,k,ms_stg_k):.4f},"
            f"{oh_k:.2f},,,"
        )
        print(
            f"abft_staging_full,{m},{n},{k},{_tflops(m,n,k,ms_stg_full):.4f},"
            f"{oh_k:.2f},{oh_f:.2f},"
            f"{stg_out['abft_abs_error'].item():.6e},{stg_out['abft_rel_error'].item():.6e}"
        )
        print("------------------------------------------------------------------")


if __name__ == "__main__":
    main()
