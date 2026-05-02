import argparse
import builtins

import torch
import triton

from matmul_abft_kernels import (
    get_abft_partial_shape_bounds,
    launch_matmul_abft_kernel_only,
    launch_matmul_naive_kernel_only,
    matmul,
    matmul_abft,
    matmul_abft_naive,
)


def run_benchmark(m_min=1, m_max=21, csv_out="benchmark_results.csv"):
    _builtin_print = builtins.print
    csv_rows = []
    ref_lib = "cublas"

    def print(*args, **kwargs):
        line = " ".join(str(x) for x in args)
        if line and not line.startswith("-"):
            csv_rows.append(line)
        _builtin_print(*args, **kwargs)

    print(
        "provider,M,N,K,TFLOPS,abft_kernel_overhead_pct,abft_full_overhead_pct,"
        "abft_abs_error,abft_rel_error"
    )
    for i in range(m_min, m_max):
        m = n = k = 256 * i
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]

        ms_ref, _, _ = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        ms_tri, _, _ = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)

        num_pid_m, num_pid_n = get_abft_partial_shape_bounds(m, n)
        c_abft = torch.empty((m, n), device="cuda", dtype=torch.float16)
        sum_a_partial = torch.zeros((num_pid_m, k), device="cuda", dtype=torch.float32)
        sum_b_partial = torch.zeros((num_pid_n, k), device="cuda", dtype=torch.float32)
        sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device="cuda", dtype=torch.float32)

        ms_abft_kernel, _, _ = triton.testing.do_bench(
            lambda: launch_matmul_abft_kernel_only(a, b, c_abft, sum_a_partial, sum_b_partial, sum_c_partial),
            quantiles=quantiles,
        )
        ms_abft_naive_kernel, _, _ = triton.testing.do_bench(
            lambda: launch_matmul_naive_kernel_only(a, b),
            quantiles=quantiles,
        )
        ms_abft_full, _, _ = triton.testing.do_bench(lambda: matmul_abft(a, b)["c"], quantiles=quantiles)
        ms_abft_naive_full, _, _ = triton.testing.do_bench(lambda: matmul_abft_naive(a, b)["c"], quantiles=quantiles)

        abft_out = matmul_abft(a, b)
        abft_naive_out = matmul_abft_naive(a, b)

        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        abft_kernel_overhead_pct = (ms_abft_kernel - ms_tri) / ms_tri * 100.0
        abft_full_overhead_pct = (ms_abft_full - ms_tri) / ms_tri * 100.0
        abft_naive_kernel_overhead_pct = (ms_abft_naive_kernel - ms_tri) / ms_tri * 100.0
        abft_naive_full_overhead_pct = (ms_abft_naive_full - ms_tri) / ms_tri * 100.0

        print(f"{ref_lib},{m},{n},{k},{perf(ms_ref):.4f},,,,")
        print(f"triton,{m},{n},{k},{perf(ms_tri):.4f},,,,")
        print(
            f"triton_abft_kernel,{m},{n},{k},{perf(ms_abft_kernel):.4f},"
            f"{abft_kernel_overhead_pct:.2f},,,"
        )
        print(
            f"triton_abft_full,{m},{n},{k},{perf(ms_abft_full):.4f},"
            f"{abft_kernel_overhead_pct:.2f},{abft_full_overhead_pct:.2f},"
            f"{abft_out['abft_abs_error'].item():.6e},{abft_out['abft_rel_error'].item():.6e}"
        )
        print(
            f"triton_abft_naive_full,{m},{n},{k},{perf(ms_abft_naive_full):.4f},"
            f"{abft_naive_kernel_overhead_pct:.2f},{abft_naive_full_overhead_pct:.2f},"
            f"{abft_naive_out['abft_abs_error'].item():.6e},{abft_naive_out['abft_rel_error'].item():.6e}"
        )
        print("------------------------------------------------------------------")

    with open(csv_out, "w") as f:
        f.write("\n".join(csv_rows) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(description="Slim ABFT benchmark for Triton matmul experiments.")
    parser.add_argument("--m-min", type=int, default=1, help="Start index i in M=N=K=256*i.")
    parser.add_argument("--m-max", type=int, default=16, help="End index i (exclusive) in M=N=K=256*i.")
    parser.add_argument("--csv-out", type=str, default="benchmark_results.csv", help="Output CSV file path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    torch.manual_seed(0)
    run_benchmark(m_min=args.m_min, m_max=args.m_max, csv_out=args.csv_out)
