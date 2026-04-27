import argparse
from pathlib import Path

import torch

import reproduce_tutorial_matmul_bench as bench


def _alloc_common_buffers(a, b):
    m, k = a.shape
    _, n = b.shape
    num_pid_m, num_pid_n = bench.get_abft_partial_shape_bounds(m, n)
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    sum_a_partial = torch.zeros((num_pid_m, k), device=a.device, dtype=torch.float32)
    sum_b_partial = torch.zeros((num_pid_n, k), device=a.device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
    return c, sum_a_partial, sum_b_partial, sum_c_partial


def _run_variant(variant, a, b):
    c, sum_a_partial, sum_b_partial, sum_c_partial = _alloc_common_buffers(a, b)
    m, k = a.shape
    _, n = b.shape
    num_pid_m, num_pid_n = bench.get_abft_partial_shape_bounds(m, n)

    if variant == "cublas":
        torch.matmul(a, b)
    elif variant == "triton":
        bench.matmul(a, b)
    elif variant == "abft_kernel":
        bench.launch_matmul_abft_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial)
    elif variant == "sum_a_only":
        bench.launch_matmul_abft_component_kernel_only(
            a, b, c, sum_a_partial, sum_b_partial, sum_c_partial, do_sum_a=True
        )
    elif variant == "sum_b_only":
        bench.launch_matmul_abft_component_kernel_only(
            a, b, c, sum_a_partial, sum_b_partial, sum_c_partial, do_sum_b=True
        )
    elif variant == "ablate_no_sum_store0":
        ablation_sink = torch.zeros((num_pid_m * num_pid_n,), device=a.device, dtype=torch.float32)
        bench.launch_matmul_abft_ablation_kernel_only(
            a, b, c, sum_a_partial, sum_b_partial, sum_c_partial, ablation_sink, 1
        )
    elif variant == "ablate_sum_no_partial_store":
        ablation_sink = torch.zeros((num_pid_m * num_pid_n,), device=a.device, dtype=torch.float32)
        bench.launch_matmul_abft_ablation_kernel_only(
            a, b, c, sum_a_partial, sum_b_partial, sum_c_partial, ablation_sink, 2
        )
    elif variant == "abft_full":
        bench.matmul_abft(a, b)
    elif variant == "two_stage_full":
        bench.matmul_abft_two_stage(a, b)
    else:
        raise ValueError(f"unsupported variant: {variant}")


def parse_args():
    parser = argparse.ArgumentParser(description="Single-entry profiling launcher for ncu.")
    parser.add_argument(
        "--variant",
        type=str,
        default="abft_kernel",
        choices=[
            "cublas",
            "triton",
            "abft_kernel",
            "sum_a_only",
            "sum_b_only",
            "ablate_no_sum_store0",
            "ablate_sum_no_partial_store",
            "abft_full",
            "two_stage_full",
        ],
        help="Target path to profile.",
    )
    parser.add_argument("--dim", type=int, default=1024, help="Use M=N=K=dim.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=1, help="Profiled iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not bench.is_cuda():
        raise RuntimeError("CUDA backend required.")
    torch.manual_seed(args.seed)
    device = "cuda"
    m = n = k = args.dim
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float16)

    # Warmup avoids JIT/first-run effects in profiled section.
    for _ in range(args.warmup):
        _run_variant(args.variant, a, b)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        _run_variant(args.variant, a, b)
    torch.cuda.synchronize()

    print(f"profile_done,variant={args.variant},dim={args.dim},iters={args.iters}")
    print(
        "ncu_cmd_hint,"
        f"ncu --clock-control none --import-source yes --set full "
        f"--target-processes all python {Path(__file__).name} "
        f"--variant {args.variant} --dim {args.dim} --warmup {args.warmup} --iters {args.iters}"
    )


if __name__ == "__main__":
    main()
