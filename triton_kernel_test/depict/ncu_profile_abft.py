import argparse
from pathlib import Path

import torch
import triton

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


def _launch_triton_fixed(a, b, c, block_m, block_n, block_k, group_m, num_warps, num_stages):
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    bench.matmul_kernel.fn[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_abft_fixed(
    a,
    b,
    c,
    sum_a_partial,
    sum_b_partial,
    sum_c_partial,
    block_m,
    block_n,
    block_k,
    group_m,
    num_warps,
    num_stages,
):
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    bench.matmul_abft_kernel.fn[grid](
        a,
        b,
        c,
        sum_a_partial,
        sum_b_partial,
        sum_c_partial,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _run_variant(variant, a, b, args):
    c, sum_a_partial, sum_b_partial, sum_c_partial = _alloc_common_buffers(a, b)
    m, k = a.shape
    _, n = b.shape
    num_pid_m, num_pid_n = bench.get_abft_partial_shape_bounds(m, n)
    fixed = args.disable_autotune

    if variant == "cublas":
        torch.matmul(a, b)
    elif variant == "triton":
        if fixed:
            _launch_triton_fixed(
                a,
                b,
                c,
                args.block_m,
                args.block_n,
                args.block_k,
                args.group_m,
                args.num_warps,
                args.num_stages,
            )
        else:
            bench.matmul(a, b)
    elif variant == "abft_kernel":
        if fixed:
            _launch_abft_fixed(
                a,
                b,
                c,
                sum_a_partial,
                sum_b_partial,
                sum_c_partial,
                args.block_m,
                args.block_n,
                args.block_k,
                args.group_m,
                args.num_warps,
                args.num_stages,
            )
        else:
            bench.launch_matmul_abft_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial)
    elif variant == "sum_a_only":
        if fixed:
            raise ValueError("sum_a_only does not support --disable-autotune currently")
        bench.launch_matmul_abft_component_kernel_only(
            a, b, c, sum_a_partial, sum_b_partial, sum_c_partial, do_sum_a=True
        )
    elif variant == "sum_b_only":
        if fixed:
            raise ValueError("sum_b_only does not support --disable-autotune currently")
        bench.launch_matmul_abft_component_kernel_only(
            a, b, c, sum_a_partial, sum_b_partial, sum_c_partial, do_sum_b=True
        )
    elif variant == "ablate_no_sum_store0":
        if fixed:
            raise ValueError("ablate_no_sum_store0 does not support --disable-autotune currently")
        ablation_sink = torch.zeros((num_pid_m * num_pid_n,), device=a.device, dtype=torch.float32)
        bench.launch_matmul_abft_ablation_kernel_only(
            a, b, c, sum_a_partial, sum_b_partial, sum_c_partial, ablation_sink, 1
        )
    elif variant == "ablate_sum_no_partial_store":
        if fixed:
            raise ValueError("ablate_sum_no_partial_store does not support --disable-autotune currently")
        ablation_sink = torch.zeros((num_pid_m * num_pid_n,), device=a.device, dtype=torch.float32)
        bench.launch_matmul_abft_ablation_kernel_only(
            a, b, c, sum_a_partial, sum_b_partial, sum_c_partial, ablation_sink, 2
        )
    elif variant == "abft_full":
        if fixed:
            raise ValueError("abft_full does not support --disable-autotune currently")
        bench.matmul_abft(a, b)
    elif variant == "two_stage_full":
        if fixed:
            raise ValueError("two_stage_full does not support --disable-autotune currently")
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
    parser.add_argument("--disable-autotune", action="store_true", help="Use fixed config and bypass Triton autotuner.")
    parser.add_argument("--block-m", type=int, default=128, help="Fixed BLOCK_SIZE_M for --disable-autotune.")
    parser.add_argument("--block-n", type=int, default=128, help="Fixed BLOCK_SIZE_N for --disable-autotune.")
    parser.add_argument("--block-k", type=int, default=32, help="Fixed BLOCK_SIZE_K for --disable-autotune.")
    parser.add_argument("--group-m", type=int, default=8, help="Fixed GROUP_SIZE_M for --disable-autotune.")
    parser.add_argument("--num-warps", type=int, default=4, help="Fixed num_warps for --disable-autotune.")
    parser.add_argument("--num-stages", type=int, default=4, help="Fixed num_stages for --disable-autotune.")
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
        _run_variant(args.variant, a, b, args)
    torch.cuda.synchronize()

    for _ in range(args.iters):
        _run_variant(args.variant, a, b, args)
    torch.cuda.synchronize()

    print(f"profile_done,variant={args.variant},dim={args.dim},iters={args.iters}")
    print(
        "ncu_cmd_hint,"
        f"ncu --clock-control none --import-source yes --set full "
        f"--target-processes all python {Path(__file__).name} "
        f"--variant {args.variant} --dim {args.dim} --warmup {args.warmup} --iters {args.iters}"
        + (
            f" --disable-autotune --block-m {args.block_m} --block-n {args.block_n} "
            f"--block-k {args.block_k} --group-m {args.group_m} --num-warps {args.num_warps} "
            f"--num-stages {args.num_stages}"
            if args.disable_autotune
            else ""
        )
    )


if __name__ == "__main__":
    main()
