import argparse

import torch
import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    ]


def get_abft_partial_shape_bounds(m, n):
    """Conservative partial-buffer bounds from autotune search space."""
    configs = get_cuda_autotune_config()
    min_block_m = min(cfg.kwargs["BLOCK_SIZE_M"] for cfg in configs)
    min_block_n = min(cfg.kwargs["BLOCK_SIZE_N"] for cfg in configs)
    num_pid_m = triton.cdiv(m, min_block_m)
    num_pid_n = triton.cdiv(n, min_block_n)
    return num_pid_m, num_pid_n


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_kernel[grid](
        a,
        b,
        c,
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
    return c


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
    reset_to_zero=["sum_a_partial_ptr", "sum_b_partial_ptr", "sum_c_partial_ptr"],
)
@triton.jit
def matmul_abft_kernel(
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

    # ABFT diff vs baseline:
    # Keep per-program partial checksum in fp32 and write once per k-block.
    # This avoids vector atomic-add hotspots on global sum_a/sum_b.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offsets = k * BLOCK_SIZE_K + offs_k
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        partial_a = tl.sum(a.to(tl.float32), axis=0)
        partial_b = tl.sum(b.to(tl.float32), axis=1)
        sum_a_ptrs = sum_a_partial_ptr + pid_m * K + k_offsets
        sum_b_ptrs = sum_b_partial_ptr + pid_n * K + k_offsets
        tl.store(sum_a_ptrs, partial_a, mask=(k_offsets < K) & (pid_n == 0))
        tl.store(sum_b_ptrs, partial_b, mask=(k_offsets < K) & (pid_m == 0))

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

    # ABFT diff vs baseline:
    # Write per-program tile sumC to a unique slot, then reduce on host.
    # This removes scalar atomic contention on one global sum_c.
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    sum_c_slot = sum_c_partial_ptr + pid_m * num_pid_n + pid_n
    sum_c_tile = tl.sum(tl.sum(accumulator, axis=1), axis=0)
    tl.store(sum_c_slot, sum_c_tile)


def matmul_abft(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # ABFT diff vs baseline:
    # Allocate partial checksum buffers and reduce after kernel launch.
    # Use autotune search-space bounds instead of hard-coded constants.
    num_pid_m, num_pid_n = get_abft_partial_shape_bounds(M, N)
    sum_a_partial = torch.zeros((num_pid_m, K), device=a.device, dtype=torch.float32)
    sum_b_partial = torch.zeros((num_pid_n, K), device=a.device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_abft_kernel[grid](
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


def matmul_abft_naive(a, b):
    """Naive ABFT: matmul first, then checksum reductions on full tensors."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    c = launch_matmul_naive_kernel_only(a, b)
    sum_a = a.to(torch.float32).sum(dim=0)
    sum_b = b.to(torch.float32).sum(dim=1)
    sum_c = c.to(torch.float32).sum().reshape(1)
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


def launch_matmul_naive_kernel_only(a, b):
    """Naive ABFT kernel stage only: plain triton matmul."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_kernel[grid](
        a,
        b,
        c,
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
    return c


def launch_matmul_abft_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial):
    """Launch ABFT kernel only (no post-kernel reduction/dot)."""
    M, K = a.shape
    _, N = b.shape
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_abft_kernel[grid](
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


def run_abft_sanity(m=1024, n=1024, k=1024, atol=1e-2, rtol=1e-2):
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    out = matmul_abft(a, b)
    c_ref = torch.matmul(a, b)
    sum_a_ref = a.to(torch.float32).sum(dim=0)
    sum_b_ref = b.to(torch.float32).sum(dim=1)
    c_ok = torch.allclose(out["c"], c_ref, atol=atol, rtol=rtol)
    sum_a_max_abs = (out["sum_a"] - sum_a_ref).abs().max().item()
    sum_b_max_abs = (out["sum_b"] - sum_b_ref).abs().max().item()
    print(
        "abft_sanity,"
        f"shape={m}x{n}x{k},"
        f"c_ok={c_ok},"
        f"sum_a_max_abs={sum_a_max_abs:.6e},"
        f"sum_b_max_abs={sum_b_max_abs:.6e},"
        f"abft_abs={out['abft_abs_error'].item():.6e},"
        f"abft_rel={out['abft_rel_error'].item():.6e}"
    )


def run_benchmark(m_min=2, m_max=33):
    if not is_cuda():
        raise RuntimeError("This reproduction script expects CUDA backend.")
    ref_lib = "cublas"
    print(
        "provider,M,N,K,TFLOPS,abft_kernel_overhead_pct,abft_full_overhead_pct,"
        "abft_abs_error,abft_rel_error"
    )
    for i in range(m_min, m_max):
        m = n = k = 128 * i
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]
        ms_ref, _, _ = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        ms_tri, _, _ = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
        perf = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)
        num_pid_m, num_pid_n = get_abft_partial_shape_bounds(m, n)
        c_abft = torch.empty((m, n), device=a.device, dtype=torch.float16)
        sum_a_partial = torch.zeros((num_pid_m, k), device=a.device, dtype=torch.float32)
        sum_b_partial = torch.zeros((num_pid_n, k), device=a.device, dtype=torch.float32)
        sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
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
        abft_kernel_overhead_pct = (ms_abft_kernel - ms_tri) / ms_tri * 100.0
        abft_naive_kernel_overhead_pct = (ms_abft_naive_kernel - ms_tri) / ms_tri * 100.0
        abft_full_overhead_pct = (ms_abft_full - ms_tri) / ms_tri * 100.0
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
            f"triton_abft_naive_kernel,{m},{n},{k},{perf(ms_abft_naive_kernel):.4f},"
            f"{abft_naive_kernel_overhead_pct:.2f},,,"
        )
        print(
            f"triton_abft_naive_full,{m},{n},{k},{perf(ms_abft_naive_full):.4f},"
            f"{abft_naive_kernel_overhead_pct:.2f},{abft_naive_full_overhead_pct:.2f},"
            f"{abft_naive_out['abft_abs_error'].item():.6e},{abft_naive_out['abft_rel_error'].item():.6e}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(description="Reproduce Triton tutorial matmul benchmark.")
    parser.add_argument("--m-min", type=int, default=2, help="Start index i in M=N=K=128*i.")
    parser.add_argument("--m-max", type=int, default=33, help="End index i (exclusive).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    torch.manual_seed(0)
    run_benchmark(m_min=args.m_min, m_max=args.m_max)
