import torch
import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_matmul_autotune_config():
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
    configs = get_matmul_autotune_config()
    min_block_m = min(cfg.kwargs["BLOCK_SIZE_M"] for cfg in configs)
    min_block_n = min(cfg.kwargs["BLOCK_SIZE_N"] for cfg in configs)
    num_pid_m = triton.cdiv(m, min_block_m)
    num_pid_n = triton.cdiv(n, min_block_n)
    return num_pid_m, num_pid_n


@triton.autotune(
    configs=get_matmul_autotune_config(),
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
    # Same candidate set as matmul_kernel; autotune result is still kernel-local.
    configs=get_matmul_autotune_config(),
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pid_n_is_zero = pid_n == 0
    pid_m_is_zero = pid_m == 0
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offsets = k * BLOCK_SIZE_K + offs_k
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        k_mask = k_offsets < K
        if pid_n_is_zero:
            partial_a = tl.sum(a.to(tl.float32), axis=0)
            sum_a_ptrs = sum_a_partial_ptr + pid_m * K + k_offsets
            tl.store(sum_a_ptrs, partial_a, mask=k_mask)
        if pid_m_is_zero:
            partial_b = tl.sum(b.to(tl.float32), axis=1)
            sum_b_ptrs = sum_b_partial_ptr + pid_n * K + k_offsets
            tl.store(sum_b_ptrs, partial_b, mask=k_mask)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    sum_c_slot = sum_c_partial_ptr + pid_m * num_pid_n + pid_n
    sum_c_tile = tl.sum(tl.sum(accumulator, axis=1), axis=0)
    tl.store(sum_c_slot, sum_c_tile)


def launch_matmul_naive_kernel_only(a, b):
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


def matmul_abft(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    num_pid_m, num_pid_n = get_abft_partial_shape_bounds(M, N)
    sum_a_partial = torch.zeros((num_pid_m, K), device=a.device, dtype=torch.float32)
    sum_b_partial = torch.zeros((num_pid_n, K), device=a.device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
    launch_matmul_abft_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial)
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


# ---------------------------------------------------------------------------
# Stage 1: tail-effect elimination variants. Three explicit ablation paths.
#   v2a (path A in plan): every block computes & stores its own (pid_m,pid_n)
#       slot in a 3D partial buffer. Redundant compute & memory, no atomic.
#   v2b (path B in plan): round-robin assignment of which block computes the
#       colsum / rowsum at each k-iter, and tl.atomic_add into a flat (K,)
#       global accumulator. Pays atomic contention but skips post-reduce.
#   v2c (round-robin store): same round-robin compute as v2b, but stores to a
#       per-pid_m / per-pid_n partial buffer like v1 (no atomic). Workload
#       balanced AND no contention; only the partial-store pattern differs.
# All three pin (BM=128, BN=128, BK=32, GM=8, warps=4, stages=4) so NCU can
# diff them apples-to-apples against the stage0 baseline.
# ---------------------------------------------------------------------------

V2_FIXED = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 4,
}


def _v2_grid_dims(M: int, N: int):
    bm = V2_FIXED["BLOCK_SIZE_M"]
    bn = V2_FIXED["BLOCK_SIZE_N"]
    return triton.cdiv(M, bm), triton.cdiv(N, bn)


# v2c autotune intentionally reuses get_matmul_autotune_config() so v2c shares
# the exact same search space as the baseline matmul_kernel / matmul_abft_kernel
# (apples-to-apples across providers).


@triton.jit
def matmul_abft_kernel_v2a(
    a_ptr,
    b_ptr,
    c_ptr,
    sum_a_partial3d_ptr,
    sum_b_partial3d_ptr,
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offsets = k * BLOCK_SIZE_K + offs_k
        k_mask = k_offsets < K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        partial_a = tl.sum(a.to(tl.float32), axis=0)
        partial_b = tl.sum(b.to(tl.float32), axis=1)
        a_slot = sum_a_partial3d_ptr + pid_m * (num_pid_n * K) + pid_n * K + k_offsets
        b_slot = sum_b_partial3d_ptr + pid_m * (num_pid_n * K) + pid_n * K + k_offsets
        tl.store(a_slot, partial_a, mask=k_mask)
        tl.store(b_slot, partial_b, mask=k_mask)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

    sum_c_slot = sum_c_partial_ptr + pid_m * num_pid_n + pid_n
    sum_c_tile = tl.sum(tl.sum(accumulator, axis=1), axis=0)
    tl.store(sum_c_slot, sum_c_tile)


@triton.jit
def matmul_abft_kernel_v2b(
    a_ptr,
    b_ptr,
    c_ptr,
    sum_a_total_ptr,
    sum_b_total_ptr,
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offsets = k * BLOCK_SIZE_K + offs_k
        k_mask = k_offsets < K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        do_colsum = pid_n == (k % num_pid_n)
        do_rowsum = pid_m == (k % num_pid_m)
        if do_colsum:
            partial_a = tl.sum(a.to(tl.float32), axis=0)
            tl.atomic_add(sum_a_total_ptr + k_offsets, partial_a, mask=k_mask)
        if do_rowsum:
            partial_b = tl.sum(b.to(tl.float32), axis=1)
            tl.atomic_add(sum_b_total_ptr + k_offsets, partial_b, mask=k_mask)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

    sum_c_slot = sum_c_partial_ptr + pid_m * num_pid_n + pid_n
    sum_c_tile = tl.sum(tl.sum(accumulator, axis=1), axis=0)
    tl.store(sum_c_slot, sum_c_tile)


@triton.jit
def matmul_abft_kernel_v2c(
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offsets = k * BLOCK_SIZE_K + offs_k
        k_mask = k_offsets < K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        do_colsum = pid_n == (k % num_pid_n)
        do_rowsum = pid_m == (k % num_pid_m)
        if do_colsum:
            partial_a = tl.sum(a.to(tl.float32), axis=0)
            tl.store(sum_a_partial_ptr + pid_m * K + k_offsets, partial_a, mask=k_mask)
        if do_rowsum:
            partial_b = tl.sum(b.to(tl.float32), axis=1)
            tl.store(sum_b_partial_ptr + pid_n * K + k_offsets, partial_b, mask=k_mask)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

    sum_c_slot = sum_c_partial_ptr + pid_m * num_pid_n + pid_n
    sum_c_tile = tl.sum(tl.sum(accumulator, axis=1), axis=0)
    tl.store(sum_c_slot, sum_c_tile)


@triton.autotune(
    configs=get_matmul_autotune_config(),
    key=["M", "N", "K"],
    reset_to_zero=["sum_a_partial_ptr", "sum_b_partial_ptr", "sum_c_partial_ptr"],
)
@triton.jit
def matmul_abft_kernel_v2c_autotune(
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
    # Same logic as v2c fixed, but let Triton pick tiling/warps/stages.
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
    num_k_iter = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_iter):
        k_offsets = k * BLOCK_SIZE_K + offs_k
        k_mask = k_offsets < K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        # Round-robin assignment across programs to avoid tail effect.
        do_colsum = pid_n == (k % num_pid_n)
        do_rowsum = pid_m == (k % num_pid_m)
        if do_colsum:
            partial_a = tl.sum(a.to(tl.float32), axis=0)
            tl.store(sum_a_partial_ptr + pid_m * K + k_offsets, partial_a, mask=k_mask)
        if do_rowsum:
            partial_b = tl.sum(b.to(tl.float32), axis=1)
            tl.store(sum_b_partial_ptr + pid_n * K + k_offsets, partial_b, mask=k_mask)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

    sum_c_slot = sum_c_partial_ptr + pid_m * num_pid_n + pid_n
    sum_c_tile = tl.sum(tl.sum(accumulator, axis=1), axis=0)
    tl.store(sum_c_slot, sum_c_tile)


@triton.autotune(
    configs=get_matmul_autotune_config(),
    key=["M", "N", "K"],
    reset_to_zero=["sum_a_partial_ptr", "sum_b_partial_ptr", "sum_c_partial_ptr"],
)
@triton.jit
def matmul_abft_kernel_v3a_dotsum(
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
    """Stage 2 v3a: identical v2c round-robin scheme, but per-iteration
    column / row sums use tl.dot(ones, a) / tl.dot(b, ones) so the reduction
    is performed by HMMA in tensor-core registers. This avoids tl.sum's
    SMEM round-trip + BAR.SYNC pairs, and folds the FP16 -> FP32 accumulation
    cast into the HMMA op for free.

    Cost trade-off: 16x more multiply-adds per partial-sum tile (since the
    HMMA tile minimum is 16xKxN), but those FMAs run on tensor cores at much
    higher throughput than tl.sum's CUDA-core path, and the extra issue cost
    is small because each block only triggers the partial-sum on
    ~K/(BK*num_pid_n) iterations (round-robin fan-out)."""
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

    rows16 = tl.arange(0, 16)

    num_k_iter = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_iter):
        k_offsets = k * BLOCK_SIZE_K + offs_k
        k_mask = k_offsets < K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        do_colsum = pid_n == (k % num_pid_n)
        do_rowsum = pid_m == (k % num_pid_m)
        # Building the "row-0 / col-0 only" ones tiles INSIDE the conditional
        # keeps their live range short (≤ 1 K-iteration) so they don't pin
        # registers across the whole loop and crush occupancy. tl.store with
        # the broadcast pointer + row/col mask collapses the (16, BK) write
        # back to a single coalesced row of HBM stores.
        if do_colsum:
            ones_M_first_row = tl.where(
                rows16[:, None] == 0,
                tl.full((16, BLOCK_SIZE_M), 1.0, dtype=tl.float16),
                tl.zeros((16, BLOCK_SIZE_M), dtype=tl.float16),
            )
            # (16, BM) @ (BM, BK) -> (16, BK) FP32, row 0 = sum_M(a) (others zeroed).
            partial_a_2d = tl.dot(ones_M_first_row, a)
            sum_a_ptrs_2d = (
                sum_a_partial_ptr
                + pid_m * K
                + (rows16[:, None] * 0)
                + k_offsets[None, :]
            )
            mask_a_2d = (rows16[:, None] == 0) & (k_mask[None, :])
            tl.store(sum_a_ptrs_2d, partial_a_2d, mask=mask_a_2d)
        if do_rowsum:
            ones_N_first_col = tl.where(
                rows16[None, :] == 0,
                tl.full((BLOCK_SIZE_N, 16), 1.0, dtype=tl.float16),
                tl.zeros((BLOCK_SIZE_N, 16), dtype=tl.float16),
            )
            # (BK, BN) @ (BN, 16) -> (BK, 16) FP32, col 0 = sum_N(b) (others zeroed).
            partial_b_2d = tl.dot(b, ones_N_first_col)
            sum_b_ptrs_2d = (
                sum_b_partial_ptr
                + pid_n * K
                + k_offsets[:, None]
                + (rows16[None, :] * 0)
            )
            mask_b_2d = (rows16[None, :] == 0) & (k_mask[:, None])
            tl.store(sum_b_ptrs_2d, partial_b_2d, mask=mask_b_2d)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

    sum_c_slot = sum_c_partial_ptr + pid_m * num_pid_n + pid_n
    sum_c_tile = tl.sum(tl.sum(accumulator, axis=1), axis=0)
    tl.store(sum_c_slot, sum_c_tile)


def launch_matmul_abft_v2a_kernel_only(a, b, c, sum_a3d, sum_b3d, sum_c_partial):
    M, K = a.shape
    _, N = b.shape
    num_pid_m, num_pid_n = _v2_grid_dims(M, N)
    grid = (num_pid_m * num_pid_n,)
    matmul_abft_kernel_v2a[grid](
        a, b, c, sum_a3d, sum_b3d, sum_c_partial,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=V2_FIXED["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=V2_FIXED["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=V2_FIXED["BLOCK_SIZE_K"],
        GROUP_SIZE_M=V2_FIXED["GROUP_SIZE_M"],
        num_warps=V2_FIXED["num_warps"],
        num_stages=V2_FIXED["num_stages"],
    )


def launch_matmul_abft_v2b_kernel_only(a, b, c, sum_a_total, sum_b_total, sum_c_partial):
    M, K = a.shape
    _, N = b.shape
    num_pid_m, num_pid_n = _v2_grid_dims(M, N)
    grid = (num_pid_m * num_pid_n,)
    matmul_abft_kernel_v2b[grid](
        a, b, c, sum_a_total, sum_b_total, sum_c_partial,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=V2_FIXED["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=V2_FIXED["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=V2_FIXED["BLOCK_SIZE_K"],
        GROUP_SIZE_M=V2_FIXED["GROUP_SIZE_M"],
        num_warps=V2_FIXED["num_warps"],
        num_stages=V2_FIXED["num_stages"],
    )


def launch_matmul_abft_v2c_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial):
    M, K = a.shape
    _, N = b.shape
    num_pid_m, num_pid_n = _v2_grid_dims(M, N)
    grid = (num_pid_m * num_pid_n,)
    matmul_abft_kernel_v2c[grid](
        a, b, c, sum_a_partial, sum_b_partial, sum_c_partial,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=V2_FIXED["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=V2_FIXED["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=V2_FIXED["BLOCK_SIZE_K"],
        GROUP_SIZE_M=V2_FIXED["GROUP_SIZE_M"],
        num_warps=V2_FIXED["num_warps"],
        num_stages=V2_FIXED["num_stages"],
    )


def launch_matmul_abft_v2c_autotune_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial):
    M, K = a.shape
    _, N = b.shape
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_abft_kernel_v2c_autotune[grid](
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


def launch_matmul_abft_v3a_dotsum_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial):
    M, K = a.shape
    _, N = b.shape
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_abft_kernel_v3a_dotsum[grid](
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


@triton.autotune(
    configs=get_matmul_autotune_config(),
    key=["M", "N", "K"],
    reset_to_zero=["sum_a_partial_ptr", "sum_b_partial_ptr", "sum_c_partial_ptr"],
)
@triton.jit
def matmul_abft_kernel_v3b_latecast(
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
    """Stage 2 v3b: same v2c round-robin scheme, but the FP16->FP32 cast is
    deferred to AFTER the in-block reduction. This shrinks the casted tile
    from BMxBK / BKxBN (huge) down to BK / BK (tiny vector), eliminating the
    register-pressure hit that the eager cast incurred without changing the
    BAR.SYNC count of tl.sum. Trades BAR.SYNC removal for register/cast cost
    reduction."""
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
    num_k_iter = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_iter):
        k_offsets = k * BLOCK_SIZE_K + offs_k
        k_mask = k_offsets < K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        do_colsum = pid_n == (k % num_pid_n)
        do_rowsum = pid_m == (k % num_pid_m)
        if do_colsum:
            # Reduce in FP16 (BM <= 256, error budget OK), cast the (BK,) result.
            partial_a = tl.sum(a, axis=0).to(tl.float32)
            tl.store(sum_a_partial_ptr + pid_m * K + k_offsets, partial_a, mask=k_mask)
        if do_rowsum:
            partial_b = tl.sum(b, axis=1).to(tl.float32)
            tl.store(sum_b_partial_ptr + pid_n * K + k_offsets, partial_b, mask=k_mask)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

    sum_c_slot = sum_c_partial_ptr + pid_m * num_pid_n + pid_n
    sum_c_tile = tl.sum(tl.sum(accumulator, axis=1), axis=0)
    tl.store(sum_c_slot, sum_c_tile)


def launch_matmul_abft_v3b_latecast_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial):
    M, K = a.shape
    _, N = b.shape
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_abft_kernel_v3b_latecast[grid](
        a, b, c, sum_a_partial, sum_b_partial, sum_c_partial,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )


# We also pin v1 to the same fixed config for an apples-to-apples comparison
# against v2a/b/c. The autotune-driven version (matmul_abft) is kept untouched.
def launch_matmul_abft_v1_fixed_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial):
    M, K = a.shape
    _, N = b.shape
    num_pid_m, num_pid_n = _v2_grid_dims(M, N)
    grid = (num_pid_m * num_pid_n,)
    matmul_abft_kernel.fn[grid](
        a, b, c, sum_a_partial, sum_b_partial, sum_c_partial,
        M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=V2_FIXED["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=V2_FIXED["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=V2_FIXED["BLOCK_SIZE_K"],
        GROUP_SIZE_M=V2_FIXED["GROUP_SIZE_M"],
        num_warps=V2_FIXED["num_warps"],
        num_stages=V2_FIXED["num_stages"],
    )


def matmul_abft_v2a(a, b):
    """Path A: redundant compute & redundant slot. Same identity as v1 but the
    partial buffer is 3D and every block writes its own slot. Post-reduce
    extracts a single (pid_m, K) slice (all (pid_m, *, K) slices are equal by
    construction) and reduces over pid_m on host."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    num_pid_m, num_pid_n = _v2_grid_dims(M, N)
    sum_a3d = torch.zeros((num_pid_m, num_pid_n, K), device=a.device, dtype=torch.float32)
    sum_b3d = torch.zeros((num_pid_m, num_pid_n, K), device=a.device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
    launch_matmul_abft_v2a_kernel_only(a, b, c, sum_a3d, sum_b3d, sum_c_partial)
    sum_a = sum_a3d[:, 0, :].sum(dim=0)
    sum_b = sum_b3d[0, :, :].sum(dim=0)
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


def matmul_abft_v2b(a, b):
    """Path B: round-robin compute + atomic-add into flat (K,) accumulators.
    No per-pid_m partial buffer for sum_a/sum_b, only sum_c_partial remains."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    num_pid_m, num_pid_n = _v2_grid_dims(M, N)
    sum_a = torch.zeros((K,), device=a.device, dtype=torch.float32)
    sum_b = torch.zeros((K,), device=a.device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
    launch_matmul_abft_v2b_kernel_only(a, b, c, sum_a, sum_b, sum_c_partial)
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


def matmul_abft_v2c(a, b):
    """Round-robin compute + per-pid_m partial buffer (no atomic). Buffer
    layout matches v1; only the assignment of who writes is round-robin."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    num_pid_m, num_pid_n = _v2_grid_dims(M, N)
    sum_a_partial = torch.zeros((num_pid_m, K), device=a.device, dtype=torch.float32)
    sum_b_partial = torch.zeros((num_pid_n, K), device=a.device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
    launch_matmul_abft_v2c_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial)
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


def matmul_abft_v2c_autotune(a, b):
    """v2c logic but with autotuned tiling/warps/stages (fix small-shape regression)."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # Use conservative bounds from the baseline config space for partial buffers.
    num_pid_m, num_pid_n = get_abft_partial_shape_bounds(M, N)
    sum_a_partial = torch.zeros((num_pid_m, K), device=a.device, dtype=torch.float32)
    sum_b_partial = torch.zeros((num_pid_n, K), device=a.device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
    launch_matmul_abft_v2c_autotune_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial)
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


def matmul_abft_v3b_latecast(a, b):
    """Stage 2 v3b: same v2c logic but with delayed FP16->FP32 cast.
    Lighter alternative to v3a (no extra HMMA tile); does not eliminate
    BAR.SYNC but slashes the register pressure of the cast."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    num_pid_m, num_pid_n = get_abft_partial_shape_bounds(M, N)
    sum_a_partial = torch.zeros((num_pid_m, K), device=a.device, dtype=torch.float32)
    sum_b_partial = torch.zeros((num_pid_n, K), device=a.device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
    launch_matmul_abft_v3b_latecast_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial)
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


def matmul_abft_v3a_dotsum(a, b):
    """Stage 2 v3a: v2c-style round-robin ABFT with HMMA-based partial sums.
    Replaces tl.sum(a.to(fp32), axis=...) by tl.dot(ones, a) so the reduction
    runs entirely on tensor cores and avoids tl.sum's BAR.SYNC + SMEM round-trip."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    num_pid_m, num_pid_n = get_abft_partial_shape_bounds(M, N)
    sum_a_partial = torch.zeros((num_pid_m, K), device=a.device, dtype=torch.float32)
    sum_b_partial = torch.zeros((num_pid_n, K), device=a.device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
    launch_matmul_abft_v3a_dotsum_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial)
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

def matmul_abft_v1_fixed(a, b):
    """v1 (pid==0 gating) but pinned to the same fixed config as v2 variants.
    Used to isolate "fixed-config v1" vs "fixed-config v2*" without autotune
    variance contaminating the comparison."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    num_pid_m, num_pid_n = _v2_grid_dims(M, N)
    sum_a_partial = torch.zeros((num_pid_m, K), device=a.device, dtype=torch.float32)
    sum_b_partial = torch.zeros((num_pid_n, K), device=a.device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=a.device, dtype=torch.float32)
    launch_matmul_abft_v1_fixed_kernel_only(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial)
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
