import argparse
import csv
from dataclasses import dataclass
from typing import List, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def abft_probe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sum_a_partial_ptr,
    sum_b_partial_ptr,
    sum_c_partial_ptr,
    sink_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    DO_SUM: tl.constexpr,
    DO_STORE: tl.constexpr,
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
    sink_acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offsets = k * BLOCK_SIZE_K + offs_k
        k_mask = k_offsets < K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        if DO_SUM:
            if pid_n_is_zero:
                partial_a = tl.sum(a.to(tl.float32), axis=0)
                if DO_STORE:
                    sum_a_ptrs = sum_a_partial_ptr + pid_m * K + k_offsets
                    tl.store(sum_a_ptrs, partial_a, mask=k_mask)
                sink_acc += tl.sum(partial_a, axis=0)
            if pid_m_is_zero:
                partial_b = tl.sum(b.to(tl.float32), axis=1)
                if DO_STORE:
                    sum_b_ptrs = sum_b_partial_ptr + pid_n * K + k_offsets
                    tl.store(sum_b_ptrs, partial_b, mask=k_mask)
                sink_acc += tl.sum(partial_b, axis=0)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

    num_pid_n_sc = tl.cdiv(N, BLOCK_SIZE_N)
    sum_c_slot = sum_c_partial_ptr + pid_m * num_pid_n_sc + pid_n
    sum_c_tile = tl.sum(tl.sum(accumulator, axis=1), axis=0)
    tl.store(sum_c_slot, sum_c_tile)
    tl.store(sink_ptr + pid, sink_acc)


@dataclass
class Cfg:
    name: str
    bm: int
    bn: int
    bk: int
    warps: int
    stages: int


DEFAULT_CFGS = [
    Cfg("k64_s4_w4", 128, 128, 64, 4, 4),
    Cfg("k32_s4_w4", 128, 128, 32, 4, 4),
    Cfg("k32_s2_w4", 128, 128, 32, 4, 2),
    Cfg("k16_s2_w4", 128, 128, 16, 4, 2),
]


def parse_args():
    p = argparse.ArgumentParser(description="ABFT register-pressure sweep launcher")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--do-sum", type=int, default=1, choices=[0, 1])
    p.add_argument("--do-store", type=int, default=1, choices=[0, 1])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config-name", type=str, required=True)
    p.add_argument("--csv-out", type=str, default="")
    return p.parse_args()


def get_cfg(name: str) -> Cfg:
    for c in DEFAULT_CFGS:
        if c.name == name:
            return c
    raise ValueError(f"unknown config-name: {name}")


def launch_once(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial, sink, cfg: Cfg, do_sum: int, do_store: int):
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, cfg.bm) * triton.cdiv(n, cfg.bn),)
    abft_probe_kernel[grid](
        a,
        b,
        c,
        sum_a_partial,
        sum_b_partial,
        sum_c_partial,
        sink,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        DO_SUM=do_sum,
        DO_STORE=do_store,
        BLOCK_SIZE_M=cfg.bm,
        BLOCK_SIZE_N=cfg.bn,
        BLOCK_SIZE_K=cfg.bk,
        GROUP_SIZE_M=8,
        num_warps=cfg.warps,
        num_stages=cfg.stages,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    m = n = k = args.dim
    cfg = get_cfg(args.config_name)
    device = "cuda"
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float16)
    c = torch.empty((m, n), device=device, dtype=torch.float16)
    num_pid_m = triton.cdiv(m, cfg.bm)
    num_pid_n = triton.cdiv(n, cfg.bn)
    sum_a_partial = torch.zeros((num_pid_m, k), device=device, dtype=torch.float32)
    sum_b_partial = torch.zeros((num_pid_n, k), device=device, dtype=torch.float32)
    sum_c_partial = torch.zeros((num_pid_m, num_pid_n), device=device, dtype=torch.float32)
    sink = torch.zeros((num_pid_m * num_pid_n,), device=device, dtype=torch.float32)

    for _ in range(args.warmup):
        launch_once(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial, sink, cfg, args.do_sum, args.do_store)
    torch.cuda.synchronize()
    for _ in range(args.iters):
        launch_once(a, b, c, sum_a_partial, sum_b_partial, sum_c_partial, sink, cfg, args.do_sum, args.do_store)
    torch.cuda.synchronize()

    print(
        f"probe_done,dim={args.dim},do_sum={args.do_sum},do_store={args.do_store},cfg={cfg.name},"
        f"bm={cfg.bm},bn={cfg.bn},bk={cfg.bk},warps={cfg.warps},stages={cfg.stages}"
    )

    if args.csv_out:
        with open(args.csv_out, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([args.dim, args.do_sum, args.do_store, cfg.name, cfg.bm, cfg.bn, cfg.bk, cfg.warps, cfg.stages])


if __name__ == "__main__":
    main()
