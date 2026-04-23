import argparse
import time
from typing import Dict, List, Tuple

import torch

from triton_gemm_abft_fused import matmul_abft_fused
from triton_gemm_baseline import matmul_baseline


def _dtype_from_str(name: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    return mapping[name]


def _parse_shapes(raw_shapes: List[str]) -> List[Tuple[int, int, int]]:
    if not raw_shapes:
        return [(4096, 4096, 4096), (8192, 4096, 4096)]
    shapes = []
    for raw in raw_shapes:
        m_str, n_str, k_str = raw.split(",")
        shapes.append((int(m_str), int(n_str), int(k_str)))
    return shapes


def _mean_runtime_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start
    return (elapsed_s / repeat) * 1000.0


def _tflops(m: int, n: int, k: int, ms: float) -> float:
    flops = 2.0 * m * n * k
    return flops / (ms * 1e-3) / 1e12


def benchmark_one_shape(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
    seed: int,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)

    torch_ms = _mean_runtime_ms(lambda: torch.matmul(a, b), warmup=warmup, repeat=repeat)
    baseline_ms = _mean_runtime_ms(lambda: matmul_baseline(a, b), warmup=warmup, repeat=repeat)
    fused_ms = _mean_runtime_ms(lambda: matmul_abft_fused(a, b)["c"], warmup=warmup, repeat=repeat)

    out = matmul_abft_fused(a, b)
    abft_abs = out["abft_abs_error"].item()
    abft_rel = out["abft_rel_error"].item()

    torch_tflops = _tflops(m, n, k, torch_ms)
    baseline_tflops = _tflops(m, n, k, baseline_ms)
    fused_tflops = _tflops(m, n, k, fused_ms)
    overhead_pct = ((fused_ms - baseline_ms) / baseline_ms) * 100.0
    baseline_vs_torch_pct = ((baseline_ms - torch_ms) / torch_ms) * 100.0
    fused_vs_torch_pct = ((fused_ms - torch_ms) / torch_ms) * 100.0

    return {
        "m": m,
        "n": n,
        "k": k,
        "torch_ms": torch_ms,
        "baseline_ms": baseline_ms,
        "fused_ms": fused_ms,
        "torch_tflops": torch_tflops,
        "baseline_tflops": baseline_tflops,
        "fused_tflops": fused_tflops,
        "overhead_pct": overhead_pct,
        "baseline_vs_torch_pct": baseline_vs_torch_pct,
        "fused_vs_torch_pct": fused_vs_torch_pct,
        "abft_abs_error": abft_abs,
        "abft_rel_error": abft_rel,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Triton baseline vs fused ABFT GEMM.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shape", action="append", help="shape as M,N,K. Can be passed multiple times.")
    return parser.parse_args()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required.")
    args = _parse_args()
    dtype = _dtype_from_str(args.dtype)
    shapes = _parse_shapes(args.shape)

    print(
        "shape,torch_ms,baseline_ms,fused_ms,torch_tflops,baseline_tflops,fused_tflops,"
        "baseline_vs_torch_pct,fused_vs_torch_pct,overhead_pct,abft_abs_error,abft_rel_error"
    )
    for m, n, k in shapes:
        stats = benchmark_one_shape(
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
        )
        shape = f"{stats['m']}x{stats['n']}x{stats['k']}"
        print(
            f"{shape},{stats['torch_ms']:.4f},{stats['baseline_ms']:.4f},{stats['fused_ms']:.4f},"
            f"{stats['torch_tflops']:.4f},{stats['baseline_tflops']:.4f},{stats['fused_tflops']:.4f},"
            f"{stats['baseline_vs_torch_pct']:.2f},{stats['fused_vs_torch_pct']:.2f},"
            f"{stats['overhead_pct']:.2f},{stats['abft_abs_error']:.6e},{stats['abft_rel_error']:.6e}"
        )
