"""CUDA Event 微基准：eager / torch.compile(max-autotune) / Triton plain / Triton fused。"""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F

from triton_protect_linear import fused_linear_outlier


def fused_eager(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    threshold: float,
) -> torch.Tensor:
    y = F.linear(x, weight, bias)
    return torch.where(y.abs() > threshold, 0.0, y)


@torch.compile(mode="max-autotune")
def fused_compiled(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    threshold: float,
) -> torch.Tensor:
    y = F.linear(x, weight, bias)
    return torch.where(y.abs() > threshold, 0.0, y)


def bench_cuda_events(
    fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    threshold: float,
    *,
    warmup: int,
    repeat: int,
) -> List[float]:
    for _ in range(warmup):
        y = fn(x, weight, bias, threshold)
    torch.cuda.synchronize()
    times_ms: List[float] = []
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        starter.record()
        y = fn(x, weight, bias, threshold)
        ender.record()
        torch.cuda.synchronize()
        times_ms.append(starter.elapsed_time(ender))
    _ = y  # noqa: F841
    return times_ms


@dataclass
class ShapeCase:
    name: str
    batch: int
    seq: int
    in_features: int
    out_features: int


def _make_tensors(
    device: torch.device,
    dtype: torch.dtype,
    case: ShapeCase,
    threshold: float,
    seed: int,
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn(
        case.batch,
        case.seq,
        case.in_features,
        device=device,
        dtype=dtype,
        generator=g,
    )
    weight = torch.randn(
        case.out_features,
        case.in_features,
        device=device,
        dtype=dtype,
        generator=g,
    )
    bias = torch.randn(case.out_features, device=device, dtype=dtype, generator=g)
    return x, weight, bias, float(threshold)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--threshold", type=float, default=5000.0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    cases: List[ShapeCase] = [
        ShapeCase("decode_bs1_seq1", 1, 1, 3584, 3584),
        ShapeCase("prefill_bs1_seq256", 1, 256, 3584, 3584),
        ShapeCase("prefill_bs1_seq512", 1, 512, 3584, 18944),
    ]

    def summ(xs: List[float]) -> Tuple[float, float]:
        return statistics.fmean(xs), statistics.pstdev(xs)

    print(
        f"device={device} dtype={dtype} threshold={args.threshold} "
        f"warmup={args.warmup} repeat={args.repeat}",
        flush=True,
    )

    for case in cases:
        x, w, b, thr = _make_tensors(device, dtype, case, args.threshold, args.seed)

        def triton_plain_fn(xx, ww, bb, t):
            return fused_linear_outlier(xx, ww, bb, t, do_outlier=False)

        def triton_fused_fn(xx, ww, bb, t):
            return fused_linear_outlier(xx, ww, bb, t, do_outlier=True)

        rows = [
            ("eager", fused_eager),
            ("compiled_max_autotune", fused_compiled),
            ("triton_plain", triton_plain_fn),
            ("triton_fused", triton_fused_fn),
        ]
        print(
            f"\n[{case.name}] x=({case.batch},{case.seq},{case.in_features}) "
            f"weight=({case.out_features},{case.in_features})",
            flush=True,
        )
        baseline_mean = None
        for label, fn in rows:
            tms = bench_cuda_events(fn, x, w, b, thr, warmup=args.warmup, repeat=args.repeat)
            m, s = summ(tms)
            if baseline_mean is None:
                baseline_mean = m
                ratio = 1.0
            else:
                ratio = m / baseline_mean if baseline_mean > 0 else float("nan")
            print(f"  {label:24s} mean={m:.4f} ms std={s:.4f} ms  (×baseline {ratio:.3f})", flush=True)


if __name__ == "__main__":
    main()
