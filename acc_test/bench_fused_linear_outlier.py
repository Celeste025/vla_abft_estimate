"""对比 fused Linear + |y|>threshold 清零：eager vs @torch.compile(max-autotune)。

仅统计 GPU kernel 时间（CUDA Event），形状贴近常见 LLM 投影层。

清零侧使用 ``torch.where(..., 0.0, y)``（标量分支），便于 Inductor 避免显式
``zeros_like`` 的大块分配；eager 下可能触发与 ``y`` 的类型提升，见 PyTorch 文档。

查看 Inductor 生成代码::

  TORCH_LOGS=output_code python bench_fused_linear_outlier.py 2>&1 | tee inductor_code.log
"""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


def fused_linear_outlier_eager(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    threshold: float,
) -> torch.Tensor:
    y = F.linear(x, weight, bias)
    return torch.where(y.abs() > threshold, 0.0, y)


@torch.compile(mode="max-autotune")
def fused_linear_outlier_pt2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    threshold: float,
) -> torch.Tensor:
    y = F.linear(x, weight, bias)
    return torch.where(y.abs() > threshold, 0.0, y)


@dataclass
class ShapeCase:
    name: str
    batch: int
    seq: int
    in_features: int
    out_features: int


def _make_inputs(
    device: torch.device,
    dtype: torch.dtype,
    case: ShapeCase,
    threshold: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
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


def bench_cuda_events(
    fn,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    threshold: float,
    *,
    warmup: int,
    repeat: int,
) -> List[float]:
    """返回每次 forward 的 GPU 耗时（毫秒）。"""
    for _ in range(warmup):
        y = fn(x, weight, bias, threshold)
    torch.cuda.synchronize()

    times_ms: List[float] = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        starter.record()
        y = fn(x, weight, bias, threshold)
        ender.record()
        torch.cuda.synchronize()
        times_ms.append(starter.elapsed_time(ender))
    _ = y  # noqa: F841
    return times_ms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
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

    # 贴近 Qwen2.5-7B 量级：hidden≈3584；decode seq=1；prefill 取 256/512
    cases: List[ShapeCase] = [
        ShapeCase("decode_bs1_seq1", 1, 1, 3584, 3584),
        ShapeCase("prefill_bs1_seq256", 1, 256, 3584, 3584),
        ShapeCase("prefill_bs1_seq512", 1, 512, 3584, 18944),  # 更大 out（类似 up/gate 展宽）
    ]

    print(
        f"device={device} dtype={dtype} threshold={args.threshold} "
        f"warmup={args.warmup} repeat={args.repeat}",
        flush=True,
    )

    for case in cases:
        x, w, b, thr = _make_inputs(device, dtype, case, args.threshold, args.seed)

        t_eager = bench_cuda_events(
            fused_linear_outlier_eager,
            x,
            w,
            b,
            thr,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        t_comp = bench_cuda_events(
            fused_linear_outlier_pt2,
            x,
            w,
            b,
            thr,
            warmup=args.warmup,
            repeat=args.repeat,
        )

        def _summ(xs: List[float]) -> Tuple[float, float]:
            return statistics.fmean(xs), statistics.pstdev(xs)

        m_e, s_e = _summ(t_eager)
        m_c, s_c = _summ(t_comp)
        ratio = m_c / m_e if m_e > 0 else float("nan")

        print(
            f"\n[{case.name}] x=({case.batch},{case.seq},{case.in_features}) "
            f"weight=({case.out_features},{case.in_features})",
            flush=True,
        )
        print(
            f"  eager:    mean={m_e:.4f} ms  std={s_e:.4f} ms",
            flush=True,
        )
        print(
            f"  compiled: mean={m_c:.4f} ms  std={s_c:.4f} ms  (vs eager ×{ratio:.3f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
