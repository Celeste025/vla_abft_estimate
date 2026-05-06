"""纯 prefill 耗时：PyTorch 原生 Linear vs Triton GEMM vs Triton + kernel 内 outlier。

单次 `model(input_ids, attention_mask)`，seq_len 较大时 Linear 侧以 GEMM 为主（非 decode GEMV）。
每个 backend 单独起一份模型，避免就地换层后无法恢复 baseline。
"""
from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch

from model_runner import ModelRunner
from protect_linear import apply_protect_linears_qwen


def _build_prompt_ids(tokenizer, device: torch.device, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = tokenizer("测 prefill 延迟 " * 512, add_special_tokens=False)["input_ids"]
    if not chunk:
        chunk = [tokenizer.eos_token_id or 0]
    ids: list[int] = []
    while len(ids) < seq_len:
        ids.extend(chunk)
    ids = ids[:seq_len]
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


@torch.inference_mode()
def _bench_one_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        model(input_ids=input_ids, attention_mask=attention_mask)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms: list[float] = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        model(input_ids=input_ids, attention_mask=attention_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms.append((time.perf_counter_ns() - t0) / 1e6)
    return ms


def _summarize_ms(values: list[float]) -> dict[str, float]:
    s = sorted(values)
    n = len(s)
    p50 = s[n // 2] if n else float("nan")
    p95 = s[int(0.95 * (n - 1))] if n > 1 else s[0] if n else float("nan")
    return {
        "mean_ms": float(statistics.mean(values)),
        "stdev_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "p50_ms": float(p50),
        "p95_ms": float(p95),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument(
        "--fault-delta",
        type=float,
        default=10000.0,
        help="threshold = |fault_delta| * clear_threshold_mul（与 run_gsm8k_latency 一致）",
    )
    ap.add_argument("--clear-threshold-mul", type=float, default=0.5)
    args = ap.parse_args()

    threshold = abs(args.fault_delta) * args.clear_threshold_mul
    backends: list[tuple[str, str | None]] = [
        ("pytorch_baseline", None),
        ("triton_plain", "triton_plain"),
        ("protect_triton", "triton"),
    ]

    print(
        f"[bench_prefill] model={args.model_id} seq_len={args.seq_len} dtype={args.dtype} "
        f"warmup={args.warmup} repeats={args.repeats} threshold={threshold}",
        flush=True,
    )

    baseline_mean: float | None = None
    for name, linear_backend in backends:
        runner = ModelRunner(
            args.model_id,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        n_wrap = 0
        if linear_backend is not None:
            n_wrap = apply_protect_linears_qwen(
                runner.model,
                float(threshold),
                linear_backend=linear_backend,  # type: ignore[arg-type]
            )
        dev = torch.device(args.device)
        input_ids, attention_mask = _build_prompt_ids(runner.tokenizer, dev, int(args.seq_len))

        times = _bench_one_forward(
            runner.model,
            input_ids,
            attention_mask,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        stat = _summarize_ms(times)
        if name == "pytorch_baseline":
            baseline_mean = stat["mean_ms"]

        ratio = stat["mean_ms"] / baseline_mean if baseline_mean and baseline_mean > 0 else float("nan")
        print(
            f"  {name}: wrap={n_wrap} mean={stat['mean_ms']:.2f}ms stdev={stat['stdev_ms']:.2f} "
            f"p50={stat['p50_ms']:.2f} p95={stat['p95_ms']:.2f}  vs_baseline_mean={ratio:.3f}x",
            flush=True,
        )

        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[bench_prefill] done.", flush=True)


if __name__ == "__main__":
    main()
