#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run baseline/ABFT variants and compare latency.")
    ap.add_argument("--python", default="python3")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompt-lens", default="128,512,1024")
    ap.add_argument("--gen-lens", default="32,128")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup-prefill", type=int, default=100)
    ap.add_argument("--warmup-decode", type=int, default=100)
    ap.add_argument("--iters-prefill", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--run-fault", action="store_true")
    ap.add_argument("--fault-probability", type=float, default=1e-7)
    ap.add_argument("--fault-magnitude", type=float, default=1.0)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--report-md", default="ABFT_QWEN_PYTORCH_REPORT.md")
    return ap.parse_args()


def run_one(cmd: List[str], cwd: str) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_p99(data: Dict[str, Any], key: str) -> float:
    return float(data["global_latency"][key]["p99_ms"])


def overhead_pct(base: float, new: float) -> float:
    if base <= 1e-9:
        return 0.0
    return (new - base) / base * 100.0


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--model-id",
        args.model_id,
        "--prompt-lens",
        args.prompt_lens,
        "--gen-lens",
        args.gen_lens,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--warmup-prefill",
        str(args.warmup_prefill),
        "--warmup-decode",
        str(args.warmup_decode),
        "--iters-prefill",
        str(args.iters_prefill),
        "--seed",
        str(args.seed),
    ]

    bench = Path("qwen_abft_benchmark.py")
    baseline_json = out_dir / "baseline.json"
    all_ops_json = out_dir / "abft_all_ops.json"
    sampled_json = out_dir / "abft_sampled.json"

    run_one(
        [args.python, str(bench), *common, "--out-json", str(baseline_json)],
        cwd=args.workdir,
    )
    run_one(
        [
            args.python,
            str(bench),
            *common,
            "--abft-enable",
            "--abft-sample-rate",
            "1.0",
            "--out-json",
            str(all_ops_json),
        ],
        cwd=args.workdir,
    )
    run_one(
        [
            args.python,
            str(bench),
            *common,
            "--abft-enable",
            "--abft-sample-rate",
            "0.25",
            "--out-json",
            str(sampled_json),
        ],
        cwd=args.workdir,
    )

    fault_json = None
    if args.run_fault:
        fault_json = out_dir / "abft_fault.json"
        run_one(
            [
                args.python,
                str(bench),
                *common,
                "--abft-enable",
                "--inject-fault",
                "--inject-probability",
                str(args.fault_probability),
                "--inject-magnitude",
                str(args.fault_magnitude),
                "--out-json",
                str(fault_json),
            ],
            cwd=args.workdir,
        )

    base = load_json(baseline_json)
    all_ops = load_json(all_ops_json)
    sampled = load_json(sampled_json)
    fault = load_json(fault_json) if fault_json else None

    base_prefill_p99 = get_p99(base, "prefill")
    base_decode_p99 = get_p99(base, "decode_step")
    all_prefill_p99 = get_p99(all_ops, "prefill")
    all_decode_p99 = get_p99(all_ops, "decode_step")
    sampled_prefill_p99 = get_p99(sampled, "prefill")
    sampled_decode_p99 = get_p99(sampled, "decode_step")

    lines = [
        "# ABFT_QWEN_PYTORCH_REPORT",
        "",
        "## Global P99 延迟对比",
        "",
        "| 方案 | prefill_p99_ms | decode_token_p99_ms | prefill_overhead_pct | decode_overhead_pct |",
        "|---|---:|---:|---:|---:|",
        (
            f"| baseline | {base_prefill_p99:.4f} | {base_decode_p99:.4f} | 0.00 | 0.00 |"
        ),
        (
            f"| abft_all_ops | {all_prefill_p99:.4f} | {all_decode_p99:.4f} | "
            f"{overhead_pct(base_prefill_p99, all_prefill_p99):.2f} | "
            f"{overhead_pct(base_decode_p99, all_decode_p99):.2f} |"
        ),
        (
            f"| abft_sampled_0.25 | {sampled_prefill_p99:.4f} | {sampled_decode_p99:.4f} | "
            f"{overhead_pct(base_prefill_p99, sampled_prefill_p99):.2f} | "
            f"{overhead_pct(base_decode_p99, sampled_decode_p99):.2f} |"
        ),
        "",
        "## ABFT统计",
        "",
        f"- abft_all_ops checks_total: {all_ops['stats']['abft']['checks_total']}",
        f"- abft_all_ops fail_total: {all_ops['stats']['abft']['fail_total']}",
        f"- abft_sampled checks_total: {sampled['stats']['abft']['checks_total']}",
        f"- abft_sampled fail_total: {sampled['stats']['abft']['fail_total']}",
    ]

    if fault is not None:
        fi = fault["stats"]["fault_injection"]
        lines.extend(
            [
                "",
                "## 故障注入统计",
                "",
                f"- injections_total: {fi['injections_total']}",
                f"- detected_injections_total: {fi['detected_injections_total']}",
                f"- detection_rate: {fi['detection_rate']:.6f}",
            ]
        )

    report_path = Path(args.report_md)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
