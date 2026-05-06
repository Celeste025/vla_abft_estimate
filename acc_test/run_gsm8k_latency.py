"""Run Qwen2.5 + GSM8K latency benchmark.

Modes:
  --mode clean           : no hooks, no protect wrappers.
  --mode protect         : InjectionContext(protect_only=True, site_strategy=qwen_decoder)
                           attaches threshold-mask + masked_fill on every hook site
                           (28 layers × 8 hooks each = 224 forward/pre registrations).
  --mode clean_compile   : torch.compile only (no ProtectLinear, no hooks). Same e2e timing
                           as protect_compile for apples-to-apples comparison.
  --mode protect_compile : replace decoder Linears with ProtectLinear, then torch.compile.
                           Latency uses cuda-sync wall time around whole generate (see
                           gsm8k_latency_task e2e_generate), not per-forward LatencyHook.
  --mode protect_triton : Triton GEMM+bias + kernel 内 outlier 清零（无 torch.compile）。
  --mode triton_plain   : 同上 Triton 路径但 **不做** outlier（仅 matmul+bias），用于与 protect_triton 对照。

By default (clean / protect) each problem uses `LatencyHook` on the root model so the
first forward is prefill and the rest are decode steps (cuda.synchronize per forward).

Modes clean_compile / protect_compile / protect_triton / triton_plain use e2e_generate timing.
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext

from gsm8k_latency_task import Gsm8kLatencyTask
from inject import SITE_STRATEGY_QWEN, InjectionContext
from model_runner import ModelRunner
from protect_linear import apply_protect_linears_qwen, compile_model_for_latency


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=[
            "clean",
            "protect",
            "clean_compile",
            "protect_compile",
            "protect_triton",
            "triton_plain",
        ],
        required=True,
    )
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--warmup-samples",
        type=int,
        default=1,
        help="Number of leading problems to discard from latency stats (CUDA warmup).",
    )
    ap.add_argument(
        "--fault-delta",
        type=float,
        default=10000.0,
        help="Used only to derive threshold = |fault_delta|*clear_threshold_mul.",
    )
    ap.add_argument("--clear-threshold-mul", type=float, default=0.5)
    ap.add_argument(
        "--no-require-hash-answer",
        action="store_true",
        help="Allow last-number fallback when '#### N' line missing.",
    )
    ap.add_argument("--out-json", required=True)
    ap.add_argument(
        "--protect-capture-stats",
        action="store_true",
        help="protect mode only: count elements cleared per site (GPU sync per tensor; "
        "use for 1-case diagnostics, not full latency sweeps).",
    )
    ap.add_argument(
        "--compile-mode",
        default="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="clean_compile / protect_compile: torch.compile mode=...",
    )
    ap.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="clean_compile / protect_compile: pass fullgraph=True to torch.compile (often breaks on generate).",
    )
    ap.add_argument(
        "--protect-linear-backend",
        default="masked_fill",
        choices=["pt2_funct", "masked_fill"],
        help="protect_compile only: pt2_funct = 子图 compile(F.linear+where)，mode=reduce-overhead；"
        "勿叠整模 compile，用 --no-full-model-compile。默认 masked_fill + 整模 compile 更稳。",
    )
    ap.add_argument(
        "--no-full-model-compile",
        action="store_true",
        help="protect_compile only: 不 torch.compile 整模，仅依赖 ProtectLinear 内联的 "
        "fused_linear_outlier_pt2（max-autotune）。与 --protect-linear-backend masked_fill 组合时 "
        "则无 compile。",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    task = Gsm8kLatencyTask(
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        require_hash_answer=not args.no_require_hash_answer,
        warmup_samples=args.warmup_samples,
        latency_timing=(
            "e2e_generate"
            if args.mode
            in ("clean_compile", "protect_compile", "protect_triton", "triton_plain")
            else "hooks"
        ),
    )

    threshold = abs(args.fault_delta) * args.clear_threshold_mul
    print(
        f"[run_gsm8k_latency] mode={args.mode} model={args.model_id} dtype={args.dtype} "
        f"max_samples={args.max_samples} warmup={args.warmup_samples} "
        f"max_new_tokens={args.max_new_tokens} threshold={threshold}",
        flush=True,
    )

    hook_site_count = 0
    linear_wrap_count = 0
    inj = None

    if args.mode == "clean":
        ctx = nullcontext(None)
    elif args.mode == "clean_compile":
        runner.model = compile_model_for_latency(
            runner.model,
            compile_mode=args.compile_mode,
            fullgraph=bool(args.compile_fullgraph),
        )
        print(
            f"[run_gsm8k_latency] clean_compile: compile_mode={args.compile_mode} "
            f"fullgraph={bool(args.compile_fullgraph)}",
            flush=True,
        )
        ctx = nullcontext(None)
    elif args.mode == "protect_triton":
        linear_wrap_count = apply_protect_linears_qwen(
            runner.model,
            float(threshold),
            linear_backend="triton",
        )
        print(
            f"[run_gsm8k_latency] protect_triton: TritonProtectLinear(do_outlier=True) "
            f"count={linear_wrap_count}",
            flush=True,
        )
        ctx = nullcontext(None)
    elif args.mode == "triton_plain":
        linear_wrap_count = apply_protect_linears_qwen(
            runner.model,
            float(threshold),
            linear_backend="triton_plain",
        )
        print(
            f"[run_gsm8k_latency] triton_plain: TritonProtectLinear(do_outlier=False) "
            f"count={linear_wrap_count}",
            flush=True,
        )
        ctx = nullcontext(None)
    elif args.mode == "protect_compile":
        linear_wrap_count = apply_protect_linears_qwen(
            runner.model,
            float(threshold),
            linear_backend=str(args.protect_linear_backend),
        )
        full_model_compiled = not bool(args.no_full_model_compile)
        if full_model_compiled:
            runner.model = compile_model_for_latency(
                runner.model,
                compile_mode=args.compile_mode,
                fullgraph=bool(args.compile_fullgraph),
            )
        print(
            f"[run_gsm8k_latency] protect_compile: ProtectLinear count={linear_wrap_count} "
            f"linear_backend={args.protect_linear_backend} full_model_compile={full_model_compiled} "
            f"compile_mode={args.compile_mode} fullgraph={bool(args.compile_fullgraph)}",
            flush=True,
        )
        ctx = nullcontext(None)
    else:
        ctx = InjectionContext(
            model=runner.model,
            target_site=None,
            fault_delta=args.fault_delta,
            seed=args.seed,
            fault_index_mode="random",
            clear_exceptions=False,
            clear_threshold_mul=args.clear_threshold_mul,
            decode_step_inject_enable=False,
            site_strategy=SITE_STRATEGY_QWEN,
            protect_only=True,
            protect_capture_stats=bool(args.protect_capture_stats),
        )

    with ctx as inj_cm:
        inj = inj_cm
        if inj is not None:
            stats = inj.collect_hook_stats()
            hook_site_count = int(stats.registered_site_count)
            print(
                f"[run_gsm8k_latency] protect hooks registered: {hook_site_count} "
                f"(expected 224 for Qwen2.5-7B: 28 layers × 8)",
                flush=True,
            )
        result = task.run(runner)

    result["mode"] = args.mode
    result["summary"]["mode"] = args.mode
    result["summary"]["model_id"] = args.model_id
    result["summary"]["dtype"] = args.dtype
    result["summary"]["max_new_tokens"] = args.max_new_tokens
    result["summary"]["threshold"] = float(threshold)
    result["summary"]["hook_site_count"] = hook_site_count
    result["summary"]["protect_linear_count"] = int(linear_wrap_count)
    if args.mode in ("clean_compile", "protect_compile"):
        result["summary"]["compile_mode"] = str(args.compile_mode)
        result["summary"]["compile_fullgraph"] = bool(args.compile_fullgraph)
    if args.mode == "protect_compile":
        result["summary"]["protect_linear_backend"] = str(args.protect_linear_backend)
        result["summary"]["full_model_compiled"] = not bool(args.no_full_model_compile)
    if args.mode == "protect_triton":
        result["summary"]["protect_linear_backend"] = "triton"
        result["summary"]["full_model_compiled"] = False
        result["summary"]["triton_do_outlier"] = True
    if args.mode == "triton_plain":
        result["summary"]["protect_linear_backend"] = "triton_plain"
        result["summary"]["full_model_compiled"] = False
        result["summary"]["triton_do_outlier"] = False
    result["summary"].setdefault(
        "triton_kernel_used", args.mode in ("protect_triton", "triton_plain")
    )
    if inj is not None and getattr(inj, "protect_capture_stats", False):
        pcs = inj.get_protect_clear_stats()
        result["protect_clear_stats"] = pcs
        result["summary"]["protect_clear_total_elements"] = pcs["total_cleared_elements"]
        result["summary"]["protect_clear_sites_with_any"] = pcs["sites_with_any_clear"]
        print(
            json.dumps(
                {
                    "protect_clear_total_elements": pcs["total_cleared_elements"],
                    "protect_clear_sites_with_any": pcs["sites_with_any_clear"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    s = result["summary"]
    short = {
        "mode": s.get("mode"),
        "latency_timing": s.get("latency_timing"),
        "hook_site_count": s.get("hook_site_count"),
        "protect_linear_count": s.get("protect_linear_count"),
        "compile_mode": s.get("compile_mode"),
        "protect_linear_backend": s.get("protect_linear_backend"),
        "full_model_compiled": s.get("full_model_compiled"),
        "triton_kernel_used": s.get("triton_kernel_used"),
        "triton_do_outlier": s.get("triton_do_outlier"),
        "accuracy": s.get("accuracy"),
        "total_decode_steps": s.get("total_decode_steps"),
    }
    if s.get("latency_timing") == "e2e_generate":
        short.update(
            {
                "n": s.get("generate_total_ms_n"),
                "generate_total_ms_mean": s.get("generate_total_ms_mean"),
                "generate_total_ms_p50": s.get("generate_total_ms_p50"),
                "generate_total_ms_p95": s.get("generate_total_ms_p95"),
            }
        )
    else:
        short.update(
            {
                "n": s.get("prefill_ms_n"),
                "prefill_ms_mean": s.get("prefill_ms_mean"),
                "prefill_ms_p50": s.get("prefill_ms_p50"),
                "prefill_ms_p95": s.get("prefill_ms_p95"),
                "decode_ms_per_token_mean": s.get("decode_ms_per_token_mean"),
                "decode_ms_per_token_p50": s.get("decode_ms_per_token_p50"),
                "decode_ms_per_token_p95": s.get("decode_ms_per_token_p95"),
            }
        )
    print(json.dumps(short, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
