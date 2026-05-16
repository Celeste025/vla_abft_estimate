#!/usr/bin/env python3
"""逐题跑 GSM8K + protect_only + protect_capture_stats，与 latency 任务同一套采样顺序。

与 `Gsm8kLatencyTask` / `run_gsm8k_latency.py` 对齐：
  load_dataset("gsm8k","main",split=test) -> shuffle(seed) -> 取前 (warmup + max_samples) 条。
  仅对其中 **非 warmup** 的每条题单独 `reset_protect_clear_counts()` 后 `generate_text`，
  统计本题全链路（prefill + decode）|x|>threshold 被清零的元素个数（按 site 聚合）。

注意：开启 capture 时每层每个 tensor 会 `mask.sum().item()`，非常慢，仅用于诊断。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset

from gsm8k_task import _build_prompt, extract_final_answer
from inject import SITE_STRATEGY_QWEN, InjectionContext
from model_runner import ModelRunner
from results_layout import default_results_root


def parse_args() -> argparse.Namespace:
    rr = default_results_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=8, help="与 latency 里非 warmup 题数一致。")
    ap.add_argument("--warmup-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--fault-delta", type=float, default=10000.0)
    ap.add_argument("--clear-threshold-mul", type=float, default=0.5)
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(rr / "gsm8k_protect_capture_per_problem.json"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    threshold = abs(args.fault_delta) * args.clear_threshold_mul

    ds = load_dataset("gsm8k", "main", split=args.split)
    n_take = int(args.max_samples) + int(args.warmup_samples)
    ds = ds.shuffle(seed=int(args.seed)).select(range(min(n_take, len(ds))))

    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    per_problem: List[Dict[str, Any]] = []
    totals: List[int] = []

    with InjectionContext(
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
        protect_capture_stats=True,
    ) as inj:
        st = inj.collect_hook_stats()
        print(
            f"[protect_capture_sweep] threshold={threshold} hooks={st.registered_site_count} "
            f"split={args.split} seed={args.seed} warmup={args.warmup_samples} n_eval={args.max_samples}",
            flush=True,
        )
        runner._active_injector = inj
        try:
            for pos, ex in enumerate(ds):
                if pos < int(args.warmup_samples):
                    continue
                q = ex["question"]
                gold = extract_final_answer(ex["answer"])
                prompt = _build_prompt(q)
                inj.reset_protect_clear_counts()

                gen = runner.generate_text(
                    prompt,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=0.0,
                    top_p=1.0,
                )
                pstats = inj.get_protect_clear_stats()
                tot = int(pstats["total_cleared_elements"])
                totals.append(tot)

                pred = None
                m = re.search(r"####\s*([-+]?\d[\d,]*)", gen)
                if m:
                    pred = m.group(1).replace(",", "")
                correct = int(gold is not None and pred is not None and pred == gold)
                gen_tok = int(len(runner.tokenizer(gen, add_special_tokens=False)["input_ids"]))

                row: Dict[str, Any] = {
                    "position_in_shuffled_batch": int(pos),
                    "dataset_index_in_batch": int(pos - args.warmup_samples),
                    "question_head": q[:120].replace("\n", " "),
                    "gold": gold,
                    "pred": pred,
                    "correct": correct,
                    "gen_tokens": gen_tok,
                    "protect_clear": pstats,
                }
                per_problem.append(row)
                print(
                    f"[pos {pos}] total_cleared={tot} sites_with_any={pstats['sites_with_any_clear']} "
                    f"correct={correct} by_site_n={len(pstats['by_site'])}",
                    flush=True,
                )
        finally:
            runner._active_injector = None

    out: Dict[str, Any] = {
        "config": {
            "model_id": args.model_id,
            "dtype": args.dtype,
            "seed": args.seed,
            "warmup_samples": args.warmup_samples,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "threshold": float(threshold),
            "fault_delta": args.fault_delta,
            "clear_threshold_mul": args.clear_threshold_mul,
        },
        "per_problem": per_problem,
        "summary": {
            "total_cleared_elements_per_problem": totals,
            "total_cleared_elements_sum": int(sum(totals)),
            "total_cleared_elements_mean": float(sum(totals) / len(totals)) if totals else 0.0,
            "problems_with_zero_clears": int(sum(1 for t in totals if t == 0)),
            "problems_with_nonzero_clears": int(sum(1 for t in totals if t > 0)),
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
