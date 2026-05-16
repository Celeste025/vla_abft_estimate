"""GSM8K ACC v2: warmup (per-site m/M) then formal run with thr-mMg + golden restore + metrics export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset

from gsm8k_task import _build_prompt as gsm8k_build_prompt, extract_final_answer
from inject import SITE_STRATEGY_QWEN, InjectionContext
from model_runner import ModelRunner
from results_layout import (
    build_run_config_segment,
    default_results_root,
    ensure_results_subdirs,
    results_run_dir,
    write_run_meta,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=200, help="Total GSM8K problems (includes warmup re-run)")
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--fault-mode", choices=["fixed", "rand2pow"], default="rand2pow")
    ap.add_argument("--fault-delta", type=float, default=10000.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--target-site", required=True, help="e.g. L12_q_proj")
    ap.add_argument(
        "--results-root",
        default=None,
        help="default: acc_test/results (results_layout.default_results_root)",
    )
    ap.add_argument("--decode-step-inject", action="store_true")
    ap.add_argument("--decode-step-max", type=int, default=150)
    ap.add_argument(
        "--acc-no-threshold",
        action="store_true",
        help="ACC v2: inject only; skip thr-mMg + golden restore. Path uses thr-none.",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root) if args.results_root else default_results_root()
    acc_thr_enabled = not bool(args.acc_no_threshold)
    run_dir = results_run_dir(
        results_root,
        model_id=args.model_id,
        dataset="gsm8k",
        n_total=int(args.max_samples),
        n_warmup=int(args.n_warmup),
        gamma=float(args.gamma),
        fault_mode=str(args.fault_mode),
        seed=int(args.seed),
        max_new_tokens=int(args.max_new_tokens),
        fault_delta=float(args.fault_delta) if args.fault_mode == "fixed" else None,
        acc_thr_enabled=acc_thr_enabled,
    )
    paths = ensure_results_subdirs(run_dir)

    meta: Dict[str, Any] = {
        "model_id": args.model_id,
        "dataset": "gsm8k",
        "split": args.split,
        "max_samples": int(args.max_samples),
        "n_warmup": int(args.n_warmup),
        "gamma": float(args.gamma),
        "fault_mode": args.fault_mode,
        "fault_delta": float(args.fault_delta),
        "acc_thr_enabled": acc_thr_enabled,
        "seed": int(args.seed),
        "max_new_tokens": int(args.max_new_tokens),
        "target_site": args.target_site,
        "run_config_segment": build_run_config_segment(
            n_total=int(args.max_samples),
            n_warmup=int(args.n_warmup),
            gamma=float(args.gamma),
            fault_mode=str(args.fault_mode),
            seed=int(args.seed),
            max_new_tokens=int(args.max_new_tokens),
            fault_delta=float(args.fault_delta) if args.fault_mode == "fixed" else None,
            acc_thr_enabled=acc_thr_enabled,
        ),
        "decode_step_inject": bool(args.decode_step_inject),
    }
    write_run_meta(paths, meta)

    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    inj = InjectionContext(
        model=runner.model,
        target_site=str(args.target_site),
        fault_delta=float(args.fault_delta),
        seed=int(args.seed),
        decode_step_inject_enable=bool(args.decode_step_inject),
        decode_step_max=int(args.decode_step_max),
        site_strategy=SITE_STRATEGY_QWEN,
        protect_only=False,
        acc_v2=True,
        thr_gamma=float(args.gamma),
        acc_v2_threshold_enable=acc_thr_enabled,
        fault_mode=str(args.fault_mode),
    )

    ds = load_dataset("gsm8k", "main", split=args.split)
    ds = ds.shuffle(seed=int(args.seed)).select(range(min(int(args.max_samples), len(ds)))))

    n_wu = int(args.n_warmup)

    with inj:
        inj.reset_site_bounds()
        inj.set_warmup(True)
        for idx in range(n_wu):
            ex = ds[idx]
            prompt = gsm8k_build_prompt(ex["question"])
            runner.generate_text(
                prompt,
                max_new_tokens=int(args.max_new_tokens),
                temperature=0.0,
                top_p=1.0,
            )
        inj.set_warmup(False)
        inj.reset_acc_metrics()

        per_rows = []
        correct = 0
        for idx in range(len(ds)):
            ex = ds[idx]
            q = ex["question"]
            gold = extract_final_answer(ex["answer"])
            prompt = gsm8k_build_prompt(q)
            if getattr(inj, "decode_step_inject_enable", False):
                inj.begin_decode()
            gen = runner.generate_text(
                prompt,
                max_new_tokens=int(args.max_new_tokens),
                temperature=0.0,
                top_p=1.0,
            )
            if getattr(inj, "decode_step_inject_enable", False):
                inj.end_decode()
            pred = extract_final_answer(gen)
            ok = int(gold is not None and pred is not None and pred == gold)
            correct += ok
            per_rows.append(
                {
                    "idx": idx,
                    "is_warmup_duplicate": int(idx < n_wu),
                    "correct": ok,
                    "gold": gold,
                    "pred": pred,
                }
            )

    inj.export_acc_v2_metrics(paths, site_id=str(args.target_site))
    acc_path = paths["json"] / "per_example_summary.json"
    with open(acc_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": float(correct) / float(len(per_rows)) if per_rows else 0.0,
                "total": len(per_rows),
                "correct": correct,
                "rows": per_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps({"run_dir": str(run_dir), "accuracy": float(correct) / len(per_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
