"""Run Qwen decoder op-output stats capture on GSM8K or HellaSwag (default 10 testcases)."""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import torch
from datasets import load_dataset

from gsm8k_task import _build_prompt as gsm8k_build_prompt
from model_runner import ModelRunner
from op_output_stats_capture import QwenDecoderOpStatsCapture


def _hellaswag_forward_one(runner: ModelRunner, ctx: str, ending: str) -> None:
    tok = runner.tokenizer
    device = runner.device
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    end_ids = tok(" " + ending, add_special_tokens=False)["input_ids"]
    full_ids = ctx_ids + end_ids
    if len(full_ids) < 2 or len(end_ids) == 0:
        return
    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    runner.forward(input_ids=input_ids, attention_mask=attention_mask)


def run_gsm8k(
    runner: ModelRunner,
    cap: QwenDecoderOpStatsCapture,
    *,
    max_samples: int,
    seed: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        cap.begin_episode()
        prompt = gsm8k_build_prompt(ex["question"])
        runner.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
        agg = cap.end_episode()
        rows.append(
            {
                "idx": idx,
                "question": ex["question"][:200],
                "aggregate_by_site": agg,
            }
        )
    return rows


def run_hellaswag(
    runner: ModelRunner,
    cap: QwenDecoderOpStatsCapture,
    *,
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    ds = load_dataset("hellaswag", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
    rows: List[Dict[str, Any]] = []
    for idx, ex in enumerate(ds):
        ctx = ex["ctx"]
        endings = ex["endings"]
        cap.begin_episode()
        for ending in endings:
            _hellaswag_forward_one(runner, ctx, ending)
        agg = cap.end_episode()
        rows.append({"idx": idx, "ind": ex.get("ind"), "aggregate_by_site": agg})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["gsm8k", "hellaswag"], required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-new-tokens", type=int, default=64, help="GSM8K generate length")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    with QwenDecoderOpStatsCapture(runner.model) as cap:
        reg = cap.collect_registration_stats()
        if reg["missing_sites"]:
            raise RuntimeError(f"missing hook sites (first 10): {reg['missing_sites'][:10]}")
        if args.benchmark == "gsm8k":
            testcases = run_gsm8k(
                runner,
                cap,
                max_samples=args.max_samples,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            testcases = run_hellaswag(runner, cap, max_samples=args.max_samples, seed=args.seed)

    out: Dict[str, Any] = {
        "meta": {
            "benchmark": args.benchmark,
            "model_id": args.model_id,
            "dtype": args.dtype,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens if args.benchmark == "gsm8k" else None,
            "registration": reg,
        },
        "testcases": testcases,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({"wrote": args.out_json, "n_testcases": len(testcases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
