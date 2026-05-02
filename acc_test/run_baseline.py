from __future__ import annotations

import argparse
import json

from hellaswag_task import HellaSwagTask
from model_runner import ModelRunner


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-json", default="baseline_hellaswag.json")
    return ap.parse_args()


def main():
    args = parse_args()
    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    task = HellaSwagTask(split=args.split, max_samples=args.max_samples, seed=args.seed)
    result = runner.run_task(task, inject_site=None, seed=args.seed)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(json.dumps(result["run_meta"], ensure_ascii=False))


if __name__ == "__main__":
    main()
