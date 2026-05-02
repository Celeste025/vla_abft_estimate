from __future__ import annotations

import argparse
import json
import random

from hellaswag_task import HellaSwagTask, attach_delta_scores
from inject import list_sites
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
    ap.add_argument("--inject-site", default="")
    ap.add_argument("--fault-delta", type=float, default=10000.0)
    ap.add_argument("--out-json", default="inject_smoke_hellaswag.json")
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

    baseline = runner.run_task(task, inject_site=None, seed=args.seed)
    site = args.inject_site.strip()
    if not site:
        rng = random.Random(args.seed)
        site = rng.choice(list_sites(runner.model))
    fault = runner.run_task(task, inject_site=site, fault_delta=args.fault_delta, seed=args.seed)

    merged = attach_delta_scores(baseline, fault)
    merged["selected_site"] = site
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"selected_site={site}")
    print(json.dumps(merged["baseline_summary"], ensure_ascii=False))
    print(json.dumps(merged["fault_summary"], ensure_ascii=False))
    print(json.dumps(merged["run_meta_fault"], ensure_ascii=False))


if __name__ == "__main__":
    main()
