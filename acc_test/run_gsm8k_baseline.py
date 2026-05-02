from __future__ import annotations

import argparse
import json

from gsm8k_task import Gsm8kTask
from model_runner import ModelRunner


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--require-hash-answer", action="store_true", default=True)
    ap.add_argument("--no-require-hash-answer", action="store_false", dest="require_hash_answer")
    ap.add_argument("--inject-site", default="")
    ap.add_argument("--fault-delta", type=float, default=0.0)
    ap.add_argument("--decode-step-max", type=int, default=150)
    ap.add_argument("--out-json", default="gsm8k_baseline.json")
    return ap.parse_args()


def main():
    args = parse_args()
    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    task = Gsm8kTask(
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        require_hash_answer=args.require_hash_answer,
    )
    inject_site = args.inject_site.strip() or None
    decode_enable = inject_site is not None and args.fault_delta != 0.0
    result = runner.run_task(
        task,
        inject_site=inject_site,
        fault_delta=args.fault_delta if decode_enable else 0.0,
        seed=args.seed,
        decode_step_inject_enable=decode_enable,
        decode_step_max=args.decode_step_max,
    )
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

