from __future__ import annotations

import argparse
import json
import os

from gsm8k_task import Gsm8kTask
from model_runner import ModelRunner
from results_layout import default_results_root


def parse_args():
    rr = default_results_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--indices-json", default=str(rr / "gsm8k_test_shared100_indices.json"))
    ap.add_argument("--max-samples", type=int, default=3)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--fault-delta", type=float, default=10000.0)
    ap.add_argument("--decode-step-max", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default=str(rr / "_trace_smoke"))
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    idx_payload = json.load(open(args.indices_json, "r", encoding="utf-8"))
    indices = idx_payload["indices"][: int(args.max_samples)]

    runner = ModelRunner(model_id=args.model_id, device=args.device, dtype=args.dtype)
    site = f"L{int(args.layer)}_v_proj"

    def run_one(tag: str, inject: bool, clear: bool):
        trace_path = os.path.join(args.out_dir, f"{tag}.jsonl")
        if os.path.exists(trace_path):
            os.remove(trace_path)

        task = Gsm8kTask(
            split="test",
            max_samples=int(args.max_samples),
            seed=int(args.seed),
            max_new_tokens=512,
            require_hash_answer=True,
            indices=[int(i) for i in indices],
            raw_generation_char_limit=0,
            trace_jsonl_path=trace_path,
            trace_run_tag=tag,
        )

        res = runner.run_task(
            task,
            inject_site=site if inject else None,
            fault_delta=args.fault_delta,
            seed=args.seed,
            fault_index_mode="max_abs",
            clear_exceptions=bool(clear),
            clear_threshold_mul=0.5,
            decode_step_inject_enable=bool(inject),
            decode_step_max=args.decode_step_max,
        )
        out_json = os.path.join(args.out_dir, f"{tag}.summary.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res["summary"], f, ensure_ascii=False, indent=2)
        print(f"[{tag}] summary={res['summary']} trace={trace_path}", flush=True)

    run_one("baseline", inject=False, clear=False)
    run_one("fault_maxabs", inject=True, clear=False)
    run_one("fault_maxabs_clearhalf", inject=True, clear=True)


if __name__ == "__main__":
    main()

