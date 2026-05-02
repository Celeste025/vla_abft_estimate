from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

from gsm8k_task import Gsm8kTask
from model_runner import ModelRunner


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--indices-json", default="results/gsm8k_test_shared100_indices.json")
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument(
        "--site",
        default=None,
        help="Optional explicit site_id (e.g. L14_v_proj). If not set, defaults to L{layer}_v_proj.",
    )
    ap.add_argument("--fault-delta", type=float, default=10000.0)
    ap.add_argument("--decode-step-max", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--clear-threshold-mul", type=float, default=0.5)
    ap.add_argument("--out-json", default="results/gsm8k_recovery_case.json")
    return ap.parse_args()


def default_site(layer_idx: int) -> str:
    return f"L{layer_idx}_v_proj"


def idx_to_map(per_example: List[Dict]) -> Dict[int, Dict]:
    return {int(ex["idx"]): ex for ex in per_example}


def _short(text: str, limit: int = 800) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[:limit] + "\n...<truncated>..."


def first_diff_token_idx(tok, a: str, b: str) -> Optional[int]:
    a_ids = tok(a, add_special_tokens=False)["input_ids"]
    b_ids = tok(b, add_special_tokens=False)["input_ids"]
    n = min(len(a_ids), len(b_ids))
    for i in range(n):
        if a_ids[i] != b_ids[i]:
            return i
    if len(a_ids) != len(b_ids):
        return n
    return None


def main():
    args = parse_args()
    idx_payload = json.load(open(args.indices_json, "r", encoding="utf-8"))
    indices = idx_payload["indices"][: int(args.max_samples)]

    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )

    task = Gsm8kTask(
        split="test",
        max_samples=int(args.max_samples),
        seed=int(args.seed),
        max_new_tokens=512,
        require_hash_answer=True,
        indices=[int(i) for i in indices],
        raw_generation_char_limit=0,
    )

    # Baseline
    base = runner.run_task(task, inject_site=None, seed=args.seed)
    base_map = idx_to_map(base["per_example"])

    site = (args.site or default_site(int(args.layer))).strip()

    fault = runner.run_task(
        task,
        inject_site=site,
        fault_delta=args.fault_delta,
        seed=args.seed,
        decode_step_inject_enable=True,
        decode_step_max=args.decode_step_max,
    )
    clear = runner.run_task(
        task,
        inject_site=site,
        fault_delta=args.fault_delta,
        seed=args.seed,
        clear_exceptions=True,
        clear_threshold_mul=args.clear_threshold_mul,
        decode_step_inject_enable=True,
        decode_step_max=args.decode_step_max,
    )
    fault_map = idx_to_map(fault["per_example"])
    clear_map = idx_to_map(clear["per_example"])

    best: Optional[int] = None
    best_payload: Optional[Dict] = None
    for idx in sorted(base_map.keys()):
        b = base_map[idx]
        f = fault_map[idx]
        c = clear_map[idx]
        if int(b["correct"]) == 1 and int(f["correct"]) == 0 and int(c["correct"]) == 1:
            best = idx
            tok = runner.tokenizer
            diff_fault = first_diff_token_idx(tok, b["raw_generation"], f["raw_generation"])
            diff_clear = first_diff_token_idx(tok, b["raw_generation"], c["raw_generation"])
            best_payload = {
                "site": site,
                "idx": idx,
                "question": b["question"],
                "gold": b["gold"],
                "baseline": {
                    "pred": b["pred"],
                    "correct": b["correct"],
                    "raw_generation": b["raw_generation"],
                },
                "fault": {
                    "pred": f["pred"],
                    "correct": f["correct"],
                    "raw_generation": f["raw_generation"],
                    "decode_target_step": f.get("decode_target_step"),
                    "decode_injected": f.get("decode_injected"),
                    "first_diff_token_idx_vs_baseline": diff_fault,
                },
                "clear": {
                    "pred": c["pred"],
                    "correct": c["correct"],
                    "raw_generation": c["raw_generation"],
                    "decode_target_step": c.get("decode_target_step"),
                    "decode_injected": c.get("decode_injected"),
                    "first_diff_token_idx_vs_baseline": diff_clear,
                },
            }
            break

    if best is None or best_payload is None:
        print("NOT_FOUND: no (baseline correct, fault wrong, clear correct) case found.", flush=True)
        return

    idx = best
    print(f"FOUND site={site} idx={idx}", flush=True)
    print("QUESTION:", best_payload["question"], flush=True)
    print("GOLD:", best_payload["gold"], flush=True)
    print("\n=== BASELINE ===", flush=True)
    print("pred:", best_payload["baseline"]["pred"], "correct:", best_payload["baseline"]["correct"], flush=True)
    print(_short(best_payload["baseline"]["raw_generation"]), flush=True)
    print("\n=== FAULT (no clear) ===", flush=True)
    print("pred:", best_payload["fault"]["pred"], "correct:", best_payload["fault"]["correct"], flush=True)
    print(_short(best_payload["fault"]["raw_generation"]), flush=True)
    print("\n=== FAULT + CLEAR ===", flush=True)
    print("pred:", best_payload["clear"]["pred"], "correct:", best_payload["clear"]["correct"], flush=True)
    print(_short(best_payload["clear"]["raw_generation"]), flush=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)
    print(f"\nEXPORTED {args.out_json}", flush=True)


if __name__ == "__main__":
    main()

