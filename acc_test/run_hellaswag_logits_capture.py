"""HellaSwag: capture full-sequence token log-probability vectors (clean vs fault)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset

from hellaswag_logits_capture import (
    build_hellaswag_inputs,
    ending_logp_sum,
    save_token_logp,
    token_logp_vector,
)
from inject import InjectionContext, SiteSet
from model_runner import ModelRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--case-idx", type=int, default=0)
    ap.add_argument("--ending-idx", type=int, default=0)
    ap.add_argument("--inject-layer", type=int, default=0)
    ap.add_argument(
        "--inject-site",
        choices=["mlp_down", "mlp_residual"],
        default="mlp_down",
    )
    ap.add_argument("--fault-delta", type=float, default=100.0)
    ap.add_argument("--fault-index-mode", choices=["random", "max_abs"], default="random")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    case_idx = int(args.case_idx)
    ending_idx = int(args.ending_idx)
    inject_layer = int(args.inject_layer)
    inject_site = str(args.inject_site)
    fault_delta = float(args.fault_delta)
    target_site = f"L{inject_layer}_{inject_site}"
    site_set: SiteSet = "matmul" if inject_site == "mlp_down" else "nonmatmul"

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent
        / "artifacts"
        / "hellaswag_logits_diff"
        / f"case{case_idx}_end{ending_idx}_L{inject_layer}_{inject_site}_fd{fault_delta:g}"
    )
    clean_path = out_dir / "clean" / "token_logp.pt"
    fault_path = out_dir / "fault" / "token_logp.pt"

    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    ds = load_dataset("hellaswag", split="validation")
    ds = ds.shuffle(seed=int(args.seed))
    ex = ds[int(case_idx)]
    ending = list(ex["endings"])[int(ending_idx)]
    ctx = ex["ctx"]
    input_ids, attention_mask, tok_meta = build_hellaswag_inputs(runner, ctx, ending)

    logits_clean = runner.forward(input_ids=input_ids, attention_mask=attention_mask)
    vec_clean = token_logp_vector(logits_clean, input_ids)
    save_token_logp(
        clean_path,
        vec_clean,
        {"run": "clean", "ending_logp_sum": ending_logp_sum(vec_clean, tok_meta)},
    )

    inject_count = 0
    last_flat_idx = None
    with InjectionContext(
        runner.model,
        target_site=target_site,
        fault_mode="fixed",
        fault_delta=fault_delta,
        seed=int(args.seed),
        fault_index_mode=str(args.fault_index_mode),
        threshold_enable=False,
        inject_enable=True,
        site_set=site_set,
    ) as inj:
        logits_fault = runner.forward(input_ids=input_ids, attention_mask=attention_mask)
        inject_count = inj.inject_count
        last_flat_idx = inj.last_inject_flat_idx

    vec_fault = token_logp_vector(logits_fault, input_ids)
    save_token_logp(
        fault_path,
        vec_fault,
        {"run": "fault", "ending_logp_sum": ending_logp_sum(vec_fault, tok_meta)},
    )

    payload: Dict[str, Any] = {
        "model_id": args.model_id,
        "benchmark": "hellaswag",
        "case_idx": case_idx,
        "ending_idx": ending_idx,
        "label": int(ex["label"]),
        "ctx_preview": ctx[:200],
        "ending": ending,
        "target_site": target_site,
        "inject_layer": inject_layer,
        "inject_site": inject_site,
        "site_set": site_set,
        "fault_mode": "fixed",
        "fault_delta": fault_delta,
        "threshold_enable": False,
        "inject_count": inject_count,
        "last_inject_flat_idx": last_flat_idx,
        "token_meta": tok_meta,
        "vector_shape": list(vec_clean.shape),
        "clean_path": str(clean_path),
        "fault_path": str(fault_path),
        "ending_logp_sum_clean": ending_logp_sum(vec_clean, tok_meta),
        "ending_logp_sum_fault": ending_logp_sum(vec_fault, tok_meta),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "meta": str(meta_path),
                "vector_shape": list(vec_clean.shape),
                "inject_count": inject_count,
                "ending_logp_sum_clean": payload["ending_logp_sum_clean"],
                "ending_logp_sum_fault": payload["ending_logp_sum_fault"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
