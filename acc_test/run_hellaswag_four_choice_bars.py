"""HellaSwag case0: four-choice log-likelihood scores (clean + mlp_down fault)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from hellaswag_logits_capture import score_ending_loglik, score_four_choices, softmax_scores
from inject import InjectionContext
from model_runner import ModelRunner

CASE_IDX = 0
INJECT_SITE = "mlp_down"
FAULT_DELTAS = [10.0, 100.0, 1000.0]
SEED = 2026


def _run_config(
    runner: ModelRunner,
    ctx: str,
    endings: List[str],
    label: int,
    *,
    inject_layer: int,
    fault_delta: Optional[float] = None,
) -> Dict[str, Any]:
    target_site = f"L{inject_layer}_{INJECT_SITE}"
    if fault_delta is None:
        scores = score_four_choices(runner, ctx, endings)
        probs = softmax_scores(scores)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        return {
            "run": "clean",
            "scores": scores,
            "probs": probs,
            "label": label,
            "pred": pred,
            "correct": int(pred == label),
            "target_site": target_site,
            "inject_layer": inject_layer,
            "inject_site": INJECT_SITE,
            "fault_delta": None,
            "inject_count": 0,
            "pre_inject_min": None,
            "pre_inject_max": None,
            "per_choice_pre_inject": [],
        }

    per_choice_pre: List[Dict[str, Optional[float]]] = []
    scores: List[float] = []

    with InjectionContext(
        runner.model,
        target_site=target_site,
        fault_mode="fixed",
        fault_delta=float(fault_delta),
        seed=SEED,
        fault_index_mode="random",
        threshold_enable=False,
        inject_enable=True,
        site_set="matmul",
    ) as inj:
        inj.reset_pre_inject_stats()
        for ending in endings:
            scores.append(score_ending_loglik(runner, ctx, ending))
            per_choice_pre.append(
                {
                    "pre_inject_min": inj.last_pre_inject_min,
                    "pre_inject_max": inj.last_pre_inject_max,
                }
            )
        inject_count = inj.inject_count
        last_flat_idx = inj.last_inject_flat_idx

    pre_mins = [x["pre_inject_min"] for x in per_choice_pre if x["pre_inject_min"] is not None]
    pre_maxs = [x["pre_inject_max"] for x in per_choice_pre if x["pre_inject_max"] is not None]
    probs = softmax_scores(scores)
    pred = int(max(range(len(scores)), key=lambda i: scores[i]))

    return {
        "run": "fault",
        "scores": scores,
        "probs": probs,
        "label": label,
        "pred": pred,
        "correct": int(pred == label),
        "target_site": target_site,
        "inject_layer": inject_layer,
        "inject_site": INJECT_SITE,
        "fault_mode": "fixed",
        "fault_delta": float(fault_delta),
        "threshold_enable": False,
        "inject_count": inject_count,
        "last_inject_flat_idx": last_flat_idx,
        "pre_inject_min": min(pre_mins) if pre_mins else None,
        "pre_inject_max": max(pre_maxs) if pre_maxs else None,
        "per_choice_pre_inject": per_choice_pre,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--inject-layer", type=int, default=12)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="default: artifacts/hellaswag_four_choice/case0_L{L}_mlp_down",
    )
    args = ap.parse_args()

    inject_layer = int(args.inject_layer)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent
        / "artifacts"
        / "hellaswag_four_choice"
        / f"case0_L{inject_layer}_{INJECT_SITE}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    ds = load_dataset("hellaswag", split="validation").shuffle(seed=SEED)
    ex = ds[CASE_IDX]
    ctx = ex["ctx"]
    endings = list(ex["endings"])
    label = int(ex["label"])

    meta_common: Dict[str, Any] = {
        "model_id": args.model_id,
        "benchmark": "hellaswag",
        "case_idx": CASE_IDX,
        "label": label,
        "ctx_preview": ctx[:200],
        "endings": endings,
        "seed": SEED,
        "inject_layer": inject_layer,
        "inject_site": INJECT_SITE,
    }

    clean_payload = {
        **meta_common,
        **_run_config(
            runner, ctx, endings, label, inject_layer=inject_layer, fault_delta=None
        ),
    }
    clean_path = out_dir / "clean.json"
    clean_path.write_text(json.dumps(clean_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fault_paths: Dict[str, str] = {}
    fault_summary: Dict[str, Any] = {}
    for fd in FAULT_DELTAS:
        fault_payload = {
            **meta_common,
            **_run_config(
                runner, ctx, endings, label, inject_layer=inject_layer, fault_delta=fd
            ),
        }
        fault_path = out_dir / f"fault_fd{fd:g}.json"
        fault_path.write_text(json.dumps(fault_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        fault_paths[f"fd{fd:g}"] = str(fault_path)
        fault_summary[f"fd{fd:g}"] = {
            "pred": fault_payload["pred"],
            "correct": fault_payload["correct"],
            "pre_inject_min": fault_payload["pre_inject_min"],
            "pre_inject_max": fault_payload["pre_inject_max"],
        }

    summary_path = out_dir / "meta.json"
    summary_path.write_text(
        json.dumps(
            {
                **meta_common,
                "clean_path": str(clean_path),
                "fault_paths": fault_paths,
                "fault_deltas": FAULT_DELTAS,
                "fault_summary": fault_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "inject_layer": inject_layer,
                "clean_pred": clean_payload["pred"],
                "clean_correct": clean_payload["correct"],
                "fault_summary": fault_summary,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
