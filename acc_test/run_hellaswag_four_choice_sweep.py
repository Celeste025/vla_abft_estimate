"""HellaSwag: sweep fault_delta at one layer/site, four-choice log-likelihood."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from hellaswag_logits_capture import score_ending_loglik, score_four_choices, softmax_scores
from inject import InjectionContext
from model_runner import ModelRunner

INJECT_LAYER = 0
INJECT_SITE = "mlp_down"
FAULT_DELTAS = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 5000.0, 10000.0]
SEED = 2026


def _parse_case_indices(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _default_out_dir(base: Path, case_idx: int, inject_layer: int, inject_site: str) -> Path:
    return base / "hellaswag_four_choice" / f"case{case_idx}_L{inject_layer}_{inject_site}_sweep"


def _score_fault_four(
    runner: ModelRunner,
    ctx: str,
    endings: List[str],
    *,
    inject_layer: int,
    inject_site: str,
    fault_delta: float,
    inject_seed: int,
) -> Dict[str, Any]:
    target_site = f"L{inject_layer}_{inject_site}"
    per_choice_pre: List[Dict[str, Optional[float]]] = []
    scores: List[float] = []

    with InjectionContext(
        runner.model,
        target_site=target_site,
        fault_mode="fixed",
        fault_delta=float(fault_delta),
        seed=int(inject_seed),
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

    pre_mins = [x["pre_inject_min"] for x in per_choice_pre if x["pre_inject_min"] is not None]
    pre_maxs = [x["pre_inject_max"] for x in per_choice_pre if x["pre_inject_max"] is not None]
    probs = softmax_scores(scores)
    pred = int(max(range(len(scores)), key=lambda i: scores[i]))

    return {
        "fault_delta": float(fault_delta),
        "scores": scores,
        "probs": probs,
        "pred": pred,
        "inject_count": inject_count,
        "pre_inject_min": min(pre_mins) if pre_mins else None,
        "pre_inject_max": max(pre_maxs) if pre_maxs else None,
        "per_choice_pre_inject": per_choice_pre,
    }


def run_case_sweep(
    runner: ModelRunner,
    ds: Any,
    *,
    case_idx: int,
    inject_layer: int,
    inject_site: str,
    fault_deltas: List[float],
    out_dir: Path,
    model_id: str,
) -> Dict[str, Any]:
    ex = ds[int(case_idx)]
    ctx = ex["ctx"]
    endings = list(ex["endings"])
    label = int(ex["label"])

    clean_scores = score_four_choices(runner, ctx, endings)
    clean_probs = softmax_scores(clean_scores)
    clean_pred = int(max(range(len(clean_scores)), key=lambda i: clean_scores[i]))

    sweep_rows: List[Dict[str, Any]] = []
    for fd in fault_deltas:
        row = _score_fault_four(
            runner,
            ctx,
            endings,
            inject_layer=inject_layer,
            inject_site=inject_site,
            fault_delta=fd,
            inject_seed=SEED + int(case_idx),
        )
        row["correct"] = int(row["pred"] == label)
        sweep_rows.append(row)
        print(
            json.dumps(
                {
                    "case_idx": case_idx,
                    "fault_delta": fd,
                    "scores": row["scores"],
                    "pred": row["pred"],
                    "label": label,
                    "correct": row["correct"],
                    "pre_inject": [row["pre_inject_min"], row["pre_inject_max"]],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload: Dict[str, Any] = {
        "model_id": model_id,
        "benchmark": "hellaswag",
        "case_idx": int(case_idx),
        "label": label,
        "ctx_preview": ctx[:200],
        "endings": endings,
        "seed": SEED,
        "inject_layer": inject_layer,
        "inject_site": inject_site,
        "target_site": f"L{inject_layer}_{inject_site}",
        "fault_deltas": fault_deltas,
        "clean": {
            "fault_delta": 0.0,
            "scores": clean_scores,
            "probs": clean_probs,
            "pred": clean_pred,
            "correct": int(clean_pred == label),
        },
        "sweep": sweep_rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out_dir": str(out_dir), "out_path": str(out_path), "payload": payload}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--inject-layer", type=int, default=INJECT_LAYER)
    ap.add_argument("--inject-site", default=INJECT_SITE, help="e.g. mlp_down, v_proj")
    ap.add_argument("--case-indices", default="0", help="comma-separated case indices, e.g. 0,1,2")
    ap.add_argument(
        "--fault-deltas",
        default=",".join(str(int(x)) if x == int(x) else str(x) for x in FAULT_DELTAS),
    )
    ap.add_argument("--out-dir", default=None, help="override base dir for a single case only")
    ap.add_argument(
        "--artifact-root",
        default=None,
        help="default: acc_test/artifacts",
    )
    args = ap.parse_args()

    inject_layer = int(args.inject_layer)
    inject_site = str(args.inject_site)
    case_indices = _parse_case_indices(args.case_indices)
    fault_deltas = [float(x.strip()) for x in str(args.fault_deltas).split(",") if x.strip()]

    artifact_root = (
        Path(args.artifact_root)
        if args.artifact_root
        else Path(__file__).resolve().parent / "artifacts"
    )

    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    ds = load_dataset("hellaswag", split="validation").shuffle(seed=SEED)

    summaries: List[Dict[str, Any]] = []
    for case_idx in case_indices:
        if args.out_dir and len(case_indices) == 1:
            out_dir = Path(args.out_dir)
        else:
            out_dir = _default_out_dir(artifact_root, case_idx, inject_layer, inject_site)

        info = run_case_sweep(
            runner,
            ds,
            case_idx=case_idx,
            inject_layer=inject_layer,
            inject_site=inject_site,
            fault_deltas=fault_deltas,
            out_dir=out_dir,
            model_id=args.model_id,
        )
        p = info["payload"]
        summaries.append(
            {
                "case_idx": case_idx,
                "label": p["label"],
                "out_dir": info["out_dir"],
                "clean_pred": p["clean"]["pred"],
                "clean_correct": p["clean"]["correct"],
                "fault_flip": [
                    {"fd": r["fault_delta"], "pred": r["pred"], "correct": r["correct"]}
                    for r in p["sweep"]
                    if not r["correct"]
                ],
            }
        )

    print(json.dumps({"cases": summaries}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
