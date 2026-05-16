"""HellaSwag ACC sweep: baseline + per-site single-op fault on selected layers.

Baseline: no threshold detection, no injection.
Fault groups: for each selected layer and each of 8 ops, run:
  - warmup first n_warmup problems to collect per-site m/M
  - formal run on all selected problems with thr-mMg (golden) or thr-mMz (zero) + metrics
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
import torch
import torch.nn.functional as F

from inject import InjectionContext, SiteSet, layer_site_ids
from model_runner import ModelRunner
from results_layout import (
    AccThrAction,
    default_results_root,
    ensure_results_subdirs,
    results_run_dir,
    write_run_meta,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-samples", type=int, default=200, help="Total HellaSwag problems.")
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--fault-mode", choices=["fixed", "rand2pow"], default="rand2pow")
    ap.add_argument("--fault-delta", type=float, default=10000.0)
    ap.add_argument(
        "--acc-no-threshold",
        action="store_true",
        help="ACC: inject fault only; skip m/M·γ bounds check and restore. "
        "Metrics: runs increment at target site; tp/fp/fn/normal stay 0 (no detection). "
        "Run directory uses thr-none.",
    )
    ap.add_argument(
        "--acc-threshold-zero",
        action="store_true",
        help="ACC: m/M·γ bounds check; clear masked elements to 0 (thr-mMz). "
        "Mutually exclusive with --acc-no-threshold.",
    )
    ap.add_argument(
        "--layer-list",
        default="0,8,16,24",
        help="Selected layers only (not all layers), comma-separated.",
    )
    ap.add_argument(
        "--results-root",
        default=None,
        help="default: acc_test/results (see results_layout.default_results_root)",
    )
    ap.add_argument(
        "--site-set",
        choices=["matmul", "nonmatmul", "all"],
        default="matmul",
        help="Which hook sites to sweep. nonmatmul writes sweep_summary_nonmatmul.csv.",
    )
    ap.add_argument(
        "--reuse-baseline",
        action="store_true",
        help="If baseline_summary.json exists in run_dir, skip re-running baseline.",
    )
    return ap.parse_args()


def _slugify_site(site_id: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(site_id))
    return s.strip("_") or "site"


def op_type_from_site(site_id: str) -> str:
    suffix = site_id.split("_", 1)[1]
    if suffix in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return suffix
    if suffix == "attn_core":
        return "attn_core(qk^t+s*v)"
    return suffix


def _score_choice(runner: ModelRunner, ctx: str, ending: str) -> float:
    tok = runner.tokenizer
    device = runner.device

    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    end_ids = tok(" " + ending, add_special_tokens=False)["input_ids"]
    full_ids = ctx_ids + end_ids
    if len(full_ids) < 2 or len(end_ids) == 0:
        return -1e9

    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    logits = runner.forward(input_ids=input_ids, attention_mask=attention_mask)[0]

    log_probs = F.log_softmax(logits[:-1], dim=-1)
    target = input_ids[0, 1:]

    start = max(len(ctx_ids) - 1, 0)
    end = start + len(end_ids)
    token_logp = log_probs[start:end, :].gather(1, target[start:end].unsqueeze(-1)).squeeze(-1)
    return float(token_logp.sum().item())


def _run_examples(runner: ModelRunner, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_example: List[Dict[str, Any]] = []
    correct = 0
    for idx, ex in enumerate(examples):
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])
        scores = [_score_choice(runner, ctx, ending) for ending in endings]
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        ok = int(pred == label)
        correct += ok
        per_example.append(
            {
                "idx": idx,
                "ind": ex["ind"],
                "label": label,
                "pred": pred,
                "correct": ok,
                "scores": scores,
            }
        )
    total = len(per_example)
    acc = float(correct) / float(total) if total else 0.0
    return {
        "summary": {"total": total, "correct": correct, "accuracy": acc},
        "per_example": per_example,
    }


def _warmup_examples(runner: ModelRunner, examples: List[Dict[str, Any]], n_warmup: int) -> None:
    n = min(int(n_warmup), len(examples))
    for i in range(n):
        ex = examples[i]
        for ending in ex["endings"]:
            _ = _score_choice(runner, ex["ctx"], ending)


def _resolve_acc_thr(args: argparse.Namespace) -> tuple[bool, AccThrAction, str]:
    if args.acc_no_threshold and args.acc_threshold_zero:
        raise ValueError("--acc-no-threshold and --acc-threshold-zero are mutually exclusive")
    if args.acc_no_threshold:
        return False, "none", "golden"
    if args.acc_threshold_zero:
        return True, "zero", "zero"
    return True, "golden", "golden"


def _sweep_csv_name(site_set: SiteSet) -> str:
    return "sweep_summary_nonmatmul.csv" if site_set == "nonmatmul" else "sweep_summary.csv"


def main() -> None:
    args = parse_args()
    site_set: SiteSet = args.site_set  # type: ignore[assignment]
    acc_thr_enabled, acc_thr_action, acc_restore_mode = _resolve_acc_thr(args)
    layers = [int(x.strip()) for x in str(args.layer_list).split(",") if x.strip()]
    if not layers:
        raise ValueError("layer_list is empty")
    sites: List[Dict[str, Any]] = []
    for li in layers:
        for site in layer_site_ids(li, site_set):
            sites.append({"layer": li, "site_id": site, "op_type": op_type_from_site(site)})
    ops_per_layer = len(layer_site_ids(layers[0], site_set))

    results_root = Path(args.results_root) if args.results_root else default_results_root()
    run_dir = results_run_dir(
        results_root,
        model_id=args.model_id,
        dataset="hellaswag",
        n_total=int(args.max_samples),
        n_warmup=int(args.n_warmup),
        gamma=float(args.gamma),
        fault_mode=str(args.fault_mode),
        seed=int(args.seed),
        max_new_tokens=None,
        fault_delta=float(args.fault_delta) if args.fault_mode == "fixed" else None,
        acc_thr_enabled=acc_thr_enabled,
        acc_thr_action=acc_thr_action,
    )
    paths = ensure_results_subdirs(run_dir)
    sweep_csv = _sweep_csv_name(site_set)

    meta_path = paths["json"] / "run_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "model_id": args.model_id,
            "dataset": "hellaswag",
            "split": args.split,
            "max_samples": int(args.max_samples),
            "n_warmup": int(args.n_warmup),
            "gamma": float(args.gamma),
            "fault_mode": str(args.fault_mode),
            "fault_delta": float(args.fault_delta),
            "acc_thr_enabled": acc_thr_enabled,
            "acc_thr_action": acc_thr_action,
            "restore_mode": acc_restore_mode,
            "seed": int(args.seed),
            "layer_list": layers,
            "l_select": len(layers),
            "site_set": site_set,
            "ops_per_layer": ops_per_layer,
            "fault_group_count": len(sites),
            "sweep_csv": sweep_csv,
            "notes": "baseline has no threshold detection and no fault injection",
        }
    )
    write_run_meta(paths, meta)

    ds = load_dataset("hellaswag", split=args.split)
    ds = ds.shuffle(seed=int(args.seed)).select(range(min(int(args.max_samples), len(ds))))
    examples = [dict(x) for x in ds]
    if not examples:
        raise ValueError("no examples selected from dataset")

    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    baseline_path = paths["json"] / "baseline_summary.json"
    if args.reuse_baseline and baseline_path.is_file():
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline = {"summary": baseline_payload["summary"]}
        print(
            f"[baseline] reuse acc={baseline['summary']['accuracy']:.6f} "
            f"total={baseline['summary']['total']}",
            flush=True,
        )
    else:
        baseline = _run_examples(runner, examples)
        baseline_payload = {
            "mode": "baseline_clean",
            "summary": baseline["summary"],
        }
        baseline_path.write_text(
            json.dumps(baseline_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[baseline] acc={baseline['summary']['accuracy']:.6f} total={baseline['summary']['total']}",
            flush=True,
        )

    rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    total_groups = len(sites)
    for cur, item in enumerate(sites, start=1):
        site = item["site_id"]
        layer = item["layer"]
        op_type = item["op_type"]
        inj = InjectionContext(
            model=runner.model,
            target_site=site,
            fault_delta=float(args.fault_delta),
            seed=int(args.seed),
            thr_gamma=float(args.gamma),
            threshold_enable=acc_thr_enabled,
            restore_mode=acc_restore_mode,
            fault_mode=str(args.fault_mode),
            site_set=site_set,
        )
        with inj:
            inj.reset_site_bounds()
            inj.set_warmup(True)
            _warmup_examples(runner, examples, int(args.n_warmup))
            inj.set_warmup(False)
            inj.reset_acc_metrics()
            fault = _run_examples(runner, examples)

        acc_m = inj.get_acc_metrics()
        row = {
            "layer": layer,
            "site_id": site,
            "op_type": op_type,
            "acc_baseline": float(baseline["summary"]["accuracy"]),
            "acc_fault": float(fault["summary"]["accuracy"]),
            "runs": int(acc_m.get("runs", 0)),
            "tp": int(acc_m.get("tp", 0)),
            "fp": int(acc_m.get("fp", 0)),
            "fn": int(acc_m.get("fn", 0)),
            "normal": int(acc_m.get("normal", 0)),
            "thr_gamma": float(args.gamma),
            "acc_thr_enabled": int(acc_thr_enabled),
            "acc_thr_action": acc_thr_action,
            "fault_mode": str(args.fault_mode),
            "fault_delta": float(args.fault_delta),
            "site_set": site_set,
        }
        rows.append(row)

        site_slug = _slugify_site(site)
        (paths["json"] / f"site_metrics_{site_slug}.json").write_text(
            json.dumps(
                {
                    "site_id": site,
                    "layer": layer,
                    "op_type": op_type,
                    "summary": fault["summary"],
                    "acc_metrics": acc_m,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        for p in fault["per_example"]:
            detail_rows.append(
                {
                    "site_id": site,
                    "idx": int(p["idx"]),
                    "ind": p["ind"],
                    "label": int(p["label"]),
                    "pred": int(p["pred"]),
                    "correct": int(p["correct"]),
                }
            )
        print(
            f"[{cur}/{total_groups}] site={site} layer={layer} op={op_type} "
            f"acc={row['acc_fault']:.6f} runs={row['runs']} tp={row['tp']} fp={row['fp']} fn={row['fn']}",
            flush=True,
        )

    with open(paths["csv"] / sweep_csv, "w", encoding="utf-8", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    with open(paths["csv"] / "sweep_per_example_min.csv", "w", encoding="utf-8", newline="") as f:
        if detail_rows:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader()
            w.writerows(detail_rows)

    summary_json = paths["json"] / (
        "sweep_summary_nonmatmul.json" if site_set == "nonmatmul" else "sweep_summary.json"
    )
    payload = {
        "baseline": baseline_payload,
        "fault_groups": rows,
        "site_set": site_set,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "site_set": site_set,
                "sweep_csv": sweep_csv,
                "fault_group_count": len(rows),
                "expected_fault_group_count": ops_per_layer * len(layers),
                "baseline_acc": baseline["summary"]["accuracy"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
