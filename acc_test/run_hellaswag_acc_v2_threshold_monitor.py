"""HellaSwag ACC v2: no fault injection, threshold check only, per-operator TP/FP/FN/normal.

Under clean tensors there is no true positive (``tp`` should stay 0); ``fp`` counts
threshold crossings (mask hits) with golden restore, ``normal`` is no mask.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from datasets import load_dataset

# Allow ``python acc_test/run_...py`` from repo root.
_ACC_ROOT = Path(__file__).resolve().parent
if str(_ACC_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACC_ROOT))

from inject import SITE_STRATEGY_QWEN, InjectionContext, list_sites
from model_runner import ModelRunner
from results_layout import default_results_root, ensure_results_subdirs, results_run_dir, write_run_meta

from run_hellaswag_acc_v2_sweep import (  # noqa: E402
    _run_examples,
    _warmup_examples,
    op_type_from_site,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument(
        "--layer-list",
        default="0,8,16,24",
        help="Comma-separated layer indices (same default as sweep: 4×8=32 sites). Empty = all layers.",
    )
    ap.add_argument(
        "--results-root",
        default=None,
        help="default: acc_test/results (see results_layout.default_results_root)",
    )
    return ap.parse_args()


def _parse_layer_list(s: str) -> Optional[List[int]]:
    t = str(s).strip()
    if not t:
        return None
    out = [int(x.strip()) for x in t.split(",") if x.strip()]
    return out or None


def _layer_from_site(site_id: str) -> int:
    # L12_q_proj -> 12
    return int(site_id.split("_", 1)[0][1:])


def _site_in_layers(site_id: str, layers: Optional[Set[int]]) -> bool:
    if layers is None:
        return True
    return _layer_from_site(site_id) in layers


def main() -> None:
    args = parse_args()
    layers_arg = _parse_layer_list(args.layer_list)
    layer_filter: Optional[Set[int]] = set(layers_arg) if layers_arg is not None else None

    results_root = Path(args.results_root) if args.results_root else default_results_root()
    run_dir = results_run_dir(
        results_root,
        model_id=args.model_id,
        dataset="hellaswag",
        n_total=int(args.max_samples),
        n_warmup=int(args.n_warmup),
        gamma=float(args.gamma),
        fault_mode="none",
        seed=int(args.seed),
        max_new_tokens=None,
        fault_delta=None,
        acc_thr_enabled=True,
    )
    paths = ensure_results_subdirs(run_dir)

    meta: Dict[str, Any] = {
        "experiment": "threshold_monitor_no_fault",
        "model_id": args.model_id,
        "dataset": "hellaswag",
        "split": args.split,
        "max_samples": int(args.max_samples),
        "n_warmup": int(args.n_warmup),
        "gamma": float(args.gamma),
        "fault_mode": "none",
        "acc_thr_enabled": True,
        "acc_v2_inject_enable": False,
        "acc_v2_metrics_scope": "all",
        "seed": int(args.seed),
        "layer_list_filter": layers_arg,
        "notes": "No injection; per-site ACC v2 threshold + golden restore; tp should be 0 on clean data.",
    }
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

    all_site_ids = list_sites(runner.model, strategy=SITE_STRATEGY_QWEN)
    inj = InjectionContext(
        model=runner.model,
        target_site=None,
        fault_delta=0.0,
        seed=int(args.seed),
        site_strategy=SITE_STRATEGY_QWEN,
        protect_only=False,
        acc_v2=True,
        thr_gamma=float(args.gamma),
        acc_v2_threshold_enable=True,
        acc_v2_inject_enable=False,
        acc_v2_metrics_scope="all",
        fault_mode="none",
    )
    with inj:
        inj.reset_site_bounds()
        inj.set_warmup(True)
        _warmup_examples(runner, examples, int(args.n_warmup))
        inj.set_warmup(False)
        inj.reset_acc_metrics()
        scored = _run_examples(runner, examples)

    by_site = inj.get_acc_v2_metrics_by_site()
    if layer_filter is not None:
        by_site_export = {k: v for k, v in by_site.items() if _site_in_layers(k, layer_filter)}
    else:
        by_site_export = dict(by_site)
    # Rows for export (same keys as JSON ``by_site`` when filter is set).
    rows: List[Dict[str, Any]] = []
    for site_id in sorted(by_site_export.keys()):
        m = by_site_export[site_id]
        runs = int(m.get("runs", 0))
        tp = int(m.get("tp", 0))
        fp = int(m.get("fp", 0))
        fn = int(m.get("fn", 0))
        normal = int(m.get("normal", 0))
        fpr = float(fp) / float(runs) if runs else 0.0
        rows.append(
            {
                "layer": _layer_from_site(site_id),
                "site_id": site_id,
                "op_type": op_type_from_site(site_id),
                "runs": runs,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "normal": normal,
                "fpr": fpr,
                "thr_gamma": float(args.gamma),
            }
        )

    payload = {
        "summary": scored["summary"],
        "by_site": by_site_export,
        "layer_list_filter": layers_arg,
        "expected_sites_qwen": all_site_ids,
    }
    (paths["json"] / "site_metrics_by_site.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(paths["csv"] / "threshold_monitor_by_site.csv", "w", encoding="utf-8", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    (paths["json"] / "hellaswag_scored_summary.json").write_text(
        json.dumps(scored["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_tp = sum(int(by_site_export[s].get("tp", 0)) for s in by_site_export)
    total_fp = sum(int(by_site_export[s].get("fp", 0)) for s in by_site_export)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "accuracy": scored["summary"]["accuracy"],
                "sites_with_metrics": len(by_site_export),
                "total_tp": total_tp,
                "total_fp": total_fp,
                "csv_rows": len(rows),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
