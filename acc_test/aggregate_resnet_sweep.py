#!/usr/bin/env python3
"""Merge shard0..shard5 CSV from parallel ResNet sweep; sort rows by canonical site order."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from inject import SITE_STRATEGY_MODULE_SCAN, list_sites


def canonical_site_order() -> List[str]:
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    return list_sites(
        m,
        strategy=SITE_STRATEGY_MODULE_SCAN,
        module_classes=(nn.Conv2d, nn.Linear),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Directory containing shard0.csv .. shard5.csv (and json).",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output CSV path (default: <result-dir>/master.csv).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rd = Path(args.result_dir).expanduser().resolve()
    if not rd.is_dir():
        raise SystemExit(f"result-dir not a directory: {rd}")

    order = canonical_site_order()
    rank = {s: i for i, s in enumerate(order)}

    rows: List[Dict[str, Any]] = []
    baselines_top1: List[float] = []
    baselines_top5: List[float] = []

    for i in range(6):
        csv_path = rd / f"shard{i}.csv"
        json_path = rd / f"shard{i}.json"
        if not csv_path.is_file():
            raise SystemExit(f"missing {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        if json_path.is_file():
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            b1 = float(meta["baseline_summary"]["top1_accuracy"])
            b5 = float(meta["baseline_summary"]["top5_accuracy"])
            baselines_top1.append(b1)
            baselines_top5.append(b5)

    if len(rows) != len(order):
        raise SystemExit(f"expected {len(order)} rows total, got {len(rows)}")

    missing = [s for s in order if s not in {r["site_id"] for r in rows}]
    if missing:
        raise SystemExit(f"missing site rows: {missing[:5]}...")

    if baselines_top1:
        if max(baselines_top1) - min(baselines_top1) > 1e-6:
            print(
                "WARNING: baseline top1 differs across shards:",
                baselines_top1,
            )
        if max(baselines_top5) - min(baselines_top5) > 1e-6:
            print(
                "WARNING: baseline top5 differs across shards:",
                baselines_top5,
            )

    rows.sort(key=lambda r: rank[str(r["site_id"])])

    out_csv = Path(args.out_csv) if args.out_csv else rd / "master.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "result_dir": str(rd),
        "n_rows": len(rows),
        "n_sites_expected": len(order),
        "baseline_top1_range": [min(baselines_top1), max(baselines_top1)] if baselines_top1 else None,
        "baseline_top5_range": [min(baselines_top5), max(baselines_top5)] if baselines_top5 else None,
        "out_csv": str(out_csv),
    }
    (rd / "master_meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
