from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Type

import torch

from imagenet_task import HF_IMAGENET_DATASET_ID, ImagenetTask
from inject import SITE_STRATEGY_MODULE_SCAN, list_sites
from vision_runner import VisionRunner


DEFAULT_SITES = (
    "conv1",
    "layer1.0.conv2",
    "layer2.0.conv2",
    "layer3.0.conv2",
    "layer4.0.conv2",
    "fc",
)


def parse_module_types(s: str) -> Tuple[Type[torch.nn.Module], ...]:
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    mapping: Dict[str, Type[torch.nn.Module]] = {
        "conv2d": torch.nn.Conv2d,
        "linear": torch.nn.Linear,
    }
    out: List[Type[torch.nn.Module]] = []
    for p in parts:
        if p not in mapping:
            raise ValueError(f"unknown module type {p!r}; choose from {list(mapping)}")
        out.append(mapping[p])
    return tuple(out)


def op_type_from_site(site_id: str) -> str:
    if site_id == "fc" or site_id.endswith(".fc") or "fc" == site_id.split(".")[-1]:
        return "linear"
    return "conv2d"


def pred_mismatch_rate(baseline: Dict, other: Dict) -> float:
    bmap = {x["index"]: int(x["pred1"]) for x in baseline["per_example"]}
    n = 0
    diff = 0
    for ex in other["per_example"]:
        i = int(ex["index"])
        if i not in bmap:
            continue
        n += 1
        if int(ex["pred1"]) != bmap[i]:
            diff += 1
    return float(diff) / float(n) if n else 0.0


def parse_args():
    ap = argparse.ArgumentParser(description="Sweep ResNet Conv/Linear inject sites on ImageNet subset.")
    ap.add_argument("--weights", default="IMAGENET1K_V2", choices=["IMAGENET1K_V1", "IMAGENET1K_V2"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fault-delta", type=float, default=10000.0)
    ap.add_argument("--fault-index-mode", default="random", choices=["random", "max_abs"])
    ap.add_argument("--clear-threshold-mul", type=float, default=0.5)
    ap.add_argument(
        "--site-list",
        default=",".join(DEFAULT_SITES),
        help="Comma-separated site ids (named_modules paths).",
    )
    ap.add_argument(
        "--site-list-file",
        default=None,
        help="If set, read comma-separated site ids from this file (overrides --site-list).",
    )
    ap.add_argument(
        "--all-sites",
        action="store_true",
        help="Sweep every Conv2d/Linear site (slow).",
    )
    ap.add_argument("--module-types", default="conv2d,linear", help="For --all-sites and hook registration.")
    ap.add_argument("--out-csv", default="resnet_imagenet_sweep.csv")
    ap.add_argument("--out-json", default="resnet_imagenet_sweep.json")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random tensors instead of HF imagenet-1k.",
    )
    ap.add_argument("--hf-dataset-id", default=HF_IMAGENET_DATASET_ID, help="HF datasets id, default ILSVRC/imagenet-1k.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    ap.add_argument(
        "--local-dataset-dir",
        type=str,
        default=None,
        help="Local validation from download_imagenet_val.py (no Hub streaming).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    module_classes = parse_module_types(args.module_types)
    runner = VisionRunner(weights=args.weights, device=args.device, dtype=args.dtype)
    task = ImagenetTask(
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        streaming=True,
        weights_name=args.weights,
        synthetic=args.synthetic,
        hf_dataset_id=args.hf_dataset_id,
        show_progress=not args.no_progress,
        local_dataset_dir=args.local_dataset_dir,
    )

    print(
        f"[sweep] baseline run: max_samples={args.max_samples} batch_size={args.batch_size} "
        f"device={args.device} inject=off",
        flush=True,
    )
    baseline = runner.run_task(
        task,
        inject_site=None,
        target_module_classes=module_classes,
    )
    print(
        f"[sweep] baseline done: top1={baseline['summary']['top1_accuracy']:.4f} "
        f"top5={baseline['summary']['top5_accuracy']:.4f}",
        flush=True,
    )

    if args.all_sites:
        sites = list_sites(
            runner.model,
            strategy=SITE_STRATEGY_MODULE_SCAN,
            module_classes=module_classes,
        )
    elif args.site_list_file:
        p = Path(args.site_list_file).expanduser().resolve()
        raw = p.read_text(encoding="utf-8").strip()
        sites = [s.strip() for s in raw.replace("\n", ",").split(",") if s.strip()]
    else:
        sites = [s.strip() for s in args.site_list.split(",") if s.strip()]

    available = set(
        list_sites(
            runner.model,
            strategy=SITE_STRATEGY_MODULE_SCAN,
            module_classes=module_classes,
        )
    )
    sites = [s for s in sites if s in available]
    if not sites:
        raise SystemExit("No valid sites after filtering; check --site-list or model.")

    print(f"[sweep] {len(sites)} site(s): {sites[0]} … {sites[-1]}", flush=True)
    rows = []
    for site in sites:
        fault = runner.run_task(
            task,
            inject_site=site,
            fault_delta=args.fault_delta,
            seed=args.seed,
            fault_index_mode=args.fault_index_mode,
            clear_exceptions=False,
            target_module_classes=module_classes,
        )
        protect = runner.run_task(
            task,
            inject_site=site,
            fault_delta=args.fault_delta,
            seed=args.seed,
            fault_index_mode=args.fault_index_mode,
            clear_exceptions=True,
            clear_threshold_mul=args.clear_threshold_mul,
            target_module_classes=module_classes,
        )
        b1 = baseline["summary"]["top1_accuracy"]
        bf = fault["summary"]["top1_accuracy"]
        bp = protect["summary"]["top1_accuracy"]
        rows.append(
            {
                "site_id": site,
                "op_type": op_type_from_site(site),
                "acc_baseline_top1": b1,
                "acc_fault_top1": bf,
                "acc_protect_top1": bp,
                "acc_baseline_top5": baseline["summary"]["top5_accuracy"],
                "acc_fault_top5": fault["summary"]["top5_accuracy"],
                "acc_protect_top5": protect["summary"]["top5_accuracy"],
                "top1_drop_fault": b1 - bf,
                "top1_recover_protect_vs_fault": bf - bp,
                "pred_mismatch_rate_fault": pred_mismatch_rate(baseline, fault),
                "inject_count_fault": fault["run_meta"]["inject_count"],
                "inject_count_protect": protect["run_meta"]["inject_count"],
                "errors_total_protect": protect["run_meta"]["errors_total"],
            }
        )
        print(
            f"site={site} top1: baseline={b1:.4f} fault={bf:.4f} protect={bp:.4f} "
            f"errors_total={protect['run_meta']['errors_total']}",
            flush=True,
        )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"baseline_summary": baseline["summary"], "rows": rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
