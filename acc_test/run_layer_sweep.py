from __future__ import annotations

import argparse
import csv
import json

from hellaswag_task import HellaSwagTask
from model_runner import ModelRunner


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fault-delta", type=float, default=10000.0)
    ap.add_argument(
        "--fault-index-mode",
        default="random",
        choices=["random", "max_abs"],
        help="Which element index to corrupt within the target tensor.",
    )
    ap.add_argument("--clear-exceptions", action="store_true", default=False)
    ap.add_argument("--clear-threshold-mul", type=float, default=0.5)
    ap.add_argument("--layer-list", default="0,8,16,24")
    ap.add_argument(
        "--scope",
        default="attn",
        choices=["attn", "mlp", "all"],
        help="Which sites to sweep per layer.",
    )
    ap.add_argument("--out-csv", default="sweep.csv")
    ap.add_argument("--out-json", default="sweep.json")
    return ap.parse_args()

def layer_sites(layer_idx: int, scope: str):
    attn_sites = [
        f"L{layer_idx}_q_proj",
        f"L{layer_idx}_k_proj",
        f"L{layer_idx}_v_proj",
        f"L{layer_idx}_attn_core",
        f"L{layer_idx}_o_proj",
    ]
    mlp_sites = [
        f"L{layer_idx}_mlp_gate",
        f"L{layer_idx}_mlp_up",
        f"L{layer_idx}_mlp_down",
    ]
    if scope == "attn":
        return attn_sites
    if scope == "mlp":
        return mlp_sites
    return attn_sites + mlp_sites


def op_type_from_site(site_id: str) -> str:
    # site_id: L{layer}_{suffix}
    suffix = site_id.split("_", 1)[1]
    if suffix in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return suffix
    if suffix == "attn_core":
        return "attn_core(qk^t+s*v)"
    if suffix in {"mlp_gate", "mlp_up", "mlp_down"}:
        return suffix
    return suffix


def mean_abs_delta_score(baseline_map, fault_per_example) -> float:
    # baseline_map: ind -> per_example item
    # fault_per_example: list of per_example items
    deltas = []
    for fx in fault_per_example:
        bx = baseline_map[fx["ind"]]
        abs_d = [abs(sf - sb) for sb, sf in zip(bx["scores"], fx["scores"])]
        deltas.append(sum(abs_d) / float(len(abs_d)))
    return float(sum(deltas) / float(len(deltas))) if deltas else 0.0


def main():
    args = parse_args()
    layers = [int(x.strip()) for x in args.layer_list.split(",") if x.strip()]
    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    task = HellaSwagTask(split=args.split, max_samples=args.max_samples, seed=args.seed)
    baseline = runner.run_task(
        task,
        inject_site=None,
        seed=args.seed,
        clear_exceptions=args.clear_exceptions,
        clear_threshold_mul=args.clear_threshold_mul,
    )
    baseline_acc = baseline["summary"]["accuracy"]
    base_map = {x["ind"]: x for x in baseline["per_example"]}

    scope_sites = []
    for li in layers:
        for site in layer_sites(li, scope=args.scope):
            scope_sites.append((li, site))
    total_sites = len(scope_sites)

    rows = []
    for cur_idx, (li, site) in enumerate(scope_sites, start=1):
        op_type = op_type_from_site(site)
        fault = runner.run_task(
            task,
            inject_site=site,
            fault_delta=args.fault_delta,
            seed=args.seed,
            fault_index_mode=args.fault_index_mode,
            clear_exceptions=args.clear_exceptions,
            clear_threshold_mul=args.clear_threshold_mul,
        )
        mad = mean_abs_delta_score(base_map, fault["per_example"])
        clear_meta = fault.get("run_meta", {})
        errors_total = clear_meta.get("errors_total", 0)
        bad_forward_count = clear_meta.get("bad_forward_count", 0)
        rows.append(
            {
                "layer": li,
                "site_id": site,
                "op_type": op_type,
                "acc_baseline": baseline["summary"]["accuracy"],
                "acc_fault": fault["summary"]["accuracy"],
                "mean_abs_delta_score": mad,
                "inject_count": fault["run_meta"]["inject_count"],
                "registered_site_count": fault["run_meta"]["registered_site_count"],
                "expected_site_count": fault["run_meta"]["expected_site_count"],
            }
        )
        print(
            f"[{cur_idx}/{total_sites}] layer={li} site={site} op_type={op_type} "
            f"acc_fault={rows[-1]['acc_fault']:.6f} mean_abs_delta={mad:.3f} "
            f"errors_total={errors_total} bad_forward={bad_forward_count}",
            flush=True,
        )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    payload = {"baseline": baseline["summary"], "rows": rows}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
