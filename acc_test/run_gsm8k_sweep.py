from __future__ import annotations

# Print before heavy imports so user sees immediate output.
print("[import] run_gsm8k_sweep.py importing...", flush=True)

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Tuple

from results_layout import default_results_root


def parse_args():
    rr = default_results_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--indices-json", default=str(rr / "gsm8k_test_shared100_indices.json"))
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--fault-delta", type=float, default=10000.0)
    ap.add_argument(
        "--fault-index-mode",
        default="random",
        choices=["random", "max_abs"],
        help="Which element index to corrupt within the target tensor.",
    )
    ap.add_argument("--clear-exceptions", action="store_true", default=False)
    ap.add_argument("--clear-threshold-mul", type=float, default=0.5)
    ap.add_argument("--decode-step-max", type=int, default=150)
    ap.add_argument("--layer-list", default="0,7,14,21,27")
    ap.add_argument("--scope", default="all", choices=["attn", "mlp", "all"])
    ap.add_argument(
        "--site-list",
        default="",
        help="Optional: comma-separated explicit site_ids (e.g. L14_v_proj). Overrides --layer-list/--scope.",
    )
    ap.add_argument("--out-csv", default=str(rr / "gsm8k_sweep_shared100.csv"))
    ap.add_argument("--out-json", default=str(rr / "gsm8k_sweep_shared100.json"))
    ap.add_argument("--out-baseline-json", default=str(rr / "gsm8k_baseline_shared100.json"))
    ap.add_argument(
        "--baseline-only",
        action="store_true",
        default=False,
        help="Run baseline only and exit before sweep.",
    )
    ap.add_argument(
        "--baseline-json-in",
        default=None,
        help="Optional: reuse an existing baseline JSON and skip running baseline.",
    )
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help="Skip baseline run (requires --baseline-json-in).",
    )
    ap.add_argument(
        "--trace-jsonl",
        default=None,
        help="Optional: write per-problem JSONL trace for this run (appended).",
    )
    ap.add_argument(
        "--trace-dir",
        default=None,
        help="Optional: directory to write per-site JSONL trace files (recommended).",
    )
    ap.add_argument(
        "--trace-run-tag",
        default="",
        help="Optional: attach a short tag to each JSONL row (e.g. baseline/fault/clear).",
    )
    return ap.parse_args()


def layer_sites(layer_idx: int, scope: str) -> List[str]:
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
    suffix = site_id.split("_", 1)[1]
    if suffix in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return suffix
    if suffix == "attn_core":
        return "attn_core(qk^t+s*v)"
    if suffix in {"mlp_gate", "mlp_up", "mlp_down"}:
        return suffix
    return suffix


def main():
    # Delay heavy imports (torch/transformers/datasets) until after we print something.
    from gsm8k_task import Gsm8kTask
    from model_runner import ModelRunner

    args = parse_args()
    print("[main] parsed args, loading indices json...", flush=True)
    progress_log_path = args.out_csv + ".progress.log"
    idx_payload = json.load(open(args.indices_json, "r", encoding="utf-8"))
    indices = idx_payload["indices"][: int(args.max_samples)]
    print(f"[main] loaded indices n={len(indices)}; building runner...", flush=True)

    sites: List[Tuple[int, str]] = []
    if args.site_list.strip():
        for raw in args.site_list.split(","):
            site = raw.strip()
            if not site:
                continue
            m = re.match(r"^L(\d+)_", site)
            if not m:
                raise ValueError(f"bad site_id={site!r} (expected like L14_v_proj)")
            li = int(m.group(1))
            sites.append((li, site))
    else:
        layers = [int(x.strip()) for x in args.layer_list.split(",") if x.strip()]
        for li in layers:
            for s in layer_sites(li, scope=args.scope):
                sites.append((li, s))

    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    print("[main] runner ready.", flush=True)

    task = Gsm8kTask(
        split=args.split,
        max_samples=int(args.max_samples),
        seed=int(args.seed),
        max_new_tokens=int(args.max_new_tokens),
        require_hash_answer=True,
        indices=[int(i) for i in indices],
        raw_generation_char_limit=0 if args.trace_jsonl else 2000,
        trace_jsonl_path=args.trace_jsonl,
        trace_run_tag=args.trace_run_tag,
    )

    def _trace_path_for(tag: str, site_id: str | None) -> str | None:
        if not args.trace_dir:
            return args.trace_jsonl
        os.makedirs(args.trace_dir, exist_ok=True)
        n = int(args.max_samples)
        seed = int(args.seed)
        delta = int(args.fault_delta) if float(args.fault_delta).is_integer() else args.fault_delta
        stepmax = int(args.decode_step_max)
        fim = str(args.fault_index_mode)
        clear_tag = "clearhalf" if bool(args.clear_exceptions) else "noclear"
        if site_id is None:
            name = f"baseline__n{n}_seed{seed}.jsonl"
        else:
            name = f"{site_id}__{fim}__{clear_tag}__delta{delta}__stepmax{stepmax}__n{n}_seed{seed}.jsonl"
        if tag:
            name = f"{tag}__{name}"
        return os.path.join(args.trace_dir, name)

    def _prepare_trace(path: str | None) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Overwrite semantics: avoid accidental mixing across runs.
        if os.path.exists(path):
            os.remove(path)

    baseline_acc: float
    baseline_summary: Dict[str, Any]
    if args.skip_baseline:
        if not args.baseline_json_in:
            raise ValueError("--skip-baseline requires --baseline-json-in")
        baseline = json.load(open(args.baseline_json_in, "r", encoding="utf-8"))
        baseline_summary = baseline["summary"]
        baseline_acc = float(baseline_summary["accuracy"])
        baseline_line = f"[baseline:reuse] acc={baseline_acc:.4f} total={baseline_summary['total']}"
        print(baseline_line, flush=True)
        with open(progress_log_path, "a", encoding="utf-8") as f:
            f.write(baseline_line + "\n")
            f.flush()
    else:
        print("[main] starting baseline...", flush=True)
        baseline_trace = _trace_path_for(args.trace_run_tag or "baseline", None)
        _prepare_trace(baseline_trace)
        task.trace_jsonl_path = baseline_trace
        task.trace_run_tag = args.trace_run_tag or "baseline"
        baseline = runner.run_task(task, inject_site=None, seed=args.seed)
        with open(args.out_baseline_json, "w", encoding="utf-8") as f:
            json.dump(baseline, f, ensure_ascii=False, indent=2)
        baseline_summary = baseline["summary"]
        baseline_acc = float(baseline_summary["accuracy"])
        baseline_line = f"[baseline] acc={baseline_acc:.4f} total={baseline_summary['total']}"
        print(baseline_line, flush=True)
        with open(progress_log_path, "a", encoding="utf-8") as f:
            f.write(baseline_line + "\n")
            f.flush()

    if args.baseline_only:
        print("[baseline-only] done", flush=True)
        return

    rows: List[Dict[str, Any]] = []
    total_sites = len(sites)
    for cur, (li, site) in enumerate(sites, start=1):
        # Per-site trace file (overwrite each run).
        site_trace = _trace_path_for(args.trace_run_tag, site)
        _prepare_trace(site_trace)
        task.trace_jsonl_path = site_trace
        task.trace_run_tag = args.trace_run_tag

        fault = runner.run_task(
            task,
            inject_site=site,
            fault_delta=args.fault_delta,
            seed=args.seed,
            fault_index_mode=args.fault_index_mode,
            clear_exceptions=args.clear_exceptions,
            clear_threshold_mul=args.clear_threshold_mul,
            decode_step_inject_enable=True,
            decode_step_max=args.decode_step_max,
        )
        summ = fault["summary"]
        total = int(summ["total"])
        format_fail = int(summ.get("pred_format_fail", 0))
        format_fail_rate = float(format_fail) / float(total) if total else 0.0
        rm = fault.get("run_meta", {})
        dp = int(rm.get("decode_problem_count", 0))
        di = int(rm.get("decode_injected_problem_count", 0))
        hit_rate = float(di) / float(dp) if dp else 0.0

        row = {
            "layer": li,
            "site_id": site,
            "op_type": op_type_from_site(site),
            "acc_baseline": baseline_acc,
            "acc_fault": float(summ["accuracy"]),
            "format_fail_rate": format_fail_rate,
            "hit_rate": hit_rate,
        }
        rows.append(row)
        line = (
            f"[{cur}/{total_sites}] layer={li} site={site} op_type={row['op_type']} "
            f"acc={row['acc_fault']:.4f} format_fail={format_fail_rate:.3f} hit_rate={hit_rate:.3f}"
        )
        print(line, flush=True)
        with open(progress_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"baseline": baseline_summary, "rows": rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("[entry] run_gsm8k_sweep.py started", flush=True)
    main()

