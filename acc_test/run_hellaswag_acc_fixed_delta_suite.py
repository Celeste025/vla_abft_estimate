#!/usr/bin/env python3
"""Run several HellaSwag ACC sweeps: fault_mode=fixed, random flat index + fault_delta.

Each sweep matches ``run_hellaswag_acc_sweep.py`` (baseline + 8*len(layers) single-site groups).
Run directories include ``fm-fixed_fd{delta}`` via ``results_layout.build_run_config_segment``.

Example::

    python run_hellaswag_acc_fixed_delta_suite.py \\
        --fault-deltas 1,10,100,1000,10000 \\
        --max-samples 200 --n-warmup 10 --gamma 3.0 --seed 2026

Or explicitly separate suite vs sweep args::

    python run_hellaswag_acc_fixed_delta_suite.py --fault-deltas 1,100 -- \\
        --max-samples 200 --layer-list 0,8,16,24

``--fault-mode`` / ``--fault-delta`` are always forced by this suite (fixed + chosen delta).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from results_layout import default_results_root, results_run_dir


def _parse_deltas(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("fault-deltas is empty")
    return out


def _sweep_argv(acc_test_dir: Path, forward: list[str], fault_delta: float, *, acc_no_threshold: bool) -> list[str]:
    sweep = acc_test_dir / "run_hellaswag_acc_sweep.py"
    tail: list[str] = [
        "--fault-mode",
        "fixed",
        "--fault-delta",
        str(fault_delta),
    ]
    if acc_no_threshold:
        tail.append("--acc-no-threshold")
    return [sys.executable, str(sweep), *forward, *tail]


def _plot_argv(acc_test_dir: Path, run_dir: Path, title: str) -> list[str]:
    plot = acc_test_dir / "plot_sweep_summary.py"
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "csv" / "sweep_summary.csv"
    return [
        sys.executable,
        str(plot),
        "--in-csv",
        str(csv_path),
        "--out-png-acc",
        str(plots / "sweep_acc_fault_by_layer_op.png"),
        "--title",
        title,
        "--out-png-tp-rate",
        str(plots / "sweep_tp_rate_by_layer_op.png"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch HellaSwag ACC sweeps: fixed additive fault at random index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Any other flags (e.g. --max-samples, --layer-list, --model-id) are forwarded to "
        "run_hellaswag_acc_sweep.py. Do not pass --fault-mode/--fault-delta; this script forces "
        "fixed mode and each delta.",
    )
    ap.add_argument(
        "--fault-deltas",
        default="1,10,100,1000,10000",
        help="Comma-separated fault_delta values (fixed mode: +delta at one random element).",
    )
    ap.add_argument("--skip-plots", action="store_true", help="Only run sweeps, do not call plot_sweep_summary.py.")
    ap.add_argument(
        "--acc-no-threshold",
        action="store_true",
        help="Forward to sweep: inject only (thr-none run dirs, detection metrics stay zero).",
    )
    args, forward = ap.parse_known_args()
    deltas = _parse_deltas(args.fault_deltas)
    if "--" in forward:
        forward = [x for x in forward if x != "--"]

    acc_test_dir = Path(__file__).resolve().parent
    # Defaults for title / run_dir preview (parse forwarded args lightly)
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    n_total, n_wu, gamma, seed = 200, 10, 3.0, 2026
    results_root = default_results_root()
    i = 0
    while i < len(forward):
        if forward[i] == "--model-id" and i + 1 < len(forward):
            model_id = forward[i + 1]
            i += 2
            continue
        if forward[i] == "--max-samples" and i + 1 < len(forward):
            n_total = int(forward[i + 1])
            i += 2
            continue
        if forward[i] == "--n-warmup" and i + 1 < len(forward):
            n_wu = int(forward[i + 1])
            i += 2
            continue
        if forward[i] == "--gamma" and i + 1 < len(forward):
            gamma = float(forward[i + 1])
            i += 2
            continue
        if forward[i] == "--seed" and i + 1 < len(forward):
            seed = int(forward[i + 1])
            i += 2
            continue
        if forward[i] == "--results-root" and i + 1 < len(forward):
            results_root = Path(forward[i + 1])
            i += 2
            continue
        i += 1

    acc_thr_enabled = not bool(args.acc_no_threshold)
    for fd in deltas:
        print(f"\n{'='*72}\n[suite] fixed fault_delta={fd} acc_thr={acc_thr_enabled}\n{'='*72}\n", flush=True)
        cmd = _sweep_argv(acc_test_dir, forward, fault_delta=fd, acc_no_threshold=bool(args.acc_no_threshold))
        print("[suite] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(acc_test_dir), check=True)

        run_dir = results_run_dir(
            results_root,
            model_id=model_id,
            dataset="hellaswag",
            n_total=n_total,
            n_warmup=n_wu,
            gamma=gamma,
            fault_mode="fixed",
            seed=seed,
            max_new_tokens=None,
            fault_delta=fd,
            acc_thr_enabled=acc_thr_enabled,
        )
        fd_lab = str(int(fd)) if float(fd).is_integer() else str(fd)
        thr_lab = "thr-mMg" if acc_thr_enabled else "thr-none"
        title = (
            f"HellaSwag ACC {thr_lab} fixed +{fd_lab} at random index "
            f"(n={n_total}, wu={n_wu}, gamma={gamma}, seed={seed})"
        )

        if not args.skip_plots:
            pcmd = _plot_argv(acc_test_dir, run_dir, title=title)
            print("[suite] " + " ".join(pcmd), flush=True)
            subprocess.run(pcmd, cwd=str(acc_test_dir), check=True)
        print(f"[suite] done fault_delta={fd} run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
