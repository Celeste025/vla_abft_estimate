#!/usr/bin/env python3
"""Batch HellaSwag ACC v2 sweeps over fault specs × threshold on/off.

Each run is ``run_hellaswag_acc_v2_sweep.py`` (baseline + 8×layers single-site groups).

Fault specs (``--fault-specs``)::

    rand2pow
    fixed:<delta>     e.g. fixed:1  fixed:10  (random flat index + that delta)

``--threshold-modes``::

    on   — only thr-mMg (threshold + golden restore)
    off  — only thr-none (inject only; ``--acc-no-threshold``)
    both — run on then off for each fault spec (2× jobs per spec)

Example::

    python run_hellaswag_acc_v2_fault_matrix_suite.py \\
        --threshold-modes off \\
        --fault-specs rand2pow,fixed:1,fixed:10,fixed:100,fixed:1000,fixed:10000 \\
        --max-samples 200 --n-warmup 10 --gamma 3.0 --seed 2026 --layer-list 0,8,16,24
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from results_layout import default_results_root, results_run_dir


def _parse_fault_specs(s: str) -> List[Tuple[str, Optional[float]]]:
    out: List[Tuple[str, Optional[float]]] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if part == "rand2pow":
            out.append(("rand2pow", None))
            continue
        if part.lower().startswith("fixed:"):
            v = float(part.split(":", 1)[1].strip())
            out.append(("fixed", v))
            continue
        raise ValueError(f"bad fault spec {part!r}; use rand2pow or fixed:<delta>")
    if not out:
        raise ValueError("fault-specs is empty")
    return out


def _sweep_argv(
    acc_test_dir: Path,
    forward: list[str],
    *,
    fault_mode: str,
    fault_delta: Optional[float],
    acc_thr_enabled: bool,
) -> list[str]:
    sweep = acc_test_dir / "run_hellaswag_acc_v2_sweep.py"
    tail: list[str] = ["--fault-mode", str(fault_mode)]
    if fault_mode == "fixed":
        if fault_delta is None:
            raise ValueError("internal: fixed requires fault_delta")
        tail += ["--fault-delta", str(fault_delta)]
    if not acc_thr_enabled:
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


def _parse_forward_defaults(forward: list[str]) -> tuple[str, int, int, float, int, Path]:
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
    return model_id, n_total, n_wu, gamma, seed, results_root


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HellaSwag ACC v2: matrix over fault specs and threshold on/off.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unknown flags are forwarded to run_hellaswag_acc_v2_sweep.py.",
    )
    ap.add_argument(
        "--fault-specs",
        default="rand2pow,fixed:1,fixed:10,fixed:100,fixed:1000,fixed:10000",
        help="Comma-separated: rand2pow | fixed:<delta>",
    )
    ap.add_argument(
        "--threshold-modes",
        default="both",
        choices=["both", "on", "off"],
        help="both: run thr-mMg then thr-none per spec; on/off: single mode.",
    )
    ap.add_argument("--skip-plots", action="store_true")
    args, forward = ap.parse_known_args()
    if "--" in forward:
        forward = [x for x in forward if x != "--"]

    specs = _parse_fault_specs(args.fault_specs)
    if args.threshold_modes == "both":
        thr_seq = [True, False]
    elif args.threshold_modes == "on":
        thr_seq = [True]
    else:
        thr_seq = [False]

    acc_test_dir = Path(__file__).resolve().parent
    model_id, n_total, n_wu, gamma, seed, results_root = _parse_forward_defaults(forward)

    for thr_on in thr_seq:
        for fm, fd in specs:
            thr_lab = "thr-mMg" if thr_on else "thr-none"
            if fm == "rand2pow":
                spec_lab = "rand2pow"
            else:
                fd_lab = str(int(fd)) if float(fd).is_integer() else str(fd)
                spec_lab = f"fixed+{fd_lab}"
            print(f"\n{'='*72}\n[suite] {thr_lab}  {spec_lab}\n{'='*72}\n", flush=True)
            cmd = _sweep_argv(
                acc_test_dir,
                forward,
                fault_mode=fm,
                fault_delta=fd,
                acc_thr_enabled=thr_on,
            )
            print("[suite] " + " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(acc_test_dir), check=True)

            run_dir = results_run_dir(
                results_root,
                model_id=model_id,
                dataset="hellaswag",
                n_total=n_total,
                n_warmup=n_wu,
                gamma=gamma,
                fault_mode=fm,
                seed=seed,
                max_new_tokens=None,
                fault_delta=float(fd) if fm == "fixed" else None,
                acc_thr_enabled=thr_on,
            )
            title = (
                f"HellaSwag ACC v2 {thr_lab} {spec_lab} "
                f"(n={n_total}, wu={n_wu}, gamma={gamma}, seed={seed})"
            )
            if not args.skip_plots:
                pcmd = _plot_argv(acc_test_dir, run_dir, title=title)
                print("[suite] " + " ".join(pcmd), flush=True)
                subprocess.run(pcmd, cwd=str(acc_test_dir), check=True)
            print(f"[suite] done run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
