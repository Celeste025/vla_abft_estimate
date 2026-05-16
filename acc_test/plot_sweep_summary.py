#!/usr/bin/env python3
"""Plot ``sweep_summary.csv`` style tables: layer × op_type curves and optional ACC v2 charts.

Input CSV is expected to have at least: ``layer``, ``op_type``, ``acc_baseline``, ``acc_fault``.
Works for HellaSwag / GSM8K ACC v2 sweeps and legacy GSM8K layer-op sweep exports.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from results_layout import default_results_root


def _fpr_per_op_type(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Aggregate FP rate per op_type across layers.

    Per op_type: if FP+TN > 0 with TN = ``normal``, use classical FPR = FP / (FP + TN).
    Otherwise use FP / runs (typical when every audited step had injection).
    """
    g = (
        df.groupby("op_type", sort=False)[["fp", "normal", "runs"]]
        .sum()
        .reset_index()
        .sort_values("op_type")
    )
    fp = g["fp"].astype(float)
    nr = g["normal"].astype(float)
    rn = g["runs"].astype(float).replace(0, float("nan"))
    den_classical = fp + nr
    g["fpr"] = (fp / den_classical).where(den_classical > 0, fp / rn).fillna(0.0)
    note = (
        "FPR per op_type: fp/(fp+normal) if fp+normal>0; else fp/runs. "
        "Here fp counts evaluations where the thr mask flagged at least one "
        "non-injected element (spurious), per inject.py ACC v2 rules."
    )
    return g, note


def plot_fpr_by_op_type(df: pd.DataFrame, out_path: str, *, title_prefix: str) -> None:
    agg, note = _fpr_per_op_type(df)
    ops = agg["op_type"].tolist()
    ys = agg["fpr"].astype(float).tolist()
    runs = agg["runs"].astype(int).tolist()

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(ops))
    bars = ax.bar(x, ys, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ops, rotation=22, ha="right")
    ax.set_ylabel("False positive rate")
    ax.set_xlabel("op_type")
    ax.set_title(f"{title_prefix} — FPR by op_type (summed over layers)")
    ymax = max(0.05, max(ys, default=0.0) * 1.2 + 1e-6)
    ax.set_ylim(0.0, ymax)
    ax.grid(True, axis="y", alpha=0.25)

    ax.bar_label(bars, labels=[f"{v:.4f}" for v in ys], fontsize=8, padding=4)

    lines = ["Total runs per op_type (sum over selected layers):"]
    for o, r in zip(ops, runs):
        lines.append(f"  {o}: {r}")
    lines.append("")
    lines.append(note)
    txt = "\n".join(lines)
    fig.text(
        0.99,
        0.01,
        txt,
        ha="right",
        va="bottom",
        fontsize=7,
        family="monospace",
        transform=fig.transFigure,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.85, edgecolor="gray"),
    )
    plt.tight_layout(rect=(0, 0.18, 1, 1))
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _is_threshold_monitor_df(df: pd.DataFrame) -> bool:
    return "fpr" in df.columns and "acc_fault" not in df.columns


def plot_fpr_by_layer_op(df: pd.DataFrame, out_path: str, *, title: str) -> None:
    """Threshold-monitor CSV: FPR vs layer, one curve per op_type."""
    layers = sorted(df["layer"].unique().tolist())
    op_types = sorted(df["op_type"].unique().tolist())
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    plt.figure(figsize=(10, 5))
    for i, op in enumerate(op_types):
        dfo = df[df["op_type"] == op].sort_values("layer")
        plt.plot(
            dfo["layer"].tolist(),
            dfo["fpr"].astype(float).tolist(),
            marker=markers[i % len(markers)],
            linewidth=1.8,
            label=op,
        )
    plt.xticks(layers, [str(x) for x in layers])
    plt.xlabel("layer")
    plt.ylabel("FPR (fp / runs)")
    plt.title(f"{title} — FPR vs layer (per op_type, no fault injection)")
    ymax = max(0.05, float(df["fpr"].max()) * 1.15 + 1e-6)
    plt.ylim(0.0, ymax)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_threshold_monitor(df: pd.DataFrame, args: argparse.Namespace) -> None:
    plots_dir = Path(args.out_png_acc).parent
    plots_dir.mkdir(parents=True, exist_ok=True)
    fpr_layer = args.out_png_acc
    if fpr_layer.endswith("sweep_acc_fault_by_layer_op.png"):
        fpr_layer = str(plots_dir / "threshold_fpr_by_layer_op.png")
    plot_fpr_by_layer_op(df, fpr_layer, title=args.title)
    print(fpr_layer)

    if "fp" in df.columns and "runs" in df.columns and "normal" in df.columns:
        fpr_out = args.out_png_fpr_by_op
        if fpr_out is None:
            fpr_out = str(plots_dir / "sweep_fpr_by_op.png")
        plot_fpr_by_op_type(df, fpr_out, title_prefix=args.title)
        print(fpr_out)


def load_sweep_dataframe(run_dir: Path | str) -> pd.DataFrame:
    """Merge matmul + nonmatmul sweep CSVs when both exist under run_dir/csv/."""
    rd = Path(run_dir)
    csv_dir = rd / "csv"
    parts: list[pd.DataFrame] = []
    for name in ("sweep_summary.csv", "sweep_summary_nonmatmul.csv"):
        p = csv_dir / name
        if p.is_file():
            parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError(f"no sweep_summary*.csv under {csv_dir}")
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts, ignore_index=True)


def load_threshold_monitor_dataframe(run_dir: Path | str) -> pd.DataFrame:
    """Merge matmul + nonmatmul threshold-monitor CSVs under run_dir/csv/."""
    rd = Path(run_dir)
    csv_dir = rd / "csv"
    parts: list[pd.DataFrame] = []
    for name in ("threshold_monitor_by_site.csv", "threshold_monitor_nonmatmul_by_site.csv"):
        p = csv_dir / name
        if p.is_file():
            parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError(f"no threshold_monitor*.csv under {csv_dir}")
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts, ignore_index=True)


def load_run_dataframe(run_dir: Path | str) -> pd.DataFrame:
    """Load sweep or threshold-monitor CSV(s) from a run directory."""
    rd = Path(run_dir)
    csv_dir = rd / "csv"
    if (csv_dir / "sweep_summary.csv").is_file() or (csv_dir / "sweep_summary_nonmatmul.csv").is_file():
        return load_sweep_dataframe(rd)
    return load_threshold_monitor_dataframe(rd)


def parse_args():
    rr = default_results_root()
    ap = argparse.ArgumentParser(
        description="Plots from sweep_summary-style CSV: acc_fault vs layer per op_type; "
        "optional tp/runs and FPR-by-op_type when columns exist.",
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help="If set, read csv/sweep_summary.csv (+ _nonmatmul.csv if present) from this run directory.",
    )
    ap.add_argument(
        "--in-csv",
        default=None,
        help="Path to sweep_summary.csv (required if --run-dir not set).",
    )
    ap.add_argument(
        "--out-png-acc",
        default=str(rr / "sweep_summary_acc_fault.png"),
        help="Output path for accuracy line plot.",
    )
    ap.add_argument(
        "--title",
        default="Layer × op_type sweep: acc_fault",
        help="Figure title for the accuracy plot (and FPR chart prefix).",
    )
    ap.add_argument(
        "--out-png-tp-rate",
        default=None,
        help="If set and CSV has tp,runs columns, also save tp/runs vs layer per op.",
    )
    ap.add_argument(
        "--out-png-fpr-by-op",
        default=None,
        help="Bar chart: false-positive rate per op_type (aggregated across layers). "
        "Default: alongside --out-png-acc in plots/sweep_fpr_by_op.png when fp/normal/runs exist.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.run_dir:
        df = load_run_dataframe(args.run_dir)
    elif args.in_csv:
        df = pd.read_csv(args.in_csv)
    else:
        raise SystemExit("error: pass --run-dir or --in-csv")
    if df.empty:
        raise RuntimeError("empty input csv")

    if _is_threshold_monitor_df(df):
        plot_threshold_monitor(df, args)
        return

    layers = sorted(df["layer"].unique().tolist())
    op_types = sorted(df["op_type"].unique().tolist())
    baseline_acc = float(df["acc_baseline"].iloc[0])

    plt.figure(figsize=(10, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, op in enumerate(op_types):
        dfo = df[df["op_type"] == op].sort_values("layer")
        plt.plot(
            dfo["layer"].tolist(),
            dfo["acc_fault"].tolist(),
            marker=markers[i % len(markers)],
            linewidth=1.8,
            label=op,
        )

    plt.axhline(y=baseline_acc, linestyle="--", color="black", linewidth=1.2, label="baseline_acc")
    plt.xticks(layers, [str(x) for x in layers])
    plt.xlabel("layer")
    plt.ylabel("acc_fault")
    plt.title(args.title)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_png_acc, dpi=160)
    print(args.out_png_acc)

    if args.out_png_tp_rate and "tp" in df.columns and "runs" in df.columns:
        plt.figure(figsize=(10, 5))
        for i, op in enumerate(op_types):
            dfo = df[df["op_type"] == op].sort_values("layer")
            rates = (dfo["tp"].astype(float) / dfo["runs"].astype(float).replace(0, float("nan"))).tolist()
            plt.plot(
                dfo["layer"].tolist(),
                rates,
                marker=markers[i % len(markers)],
                linewidth=1.8,
                label=op,
            )
        plt.xticks(layers, [str(x) for x in layers])
        plt.xlabel("layer")
        plt.ylabel("tp / runs")
        plt.title(
            "ACC: TP rate vs layer (per op_type)\n"
            "tp = thr fired while the site was fault-injected; "
            "runs = hook evaluations at the target site."
        )
        plt.ylim(0.0, 1.02)
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(args.out_png_tp_rate, dpi=160)
        print(args.out_png_tp_rate)

    if "fp" in df.columns and "runs" in df.columns and "normal" in df.columns:
        fpr_out = args.out_png_fpr_by_op
        if fpr_out is None:
            fpr_out = str(Path(args.out_png_acc).parent / "sweep_fpr_by_op.png")
        plot_fpr_by_op_type(df, fpr_out, title_prefix=args.title)
        print(fpr_out)


if __name__ == "__main__":
    main()
