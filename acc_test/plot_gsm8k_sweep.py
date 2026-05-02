from __future__ import annotations

import argparse

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="results/gsm8k_sweep_shared100.csv")
    ap.add_argument("--out-png-acc", default="results/gsm8k_sweep_shared100_acc.png")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.in_csv)
    if df.empty:
        raise RuntimeError("empty input csv")

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
    plt.title("GSM8K sweep (shared100) accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_png_acc, dpi=160)
    print(args.out_png_acc)


if __name__ == "__main__":
    main()

