from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="sweep.csv")
    ap.add_argument("--out-png-acc", default="sweep_acc_line.png")
    ap.add_argument("--out-png-delta", default="sweep_logits_abs_delta_line.png")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.in_csv)

    layers = sorted(df["layer"].unique().tolist())
    op_types = sorted(df["op_type"].unique().tolist())
    baseline_acc = float(df["acc_baseline"].iloc[0])

    # Plot 1: accuracy per op_type across layers
    plt.figure(figsize=(10, 5))
    for op in op_types:
        sub = df[df["op_type"] == op].set_index("layer").sort_index()
        ys = [float(sub.loc[l, "acc_fault"]) if l in sub.index else float("nan") for l in layers]
        plt.plot(layers, ys, marker="o", linewidth=1.8, label=op)
    plt.axhline(y=baseline_acc, linestyle="--", linewidth=1.2, color="black", label="baseline_acc")
    plt.xlabel("Layer")
    plt.ylabel("acc_fault")
    plt.xticks(layers, [str(l) for l in layers])
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_png_acc, dpi=160)

    # Plot 2: mean abs logits-change proxy per op_type across layers
    plt.figure(figsize=(10, 5))
    for op in op_types:
        sub = df[df["op_type"] == op].set_index("layer").sort_index()
        ys = [float(sub.loc[l, "mean_abs_delta_score"]) if l in sub.index else float("nan") for l in layers]
        plt.plot(layers, ys, marker="o", linewidth=1.8, label=op)
    plt.axhline(y=0.0, linestyle="--", linewidth=1.2, color="black", label="0")
    plt.xlabel("Layer")
    plt.ylabel("mean|Δlogits|(proxy by score)")
    plt.xticks(layers, [str(l) for l in layers])
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_png_delta, dpi=160)

    print(args.out_png_acc)
    print(args.out_png_delta)


if __name__ == "__main__":
    main()
