#!/usr/bin/env python3
"""Plot ResNet inject sweep: top1 and top5 (fault / protect / baseline horizontal)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def stage_for_site(site_id: str) -> str:
    if site_id == "conv1":
        return "stem"
    if site_id.startswith("layer1."):
        return "L1"
    if site_id.startswith("layer2."):
        return "L2"
    if site_id.startswith("layer3."):
        return "L3"
    if site_id.startswith("layer4."):
        return "L4"
    if site_id == "fc":
        return "fc"
    return "other"


def stage_spans(site_ids: List[str]) -> List[Tuple[int, int, str]]:
    """Inclusive [start, end] index ranges per stage label."""
    if not site_ids:
        return []
    spans: List[Tuple[int, int, str]] = []
    cur = stage_for_site(site_ids[0])
    start = 0
    for i, s in enumerate(site_ids[1:], start=1):
        st = stage_for_site(s)
        if st != cur:
            spans.append((start, i - 1, cur))
            start = i
            cur = st
    spans.append((start, len(site_ids) - 1, cur))
    return spans


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", type=str, required=True, help="master.csv from aggregate_resnet_sweep.py")
    ap.add_argument("--out-dir", type=str, required=True, help="Directory for top1.png / top5.png")
    ap.add_argument(
        "--inject-tag",
        type=str,
        default="random idx",
        help="Short label for plot title (e.g. 'random idx', 'max_abs idx').",
    )
    return ap.parse_args()


def _plot_one(
    df: pd.DataFrame,
    *,
    y_fault: str,
    y_protect: str,
    y_baseline: str,
    title: str,
    out_png: Path,
) -> None:
    x = range(len(df))
    b0 = float(df[y_baseline].iloc[0])

    fig, ax = plt.subplots(figsize=(22, 5))
    colors = {"stem": "#f0f0f0", "L1": "#e8f4fc", "L2": "#e8fce8", "L3": "#fcf8e8", "L4": "#fce8f8", "fc": "#eeeeee"}
    sites = df["site_id"].astype(str).tolist()
    for lo, hi, st in stage_spans(sites):
        ax.axvspan(lo - 0.5, hi + 0.5, color=colors.get(st, "#f5f5f5"), alpha=0.9, zorder=0)

    ax.plot(x, df[y_fault].astype(float), marker=".", linestyle="-", linewidth=1.2, label="fault", zorder=2)
    ax.plot(
        x,
        df[y_protect].astype(float),
        marker=".",
        linestyle="-",
        linewidth=1.2,
        label="fault+protect",
        zorder=2,
    )
    ax.axhline(y=b0, linestyle="--", color="black", linewidth=1.2, label="baseline", zorder=1)

    ax.set_xticks(list(x))
    ax.set_xticklabels(sites, rotation=75, ha="right", fontsize=6)
    ax.set_xlabel("inject site (ResNet-50 named_modules order)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, zorder=1)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.in_csv)
    if df.empty:
        raise SystemExit("empty csv")

    need = {
        "site_id",
        "acc_fault_top1",
        "acc_protect_top1",
        "acc_baseline_top1",
        "acc_fault_top5",
        "acc_protect_top5",
        "acc_baseline_top5",
    }
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"csv missing columns: {sorted(miss)}")

    tag = str(args.inject_tag).strip() or "inject"
    base_title = (
        f"ResNet-50 ImageNet-1k inject sweep (1000 samples, bs=1, {tag}, delta=10000)"
    )
    p1 = out_dir / "sweep_top1.png"
    p5 = out_dir / "sweep_top5.png"
    _plot_one(
        df,
        y_fault="acc_fault_top1",
        y_protect="acc_protect_top1",
        y_baseline="acc_baseline_top1",
        title=f"{base_title} — Top-1",
        out_png=p1,
    )
    _plot_one(
        df,
        y_fault="acc_fault_top5",
        y_protect="acc_protect_top5",
        y_baseline="acc_baseline_top5",
        title=f"{base_title} — Top-5",
        out_png=p5,
    )
    print(str(p1))
    print(str(p5))


if __name__ == "__main__":
    main()
