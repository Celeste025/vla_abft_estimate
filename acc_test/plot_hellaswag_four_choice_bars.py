"""Bar charts: softmax-normalized HellaSwag four-choice scores (clean vs fault)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from results_layout import dataset_slug, model_slug_from_id

FAULT_STEMS = ["fault_fd10", "fault_fd100", "fault_fd1000"]
_LOG_Y_FLOOR = 1e-45


def _positive_for_log(values: List[float]) -> List[float]:
    return [max(float(v), _LOG_Y_FLOOR) for v in values]


def _setup_log_prob_yaxis(ax: plt.Axes, *arrays: List[float]) -> None:
    flat = [float(v) for arr in arrays for v in arr if v is not None]
    positive = [v for v in flat if v > 0]
    ymin = min(positive) * 0.1 if positive else _LOG_Y_FLOOR
    ymin = max(ymin, _LOG_Y_FLOOR)
    ymax = max(positive) if positive else 1.0
    ymax = min(1.05, max(ymax * 2.0, ymin * 10))
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("softmax probability (log scale)")


def _format_activation_note(
    site: str,
    pre_lo: Any,
    pre_hi: Any,
    inject_count: Any,
) -> str:
    if pre_lo is None or pre_hi is None:
        return f"{site} pre-inject activation: n/a"
    return (
        f"{site} pre-inject activation (before fault, 4 forwards):\n"
        f"  min = {float(pre_lo):.6g}    max = {float(pre_hi):.6g}\n"
        f"  inject_count = {inject_count}"
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_plot_dir(artifact_dir: Path, meta: Dict[str, Any]) -> Path:
    model_id = str(meta.get("model_id", "Qwen/Qwen2.5-7B-Instruct"))
    dataset = str(meta.get("benchmark", "hellaswag"))
    return (
        Path(__file__).resolve().parent
        / "results"
        / "distribution"
        / f"{model_slug_from_id(model_id)}_{dataset_slug(dataset)}"
        / "four_choice"
        / artifact_dir.name
    )


def _plot_clean_bars(payload: Dict[str, Any], out_png: Path) -> None:
    probs = payload["probs"]
    plot_probs = _positive_for_log(probs)
    label = int(payload["label"])
    n = len(probs)
    x = np.arange(n)
    colors = ["#4daf4a" if i == label else "#377eb8" for i in range(n)]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x, plot_probs, color=colors, edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"choice {i}" for i in range(n)])
    _setup_log_prob_yaxis(ax, probs)
    ax.set_title(
        f"HellaSwag case {payload.get('case_idx', 0)} — clean (no fault)\n"
        f"pred={payload['pred']}  label={label}  "
        f"({'correct' if payload.get('correct') else 'wrong'})",
        fontsize=10,
    )
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.subplots_adjust(bottom=0.14, top=0.88)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_fault_bars(
    fault: Dict[str, Any],
    clean: Dict[str, Any],
    out_png: Path,
) -> None:
    fault_probs = fault["probs"]
    clean_probs = clean["probs"]
    fault_plot = _positive_for_log(fault_probs)
    clean_plot = _positive_for_log(clean_probs)
    n = len(fault_plot)
    x = np.arange(n)
    label = int(fault["label"])
    fd = fault.get("fault_delta")
    site = fault.get("target_site", "L12_mlp_down")
    act_note = _format_activation_note(
        site,
        fault.get("pre_inject_min"),
        fault.get("pre_inject_max"),
        fault.get("inject_count", 0),
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar(
        x,
        fault_plot,
        color="#e41a1c",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
        label="fault (softmax)",
    )

    for i in range(n):
        ax.hlines(
            clean_plot[i],
            i - 0.4,
            i + 0.4,
            colors="#2166ac",
            linewidth=2.4,
            label="clean ref" if i == 0 else None,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"choice {i}" for i in range(n)])
    _setup_log_prob_yaxis(ax, fault_probs, clean_probs)
    ax.set_title(
        f"HellaSwag case {fault.get('case_idx', 0)} — {site} fixed+{fd:g} thr-none\n"
        f"pred={fault['pred']}  label={label}  "
        f"({'correct' if fault.get('correct') else 'wrong'})",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.3)

    fig.text(
        0.5,
        0.02,
        act_note,
        ha="center",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff3cd", edgecolor="#856404", alpha=0.95),
    )

    fig.subplots_adjust(bottom=0.28, top=0.86)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--plot-dir", default=None)
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    clean_path = artifact_dir / "clean.json"
    if not clean_path.is_file():
        raise FileNotFoundError(f"missing {clean_path}")
    clean = _load_json(clean_path)

    plot_dir = Path(args.plot_dir) if args.plot_dir else _default_plot_dir(artifact_dir, clean)
    plot_dir.mkdir(parents=True, exist_ok=True)

    wrote: List[str] = []
    clean_png = plot_dir / "bars_clean.png"
    _plot_clean_bars(clean, clean_png)
    wrote.append(str(clean_png))

    fault_pngs: Dict[str, str] = {}
    for stem in FAULT_STEMS:
        fault_path = artifact_dir / f"{stem}.json"
        if not fault_path.is_file():
            continue
        fault = _load_json(fault_path)
        out_png = plot_dir / f"bars_{stem}.png"
        _plot_fault_bars(fault, clean, out_png)
        wrote.append(str(out_png))
        fault_pngs[stem] = str(out_png)

    plot_meta = {
        "artifact_dir": str(artifact_dir),
        "plot_dir": str(plot_dir),
        "y_scale": "log on softmax probability",
        "clean_png": str(clean_png),
        "fault_pngs": fault_pngs,
        "clean_probs": clean["probs"],
    }
    meta_path = plot_dir / "plot_meta.json"
    meta_path.write_text(json.dumps(plot_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": wrote, "plot_meta": str(meta_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
