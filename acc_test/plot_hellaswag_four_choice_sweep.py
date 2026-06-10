"""Line plot: four-choice log-likelihood vs L0 mlp_down fault_delta sweep."""
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

CHOICE_COLORS = ["#4daf4a", "#377eb8", "#ff7f00", "#984ea3"]
CHOICE_MARKERS = ["o", "s", "^", "D"]


def _load_sweep(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_plot_dir(artifact_dir: Path, data: Dict[str, Any]) -> Path:
    model_id = str(data.get("model_id", "Qwen/Qwen2.5-7B-Instruct"))
    dataset = str(data.get("benchmark", "hellaswag"))
    return (
        Path(__file__).resolve().parent
        / "results"
        / "distribution"
        / f"{model_slug_from_id(model_id)}_{dataset_slug(dataset)}"
        / "four_choice"
        / artifact_dir.name
    )


def plot_sweep(data: Dict[str, Any], out_png: Path) -> None:
    clean = data["clean"]
    sweep = data["sweep"]
    label = int(data["label"])
    site = str(data.get("target_site", "L0_mlp_down"))
    case_idx = int(data.get("case_idx", 0))

    xs = [float(r["fault_delta"]) for r in sweep]
    n_choices = len(clean["scores"])

    fig, ax = plt.subplots(figsize=(9, 5))

    for i in range(n_choices):
        ys = [float(r["scores"][i]) for r in sweep]
        lbl = f"choice {i}" + (" (label)" if i == label else "")
        ax.plot(
            xs,
            ys,
            color=CHOICE_COLORS[i % len(CHOICE_COLORS)],
            marker=CHOICE_MARKERS[i % len(CHOICE_MARKERS)],
            linewidth=1.8,
            markersize=6,
            label=lbl,
        )
        ax.axhline(
            float(clean["scores"][i]),
            color=CHOICE_COLORS[i % len(CHOICE_COLORS)],
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
        )

    ax.set_xscale("log")
    ax.set_xlabel("fault_delta (fixed +, log scale)")
    ax.set_ylabel("ending log-likelihood (Σ log p, teacher forcing)")
    ax.set_title(
        f"HellaSwag case {case_idx} — {site} fault sweep (four choices)\n"
        f"dashed = clean (no fault)  |  label = choice {label}",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.35)

    pre_maxs = [
        float(r["pre_inject_max"])
        for r in sweep
        if r.get("pre_inject_max") is not None
    ]
    activation_M = max(pre_maxs) if pre_maxs else None
    if activation_M is not None and activation_M > 0:
        for mult, style, color in ((2.0, "-", "#d62728"), (5.0, "--", "#9467bd")):
            x_ref = mult * activation_M
            ax.axvline(
                x_ref,
                color=color,
                linewidth=1.6,
                linestyle=style,
                alpha=0.85,
                label=f"{mult:g}×M (M={activation_M:.4g})",
            )

    ax.legend(loc="best", fontsize=8, ncol=2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--plot-dir", default=None)
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    sweep_path = artifact_dir / "sweep.json"
    if not sweep_path.is_file():
        raise FileNotFoundError(f"missing {sweep_path}")

    data = _load_sweep(sweep_path)
    plot_dir = Path(args.plot_dir) if args.plot_dir else _default_plot_dir(artifact_dir, data)
    plot_dir.mkdir(parents=True, exist_ok=True)

    out_png = plot_dir / "four_choice_loglik_sweep.png"
    plot_sweep(data, out_png)

    pre_maxs = [
        float(r["pre_inject_max"])
        for r in data.get("sweep", [])
        if r.get("pre_inject_max") is not None
    ]
    activation_M = max(pre_maxs) if pre_maxs else None
    meta = {
        "artifact_dir": str(artifact_dir),
        "sweep_json": str(sweep_path),
        "plot_png": str(out_png),
        "fault_deltas": data.get("fault_deltas"),
        "clean_scores": data["clean"]["scores"],
        "activation_M": activation_M,
        "vlines_fault_delta": (
            [2.0 * activation_M, 5.0 * activation_M] if activation_M else None
        ),
    }
    meta_path = plot_dir / "plot_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
