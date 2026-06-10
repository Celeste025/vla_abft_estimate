"""Plot clean / fault / diff heatmaps for HellaSwag token log-probability vectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from results_layout import dataset_slug, model_slug_from_id

CMAP_LOGP = "viridis"
CMAP_DIFF = "coolwarm"
CMAP_REL = "plasma"


def _load_vector(path: Path) -> np.ndarray:
    rec = torch.load(path, map_location="cpu", weights_only=False)
    v = rec["token_logp"] if isinstance(rec, dict) else rec
    arr = v.float().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[1] != 1:
        arr = arr.reshape(-1, 1)
    return arr


def _logp_limits(arr: np.ndarray, percentile: float) -> Tuple[float, float]:
    lo = float(np.percentile(arr, 100.0 - percentile))
    hi = float(np.percentile(arr, percentile))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _diff_symmetric_limit(arr: np.ndarray, percentile: float) -> float:
    lim = float(np.percentile(np.abs(arr), percentile))
    return max(lim, 1e-12)


def _logp_to_prob(logp: np.ndarray, *, logp_floor: float = -500.0) -> np.ndarray:
    """exp(log p) with clamp so exp does not underflow to 0."""
    return np.exp(np.clip(logp, logp_floor, 0.0))


def _relative_prob_perturbation(
    clean: np.ndarray, fault: np.ndarray, *, logp_floor: float = -500.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """|p_fault - p_clean| / p_clean in stable log-space.

    Since p = exp(log p), ratio equals |exp(Δlog p) - 1| (no division by tiny p).
    Returns (rel, p_clean, p_fault), each shape (T, 1).
    """
    delta = np.clip(fault - clean, logp_floor, -logp_floor)
    rel = np.abs(np.expm1(delta))
    p_clean = _logp_to_prob(clean, logp_floor=logp_floor)
    p_fault = _logp_to_prob(fault, logp_floor=logp_floor)
    return rel, p_clean, p_fault


def _nonnegative_limit(arr: np.ndarray, percentile: float) -> float:
    hi = float(np.percentile(arr, percentile))
    mx = float(np.max(arr))
    if percentile < 100.0:
        hi = min(hi, mx)
    return max(hi, 1e-12)


def _default_plot_dir(meta: Dict[str, Any], artifact_dir: Path) -> Path:
    model_id = str(meta.get("model_id", "Qwen/Qwen2.5-7B-Instruct"))
    dataset = str(meta.get("benchmark", "hellaswag"))
    case_idx = int(meta.get("case_idx", 0))
    ending_idx = int(meta.get("ending_idx", 0))
    fd = float(meta.get("fault_delta", 100))
    site = str(meta.get("target_site", "L0_mlp_down"))
    stem = f"logits_case{case_idx}_end{ending_idx}_{site}_fd{fd:g}"
    return (
        Path(__file__).resolve().parent
        / "results"
        / "distribution"
        / f"{model_slug_from_id(model_id)}_{dataset_slug(dataset)}"
        / "logits_diff"
        / stem
    )


def plot_heatmaps(
    clean: np.ndarray,
    fault: np.ndarray,
    meta: Dict[str, Any],
    out_png: Path,
    *,
    scale_percentile: float = 99.5,
    dpi: int = 120,
) -> Dict[str, Any]:
    if clean.shape != fault.shape:
        raise ValueError(f"shape mismatch clean={clean.shape} fault={fault.shape}")
    diff = fault - clean
    rel, p_clean, p_fault = _relative_prob_perturbation(clean, fault)

    vmin_c, vmax_c = _logp_limits(clean, scale_percentile)
    vmin_f, vmax_f = _logp_limits(fault, scale_percentile)
    lim_d = _diff_symmetric_limit(diff, scale_percentile)
    vmax_rel = _nonnegative_limit(rel, scale_percentile)

    tok_meta = meta.get("token_meta", {})
    boundary = int(tok_meta.get("ctx_predict_end", tok_meta.get("ctx_token_len", 0)))
    T = clean.shape[0]
    extent = [0, 1, T, 0]

    target = str(meta.get("target_site", "L0_mlp_down"))
    fd = meta.get("fault_delta", 100)
    case_idx = meta.get("case_idx", 0)
    ending_idx = meta.get("ending_idx", 0)

    fig, axes = plt.subplots(1, 4, figsize=(11, max(6, T / 18)), sharey=True)
    panels = [
        (axes[0], clean, "clean log p", CMAP_LOGP, (vmin_c, vmax_c)),
        (axes[1], fault, "fault log p", CMAP_LOGP, (vmin_f, vmax_f)),
        (axes[2], diff, "Δ log p", CMAP_DIFF, (-lim_d, lim_d)),
        (axes[3], rel, "|Δp|/p_clean", CMAP_REL, (0.0, vmax_rel)),
    ]

    xlabels = ("log p", "log p", "log p", "rel. prob.")
    for ax, arr, title, cmap, (vmin, vmax), xlab in zip(
        [p[0] for p in panels],
        [p[1] for p in panels],
        [p[2] for p in panels],
        [p[3] for p in panels],
        [p[4] for p in panels],
        xlabels,
    ):
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax.axhline(boundary - 0.5, color="white", linewidth=1.2, linestyle="--", alpha=0.9)
        ax.set_title(f"{title}\n[{vmin:.3g}, {vmax:.3g}]")
        ax.set_xlabel(xlab)
        fig.colorbar(im, ax=ax, fraction=0.08, pad=0.02)
    axes[0].set_ylabel("token position (predict t+1)")

    fig.suptitle(
        f"HellaSwag case {case_idx} end {ending_idx} — token log-probability (T={T})\n"
        f"{target} fixed+{fd:g} thr-none  |  dashed: prompt/ending boundary",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    rel_path = out_png.parent / "token_prob_rel_perturbation.csv"
    with open(rel_path, "w", encoding="utf-8") as f:
        f.write(
            "position,segment,logp_clean,logp_fault,p_clean,p_fault,delta_p,rel_perturbation\n"
        )
        for i in range(T):
            seg = "ctx" if i < boundary else "end"
            dp = float(p_fault[i, 0] - p_clean[i, 0])
            f.write(
                f"{i},{seg},{float(clean[i,0]):.8g},{float(fault[i,0]):.8g},"
                f"{float(p_clean[i,0]):.8g},{float(p_fault[i,0]):.8g},{dp:.8g},"
                f"{float(rel[i,0]):.8g}\n"
            )

    return {
        "vmin_clean": vmin_c,
        "vmax_clean": vmax_c,
        "vmin_fault": vmin_f,
        "vmax_fault": vmax_f,
        "lim_diff": lim_d,
        "vmax_rel_perturbation": vmax_rel,
        "rel_metric": "|p_fault-p_clean|/p_clean via |expm1(delta_logp)|",
        "rel_perturbation_max": float(np.max(rel)),
        "rel_perturbation_mean": float(np.mean(rel)),
        "rel_perturbation_csv": str(rel_path),
        "scale_percentile": scale_percentile,
        "boundary_token": boundary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument("--scale-percentile", type=float, default=99.5)
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    meta_path = artifact_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    clean = _load_vector(artifact_dir / "clean" / "token_logp.pt")
    fault = _load_vector(artifact_dir / "fault" / "token_logp.pt")

    plot_dir = Path(args.plot_dir) if args.plot_dir else _default_plot_dir(meta, artifact_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_png = plot_dir / "token_logp_heatmaps.png"

    scale_info = plot_heatmaps(
        clean,
        fault,
        meta,
        out_png,
        scale_percentile=float(args.scale_percentile),
    )

    out_json = plot_dir / "plot_meta.json"
    out_json.write_text(
        json.dumps(
            {"artifact_dir": str(artifact_dir), "meta": meta, "plot_png": str(out_png), **scale_info},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"wrote_png": str(out_png), "wrote_json": str(out_json), **scale_info}, ensure_ascii=False))


if __name__ == "__main__":
    main()
