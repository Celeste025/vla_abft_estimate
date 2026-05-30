"""Plot per-layer hidden-state diff between clean and fault runs."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from results_layout import dataset_slug, model_slug_from_id


def _load_layer_tensors(run_dir: Path) -> Dict[int, torch.Tensor]:
    pat = re.compile(r"^L(\d+)\.pt$")
    out: Dict[int, torch.Tensor] = {}
    for p in sorted(run_dir.glob("L*.pt")):
        m = pat.match(p.name)
        if not m:
            continue
        rec = torch.load(p, map_location="cpu", weights_only=False)
        t = rec["tensor"] if isinstance(rec, dict) else rec
        out[int(m.group(1))] = t.float()
    if not out:
        raise FileNotFoundError(f"no L*.pt under {run_dir}")
    return out


def _compute_metrics(clean: Dict[int, torch.Tensor], fault: Dict[int, torch.Tensor]) -> List[Dict[str, Any]]:
    layers = sorted(set(clean.keys()) & set(fault.keys()))
    if not layers:
        raise ValueError("no overlapping layers between clean and fault")
    rows: List[Dict[str, Any]] = []
    eps = 1e-12
    for i in layers:
        hc = clean[i]
        hf = fault[i]
        if hc.shape != hf.shape:
            raise ValueError(f"L{i}: shape mismatch clean={hc.shape} fault={hf.shape}")
        diff = hf - hc
        flat_c = hc.reshape(-1)
        flat_d = diff.reshape(-1)
        l2_clean = float(torch.linalg.vector_norm(flat_c).item())
        l1_clean = float(torch.linalg.vector_norm(flat_c, ord=1).item())
        l2_diff = float(torch.linalg.vector_norm(flat_d).item())
        l1_diff = float(torch.linalg.vector_norm(flat_d, ord=1).item())
        rows.append(
            {
                "layer": i,
                "shape": list(hc.shape),
                "numel": int(hc.numel()),
                "max_abs_diff": float(diff.abs().max().item()),
                "mean_abs_diff": float(diff.abs().mean().item()),
                "l2_diff": l2_diff,
                "l1_diff": l1_diff,
                "rel_l2": l2_diff / (l2_clean + eps),
                "rel_l1": l1_diff / (l1_clean + eps),
                "l2_clean": l2_clean,
                "l1_clean": l1_clean,
            }
        )
    return rows


def _default_plot_dir(meta: Dict[str, Any], artifact_dir: Path) -> Path:
    model_id = str(meta.get("model_id", "Qwen/Qwen2.5-7B-Instruct"))
    dataset = str(meta.get("benchmark", "hellaswag"))
    case_idx = int(meta.get("case_idx", 0))
    ending_idx = int(meta.get("ending_idx", 0))
    fd = float(meta.get("fault_delta", 100))
    site = str(meta.get("target_site", meta.get("inject_site", "L0_mlp_down")))
    if not site.startswith("L"):
        layer = int(meta.get("inject_layer", 0))
        site = f"L{layer}_{meta.get('inject_site', 'mlp_down')}"
    stem = f"layer_hidden_diff_case{case_idx}_end{ending_idx}_{site}_fd{fd:g}"
    return (
        Path(__file__).resolve().parent
        / "results"
        / "distribution"
        / f"{model_slug_from_id(model_id)}_{dataset_slug(dataset)}"
        / "layer_hidden_diff"
        / stem
    )


def plot_metrics(rows: List[Dict[str, Any]], meta: Dict[str, Any], out_png: Path) -> None:
    layers = [r["layer"] for r in rows]
    max_abs = [r["max_abs_diff"] for r in rows]
    rel_l2 = [r["rel_l2"] for r in rows]
    rel_l1 = [r["rel_l1"] for r in rows]
    mean_abs = [r["mean_abs_diff"] for r in rows]
    l2_clean = [r["l2_clean"] for r in rows]
    l1_clean = [r["l1_clean"] for r in rows]

    target = str(meta.get("target_site", "L0_mlp_down"))
    fd = meta.get("fault_delta", 100)
    case_idx = meta.get("case_idx", 0)
    ending_idx = meta.get("ending_idx", 0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax0, ax1, ax2 = axes

    ax0.plot(layers, max_abs, "o-", color="crimson", linewidth=1.8, markersize=5, label="max |Δh|")
    ax0.set_ylabel("max |Δh|", color="crimson")
    ax0.tick_params(axis="y", labelcolor="crimson")
    ax0.set_ylim(bottom=0)

    ax0_r = ax0.twinx()
    ax0_r.plot(layers, mean_abs, "s--", color="steelblue", linewidth=1.4, markersize=4, label="mean |Δh|")
    ax0_r.set_ylabel("mean |Δh|", color="steelblue")
    ax0_r.tick_params(axis="y", labelcolor="steelblue")
    ax0_r.set_ylim(bottom=0)

    lines_l, labels_l = ax0.get_legend_handles_labels()
    lines_r, labels_r = ax0_r.get_legend_handles_labels()
    ax0.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", fontsize=9)
    ax0.grid(True, alpha=0.25)
    ax0.set_title(
        f"HellaSwag case {case_idx} ending {ending_idx} — layer hidden diff\n"
        f"{target} fixed+{fd:g} thr-none (down_proj + residual)"
    )

    ax1.plot(
        layers,
        rel_l2,
        "^-",
        color="darkorange",
        linewidth=1.8,
        markersize=5,
        label="||Δh||₂ / ||h_clean||₂",
    )
    ax1.set_ylabel("rel L2 diff", color="darkorange")
    ax1.tick_params(axis="y", labelcolor="darkorange")
    ax1.set_ylim(bottom=0)

    ax1_r = ax1.twinx()
    ax1_r.plot(
        layers,
        rel_l1,
        "d--",
        color="mediumpurple",
        linewidth=1.4,
        markersize=4,
        label="||Δh||₁ / ||h_clean||₁",
    )
    ax1_r.set_ylabel("rel L1 diff", color="mediumpurple")
    ax1_r.tick_params(axis="y", labelcolor="mediumpurple")
    ax1_r.set_ylim(bottom=0)
    lines_rl2, labels_rl2 = ax1.get_legend_handles_labels()
    lines_rl1, labels_rl1 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines_rl2 + lines_rl1, labels_rl2 + labels_rl1, loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.plot(layers, l2_clean, "D-", color="seagreen", linewidth=1.8, markersize=5, label="||h_clean||₂")
    ax2.set_ylabel("||h_clean||₂", color="seagreen")
    ax2.tick_params(axis="y", labelcolor="seagreen")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.25)

    ax2_r = ax2.twinx()
    ax2_r.plot(
        layers,
        l1_clean,
        "v:",
        color="mediumpurple",
        linewidth=1.5,
        markersize=4,
        label="||h_clean||₁",
    )
    ax2_r.set_ylabel("||h_clean||₁", color="mediumpurple")
    ax2_r.tick_params(axis="y", labelcolor="mediumpurple")
    ax2_r.set_ylim(bottom=0)
    lines_l2, labels_l2 = ax2.get_legend_handles_labels()
    lines_r2, labels_r2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines_l2 + lines_r2, labels_l2 + labels_r2, loc="upper right", fontsize=9)

    ax2.set_xlabel("layer index")
    ax2.annotate(
        "L2–L3 jump in ||h_clean|| (L2/L1) drives rel_L2 / rel_L1 drop",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=8,
        color="dimgray",
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _tensor_to_2d(t: torch.Tensor) -> np.ndarray:
    """(1, seq, hidden) or (seq, hidden) -> (seq, hidden) float32 numpy."""
    if t.ndim == 3 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 2:
        raise ValueError(f"expected 2D or (1,T,H) tensor, got shape {tuple(t.shape)}")
    return t.numpy()


def _symmetric_limit(
    arr: np.ndarray,
    *,
    percentile: float = 99.5,
    floor: float = 1e-12,
) -> Tuple[float, float]:
    """Half-width L for (-L,L). Uses percentile of |arr| to avoid outlier-dominated scales."""
    flat = np.abs(arr).reshape(-1)
    if flat.size == 0:
        return max(floor, 1.0), 0.0
    p = float(np.percentile(flat, percentile))
    mx = float(flat.max())
    lim = max(p, floor)
    return lim, mx


def _layer_panel_limits(
    hc: np.ndarray,
    hf: np.ndarray,
    d: np.ndarray,
    ad: np.ndarray,
    *,
    percentile: float,
) -> Dict[str, float]:
    """Per-layer symmetric limits; clean/fault share hs_lim for side-by-side comparison."""
    lc, mc = _symmetric_limit(hc, percentile=percentile)
    lf, mf = _symmetric_limit(hf, percentile=percentile)
    ld, md = _symmetric_limit(d, percentile=percentile)
    la, ma = _symmetric_limit(ad, percentile=percentile)
    return {
        "hs_lim": max(lc, lf),
        "hs_max": max(mc, mf),
        "diff_lim": ld,
        "diff_max": md,
        "abs_lim": la,
        "abs_max": ma,
    }


def _format_layer_stats(row: Dict[str, Any], lim: Dict[str, float], *, scale_pct: float) -> str:
    return (
        f"color scale (symmetric)\n"
        f"  p{scale_pct:g}% |h|: ±{lim['hs_lim']:.4g}\n"
        f"  true max|h|: {lim['hs_max']:.4g}\n"
        f"  p{scale_pct:g}% |Δh|: ±{lim['diff_lim']:.4g}\n"
        f"  true max|Δh|: {lim['diff_max']:.4g}\n"
        f"\n"
        f"|Δh| absolute\n"
        f"  max  = {row['max_abs_diff']:.6g}\n"
        f"  mean = {row['mean_abs_diff']:.6g}\n"
        f"  L2   = {row['l2_diff']:.6g}\n"
        f"\n"
        f"|Δh| relative\n"
        f"  L2/||h_clean||₂ = {row['rel_l2']:.6g}\n"
        f"  L1/||h_clean||₁ = {row['rel_l1']:.6g}\n"
        f"  max/||h_clean||₂ = {row['max_abs_diff'] / (row['l2_clean'] + 1e-12):.6g}\n"
        f"  mean/||h_clean||₂ = {row['mean_abs_diff'] / (row['l2_clean'] + 1e-12):.6g}"
    )


# Sequential: blue→green→yellow; diverging diff: blue↔red
CMAP_HIDDEN = "viridis"
CMAP_DIFF = "coolwarm"
CMAP_ABS = "viridis"


def plot_layer_heatmaps(
    clean: Dict[int, torch.Tensor],
    fault: Dict[int, torch.Tensor],
    meta: Dict[str, Any],
    rows: List[Dict[str, Any]],
    out_dir: Path,
    *,
    scale_percentile: float = 99.5,
    dpi: int = 90,
) -> Dict[str, Any]:
    """Per layer: 4-panel figure; each panel uses symmetric (-L, L) limits (per layer)."""
    layers = sorted(set(clean.keys()) & set(fault.keys()))
    metrics_by_layer = {int(r["layer"]): r for r in rows}
    if not layers:
        raise ValueError("no layers to plot")

    hc0 = _tensor_to_2d(clean[layers[0]])
    seq_len, hidden_dim = hc0.shape

    target = str(meta.get("target_site", "L0_mlp_down"))
    fd = meta.get("fault_delta", 100)
    case_idx = meta.get("case_idx", 0)
    ending_idx = meta.get("ending_idx", 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    per_layer_scales: List[Dict[str, Any]] = []
    extent = [0, hidden_dim, seq_len, 0]
    fig_w = max(16.0, hidden_dim / 180.0)

    for i in layers:
        hc = _tensor_to_2d(clean[i])
        hf = _tensor_to_2d(fault[i])
        d = hf - hc
        ad = np.abs(d)
        lim = _layer_panel_limits(hc, hf, d, ad, percentile=scale_percentile)
        hs_lim = lim["hs_lim"]
        diff_lim = lim["diff_lim"]
        abs_lim = lim["abs_lim"]
        per_layer_scales.append({"layer": i, "scale_percentile": scale_percentile, **lim})

        fig = plt.figure(figsize=(fig_w, 6.5))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.42], height_ratios=[1, 1], wspace=0.28, hspace=0.32)
        ax_c = fig.add_subplot(gs[0, 0])
        ax_f = fig.add_subplot(gs[0, 1])
        ax_d = fig.add_subplot(gs[1, 0])
        ax_a = fig.add_subplot(gs[1, 1])
        ax_txt = fig.add_subplot(gs[:, 2])
        ax_txt.axis("off")

        im_kw = dict(aspect="auto", extent=extent, interpolation="nearest")
        pct = scale_percentile
        im0 = ax_c.imshow(hc, cmap=CMAP_HIDDEN, vmin=-hs_lim, vmax=hs_lim, **im_kw)
        ax_c.set_title(f"clean  ±p{pct:g}%={hs_lim:.3g}  (max|h|={lim['hs_max']:.3g})")
        ax_c.set_ylabel("token")
        fig.colorbar(im0, ax=ax_c, fraction=0.046, pad=0.02)

        im1 = ax_f.imshow(hf, cmap=CMAP_HIDDEN, vmin=-hs_lim, vmax=hs_lim, **im_kw)
        ax_f.set_title(f"fault  ±p{pct:g}%={hs_lim:.3g}  (max|h|={lim['hs_max']:.3g})")
        fig.colorbar(im1, ax=ax_f, fraction=0.046, pad=0.02)

        im2 = ax_d.imshow(d, cmap=CMAP_DIFF, vmin=-diff_lim, vmax=diff_lim, **im_kw)
        ax_d.set_title(f"Δ  ±p{pct:g}%={diff_lim:.3g}  (max|Δ|={lim['diff_max']:.3g})")
        ax_d.set_ylabel("token")
        ax_d.set_xlabel("hidden dim")
        fig.colorbar(im2, ax=ax_d, fraction=0.046, pad=0.02)

        im3 = ax_a.imshow(ad, cmap=CMAP_ABS, vmin=0.0, vmax=abs_lim, **im_kw)
        ax_a.set_title(f"|Δ|  p{pct:g}%={abs_lim:.3g}  (max|Δ|={lim['abs_max']:.3g})")
        ax_a.set_xlabel("hidden dim")
        fig.colorbar(im3, ax=ax_a, fraction=0.046, pad=0.02)

        row = metrics_by_layer[i]
        stats_txt = _format_layer_stats(row, lim, scale_pct=scale_percentile)
        ax_txt.text(
            0.02,
            0.98,
            stats_txt,
            transform=ax_txt.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
        )

        fig.suptitle(
            f"Layer {i} — HellaSwag case {case_idx} ending {ending_idx}  "
            f"(seq={seq_len}, hidden={hidden_dim})\n"
            f"{target} fixed+{fd:g} thr-none  |  scale: p{scale_percentile:g}% symmetric (hidden/diff), p{pct:g}% [0,L] (|Δ|)",
            fontsize=11,
            y=1.02,
        )
        out_png = out_dir / f"L{i:02d}_layer_panels.png"
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(out_png))

    scale_path = out_dir / "heatmap_scale.json"
    scale_payload = {
        "layout": "2x2_panels_plus_stats",
        "scale_mode": "per_layer_symmetric_percentile",
        "scale_percentile": scale_percentile,
        "cmaps": {"hidden": CMAP_HIDDEN, "diff": CMAP_DIFF, "abs": CMAP_ABS},
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "per_layer_limits": per_layer_scales,
        "layers": layers,
        "pngs": paths,
    }
    scale_path.write_text(json.dumps(scale_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return scale_payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifact-dir",
        required=True,
        help="Root with clean/, fault/, meta.json (from run_hellaswag_layer_hidden_diff.py)",
    )
    ap.add_argument("--plot-dir", default=None, help="Directory for png/csv/json; default from meta")
    ap.add_argument("--no-heatmaps", action="store_true", help="Skip per-layer 4-panel heatmaps")
    ap.add_argument(
        "--scale-percentile",
        type=float,
        default=99.5,
        help="Symmetric color half-width = percentile of |values| per panel (avoids outlier washout)",
    )
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    meta_path = artifact_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    clean_dir = artifact_dir / "clean"
    fault_dir = artifact_dir / "fault"
    clean = _load_layer_tensors(clean_dir)
    fault = _load_layer_tensors(fault_dir)
    rows = _compute_metrics(clean, fault)

    plot_dir = Path(args.plot_dir) if args.plot_dir else _default_plot_dir(meta, artifact_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    out_json = plot_dir / "layer_hidden_diff_metrics.json"
    out_csv = plot_dir / "layer_hidden_diff_metrics.csv"
    out_png = plot_dir / "layer_hidden_diff.png"

    payload = {
        "artifact_dir": str(artifact_dir),
        "meta": meta,
        "metrics": rows,
        "plot_png": str(out_png),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "layer",
        "max_abs_diff",
        "mean_abs_diff",
        "l2_diff",
        "l1_diff",
        "rel_l2",
        "rel_l1",
        "l2_clean",
        "l1_clean",
        "numel",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    plot_metrics(rows, meta, out_png)

    heatmap_info: Optional[Dict[str, Any]] = None
    if not args.no_heatmaps:
        heatmap_dir = plot_dir / "heatmaps"
        heatmap_info = plot_layer_heatmaps(
            clean,
            fault,
            meta,
            rows,
            heatmap_dir,
            scale_percentile=float(args.scale_percentile),
        )
        payload["heatmap_dir"] = str(heatmap_dir)
        payload["heatmap_scale"] = {
            "scale_mode": heatmap_info.get("scale_mode"),
            "seq_len": heatmap_info["seq_len"],
            "hidden_dim": heatmap_info["hidden_dim"],
            "per_layer_limits": heatmap_info.get("per_layer_limits"),
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "wrote_json": str(out_json),
        "wrote_csv": str(out_csv),
        "wrote_png": str(out_png),
        "layers": len(rows),
        "L0_max_abs_diff": rows[0]["max_abs_diff"] if rows else None,
        "L27_max_abs_diff": rows[-1]["max_abs_diff"] if rows else None,
    }
    if heatmap_info is not None:
        summary["heatmap_dir"] = str(plot_dir / "heatmaps")
        summary["scale_mode"] = heatmap_info.get("scale_mode")
        summary["n_heatmaps"] = len(heatmap_info["pngs"])

    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
