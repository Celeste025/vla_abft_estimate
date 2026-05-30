"""Pool first N HellaSwag cases (×4 forwards), fit Normal / Laplace / Student-t, plot histograms."""
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
from scipy import stats

from results_layout import dataset_slug, model_slug_from_id


def _parse_tensor_files(data_dir: Path) -> Dict[int, List[Path]]:
    pat = re.compile(r"^tc_(\d+)_fwd_(\d+)\.pt$")
    by_case: Dict[int, List[tuple[int, Path]]] = {}
    for p in sorted(data_dir.glob("tc_*_fwd_*.pt")):
        m = pat.match(p.name)
        if not m:
            continue
        case_idx = int(m.group(1))
        fwd_idx = int(m.group(2))
        by_case.setdefault(case_idx, []).append((fwd_idx, p))
    out: Dict[int, List[Path]] = {}
    for case_idx in sorted(by_case.keys()):
        out[case_idx] = [pp for _, pp in sorted(by_case[case_idx], key=lambda t: t[0])]
    return out


def _load_pool_values(paths: List[Path]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for p in paths:
        rec = torch.load(p, map_location="cpu", weights_only=False)
        t = rec["tensor"] if isinstance(rec, dict) else rec
        chunks.append(t.flatten().float())
    if not chunks:
        raise ValueError("no tensors to fit")
    return torch.cat(chunks)


def _subsample_np(values: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    n = int(values.size)
    if n <= max_points:
        return values
    rng = np.random.default_rng(int(seed))
    return values[rng.choice(n, size=max_points, replace=False)]


def _fit_distributions(fit_sample: np.ndarray) -> Dict[str, Any]:
    """MLE / scipy.fit on fit_sample (may be subsampled)."""
    mu = float(np.mean(fit_sample))
    var = float(np.var(fit_sample))
    std = float(np.std(fit_sample))

    out: Dict[str, Any] = {
        "normal": {"loc": mu, "scale": std, "variance": var},
    }

    try:
        lap_loc, lap_scale = stats.laplace.fit(fit_sample)
        out["laplace"] = {
            "loc": float(lap_loc),
            "scale": float(lap_scale),
            "variance": float(2.0 * lap_scale**2),
        }
    except Exception as e:
        out["laplace"] = {"error": str(e)}

    try:
        t_df, t_loc, t_scale = stats.t.fit(fit_sample)
        out["student_t"] = {
            "df": float(t_df),
            "loc": float(t_loc),
            "scale": float(t_scale),
            "variance": float(t_scale**2 * t_df / (t_df - 2.0)) if t_df > 2.0 else None,
        }
    except Exception as e:
        out["student_t"] = {"error": str(e)}

    return out


def _pdf_curves(
    fits: Dict[str, Any],
    xs: np.ndarray,
) -> List[Tuple[str, np.ndarray, str]]:
    curves: List[Tuple[str, np.ndarray, str]] = []
    n = fits.get("normal", {})
    if n.get("scale", 0) > 1e-12:
        loc, sc = float(n["loc"]), float(n["scale"])
        pdf = stats.norm.pdf(xs, loc=loc, scale=sc)
        curves.append(("normal", pdf, f"Normal(μ={loc:.4g}, σ={sc:.4g})"))
    lap = fits.get("laplace", {})
    if "scale" in lap and float(lap["scale"]) > 1e-12:
        loc, sc = float(lap["loc"]), float(lap["scale"])
        pdf = stats.laplace.pdf(xs, loc=loc, scale=sc)
        curves.append(("laplace", pdf, f"Laplace(μ={loc:.4g}, b={sc:.4g})"))
    st = fits.get("student_t", {})
    if "df" in st and "scale" in st and float(st["scale"]) > 1e-12:
        df, loc, sc = float(st["df"]), float(st["loc"]), float(st["scale"])
        pdf = stats.t.pdf(xs, df=df, loc=loc, scale=sc)
        curves.append(("student_t", pdf, f"Student-t(df={df:.2g}, μ={loc:.4g}, σ={sc:.4g})"))
    return curves


def default_distribution_dir(
    *,
    results_root: Path,
    model_id: str,
    dataset: str,
    layer: int,
    site_suffix: str,
    max_cases: int,
) -> Path:
    seg = f"L{int(layer)}_{site_suffix}_n{int(max_cases)}"
    return results_root / "distribution" / f"{model_slug_from_id(model_id)}_{dataset_slug(dataset)}" / seg


def plot_distribution(
    values: torch.Tensor,
    *,
    fits: Dict[str, Any],
    n_cases: int,
    site_id: str,
    out_png: Path,
    hist_bins: int = 120,
    max_plot_points: int = 500_000,
    seed: int = 2026,
) -> None:
    full = values.numpy()
    hist_sample = _subsample_np(full, max_plot_points, seed)
    fit_sample = _subsample_np(full, min(max_plot_points, 200_000), seed + 1)

    vmin, vmax = float(np.percentile(hist_sample, 0.5)), float(np.percentile(hist_sample, 99.5))
    if vmax <= vmin:
        vmin, vmax = float(hist_sample.min()), float(hist_sample.max())
    bins = np.linspace(vmin, vmax, int(hist_bins))
    xs = np.linspace(vmin, vmax, 400)

    style = {
        "normal": ("r-", 2.0),
        "laplace": ("g-", 2.0),
        "student_t": ("darkorange", 2.0),
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(
        hist_sample,
        bins=bins,
        density=True,
        alpha=0.5,
        color="steelblue",
        edgecolor="none",
        label="empirical (subsample)",
    )
    for name, pdf, label in _pdf_curves(fits, xs):
        ls, lw = style.get(name, ("k-", 1.8))
        ax.plot(xs, pdf, ls, linewidth=lw, label=label)
    ax.set_xlabel("activation value")
    ax.set_ylabel("density")
    ax.set_title(
        f"{site_id} — first {n_cases} cases × 4 forwards\n"
        f"fits on {fit_sample.size:,} subsampled elements (MLE / scipy.fit)"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory with tc_*_fwd_*.pt and meta.json")
    ap.add_argument("--case-limits", nargs="+", type=int, default=[5, 10, 20, 50, 100])
    ap.add_argument(
        "--plot-dir",
        default=None,
        help="default: results/distribution/{model}_{dataset}/L{layer}_{site}_n{max_cases}",
    )
    ap.add_argument("--results-root", default=None, help="default: acc_test/results")
    ap.add_argument("--max-cases-label", type=int, default=None, help="n in path segment, default max(case-limits)")
    ap.add_argument("--hist-bins", type=int, default=120)
    ap.add_argument("--max-plot-points", type=int, default=500_000)
    ap.add_argument("--fit-subsample", type=int, default=200_000, help="Max elements for scipy MLE fit")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    by_case = _parse_tensor_files(data_dir)
    if not by_case:
        raise FileNotFoundError(f"no tc_*_fwd_*.pt under {data_dir}")

    meta_path = data_dir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    model_id = str(meta.get("model_id", "Qwen/Qwen2.5-7B-Instruct"))
    dataset = str(meta.get("benchmark", "hellaswag"))
    site_id = str(meta.get("site_id", "L24_o_proj"))
    layer = int(meta.get("layer", 24))
    site_suffix = site_id.split("_", 1)[1] if "_" in site_id else "o_proj"
    seed = int(meta.get("seed", 2026))
    max_cases_label = int(args.max_cases_label) if args.max_cases_label is not None else max(
        int(x) for x in args.case_limits
    )

    results_root = Path(args.results_root) if args.results_root else Path(__file__).resolve().parent / "results"
    plot_dir = Path(args.plot_dir) if args.plot_dir else default_distribution_dir(
        results_root=results_root,
        model_id=model_id,
        dataset=dataset,
        layer=layer,
        site_suffix=site_suffix,
        max_cases=max_cases_label,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    plot_paths: List[str] = []
    for n_cases in sorted(set(int(x) for x in args.case_limits)):
        case_ids = sorted(by_case.keys())[:n_cases]
        if len(case_ids) < n_cases:
            print(
                f"[warn] requested n_cases={n_cases} but only {len(by_case)} cases available; using {len(case_ids)}",
                flush=True,
            )
        paths: List[Path] = []
        for cid in case_ids:
            paths.extend(by_case[cid])
        values = _load_pool_values(paths)
        full_np = values.numpy()
        fit_sample = _subsample_np(full_np, int(args.fit_subsample), seed + int(n_cases))
        fits = _fit_distributions(fit_sample)
        stats = {
            "n_cases": len(case_ids),
            "n_cases_requested": n_cases,
            "n_forwards_per_case": len(by_case[case_ids[0]]) if case_ids else 0,
            "n_tensors": len(paths),
            "n_elements": int(values.numel()),
            "fit_subsample_size": int(fit_sample.size),
            "sample_mean": float(np.mean(full_np)),
            "sample_variance": float(np.var(full_np)),
            "sample_std": float(np.std(full_np)),
            "fits": fits,
        }
        rows.append(stats)

        if not args.no_plots:
            out_png = plot_dir / f"dist_pooled_n{len(case_ids)}_cases_x4.png"
            plot_distribution(
                values,
                fits=fits,
                n_cases=len(case_ids),
                site_id=site_id,
                out_png=out_png,
                hist_bins=int(args.hist_bins),
                max_plot_points=int(args.max_plot_points),
                seed=seed,
            )
            plot_paths.append(str(out_png))
            print(str(out_png), flush=True)

    out_json = plot_dir / "distribution_fit_summary.json"
    out_csv = plot_dir / "distribution_fit_summary.csv"
    payload = {
        "data_dir": str(data_dir),
        "plot_dir": str(plot_dir),
        "distributions": ["normal", "laplace", "student_t"],
        "meta": {
            "site_id": site_id,
            "model_id": model_id,
            "dataset": dataset,
            "seed": seed,
            "n_cases_total": len(by_case),
        },
        "case_limits": [int(x) for x in args.case_limits],
        "rows": rows,
        "plots": plot_paths,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "n_cases",
        "n_cases_requested",
        "n_tensors",
        "n_elements",
        "sample_mean",
        "sample_std",
        "normal_loc",
        "normal_scale",
        "laplace_loc",
        "laplace_scale",
        "student_t_df",
        "student_t_loc",
        "student_t_scale",
    ]

    def _row_csv(r: Dict[str, Any]) -> Dict[str, Any]:
        f = r.get("fits", {})
        n, lap, st = f.get("normal", {}), f.get("laplace", {}), f.get("student_t", {})
        return {
            "n_cases": r.get("n_cases"),
            "n_cases_requested": r.get("n_cases_requested"),
            "n_tensors": r.get("n_tensors"),
            "n_elements": r.get("n_elements"),
            "sample_mean": r.get("sample_mean"),
            "sample_std": r.get("sample_std"),
            "normal_loc": n.get("loc"),
            "normal_scale": n.get("scale"),
            "laplace_loc": lap.get("loc"),
            "laplace_scale": lap.get("scale"),
            "student_t_df": st.get("df"),
            "student_t_loc": st.get("loc"),
            "student_t_scale": st.get("scale"),
        }

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(_row_csv(r))

    print(
        json.dumps(
            {"wrote_json": str(out_json), "wrote_csv": str(out_csv), "plot_dir": str(plot_dir), "fits": rows},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
