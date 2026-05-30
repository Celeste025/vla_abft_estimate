#!/usr/bin/env python3
"""Plot mean acc_fault (and optional baseline) vs gamma for a family of ACC sweep runs.

Example (thr-mMg, fixed fd=1000, gamma 2/3/5/10):
  python plot_sweep_average_acc.py \\
    --fault-mode fixed --fault-delta 1000 \\
    --gammas 2 3 5 10 \\
    --out-dir results/qwen-qwen2.5-7b-instruct_hellaswag/average
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from results_layout import build_run_config_segment, default_results_root, results_run_dir


def _load_sweep_for_site_set(run_dir: Path, site_set: str) -> pd.DataFrame:
    csv_dir = run_dir / "csv"
    if site_set == "matmul":
        path = csv_dir / "sweep_summary.csv"
    elif site_set == "nonmatmul":
        path = csv_dir / "sweep_summary_nonmatmul.csv"
    elif site_set == "merge":
        from plot_sweep_summary import load_sweep_dataframe

        return load_sweep_dataframe(run_dir)
    else:
        raise ValueError(f"site_set must be matmul|nonmatmul|merge, got {site_set!r}")
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_csv(path)


def _parse_gammas(raw: List[str]) -> List[float]:
    out: List[float] = []
    for x in raw:
        out.append(float(x))
    return sorted(set(out))


def _dataset_parent(results_root: Path, model_id: str, dataset: str) -> Path:
    slug = re.sub(r"[^\w.-]+", "_", model_id.strip().lower().replace("/", "-"))
    dslug = re.sub(r"[^\w.-]+", "_", dataset.strip().lower())
    return results_root / f"{slug}_{dslug}"


def _resolve_run_dir(
    *,
    results_root: Path,
    model_id: str,
    dataset: str,
    n_total: int,
    n_warmup: int,
    seed: int,
    gamma: float,
    fault_mode: str,
    fault_delta: Optional[float],
    acc_thr_enabled: bool = True,
    acc_thr_action: str = "golden",
) -> Path:
    seg = build_run_config_segment(
        n_total=n_total,
        n_warmup=n_warmup,
        gamma=gamma,
        fault_mode=fault_mode,
        seed=seed,
        fault_delta=fault_delta,
        acc_thr_enabled=acc_thr_enabled,
        acc_thr_action=acc_thr_action,  # type: ignore[arg-type]
    )
    return _dataset_parent(results_root, model_id, dataset) / seg


def find_thr_none_run_dir(
    *,
    results_root: Path,
    model_id: str,
    dataset: str,
    n_total: int,
    n_warmup: int,
    seed: int,
    fault_mode: str,
    fault_delta: Optional[float],
    site_set: str,
) -> Optional[Path]:
    """thr-none sweeps are gamma-independent; pick any matching run with the requested CSV."""
    parent = _dataset_parent(results_root, model_id, dataset)
    if fault_mode == "fixed":
        if fault_delta is None:
            return None
        fd = float(fault_delta)
        glob_pat = f"n{int(n_total)}_wu{int(n_warmup)}_g*_thr-none_fm-fixed_fd{fd:g}_s{int(seed)}"
    else:
        glob_pat = f"n{int(n_total)}_wu{int(n_warmup)}_g*_thr-none_fm-rand2pow_s{int(seed)}"
    csv_name = "sweep_summary.csv" if site_set == "matmul" else "sweep_summary_nonmatmul.csv"
    candidates = sorted(parent.glob(glob_pat), key=lambda p: p.name)
    for run_dir in candidates:
        if (run_dir / "csv" / csv_name).is_file():
            return run_dir
    return None


def _mean_acc_from_run(run_dir: Path, *, site_set: str) -> Dict[str, float]:
    df = _load_sweep_for_site_set(run_dir, site_set)
    if df.empty or "acc_fault" not in df.columns:
        raise ValueError(f"empty or missing acc_fault in {run_dir}")
    out = {
        "mean_acc_fault": float(df["acc_fault"].mean()),
        "std_acc_fault": float(df["acc_fault"].std(ddof=0)) if len(df) > 1 else 0.0,
        "n_sites": int(len(df)),
    }
    if "acc_baseline" in df.columns:
        out["mean_acc_baseline"] = float(df["acc_baseline"].iloc[0])
    return out


def collect_series(
    *,
    gammas: List[float],
    results_root: Path,
    model_id: str,
    dataset: str,
    n_total: int,
    n_warmup: int,
    seed: int,
    fault_mode: str,
    fault_delta: Optional[float],
    site_set: str,
    skip_missing: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for g in gammas:
        run_dir = _resolve_run_dir(
            results_root=results_root,
            model_id=model_id,
            dataset=dataset,
            n_total=n_total,
            n_warmup=n_warmup,
            seed=seed,
            gamma=g,
            fault_mode=fault_mode,
            fault_delta=fault_delta,
        )
        if not run_dir.is_dir():
            if skip_missing:
                print(f"[warn] skip gamma={g}: missing {run_dir}", flush=True)
                continue
            raise FileNotFoundError(f"missing run directory: {run_dir}")
        try:
            stats = _mean_acc_from_run(run_dir, site_set=site_set)
        except FileNotFoundError as e:
            if skip_missing:
                print(f"[warn] skip gamma={g}: {e}", flush=True)
                continue
            raise
        rows.append(
            {
                "gamma": float(g),
                "run_dir": str(run_dir),
                "site_set": site_set,
                **stats,
            }
        )
    if not rows:
        raise RuntimeError("no gamma points loaded")
    return rows


def _fault_label(fault_mode: str, fault_delta: Optional[float]) -> str:
    if fault_mode == "rand2pow":
        return "rand2pow"
    if fault_mode == "fixed" and fault_delta is not None:
        return f"fixed+{fault_delta:g}"
    return fault_mode


def plot_average_acc(
    rows: List[Dict[str, Any]],
    *,
    out_png: Path,
    title: str,
    fault_label: str,
    site_set: str,
    thr_none: Optional[Dict[str, Any]] = None,
) -> None:
    gammas = [r["gamma"] for r in rows]
    mean_fault = [r["mean_acc_fault"] for r in rows]
    baseline = rows[0].get("mean_acc_baseline") if rows else None
    thr_none_mean = float(thr_none["mean_acc_fault"]) if thr_none else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        gammas,
        mean_fault,
        "o-",
        linewidth=2.0,
        markersize=8,
        color="C0",
        label="thr-mMg mean acc_fault",
    )
    if baseline is not None:
        ax.axhline(
            float(baseline),
            linestyle="--",
            color="black",
            linewidth=1.2,
            label=f"baseline acc ({baseline:.4f})",
        )
    if thr_none_mean is not None:
        ax.axhline(
            thr_none_mean,
            linestyle="--",
            color="C3",
            linewidth=1.5,
            label=f"thr-none mean acc_fault ({thr_none_mean:.4f})",
        )
        x_mid = gammas[len(gammas) // 2]
        ax.annotate(
            f"no protection\n{thr_none_mean:.4f}",
            xy=(x_mid, thr_none_mean),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="C3",
        )
    for g, y in zip(gammas, mean_fault):
        ax.annotate(
            f"{y:.4f}",
            (g, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color="C0",
        )
    ax.set_xlabel("gamma (thr-mMg)")
    ax.set_ylabel("mean accuracy")
    ax.set_title(title)
    ax.set_xticks(gammas)
    ax.set_xticklabels([str(g).rstrip("0").rstrip(".") if g == int(g) else str(g) for g in gammas])
    y_refs = list(mean_fault)
    if baseline is not None:
        y_refs.append(float(baseline))
    if thr_none_mean is not None:
        y_refs.append(thr_none_mean)
    ymin = min(y_refs)
    ymax = max(y_refs)
    pad = max(0.02, (ymax - ymin) * 0.15)
    ax.set_ylim(max(0.0, ymin - pad), min(1.02, ymax + pad))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    foot = f"thr-mMg vs gamma | fault={fault_label} | site_set={site_set}"
    if thr_none is not None:
        foot += f" | thr-none ref: {Path(thr_none['run_dir']).name}"
    fig.text(
        0.99,
        0.01,
        foot,
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="hellaswag")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--n-total", type=int, default=200)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fault-mode", choices=["fixed", "rand2pow"], required=True)
    ap.add_argument("--fault-delta", type=float, default=None, help="Required when fault-mode=fixed")
    ap.add_argument("--gammas", nargs="+", default=["2", "3", "5", "10"])
    ap.add_argument(
        "--site-set",
        choices=["matmul", "nonmatmul", "merge"],
        default="matmul",
        help="Which sweep CSV to average (default matmul, matches layer-list 0,8,16,24 ACC sweeps).",
    )
    ap.add_argument(
        "--skip-missing-gamma",
        action="store_true",
        help="Skip gammas whose run dir or CSV is missing instead of failing.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="default: <results-root>/<model>_<dataset>/average",
    )
    ap.add_argument("--out-stem", default=None, help="PNG/JSON stem without extension")
    ap.add_argument(
        "--thr-none-run-dir",
        default=None,
        help="Optional thr-none run dir; default: auto-find (gamma-independent, same fault config).",
    )
    ap.add_argument(
        "--no-thr-none",
        action="store_true",
        help="Do not overlay thr-none (no protection) reference line.",
    )
    args = ap.parse_args()

    fault_mode = str(args.fault_mode)
    fault_delta: Optional[float] = float(args.fault_delta) if args.fault_delta is not None else None
    if fault_mode == "fixed" and fault_delta is None:
        ap.error("--fault-delta required when --fault-mode=fixed")

    results_root = Path(args.results_root) if args.results_root else default_results_root()
    gammas = _parse_gammas([str(x) for x in args.gammas])

    rows = collect_series(
        gammas=gammas,
        results_root=results_root,
        model_id=args.model_id,
        dataset=args.dataset,
        n_total=int(args.n_total),
        n_warmup=int(args.n_warmup),
        seed=int(args.seed),
        fault_mode=fault_mode,
        fault_delta=fault_delta if fault_mode == "fixed" else None,
        site_set=str(args.site_set),
        skip_missing=bool(args.skip_missing_gamma),
    )

    thr_none: Optional[Dict[str, Any]] = None
    if not args.no_thr_none:
        thr_run = Path(args.thr_none_run_dir) if args.thr_none_run_dir else find_thr_none_run_dir(
            results_root=results_root,
            model_id=args.model_id,
            dataset=args.dataset,
            n_total=int(args.n_total),
            n_warmup=int(args.n_warmup),
            seed=int(args.seed),
            fault_mode=fault_mode,
            fault_delta=fault_delta if fault_mode == "fixed" else None,
            site_set=str(args.site_set),
        )
        if thr_run is None:
            print("[warn] no thr-none run found for overlay", flush=True)
        else:
            stats = _mean_acc_from_run(thr_run, site_set=str(args.site_set))
            thr_none = {"run_dir": str(thr_run), **stats}
            print(
                json.dumps(
                    {"thr_none_ref": thr_none["run_dir"], "mean_acc_fault": thr_none["mean_acc_fault"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    fault_label = _fault_label(fault_mode, fault_delta)
    out_dir = Path(args.out_dir) if args.out_dir else (
        results_run_dir(
            results_root,
            model_id=args.model_id,
            dataset=args.dataset,
            n_total=int(args.n_total),
            n_warmup=int(args.n_warmup),
            gamma=gammas[0],
            fault_mode=fault_mode,
            seed=int(args.seed),
            fault_delta=fault_delta if fault_mode == "fixed" else None,
        ).parent
        / "average"
    )
    stem = args.out_stem or f"mean_acc_fault_thr-mMg_{fault_label.replace('+', '_')}"
    out_png = out_dir / f"{stem}.png"
    out_json = out_dir / f"{stem}.json"
    out_csv = out_dir / f"{stem}.csv"

    title = f"HellaSwag mean acc_fault vs gamma (thr-mMg, {fault_label}, n{args.n_total} wu{args.n_warmup} s{args.seed})"
    plot_average_acc(
        rows,
        out_png=out_png,
        title=title,
        fault_label=fault_label,
        site_set=str(args.site_set),
        thr_none=thr_none,
    )

    out_rows = [{**r, "thr": "mMg"} for r in rows]
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    payload = {
        "model_id": args.model_id,
        "dataset": args.dataset,
        "fault_mode": fault_mode,
        "fault_delta": fault_delta,
        "site_set": str(args.site_set),
        "gammas_requested": gammas,
        "gammas_plotted": [r["gamma"] for r in rows],
        "thr_mMg_rows": rows,
        "thr_none_ref": thr_none,
        "plot": str(out_png),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"png": str(out_png), "csv": str(out_csv), "json": str(out_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
