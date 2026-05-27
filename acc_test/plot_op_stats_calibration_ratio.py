"""Ratio curves: M_all/M and m_all/m vs testcase index (one operator per figure).

- From the first ``--calib-k`` testcases (default 5), per site:
    M = max_t global_max[t]   (t in [0, calib_k))
    m = min_t global_min[t]
- For testcase index i = 0..N-1 (x-axis):
    M_all(i) = max_{t <= i} global_max[t]
    m_all(i) = min_{t <= i} global_min[t]
    r_M(i) = M_all(i) / M,  r_m(i) = m_all(i) / m  (skip if |M| or |m| < eps)

Default output (9 PNGs under ``<data-dir>/plots/``):
  - 5 random sites (seed ``--seed``), excluding stable/unstable picks
  - 2 sites with **smallest** ratio-curve fluctuation:
        ptp(r_M) + ptp(r_m)  over testcase index (finite values only)
  - 2 sites with **largest** fluctuation

Optional: ``--include-median`` adds one aggregate figure (nan-median across sites).
Optional: ``--sites a b c`` plots only those sites (no random/stable/unstable batch).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _find_capture_json(data_dir: str) -> str:
    def _is_capture(path: str) -> bool:
        b = os.path.basename(path).lower()
        if "analysis" in b:
            return False
        if b.startswith("m_ratio") or b.endswith("_summary.json") or "meta_calib" in b:
            return False
        return True

    paths = sorted(p for p in glob.glob(os.path.join(data_dir, "*.json")) if _is_capture(p))
    preferred = sorted(p for p in paths if os.path.basename(p).startswith("op_stats"))
    if preferred:
        paths = preferred
    if not paths:
        raise FileNotFoundError(f"no capture JSON in {data_dir!r}")
    if len(paths) > 1:
        raise RuntimeError(f"multiple capture JSON: {paths}")
    return paths[0]


def _load(data_path: str) -> Tuple[List[str], int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    data = json.load(open(data_path, encoding="utf-8"))
    tcs: List[Dict[str, Any]] = data["testcases"]
    n = len(tcs)
    if n == 0:
        raise ValueError("empty testcases")
    site_ids = sorted(tcs[0]["aggregate_by_site"].keys())
    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid in site_ids:
        mins = np.empty(n, dtype=np.float64)
        maxs = np.empty(n, dtype=np.float64)
        for i, tc in enumerate(tcs):
            d = tc["aggregate_by_site"].get(sid, {})
            mins[i] = float(d.get("global_min", float("nan")))
            maxs[i] = float(d.get("global_max", float("nan")))
        series[sid] = (mins, maxs)
    return site_ids, n, series


def _cummax_max(mins: np.ndarray, maxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m_all = np.minimum.accumulate(mins)
    M_all = np.maximum.accumulate(maxs)
    return m_all, M_all


def _ratios_for_site(
    mins: np.ndarray,
    maxs: np.ndarray,
    calib_k: int,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    k = min(int(calib_k), len(mins))
    if k < 1:
        raise ValueError("calib_k must be >= 1")
    M_ref = float(np.nanmax(maxs[:k]))
    m_ref = float(np.nanmin(mins[:k]))
    m_all, M_all = _cummax_max(mins, maxs)
    r_M = np.full_like(maxs, np.nan, dtype=np.float64)
    r_m = np.full_like(mins, np.nan, dtype=np.float64)
    if abs(M_ref) > eps:
        r_M = M_all / M_ref
    if abs(m_ref) > eps:
        r_m = m_all / m_ref
    return r_M, r_m, M_ref, m_ref


def _ratio_fluctuation(r_M: np.ndarray, r_m: np.ndarray) -> float:
    a = r_M[np.isfinite(r_M)]
    b = r_m[np.isfinite(r_m)]
    p1 = float(np.ptp(a)) if a.size else 0.0
    p2 = float(np.ptp(b)) if b.size else 0.0
    return p1 + p2


def _nanmedian_rows(r_mat: np.ndarray) -> np.ndarray:
    out = np.empty(r_mat.shape[1], dtype=np.float64)
    for j in range(r_mat.shape[1]):
        col = r_mat[:, j]
        col = col[np.isfinite(col)]
        out[j] = float(np.nanmedian(col)) if col.size else float("nan")
    return out


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", s)


def _plot_ratio(
    out_path: str,
    sid: str,
    x: np.ndarray,
    r_M: np.ndarray,
    r_m: np.ndarray,
    M_ref: float,
    m_ref: float,
    calib_k: int,
    tag: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, r_M, "o-", label="M_all/M", markersize=3, linewidth=1.2)
    ax.plot(x, r_m, "s-", label="m_all/m", markersize=3, linewidth=1.2)
    ax.axvline(calib_k - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label=f"calib k={calib_k}")
    ax.set_xlabel("testcase index")
    ax.set_ylabel("ratio")
    ax.set_title(f"{tag}: {sid}  M_ref={M_ref:g}  m_ref={m_ref:g}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--calib-k", type=int, default=5)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--random-k", type=int, default=5)
    ap.add_argument(
        "--include-median",
        action="store_true",
        help="Also write nan-median aggregate across sites (one extra PNG)",
    )
    ap.add_argument(
        "--sites",
        nargs="*",
        default=None,
        help="If set: only plot these site_ids (ignores random/stable/unstable batch)",
    )
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    cap = _find_capture_json(data_dir)
    site_ids, n, series = _load(cap)
    calib_k = int(args.calib_k)
    eps = float(args.eps)
    x = np.arange(n)
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ratio_by_site: Dict[str, Tuple[np.ndarray, np.ndarray, float, float]] = {}
    fluct: List[Tuple[float, str]] = []
    for sid in site_ids:
        mins, maxs = series[sid]
        r_M, r_m, M_ref, m_ref = _ratios_for_site(mins, maxs, calib_k, eps)
        ratio_by_site[sid] = (r_M, r_m, M_ref, m_ref)
        fluct.append((_ratio_fluctuation(r_M, r_m), sid))
    fluct.sort(key=lambda t: t[0])

    meta: Dict[str, Any] = {
        "capture_json": cap,
        "calib_k": calib_k,
        "n_testcases": n,
        "n_sites": len(site_ids),
        "seed": int(args.seed),
    }
    manifest: List[Dict[str, Any]] = []

    if args.sites:
        for sid in args.sites:
            sid = str(sid)
            if sid not in ratio_by_site:
                raise KeyError(f"unknown site_id={sid!r}")
            r_M, r_m, M_ref, m_ref = ratio_by_site[sid]
            fn = f"ratio_calib{calib_k}_manual_{_safe_name(sid)}.png"
            outp = os.path.join(plots_dir, fn)
            _plot_ratio(outp, sid, x, r_M, r_m, M_ref, m_ref, calib_k, "manual")
            manifest.append(
                {
                    "tag": "manual",
                    "site_id": sid,
                    "path": outp,
                    "ratio_fluctuation": _ratio_fluctuation(r_M, r_m),
                    "M_ref": M_ref,
                    "m_ref": m_ref,
                }
            )
        meta["plots"] = manifest
        with open(os.path.join(plots_dir, f"ratio_calibration_meta_calib{calib_k}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(json.dumps({"plots_dir": plots_dir, "n_plots": len(manifest)}, ensure_ascii=False))
        return

    smallest = [fluct[0][1], fluct[1][1]]
    largest = [fluct[-1][1], fluct[-2][1]]
    reserved = set(smallest + largest)
    rng = random.Random(int(args.seed))
    pool = [s for s in site_ids if s not in reserved]
    rk = min(int(args.random_k), len(pool))
    random_sites = rng.sample(pool, k=rk)

    def emit(prefix: str, tag: str, sid: str) -> None:
        r_M, r_m, M_ref, m_ref = ratio_by_site[sid]
        outp = os.path.join(plots_dir, f"{prefix}_{_safe_name(sid)}.png")
        _plot_ratio(outp, sid, x, r_M, r_m, M_ref, m_ref, calib_k, tag)
        manifest.append(
            {
                "tag": tag,
                "site_id": sid,
                "path": outp,
                "ratio_fluctuation": _ratio_fluctuation(r_M, r_m),
                "M_ref": M_ref,
                "m_ref": m_ref,
            }
        )

    for i, sid in enumerate(random_sites, start=1):
        emit(f"ratio_calib{calib_k}_01_random_{i:02d}", "random", sid)
    emit(f"ratio_calib{calib_k}_02_stable_01", "most_stable_ratio", smallest[0])
    emit(f"ratio_calib{calib_k}_03_stable_02", "most_stable_ratio", smallest[1])
    emit(f"ratio_calib{calib_k}_04_unstable_01", "most_unstable_ratio", largest[0])
    emit(f"ratio_calib{calib_k}_05_unstable_02", "most_unstable_ratio", largest[1])

    meta["random_sites"] = random_sites
    meta["most_stable_ratio"] = [
        {
            "site_id": s,
            "ratio_fluctuation": _ratio_fluctuation(ratio_by_site[s][0], ratio_by_site[s][1]),
        }
        for s in smallest
    ]
    meta["most_unstable_ratio"] = [
        {
            "site_id": s,
            "ratio_fluctuation": _ratio_fluctuation(ratio_by_site[s][0], ratio_by_site[s][1]),
        }
        for s in largest
    ]

    if args.include_median:
        rM_stack = [ratio_by_site[s][0] for s in site_ids]
        rm_stack = [ratio_by_site[s][1] for s in site_ids]
        rM_mat = np.stack(rM_stack, axis=0)
        rm_mat = np.stack(rm_stack, axis=0)
        med_rM = _nanmedian_rows(rM_mat)
        med_rm = _nanmedian_rows(rm_mat)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, med_rM, "o-", label="median(M_all/M)", markersize=3, linewidth=1.2)
        ax.plot(x, med_rm, "s-", label="median(m_all/m)", markersize=3, linewidth=1.2)
        ax.axvline(calib_k - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("testcase index")
        ax.set_ylabel("ratio")
        ax.set_title(f"nan-median across {len(site_ids)} sites (optional)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_med = os.path.join(plots_dir, f"ratio_calib{calib_k}_00_median_aggregate.png")
        fig.savefig(out_med, dpi=150)
        plt.close(fig)
        meta["plot_median_aggregate"] = out_med
        meta["median_rM"] = med_rM.tolist()
        meta["median_rm"] = med_rm.tolist()

    meta["plots"] = manifest
    with open(os.path.join(plots_dir, f"ratio_calibration_meta_calib{calib_k}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({"plots_dir": plots_dir, "n_plots": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
