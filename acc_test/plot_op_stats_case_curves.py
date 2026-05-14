"""Plot per-site global_min / global_max vs testcase index from op_stats capture JSON.

Reads a single capture file (not *_analysis.json) under --data-dir, writes PNGs to
``<data-dir>/plots/``: 5 random sites, 2 smallest-fluctuation sites, 2 largest-fluctuation
sites (fluctuation = ptp(mins) + ptp(maxs) across testcases, same spirit as analysis).
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


def _find_capture_json(data_dir: str) -> str:
    paths = sorted(
        p
        for p in glob.glob(os.path.join(data_dir, "*.json"))
        if "analysis" not in os.path.basename(p).lower()
    )
    if not paths:
        raise FileNotFoundError(f"no capture JSON (exclude *analysis*) in {data_dir!r}")
    if len(paths) > 1:
        raise RuntimeError(f"multiple capture JSON in {data_dir}: {paths}")
    return paths[0]


def _load_series(
    path: str,
) -> Tuple[List[str], Dict[str, Tuple[List[float], List[float]]], int]:
    data = json.load(open(path, encoding="utf-8"))
    tcs: List[Dict[str, Any]] = data["testcases"]
    if not tcs:
        raise ValueError("empty testcases")
    site_ids = sorted(tcs[0]["aggregate_by_site"].keys())
    series: Dict[str, Tuple[List[float], List[float]]] = {}
    for sid in site_ids:
        mins: List[float] = []
        maxs: List[float] = []
        for tc in tcs:
            d = tc["aggregate_by_site"].get(sid, {})
            mins.append(float(d.get("global_min", float("nan"))))
            maxs.append(float(d.get("global_max", float("nan"))))
        series[sid] = (mins, maxs)
    x = list(range(len(tcs)))
    return site_ids, series, len(tcs)


def _fluctuation(mins: List[float], maxs: List[float]) -> float:
    import math

    mns = [v for v in mins if not math.isnan(v)]
    mxs = [v for v in maxs if not math.isnan(v)]
    if not mns or not mxs:
        return float("inf")
    return float(max(mns) - min(mns) + max(mxs) - min(mxs))


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", s)


def _plot_one(
    out_path: str,
    site_id: str,
    x: List[int],
    mins: List[float],
    maxs: List[float],
    tag: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, mins, "o-", label="global_min", markersize=3, linewidth=1.2)
    ax.plot(x, maxs, "s-", label="global_max", markersize=3, linewidth=1.2)
    ax.set_xlabel("testcase index")
    ax.set_ylabel("value")
    ax.set_title(f"{tag}: {site_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        required=True,
        help="Folder containing one op_stats capture JSON (e.g. artifacts/hellaswag_n50)",
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--random-k",
        type=int,
        default=5,
        help="number of random sites to plot",
    )
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    cap_path = _find_capture_json(data_dir)
    site_ids, series, n = _load_series(cap_path)

    flucs: List[Tuple[float, str]] = []
    for sid in site_ids:
        mins, maxs = series[sid]
        flucs.append((_fluctuation(mins, maxs), sid))
    flucs.sort(key=lambda t: t[0])

    smallest = [flucs[0][1], flucs[1][1]]
    largest = [flucs[-1][1], flucs[-2][1]]

    rng = random.Random(int(args.seed))
    pool = [s for s in site_ids if s not in set(smallest + largest)]
    k = min(int(args.random_k), len(pool))
    random_sites = rng.sample(pool, k=k)

    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    x = list(range(n))
    manifest: List[Dict[str, Any]] = []

    def do_plot(tag: str, sid: str, fname: str) -> None:
        mins, maxs = series[sid]
        out_path = os.path.join(plots_dir, fname)
        _plot_one(out_path, sid, x, mins, maxs, tag)
        manifest.append({"tag": tag, "site_id": sid, "path": out_path, "fluctuation": _fluctuation(mins, maxs)})

    for i, sid in enumerate(random_sites, start=1):
        do_plot("random", sid, f"01_random_{i:02d}_{_safe_name(sid)}.png")

    do_plot("least_fluctuation", smallest[0], f"02_least_fluct_01_{_safe_name(smallest[0])}.png")
    do_plot("least_fluctuation", smallest[1], f"03_least_fluct_02_{_safe_name(smallest[1])}.png")
    do_plot("most_fluctuation", largest[0], f"04_most_fluct_01_{_safe_name(largest[0])}.png")
    do_plot("most_fluctuation", largest[1], f"05_most_fluct_02_{_safe_name(largest[1])}.png")

    meta = {
        "capture_json": cap_path,
        "n_testcases": n,
        "seed": int(args.seed),
        "random_sites": random_sites,
        "least_fluctuation": [{"site_id": s, "fluctuation": _fluctuation(*series[s])} for s in smallest],
        "most_fluctuation": [{"site_id": s, "fluctuation": _fluctuation(*series[s])} for s in largest],
        "plots": manifest,
    }
    with open(os.path.join(plots_dir, "plot_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps({"plots_dir": plots_dir, "n_plots": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
