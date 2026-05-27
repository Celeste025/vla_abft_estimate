"""Summarize max fluctuation ratios M_50/M_5 and m_50/m_5 from op_stats capture JSON."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _pct(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = int(round(p * (len(sorted_vals) - 1)))
    return sorted_vals[i]


def _summarize(vals: List[float]) -> Dict[str, Any]:
    v = sorted(x for x in vals if np.isfinite(x))
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "min": float(v[0]),
        "p10": _pct(v, 0.10),
        "p25": _pct(v, 0.25),
        "median": float(statistics.median(v)),
        "mean": float(statistics.mean(v)),
        "p75": _pct(v, 0.75),
        "p90": _pct(v, 0.90),
        "p95": _pct(v, 0.95),
        "max": float(v[-1]),
        "frac_in_2_3": float(sum(1 for x in v if 2.0 <= x <= 3.0) / len(v)),
        "frac_in_2_5_3_5": float(sum(1 for x in v if 2.5 <= x <= 3.5) / len(v)),
    }


def compute_ratios(
    capture: Dict[str, Any], *, k5: int = 5, k50: int = 50, eps: float = 1e-12
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    tcs = capture["testcases"]
    n = len(tcs)
    k50 = min(int(k50), n)
    k5 = min(int(k5), n)
    site_ids = sorted(tcs[0]["aggregate_by_site"].keys())
    rows: List[Dict[str, Any]] = []
    rM: List[float] = []
    rm: List[float] = []
    spread: List[float] = []

    for sid in site_ids:
        maxs = np.array(
            [float(tc["aggregate_by_site"][sid]["global_max"]) for tc in tcs[:k50]],
            dtype=np.float64,
        )
        mins = np.array(
            [float(tc["aggregate_by_site"][sid]["global_min"]) for tc in tcs[:k50]],
            dtype=np.float64,
        )
        M5 = float(np.nanmax(maxs[:k5]))
        M50 = float(np.nanmax(maxs))
        m5 = float(np.nanmin(mins[:k5]))
        m50 = float(np.nanmin(mins))
        ratio_M = M50 / M5 if abs(M5) > eps else float("nan")
        ratio_m = m50 / m5 if abs(m5) > eps else float("nan")
        lo, hi = float(np.nanmin(maxs)), float(np.nanmax(maxs))
        ratio_spread = hi / lo if abs(lo) > eps else float("nan")

        m = re.match(r"^L(\d+)_(.+)$", sid)
        layer = int(m.group(1)) if m else -1
        suffix = m.group(2) if m else sid

        rows.append(
            {
                "site_id": sid,
                "layer": layer,
                "suffix": suffix,
                "M5": M5,
                "M50": M50,
                "M50_over_M5": ratio_M,
                "m5": m5,
                "m50": m50,
                "m50_over_m5": ratio_m,
                "max_over_min_global_max_50tc": ratio_spread,
            }
        )
        if np.isfinite(ratio_M):
            rM.append(ratio_M)
        if np.isfinite(ratio_m):
            rm.append(ratio_m)
        if np.isfinite(ratio_spread):
            spread.append(ratio_spread)

    summary = {
        "meta": capture.get("meta", {}),
        "n_testcases": n,
        "k5": k5,
        "k50": k50,
        "M50_over_M5": _summarize(rM),
        "m50_over_m5": _summarize(rm),
        "max_min_global_max_over_50tc": _summarize(spread),
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--k5", type=int, default=5)
    ap.add_argument("--k50", type=int, default=50)
    args = ap.parse_args()

    capture = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
    rows, summary = compute_ratios(capture, k5=args.k5, k50=args.k50)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps({"summary": summary, "top_M50_over_M5": sorted(rows, key=lambda r: -r["M50_over_M5"])[:20]},
                   ensure_ascii=False,
                   indent=2),
        encoding="utf-8",
    )

    import csv

    fields = list(rows[0].keys()) if rows else []
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({"wrote_json": args.out_json, "wrote_csv": args.out_csv, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
