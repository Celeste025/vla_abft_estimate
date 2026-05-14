"""Analyze op-output capture JSON: stability of global_min / global_max / weighted_mean across testcases."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any, Dict, List


def _summarize(vals: List[float]) -> Dict[str, float]:
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "mean": float(statistics.mean(vals)),
        "stdev": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
        "range": float(max(vals) - min(vals)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--out-json", default=None, help="default: <in-json> with _analysis suffix")
    ap.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Report top-K sites by combined min/max range across testcases",
    )
    args = ap.parse_args()

    data = json.load(open(args.in_json, encoding="utf-8"))
    testcases: List[Dict[str, Any]] = data["testcases"]

    site_ids: List[str] = []
    if testcases:
        site_ids = sorted(testcases[0]["aggregate_by_site"].keys())

    per_site: Dict[str, Any] = {}
    score_pairs: List[tuple[float, str]] = []
    for sid in site_ids:
        gmins: List[float] = []
        gmaxs: List[float] = []
        means: List[float] = []
        shape_first_set = set()
        for tc in testcases:
            d = tc["aggregate_by_site"].get(sid, {})
            gmins.append(float(d.get("global_min", float("nan"))))
            gmaxs.append(float(d.get("global_max", float("nan"))))
            means.append(float(d.get("weighted_mean", float("nan"))))
            sf = d.get("shape_first")
            if sf is not None:
                shape_first_set.add(json.dumps(sf))

        smin = _summarize(gmins)
        smax = _summarize(gmaxs)
        smean = _summarize(means)
        combined_range = float(smin.get("range", 0.0) + smax.get("range", 0.0))
        score_pairs.append((combined_range, sid))

        per_site[sid] = {
            "global_min_across_testcases": smin,
            "global_max_across_testcases": smax,
            "weighted_mean_across_testcases": smean,
            "combined_min_max_range": combined_range,
            "shape_first_unique_json_count": len(shape_first_set),
            "shape_first_stable": len(shape_first_set) <= 1,
        }

    score_pairs.sort(reverse=True)
    top = [
        {
            "site_id": sid,
            "combined_min_max_range": r,
            "global_min_range": per_site[sid]["global_min_across_testcases"].get("range"),
            "global_max_range": per_site[sid]["global_max_across_testcases"].get("range"),
            "weighted_mean_range": per_site[sid]["weighted_mean_across_testcases"].get("range"),
            "shape_first_stable": per_site[sid]["shape_first_stable"],
        }
        for r, sid in score_pairs[: max(0, int(args.top_k))]
    ]

    summary: Dict[str, Any] = {
        "meta": data.get("meta", {}),
        "n_testcases": len(testcases),
        "n_sites": len(site_ids),
        "per_site": per_site,
        "top_unstable_by_combined_min_max_range": top,
    }

    out_path = args.out_json
    if not out_path:
        if args.in_json.endswith(".json"):
            out_path = args.in_json[:-5] + "_analysis.json"
        else:
            out_path = args.in_json + "_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"wrote": out_path, "n_sites": len(site_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
