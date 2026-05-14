"""Join two analyze_op_output_stats.py outputs on site_id (e.g. gsm8k n5 vs n50)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-json", required=True, help="First *_analysis.json")
    ap.add_argument("--b-json", required=True, help="Second *_analysis.json")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    da = json.load(open(args.a_json, encoding="utf-8"))
    db = json.load(open(args.b_json, encoding="utf-8"))
    pa, pb = da["per_site"], db["per_site"]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sites = sorted(set(pa.keys()) & set(pb.keys()))
    fields = [
        "site_id",
        "a_combined_range",
        "b_combined_range",
        "delta_combined_range",
        "a_min_range",
        "b_min_range",
        "a_max_range",
        "b_max_range",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sid in sites:
            ca = pa[sid]["combined_min_max_range"]
            cb = pb[sid]["combined_min_max_range"]
            w.writerow(
                {
                    "site_id": sid,
                    "a_combined_range": ca,
                    "b_combined_range": cb,
                    "delta_combined_range": float(cb) - float(ca),
                    "a_min_range": pa[sid]["global_min_across_testcases"].get("range"),
                    "b_min_range": pb[sid]["global_min_across_testcases"].get("range"),
                    "a_max_range": pa[sid]["global_max_across_testcases"].get("range"),
                    "b_max_range": pb[sid]["global_max_across_testcases"].get("range"),
                }
            )

    print(json.dumps({"wrote": str(out_path), "n_sites": len(sites)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
