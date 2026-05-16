from __future__ import annotations

import argparse
import json

from datasets import load_dataset

from results_layout import default_results_root


def parse_args():
    rr = default_results_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-json", default=str(rr / "gsm8k_test_shared100_indices.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    ds = load_dataset("gsm8k", "main", split=args.split)
    n = min(int(args.n), len(ds))
    ds_shuffled = ds.shuffle(seed=int(args.seed))
    # datasets keeps an indices mapping after shuffle; persist the original row indices
    # so we can re-select the same subset deterministically later.
    if getattr(ds_shuffled, "_indices", None) is None:
        raise RuntimeError("datasets shuffle did not expose _indices mapping; cannot persist stable indices.")
    # ds_shuffled._indices is a small pyarrow Table with a single column `indices`.
    orig_indices = [int(d["indices"]) for d in ds_shuffled._indices[:n].to_pylist()]

    payload = {
        "dataset": "gsm8k",
        "config": "main",
        "split": args.split,
        "seed": int(args.seed),
        "n": n,
        "shuffled_take_first_n": True,
        "indices_kind": "original_row_indices",
        "indices": orig_indices,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(args.out_json)


if __name__ == "__main__":
    main()

