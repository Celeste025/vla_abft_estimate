#!/usr/bin/env python3
"""
只下载 ILSVRC/imagenet-1k 的 validation parquet（绝不拉 train）。

- 默认：同步全部 validation 分片（约 14 个 parquet，~5 万条）。
- --max-samples N（如 5000）：按文件名顺序逐个 hf_hub_download 分片，凑够 N 条即停，节省流量与时间。

datasets.load_dataset(repo_id, split=\"validation\") 会先拉完全部 train parquet，请勿那样做。

需要：hf auth login + 数据集网页授权。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from imagenet_task import HF_IMAGENET_DATASET_ID


def parse_args():
    ap = argparse.ArgumentParser(
        description="Download ImageNet-1k validation only (optionally first N samples / shards).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for Dataset.save_to_disk (e.g. ~/data/imagenet1k_val_hf).",
    )
    ap.add_argument("--hf-dataset-id", default=HF_IMAGENET_DATASET_ID)
    ap.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional dataset repo revision (commit hash).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = 下载全部 validation（~50k）；>0 则按分片顺序只下到凑满该条数（如前 5000 张）。",
    )
    return ap.parse_args()


def _list_validation_parquet_paths(repo_id: str, revision: str | None) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    vals: list[str] = []
    for f in files:
        if not f.endswith(".parquet"):
            continue
        name = f.split("/")[-1]
        if name.startswith("validation-"):
            vals.append(f)
    return sorted(vals)


def _collect_validation_parquets(snapshot_root: Path) -> list[str]:
    files = sorted(snapshot_root.rglob("validation-*.parquet"))
    paths = [str(p.resolve()) for p in files if p.is_file()]
    if not paths:
        raise FileNotFoundError(
            f"未找到 validation-*.parquet。snapshot 根目录: {snapshot_root}"
        )
    return paths


def _download_until_n(repo_id: str, revision: str | None, max_samples: int):
    rel_paths = _list_validation_parquet_paths(repo_id, revision)
    if not rel_paths:
        raise RuntimeError(f"Hub 上未列出 validation parquet: {repo_id}")

    chunks = []
    n_rows = 0
    n_files = 0
    for rel in rel_paths:
        print(f"[download_imagenet_val] 下载分片 ({n_files + 1}/{len(rel_paths)}): {rel}", flush=True)
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=rel,
            revision=revision,
        )
        part = load_dataset("parquet", data_files=local_path, split="train")
        chunks.append(part)
        n_rows += len(part)
        n_files += 1
        if n_rows >= max_samples:
            break

    ds = concatenate_datasets(chunks)
    if len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    print(
        f"[download_imagenet_val] 共使用 {n_files} 个 parquet，导出样本数 {len(ds)}（目标 ≤ {max_samples}）",
        flush=True,
    )
    if len(ds) < max_samples:
        print(
            "[download_imagenet_val] 警告: 样本仍少于目标，可能 validation 分片不足。",
            flush=True,
        )
    return ds


def main():
    args = parse_args()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    repo_id = args.hf_dataset_id
    max_samples = int(args.max_samples)

    if max_samples > 0:
        print(
            f"[download_imagenet_val] 按需下载 validation（至多 {max_samples} 条，不拉 train）: {repo_id}",
            flush=True,
        )
        ds = _download_until_n(repo_id, args.revision, max_samples)
    else:
        print(
            f"[download_imagenet_val] 同步全部 validation 分片（不拉 train）: {repo_id}",
            flush=True,
        )
        snapshot_root = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=args.revision,
                allow_patterns=["data/validation-*"],
            )
        )
        parquet_files = _collect_validation_parquets(snapshot_root)
        print(
            f"[download_imagenet_val] 找到 {len(parquet_files)} 个 validation parquet，加载中...",
            flush=True,
        )
        ds = load_dataset("parquet", data_files={"validation": parquet_files})["validation"]
        n = len(ds)
        print(f"[download_imagenet_val] 样本数 {n}（期望约 50000）", flush=True)
        if n < 40_000 or n > 60_000:
            print(
                "[download_imagenet_val] 警告: 样本数异常，请核对数据集版本。",
                flush=True,
            )

    print(f"[download_imagenet_val] save_to_disk -> {out} ...", flush=True)
    ds.save_to_disk(str(out))
    print(f"[download_imagenet_val] 完成。评测: --local-dataset-dir {out}", flush=True)


if __name__ == "__main__":
    main()
