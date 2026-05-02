from __future__ import annotations

import argparse
import json
import sys

from imagenet_task import COMMUNITY_TOP1_TOP5, HF_IMAGENET_DATASET_ID, ImagenetTask
from vision_runner import VisionRunner


def parse_args():
    ap = argparse.ArgumentParser(description="ResNet-50 ImageNet-1k baseline (HF streaming).")
    ap.add_argument("--weights", default="IMAGENET1K_V2", choices=["IMAGENET1K_V1", "IMAGENET1K_V2"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--streaming", action="store_true", default=True)
    ap.add_argument("--no-streaming", action="store_false", dest="streaming")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Random tensors instead of HF imagenet-1k (no login; acc vs community meaningless).",
    )
    ap.add_argument(
        "--hf-dataset-id",
        default=HF_IMAGENET_DATASET_ID,
        help="HuggingFace datasets 仓库 id，默认 ILSVRC/imagenet-1k。",
    )
    ap.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条。")
    ap.add_argument(
        "--local-dataset-dir",
        type=str,
        default=None,
        help="本地 validation（download_imagenet_val.py --out-dir 生成）；指定后不走 Hub 流式。",
    )
    ap.add_argument("--out-json", default="resnet_imagenet_baseline.json")
    return ap.parse_args()


def main():
    args = parse_args()
    runner = VisionRunner(weights=args.weights, device=args.device, dtype=args.dtype)
    task = ImagenetTask(
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        streaming=args.streaming,
        weights_name=args.weights,
        synthetic=args.synthetic,
        hf_dataset_id=args.hf_dataset_id,
        show_progress=not args.no_progress,
        local_dataset_dir=args.local_dataset_dir,
    )
    result = runner.run_task(task, inject_site=None)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    s = result["summary"]
    comm = COMMUNITY_TOP1_TOP5[args.weights]
    print(
        json.dumps(
            {
                "top1_accuracy": s["top1_accuracy"],
                "top5_accuracy": s["top5_accuracy"],
                "community_top1": comm[0],
                "community_top5": comm[1],
                "delta_top1": s["top1_accuracy"] - comm[0],
                "delta_top5": s["top5_accuracy"] - comm[1],
                "total": s["total"],
            },
            ensure_ascii=False,
        )
    )
    print(json.dumps(result["run_meta"], ensure_ascii=False))
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
