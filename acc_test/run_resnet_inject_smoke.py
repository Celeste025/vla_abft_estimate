from __future__ import annotations

import argparse
import json

from imagenet_task import HF_IMAGENET_DATASET_ID, ImagenetTask
from vision_runner import VisionRunner


def parse_args():
    ap = argparse.ArgumentParser(description="ResNet fault inject smoke: baseline / fault / fault+protect.")
    ap.add_argument("--weights", default="IMAGENET1K_V2", choices=["IMAGENET1K_V1", "IMAGENET1K_V2"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--site", default="layer3.0.conv2", help="Module path from named_modules(), e.g. layer3.0.conv2")
    ap.add_argument("--fault-delta", type=float, default=10000.0)
    ap.add_argument(
        "--fault-index-mode",
        default="random",
        choices=["random", "max_abs"],
    )
    ap.add_argument("--clear-threshold-mul", type=float, default=0.5)
    ap.add_argument("--out-json", default=None, help="If set, write all three runs to this JSON file.")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random tensors instead of HF imagenet-1k.",
    )
    ap.add_argument("--hf-dataset-id", default=HF_IMAGENET_DATASET_ID, help="HF datasets id, default ILSVRC/imagenet-1k.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    ap.add_argument(
        "--local-dataset-dir",
        type=str,
        default=None,
        help="Local validation from download_imagenet_val.py (no Hub streaming).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    runner = VisionRunner(weights=args.weights, device=args.device, dtype=args.dtype)
    task = ImagenetTask(
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        streaming=True,
        weights_name=args.weights,
        synthetic=args.synthetic,
        hf_dataset_id=args.hf_dataset_id,
        show_progress=not args.no_progress,
        local_dataset_dir=args.local_dataset_dir,
    )

    baseline = runner.run_task(task, inject_site=None)
    fault = runner.run_task(
        task,
        inject_site=args.site,
        fault_delta=args.fault_delta,
        seed=args.seed,
        fault_index_mode=args.fault_index_mode,
        clear_exceptions=False,
    )
    protect = runner.run_task(
        task,
        inject_site=args.site,
        fault_delta=args.fault_delta,
        seed=args.seed,
        fault_index_mode=args.fault_index_mode,
        clear_exceptions=True,
        clear_threshold_mul=args.clear_threshold_mul,
    )

    out = {"baseline": baseline, "fault": fault, "fault_protect": protect}
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def brief(tag: str, r: dict) -> None:
        sm = r["summary"]
        meta = r.get("run_meta", {})
        print(
            json.dumps(
                {
                    "tag": tag,
                    "top1": sm["top1_accuracy"],
                    "top5": sm["top5_accuracy"],
                    "inject_count": meta.get("inject_count"),
                    "errors_total": meta.get("errors_total"),
                },
                ensure_ascii=False,
            )
        )

    brief("baseline", baseline)
    brief("fault", fault)
    brief("fault+protect", protect)


if __name__ == "__main__":
    main()
