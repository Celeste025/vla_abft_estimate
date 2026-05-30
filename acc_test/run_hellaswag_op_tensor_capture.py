"""Capture per-forward o_proj (or other attn proj) output tensors on HellaSwag."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset

from model_runner import ModelRunner
from op_tensor_capture import SingleSiteTensorCapture


def _hellaswag_forward_one(runner: ModelRunner, ctx: str, ending: str) -> None:
    tok = runner.tokenizer
    device = runner.device
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    end_ids = tok(" " + ending, add_special_tokens=False)["input_ids"]
    full_ids = ctx_ids + end_ids
    if len(full_ids) < 2 or len(end_ids) == 0:
        return
    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    runner.forward(input_ids=input_ids, attention_mask=attention_mask)


def _tensor_filename(case_idx: int, fwd_idx: int) -> str:
    return f"tc_{case_idx:03d}_fwd_{fwd_idx}.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--site-suffix", default="o_proj")
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--out-dir",
        default="artifacts/hellaswag_o_proj_tensors/L24_n50_s2026",
        help="Output directory for .pt files and meta.json",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    ds = load_dataset("hellaswag", split="validation")
    ds = ds.shuffle(seed=int(args.seed)).select(range(min(int(args.max_samples), len(ds))))

    manifest: List[Dict[str, Any]] = []
    shape_examples: List[List[int]] = []

    with SingleSiteTensorCapture(
        runner.model,
        layer_idx=int(args.layer),
        site_suffix=str(args.site_suffix),
    ) as cap:
        for case_idx, ex in enumerate(ds):
            ctx = ex["ctx"]
            endings = ex["endings"]
            cap.begin_episode()
            for ending in endings:
                _hellaswag_forward_one(runner, ctx, ending)
            tensors = cap.end_episode()

            if len(tensors) != len(endings):
                print(
                    f"[warn] case {case_idx}: expected {len(endings)} tensors, got {len(tensors)}",
                    flush=True,
                )

            for fwd_idx, tensor in enumerate(tensors):
                rec = {
                    "tensor": tensor,
                    "case_idx": int(case_idx),
                    "fwd_idx": int(fwd_idx),
                    "site_id": cap.site_id,
                    "shape": list(tensor.shape),
                }
                path = out_dir / _tensor_filename(case_idx, fwd_idx)
                torch.save(rec, path)
                manifest.append(
                    {
                        "file": path.name,
                        "case_idx": int(case_idx),
                        "fwd_idx": int(fwd_idx),
                        "shape": list(tensor.shape),
                        "numel": int(tensor.numel()),
                    }
                )
                if len(shape_examples) < 4:
                    shape_examples.append(list(tensor.shape))

    meta = {
        "benchmark": "hellaswag",
        "model_id": args.model_id,
        "dtype": args.dtype,
        "layer": int(args.layer),
        "site_id": f"L{int(args.layer)}_{args.site_suffix}",
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
        "n_cases": len(ds),
        "forwards_per_case": 4,
        "n_tensor_files": len(manifest),
        "shape_examples": shape_examples,
        "files": manifest,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "wrote_meta": str(meta_path),
                "out_dir": str(out_dir),
                "n_tensor_files": len(manifest),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
