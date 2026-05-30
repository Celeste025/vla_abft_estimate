"""HellaSwag: inject at L{layer}_mlp_down (fixed), capture per-layer hidden states (clean vs fault)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset

from inject import InjectionContext, SiteSet
from layer_hidden_state_capture import LayerHiddenStateCapture
from model_runner import ModelRunner
from results_layout import dataset_slug, model_slug_from_id


def _build_inputs(
    runner: ModelRunner, ctx: str, ending: str
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    tok = runner.tokenizer
    device = runner.device
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    end_ids = tok(" " + ending, add_special_tokens=False)["input_ids"]
    full_ids = ctx_ids + end_ids
    if len(full_ids) < 2 or len(end_ids) == 0:
        raise ValueError("prompt too short for scoring forward")
    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    meta = {
        "seq_len": len(full_ids),
        "ctx_token_len": len(ctx_ids),
        "ending_token_len": len(end_ids),
    }
    return input_ids, attention_mask, meta


def _save_layers(out_dir: Path, layers: Dict[int, torch.Tensor]) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for i in sorted(layers.keys()):
        t = layers[i]
        name = f"L{i:02d}.pt"
        path = out_dir / name
        torch.save({"layer_idx": i, "tensor": t, "shape": list(t.shape)}, path)
        manifest.append({"file": name, "layer_idx": i, "shape": list(t.shape), "numel": int(t.numel())})
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--case-idx", type=int, default=0)
    ap.add_argument("--ending-idx", type=int, default=0)
    ap.add_argument("--inject-layer", type=int, default=0)
    ap.add_argument(
        "--inject-site",
        choices=["mlp_down", "mlp_residual"],
        default="mlp_down",
        help="mlp_down: down_proj output; mlp_residual: down_proj+residual (layer output)",
    )
    ap.add_argument("--fault-delta", type=float, default=100.0)
    ap.add_argument("--fault-index-mode", choices=["random", "max_abs"], default="random")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="default: artifacts/.../case{C}_end{E}_L{L}_{site}_fd{delta}",
    )
    args = ap.parse_args()

    case_idx = int(args.case_idx)
    ending_idx = int(args.ending_idx)
    inject_layer = int(args.inject_layer)
    inject_site = str(args.inject_site)
    fault_delta = float(args.fault_delta)
    target_site = f"L{inject_layer}_{inject_site}"
    site_set: SiteSet = "matmul" if inject_site == "mlp_down" else "nonmatmul"

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).resolve().parent
        / "artifacts"
        / "hellaswag_layer_hidden_diff"
        / f"case{case_idx}_end{ending_idx}_L{inject_layer}_{inject_site}_fd{fault_delta:g}"
    )
    clean_dir = out_dir / "clean"
    fault_dir = out_dir / "fault"

    runner = ModelRunner(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    ds = load_dataset("hellaswag", split="validation")
    ds = ds.shuffle(seed=int(args.seed))
    if case_idx < 0 or case_idx >= len(ds):
        raise IndexError(f"case_idx={case_idx} out of range [0, {len(ds)})")
    ex = ds[case_idx]
    endings: List[str] = list(ex["endings"])
    if ending_idx < 0 or ending_idx >= len(endings):
        raise IndexError(f"ending_idx={ending_idx} out of range [0, {len(endings)})")

    ctx = ex["ctx"]
    ending = endings[ending_idx]
    input_ids, attention_mask, tok_meta = _build_inputs(runner, ctx, ending)

    num_layers = int(runner.model.config.num_hidden_layers)

    # --- clean run (no injection hooks) ---
    with LayerHiddenStateCapture(runner.model) as cap:
        cap.begin_episode()
        runner.forward(input_ids=input_ids, attention_mask=attention_mask)
        clean_layers = cap.end_episode()
    clean_manifest = _save_layers(clean_dir, clean_layers)

    # --- fault run ---
    inject_count = 0
    last_flat_idx = None
    with LayerHiddenStateCapture(runner.model) as cap:
        with InjectionContext(
            runner.model,
            target_site=target_site,
            fault_mode="fixed",
            fault_delta=fault_delta,
            seed=int(args.seed),
            fault_index_mode=str(args.fault_index_mode),
            threshold_enable=False,
            inject_enable=True,
            site_set=site_set,
        ) as inj:
            cap.begin_episode()
            runner.forward(input_ids=input_ids, attention_mask=attention_mask)
            fault_layers = cap.end_episode()
            inject_count = inj.inject_count
            last_flat_idx = inj.last_inject_flat_idx

    fault_manifest = _save_layers(fault_dir, fault_layers)

    payload: Dict[str, Any] = {
        "model_id": args.model_id,
        "benchmark": "hellaswag",
        "case_idx": case_idx,
        "ending_idx": ending_idx,
        "label": int(ex["label"]),
        "ctx_preview": ctx[:200],
        "ending": ending,
        "target_site": target_site,
        "inject_layer": inject_layer,
        "inject_site": inject_site,
        "site_set": site_set,
        "fault_mode": "fixed",
        "fault_delta": fault_delta,
        "fault_index_mode": args.fault_index_mode,
        "threshold_enable": False,
        "inject_count": inject_count,
        "last_inject_flat_idx": last_flat_idx,
        "num_layers": num_layers,
        "token_meta": tok_meta,
        "clean_dir": str(clean_dir),
        "fault_dir": str(fault_dir),
        "clean_manifest": clean_manifest,
        "fault_manifest": fault_manifest,
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "meta": str(meta_path),
                "inject_count": inject_count,
                "last_inject_flat_idx": last_flat_idx,
                "num_layers": num_layers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
