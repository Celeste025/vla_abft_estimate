from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset
import torch
import torch.nn.functional as F


@dataclass
class HellaSwagTask:
    split: str = "validation"
    max_samples: int = 100
    seed: int = 2026

    def run(self, runner) -> Dict[str, Any]:
        ds = load_dataset("hellaswag", split=self.split)
        if self.max_samples > 0:
            ds = ds.shuffle(seed=self.seed).select(range(min(self.max_samples, len(ds))))

        per_example: List[Dict[str, Any]] = []
        correct = 0

        for ex in ds:
            ctx = ex["ctx"]
            endings = ex["endings"]
            label = int(ex["label"])

            scores = []
            for ending in endings:
                score = self._score_choice(runner, ctx, ending)
                scores.append(score)

            pred = int(max(range(len(scores)), key=lambda i: scores[i]))
            correct += int(pred == label)
            per_example.append(
                {
                    "ind": ex["ind"],
                    "label": label,
                    "pred": pred,
                    "correct": int(pred == label),
                    "scores": scores,
                }
            )

        total = len(per_example)
        acc = float(correct) / float(total) if total > 0 else 0.0
        return {
            "benchmark": "hellaswag",
            "mode": "prefill",
            "summary": {"total": total, "correct": correct, "accuracy": acc},
            "per_example": per_example,
        }

    def _score_choice(self, runner, ctx: str, ending: str) -> float:
        tok = runner.tokenizer
        device = runner.device

        ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        end_ids = tok(" " + ending, add_special_tokens=False)["input_ids"]
        full_ids = ctx_ids + end_ids
        if len(full_ids) < 2 or len(end_ids) == 0:
            return -1e9

        input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        logits = runner.forward(input_ids=input_ids, attention_mask=attention_mask)[0]

        # logits[t] predicts token at t+1.
        log_probs = F.log_softmax(logits[:-1], dim=-1)
        target = input_ids[0, 1:]

        start = max(len(ctx_ids) - 1, 0)
        end = start + len(end_ids)
        token_logp = log_probs[start:end, :].gather(1, target[start:end].unsqueeze(-1)).squeeze(-1)
        return float(token_logp.sum().item())


def attach_delta_scores(
    baseline_result: Dict[str, Any], fault_result: Dict[str, Any]
) -> Dict[str, Any]:
    base_map = {x["ind"]: x for x in baseline_result["per_example"]}
    merged = []
    for fx in fault_result["per_example"]:
        bx = base_map[fx["ind"]]
        choices = []
        for sb, sf in zip(bx["scores"], fx["scores"]):
            choices.append(
                {
                    "score_baseline": sb,
                    "score_fault": sf,
                    "delta_score": sf - sb,
                }
            )
        merged.append(
            {
                "ind": fx["ind"],
                "label": fx["label"],
                "pred_baseline": bx["pred"],
                "pred_fault": fx["pred"],
                "correct_baseline": bx["correct"],
                "correct_fault": fx["correct"],
                "choices": choices,
            }
        )
    return {
        "benchmark": "hellaswag",
        "mode": "prefill",
        "baseline_summary": baseline_result["summary"],
        "fault_summary": fault_result["summary"],
        "run_meta_baseline": baseline_result.get("run_meta", {}),
        "run_meta_fault": fault_result.get("run_meta", {}),
        "per_example": merged,
    }
