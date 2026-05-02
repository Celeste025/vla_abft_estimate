from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

import json

from datasets import load_dataset


_RE_HASH_ANSWER = re.compile(r"####\s*([-+]?\d[\d,]*)")
_RE_NUMBER = re.compile(r"[-+]?\d[\d,]*")


def extract_final_answer(text: str) -> Optional[str]:
    """
    GSM8K convention: many solutions use a final line like '#### 42'.
    We first try to extract that. Fallback: take the last integer-like number.
    Returns normalized integer string (commas removed) or None.
    """
    m = _RE_HASH_ANSWER.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = _RE_NUMBER.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def _build_prompt(question: str) -> str:
    # 0-shot CoT with a strict answer-format instruction.
    # Community GSM8K eval commonly extracts the final answer after a `####` delimiter.
    return (
        "Solve the following math word problem.\n\n"
        f"Question: {question}\n\n"
        "Let's think step by step.\n"
        "Give your final answer as a single line in the format:\n"
        "#### <integer>\n"
        "Answer:"
    )


@dataclass
class Gsm8kTask:
    split: str = "test"
    max_samples: int = 16
    seed: int = 2026
    max_new_tokens: int = 512
    require_hash_answer: bool = True
    decode_step_inject_enable: bool = False
    decode_step_max: int = 150
    decode_step_min: int = 0
    indices: Optional[List[int]] = None
    raw_generation_char_limit: int = 2000
    trace_jsonl_path: Optional[str] = None
    trace_run_tag: str = ""

    def run(self, runner) -> Dict[str, Any]:
        ds = load_dataset("gsm8k", "main", split=self.split)
        if self.indices is not None:
            ds = ds.select([int(i) for i in self.indices])
        elif self.max_samples > 0:
            ds = ds.shuffle(seed=self.seed).select(range(min(self.max_samples, len(ds))))

        per_example: List[Dict[str, Any]] = []
        correct = 0
        pred_none = 0
        pred_format_fail = 0

        for idx, ex in enumerate(ds):
            q = ex["question"]
            gold_text = ex["answer"]
            gold = extract_final_answer(gold_text)

            prompt = _build_prompt(q)
            gen = runner.generate_text(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            inj = getattr(runner, "_active_injector", None)
            decode_target_step = None
            decode_injected = None
            decode_injected_step = None
            if inj is not None and getattr(inj, "decode_step_inject_enable", False):
                decode_target_step = inj.get_decode_target_step()
                decode_injected = inj.get_decode_injected()
                decode_injected_step = inj.get_decode_injected_step()
            if self.require_hash_answer:
                m = _RE_HASH_ANSWER.search(gen)
                if m:
                    pred = m.group(1).replace(",", "")
                else:
                    pred = None
                    pred_format_fail += 1
            else:
                pred = extract_final_answer(gen)
            if pred is None:
                pred_none += 1

            is_correct = int(gold is not None and pred is not None and pred == gold)
            correct += is_correct
            row = {
                "idx": idx,
                "question": q,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "raw_generation": gen if self.raw_generation_char_limit <= 0 else gen[: self.raw_generation_char_limit],
                "decode_target_step": decode_target_step,
                "decode_injected": decode_injected,
                "decode_injected_step": decode_injected_step,
            }
            if self.trace_run_tag:
                row["run_tag"] = self.trace_run_tag
            per_example.append(row)
            if self.trace_jsonl_path:
                with open(self.trace_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            # Progress: print every 10 problems for long runs.
            cur = idx + 1
            if cur % 10 == 0:
                acc_so_far = float(correct) / float(cur) if cur else 0.0
                hit_info = ""
                if inj is not None and getattr(inj, "decode_step_inject_enable", False):
                    st = inj.collect_hook_stats()
                    dp = int(getattr(st, "decode_problem_count", 0))
                    di = int(getattr(st, "decode_injected_problem_count", 0))
                    hr = float(di) / float(dp) if dp else 0.0
                    hit_info = f" hit={di}/{dp}({hr:.3f})"
                print(
                    f"[gsm8k {cur}/{len(ds)}] acc_so_far={acc_so_far:.4f} "
                    f"format_fail={pred_format_fail} pred_none={pred_none}{hit_info}",
                    flush=True,
                )

        total = len(per_example)
        acc = float(correct) / float(total) if total else 0.0
        return {
            "benchmark": "gsm8k",
            "mode": "generate",
            "summary": {
                "total": total,
                "correct": correct,
                "accuracy": acc,
                "pred_none": pred_none,
                    "pred_format_fail": pred_format_fail,
            },
            "per_example": per_example,
        }

