"""GSM8K latency task: measure latency on a few problems, bs=1.

Default (`latency_timing="hooks"`): wraps each `runner.generate_text(...)` with
`LatencyHook` on the root model so the first forward is prefill and the rest are
decode steps.

`latency_timing="e2e_generate"`: cuda.synchronize + wall time around the entire
`generate_text` call (used with `torch.compile`, where root forward hooks are not
reliable inside HF `generate`). Reports total generation time only, not prefill vs decode.
Accuracy is computed for sanity, but not reported as the main result.
"""
from __future__ import annotations

import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import torch
from datasets import load_dataset

from gsm8k_task import _RE_HASH_ANSWER, _build_prompt, extract_final_answer
from model_runner import LatencyHook


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


@dataclass
class Gsm8kLatencyTask:
    split: str = "test"
    max_samples: int = 8
    seed: int = 2026
    max_new_tokens: int = 256
    require_hash_answer: bool = True
    warmup_samples: int = 1
    raw_generation_char_limit: int = 1000
    latency_timing: Literal["hooks", "e2e_generate"] = "hooks"

    def run(self, runner) -> Dict[str, Any]:
        ds = load_dataset("gsm8k", "main", split=self.split)
        n_take = max(0, int(self.max_samples) + max(0, int(self.warmup_samples)))
        ds = ds.shuffle(seed=self.seed).select(range(min(n_take, len(ds))))

        per_example: List[Dict[str, Any]] = []
        prefill_ms: List[float] = []
        decode_ms_per_token: List[float] = []
        decode_ms_means_per_problem: List[float] = []
        generate_total_ms: List[float] = []
        correct = 0
        total = 0
        warmup_done = 0

        for idx, ex in enumerate(ds):
            is_warmup = warmup_done < int(self.warmup_samples)
            q = ex["question"]
            gold = extract_final_answer(ex["answer"])
            prompt = _build_prompt(q)

            if self.latency_timing == "hooks":
                hook_ctx = LatencyHook(runner.model)
            else:
                hook_ctx = nullcontext(None)

            with hook_ctx as lh:
                if self.latency_timing == "e2e_generate":
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0_ns = time.perf_counter_ns()
                    gen = runner.generate_text(
                        prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    generate_total_ns = int(time.perf_counter_ns() - t0_ns)
                else:
                    gen = runner.generate_text(
                        prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                    )
                    generate_total_ns = 0

            prompt_tokens = int(len(runner.tokenizer(prompt)["input_ids"]))
            gen_tokens = int(len(runner.tokenizer(gen, add_special_tokens=False)["input_ids"]))
            if self.latency_timing == "hooks" and lh is not None:
                prefill_ns = int(lh.prefill_ns) if lh.prefill_ns is not None else 0
                decode_ns_list = list(lh.decode_ns_list)
                decode_total_ns = int(sum(decode_ns_list))
                decode_steps = int(len(decode_ns_list))
                decode_per_token_ns = (decode_total_ns / decode_steps) if decode_steps > 0 else 0.0
            else:
                prefill_ns = 0
                decode_ns_list = []
                decode_total_ns = 0
                decode_steps = int(gen_tokens)
                decode_per_token_ns = 0.0

            if self.require_hash_answer:
                m = _RE_HASH_ANSWER.search(gen)
                pred = m.group(1).replace(",", "") if m else None
            else:
                pred = extract_final_answer(gen)
            is_correct = int(gold is not None and pred is not None and pred == gold)

            row = {
                "idx": idx,
                "is_warmup": int(is_warmup),
                "question": q,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "raw_generation": gen[: self.raw_generation_char_limit] if self.raw_generation_char_limit > 0 else gen,
                "prompt_tokens": prompt_tokens,
                "gen_tokens": gen_tokens,
                "decode_steps": decode_steps,
                "prefill_ns": prefill_ns,
                "decode_total_ns": decode_total_ns,
                "decode_per_token_ns": decode_per_token_ns,
                "prefill_ms": prefill_ns / 1e6,
                "decode_total_ms": decode_total_ns / 1e6,
                "decode_per_token_ms": decode_per_token_ns / 1e6,
                "generate_total_ns": int(generate_total_ns),
                "generate_total_ms": generate_total_ns / 1e6,
                "latency_timing": self.latency_timing,
            }
            per_example.append(row)

            if is_warmup:
                warmup_done += 1
                if self.latency_timing == "e2e_generate":
                    print(
                        f"[gsm8k-latency warmup {warmup_done}/{int(self.warmup_samples)}] "
                        f"generate_total={row['generate_total_ms']:.2f}ms gen_tokens={gen_tokens} "
                        f"correct={is_correct}",
                        flush=True,
                    )
                else:
                    print(
                        f"[gsm8k-latency warmup {warmup_done}/{int(self.warmup_samples)}] "
                        f"prefill={row['prefill_ms']:.2f}ms decode_per_tok={row['decode_per_token_ms']:.2f}ms "
                        f"steps={decode_steps} correct={is_correct}",
                        flush=True,
                    )
                continue

            total += 1
            correct += is_correct
            if self.latency_timing == "e2e_generate":
                generate_total_ms.append(row["generate_total_ms"])
                print(
                    f"[gsm8k-latency {total}] generate_total={row['generate_total_ms']:.2f}ms "
                    f"gen_tokens={gen_tokens} correct={is_correct}",
                    flush=True,
                )
            else:
                prefill_ms.append(row["prefill_ms"])
                if decode_steps > 0:
                    decode_ms_per_token.append(row["decode_per_token_ms"])
                    decode_ms_means_per_problem.append(row["decode_per_token_ms"])
                print(
                    f"[gsm8k-latency {total}] prefill={row['prefill_ms']:.2f}ms "
                    f"decode_per_tok={row['decode_per_token_ms']:.2f}ms steps={decode_steps} "
                    f"correct={is_correct}",
                    flush=True,
                )

        def _stats(name: str, values: List[float]) -> Dict[str, Any]:
            if not values:
                return {f"{name}_n": 0}
            return {
                f"{name}_n": len(values),
                f"{name}_mean": float(statistics.fmean(values)),
                f"{name}_p50": _percentile(values, 0.5),
                f"{name}_p95": _percentile(values, 0.95),
                f"{name}_min": float(min(values)),
                f"{name}_max": float(max(values)),
            }

        summary: Dict[str, Any] = {
            "total": total,
            "correct": correct,
            "accuracy": (float(correct) / float(total)) if total else 0.0,
            "warmup_samples": int(self.warmup_samples),
            "latency_timing": self.latency_timing,
        }
        if self.latency_timing == "e2e_generate":
            summary.update(_stats("generate_total_ms", generate_total_ms))
            summary["total_decode_steps"] = int(
                sum(int(r["decode_steps"]) for r in per_example if not r["is_warmup"])
            )
        else:
            summary.update(_stats("prefill_ms", prefill_ms))
            summary.update(_stats("decode_ms_per_token", decode_ms_per_token))
            summary["total_decode_steps"] = int(
                sum(int(r["decode_steps"]) for r in per_example if not r["is_warmup"])
            )

        return {
            "benchmark": "gsm8k",
            "mode": "latency",
            "summary": summary,
            "per_example": per_example,
        }
