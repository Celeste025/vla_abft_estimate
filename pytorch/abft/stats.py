from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class CheckResult:
    ok: bool
    abs_err: float
    rel_err: float
    abft_corner: float
    sum_c: float


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return data[lo]
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def summarize_ms(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p99_ms": 0.0,
        }
    xs = sorted(samples)
    return {
        "count": len(xs),
        "mean_ms": statistics.fmean(xs),
        "p50_ms": percentile(xs, 0.50),
        "p90_ms": percentile(xs, 0.90),
        "p99_ms": percentile(xs, 0.99),
    }


class AbftStatsCollector:
    def __init__(self) -> None:
        self.current_phase: str = "prefill"
        self.total_checks: int = 0
        self.total_failures: int = 0
        self.total_injections: int = 0
        self.total_detected_injections: int = 0

        self.phase_checks: Counter[str] = Counter()
        self.phase_failures: Counter[str] = Counter()
        self.phase_injections: Counter[str] = Counter()
        self.phase_detected_injections: Counter[str] = Counter()

        self.op_checks: Counter[str] = Counter()
        self.op_failures: Counter[str] = Counter()
        self.shape_checks: Counter[Tuple[str, str]] = Counter()
        self.shape_failures: Counter[Tuple[str, str]] = Counter()

        self.prefill_ms: List[float] = []
        self.decode_step_ms: List[float] = []
        self.e2e_request_ms: List[float] = []
        self.by_prompt_decode_ms: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"prefill_ms": [], "decode_step_ms": [], "e2e_request_ms": []}
        )

    def set_phase(self, phase: str) -> None:
        self.current_phase = phase

    def record_check(
        self,
        op_name: str,
        shape_key: str,
        result: CheckResult,
        injected: bool = False,
    ) -> None:
        self.total_checks += 1
        self.phase_checks[self.current_phase] += 1
        self.op_checks[op_name] += 1
        self.shape_checks[(self.current_phase, f"{op_name}:{shape_key}")] += 1

        if not result.ok:
            self.total_failures += 1
            self.phase_failures[self.current_phase] += 1
            self.op_failures[op_name] += 1
            self.shape_failures[(self.current_phase, f"{op_name}:{shape_key}")] += 1
            if injected:
                self.total_detected_injections += 1
                self.phase_detected_injections[self.current_phase] += 1

    def record_injection(self) -> None:
        self.total_injections += 1
        self.phase_injections[self.current_phase] += 1

    def record_prefill_latency(self, prompt_len: int, ms: float) -> None:
        self.prefill_ms.append(ms)
        self.by_prompt_decode_ms[str(prompt_len)]["prefill_ms"].append(ms)

    def record_decode_step_latency(self, prompt_len: int, ms: float) -> None:
        self.decode_step_ms.append(ms)
        self.by_prompt_decode_ms[str(prompt_len)]["decode_step_ms"].append(ms)

    def record_e2e_latency(self, prompt_len: int, ms: float) -> None:
        self.e2e_request_ms.append(ms)
        self.by_prompt_decode_ms[str(prompt_len)]["e2e_request_ms"].append(ms)

    def _top_items(self, counter: Counter[Any], k: int = 20) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key, v in counter.most_common(k):
            rows.append({"key": str(key), "count": int(v)})
        return rows

    def export_summary(self) -> Dict[str, Any]:
        detect_rate = (
            float(self.total_detected_injections) / float(self.total_injections)
            if self.total_injections > 0
            else 0.0
        )
        return {
            "latency": {
                "prefill": summarize_ms(self.prefill_ms),
                "decode_step": summarize_ms(self.decode_step_ms),
                "e2e_request": summarize_ms(self.e2e_request_ms),
                "by_prompt_len": {
                    k: {
                        "prefill": summarize_ms(v["prefill_ms"]),
                        "decode_step": summarize_ms(v["decode_step_ms"]),
                        "e2e_request": summarize_ms(v["e2e_request_ms"]),
                    }
                    for k, v in self.by_prompt_decode_ms.items()
                },
            },
            "abft": {
                "checks_total": int(self.total_checks),
                "fail_total": int(self.total_failures),
                "checks_by_phase": dict(self.phase_checks),
                "fails_by_phase": dict(self.phase_failures),
                "checks_by_op": dict(self.op_checks),
                "fails_by_op": dict(self.op_failures),
                "top_shapes_by_checks": self._top_items(self.shape_checks),
                "top_shapes_by_fails": self._top_items(self.shape_failures),
            },
            "fault_injection": {
                "injections_total": int(self.total_injections),
                "detected_injections_total": int(self.total_detected_injections),
                "injections_by_phase": dict(self.phase_injections),
                "detected_injections_by_phase": dict(self.phase_detected_injections),
                "detection_rate": detect_rate,
            },
        }
