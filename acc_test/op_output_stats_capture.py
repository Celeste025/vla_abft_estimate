"""Capture per-site tensor stats for Qwen decoder matmul-related hooks (same sites as inject.py).

Aligns with ``InjectionContext`` qwen_decoder registration:
q_proj, k_proj, v_proj, attn_core (o_proj first-arg pre), o_proj, mlp_gate, mlp_up, mlp_down.

Per testcase, aggregates over **all** root-model forward passes in that episode (e.g. prefill+decode
for GSM8K ``generate``, or four choice forwards for HellaSwag):
  global_min / global_max / weighted_mean / total_numel / n_hook_fires / shape_first / shape_last
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from inject import SITE_STRATEGY_QWEN, list_sites


def _extract_primary_tensor(output: Any) -> Tuple[Optional[torch.Tensor], bool]:
    if isinstance(output, torch.Tensor):
        return output, False
    if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0], True
    return None, False


@dataclass
class _SiteRunningAgg:
    global_min: float = float("inf")
    global_max: float = float("-inf")
    sum_val: float = 0.0
    total_numel: int = 0
    n_hook_fires: int = 0
    shape_first: Optional[List[int]] = None
    shape_last: Optional[List[int]] = None

    def observe(self, t: torch.Tensor) -> None:
        if t.numel() == 0:
            return
        with torch.no_grad():
            x = t.detach().float()
            mn = float(x.amin().item())
            mx = float(x.amax().item())
            s = float(x.sum().item())
            n = int(x.numel())
        self.global_min = min(self.global_min, mn)
        self.global_max = max(self.global_max, mx)
        self.sum_val += s
        self.total_numel += n
        self.n_hook_fires += 1
        sh = list(t.shape)
        if self.shape_first is None:
            self.shape_first = sh
        self.shape_last = sh

    def to_dict(self) -> Dict[str, Any]:
        mean = self.sum_val / self.total_numel if self.total_numel > 0 else float("nan")
        return {
            "global_min": self.global_min if self.global_min != float("inf") else float("nan"),
            "global_max": self.global_max if self.global_max != float("-inf") else float("nan"),
            "weighted_mean": float(mean),
            "total_numel": int(self.total_numel),
            "n_hook_fires": int(self.n_hook_fires),
            "shape_first": self.shape_first,
            "shape_last": self.shape_last,
        }


@dataclass
class QwenDecoderOpStatsCapture:
    """Read-only hooks; register once, then begin_episode / end_episode around each testcase."""

    model: nn.Module
    _handles: List[Any] = field(default_factory=list)
    _site_to_handle: Dict[str, Any] = field(default_factory=dict)
    _agg: Dict[str, _SiteRunningAgg] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._register_hooks()

    def begin_episode(self) -> None:
        self._agg = {}

    def end_episode(self) -> Dict[str, Any]:
        return {sid: a.to_dict() for sid, a in sorted(self._agg.items(), key=lambda kv: kv[0])}

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._site_to_handle.clear()

    def __enter__(self) -> "QwenDecoderOpStatsCapture":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _ensure(self, site_id: str) -> _SiteRunningAgg:
        if site_id not in self._agg:
            self._agg[site_id] = _SiteRunningAgg()
        return self._agg[site_id]

    def _on_tensor(self, site_id: str, t: Optional[torch.Tensor]) -> None:
        if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
            return
        self._ensure(site_id).observe(t)

    def _register_hooks(self) -> None:
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise ValueError("expected HF CausalLM with .model.layers (Qwen-style)")
        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            mlp = layer.mlp
            self._register_forward(attn.q_proj, f"L{i}_q_proj")
            self._register_forward(attn.k_proj, f"L{i}_k_proj")
            self._register_forward(attn.v_proj, f"L{i}_v_proj")
            self._register_pre_first_arg(attn.o_proj, f"L{i}_attn_core")
            self._register_forward(attn.o_proj, f"L{i}_o_proj")
            self._register_forward(mlp.gate_proj, f"L{i}_mlp_gate")
            self._register_forward(mlp.up_proj, f"L{i}_mlp_up")
            self._register_forward(mlp.down_proj, f"L{i}_mlp_down")

    def _register_forward(self, module: nn.Module, site_id: str) -> None:
        def _hook(_m: nn.Module, _inp: Any, output: Any) -> None:
            tensor, _ = _extract_primary_tensor(output)
            self._on_tensor(site_id, tensor)

        h = module.register_forward_hook(_hook, with_kwargs=False)
        self._handles.append(h)
        self._site_to_handle[site_id] = h

    def _register_pre_first_arg(self, module: nn.Module, site_id: str) -> None:
        def _pre(_m: nn.Module, inputs: Tuple[object, ...]) -> None:
            if len(inputs) == 0:
                return
            x0 = inputs[0]
            if isinstance(x0, torch.Tensor):
                self._on_tensor(site_id, x0)

        h = module.register_forward_pre_hook(_pre, with_kwargs=False)
        self._handles.append(h)
        self._site_to_handle[site_id] = h

    def collect_registration_stats(self) -> Dict[str, Any]:
        expected = set(list_sites(self.model, strategy=SITE_STRATEGY_QWEN))
        registered = set(self._site_to_handle.keys())
        missing = sorted(expected - registered)
        return {
            "expected_site_count": len(expected),
            "registered_site_count": len(registered),
            "missing_sites": missing,
        }
