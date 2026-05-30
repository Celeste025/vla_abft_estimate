"""Capture full output tensors from a single Qwen decoder matmul hook site per forward."""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from op_output_stats_capture import _extract_primary_tensor


@dataclass
class SingleSiteTensorCapture(AbstractContextManager):
    """Register one forward hook on ``layers[layer_idx].self_attn.o_proj`` (by default)."""

    model: nn.Module
    layer_idx: int
    site_suffix: str = "o_proj"
    _handles: List[Any] = field(default_factory=list)
    _episode_tensors: List[torch.Tensor] = field(default_factory=list)
    _active: bool = False

    def __post_init__(self) -> None:
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise ValueError("expected HF CausalLM with .model.layers (Qwen-style)")
        layers = self.model.model.layers
        if self.layer_idx < 0 or self.layer_idx >= len(layers):
            raise ValueError(f"layer_idx={self.layer_idx} out of range [0, {len(layers)})")
        layer = layers[self.layer_idx]
        attn = layer.self_attn
        suffix = str(self.site_suffix).strip()
        if suffix == "o_proj":
            module = attn.o_proj
        elif suffix == "q_proj":
            module = attn.q_proj
        elif suffix == "k_proj":
            module = attn.k_proj
        elif suffix == "v_proj":
            module = attn.v_proj
        else:
            raise ValueError(f"unsupported site_suffix={suffix!r} (use o_proj|q_proj|k_proj|v_proj)")

        def _hook(_m: nn.Module, _inp: Any, output: Any) -> None:
            if not self._active:
                return
            tensor, _ = _extract_primary_tensor(output)
            if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                return
            self._episode_tensors.append(tensor.detach().cpu().float().clone())

        self._handles.append(module.register_forward_hook(_hook, with_kwargs=False))

    @property
    def site_id(self) -> str:
        return f"L{self.layer_idx}_{self.site_suffix}"

    def begin_episode(self) -> None:
        self._episode_tensors = []
        self._active = True

    def end_episode(self) -> List[torch.Tensor]:
        self._active = False
        return list(self._episode_tensors)

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "SingleSiteTensorCapture":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
