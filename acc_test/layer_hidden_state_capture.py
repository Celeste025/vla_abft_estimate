"""Capture per-layer decoder outputs (down_proj + residual) on each forward."""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn as nn

from op_output_stats_capture import _extract_primary_tensor


@dataclass
class LayerHiddenStateCapture(AbstractContextManager):
    """Read-only forward hooks on every ``model.model.layers[i]`` (layer output)."""

    model: nn.Module
    _handles: List[Any] = field(default_factory=list)
    _by_layer: Dict[int, torch.Tensor] = field(default_factory=dict)
    _active: bool = False

    def __post_init__(self) -> None:
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise ValueError("expected HF CausalLM with .model.layers (Qwen-style)")
        layers = self.model.model.layers
        for i, layer in enumerate(layers):

            def _hook(layer_idx: int):
                def _fn(_m: nn.Module, _inp: Any, output: Any) -> None:
                    if not self._active:
                        return
                    tensor, _ = _extract_primary_tensor(output)
                    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                        return
                    self._by_layer[layer_idx] = tensor.detach().cpu().float().clone()

                return _fn

            self._handles.append(layer.register_forward_hook(_hook(i), with_kwargs=False))

    @property
    def num_layers(self) -> int:
        return len(self.model.model.layers)

    def begin_episode(self) -> None:
        self._by_layer = {}
        self._active = True

    def end_episode(self) -> Dict[int, torch.Tensor]:
        self._active = False
        n = self.num_layers
        missing = [i for i in range(n) if i not in self._by_layer]
        if missing:
            raise RuntimeError(f"layer hidden capture incomplete; missing layers: {missing}")
        return {i: self._by_layer[i] for i in range(n)}

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "LayerHiddenStateCapture":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
