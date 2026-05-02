from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple, Type

import torch


MLP_SITE_SUFFIXES = ("mlp_gate", "mlp_up", "mlp_down")
ATTN_SITE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_core",  # fused together q*k^t and s*v by injecting o_proj input
    "o_proj",
)

SITE_STRATEGY_QWEN = "qwen_decoder"
SITE_STRATEGY_MODULE_SCAN = "module_class_scan"


def list_sites(
    model: torch.nn.Module,
    *,
    strategy: str = SITE_STRATEGY_QWEN,
    module_classes: Optional[Tuple[Type[torch.nn.Module], ...]] = None,
) -> List[str]:
    """
    Enumerate injectable site ids.

    - qwen_decoder: HuggingFace-style causal LM decoder (Qwen2, etc.).
    - module_class_scan: every submodule whose type is in module_classes
      (default: Conv2d + Linear), using the dotted name from named_modules().
    """
    strategy = str(strategy).strip().lower()
    if strategy == SITE_STRATEGY_QWEN:
        num_layers = int(model.config.num_hidden_layers)
        sites: List[str] = []
        for i in range(num_layers):
            for suf in ATTN_SITE_SUFFIXES:
                sites.append(f"L{i}_{suf}")
            for suf in MLP_SITE_SUFFIXES:
                sites.append(f"L{i}_{suf}")
        return sites
    if strategy == SITE_STRATEGY_MODULE_SCAN:
        classes = module_classes or (torch.nn.Conv2d, torch.nn.Linear)
        sites = []
        for name, mod in model.named_modules():
            if not name:
                continue
            if any(isinstance(mod, c) for c in classes):
                sites.append(name)
        return sites
    raise ValueError(f"unsupported list_sites strategy={strategy!r}")


@dataclass
class HookStats:
    expected_site_count: int
    registered_site_count: int
    missing_sites: List[str]
    registered_sites: List[str]
    injected_forward_count: int
    bad_forward_count: int
    errors_total: int
    warning_printed: int
    decode_problem_count: int
    decode_injected_problem_count: int


class InjectionContext(AbstractContextManager):
    def __init__(
        self,
        model: torch.nn.Module,
        target_site: Optional[str],
        fault_delta: float = 10000.0,
        seed: int = 2026,
        fault_index_mode: str = "random",
        clear_exceptions: bool = False,
        clear_threshold_mul: float = 0.5,
        warning_print_limit: int = 5,
        decode_step_inject_enable: bool = False,
        decode_step_max: int = 150,
        site_strategy: str = SITE_STRATEGY_QWEN,
        target_module_classes: Tuple[Type[torch.nn.Module], ...] = (),
    ) -> None:
        self.model = model
        self.target_site = target_site
        self.fault_delta = float(fault_delta)
        self.rng = random.Random(seed)
        self.fault_index_mode = str(fault_index_mode).strip().lower()
        if self.fault_index_mode not in {"random", "max_abs"}:
            raise ValueError(f"unsupported fault_index_mode={fault_index_mode!r}")
        self.clear_exceptions = bool(clear_exceptions)
        self.clear_threshold_mul = float(clear_threshold_mul)
        self.warning_print_limit = int(warning_print_limit)
        self.decode_step_inject_enable = bool(decode_step_inject_enable)
        self.decode_step_max = int(decode_step_max)

        self.site_strategy = str(site_strategy).strip().lower()
        if self.site_strategy not in {SITE_STRATEGY_QWEN, SITE_STRATEGY_MODULE_SCAN}:
            raise ValueError(f"unsupported site_strategy={site_strategy!r}")
        if self.site_strategy == SITE_STRATEGY_MODULE_SCAN:
            self._target_module_classes: Tuple[Type[torch.nn.Module], ...] = (
                target_module_classes
                if target_module_classes
                else (torch.nn.Conv2d, torch.nn.Linear)
            )
        else:
            self._target_module_classes = target_module_classes

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._site_to_handle: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._injected_this_forward = False
        self._inject_count = 0
        self._errors_this_forward = 0
        self._injected_forward_count = 0
        self._bad_forward_count = 0
        self._errors_total = 0
        self._warning_printed = 0

        # decode-step injection gating (for generate/decode scenarios)
        self._decode_active: bool = False
        self._decode_target_step: Optional[int] = None
        self._decode_current_step: Optional[int] = None
        self._decode_already_injected: bool = False
        self._decode_injected_step: Optional[int] = None
        self._decode_problem_count: int = 0
        self._decode_injected_problem_count: int = 0

    def __enter__(self) -> "InjectionContext":
        self._register_model_pre_hook()
        self._register_model_forward_hook()
        self._register_layer_hooks()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._site_to_handle.clear()
        return None

    @property
    def inject_count(self) -> int:
        return self._inject_count

    def collect_hook_stats(self) -> HookStats:
        if self.site_strategy == SITE_STRATEGY_MODULE_SCAN:
            expected = set(
                list_sites(
                    self.model,
                    strategy=SITE_STRATEGY_MODULE_SCAN,
                    module_classes=self._target_module_classes,
                )
            )
        else:
            expected = set(list_sites(self.model, strategy=SITE_STRATEGY_QWEN))
        registered = set(self._site_to_handle.keys())
        missing = sorted(expected - registered)
        return HookStats(
            expected_site_count=len(expected),
            registered_site_count=len(registered),
            missing_sites=missing,
            registered_sites=sorted(registered),
            injected_forward_count=self._injected_forward_count,
            bad_forward_count=self._bad_forward_count,
            errors_total=self._errors_total,
            warning_printed=self._warning_printed,
            decode_problem_count=self._decode_problem_count,
            decode_injected_problem_count=self._decode_injected_problem_count,
        )

    def begin_decode(self, step_max: Optional[int] = None) -> None:
        """
        Called at the beginning of one generate() call (one 'problem' in GSM8K).
        We randomly pick a target generation step x in [0, step_max].
        If the generation ends before reaching x, no injection happens (as desired).
        """
        if not self.decode_step_inject_enable:
            self._decode_active = False
            self._decode_target_step = None
            self._decode_current_step = None
            self._decode_already_injected = False
            return
        mx = self.decode_step_max if step_max is None else int(step_max)
        if mx < 0:
            mx = 0
        self._decode_active = True
        self._decode_target_step = self.rng.randrange(mx + 1)
        self._decode_current_step = None
        self._decode_already_injected = False
        self._decode_injected_step = None

    def end_decode(self) -> None:
        if not self._decode_active:
            return
        self._decode_problem_count += 1
        if self._decode_already_injected:
            self._decode_injected_problem_count += 1

    def get_decode_target_step(self) -> Optional[int]:
        return self._decode_target_step

    def get_decode_injected(self) -> bool:
        return bool(self._decode_already_injected)

    def get_decode_injected_step(self) -> Optional[int]:
        return self._decode_injected_step

    def set_decode_step(self, step_idx: int) -> None:
        if not self._decode_active:
            return
        self._decode_current_step = int(step_idx)

    def _register_model_pre_hook(self) -> None:
        def _on_forward_start(module: torch.nn.Module, args) -> None:
            self._injected_this_forward = False
            self._errors_this_forward = 0

        h = self.model.register_forward_pre_hook(_on_forward_start)
        self._handles.append(h)

    def _register_model_forward_hook(self) -> None:
        def _on_forward_end(module: torch.nn.Module, args, output) -> None:
            if not self._injected_this_forward:
                return
            self._injected_forward_count += 1
            if self.clear_exceptions:
                self._errors_total += int(self._errors_this_forward)
                if self._errors_this_forward != 1:
                    self._bad_forward_count += 1
                    if self._warning_printed < self.warning_print_limit:
                        print(
                            f"WARNING clear-exc: target_site={self.target_site} "
                            f"errors_this_forward={self._errors_this_forward} (expected 1)",
                            flush=True,
                        )
                        self._warning_printed += 1

        h = self.model.register_forward_hook(_on_forward_end, with_kwargs=False)
        self._handles.append(h)

    def _register_layer_hooks(self) -> None:
        if self.site_strategy == SITE_STRATEGY_MODULE_SCAN:
            for name, mod in self.model.named_modules():
                if not name:
                    continue
                if any(isinstance(mod, c) for c in self._target_module_classes):
                    self._register_site_forward_hook(mod, name)
            return

        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            attn = layer.self_attn

            # q/k/v projections: inject right after each Linear output.
            self._register_site_forward_hook(attn.q_proj, f"L{i}_q_proj")
            self._register_site_forward_hook(attn.k_proj, f"L{i}_k_proj")
            self._register_site_forward_hook(attn.v_proj, f"L{i}_v_proj")

            # q*k^t and s*v together: inject at attention output right before o_proj
            # by modifying o_proj input tensor (pre-hook).
            self._register_site_pre_hook_on_first_arg(attn.o_proj, f"L{i}_attn_core")

            # o projection: inject right after o_proj output.
            self._register_site_forward_hook(attn.o_proj, f"L{i}_o_proj")

            # MLP projections: inject right after Linear outputs.
            self._register_site_forward_hook(layer.mlp.gate_proj, f"L{i}_mlp_gate")
            self._register_site_forward_hook(layer.mlp.up_proj, f"L{i}_mlp_up")
            self._register_site_forward_hook(layer.mlp.down_proj, f"L{i}_mlp_down")

    def _register_site_forward_hook(self, module: torch.nn.Module, site_id: str) -> None:
        def _hook(_module: torch.nn.Module, _inputs, output):
            return self._maybe_inject_output(site_id, output)

        h = module.register_forward_hook(_hook, with_kwargs=False)
        self._handles.append(h)
        self._site_to_handle[site_id] = h

    def _register_site_pre_hook_on_first_arg(self, module: torch.nn.Module, site_id: str) -> None:
        def _pre_hook(_module: torch.nn.Module, inputs: Tuple[object, ...]):
            if len(inputs) == 0:
                return inputs
            x0 = inputs[0]
            if not isinstance(x0, torch.Tensor):
                return inputs
            x0_mod = self._maybe_inject_tensor(site_id, x0)
            if x0_mod is x0:
                return inputs
            return (x0_mod, *inputs[1:])

        h = module.register_forward_pre_hook(_pre_hook, with_kwargs=False)
        self._handles.append(h)
        self._site_to_handle[site_id] = h

    def _maybe_inject_output(self, site_id: str, output):
        if self.target_site is None:
            return output
        if site_id != self.target_site:
            return output
        if self._injected_this_forward:
            return output

        tensor, tuple_output = self._extract_primary_tensor(output)
        if tensor is None or tensor.numel() == 0:
            return output
        return self._inject_and_maybe_rebuild(site_id, output, tensor, tuple_output)

    def _maybe_inject_tensor(self, site_id: str, x: torch.Tensor) -> torch.Tensor:
        if self.target_site is None:
            return x
        if site_id != self.target_site:
            return x
        if self._injected_this_forward:
            return x
        if self._decode_active:
            # Only inject once at the selected generation step.
            if self._decode_already_injected:
                return x
            if self._decode_target_step is None or self._decode_current_step is None:
                return x
            if self._decode_current_step != self._decode_target_step:
                return x
        if x.numel() == 0:
            return x

        out = x.clone()
        flat = out.reshape(-1)
        if self.fault_index_mode == "max_abs":
            idx = int(flat.abs().argmax().item())
        else:
            idx = self.rng.randrange(flat.numel())
        flat[idx] += self.fault_delta

        if self.clear_exceptions:
            # Clear out-of-range values after the fault injection.
            threshold = abs(self.fault_delta) * self.clear_threshold_mul
            mask = out.abs() > threshold
            detected = int(mask.sum().item())
            if detected > 0:
                out[mask] = 0
            self._errors_this_forward += detected

        self._injected_this_forward = True
        self._inject_count += 1
        if self._decode_active:
            self._decode_already_injected = True
            self._decode_injected_step = self._decode_current_step
        return out

    def _inject_and_maybe_rebuild(
        self,
        site_id: str,
        original_output,
        tensor: torch.Tensor,
        tuple_output: bool,
    ):
        out = self._maybe_inject_tensor(site_id, tensor)
        if tuple_output:
            assert isinstance(original_output, tuple)
            return (out, *original_output[1:])
        return out

    @staticmethod
    def _extract_primary_tensor(output):
        if isinstance(output, torch.Tensor):
            return output, False
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0], True
        return None, False
