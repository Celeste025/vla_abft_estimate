from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

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
        protect_only: bool = False,
        protect_capture_stats: bool = False,
        acc_v2: bool = False,
        thr_gamma: float = 3.0,
        acc_v2_threshold_enable: bool = True,
        acc_v2_restore_mode: str = "golden",
        acc_v2_inject_enable: bool = True,
        acc_v2_metrics_scope: str = "target",
        fault_mode: str = "fixed",
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
        # protect_only: 不做任何 fault injection；对每个注册站点的张量都执行
        # mask = abs(x) > fault_delta * clear_threshold_mul; x[mask] = 0。
        # 用于测量"仅保护"开销（latency benchmark）。
        self.protect_only = bool(protect_only)
        # protect_only + True：累计每层 site 上 |x|>threshold 的元素个数（含 GPU→CPU sync）。
        self.protect_capture_stats = bool(protect_capture_stats)

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
        # protect_only + protect_capture_stats：site_id -> 累计清零元素个数
        self._protect_clear_counts: Dict[str, int] = {}

        # --- ACC v2: warmup min/max per site, interval [m*γ, M*γ], golden restore, metrics ---
        self.acc_v2 = bool(acc_v2)
        self.thr_gamma = float(thr_gamma)
        self.acc_v2_threshold_enable = bool(acc_v2_threshold_enable)
        self.acc_v2_restore_mode = str(acc_v2_restore_mode).strip().lower()
        if self.acc_v2_restore_mode not in {"golden", "zero"}:
            raise ValueError(
                f"acc_v2_restore_mode must be golden|zero, got {acc_v2_restore_mode!r}"
            )
        self.acc_v2_inject_enable = bool(acc_v2_inject_enable)
        self.acc_v2_metrics_scope = str(acc_v2_metrics_scope).strip().lower()
        if self.acc_v2_metrics_scope not in {"target", "all"}:
            raise ValueError(
                f"acc_v2_metrics_scope must be target|all, got {acc_v2_metrics_scope!r}"
            )
        self.fault_mode = str(fault_mode).strip().lower()
        if self.fault_mode not in {"fixed", "rand2pow", "none"}:
            raise ValueError(f"fault_mode must be fixed|rand2pow|none, got {fault_mode!r}")
        self._warmup_active: bool = False
        self._site_min_max: Dict[str, Tuple[float, float]] = {}
        self._acc_metrics: Dict[str, int] = {
            "runs": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "normal": 0,
        }
        # acc_v2_metrics_scope=="all": per-site buckets (same keys as _acc_metrics).
        self._acc_metrics_by_site: Dict[str, Dict[str, int]] = {}
        # Flat index of the single element corrupted in this forward at target_site (ACC v2).
        self._acc_v2_inj_flat_idx: Optional[int] = None

    def __enter__(self) -> "InjectionContext":
        if self.acc_v2 and self.site_strategy != SITE_STRATEGY_QWEN:
            raise NotImplementedError("acc_v2 is only implemented for site_strategy=qwen_decoder")
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

    # --- ACC v2 API ---
    def set_warmup(self, active: bool) -> None:
        self._warmup_active = bool(active)

    def reset_site_bounds(self) -> None:
        self._site_min_max.clear()

    def reset_acc_metrics(self) -> None:
        for k in self._acc_metrics:
            self._acc_metrics[k] = 0
        self._acc_metrics_by_site.clear()

    def get_acc_v2_metrics(self) -> Dict[str, int]:
        return dict(self._acc_metrics)

    def get_acc_v2_metrics_by_site(self) -> Dict[str, Dict[str, int]]:
        """Per-site ACC v2 counters (meaningful when acc_v2_metrics_scope=='all')."""
        return {k: dict(v) for k, v in sorted(self._acc_metrics_by_site.items())}

    def export_acc_v2_metrics(self, paths: Dict[str, Path], *, site_id: str) -> None:
        """Write site_metrics.json + site_metrics.csv under paths['json']/paths['csv']."""
        row = {
            "site_id": site_id,
            **self.get_acc_v2_metrics(),
            "thr_gamma": self.thr_gamma,
            "fault_mode": self.fault_mode,
        }
        jp = paths["json"] / "site_metrics.json"
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)
        cp = paths["csv"] / "site_metrics.csv"
        with open(cp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)

    def _warmup_obs(self, site_id: str, x: torch.Tensor) -> None:
        with torch.no_grad():
            xf = x.detach().float()
            lo = float(xf.amin().item())
            hi = float(xf.amax().item())
        prev = self._site_min_max.get(site_id)
        if prev is None:
            self._site_min_max[site_id] = (lo, hi)
        else:
            self._site_min_max[site_id] = (min(prev[0], lo), max(prev[1], hi))

    def _can_inject_fault(self, site_id: str) -> bool:
        if not self.acc_v2_inject_enable:
            return False
        if self.target_site is None or site_id != self.target_site:
            return False
        if self._injected_this_forward:
            return False
        if self._decode_active:
            if self._decode_already_injected:
                return False
            if self._decode_target_step is None or self._decode_current_step is None:
                return False
            if self._decode_current_step != self._decode_target_step:
                return False
        return True

    def _inject_fault_into(self, out: torch.Tensor) -> None:
        if self.fault_mode == "none":
            return
        flat = out.reshape(-1)
        if self.fault_mode == "rand2pow":
            k = self.rng.randint(-14, 15)
            idx = self.rng.randrange(flat.numel())
            flat[idx] = flat[idx] * (2.0**k)
        else:
            if self.fault_index_mode == "max_abs":
                idx = int(flat.abs().argmax().item())
            else:
                idx = self.rng.randrange(flat.numel())
            flat[idx] = flat[idx] + self.fault_delta
        self._acc_v2_inj_flat_idx = int(idx)
        if self._decode_active:
            self._decode_already_injected = True
            self._decode_injected_step = self._decode_current_step

    def _acc_site_bucket(self, site_id: str) -> Optional[Dict[str, int]]:
        if self.acc_v2_metrics_scope == "all":
            return self._acc_metrics_by_site.setdefault(
                site_id,
                {"runs": 0, "tp": 0, "fp": 0, "fn": 0, "normal": 0},
            )
        if self.target_site is None or site_id != self.target_site:
            return None
        return self._acc_metrics

    def _acc_tick_metrics(
        self,
        site_id: str,
        injected: bool,
        mask: torch.Tensor,
        inj_flat_idx: Optional[int],
    ) -> None:
        """ACC v2 counters (one tick per hook evaluation).

        - **tp**: fault injected and the **injected flat element** is out of range (caught).
        - **fn**: fault injected but the injected element is **not** flagged.
        - **fp**: threshold flags at least one element that was **not** the injected one
          (spurious alarm on clean positions); if there was no injection, any flag is fp.
        Multiple of tp/fp/fn may increment in the same forward when both the fault site
        and spurious positions are flagged.

        When ``acc_v2_metrics_scope=='all'``, counts are accumulated per ``site_id``;
        otherwise only ``target_site`` updates the aggregate ``_acc_metrics``.
        """
        bucket = self._acc_site_bucket(site_id)
        if bucket is None:
            return
        bucket["runs"] += 1
        if mask.numel() == 0:
            if not injected:
                bucket["normal"] += 1
            else:
                bucket["fn"] += 1
            return

        flat_mask = mask.reshape(-1)
        if not injected:
            if bool(flat_mask.any().item()):
                bucket["fp"] += 1
            else:
                bucket["normal"] += 1
            return

        n = int(flat_mask.numel())
        if inj_flat_idx is None or n == 0:
            if bool(flat_mask.any().item()):
                bucket["tp"] += 1
            else:
                bucket["fn"] += 1
            return

        ii = int(inj_flat_idx) % n
        inj_hit = bool(flat_mask[ii].item())
        sel = torch.zeros_like(flat_mask, dtype=torch.bool)
        sel[ii] = True
        has_spurious = bool((flat_mask & ~sel).any().item())

        if has_spurious:
            bucket["fp"] += 1
        if inj_hit:
            bucket["tp"] += 1
        else:
            bucket["fn"] += 1

    def _acc_v2_transform_tensor(self, site_id: str, tensor: torch.Tensor, *, allow_inject: bool) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        if self._warmup_active:
            self._warmup_obs(site_id, tensor)
            return tensor
        golden = tensor.detach().clone()
        work = tensor.clone()
        injected = False
        if allow_inject and self._can_inject_fault(site_id):
            self._inject_fault_into(work)
            injected = True
            self._injected_this_forward = True
            self._inject_count += 1
        if not self.acc_v2_threshold_enable:
            if not self._warmup_active:
                b = self._acc_site_bucket(site_id)
                if b is not None:
                    b["runs"] += 1
            return work
        bounds = self._site_min_max.get(site_id)
        if bounds is None:
            return work
        m, M = bounds
        lo = m * self.thr_gamma
        hi = M * self.thr_gamma
        wf = work.float()
        mask = (wf < lo) | (wf > hi)
        out = work.clone()
        if self.acc_v2_restore_mode == "zero":
            out = out.masked_fill(mask, 0)
        else:
            out[mask] = golden[mask]
        inj_idx = self._acc_v2_inj_flat_idx if injected else None
        self._acc_tick_metrics(site_id, injected, mask, inj_idx)
        return out

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
            self._acc_v2_inj_flat_idx = None

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
        if self.acc_v2:
            tensor, tuple_output = self._extract_primary_tensor(output)
            if tensor is None or tensor.numel() == 0:
                return output
            allow_inject = (
                self.acc_v2_inject_enable
                and not self.protect_only
                and self.target_site is not None
                and site_id == self.target_site
            )
            new_tensor = self._acc_v2_transform_tensor(site_id, tensor, allow_inject=allow_inject)
            if new_tensor is tensor:
                return output
            if tuple_output:
                assert isinstance(output, tuple)
                return (new_tensor, *output[1:])
            return new_tensor
        if self.protect_only:
            tensor, tuple_output = self._extract_primary_tensor(output)
            if tensor is None or tensor.numel() == 0:
                return output
            new_tensor = self._apply_protect(site_id, tensor)
            if new_tensor is tensor:
                return output
            if tuple_output:
                assert isinstance(output, tuple)
                return (new_tensor, *output[1:])
            return new_tensor
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
        if self.acc_v2:
            if not isinstance(x, torch.Tensor) or x.numel() == 0:
                return x
            allow_inject = (
                self.acc_v2_inject_enable
                and not self.protect_only
                and self.target_site is not None
                and site_id == self.target_site
            )
            return self._acc_v2_transform_tensor(site_id, x, allow_inject=allow_inject)
        if self.protect_only:
            if not isinstance(x, torch.Tensor) or x.numel() == 0:
                return x
            return self._apply_protect(site_id, x)
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

    def _apply_protect(self, site_id: str, x: torch.Tensor) -> torch.Tensor:
        """protect_only 模式下的清零保护：threshold mask + masked_fill。
        默认不计数（无 GPU→CPU sync），latency 友好。
        protect_capture_stats=True 时累计该 site 上被清零的元素个数（mask.sum）。
        """
        threshold = abs(self.fault_delta) * self.clear_threshold_mul
        mask = x.abs() > threshold
        if self.protect_capture_stats:
            n = int(mask.sum().item())
            if n > 0:
                self._protect_clear_counts[site_id] = self._protect_clear_counts.get(site_id, 0) + n
        return x.masked_fill(mask, 0)

    def reset_protect_clear_counts(self) -> None:
        """下一轮 generate 前清零累计（同一 InjectionContext 内连跑多题时使用）。"""
        self._protect_clear_counts.clear()

    def get_protect_clear_stats(self) -> Dict[str, Any]:
        """仅在 protect_only + protect_capture_stats 有意义。"""
        total = int(sum(self._protect_clear_counts.values()))
        by_site = dict(sorted(self._protect_clear_counts.items(), key=lambda kv: kv[0]))
        nonzero_sites = sorted(k for k, v in self._protect_clear_counts.items() if v > 0)
        return {
            "threshold": abs(self.fault_delta) * self.clear_threshold_mul,
            "total_cleared_elements": total,
            "sites_with_any_clear": len(nonzero_sites),
            "by_site": by_site,
            "nonzero_site_ids": nonzero_sites,
        }

    @staticmethod
    def _extract_primary_tensor(output):
        if isinstance(output, torch.Tensor):
            return output, False
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0], True
        return None, False
