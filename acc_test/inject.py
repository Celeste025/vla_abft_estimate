from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch


MLP_MATMUL_SITE_SUFFIXES = ("mlp_gate", "mlp_up", "mlp_down")
ATTN_MATMUL_SITE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_core",  # fused q*k^t and s*v: inject at o_proj input (pre-hook)
    "o_proj",
)
NON_MATMUL_SITE_SUFFIXES = (
    "input_norm",
    "post_attn_norm",
    "attn_residual",
    "mlp_act",
    "mlp_residual",
)

MLP_SITE_SUFFIXES = MLP_MATMUL_SITE_SUFFIXES
ATTN_SITE_SUFFIXES = ATTN_MATMUL_SITE_SUFFIXES

SITE_STRATEGY_QWEN = "qwen_decoder"
SiteSet = Literal["matmul", "nonmatmul", "all"]


def _normalize_site_set(site_set: str) -> SiteSet:
    s = str(site_set).strip().lower()
    if s not in {"matmul", "nonmatmul", "all"}:
        raise ValueError(f"site_set must be matmul|nonmatmul|all, got {site_set!r}")
    return s  # type: ignore[return-value]


def layer_site_ids(layer_idx: int, site_set: SiteSet = "matmul") -> List[str]:
    ss = _normalize_site_set(site_set)
    out: List[str] = []
    if ss in {"matmul", "all"}:
        for suf in ATTN_MATMUL_SITE_SUFFIXES:
            out.append(f"L{layer_idx}_{suf}")
        for suf in MLP_MATMUL_SITE_SUFFIXES:
            out.append(f"L{layer_idx}_{suf}")
    if ss in {"nonmatmul", "all"}:
        for suf in NON_MATMUL_SITE_SUFFIXES:
            out.append(f"L{layer_idx}_{suf}")
    return out


def list_sites(
    model: torch.nn.Module,
    *,
    strategy: str = SITE_STRATEGY_QWEN,
    site_set: SiteSet = "matmul",
) -> List[str]:
    """Enumerate injectable site ids for a HuggingFace-style Qwen decoder LM."""
    if str(strategy).strip().lower() != SITE_STRATEGY_QWEN:
        raise ValueError(f"unsupported list_sites strategy={strategy!r}")
    ss = _normalize_site_set(site_set)
    num_layers = int(model.config.num_hidden_layers)
    sites: List[str] = []
    for i in range(num_layers):
        sites.extend(layer_site_ids(i, ss))
    return sites


@dataclass
class HookStats:
    expected_site_count: int
    registered_site_count: int
    missing_sites: List[str]
    registered_sites: List[str]
    injected_forward_count: int
    decode_problem_count: int
    decode_injected_problem_count: int


class InjectionContext(AbstractContextManager):
    """Qwen decoder ACC: warmup min/max per site, threshold [m·γ, M·γ], optional fault inject."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_site: Optional[str],
        fault_delta: float = 10000.0,
        seed: int = 2026,
        fault_index_mode: str = "random",
        thr_gamma: float = 3.0,
        threshold_enable: bool = True,
        restore_mode: str = "golden",
        inject_enable: bool = True,
        metrics_scope: str = "target",
        fault_mode: str = "fixed",
        decode_step_inject_enable: bool = False,
        decode_step_max: int = 150,
        site_set: SiteSet = "matmul",
    ) -> None:
        self.model = model
        self.site_set = _normalize_site_set(site_set)
        self.target_site = target_site
        self.fault_delta = float(fault_delta)
        self.rng = random.Random(seed)
        self.fault_index_mode = str(fault_index_mode).strip().lower()
        if self.fault_index_mode not in {"random", "max_abs"}:
            raise ValueError(f"unsupported fault_index_mode={fault_index_mode!r}")
        self.decode_step_inject_enable = bool(decode_step_inject_enable)
        self.decode_step_max = int(decode_step_max)

        self.thr_gamma = float(thr_gamma)
        self.threshold_enable = bool(threshold_enable)
        self.restore_mode = str(restore_mode).strip().lower()
        if self.restore_mode not in {"golden", "zero"}:
            raise ValueError(f"restore_mode must be golden|zero, got {restore_mode!r}")
        self.inject_enable = bool(inject_enable)
        self.metrics_scope = str(metrics_scope).strip().lower()
        if self.metrics_scope not in {"target", "all"}:
            raise ValueError(f"metrics_scope must be target|all, got {metrics_scope!r}")
        self.fault_mode = str(fault_mode).strip().lower()
        if self.fault_mode not in {"fixed", "rand2pow", "none"}:
            raise ValueError(f"fault_mode must be fixed|rand2pow|none, got {fault_mode!r}")

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._site_to_handle: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._injected_this_forward = False
        self._inject_count = 0
        self._injected_forward_count = 0

        self._decode_active: bool = False
        self._decode_target_step: Optional[int] = None
        self._decode_current_step: Optional[int] = None
        self._decode_already_injected: bool = False
        self._decode_injected_step: Optional[int] = None
        self._decode_problem_count: int = 0
        self._decode_injected_problem_count: int = 0

        self._warmup_active: bool = False
        self._site_min_max: Dict[str, Tuple[float, float]] = {}
        self._acc_metrics: Dict[str, int] = {
            "runs": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "normal": 0,
        }
        self._acc_metrics_by_site: Dict[str, Dict[str, int]] = {}
        self._inj_flat_idx: Optional[int] = None

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
        expected = set(list_sites(self.model, site_set=self.site_set))
        registered = set(self._site_to_handle.keys())
        missing = sorted(expected - registered)
        return HookStats(
            expected_site_count=len(expected),
            registered_site_count=len(registered),
            missing_sites=missing,
            registered_sites=sorted(registered),
            injected_forward_count=self._injected_forward_count,
            decode_problem_count=self._decode_problem_count,
            decode_injected_problem_count=self._decode_injected_problem_count,
        )

    def set_warmup(self, active: bool) -> None:
        self._warmup_active = bool(active)

    def reset_site_bounds(self) -> None:
        self._site_min_max.clear()

    def reset_acc_metrics(self) -> None:
        for k in self._acc_metrics:
            self._acc_metrics[k] = 0
        self._acc_metrics_by_site.clear()

    def get_acc_metrics(self) -> Dict[str, int]:
        return dict(self._acc_metrics)

    def get_acc_metrics_by_site(self) -> Dict[str, Dict[str, int]]:
        """Per-site counters when metrics_scope=='all'."""
        return {k: dict(v) for k, v in sorted(self._acc_metrics_by_site.items())}

    def export_acc_metrics(self, paths: Dict[str, Path], *, site_id: str) -> None:
        row = {
            "site_id": site_id,
            **self.get_acc_metrics(),
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
        if not self.inject_enable:
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
        self._inj_flat_idx = int(idx)
        if self._decode_active:
            self._decode_already_injected = True
            self._decode_injected_step = self._decode_current_step

    def _acc_site_bucket(self, site_id: str) -> Optional[Dict[str, int]]:
        if self.metrics_scope == "all":
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

    def _transform_tensor(self, site_id: str, tensor: torch.Tensor, *, allow_inject: bool) -> torch.Tensor:
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
        if not self.threshold_enable:
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
        if self.restore_mode == "zero":
            out = out.masked_fill(mask, 0)
        else:
            out[mask] = golden[mask]
        inj_idx = self._inj_flat_idx if injected else None
        self._acc_tick_metrics(site_id, injected, mask, inj_idx)
        return out

    def begin_decode(self, step_max: Optional[int] = None) -> None:
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
        def _on_forward_start(_module: torch.nn.Module, _args) -> None:
            self._injected_this_forward = False
            self._inj_flat_idx = None

        self._handles.append(self.model.register_forward_pre_hook(_on_forward_start))

    def _register_model_forward_hook(self) -> None:
        def _on_forward_end(_module: torch.nn.Module, _args, _output) -> None:
            if self._injected_this_forward:
                self._injected_forward_count += 1

        self._handles.append(self.model.register_forward_hook(_on_forward_end, with_kwargs=False))

    def _register_layer_hooks(self) -> None:
        layers = self.model.model.layers
        ss = self.site_set
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            if ss in {"matmul", "all"}:
                self._register_site_forward_hook(attn.q_proj, f"L{i}_q_proj")
                self._register_site_forward_hook(attn.k_proj, f"L{i}_k_proj")
                self._register_site_forward_hook(attn.v_proj, f"L{i}_v_proj")
                self._register_site_pre_hook_on_first_arg(attn.o_proj, f"L{i}_attn_core")
                self._register_site_forward_hook(attn.o_proj, f"L{i}_o_proj")
                self._register_site_forward_hook(layer.mlp.gate_proj, f"L{i}_mlp_gate")
                self._register_site_forward_hook(layer.mlp.up_proj, f"L{i}_mlp_up")
                self._register_site_forward_hook(layer.mlp.down_proj, f"L{i}_mlp_down")
            if ss in {"nonmatmul", "all"}:
                self._register_site_forward_hook(layer.input_layernorm, f"L{i}_input_norm")
                self._register_site_forward_hook(
                    layer.post_attention_layernorm, f"L{i}_post_attn_norm"
                )
                self._register_site_pre_hook_on_first_arg(
                    layer.post_attention_layernorm, f"L{i}_attn_residual"
                )
                self._register_site_pre_hook_on_first_arg(layer.mlp.down_proj, f"L{i}_mlp_act")
                self._register_site_forward_hook(layer, f"L{i}_mlp_residual")

    def _register_site_forward_hook(self, module: torch.nn.Module, site_id: str) -> None:
        def _hook(_module: torch.nn.Module, _inputs, output):
            return self._maybe_transform_output(site_id, output)

        h = module.register_forward_hook(_hook, with_kwargs=False)
        self._handles.append(h)
        self._site_to_handle[site_id] = h

    def _register_site_pre_hook_on_first_arg(self, module: torch.nn.Module, site_id: str) -> None:
        def _pre_hook(_module: torch.nn.Module, inputs: Tuple[object, ...]):
            if not inputs:
                return inputs
            x0 = inputs[0]
            if not isinstance(x0, torch.Tensor):
                return inputs
            x0_mod = self._maybe_transform_tensor(site_id, x0)
            if x0_mod is x0:
                return inputs
            return (x0_mod, *inputs[1:])

        h = module.register_forward_pre_hook(_pre_hook, with_kwargs=False)
        self._handles.append(h)
        self._site_to_handle[site_id] = h

    def _allow_inject(self, site_id: str) -> bool:
        return (
            self.inject_enable
            and self.target_site is not None
            and site_id == self.target_site
        )

    def _maybe_transform_output(self, site_id: str, output):
        tensor, tuple_output = self._extract_primary_tensor(output)
        if tensor is None or tensor.numel() == 0:
            return output
        new_tensor = self._transform_tensor(
            site_id, tensor, allow_inject=self._allow_inject(site_id)
        )
        if new_tensor is tensor:
            return output
        if tuple_output:
            assert isinstance(output, tuple)
            return (new_tensor, *output[1:])
        return new_tensor

    def _maybe_transform_tensor(self, site_id: str, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.numel() == 0:
            return x
        return self._transform_tensor(site_id, x, allow_inject=self._allow_inject(site_id))

    @staticmethod
    def _extract_primary_tensor(output):
        if isinstance(output, torch.Tensor):
            return output, False
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0], True
        return None, False
