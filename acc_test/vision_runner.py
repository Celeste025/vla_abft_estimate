from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torchvision.models import ResNet50_Weights, resnet50

from inject import SITE_STRATEGY_MODULE_SCAN, InjectionContext
from model_runner import RunMeta


def get_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


class VisionRunner:
    """ResNet-50 ImageNet weights + optional fault injection (Conv2d/Linear sites)."""

    def __init__(
        self,
        weights: str = "IMAGENET1K_V2",
        device: str = "cuda",
        dtype: str = "float32",
    ) -> None:
        self.weights_name = weights
        self.weights_enum = ResNet50_Weights[weights]
        self.transforms = self.weights_enum.transforms()
        self.model_id = f"torchvision/resnet50/{weights}"
        self.device = device
        self.dtype_name = dtype
        self.dtype = get_dtype(dtype)
        self.attn_implementation: Optional[str] = None

        self.model = resnet50(weights=self.weights_enum)
        self.model.eval().to(device=device, dtype=self.dtype)
        self._active_injector: Optional[InjectionContext] = None

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def run_task(
        self,
        task,
        inject_site: Optional[str] = None,
        fault_delta: float = 10000.0,
        seed: int = 2026,
        fault_index_mode: str = "random",
        clear_exceptions: bool = False,
        clear_threshold_mul: float = 0.5,
        decode_step_inject_enable: bool = False,
        decode_step_max: int = 150,
        site_strategy: str = SITE_STRATEGY_MODULE_SCAN,
        target_module_classes: Tuple[Type[torch.nn.Module], ...] = (),
    ) -> Dict[str, Any]:
        with InjectionContext(
            model=self.model,
            target_site=inject_site,
            fault_delta=fault_delta,
            seed=seed,
            fault_index_mode=fault_index_mode,
            clear_exceptions=clear_exceptions,
            clear_threshold_mul=clear_threshold_mul,
            warning_print_limit=5,
            decode_step_inject_enable=decode_step_inject_enable,
            decode_step_max=decode_step_max,
            site_strategy=site_strategy,
            target_module_classes=target_module_classes,
        ) as inj:
            self._active_injector = inj
            result = task.run(self)
            self._active_injector = None
            stats = inj.collect_hook_stats()
            meta = RunMeta(
                model_id=self.model_id,
                device=self.device,
                dtype=self.dtype_name,
                attn_implementation=None,
                inject_site=inject_site,
                fault_delta=fault_delta,
                inject_count=inj.inject_count,
                expected_site_count=stats.expected_site_count,
                registered_site_count=stats.registered_site_count,
                missing_sites=stats.missing_sites,
                injected_forward_count=stats.injected_forward_count,
                bad_forward_count=stats.bad_forward_count,
                errors_total=stats.errors_total,
                warning_printed=stats.warning_printed,
                decode_problem_count=stats.decode_problem_count,
                decode_injected_problem_count=stats.decode_injected_problem_count,
            )
            result["run_meta"] = asdict(meta)
            return result
