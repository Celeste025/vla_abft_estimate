from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from .checkers import AbftChecker, OpMeta, flatten_to_2d_for_mm
from .config import AbftConfig
from .stats import AbftStatsCollector


def _shape_key(a2: torch.Tensor, b2: torch.Tensor, c2: torch.Tensor) -> str:
    return f"A{tuple(a2.shape)}_B{tuple(b2.shape)}_C{tuple(c2.shape)}"


class AbftInjector(AbstractContextManager):
    def __init__(self, cfg: AbftConfig, checker: AbftChecker, stats: AbftStatsCollector):
        self.cfg = cfg
        self.checker = checker
        self.stats = stats
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(cfg.seed)

        self._orig_linear: Optional[Callable[..., torch.Tensor]] = None
        self._orig_matmul: Optional[Callable[..., torch.Tensor]] = None
        self._orig_bmm: Optional[Callable[..., torch.Tensor]] = None
        self._dump_done: bool = False

    def __enter__(self) -> "AbftInjector":
        self._orig_linear = F.linear
        self._orig_matmul = torch.matmul
        self._orig_bmm = torch.bmm

        if self.cfg.enable_linear:
            F.linear = self._wrap_linear(self._orig_linear)
        if self.cfg.enable_matmul:
            torch.matmul = self._wrap_matmul(self._orig_matmul)
        if self.cfg.enable_bmm:
            torch.bmm = self._wrap_bmm(self._orig_bmm)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._orig_linear is not None:
            F.linear = self._orig_linear
        if self._orig_matmul is not None:
            torch.matmul = self._orig_matmul
        if self._orig_bmm is not None:
            torch.bmm = self._orig_bmm
        return None

    def _sample(self) -> bool:
        if self.cfg.sample_rate >= 1.0:
            return True
        return bool(torch.rand(1, generator=self.rng).item() < self.cfg.sample_rate)

    def _maybe_inject_fault(self, out: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if not self.cfg.inject_fault:
            return out, False
        if not self.cfg.inject_phase_enabled(self.stats.current_phase):
            return out, False
        if self.cfg.inject_probability <= 0.0:
            return out, False
        if torch.rand(1, generator=self.rng).item() >= self.cfg.inject_probability:
            return out, False

        flat = out.reshape(-1)
        if flat.numel() == 0:
            return out, False
        idx = int(torch.randint(0, flat.numel(), (1,), generator=self.rng).item())
        perturbed = out.clone()
        perturbed.reshape(-1)[idx] += self.cfg.inject_magnitude
        self.stats.record_injection()
        return perturbed, True

    def _maybe_check(
        self,
        op_name: str,
        a_2d: Optional[torch.Tensor],
        b_2d: Optional[torch.Tensor],
        c_2d: Optional[torch.Tensor],
        injected: bool,
        bias_1d: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.cfg.enable:
            return
        if not self.cfg.phase_enabled(self.stats.current_phase):
            return
        if not self._sample():
            return
        if a_2d is None or b_2d is None or c_2d is None:
            return
        if a_2d.numel() == 0 or b_2d.numel() == 0 or c_2d.numel() == 0:
            return
        if a_2d.shape[1] != b_2d.shape[0]:
            return
        shape_key = _shape_key(a_2d, b_2d, c_2d)
        result = self.checker.check(
            a_2d,
            b_2d,
            c_2d,
            OpMeta(op_name=op_name, phase=self.stats.current_phase, shape_key=shape_key),
            bias_1d=bias_1d,
        )
        self.stats.record_check(op_name=op_name, shape_key=shape_key, result=result, injected=injected)
        self._maybe_dump_tensors(
            op_name=op_name,
            shape_key=shape_key,
            a_2d=a_2d,
            b_2d=b_2d,
            c_2d=c_2d,
            bias_1d=bias_1d,
        )

    def _maybe_dump_tensors(
        self,
        op_name: str,
        shape_key: str,
        a_2d: torch.Tensor,
        b_2d: torch.Tensor,
        c_2d: torch.Tensor,
        bias_1d: Optional[torch.Tensor],
    ) -> None:
        if self._dump_done:
            return
        if not self.cfg.dump_file:
            return
        if not self.cfg.dump_shape_key:
            return
        if not self.cfg.dump_phase_enabled(self.stats.current_phase):
            return
        if shape_key != self.cfg.dump_shape_key:
            return
        payload = {
            "meta": {
                "phase": self.stats.current_phase,
                "op_name": op_name,
                "shape_key": shape_key,
            },
            "a": a_2d.detach().cpu(),
            "b": b_2d.detach().cpu(),
            "c": c_2d.detach().cpu(),
            "bias": None if bias_1d is None else bias_1d.detach().cpu(),
        }
        torch.save(payload, self.cfg.dump_file)
        self._dump_done = True

    def _wrap_linear(self, fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def wrapped(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            out = fn(input, weight, bias)
            out2, injected = self._maybe_inject_fault(out)
            a2 = flatten_to_2d_for_mm(input, right_operand=False)
            b2 = flatten_to_2d_for_mm(weight.t(), right_operand=True)  # [k, n]
            c2 = flatten_to_2d_for_mm(out2, right_operand=False)
            self._maybe_check("linear", a2, b2, c2, injected, bias_1d=bias)
            return out2

        return wrapped

    def _wrap_matmul(self, fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def wrapped(input: torch.Tensor, other: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            out = fn(input, other, *args, **kwargs)
            out2, injected = self._maybe_inject_fault(out)
            a2 = flatten_to_2d_for_mm(input, right_operand=False)
            b2 = flatten_to_2d_for_mm(other, right_operand=True)
            c2 = flatten_to_2d_for_mm(out2, right_operand=False)
            self._maybe_check("matmul", a2, b2, c2, injected)
            return out2

        return wrapped

    def _wrap_bmm(self, fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def wrapped(input: torch.Tensor, mat2: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            out = fn(input, mat2, *args, **kwargs)
            out2, injected = self._maybe_inject_fault(out)
            # bmm 为 3D，逐 batch 做 2D 校验，保证任意 batch 大小均可覆盖
            bsz = input.shape[0]
            for bi in range(bsz):
                a2 = input[bi]
                b2 = mat2[bi]
                c2 = out2[bi]
                self._maybe_check("bmm", a2, b2, c2, injected)
            return out2

        return wrapped
