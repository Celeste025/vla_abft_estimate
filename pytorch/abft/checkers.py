from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from .config import AbftConfig
from .fused_backend import fused_corner_sum
from .stats import CheckResult
from .triton_backend import available as triton_available
from .triton_backend import triton_corner_sum


@dataclass
class OpMeta:
    op_name: str
    phase: str
    shape_key: str


class AbftChecker(ABC):
    @abstractmethod
    def check(
        self,
        a_2d: torch.Tensor,
        b_2d: torch.Tensor,
        c_2d: torch.Tensor,
        meta: OpMeta,
        bias_1d: Optional[torch.Tensor] = None,
    ) -> CheckResult:
        raise NotImplementedError


class CheapSumChecker(AbftChecker):
    """
    经济型 ABFT:
    sA = A按列求和 (k)
    sB = B按行求和 (k)
    abft_corner = dot(sA, sB)
    sumC = sum(C)
    """

    def __init__(self, cfg: AbftConfig) -> None:
        self.cfg = cfg
        self._fused_warned: bool = False
        self._triton_warned: bool = False

    def _cast(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.force_fp32_check:
            return x.float()
        return x

    def check(
        self,
        a_2d: torch.Tensor,
        b_2d: torch.Tensor,
        c_2d: torch.Tensor,
        meta: OpMeta,
        bias_1d: Optional[torch.Tensor] = None,
    ) -> CheckResult:
        use_fused = (
            self.cfg.checker_backend == "fused"
            and a_2d.is_cuda
            and b_2d.is_cuda
            and c_2d.is_cuda
        )
        use_triton = (
            self.cfg.checker_backend == "triton"
            and triton_available()
            and a_2d.is_cuda
            and b_2d.is_cuda
            and c_2d.is_cuda
        )
        if use_triton:
            try:
                abft_corner, sum_c = triton_corner_sum(a_2d, b_2d, c_2d, bias_1d)
            except Exception:
                if not self._triton_warned:
                    self._triton_warned = True
                    print("[ABFT-WARN] triton checker unavailable, fallback to python backend.")
                abft_corner, sum_c = self._python_corner_sum(a_2d, b_2d, c_2d, bias_1d)
        elif use_fused:
            try:
                abft_corner, sum_c = fused_corner_sum(a_2d, b_2d, c_2d, bias_1d)
            except Exception:
                # 融合后端异常时自动回退，避免中断基准流程。
                if not self._fused_warned:
                    self._fused_warned = True
                    print("[ABFT-WARN] fused checker unavailable, fallback to python backend.")
                abft_corner, sum_c = self._python_corner_sum(a_2d, b_2d, c_2d, bias_1d)
        else:
            abft_corner, sum_c = self._python_corner_sum(a_2d, b_2d, c_2d, bias_1d)

        if not self.cfg.record_enable:
            return CheckResult(ok=True, abs_err=0.0, rel_err=0.0, abft_corner=0.0, sum_c=0.0)

        diff = torch.abs(abft_corner - sum_c)
        denom = torch.clamp(torch.abs(sum_c), min=1.0)
        rel = diff / denom
        ok = bool((diff <= self.cfg.tol_abs) or (rel <= self.cfg.tol_rel))
        vals = torch.stack((diff, rel, abft_corner, sum_c)).detach().cpu().tolist()
        return CheckResult(
            ok=ok,
            abs_err=float(vals[0]),
            rel_err=float(vals[1]),
            abft_corner=float(vals[2]),
            sum_c=float(vals[3]),
        )

    def _python_corner_sum(
        self,
        a_2d: torch.Tensor,
        b_2d: torch.Tensor,
        c_2d: torch.Tensor,
        bias_1d: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_a = a_2d.sum(dim=0)  # [k]
        s_b = b_2d.sum(dim=1)  # [k]
        abft_corner = torch.dot(s_a, s_b)
        if bias_1d is not None:
            m = a_2d.shape[0]
            abft_corner = abft_corner + float(m) * bias_1d.sum()
        sum_c = c_2d.sum()
        return abft_corner, sum_c


def flatten_to_2d_for_mm(x: torch.Tensor, right_operand: bool = False) -> Optional[torch.Tensor]:
    """
    将任意可能用于 matmul/linear 的张量规整为2D矩阵。
    - 左操作数: [..., m, k] -> [m*, k]
    - 右操作数: [..., k, n] -> [k, n*] (取最后两维，前缀要求可压平)
    """
    if x.dim() < 2:
        return None
    if not right_operand:
        k = x.shape[-1]
        return x.reshape(-1, k)
    k = x.shape[-2]
    n = x.shape[-1]
    if x.dim() == 2:
        return x
    # 对于高维右操作数，常见场景是广播权重；这里只处理前缀乘积为1的情况
    prefix = 1
    for d in x.shape[:-2]:
        prefix *= int(d)
    if prefix != 1:
        return None
    return x.reshape(k, n)
