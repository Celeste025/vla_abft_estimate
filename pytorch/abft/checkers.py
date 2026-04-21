from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from .config import AbftConfig
from .stats import CheckResult


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
    ) -> CheckResult:
        raise NotImplementedError


class CheapSumChecker(AbftChecker):
    """
    经济型 ABFT:
    sA = A按行求和 (m)
    sB = B按列求和 (n)
    abft_corner = sum(sA) * sum(sB)
    sumC = sum(C)
    """

    def __init__(self, cfg: AbftConfig) -> None:
        self.cfg = cfg

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
    ) -> CheckResult:
        a = self._cast(a_2d)
        b = self._cast(b_2d)
        c = self._cast(c_2d)

        s_a = a.sum(dim=1)  # [m]
        s_b = b.sum(dim=0)  # [n]
        abft_corner = s_a.sum() * s_b.sum()
        sum_c = c.sum()

        diff = torch.abs(abft_corner - sum_c)
        denom = torch.maximum(torch.abs(sum_c), torch.tensor(1.0, device=sum_c.device))
        rel = diff / denom
        ok = bool((diff <= self.cfg.tol_abs) or (rel <= self.cfg.tol_rel))
        return CheckResult(
            ok=ok,
            abs_err=float(diff.item()),
            rel_err=float(rel.item()),
            abft_corner=float(abft_corner.item()),
            sum_c=float(sum_c.item()),
        )


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
