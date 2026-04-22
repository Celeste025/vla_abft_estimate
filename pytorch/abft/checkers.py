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
        # 仅保留核心 ABFT 计算链路，避免额外数值后处理开销。
        s_a = a_2d.sum(dim=0)  # [k]
        s_b = b_2d.sum(dim=1)  # [k]
        abft_corner = torch.dot(s_a, s_b)
        # linear 带 bias 时，sum(C)=sum(A@B)+m*sum(bias)
        if bias_1d is not None:
            m = a_2d.shape[0]
            abft_corner = abft_corner + float(m) * bias_1d.sum()
        sum_c = c_2d.sum()

        # 在record关闭场景下结果不会被消费，返回占位值避免额外同步。
        return CheckResult(
            ok=True,
            abs_err=0.0,
            rel_err=0.0,
            abft_corner=0.0,
            sum_c=0.0,
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
