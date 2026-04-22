from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover - runtime env dependent
    _TRITON_OK = False


def available() -> bool:
    return _TRITON_OK


if _TRITON_OK:
    # 保留 triton 依赖检查与后端入口，具体归约先走高性能 torch reduce，
    # 避免当前 Triton 在跨步/小规模归约下产生过多 launch 与低效访存。
    pass

def triton_corner_sum(
    a_2d: torch.Tensor,
    b_2d: torch.Tensor,
    c_2d: torch.Tensor,
    bias_1d: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not _TRITON_OK:
        raise RuntimeError("triton is not available")
    if not (a_2d.is_cuda and b_2d.is_cuda and c_2d.is_cuda):
        raise RuntimeError("triton backend requires CUDA tensors")
    if a_2d.dim() != 2 or b_2d.dim() != 2 or c_2d.dim() != 2:
        raise RuntimeError("triton backend expects 2D tensors")
    a = a_2d if a_2d.is_contiguous() else a_2d.contiguous()
    b = b_2d if b_2d.is_contiguous() else b_2d.contiguous()
    c = c_2d if c_2d.is_contiguous() else c_2d.contiguous()
    m, _ = a.shape
    # 使用单次高效归约，减少 kernel launch 与中间张量管理开销。
    s_a = torch.sum(a, dim=0, dtype=torch.float32)
    s_b = torch.sum(b, dim=1, dtype=torch.float32)
    abft_corner = torch.dot(s_a, s_b)
    sum_c = torch.sum(c, dtype=torch.float32)

    if bias_1d is not None:
        bias = bias_1d if bias_1d.is_contiguous() else bias_1d.contiguous()
        abft_corner = abft_corner + float(m) * torch.sum(bias, dtype=torch.float32)

    return abft_corner, sum_c
