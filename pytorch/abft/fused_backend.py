from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load

_EXT = None
_EXT_LOAD_ERR: Optional[Exception] = None


def _load_ext():
    global _EXT, _EXT_LOAD_ERR
    if _EXT is not None:
        return _EXT
    if _EXT_LOAD_ERR is not None:
        raise _EXT_LOAD_ERR

    base = Path(__file__).resolve().parent / "kernels"
    cpp_src = str(base / "abft_fused.cpp")
    cu_src = str(base / "abft_fused.cu")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        cc = Path(conda_prefix) / "bin" / "x86_64-conda-linux-gnu-cc"
        cxx = Path(conda_prefix) / "bin" / "x86_64-conda-linux-gnu-c++"
        if cc.exists():
            os.environ.setdefault("CC", str(cc))
        if cxx.exists():
            os.environ.setdefault("CXX", str(cxx))
    try:
        _EXT = load(
            name="abft_fused_ext",
            sources=[cpp_src, cu_src],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
        return _EXT
    except Exception as e:  # pragma: no cover - runtime env dependent
        _EXT_LOAD_ERR = e
        raise


def fused_corner_sum(
    a_2d: torch.Tensor,
    b_2d: torch.Tensor,
    c_2d: torch.Tensor,
    bias_1d: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ext = _load_ext()
    bias_arg = None if bias_1d is None else bias_1d.contiguous()
    out = ext.abft_fused(
        a_2d.contiguous(),
        b_2d.contiguous(),
        c_2d.contiguous(),
        bias_arg,
    )
    return out[0], out[1]
