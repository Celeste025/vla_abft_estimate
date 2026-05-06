"""将 Qwen 风格解码层中的 Linear 替换为「可选输入清零 + 线性 + 输出清零」包装。

``linear_backend=triton`` / ``triton_plain`` 使用 [triton_protect_linear](triton_protect_linear.py) 的展平 2D Triton GEMM；
前者 kernel 内带 outlier 清零，后者仅 matmul+bias（对照用，无 torch.compile）。
``linear_backend=pt2_funct`` 使用 ``@torch.compile(mode="reduce-overhead")`` 包一截
``F.linear + torch.where``（decode 形状多变时 ``max-autotune`` 易反复 autotune，故默认不用）。
并尝试拉高 ``torch._dynamo.config.cache_size_limit`` 减轻重编译。
请勿与「整模 torch.compile」叠用；推荐 ``masked_fill + 整模 compile`` 作默认路径。

语义对齐 inject.py 中 qwen_decoder 站点：
- q/k/v/gate/up/down：仅在线性输出上做 |x|>threshold → 0。
- o_proj：与 attn_core（o_proj 输入）+ o_proj 输出两处 hook 一致：
  先对输入做阈值清零，再 matmul，再对输出做阈值清零。
"""
from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LinearBackend = Literal["pt2_funct", "masked_fill", "triton", "triton_plain"]


@torch.compile(mode="reduce-overhead")
def fused_linear_outlier_pt2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    threshold: float,
) -> torch.Tensor:
    """F.linear + 输出侧异常值清零；Inductor 有机会融成更少 kernel。"""
    y = F.linear(x, weight, bias)
    return torch.where(y.abs() > threshold, torch.zeros_like(y), y)


@torch.compile(mode="reduce-overhead")
def fused_linear_outlier_pt2_inout(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    threshold: float,
) -> torch.Tensor:
    """o_proj：输入清零 + F.linear + 输出清零（对齐 attn_core + o_proj 双站点）。"""
    x = torch.where(x.abs() > threshold, torch.zeros_like(x), x)
    y = F.linear(x, weight, bias)
    return torch.where(y.abs() > threshold, torch.zeros_like(y), y)


class ProtectLinear(nn.Module):
    """Wrap nn.Linear：可选输入保护 + F.linear + 输出保护。"""

    def __init__(
        self,
        linear: nn.Module,
        threshold: float,
        *,
        protect_input: bool = False,
        linear_backend: Literal["pt2_funct", "masked_fill"] = "masked_fill",
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(linear)}")
        if linear_backend not in ("pt2_funct", "masked_fill"):
            raise ValueError(f"unsupported linear_backend={linear_backend!r}")
        self.threshold = float(threshold)
        self.protect_input = bool(protect_input)
        self.linear_backend: Literal["pt2_funct", "masked_fill"] = linear_backend
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.threshold
        if self.linear_backend == "pt2_funct":
            if self.protect_input:
                return fused_linear_outlier_pt2_inout(x, self.weight, self.bias, t)
            return fused_linear_outlier_pt2(x, self.weight, self.bias, t)
        if self.protect_input:
            x = x.masked_fill(x.abs() > t, 0)
        y = F.linear(x, self.weight, self.bias)
        return y.masked_fill(y.abs() > t, 0)


def apply_protect_linears_qwen(
    model: torch.nn.Module,
    threshold: float,
    *,
    linear_backend: LinearBackend = "masked_fill",
) -> int:
    """
    就地替换 `model.model.layers[*]` 下注意力与 MLP 中的 Linear。
    返回替换的 Linear 个数（应与 7 * num_hidden_layers 一致）。
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("expected HuggingFace-style CausalLM with .model.layers")
    layers = model.model.layers
    n = 0
    for layer in layers:
        attn = layer.self_attn
        mlp = layer.mlp
        pairs: Tuple[Tuple[str, nn.Module, bool], ...] = (
            ("q_proj", attn.q_proj, False),
            ("k_proj", attn.k_proj, False),
            ("v_proj", attn.v_proj, False),
            ("o_proj", attn.o_proj, True),
            ("gate_proj", mlp.gate_proj, False),
            ("up_proj", mlp.up_proj, False),
            ("down_proj", mlp.down_proj, False),
        )
        lb = str(linear_backend).strip().lower()
        use_triton = lb in ("triton", "triton_plain")
        triton_do_outlier = lb == "triton"
        if use_triton:
            from triton_protect_linear import TritonProtectLinear
        for name, mod, pin in pairs:
            if not isinstance(mod, nn.Linear):
                raise TypeError(
                    f"expected nn.Linear at {name}, got {type(mod)} "
                    "(already wrapped or unexpected architecture)"
                )
            if use_triton:
                wrapped = TritonProtectLinear(
                    mod,
                    threshold,
                    protect_input=False,
                    do_outlier=triton_do_outlier,
                )
            else:
                wrapped = ProtectLinear(
                    mod,
                    threshold,
                    protect_input=pin,
                    linear_backend=linear_backend,
                )
            if name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(attn, name, wrapped)
            else:
                setattr(mlp, name, wrapped)
            n += 1
    return n


def compile_model_for_latency(
    model: torch.nn.Module,
    *,
    compile_mode: str = "default",
    fullgraph: bool = False,
) -> torch.nn.Module:
    """
    torch.compile 整模（generate 下序列长变化，dynamic=True 更稳）。
    """
    mode = str(compile_mode).strip().lower()
    if mode not in {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}:
        raise ValueError(f"unsupported compile_mode={compile_mode!r}")
    return torch.compile(model, mode=mode, dynamic=True, fullgraph=fullgraph)
