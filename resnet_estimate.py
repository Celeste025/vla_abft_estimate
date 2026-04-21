#!/usr/bin/env python3
"""
ResNet-152 + ABFT 解析估算：
- Conv 通过 im2col 等价到 GEMM，统计 (M,K,N) 与 FLOPs
- 非矩阵乘向量算子（BN/ReLU/Add/Pool）粗算 FLOPs
- 参数量与权重访存
- ABFT：单维校验 + 16 对齐，取较小分支
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


def flops_mm(m: int, k: int, n: int) -> float:
    return 2.0 * m * k * n


def ceil16(x: int) -> int:
    return ((x + 15) // 16) * 16


@dataclass
class GemmOp:
    name: str
    module: str
    m: int
    k: int
    n: int
    count: int = 1
    source: str = ""

    def flops(self) -> float:
        return flops_mm(self.m, self.k, self.n) * self.count

    def abft_vector_adds(self) -> float:
        if self.k <= 0 or self.m <= 0 or self.n <= 0:
            return 0.0
        a_row_sum = self.m * max(0, self.k - 1)
        b_col_sum = self.n * max(0, self.k - 1)
        dot_reduce = max(0, self.k - 1)
        c_sum = max(0, self.m * self.n - 1)
        return float((a_row_sum + b_col_sum + dot_reduce + c_sum) * self.count)

    def abft_vector_muls(self) -> float:
        if self.k <= 0:
            return 0.0
        return float(self.k * self.count)

    def abft_vector_nonlinear(self) -> float:
        return 0.0


@dataclass
class ModuleResult:
    name: str
    gemms: List[GemmOp] = field(default_factory=list)
    vector_adds: float = 0.0
    vector_muls: float = 0.0
    vector_nonlinear: float = 0.0
    params: int = 0
    notes: str = ""

    def finalize(self, bpe: float) -> Dict[str, Any]:
        mm = sum(g.flops() for g in self.gemms)
        abft_adds = sum(g.abft_vector_adds() for g in self.gemms)
        abft_muls = sum(g.abft_vector_muls() for g in self.gemms)
        abft_nonlin = sum(g.abft_vector_nonlinear() for g in self.gemms)
        vector_total = self.vector_adds + self.vector_muls + self.vector_nonlinear
        abft_total = abft_adds + abft_muls + abft_nonlin
        out = {
            "matmul_flops": mm,
            "matmul_flops_abft": mm,
            "delta_matmul_flops": 0.0,
            "vector_adds": self.vector_adds,
            "vector_muls": self.vector_muls,
            "vector_nonlinear": self.vector_nonlinear,
            "vector_flops": vector_total,
            "vector_abft_extra_adds": abft_adds,
            "vector_abft_extra_muls": abft_muls,
            "vector_abft_extra_nonlinear": abft_nonlin,
            "vector_flops_abft_extra": abft_total,
            "params": int(self.params),
            "weight_bytes": self.params * bpe,
            "notes": self.notes,
        }
        enrich_pct(out)
        return out


def enrich_pct(d: Dict[str, Any]) -> None:
    mm = float(d["matmul_flops"])
    dmm = float(d["delta_matmul_flops"])
    vec = float(d.get("vector_flops", 0.0))
    chk = float(d.get("vector_flops_abft_extra", 0.0))
    d["pct_delta_matmul_vs_baseline_mm"] = (dmm / mm * 100.0) if mm > 1e-12 else 0.0
    d["pct_matmul_abft_vs_baseline_mm"] = (
        d["matmul_flops_abft"] / mm * 100.0 if mm > 1e-12 else 100.0
    )
    d["pct_checksum_adds_vs_baseline_vector"] = (chk / vec * 100.0) if vec > 1e-12 else 0.0


def conv_out_hw(h: int, w: int, k: int, s: int, p: int, d: int = 1) -> Tuple[int, int]:
    oh = (h + 2 * p - d * (k - 1) - 1) // s + 1
    ow = (w + 2 * p - d * (k - 1) - 1) // s + 1
    return oh, ow


def conv_to_gemm(
    name: str,
    module: str,
    cin: int,
    cout: int,
    kh: int,
    kw: int,
    oh: int,
    ow: int,
    groups: int = 1,
    count: int = 1,
    stride: int = 1,
) -> GemmOp:
    m = oh * ow
    k = (cin // groups) * kh * kw
    n = cout
    src = f"conv{kh}x{kw} Cin={cin} Cout={cout} HoutxWout={oh}x{ow} stride={stride} groups={groups}"
    return GemmOp(name=name, module=module, m=m, k=k, n=n, count=count, source=src)


def add_bn_relu_pool_flops(
    v_add: float,
    v_mul: float,
    v_nonlin: float,
    c: int,
    h: int,
    w: int,
    with_bn: bool,
    with_relu: bool,
) -> Tuple[float, float, float]:
    n = c * h * w
    if with_bn:
        # 近似拆分：减均值1加1乘，方差归一1非线性，缩放平移1乘1加
        v_add += 2.0 * n
        v_mul += 1.0 * n
        v_nonlin += 1.0 * n
    if with_relu:
        v_nonlin += 1.0 * n
    return v_add, v_mul, v_nonlin


def resnet152_stats(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model = cfg["model"]
    inp = cfg["input"]
    bpe = float(cfg["memory_model"]["bytes_per_param"])

    layers = model["layers"]  # [3,8,36,3]
    exp = int(model.get("bottleneck_expansion", 4))
    num_classes = int(model.get("num_classes", 1000))
    b = int(inp["batch"])
    c = int(inp["channels"])
    h = int(inp["height"])
    w = int(inp["width"])
    assert b == 1, "当前脚本按 batch=1 统计。"

    stem = ModuleResult("stem")
    stage1 = ModuleResult("layer1")
    stage2 = ModuleResult("layer2")
    stage3 = ModuleResult("layer3")
    stage4 = ModuleResult("layer4")
    head = ModuleResult("fc_head")
    table_rows: List[Dict[str, Any]] = []

    # conv1: 7x7, 64, stride2, pad3
    oh, ow = conv_out_hw(h, w, 7, 2, 3)
    g = conv_to_gemm("stem_conv1", "stem", c, 64, 7, 7, oh, ow, stride=2)
    stem.gemms.append(g)
    table_rows.append({"module": "stem", "name": g.name, "m": g.m, "k": g.k, "n": g.n, "count": 1, "source": g.source, "flops": g.flops()})
    stem.params += c * 64 * 7 * 7  # conv no bias
    stem.params += 2 * 64  # BN
    stem.vector_adds, stem.vector_muls, stem.vector_nonlinear = add_bn_relu_pool_flops(
        stem.vector_adds, stem.vector_muls, stem.vector_nonlinear, 64, oh, ow, with_bn=True, with_relu=True
    )

    # maxpool 3x3 s2 p1
    ph, pw = conv_out_hw(oh, ow, 3, 2, 1)
    stem.vector_nonlinear += 3.0 * 3.0 * 64 * ph * pw

    in_c = 64
    cur_h, cur_w = ph, pw
    stage_modules = [stage1, stage2, stage3, stage4]
    planes_list = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]

    for si, (nblock, planes, s) in enumerate(zip(layers, planes_list, strides), start=1):
        mod = stage_modules[si - 1]
        out_c = planes * exp
        for bi in range(nblock):
            bstride = s if bi == 0 else 1
            # bottleneck conv1 1x1
            oh1, ow1 = conv_out_hw(cur_h, cur_w, 1, bstride, 0)
            g1 = conv_to_gemm(
                name=f"layer{si}_block{bi}_conv1",
                module=f"layer{si}",
                cin=in_c,
                cout=planes,
                kh=1,
                kw=1,
                oh=oh1,
                ow=ow1,
                stride=bstride,
            )
            mod.gemms.append(g1)
            table_rows.append({"module": f"layer{si}", "name": g1.name, "m": g1.m, "k": g1.k, "n": g1.n, "count": 1, "source": g1.source, "flops": g1.flops()})
            mod.params += in_c * planes
            mod.params += 2 * planes
            mod.vector_adds, mod.vector_muls, mod.vector_nonlinear = add_bn_relu_pool_flops(
                mod.vector_adds, mod.vector_muls, mod.vector_nonlinear, planes, oh1, ow1, with_bn=True, with_relu=True
            )

            # conv2 3x3
            oh2, ow2 = conv_out_hw(oh1, ow1, 3, 1, 1)
            g2 = conv_to_gemm(
                name=f"layer{si}_block{bi}_conv2",
                module=f"layer{si}",
                cin=planes,
                cout=planes,
                kh=3,
                kw=3,
                oh=oh2,
                ow=ow2,
                stride=1,
            )
            mod.gemms.append(g2)
            table_rows.append({"module": f"layer{si}", "name": g2.name, "m": g2.m, "k": g2.k, "n": g2.n, "count": 1, "source": g2.source, "flops": g2.flops()})
            mod.params += planes * planes * 3 * 3
            mod.params += 2 * planes
            mod.vector_adds, mod.vector_muls, mod.vector_nonlinear = add_bn_relu_pool_flops(
                mod.vector_adds, mod.vector_muls, mod.vector_nonlinear, planes, oh2, ow2, with_bn=True, with_relu=True
            )

            # conv3 1x1
            oh3, ow3 = conv_out_hw(oh2, ow2, 1, 1, 0)
            g3 = conv_to_gemm(
                name=f"layer{si}_block{bi}_conv3",
                module=f"layer{si}",
                cin=planes,
                cout=out_c,
                kh=1,
                kw=1,
                oh=oh3,
                ow=ow3,
                stride=1,
            )
            mod.gemms.append(g3)
            table_rows.append({"module": f"layer{si}", "name": g3.name, "m": g3.m, "k": g3.k, "n": g3.n, "count": 1, "source": g3.source, "flops": g3.flops()})
            mod.params += planes * out_c
            mod.params += 2 * out_c
            mod.vector_adds, mod.vector_muls, mod.vector_nonlinear = add_bn_relu_pool_flops(
                mod.vector_adds, mod.vector_muls, mod.vector_nonlinear, out_c, oh3, ow3, with_bn=True, with_relu=False
            )

            # downsample if needed
            if bi == 0 and (bstride != 1 or in_c != out_c):
                gd = conv_to_gemm(
                    name=f"layer{si}_block{bi}_downsample",
                    module=f"layer{si}",
                    cin=in_c,
                    cout=out_c,
                    kh=1,
                    kw=1,
                    oh=oh3,
                    ow=ow3,
                    stride=bstride,
                )
                mod.gemms.append(gd)
                table_rows.append({"module": f"layer{si}", "name": gd.name, "m": gd.m, "k": gd.k, "n": gd.n, "count": 1, "source": gd.source, "flops": gd.flops()})
                mod.params += in_c * out_c
                mod.params += 2 * out_c
                # BN for downsample
                mod.vector_adds += 2.0 * out_c * oh3 * ow3
                mod.vector_muls += 1.0 * out_c * oh3 * ow3
                mod.vector_nonlinear += 1.0 * out_c * oh3 * ow3

            # residual add + final relu
            mod.vector_adds += 1.0 * out_c * oh3 * ow3
            mod.vector_nonlinear += 1.0 * out_c * oh3 * ow3

            in_c = out_c
            cur_h, cur_w = oh3, ow3

    # avgpool + fc
    head.vector_adds += 1.0 * in_c * cur_h * cur_w
    head.gemms.append(GemmOp(name="fc", module="fc_head", m=1, k=in_c, n=num_classes, count=1, source=f"fc in={in_c} out={num_classes}"))
    table_rows.append({"module": "fc_head", "name": "fc", "m": 1, "k": in_c, "n": num_classes, "count": 1, "source": f"fc in={in_c} out={num_classes}", "flops": flops_mm(1, in_c, num_classes)})
    head.params += in_c * num_classes + num_classes

    module_map = {
        "stem": stem.finalize(bpe),
        "layer1": stage1.finalize(bpe),
        "layer2": stage2.finalize(bpe),
        "layer3": stage3.finalize(bpe),
        "layer4": stage4.finalize(bpe),
        "fc_head": head.finalize(bpe),
    }

    # totals
    total_mm = sum(v["matmul_flops"] for v in module_map.values())
    total_mm_abft = sum(v["matmul_flops_abft"] for v in module_map.values())
    total_add = sum(v["vector_adds"] for v in module_map.values())
    total_mul = sum(v["vector_muls"] for v in module_map.values())
    total_nonlin = sum(v["vector_nonlinear"] for v in module_map.values())
    total_vec = sum(v["vector_flops"] for v in module_map.values())
    total_chk = sum(v["vector_flops_abft_extra"] for v in module_map.values())
    total_abft_add = sum(v["vector_abft_extra_adds"] for v in module_map.values())
    total_abft_mul = sum(v["vector_abft_extra_muls"] for v in module_map.values())
    total_abft_nonlin = sum(v["vector_abft_extra_nonlinear"] for v in module_map.values())
    total_params = sum(int(v["params"]) for v in module_map.values())
    totals = {
        "matmul_flops": total_mm,
        "matmul_flops_abft": total_mm_abft,
        "delta_matmul_flops": 0.0,
        "vector_adds": total_add,
        "vector_muls": total_mul,
        "vector_nonlinear": total_nonlin,
        "vector_flops": total_vec,
        "vector_abft_extra_adds": total_abft_add,
        "vector_abft_extra_muls": total_abft_mul,
        "vector_abft_extra_nonlinear": total_abft_nonlin,
        "vector_flops_abft_extra": total_chk,
        "params": total_params,
        "weight_bytes": total_params * bpe,
    }
    enrich_pct(totals)

    # aggregated GEMM histogram
    agg: Dict[Tuple[int, int, int, str], Dict[str, Any]] = {}
    all_gemms: List[GemmOp] = []
    for mname in ("stem", "layer1", "layer2", "layer3", "layer4", "fc_head"):
        # recreate list from module objects
        pass

    # reuse table_rows for top
    top_rows = sorted(table_rows, key=lambda x: -x["flops"])[:40]

    out = {
        "model": "resnet152",
        "input_shape": [1, 3, 224, 224],
        "abft_rule": "rowcol_checksum_dot_and_outputsum",
        "modules": module_map,
        "totals": totals,
        "conv_to_gemm_table": table_rows,
        "top_gemm_table": top_rows,
    }
    return out, table_rows


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def build_markdown(out: Dict[str, Any]) -> str:
    t = out["totals"]
    lines = [
        "# ResNet-152 + ABFT 估算报告",
        "",
        "## 设定",
        f"- 输入：`{out['input_shape']}`",
        "- ABFT（新模式）：A行和 + B列和 + 点乘 + 输出矩阵全元素和校验；不做额外矩阵乘",
        "- 访存：仅权重访存 `params * 2`（BF16）",
        "",
        "## 基线统计（按模块）",
        "",
        "| 模块 | 矩阵乘 FLOPs | 向量加法 | 向量乘法 | 向量非线性 | 向量总计 | 参数量 | 权重访存(bytes) |",
        "|------|-------------|----------|----------|------------|----------|--------|----------------|",
    ]
    for key in ("stem", "layer1", "layer2", "layer3", "layer4", "fc_head"):
        x = out["modules"][key]
        lines.append(
            f"| {key} | {_fmt(x['matmul_flops'])} | {_fmt(x['vector_adds'])} | {_fmt(x['vector_muls'])} | "
            f"{_fmt(x['vector_nonlinear'])} | {_fmt(x['vector_flops'])} | {int(x['params']):,} | {_fmt(x['weight_bytes'])} |"
        )
    lines.append(
        f"| **总和** | {_fmt(t['matmul_flops'])} | {_fmt(t['vector_adds'])} | {_fmt(t['vector_muls'])} | "
        f"{_fmt(t['vector_nonlinear'])} | {_fmt(t['vector_flops'])} | {int(t['params']):,} | {_fmt(t['weight_bytes'])} |"
    )

    lines.extend(
        [
            "",
            "## ABFT 后开销与占比（按模块）",
            "",
            "| 模块 | 矩阵乘 FLOPs(ABFT) | Δ矩阵乘 | 新增加法 | 新增乘法 | 新增非线性 | 新增向量总计 | 新增总计/基线向量% |",
            "|------|---------------------|---------|----------|----------|------------|--------------|--------------------|",
        ]
    )
    for key in ("stem", "layer1", "layer2", "layer3", "layer4", "fc_head"):
        x = out["modules"][key]
        lines.append(
            f"| {key} | {_fmt(x['matmul_flops_abft'])} | {_fmt(x['delta_matmul_flops'])} | {_fmt(x['vector_abft_extra_adds'])} | "
            f"{_fmt(x['vector_abft_extra_muls'])} | {_fmt(x['vector_abft_extra_nonlinear'])} | {_fmt(x['vector_flops_abft_extra'])} | "
            f"{x['pct_checksum_adds_vs_baseline_vector']:.2f}% |"
        )
    lines.append(
        f"| **总和** | {_fmt(t['matmul_flops_abft'])} | {_fmt(t['delta_matmul_flops'])} | {_fmt(t['vector_abft_extra_adds'])} | "
        f"{_fmt(t['vector_abft_extra_muls'])} | {_fmt(t['vector_abft_extra_nonlinear'])} | {_fmt(t['vector_flops_abft_extra'])} | "
        f"{t['pct_checksum_adds_vs_baseline_vector']:.2f}% |"
    )

    lines.extend(
        [
            "",
            "## Conv->GEMM 维度表（Top 40 by FLOPs）",
            "",
            "| module | op | M | K | N | count | FLOPs | conv来源 |",
            "|--------|----|---|---|---|-------|-------|----------|",
        ]
    )
    for r in out["top_gemm_table"]:
        lines.append(
            f"| {r['module']} | {r['name']} | {r['m']} | {r['k']} | {r['n']} | {r['count']} | {_fmt(r['flops'])} | {r['source']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "resnet_config.json"))
    ap.add_argument("--json", default="resnet_out.json")
    ap.add_argument("--markdown", default="RESNET_RESULTS.md")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    out, _ = resnet152_stats(cfg)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已写入 {args.json}")

    md = build_markdown(out)
    with open(args.markdown, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"已写入 {args.markdown}")


if __name__ == "__main__":
    main()
