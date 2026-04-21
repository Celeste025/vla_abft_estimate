#!/usr/bin/env python3
"""
CogACT 风格：DINOv2-L + SigLIP vision + Qwen2.5-1.5B + DiT-L（单步×n）解析 FLOPs / 参数量 / ABFT。
访存按配置：权重访存 ≈ params × bytes_per_param；不含激活。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def flops_mm(m: int, k: int, n: int) -> float:
    return 2.0 * m * k * n


def ceil16(x: int) -> int:
    return ((x + 15) // 16) * 16


def bytes_mm(m: int, k: int, n: int, bpe: float) -> float:
    return bpe * (m * k + k * n + m * n)


def softmax_ops(num_vectors: int, vec_len: int) -> Tuple[float, float, float]:
    """按 softmax 分解统计：减max、exp、求和、归一化（除法按乘法计）。"""
    if num_vectors <= 0 or vec_len <= 0:
        return 0.0, 0.0, 0.0
    adds = num_vectors * (2 * vec_len - 1)
    muls = num_vectors * vec_len
    nonlin = num_vectors * vec_len
    return float(adds), float(muls), float(nonlin)


def layernorm_ops(num_vectors: int, hidden: int) -> Tuple[float, float, float]:
    """按 LayerNorm 分解统计：均值/方差归约 + 仿射。"""
    if num_vectors <= 0 or hidden <= 0:
        return 0.0, 0.0, 0.0
    adds = num_vectors * (4 * hidden - 1)
    muls = num_vectors * (3 * hidden + 2)
    nonlin = num_vectors  # rsqrt
    return float(adds), float(muls), float(nonlin)


def fetch_config(repo: str) -> Optional[Dict[str, Any]]:
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"[warn] 无法拉取 {url}: {e}", file=sys.stderr)
        return None


@dataclass
class GemmOp:
    name: str
    m: int
    k: int
    n: int
    count: int = 1
    module: str = ""

    def flops(self) -> float:
        return flops_mm(self.m, self.k, self.n) * self.count

    def abft_vector_adds(self) -> float:
        """新 ABFT 口径新增加法：A行和 + B列和 + 点乘归约 + C全元素和。"""
        if self.k <= 0 or self.m <= 0 or self.n <= 0:
            return 0.0
        a_row_sum = self.m * max(0, self.k - 1)
        b_col_sum = self.n * max(0, self.k - 1)
        dot_reduce = max(0, self.k - 1)
        c_sum = max(0, self.m * self.n - 1)
        return float((a_row_sum + b_col_sum + dot_reduce + c_sum) * self.count)

    def abft_vector_muls(self) -> float:
        """新 ABFT 口径新增乘法：A行和/B列和点乘。"""
        if self.k <= 0:
            return 0.0
        return float(self.k * self.count)

    def abft_vector_nonlinear(self) -> float:
        return 0.0


@dataclass
class ModuleResult:
    name: str
    gemms: List[GemmOp] = field(default_factory=list)
    matmul_flops: float = 0.0
    vector_adds: float = 0.0
    vector_muls: float = 0.0
    vector_nonlinear: float = 0.0
    params: int = 0
    notes: str = ""

    def finalize(self, p_abft: int) -> Dict[str, Any]:
        self.matmul_flops = sum(g.flops() for g in self.gemms)
        abft_adds = sum(g.abft_vector_adds() for g in self.gemms)
        abft_muls = sum(g.abft_vector_muls() for g in self.gemms)
        abft_nonlin = sum(g.abft_vector_nonlinear() for g in self.gemms)
        vector_total = self.vector_adds + self.vector_muls + self.vector_nonlinear
        abft_extra_total = abft_adds + abft_muls + abft_nonlin
        return {
            "matmul_flops": self.matmul_flops,
            "matmul_flops_abft": self.matmul_flops,
            "delta_matmul_flops": 0.0,
            "vector_adds": self.vector_adds,
            "vector_muls": self.vector_muls,
            "vector_nonlinear": self.vector_nonlinear,
            "vector_flops": vector_total,
            "vector_abft_extra_adds": abft_adds,
            "vector_abft_extra_muls": abft_muls,
            "vector_abft_extra_nonlinear": abft_nonlin,
            "vector_flops_abft_extra": abft_extra_total,
            "params": self.params,
            "weight_bytes": self.params * 2,
        }


def _vit_like_encoder(
    name: str,
    N: int,
    h: int,
    layers: int,
    heads: int,
    inter: int,
    module_tag: str,
) -> Tuple[List[GemmOp], float, float, float, float]:
    """HF ViT/DINOv2/SigLIP vision block 风格：4×Linear(h,h) + MLP。"""
    hd = h // heads
    gemms: List[GemmOp] = []
    v_add = 0.0
    v_mul = 0.0
    v_nonlin = 0.0
    for li in range(layers):
        for pname in ("q", "k", "v", "out"):
            gemms.append(
                GemmOp(f"{name}_L{li}_{pname}", N, h, h, 1, module_tag)
            )
        gemms.append(GemmOp(f"{name}_L{li}_qk", N, hd, N, heads, module_tag))
        gemms.append(GemmOp(f"{name}_L{li}_av", N, N, hd, heads, module_tag))
        # softmax: 按减max/exp/求和/归一化拆分
        a_sm, m_sm, n_sm = softmax_ops(heads * N, N)
        v_add += a_sm
        v_mul += m_sm
        v_nonlin += n_sm
        # 两次 LayerNorm（attention 前 + MLP 前）
        a_ln, m_ln, n_ln = layernorm_ops(2 * N, h)
        v_add += a_ln
        v_mul += m_ln
        v_nonlin += n_ln
        # MLP激活近似
        v_nonlin += 8.0 * N * inter
        gemms.append(GemmOp(f"{name}_L{li}_mlp1", N, h, inter, 1, module_tag))
        gemms.append(GemmOp(f"{name}_L{li}_mlp2", N, inter, h, 1, module_tag))
    return gemms, v_add + v_mul + v_nonlin, v_add, v_mul, v_nonlin


def dinov2_module(cfg: Dict[str, Any], module: str = "vision_dinov2") -> ModuleResult:
    h = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    ps = cfg["patch_size"]
    im = cfg["image_size"]
    mlp_ratio = cfg.get("mlp_ratio", 4)
    inter = int(h * mlp_ratio)
    gh = im // ps
    N = 1 + gh * gh
    r = ModuleResult("DINOv2-L")
    conv_flops = 2.0 * 3 * ps * ps * h * gh * gh
    r.vector_muls += conv_flops
    glist, _, v_add, v_mul, v_nonlin = _vit_like_encoder("dinov2", N, h, L, heads, inter, module)
    r.gemms.extend(glist)
    r.vector_adds += v_add
    r.vector_muls += v_mul
    r.vector_nonlinear += v_nonlin
    r.params = _params_dinov2(cfg, N, gh)
    r.notes = f"N_tokens={N}, patch_grid={gh}x{gh}"
    return r


def _params_dinov2(cfg: Dict[str, Any], N: int, gh: int) -> int:
    h = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    ps = cfg["patch_size"]
    p = 0
    p += 3 * ps * ps * h + h
    p += h + h + N * h
    p += 2 * h
    for _ in range(L):
        p += 4 * (h * h + h)
        p += 4 * h
        p += (h * (4 * h) + 4 * h) + ((4 * h) * h + h)
        p += 2 * h + 2 * (4 * h)
    p += 2 * h
    return p


def siglip_vision_module(vcfg: Dict[str, Any], module: str = "vision_siglip") -> ModuleResult:
    h = vcfg["hidden_size"]
    L = vcfg["num_hidden_layers"]
    heads = vcfg["num_attention_heads"]
    ps = vcfg["patch_size"]
    im = vcfg["image_size"]
    inter = vcfg["intermediate_size"]
    gh = im // ps
    Np = gh * gh
    N = Np + 1
    r = ModuleResult("SigLIP-Vision-So400m")
    conv_flops = 2.0 * 3 * ps * ps * h * gh * gh
    r.vector_muls += conv_flops
    glist, _, v_add, v_mul, v_nonlin = _vit_like_encoder("siglip", N, h, L, heads, inter, module)
    r.gemms.extend(glist)
    r.vector_adds += v_add
    r.vector_muls += v_mul
    r.vector_nonlinear += v_nonlin
    r.params = _params_siglip_vision(vcfg, N, gh)
    r.notes = f"N_tokens={N}, patch_grid={gh}x{gh}"
    return r


def _params_siglip_vision(vcfg: Dict[str, Any], N: int, gh: int) -> int:
    h = vcfg["hidden_size"]
    L = vcfg["num_hidden_layers"]
    ps = vcfg["patch_size"]
    inter = vcfg["intermediate_size"]
    p = 0
    p += 3 * ps * ps * h + h
    p += N * h + h
    p += 2 * h
    for _ in range(L):
        p += 4 * (h * h + h)
        p += 4 * h
        p += (h * inter + inter) + (inter * h + h)
        p += 2 * h + 2 * inter
    p += 2 * h
    return p


def qwen_prefill_module(
    cfg: Dict[str, Any],
    L: int,
    module: str = "llm",
) -> ModuleResult:
    h = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    nh = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    tie = cfg.get("tie_word_embeddings", True)
    hd = h // nh
    kvd = hd * nkv

    r = ModuleResult("Qwen2.5-1.5B_prefill")
    p_ct = vocab * h + h
    if not tie:
        p_ct += vocab * h
    r.params = p_ct

    gemms: List[GemmOp] = []
    v_add = 0.0
    v_mul = 0.0
    v_nonlin = 0.0

    for li in range(n_layers):
        gemms.append(GemmOp(f"qwen_L{li}_q", L, h, h, 1, module))
        gemms.append(GemmOp(f"qwen_L{li}_k", L, h, kvd, 1, module))
        gemms.append(GemmOp(f"qwen_L{li}_v", L, h, kvd, 1, module))
        gemms.append(GemmOp(f"qwen_L{li}_o", L, h, h, 1, module))
        gemms.append(GemmOp(f"qwen_L{li}_qk", L, hd, L, nh, module))
        gemms.append(GemmOp(f"qwen_L{li}_av", L, L, hd, nh, module))
        a_sm, m_sm, n_sm = softmax_ops(nh * L, L)
        v_add += a_sm
        v_mul += m_sm
        v_nonlin += n_sm
        # 预归一化 Transformer：每层两次 LN/RMSNorm，按 LN 近似拆分
        a_ln, m_ln, n_ln = layernorm_ops(2 * L, h)
        v_add += a_ln
        v_mul += m_ln
        v_nonlin += n_ln
        gemms.append(GemmOp(f"qwen_L{li}_gate", L, h, inter, 1, module))
        gemms.append(GemmOp(f"qwen_L{li}_up", L, h, inter, 1, module))
        v_nonlin += 4.0 * L * inter + L * inter
        gemms.append(GemmOp(f"qwen_L{li}_down", L, inter, h, 1, module))

    p_layers = 0
    for _ in range(n_layers):
        p_layers += 2 * (h + h)
        p_layers += (h * h + h) + 2 * (h * kvd + kvd) + (h * h + h)
        p_layers += 2 * (h * inter + inter) + (inter * h + h)
    r.params += p_layers
    if not tie:
        r.params += vocab * h + h

    r.gemms = gemms
    r.vector_adds = v_add
    r.vector_muls = v_mul
    r.vector_nonlinear = v_nonlin
    r.notes = f"L_prefill={L}, GQA kv_heads={nkv}"
    return r


def dit_single_step(
    T: int,
    h: int,
    layers: int,
    heads: int,
    mlp_ratio: int,
    in_c: int,
    patch_p: int,
    cond_dim: int,
    module: str = "dit",
) -> ModuleResult:
    inter = h * mlp_ratio
    hd = h // heads
    pdim = in_c * patch_p * patch_p
    r = ModuleResult("DiT-L_one_step")
    gemms: List[GemmOp] = []
    v_add = 0.0
    v_mul = 0.0
    v_nonlin = 0.0

    gemms.append(GemmOp("dit_patch_embed", T, pdim, h, 1, module))
    v_mul += 2.0 * T * h

    for li in range(layers):
        gemms.append(GemmOp(f"dit_L{li}_ada", 1, cond_dim, 9 * h, 1, module))
        for pname in ("q", "k", "v", "out"):
            gemms.append(GemmOp(f"dit_L{li}_{pname}", T, h, h, 1, module))
        gemms.append(GemmOp(f"dit_L{li}_qk", T, hd, T, heads, module))
        gemms.append(GemmOp(f"dit_L{li}_av", T, T, hd, heads, module))
        a_sm, m_sm, n_sm = softmax_ops(heads * T, T)
        v_add += a_sm
        v_mul += m_sm
        v_nonlin += n_sm
        a_ln, m_ln, n_ln = layernorm_ops(2 * T, h)
        v_add += a_ln
        v_mul += m_ln
        v_nonlin += n_ln
        v_nonlin += 8.0 * T * inter
        gemms.append(GemmOp(f"dit_L{li}_mlp1", T, h, inter, 1, module))
        gemms.append(GemmOp(f"dit_L{li}_mlp2", T, inter, h, 1, module))
        v_nonlin += 3.0 * T * h * 4

    gemms.append(GemmOp("dit_final_linear", T, h, pdim, 1, module))
    r.gemms = gemms
    r.vector_adds = v_add
    r.vector_muls = v_mul
    r.vector_nonlinear = v_nonlin
    r.params = _params_dit(T, h, layers, heads, inter, in_c, patch_p, cond_dim, pdim)
    r.notes = f"T_tokens={T}, layers={layers}"
    return r


def _params_dit(
    T: int,
    h: int,
    layers: int,
    heads: int,
    inter: int,
    in_c: int,
    patch_p: int,
    cond_dim: int,
    pdim: int,
) -> int:
    p = 0
    p += pdim * h + h
    p += T * h + h
    for _ in range(layers):
        p += cond_dim * (9 * h) + 9 * h
        p += 4 * (h * h + h)
        p += 4 * h
        p += (h * inter + inter) + (inter * h + h)
        p += 2 * h + 2 * inter
    p += h * pdim + pdim
    p += 2 * h
    return p


def _pct_delta_over_base(delta: float, base: float) -> float:
    """delta 相对 base 的增幅 %（如 Δ矩阵乘 / 基线矩阵乘）。"""
    return (delta / base * 100.0) if base > 1e-12 else 0.0


def _pct_ratio(num: float, den: float) -> float:
    return (num / den * 100.0) if den > 1e-12 else 0.0


def enrich_abft_pct(d: Dict[str, Any]) -> None:
    mm = float(d["matmul_flops"])
    dmm = float(d["delta_matmul_flops"])
    vec = float(d.get("vector_flops", 0.0))
    chk = float(d.get("vector_flops_abft_extra", 0.0))
    chk_add = float(d.get("vector_abft_extra_adds", 0.0))
    chk_mul = float(d.get("vector_abft_extra_muls", 0.0))
    # Δ矩阵乘 相对 ABFT 前矩阵乘
    d["pct_delta_matmul_vs_baseline_mm"] = _pct_delta_over_base(dmm, mm)
    # ABFT 后矩阵乘 相对 ABFT 前 = 100% + 上行（便于表格一眼读总倍率）
    d["pct_matmul_abft_vs_baseline_mm"] = (
        (float(d["matmul_flops_abft"]) / mm * 100.0) if mm > 1e-12 else 100.0
    )
    # 校验和加法 相对 ABFT 前向量类 FLOPs
    d["pct_checksum_adds_vs_baseline_vector"] = _pct_ratio(chk, vec)
    d["pct_abft_extra_adds_vs_baseline_vector"] = _pct_ratio(chk_add, vec)
    d["pct_abft_extra_muls_vs_baseline_vector"] = _pct_ratio(chk_mul, vec)


def gemm_histogram(gemms: List[GemmOp], top: int = 25) -> List[Dict[str, Any]]:
    agg: Dict[Tuple[int, int, int, str], Dict[str, Any]] = {}
    for g in gemms:
        key = (g.m, g.k, g.n, g.module)
        if key not in agg:
            agg[key] = {"m": g.m, "k": g.k, "n": g.n, "module": g.module, "count": 0, "flops": 0.0}
        agg[key]["count"] += g.count
        agg[key]["flops"] += g.flops()
    rows = sorted(agg.values(), key=lambda x: -x["flops"])[:top]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "cogact_config.json"))
    ap.add_argument("--dit-steps", type=int, default=None, help="覆盖 config 中的扩散步数 n")
    ap.add_argument("--offline", action="store_true", help="不访问 Hugging Face，仅用内置 fallback 与本地 config")
    ap.add_argument("--json", default="", help="输出 JSON")
    ap.add_argument("--markdown", default="", help="输出 Markdown 报告")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        bundle = json.load(f)

    p_abft = int(bundle["abft"]["pad_each_side"])
    bpe = float(bundle["memory_model"]["bytes_per_param"])

    fb_dino = {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "patch_size": 14,
        "image_size": 518,
        "mlp_ratio": 4,
    }
    fb_sig = {
        "hidden_size": 1152,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "patch_size": 14,
        "image_size": 384,
        "intermediate_size": 4304,
    }
    fb_qwen = {
        "hidden_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "intermediate_size": 8960,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    if args.offline:
        d_cfg, sig_full, q_cfg = fb_dino, {}, fb_qwen
        s_cfg = fb_sig
    else:
        d_cfg = fetch_config(bundle["huggingface_ids"]["dinov2"]) or fb_dino
        sig_full = fetch_config(bundle["huggingface_ids"]["siglip_vision"]) or {}
        s_cfg = sig_full.get("vision_config") or fb_sig
        q_cfg = fetch_config(bundle["huggingface_ids"]["llm"]) or fb_qwen

    dit_cfg = bundle["dit"]
    n_steps = int(args.dit_steps if args.dit_steps is not None else dit_cfg["num_diffusion_steps"])
    L_llm = int(bundle["llm"]["vision_fused_tokens"]) + int(bundle["llm"]["prefill_text_tokens"])

    mod_dino = dinov2_module(d_cfg)
    mod_sig = siglip_vision_module(s_cfg)
    mod_q = qwen_prefill_module(q_cfg, L_llm)
    mod_dit_1 = dit_single_step(
        T=int(dit_cfg["num_tokens"]),
        h=int(dit_cfg["hidden_size"]),
        layers=int(dit_cfg["num_layers"]),
        heads=int(dit_cfg["num_heads"]),
        mlp_ratio=int(dit_cfg["mlp_ratio"]),
        in_c=int(dit_cfg["in_channels"]),
        patch_p=int(dit_cfg["patch_size"]),
        cond_dim=int(dit_cfg.get("cond_dim", 1024)),
    )

    def pack(m: ModuleResult) -> Dict[str, Any]:
        d = m.finalize(p_abft)
        d["name"] = m.name
        d["notes"] = m.notes
        d["top_gemms"] = gemm_histogram(m.gemms, 15)
        return d

    pd = pack(mod_dino)
    ps = pack(mod_sig)
    pq = pack(mod_q)
    pdit1 = pack(mod_dit_1)

    pdit_n = {
        "name": f"DiT-L_x{n_steps}",
        "matmul_flops": pdit1["matmul_flops"] * n_steps,
        "matmul_flops_abft": pdit1["matmul_flops_abft"] * n_steps,
        "delta_matmul_flops": pdit1["delta_matmul_flops"] * n_steps,
        "vector_adds": pdit1["vector_adds"] * n_steps,
        "vector_muls": pdit1["vector_muls"] * n_steps,
        "vector_nonlinear": pdit1["vector_nonlinear"] * n_steps,
        "vector_flops": pdit1["vector_flops"] * n_steps,
        "vector_abft_extra_adds": pdit1["vector_abft_extra_adds"] * n_steps,
        "vector_abft_extra_muls": pdit1["vector_abft_extra_muls"] * n_steps,
        "vector_abft_extra_nonlinear": pdit1["vector_abft_extra_nonlinear"] * n_steps,
        "vector_flops_abft_extra": pdit1["vector_flops_abft_extra"] * n_steps,
        "params": int(pdit1["params"]),
        "weight_bytes": int(pdit1["params"]) * bpe,
        "num_diffusion_steps": n_steps,
    }
    enrich_abft_pct(pd)
    enrich_abft_pct(ps)
    enrich_abft_pct(pq)
    enrich_abft_pct(pdit1)
    enrich_abft_pct(pdit_n)

    vision_mm = pd["matmul_flops"] + ps["matmul_flops"]
    vision_vec = pd["vector_flops"] + ps["vector_flops"]
    vision_add = pd["vector_adds"] + ps["vector_adds"]
    vision_mul = pd["vector_muls"] + ps["vector_muls"]
    vision_nonlin = pd["vector_nonlinear"] + ps["vector_nonlinear"]
    vision_p = pd["params"] + ps["params"]
    vision_mm_abft = pd["matmul_flops_abft"] + ps["matmul_flops_abft"]
    vision_dmm = pd["delta_matmul_flops"] + ps["delta_matmul_flops"]
    vision_chk = pd["vector_flops_abft_extra"] + ps["vector_flops_abft_extra"]
    vision_abft_add = pd["vector_abft_extra_adds"] + ps["vector_abft_extra_adds"]
    vision_abft_mul = pd["vector_abft_extra_muls"] + ps["vector_abft_extra_muls"]
    vision_abft_nonlin = pd["vector_abft_extra_nonlinear"] + ps["vector_abft_extra_nonlinear"]

    total_mm = vision_mm + pq["matmul_flops"] + pdit_n["matmul_flops"]
    total_vec = vision_vec + pq["vector_flops"] + pdit_n["vector_flops"]
    total_add = vision_add + pq["vector_adds"] + pdit_n["vector_adds"]
    total_mul = vision_mul + pq["vector_muls"] + pdit_n["vector_muls"]
    total_nonlin = vision_nonlin + pq["vector_nonlinear"] + pdit_n["vector_nonlinear"]
    total_p = vision_p + pq["params"] + pdit1["params"]
    total_mm_abft = vision_mm_abft + pq["matmul_flops_abft"] + pdit_n["matmul_flops_abft"]
    total_dmm = total_mm_abft - total_mm
    total_chk = (
        pd["vector_flops_abft_extra"]
        + ps["vector_flops_abft_extra"]
        + pq["vector_flops_abft_extra"]
        + pdit_n["vector_flops_abft_extra"]
    )
    total_abft_add = (
        pd["vector_abft_extra_adds"]
        + ps["vector_abft_extra_adds"]
        + pq["vector_abft_extra_adds"]
        + pdit_n["vector_abft_extra_adds"]
    )
    total_abft_mul = (
        pd["vector_abft_extra_muls"]
        + ps["vector_abft_extra_muls"]
        + pq["vector_abft_extra_muls"]
        + pdit_n["vector_abft_extra_muls"]
    )
    total_abft_nonlin = (
        pd["vector_abft_extra_nonlinear"]
        + ps["vector_abft_extra_nonlinear"]
        + pq["vector_abft_extra_nonlinear"]
        + pdit_n["vector_abft_extra_nonlinear"]
    )

    vision_combined: Dict[str, Any] = {
        "matmul_flops": vision_mm,
        "matmul_flops_abft": vision_mm_abft,
        "delta_matmul_flops": vision_dmm,
        "vector_adds": vision_add,
        "vector_muls": vision_mul,
        "vector_nonlinear": vision_nonlin,
        "vector_flops": vision_vec,
        "vector_abft_extra_adds": vision_abft_add,
        "vector_abft_extra_muls": vision_abft_mul,
        "vector_abft_extra_nonlinear": vision_abft_nonlin,
        "vector_flops_abft_extra": vision_chk,
        "params": vision_p,
        "weight_bytes": vision_p * bpe,
    }
    enrich_abft_pct(vision_combined)

    out: Dict[str, Any] = {
        "config_path": args.config,
        "L_llm_prefill": L_llm,
        "dit_diffusion_steps_n": n_steps,
        "abft_pad_each_side": p_abft,
        "memory_model_weight_bytes_only": True,
        "modules": {
            "vision_dinov2": pd,
            "vision_siglip": ps,
            "vision_combined": vision_combined,
            "llm_qwen_prefill": pq,
            "dit_single_step": pdit1,
            "dit_scaled_n_steps": pdit_n,
        },
        "totals": {
            "matmul_flops": total_mm,
            "matmul_flops_abft": total_mm_abft,
            "delta_matmul_flops": total_dmm,
            "relative_delta_matmul": total_dmm / max(total_mm, 1e-9),
            "vector_adds": total_add,
            "vector_muls": total_mul,
            "vector_nonlinear": total_nonlin,
            "vector_flops": total_vec,
            "vector_abft_extra_adds": total_abft_add,
            "vector_abft_extra_muls": total_abft_mul,
            "vector_abft_extra_nonlinear": total_abft_nonlin,
            "vector_flops_abft_checksum_adds": total_chk,
            "params": total_p,
            "weight_bytes": total_p * bpe,
            "pct_delta_matmul_vs_baseline_mm": _pct_delta_over_base(total_dmm, total_mm),
            "pct_matmul_abft_vs_baseline_mm": (
                (total_mm_abft / total_mm * 100.0) if total_mm > 1e-12 else 100.0
            ),
            "pct_checksum_adds_vs_baseline_vector": _pct_ratio(total_chk, total_vec),
        },
        "characteristic_gemm_table": gemm_histogram(
            mod_dino.gemms + mod_sig.gemms + mod_q.gemms + mod_dit_1.gemms, 40
        ),
    }

    js = json.dumps(out, indent=2, ensure_ascii=False)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(js)
        print(f"已写入 {args.json}")
    else:
        print(js)

    if args.markdown:
        lines = [
            "# CogACT 风格估算：DINOv2-L + SigLIP + Qwen2.5-1.5B + DiT-L×n",
            "",
            "## 设定",
            f"- LLM prefill 序列长度 L = vision_fused_tokens + text = **{L_llm}**",
            f"- DiT 扩散步数 **n = {n_steps}**（单步 FLOPs × n）",
            "- ABFT（新模式）：A 行和 + B 列和 + 二者点乘 + C 全元素和校验，不做额外矩阵乘与16对齐。",
            "- **访存**：仅统计权重 bytes ≈ params×2（BF16）；校验和现场算，不增加持久参数",
            "",
            "## 矩阵乘与向量算子（单次「感知+语言 prefill+扩散步」）",
            "",
            "| 模块 | 矩阵乘 FLOPs | 向量加法 | 向量乘法 | 向量非线性 | 向量总计 | 参数量 | 权重访存(bytes) |",
            "|------|-------------|----------|----------|------------|----------|--------|----------------|",
        ]
        for label, key in [
            ("Vision DINOv2-L", "vision_dinov2"),
            ("Vision SigLIP", "vision_siglip"),
            ("Vision 合计", "vision_combined"),
            ("LLM Qwen2.5-1.5B prefill", "llm_qwen_prefill"),
            (f"DiT-L × n={n_steps}", "dit_scaled_n_steps"),
        ]:
            x = out["modules"][key]
            wb = x.get("weight_bytes", x["params"] * bpe)
            lines.append(
                f"| {label} | {_fmt(x['matmul_flops'])} | {_fmt(x['vector_adds'])} | "
                f"{_fmt(x['vector_muls'])} | {_fmt(x['vector_nonlinear'])} | {_fmt(x['vector_flops'])} | "
                f"{x['params']:,} | {_fmt(wb)} |"
            )
        t = out["totals"]
        lines.append(
            f"| **总和** | {_fmt(t['matmul_flops'])} | {_fmt(t['vector_adds'])} | {_fmt(t['vector_muls'])} | "
            f"{_fmt(t['vector_nonlinear'])} | {_fmt(t['vector_flops'])} | {t['params']:,} | {_fmt(t['weight_bytes'])} |"
        )
        lines.extend(
            [
                "",
                "## ABFT 新增向量开销（按新模式）",
                "",
                "列说明：新增向量总计=新增加法+新增乘法+新增非线性，后两列给出新增总计相对基线向量总量的比例。",
                "",
                "| 模块 | 矩阵乘 FLOPs (ABFT) | Δ矩阵乘 | 新增加法 | 新增乘法 | 新增非线性 | 新增向量总计 | 新增总计/基线向量% |",
                "|------|---------------------|---------|----------|----------|------------|--------------|--------------------|",
            ]
        )
        for label, key in [
            ("Vision DINOv2-L", "vision_dinov2"),
            ("Vision SigLIP", "vision_siglip"),
            ("Vision 合计", "vision_combined"),
            ("LLM prefill", "llm_qwen_prefill"),
            (f"DiT-L × n", "dit_scaled_n_steps"),
        ]:
            x = out["modules"][key]
            lines.append(
                f"| {label} | {_fmt(x['matmul_flops_abft'])} | {_fmt(x['delta_matmul_flops'])} | "
                f"{_fmt(x.get('vector_abft_extra_adds', 0))} | {_fmt(x.get('vector_abft_extra_muls', 0))} | "
                f"{_fmt(x.get('vector_abft_extra_nonlinear', 0))} | {_fmt(x.get('vector_flops_abft_extra', 0))} | "
                f"{x['pct_checksum_adds_vs_baseline_vector']:.2f}% |"
            )
        lines.append(
            f"| **总和** | {_fmt(t['matmul_flops_abft'])} | {_fmt(t['delta_matmul_flops'])} | "
            f"{_fmt(t['vector_abft_extra_adds'])} | {_fmt(t['vector_abft_extra_muls'])} | "
            f"{_fmt(t['vector_abft_extra_nonlinear'])} | {_fmt(t['vector_flops_abft_checksum_adds'])} | "
            f"{t['pct_checksum_adds_vs_baseline_vector']:.2f}% |"
        )
        lines.extend(
            [
                "",
                "## 各模块：特征矩阵乘维度（按 FLOPs 排序 Top）",
            ]
        )

        def _mod_table(title: str, rows: List[Dict[str, Any]]) -> None:
            lines.append("")
            lines.append(f"### {title}")
            lines.append("")
            lines.append("| M | K | N | 累计次数 | FLOPs |")
            lines.append("|---|---|---|---------|-------|")
            for row in rows[:12]:
                lines.append(
                    f"| {row['m']} | {row['k']} | {row['n']} | {row['count']} | {_fmt(row['flops'])} |"
                )

        _mod_table("Vision DINOv2-L", out["modules"]["vision_dinov2"]["top_gemms"])
        _mod_table("Vision SigLIP", out["modules"]["vision_siglip"]["top_gemms"])
        _mod_table("LLM Qwen2.5-1.5B (prefill)", out["modules"]["llm_qwen_prefill"]["top_gemms"])
        _mod_table("DiT-L 单步", out["modules"]["dit_single_step"]["top_gemms"])

        lines.extend(
            [
                "",
                "## 代表性 GEMM 形状（全模块合并 Top）",
                "",
                "| M | K | N | 出现次数(含层重复) | FLOPs | module |",
                "|---|---|---|-------------------|-------|--------|",
            ]
        )
        for row in out["characteristic_gemm_table"][:30]:
            lines.append(
                f"| {row['m']} | {row['k']} | {row['n']} | {row['count']} | {_fmt(row['flops'])} | {row['module']} |"
            )
        lines.extend(
            [
                "",
                "## 说明",
                "- DiT 为 **自注意力** 块 + adaLN 线性（cond→9h）简化；若 CogACT 在动作头加 cross-attn，请在代码中追加对应 GEMM。",
                "- 向量分解口径：Softmax 按 `adds=2S-1, muls=S, nonlinear=S`（每个长度S向量）统计；LayerNorm 按 `adds=4H-1, muls=3H+2, nonlinear=1`（每个长度H向量）统计。",
                "- Qwen 的 RMSNorm 近似按 LayerNorm 口径计入向量加/乘/非线性；除法按乘法等价成本统计。",
                "- 无网络时用 `--offline`（与 HF `config.json` 一致的默认超参）。",
                "- 重新生成：`python3 cogact_estimate.py --offline --config cogact_config.json --json cogact_out.json --markdown COGACT_RESULTS.md --dit-steps 10`",
            ]
        )
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"已写入 {args.markdown}")


def _fmt(x: float) -> str:
    return f"{x:.4g}"


if __name__ == "__main__":
    main()
