import argparse
from typing import List, Tuple

import torch

from triton_gemm_abft_fused import matmul_abft_fused
from triton_gemm_baseline import matmul_baseline


def _dtype_from_str(name: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    return mapping[name]


def _default_shapes() -> List[Tuple[int, int, int]]:
    return [(4096, 4096, 4096), (8192, 4096, 4096)]


def verify_one_shape(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    matmul_atol: float,
    matmul_rtol: float,
    abft_tol: float,
    seed: int,
):
    torch.manual_seed(seed)
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)

    c_baseline = matmul_baseline(a, b)
    out = matmul_abft_fused(a, b)
    c_fused = out["c"]
    c_torch = torch.matmul(a, b).to(torch.float32)

    baseline_ok = torch.allclose(c_baseline, c_torch, atol=matmul_atol, rtol=matmul_rtol)
    fused_ok = torch.allclose(c_fused, c_torch, atol=matmul_atol, rtol=matmul_rtol)
    fused_vs_baseline_ok = torch.allclose(c_fused, c_baseline, atol=matmul_atol, rtol=matmul_rtol)
    abft_abs = out["abft_abs_error"].item()
    abft_rel = out["abft_rel_error"].item()
    abft_ok = abft_abs <= abft_tol

    return {
        "shape": (m, n, k),
        "baseline_ok": baseline_ok,
        "fused_ok": fused_ok,
        "fused_vs_baseline_ok": fused_vs_baseline_ok,
        "abft_ok": abft_ok,
        "abft_abs_error": abft_abs,
        "abft_rel_error": abft_rel,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Verify Triton baseline/fused ABFT GEMM.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--matmul-atol", type=float, default=1e-2)
    parser.add_argument("--matmul-rtol", type=float, default=1e-2)
    parser.add_argument("--abft-tol", type=float, default=5e-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shape",
        action="append",
        help="shape as M,N,K. Can be passed multiple times.",
    )
    return parser.parse_args()


def _parse_shapes(raw_shapes):
    if not raw_shapes:
        return _default_shapes()
    parsed = []
    for raw in raw_shapes:
        m_str, n_str, k_str = raw.split(",")
        parsed.append((int(m_str), int(n_str), int(k_str)))
    return parsed


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required.")
    args = _parse_args()
    dtype = _dtype_from_str(args.dtype)
    shapes = _parse_shapes(args.shape)

    all_pass = True
    for m, n, k in shapes:
        stats = verify_one_shape(
            m=m,
            n=n,
            k=k,
            dtype=dtype,
            matmul_atol=args.matmul_atol,
            matmul_rtol=args.matmul_rtol,
            abft_tol=args.abft_tol,
            seed=args.seed,
        )
        all_pass = all_pass and all(
            [
                stats["baseline_ok"],
                stats["fused_ok"],
                stats["fused_vs_baseline_ok"],
                stats["abft_ok"],
            ]
        )
        print(
            f"[verify] shape={stats['shape']} "
            f"baseline_ok={stats['baseline_ok']} fused_ok={stats['fused_ok']} "
            f"fused_vs_baseline_ok={stats['fused_vs_baseline_ok']} abft_ok={stats['abft_ok']} "
            f"abft_abs={stats['abft_abs_error']:.6e} abft_rel={stats['abft_rel_error']:.6e}"
        )

    print(f"[verify] overall_pass={all_pass}")
