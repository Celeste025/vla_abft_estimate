# Triton GEMM Baseline + Fused ABFT

本目录实现了基于 Triton `03-matrix-multiplication` 官方教程结构的 GEMM baseline 与融合 ABFT 校验版本，并提供统一验证与 benchmark 框架。

## 文件说明

- `triton_gemm_baseline.py`  
  教程风格 GEMM kernel（无 ABFT），输出 `float32`。
- `triton_gemm_abft_fused.py`  
  在 GEMM tile 循环中融合 `sumA/sumB/sumC` 累积，并输出 `dot(sumA,sumB)` 与 `sumC` 的误差。
- `verify_abft.py`  
  正确性验证脚本：baseline/fused vs `torch.matmul`，并检查 ABFT 标量一致性。
- `benchmark_abft.py`  
  统一性能测试脚本，输出 `torch.matmul(=cuBLAS)` / `baseline` / `fused ABFT` 的时间、TFLOPS 与差异。
- `run_bench.sh`  
  一键执行验证和 benchmark。

## 环境要求

- Python 3.10+
- CUDA 可用 GPU
- `torch`（CUDA 版本）
- `triton`

## 快速开始

```bash
cd /data/home/jinqiwen/workspace/vla_abft_estimate/triton_kernel_test
python verify_abft.py --dtype fp16 --shape 4096,4096,4096 --shape 8192,4096,4096
python benchmark_abft.py --dtype fp16 --warmup 20 --repeat 100 --shape 4096,4096,4096 --shape 8192,4096,4096
```

或直接运行：

```bash
./run_bench.sh
```

## 输出指标

- 正确性：
  - `baseline_ok`、`fused_ok`、`fused_vs_baseline_ok`
- ABFT 一致性：
  - `abft_abs_error = |dot(sumA,sumB) - sumC|`
  - `abft_rel_error = abft_abs_error / max(|sumC|, 1e-8)`
- 性能：
  - `torch_ms`、`baseline_ms`、`fused_ms`
  - `torch_tflops`、`baseline_tflops`、`fused_tflops`
  - `baseline_vs_torch_pct = (baseline_ms - torch_ms) / torch_ms * 100`
  - `fused_vs_torch_pct = (fused_ms - torch_ms) / torch_ms * 100`
  - `overhead_pct = (fused_ms - baseline_ms) / baseline_ms * 100`

## 形状与阈值建议

- 默认形状：
  - `4096 x 4096 x 4096`
  - `8192 x 4096 x 4096`
- 容差建议：
  - matmul 一致性默认 `atol=1e-2, rtol=1e-2`
  - ABFT 标量检查可先用 `abft_tol=5e-1`，再按 dtype 和规模收紧

## 后续优化建议

- 使用两级归约（block 局部缓冲 -> 最终归约）降低 `sumA/sumB/sumC` 的原子竞争。
- 分 dtype 调整 `BLOCK_M/N/K`、`num_warps`、`num_stages`。
- 对 ABFT 累积路径单独 profile，评估寄存器压力与 occupancy 变化。

## 参考实现

- Triton 官方教程源码：`python/tutorials/03-matrix-multiplication.py`
- Triton 教程文档（含基准测试）：HyperAI 镜像文档的 Matrix Multiplication 章节
