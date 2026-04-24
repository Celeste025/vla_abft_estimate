# Triton Matmul + ABFT Benchmark

当前目录仅保留一个主脚本：`reproduce_tutorial_matmul_bench.py`。  
该脚本用于对比以下实现的性能与 ABFT 指标：

- `cublas`：`torch.matmul`
- `triton`：教程风格 Triton matmul
- `triton_abft_kernel`：仅 ABFT kernel（不含 host 端后规约）
- `triton_abft_full`：ABFT kernel + 后规约
- `triton_abft_naive_full`：先 matmul，再独立计算 `sum(A列)`/`sum(B行)`/`sum(C全部)`/点积

进阶（独立模块，不影响主脚本输出）：K 向批量 flush 的融合 ABFT 见 `abft_fused_smem.py`；对比性能可运行 `python bench_abft_fused_smem.py --m-min 8 --m-max 17`。

## 环境要求

- Python 3.10+
- CUDA GPU
- `torch`（CUDA 版本）
- `triton`

## 运行方式

在目录下执行：

```bash
python reproduce_tutorial_matmul_bench.py
```

可选参数：

```bash
python reproduce_tutorial_matmul_bench.py --m-min 2 --m-max 33
```

其中矩阵规模为：`M=N=K=128*i`，`i` 从 `m-min` 到 `m-max-1`。

## 输出字段说明

脚本输出 CSV 格式，表头为：

`provider,M,N,K,TFLOPS,abft_kernel_overhead_pct,abft_full_overhead_pct,abft_abs_error,abft_rel_error`

- `provider`：实现名称
- `TFLOPS`：按 `2*M*N*K / time` 计算
- `abft_kernel_overhead_pct`：`(ms_abft_kernel - ms_triton) / ms_triton * 100`
- `abft_full_overhead_pct`：`(ms_abft_full - ms_triton) / ms_triton * 100`
- `abft_abs_error`：`|dot(sum_a, sum_b) - sum_c|`
- `abft_rel_error`：`abft_abs_error / max(|sum_c|, 1e-8)`

说明：对于 `cublas` / `triton` 行，ABFT 相关字段为空。

## 当前实现要点

- ABFT 版本在 Triton kernel 内融合了部分 checksum 计算，并写入 partial buffer。
- `triton_abft_full` 在 kernel 结束后会做 `sum_a_partial`、`sum_b_partial`、`sum_c_partial` 的归约并计算点积误差。
- `triton_abft_naive_full` 作为参考基线，采用“先乘法后校验”的直接流程。
