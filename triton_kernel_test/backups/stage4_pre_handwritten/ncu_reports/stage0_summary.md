# Stage 0 — ABFT-fused matmul Baseline 复现与归档

> 目的：在动手优化前，把当前 v1 实现（`matmul_abft_kernel`，`pid==0` 才做 partial）的性能 / NCU / SASS 指纹固化下来，作为后续阶段的 1:1 对照基线。

## 0.1 环境

- GPU：NVIDIA RTX 4090（CC 8.9，sm_89），可见 GPU 1（GPU 0 NVML 错误，整轮统一用 `CUDA_VISIBLE_DEVICES=1`）
- Toolchain：CUDA 12.4 ncu / cuobjdump，PyTorch 2.5.1+cu121，Triton 3.1.0，conda env `abft_cost`
- Fixed config（NCU 用、屏蔽 autotune 噪声）：`BM=128, BN=128, BK=32, GROUP_M=8, num_warps=4, num_stages=4`，与历史 `clean_std_*` 系列对齐。

## 0.2 改了哪些地方

仅"工具与归档"层面的变更，**没有改任何 kernel 代码**：

- 新增：[triton_kernel_test/run_ncu_compare.py](../run_ncu_compare.py) — 单入口、autotune 关闭的 NCU 启动器，支持 `--variant {triton,abft_v1}` 与可调 fixed config。后续每加一个 `abft_v{N}` 在这里追加一个 dispatch。
- 新增目录：`triton_kernel_test/backups/stage0_post/`，里面是 `matmul_abft_kernels.py` / `bench_matmul_abft.py` / `plot_benchmark_tflops.py` / `benchmark_stage0.csv` 的快照，**回退命令** `cp -r backups/stage0_post/* ./`。
- 新增基线产物：
  - `benchmark_stage0.csv`（256→3840 全量）
  - `ncu_reports/stage0_1024_triton.ncu-rep` + `stage0_1024_triton_sass.txt`
  - `ncu_reports/stage0_1024_abft_v1.ncu-rep` + `stage0_1024_abft_v1_sass.txt`
  - `plots/stage0_tflops_main.png`

## 0.3 思路

阶段 0 不做优化，只做"可对比基线"，确保后续每个 stage 都能用同一套数据格式回看：
- 用 `do_bench` 给宏观 TFLOPS / overhead；
- 用 1024^3 fixed config 的 NCU 给 occupancy / scheduler / 指令指纹；
- 用 SASS 直接数 `BAR.SYNC / SHFL.BFLY / LDS.128 / STG.E / HMMA` 这几条关键指令，作为后续每一步是否真的"动到了"硬件路径的硬证据。

## 0.4 指标 — 性能（do_bench, autotune 开）

`triton_abft_kernel_overhead_pct = (abft_kernel - triton) / triton * 100`

| Shape M=N=K | triton TFLOPS | abft_kernel TFLOPS | abft_kernel overhead | abft_full TFLOPS | abft_full overhead | abft_rel_error |
|------------:|--------------:|-------------------:|---------------------:|-----------------:|-------------------:|---------------:|
|         512 |         26.09 |              18.72 |              +39.35% |             6.90 |           +278.23% | 4.24e-7 |
|        1024 |         99.86 |              58.25 |              +71.43% |            34.38 |           +190.48% | 1.32e-6 |
|        2048 |        158.28 |             117.32 |              +34.91% |            99.86 |            +58.49% | 1.60e-6 |
|        3840 |        153.71 |             138.07 |              +11.33% |           132.36 |            +16.13% | 7.62e-6 |

观察：
- 小 shape (512–1024) 的 `abft_full` overhead 主要来自后置 `torch.sum + torch.dot`，因为 kernel 自身已经把 `partial_a/b/c` 写好但 host 上还要 `torch.sum(dim=0)` × 2 + `torch.dot`。
- 大 shape (3840) overhead 收敛到 ~16%，主要来自 kernel 内部规约。
- `abft_rel_error` 全程 ≤ 1e-5，远低于"可接受 ≥1e-2 检错门限"的预算。

## 0.5 指标 — NCU（1024^3, fixed config）

| 指标 | matmul_kernel (baseline) | matmul_abft_kernel (v1) | Δ |
|---|---:|---:|---:|
| Duration | 30.14 μs | 62.11 μs | +106% |
| Issue Slots Busy | 8.12% | 6.37% | -1.75 pp |
| One-or-More Eligible | 8.11% | 6.38% | -1.73 pp |
| **No Eligible** | 91.89% | **93.62%** | +1.73 pp |
| Eligible Warps / Sched | 0.08 | **0.06** | -0.02 |
| Achieved Occupancy | 8.33% | 8.32% | ≈ 0（被 reg=255 卡住）|
| Reg / Thread | 254 | 255 | +1 |
| Warp Cycles per Issued Inst | 12.30 | **15.66** | +3.36 cycles |
| Memory Throughput | 24.06% | 18.18% | -5.9 pp |
| L1/TEX Throughput | 23.01% | 46.18% | +23.2 pp |
| L2 Hit Rate | 90.70% | 90.47% | ≈ 0 |

解读：duration 翻倍、`No Eligible` 上抬，warp 平均要等更多 cycle 才能 issue —— 与"流水线被新插入的 reduce/barrier 打断"完全吻合。L1/TEX 翻倍则是 `tl.sum(a)` 走 SMEM round-trip 的直接表现。

## 0.6 指标 — SASS（1024^3, fixed config，每条指令在 PC 维度计数）

| 指令 | matmul_kernel | matmul_abft_kernel | Δ | 含义 |
|---|---:|---:|---:|---|
| **BAR.SYNC** | 78 | **150** | **+72** | 多出来的 ~72 条全部来自 `tl.sum` 的 cross-warp 规约（每次 reduce 4 条，K-iter ≈ 32 次，但只有 pid==0 路径 → ~80 条理论上限）|
| LDS.128 | 96 | 144 | +48 | tl.sum 把 a/b 从 SMEM 重读进规约 |
| **SHFL.BFLY** | 0 | **384** | **+384** | tl.sum 的 warp 内树状 reduce，本来 baseline 完全不需要 |
| STG.E | 96 | 114 | +18 | `pid==0` 路径每 K-iter 写 partial（被门控压低）|
| HMMA | 384 | 384 | 0 | MMA 主体不变 |
| F2F | 384 | 384 | 0 | accumulator → fp16 的转换不变 |
| FADD | 0 | 1458 | +1458 | partial 累加 |

> 注：以上是 PC 维度的 distinct count（每条 SASS 至少出现一次就 +1），不是动态 issued 计数；用于结构性对比，不当作执行次数。

这组 SASS 直接复现了专家的诊断：

1. `tl.sum` 在主循环里塞进 4×N 条 BAR.SYNC（+72），是 "Pipeline 气泡剧增" 的直接证据。
2. SHFL.BFLY 大量出现说明 Triton 已经在用 warp 内 shuffle，但是要先 LDS.128 → 才 SHFL → 才 BAR.SYNC，**走了 SMEM round-trip**，没有专家说的 "MMA-aware shuffle" 那种"直接利用寄存器里 fragment"的形态。
3. `pid==0` 门控让 STG.E 只多了 +18（store partial 不是主因），与 NCU 的 `attribution_*` 历史结论一致。

## 0.7 回退点

- 代码快照：`triton_kernel_test/backups/stage0_post/`
- 数据快照：`benchmark_stage0.csv`、`ncu_reports/stage0_1024_*.{ncu-rep, sass.txt}`、`plots/stage0_tflops_main.png`
- 回退命令：`cd triton_kernel_test && cp -r backups/stage0_post/* ./`

## 0.8 下一步（阶段 1）

按 plan 进入阶段 1：消除 tail effect（每个 block 都做自己的 colsum_A / rowsum_B）。
预期改善方向：
- 先让 `pid_n != 0 / pid_m != 0` 的 block 不再"早早闲置"。NCU 上看 SM Active Cycles 会更均匀，整体 wall time 在 1024^3 上目标 < 50 μs。
- BAR.SYNC 数量 **大概率不会下降**（仍然有 tl.sum）；它会下降是阶段 2 (`tl.dot(ones, X)` 替换 `tl.sum`) 的事。

请确认是否进入阶段 1。
