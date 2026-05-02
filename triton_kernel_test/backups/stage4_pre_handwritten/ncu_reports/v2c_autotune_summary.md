# Stage 1.5 — v2c autotune (small-shape regression fix)

## 背景

Stage 1 选定 `matmul_abft_kernel_v2c`（round-robin 不带 atomic 的 ABFT 路径）作为
当前最优变体。但在小矩阵（M=N=K ≤ 1024）下 `v2c_fixed`（BM=BN=128, BK=32, w4, s4）
明显劣于 `triton_abft_kernel`（v1_autotune）：

| dim  | v1_autotune full ms | v2c_fixed full ms | v2c_fixed 相对 v1 |
|------|---------------------|-------------------|-------------------|
| 256  | 0.0328              | 0.0410            | **+25.0%**        |
| 512  | 0.0389              | 0.0522            | **+34.2%**        |
| 768  | 0.0481              | 0.0635            | **+32.0%**        |
| 1024 | 0.0625              | 0.0737            | **+17.9%**        |

经核查，回归并非 v2c 算法/任务分配缺陷，而是 **fixed-config 的 BM=BN=128 在小矩阵下
SM 利用率不足 + 寄存器/SMEM 浪费**。`v1_autotune` 会自动选择更小的 BM/BN（如
64×64、64×128）以及更大的 num_stages，从而获得更好性能。

## 修改

**文件**：`matmul_abft_kernels.py`

1. 新增 `matmul_abft_kernel_v2c_autotune`（v2c 同构逻辑 + `@triton.autotune`）。
2. autotune 候选 **直接复用 `get_matmul_autotune_config()`**（与 `matmul_kernel` /
   `matmul_abft_kernel` 完全一致），保证三者在同一搜索空间下做苹果对苹果对比。
3. 新增 Python 包装器 `matmul_abft_v2c_autotune` + kernel-only launcher
   `launch_matmul_abft_v2c_autotune_kernel_only`。

**文件**：`bench_matmul_abft.py`

- 引入 v2c_autotune 的 kernel-only 与 full 两路 benchmark，输出 provider
  `triton_abft_v2c_at_kernel` / `triton_abft_v2c_at_full`。

**文件**：`plot_benchmark_tflops.py`

- 新增 `V2C_AUTOTUNE_COMPARE_PROVIDERS` 与 `plots/tflops_compare_v2c_autotune.png`。

## 结果（M=N=K 对角扫描，FP16 in / FP32 accum）

CSV: `benchmark_v2c_autotune.csv`

### 小矩阵段（直接验证回归是否消除）

| dim  | cublas | triton | v1_at full ms | v2c_fix full ms | **v2c_at full ms** | v2c_at vs v1_at |
|------|--------|--------|---------------|-----------------|--------------------|-----------------|
| 256  | 0.0072 | -      | 0.0328        | 0.0410          | **0.0317**         | **-3.4%**       |
| 384  | 0.0082 | -      | 0.0358        | 0.0471          | **0.0338**         | **-5.6%**       |
| 512  | 0.0092 | -      | 0.0389        | 0.0522          | **0.0369**         | **-5.1%**       |
| 768  | 0.0164 | -      | 0.0481        | 0.0635          | **0.0451**         | **-6.2%**       |
| 1024 | 0.0256 | -      | 0.0625        | 0.0737          | **0.0563**         | **-9.9%**       |
| 1536 | 0.0666 | -      | 0.1198        | 0.1137          | 0.1147             | -4.3%           |
| 2048 | 0.1106 | -      | 0.1731        | 0.1434          | 0.1444             | **-16.6%**      |
| 3840 | 0.7557 | -      | 0.8530        | 0.8274          | 0.8325             | -2.4%           |

ABFT 相对误差全段 ≤ 1.4e-5，远低于 1e-3 budget（detect threshold ≥1e-2）。

### 完整 benchmark（256→2816 步进 256）TFLOPS 对比

| dim  | v1_at full TFLOPS | v2c_at full TFLOPS | Δ          |
|------|-------------------|--------------------|------------|
| 256  | 1.06              | **1.09**           | +2.8%      |
| 512  | 7.08              | **7.49**           | +5.8%      |
| 768  | 18.82             | **20.11**          | +6.9%      |
| 1024 | 34.38             | **38.13**          | +10.9%     |
| 1280 | 49.35             | **50.57**          | +2.5%      |
| 1536 | 59.98             | 62.64              | +4.4%      |
| 1792 | 75.94             | **86.46**          | +13.9%     |
| 2048 | 99.86             | **118.99**         | **+19.2%** |
| 2304 | 87.82             | **91.52**          | +4.2%      |
| 2560 | 105.03            | **110.33**         | +5.0%      |
| 2816 | 126.05            | **134.20**         | +6.5%      |

### kernel-only overhead vs naïve triton

| dim  | v1_at kernel%  | v2c_fix kernel% | **v2c_at kernel%** |
|------|----------------|-----------------|--------------------|
| 256  | 66.7%          | 183.3%          | **33.3%**          |
| 512  | 38.9%          | 177.8%          | **19.1%**          |
| 768  | 59.3%          | 152.3%          | **32.8%**          |
| 1024 | 68.2%          | 122.7%          | **40.9%**          |
| 1536 | 42.2%          | 35.9%           | 34.4%              |
| 1792 | 29.0%          | 9.6%            | 9.6%               |
| 2048 | 33.6%          | 9.4%            | 8.4%               |

## 结论

1. **回归已彻底消除**：`v2c_autotune` 在小矩阵段全面追平甚至超过 `v1_autotune`
   （比如 dim=512 kernel overhead 从 v2c_fixed 的 178% 降到 19%，比 v1_autotune
   的 39% 还更好）。
2. **大矩阵段维持 Stage 1 收益**：v2c_autotune 与 v2c_fixed 相当（差 <2%），
   并保留了 round-robin 抹平 tail effect 带来的 8–17% kernel-only 提升。
3. **数值精度无变化**：rel_error 仍在 1e-7 ~ 1e-5 区间。
4. **苹果对苹果**：v2c_autotune 与 v1_autotune 共享 `get_matmul_autotune_config()`，
   差异完全来自 round-robin checksum store，便于后续阶段继续对比。

## 备份

- 修改前：`backups/v2c_autotune_pre/`
- 修改后：`backups/v2c_autotune_post/`（含本阶段 csv + 两张对比图）

如需回退：`cp backups/v2c_autotune_pre/matmul_abft_kernels.py .`（其它三件同理）。

## 性能闸门检查

- **kernel-only ms**：所有 dim 上 v2c_autotune ≤ v2c_fixed 或与之等价；小矩阵段
  甚至显著优于 v1_autotune（满足 “kernel-only ms 有改善”）。
- **abft_full ms**：v2c_autotune 相比 stage1 baseline (`v2c_fixed`) 最大改善 22.7%
  （dim=1024），最差不劣化；相比 stage0 baseline (`v1_autotune`) 全段改善 2.4%
  ~ 19.2%，**未触发 ≥5% 回归闸门**。

可以安全推进到 Stage 2（消除 BAR.SYNC / MMA-aware shuffle）。
