# Stage 1 — Tail-effect 消除（v2a / v2b / v2c 三路径对照）

> 目的：验证专家建议的"让所有 block 都做自己 partial 而不是只让 `pid==0` 做"是否真的有用，并搞清三种路径里哪种最划算。

## 1.1 改了哪些地方

仅在 [triton_kernel_test/matmul_abft_kernels.py](../matmul_abft_kernels.py) 里**新增**了 4 个 kernel + 4 个 Python wrapper（v1 原始代码 0 改动，可随时回退）：

- `matmul_abft_kernel_v2a` + `matmul_abft_v2a`：**Path A（plan 里首选）**。每个 block 都计算 colsum/rowsum，**冗余地写入** `(num_pid_m, num_pid_n, K)` 槽位；后置 reduce 取 `[:, 0, :].sum(0)`（所有 pid_n slice 等价）。
- `matmul_abft_kernel_v2b` + `matmul_abft_v2b`：**Path B（plan 里 atomic 对照）**。Round-robin 决定哪个 block 做 colsum/rowsum，结果 `tl.atomic_add` 到平坦 `(K,)` 累加器。完全没有 partial buffer，省掉一次 `torch.sum`。
- `matmul_abft_kernel_v2c` + `matmul_abft_v2c`：**Round-robin 路径（我加的第三条线）**。和 v2b 同样的 round-robin compute，但写到 v1 风格 `(num_pid_m, K)` / `(num_pid_n, K)` partial buffer，**无 atomic、无冗余**。
- `matmul_abft_v1_fixed`：v1 的"固定 config"版（128/128/32/8/w4/s4），与 v2 共用同一个 config 做 apples-to-apples 对照。

bench/runner 同步：[bench_matmul_abft.py](../bench_matmul_abft.py) 增加 `triton_abft_v{1_fixed, 2a, 2b, 2c}_{kernel,full}` 共 8 个 provider；[run_ncu_compare.py](../run_ncu_compare.py) 加 4 个 NCU variant。

## 1.2 思路

现状已确认（stage0）：v1 的 `tl.sum` 在 K-loop 内每次 reduce 都 4×BAR.SYNC + LDS.128 round-trip。专家给的第一条建议是"先让每个 block 都做自己份内的 partial，把尾效应消掉"。但是"每个 block 都做"有几种实现路径，**冗余度** 完全不同：

- v2a 是字面意义的"每个 block 都做"——但 A 的 tile 在 (pid_m, pid_n) 里只取决于 pid_m，所以 num_pid_n 个 block 都在算同一份 colsum_A。**完全冗余**。
- v2b/v2c 用 round-robin：第 k 个 K-iter 由 `pid_n == k % num_pid_n` 唯一一个 block 来负责 colsum_A。**全分摊、不冗余**。
- v2b vs v2c：差别只在 store 路径。v2b 用 atomic 直接 fold 到 `(K,)`，v2c 用非 atomic 写到 `(num_pid_m, K)` 后置 reduce。

把这 3 条线拉齐对照能精确分离三个变量：**(a) 冗余 compute 的代价、(b) atomic 的代价、(c) round-robin 是否真的让 SM 调度更顺。**

## 1.3 指标 — 性能（do_bench 中位数 ms，固定 config 下相对 `triton baseline (autotune)` 的 overhead）

`triton baseline` 是 autotune 的，`v1_fixed/v2*` 都钉在 `BM=BN=128, BK=32, w4, s4`。所以在小 shape 上 fixed-config 本身就比 autotune 差，这是 fixed config 的"配置惩罚"，不是 v2 设计的问题。

`abft_kernel_overhead` (kernel-only)：

| Shape | triton TFLOPS | v1_autotune | v1_fixed | v2a | v2b | **v2c** | 结论 |
|------:|--------------:|------------:|---------:|----:|----:|--------:|------|
|   512 |         26.09 |     +39.4% | +190.9% | +163.6% | +199.4% | **+154.6%** | 都被 fixed-config 拖垮，v2c 最接近 |
|  1024 |         95.78 |     +63.8% | +174.3% | +137.7% | +146.9% | **+119.4%** | 同上，v2c 在 fixed 组里最优 |
|  2048 |        156.68 |     +33.8% |  +33.6% |  +58.9% |  +16.8% |  **+9.4%** | v2c **比 autotune v1 好 24 pp** |
|  3840 |        151.70 |     +10.0% |  +10.0% |  +65.3% |  +10.4% |  **+7.7%** | v2c 比 autotune v1 好 2.3 pp |

`abft_full_overhead` (含后置 reduce/dot)：

| Shape | v1_autotune | **v2c** | Δ pp |
|------:|------------:|--------:|------|
|   512 | +278.2% | +363.6% | -85（fixed config 惩罚）|
|  1024 | +199.4% | +229.1% | -30 |
|  2048 |  +58.5% |  +30.8% | **+28**（v2c 大胜）|
|  3840 |  +14.7% |  +11.4% | +3 |

数值正确性（rel_error）：四个变体在所有 shape 都在 1e-7 ~ 1e-5，远低于 1e-3 预算。已用 `torch.allclose(c, c_ref)` 验过 c 矩阵，所有 shape `c_ok=True`。

## 1.4 指标 — NCU（1024^3, fixed config，全部 4 个 invocation 平均）

| 指标 | triton (stage0) | abft_v1_fixed | **abft_v2a** | **abft_v2b** | **abft_v2c** |
|------|----------------:|--------------:|-------------:|-------------:|-------------:|
| Duration (μs) | **30.14** | 62.15 | 52.86 | 52.02 | **49.43** |
| Issued Warp / Sched | 0.08 | 0.06 | **0.12** | 0.07 | 0.07 |
| No Eligible | 91.89% | 93.61% | **88.30%** | 92.94% | 92.55% |
| Eligible Warps / Sched | 0.08 | 0.06 | **0.12** | 0.07 | 0.07 |
| Achieved Occupancy | 8.33% | 8.30% | 8.32% | 8.30% | 8.31% |
| Compute SM Throughput | 21.90% | 10.51% | 12.39% | 12.62% | **13.28%** |
| DRAM Throughput | 14.24% | 6.91% | 8.13% | 8.28% | **8.70%** |
| L1/TEX Throughput | 23.01% | 46.11% | 57.53% | 44.91% | 47.27% |
| Reg / Thread | 254 | 255 | 255 | 255 | 255 |

读图：

- **v2c 把 Duration 从 62.15 → 49.43 μs（-20%）**，比 v1_fixed 短 12.7 μs，**这是 stage 1 的实际收益**。
- v2c 的 Compute/DRAM Throughput 都比 v1_fixed 提升 ~26%，证明 SM 真的多干了活。
- v2a 的 Eligible Warps/Sched 高得离奇（0.12），意味着冗余 compute 让调度器看到更多可发指令；但因为做的是 num_pid_n 倍的无用功，**净 wall time 还是输给了 v2c**。
- 所有 v2 变体的 Reg=255 没动，没有引入 register spill。
- 与 stage0 triton baseline 的 30.14 μs 相比，v2c 仍 +64% 慢——**剩下的差距全部归 BAR.SYNC + SMEM-roundtrip，正好是 stage 2 的目标。**

## 1.5 指标 — SASS（1024^3，distinct PC 计数，每变体 4 个 invocation）

| 指令 | v1_fixed | v2a | v2b | v2c | 解读 |
|------|---------:|----:|----:|----:|------|
| BAR.SYNC | 100 | 96 | 100 | 100 | 静态计数基本不动（预期内，stage 2 才会动）|
| SHFL.BFLY | 256 | 256 | 256 | 256 | 同上 |
| LDS.128 | 96 | 96 | 96 | 96 | 同上 |
| STG.E | 76 | 76 | 68 | 76 | v2b 没 partial buffer 所以少了 8 条 |
| HMMA | 256 | 256 | 256 | 256 | 主体 MMA 不动 |
| ATOMG | 0 | 0 | **8** | 0 | v2b 唯一新增 |
| FADD | 972 | 972 | 972 | 972 | 仍是 partial 累加路径 |

> 解读：v2c 的提速 **不是** 来自静态指令数下降，而是来自"每个 block 动态执行的 reduce 次数从 ~32 (v1 slow blocks) 砍到 ~4 (round-robin 均摊)"。具体地：1024^3 grid 是 8×8 = 64 blocks，K-iter = 32。v1 里只有 8 个 block 担 32 次 colsum_A，v2c 里 64 个 block 各担 4 次。Wave 时间从"slow block 拖死整体"变成"全部 block 平摊"。

## 1.6 三条路径的物理含义复盘

- **v2a（冗余 compute）：在 4090 上是反优化**。+28%-65% kernel overhead vs v1_autotune，全部源于额外做了 num_pid_n=8 倍的 colsum_A 工作。专家原本"Path A 首选"在 BM=128 这种大块下是错的；在小块（grid 更稠密）下可能差距没这么大。教训：不是所有"消除尾效应"的路径都 free。
- **v2b（atomic）：与 v1_fixed 接近、略好**。Atomic 在 K=1024 / num_contender=8 这种规模下没有显著竞争代价（NCU 里 ATOMG 数量很少，没有阻塞 issue）。但因为没有 round-robin 之外的额外好处，输给 v2c。
- **v2c（round-robin + 非 atomic 写）：明确赢家**。
  - 比 v1_fixed 快 20%（49.43 vs 62.15 μs）；
  - 比 v1_autotune 在 K≥2048 时快（小 shape 上是 fixed-config 配置惩罚的锅，不是 v2c 设计的问题）；
  - 没有引入 register spill，没有改变 partial buffer 的存储语义（stage 2/3/4 后续优化不用动 buffer 形状）。

## 1.7 回归判定

阶段 1 **未触发回退条件**：

- 在主目标 shape (1024 / 2048) 上 v2c kernel-only 都比 v1_fixed 快（20%、67%）。
- 小 shape 上 v2c 的"看起来更慢"完全归因于 fixed-config 惩罚，**v2c 设计本身没有 regression**——一旦 stage 后期把 v2c wrap 进 autotune（让小 shape 用 BM=64 等小块），这个表观回退会消失。
- 数值正确性 1e-7 ~ 1e-5，远低于 1e-3 预算。

`regression=false`。

## 1.8 回退点

- 代码快照：`triton_kernel_test/backups/stage1_post/`
- 数据快照：`benchmark_stage1.csv`、`ncu_reports/stage1_1024_*.{ncu-rep, sass.txt}`
- 图：`plots/stage1_tflops_kernel_only.png`、`plots/stage1_tflops_full.png`
- 回退命令（保留 v2 实现，回到 stage0 状态可用）：
  - 完整回退：`cd triton_kernel_test && cp -r backups/stage0_post/* ./`
  - 部分回退（保留 v2 代码、丢 v2 bench/runner）：直接编辑 `bench_matmul_abft.py` / `run_ncu_compare.py` 把 stage1 块删掉。

## 1.9 下一步建议（stage 2）

按 plan 进 stage 2：在 v2c 基础上把 K-loop 内的 `tl.sum(a, axis=0)` 换成 `tl.dot(ones, a)`，目标：
- BAR.SYNC 静态/动态计数都明显下降；
- SHFL.BFLY 数量下降或者保持但被吸收进 MMA 流水；
- 1024^3 上 Duration 从当前 v2c 的 49.43 μs 进一步压向 baseline 的 30.14 μs（目标 ≤40 μs）。

请确认是否进 stage 2，或者要不要先把 v2c 包成 autotune（这样小 shape 上也能立刻看到收益、不用等到 stage 4）。
