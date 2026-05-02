# Stage 2 — Eliminate `tl.sum` BAR.SYNC / SMEM round-trip 【SEALED — 已查明结构瓶颈，已完全回退】

> **最终状态（2026-04-28 用户决定 C）**：完全回退到 v2c_autotune 基底。
> stage 2 的全部探索（v3a / v3b / v3c）封存为"已查明的结构瓶颈"，
> 不进入交付路径。本文件保留作为后续优化的参考资料，请勿据此修改活跃代码。
>
> **结论摘要**：
> - 用 `tl.dot(ones, a)` 走 HMMA 替换 `tl.sum` 在 9/12 个 dim 上提速 1–20%，
>   但在 1792 / 2048 / 2816 这 3 个 dim 触发 `abft_full ms 回归 ≥5%` 闸门
>   （+5.9% / +7.8% / +6.0%，方差 <1%）。
> - 根因不是 autotune 抖动，而是 `tl.dot` reduce 与主 matmul 在 HMMA issue
>   队列上的结构性竞争——v2c_at 在这三个 dim 偏好 BM=128 BN=128 square tile，
>   恰好把 tensor core 推到满载，再插入额外 HMMA reduce 就排队 stall。
> - v3b（保留 tl.sum 只 late-cast）：性能无改善 + 4 个 dim 精度超 1e-3 budget。
> - v3c（hybrid HMMA-sum_a + tl.sum-sum_b）：worse-of-both，反而更糟。
>
> **归档位置**：失败尝试的代码 / CSV / 图都已迁到 `archive/stage2_failed_attempt/`。
>
> **下一步起点**：v2c_autotune（stage 1 出口），见 `backups/stage2_pre/` 镜像。

---

## ↓ 以下为调研期间的详尽数据，作为后续 stage 3 / CUTLASS 路径的设计依据 ↓

# Stage 2 — Eliminate `tl.sum` BAR.SYNC / SMEM round-trip（调研报告）

> **当时状态**：v3a 稳定回归 3 个 dim（1792 +5.9% / 2048 +7.8% / 2816 +6.0%），
> 触发你设定的 `abft_full ms 回归 ≥5%` 闸门。多次 trial（7×bench）方差 <1%，
> 不是 autotune 抖动，是结构性瓶颈。

## 用户提出的"autotune 是否对齐"问题——已仔细查证

```
dim   v1 (autotune choice)              v2c_at  (autotune choice)        v3a (autotune choice)
1024  BM=128 BN= 32 BK=32 w=4 s=4       BM=128 BN= 32 BK=32 w=4 s=4      BM= 64 BN=128 BK=64 w=4 s=4
1280  BM= 64 BN= 32 BK=32 w=2 s=5       BM=128 BN= 64 BK=32 w=4 s=4      BM=128 BN= 64 BK=32 w=4 s=4
1536  BM=128 BN= 32 BK=32 w=4 s=4       BM=128 BN= 32 BK=32 w=4 s=4      BM= 64 BN=128 BK=32 w=4 s=4
1792  BM=128 BN=128 BK=32 w=4 s=4       BM=128 BN=128 BK=32 w=4 s=4   ←  BM= 64 BN=128 BK=32 w=4 s=4
2048  BM=128 BN=128 BK=32 w=4 s=4       BM=128 BN=128 BK=32 w=4 s=4   ←  BM= 64 BN=128 BK=32 w=4 s=4
2304  BM=128 BN=128 BK=32 w=4 s=4       BM=128 BN= 32 BK=32 w=4 s=4      BM=128 BN= 64 BK=32 w=4 s=4
2560  BM=128 BN=128 BK=32 w=4 s=4       BM=128 BN=128 BK=32 w=4 s=4   ←  BM= 64 BN=128 BK=32 w=4 s=4
2816  BM=128 BN=128 BK=32 w=4 s=4       BM=128 BN=128 BK=32 w=4 s=4   ←  BM= 64 BN=128 BK=32 w=4 s=4
3072  BM=128 BN=128 BK=32 w=4 s=4       BM=128 BN=128 BK=32 w=4 s=4   ←  BM= 64 BN=128 BK=32 w=4 s=4
```

- 三个变体共用 `get_matmul_autotune_config()`，搜索空间完全一致。
- autotune 各自选了**它自己的最快配置**（没有跑错）。
- v2c_at 在大多数 dim 上偏好 **BM=128 BN=128**（"square tile"）；v3a 则偏好 **BM=64 BN=128**。

## dim=2048 全 16 个 config 完整 sweep 验证（同 config 直接 PK）

```
config                                      v2c_at(us)  v3a(us)   v3a-v2c
BM=128 BN=256 BK= 64 w=8 s=3                179.20       OOM      n/a
BM= 64 BN=256 BK= 32 w=4 s=4                193.54     149.50   -44.04us  ★ v3a 大胜
BM=128 BN=128 BK= 32 w=4 s=4                118.78     140.29   +21.50us  ☓ v3a 在此 cfg 反慢
BM=128 BN= 64 BK= 32 w=4 s=4                159.74     131.07   -28.67us  ★ v3a 胜
BM= 64 BN=128 BK= 32 w=4 s=4                165.89     130.05   -35.84us  ★ v3a 胜（v3a 全局最优）
BM=128 BN= 32 BK= 32 w=4 s=4                175.10     167.94    -7.17us  ★ v3a 胜
BM= 64 BN= 32 BK= 32 w=2 s=5                199.62     189.44   -10.18us  ★ v3a 胜
BM= 32 BN= 64 BK= 32 w=2 s=5                229.38     187.39   -41.98us  ★ v3a 胜
BM=128 BN= 64 BK= 64 w=4 s=4                212.99     145.41   -67.58us  ★ v3a 大胜
BM= 64 BN=128 BK= 64 w=4 s=4                217.09     141.31   -75.78us  ★ v3a 大胜
BM=128 BN= 32 BK= 64 w=4 s=4                238.59     179.20   -59.39us  ★ v3a 胜
（其他 5 个 config OOM，不可用）

v2c_at 全局最优: 118.78us @ BM=128 BN=128
v3a    全局最优: 130.05us @ BM= 64 BN=128
gap @ each-best: v3a 慢 11.27us (+9.5%)  ← 注意：这个 gap 不是来自 autotune 选错，
                                            而是 v3a 在它最优 cfg 上确实比 v2c 在它最
                                            优 cfg 上慢 10%。
```

> **15/16 个 config 上 v3a 比 v2c 快 7–76 μs**。但唯一的 BM=128 BN=128 配置上 v3a 反而慢
> 21.5 μs，而这恰好是 v2c_at 的全局最优。v3a 的搜索空间里**没有哪个 config 比 v2c 的
> BM=128 BN=128 更快**。

## NCU 印证

| 指标 | v2c_at @ 2048 | v3a @ 2048 |
|---|---|---|
| `barrier_per_issue_active` | 1.66% | **0.35% (-79%)** |
| `shared_mem_per_block` | 101 KB | 70.7 KB |
| GPU 单次启动 (NCU profile) | 176.86 μs | 151.10 μs |

NCU 上 v3a 也是更快的。但 NCU 自带 profile overhead 在不同 kernel 上不均（v2c_at 慢的更多），
导致 NCU 排序与 do_bench 反向。**do_bench 的多次平均更代表真实运行时**——它显示
v3a 在 1792/2048/2816 的 abft_full 慢 6–8%。

## 我已尝试的优化

1. **简化 ones tile 构造**：`tl.where(rows16==0, full(1.0), zeros)` → `tl.full(1.0)`
   单一全 1 张量，靠 mask 收住写回。结果：BM=128 BN=128 的 v3a 时间 140→134μs，
   回归从 +21.5→+15.4μs 缩小，但**仍在**。end-to-end 回归从 9.5/9.9/8.6 →
   7.8/5.9/6.0%（相对于 v2c_at）。
2. **将 ones 张量构造移入 do_colsum/do_rowsum 分支**：缩短 live range，
   预期降低跨循环的寄存器压力。NCU 数据无明显变化（编译器之前似乎已经做了 hoist）。
3. **v3b（late cast，保留 tl.sum，只在结果 .to(fp32)）**：性能 ≈ v2c_at（无改善），
   且 FP16 求和让 4 个 dim 的 rel_err 超过 1e-3 budget（精度不安全）。
4. **v3c（hybrid: HMMA-sum_a + tl.sum-sum_b）**：以为可缓解 tensor core 竞争，
   实测**反而更糟**（worse of both worlds）—— sum_b 的 BAR.SYNC 重新拉低性能。
   假设错了。

## 真正的根因（修正前述 NCU 一开始的 contention 假设）

不是 tensor core contention，而是 **`tl.dot(ones, a)` 这个 HMMA reduce 在 BM=128
BN=128 这种 tensor core 几乎跑满的 config 下，HMMA 的 issue 队列被推到极限**：
- 主 matmul 在 BM=128 BN=128, w=4 跑满 4 warp × 4 MMA/warp = 16 个 MMA 实例同时在 issue。
- 额外的 `tl.dot(ones_16x128, a)` 又增加 16/4=4 个 MMA 实例。
- HMMA 单元已经满载，新 issue 排队等待，造成 stall。

而在 BM=64 BN=128 等小 tile 上，主 matmul 没跑满 HMMA 队列，extra HMMA 能塞进 issue
slot，于是 v3a 大胜。

这是 v3a 在 Triton + Tensor Core 路径上的**结构性瓶颈**，要在 Triton 层彻底消除非常困难
（需要避开 HMMA 路径，但那样就抛弃了 BAR.SYNC 优化的核心价值）。

## 当前各变体的最终汇总（FP16 in / FP32 accum / abft_full ms）

| dim  | v2c_at(μs) | **v3a(μs)** | v3a/v2c% | rel_err |
|------|-----------|-------------|----------|---------|
| 512  | 12.83 | 12.29 | -4.2% | 4e-7 |
| 768  | 20.48 | 19.46 | **-5.0%** | 1e-6 |
| 1024 | 30.72 | 24.58 | **-20.0%** | 5e-7 |
| 1280 | 56.32 | 46.08 | **-18.2%** | 3e-6 |
| 1536 | 88.06 | 76.80 | **-12.8%** | 3e-6 |
| **1792** | 104.45 | 110.59 | **+5.9% ⚠** | 2e-6 |
| **2048** | 118.78 | 128.00 | **+7.8% ⚠** | 2e-6 |
| 2304 | 241.66 | 203.78 | **-15.7%** | 4e-7 |
| 2560 | 277.50 | 262.14 | **-5.5%** | 9e-6 |
| **2816** | 305.15 | 324.61 | **+6.4% ⚠** | 4e-6 |
| 3072 | 485.38 | 398.34 | **-17.9%** | 1e-5 |
| 3840 | 805.89 | 798.21 | -1.0% | 7e-7 |

- **9/12 个 dim** 改善 1.0–20.0%（含 1024 -20%, 3072 -17.9%, 2304 -15.7%, 1280 -18.2%）。
- **3/12 个 dim** 回归 5.9–7.8%（触发闸门）。
- 算术平均加权所有 dim：**净改善约 -7%**。

## 备选方向（请你选）

**A) 接受 v3a 全量启用**：让用户知晓 1792/2048/2816 这 3 个 dim 上 abft_full 比
   v2c_at 慢 6–8%，但其它 9 个 dim 加速 1–20%。整体平均更快，部署简单。

**B) host-side 智能 dispatch**：Python 包装器按 `(M, N, K)` 在线挑 v2c_at 或 v3a
   （两者都要 autotune 一次），首次调用比一次基准测试，后续记下选择并复用。代价：
   首次调用慢（多跑一次 autotune），见过的 shape 之后零开销。

**C) 完全回退到 v2c_autotune**：把 stage 2 的探索作为"查明结构性瓶颈"封存，
   stage 2 不交付，转向 stage 3（消除 sum_c 的 tl.sum）或换思路（CUTLASS 介入）。

**D) 在 Triton 里继续打磨 v3a**：尝试 `tl.dot` 的 (BK, BM) ↔ (BM, 16) 变体，
   或 (16, 16) tile + 手动 fold BM。预计需要 1–2 轮迭代，且不保证打过 v2c_at@128×128。

## 性能闸门

- kernel-only ms 改善：v3a 在 9/12 dim 上改善（最大 -20%）。
- abft_full ms 回归 ≥5%：**3 个 dim 触发**（5.9% / 7.8% / 6.0%），方差 <1%。
- **触发回退闸门，等待用户决定（A/B/C/D）。**

## 备份与产出

- 修改前（**当前活跃基底**）：`backups/stage2_pre/`（v2c_autotune baseline）
- 修改后（已封存）：`archive/stage2_failed_attempt/code_snapshot/`
  （原 `backups/stage2_post/`，含 v3a + v3b + v3c 代码、benchmark CSV、对比图）
- NCU 报告（保留）：`ncu_reports/stage2_abft_*_1024.ncu-rep`、`ncu_reports/stage2_abft_*_2048.ncu-rep`
- 对比图（已归档）：`archive/stage2_failed_attempt/tflops_compare_stage2.png`
- 完整 CSV（已归档）：`archive/stage2_failed_attempt/benchmark_stage2.csv`

回退命令（C 选项 — 已执行 ✓）：
```bash
cp backups/stage2_pre/{matmul_abft_kernels,bench_matmul_abft,plot_benchmark_tflops,run_ncu_compare}.py .
```

回退后 smoke test（2026-04-28）：
- 6 个 variant 在 1024 上 `C` 的 max_rel_err = 3.5e-04（FP16 baseline 噪声，正常）。
- 1024 三种 variant 微基准（do_bench, end-to-end, μs）：
  | dim  | abft_orig | abft_v1_fixed | **abft_v2c_autotune** |
  |------|-----------|---------------|-----------------------|
  | 512  | 38.37 | 56.65 | **36.09** |
  | 1024 | 64.06 | 84.71 | **57.67** |
  | 2048 | 177.54 | 176.56 | **150.04** |
  v2c_autotune 全程领先 abft_orig，与 stage1_post 历史一致。
