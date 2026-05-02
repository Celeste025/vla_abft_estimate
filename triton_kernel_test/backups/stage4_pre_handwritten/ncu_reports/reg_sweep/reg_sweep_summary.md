# ABFT 寄存器压力敏感性扫描（1024）

## 实验目的
- 验证专家猜测中“`for` 循环内 `partial_a/partial_b` 导致寄存器/依赖压力”的可信度。
- 方法：固定同一 probe kernel，扫描 `BLOCK_K / num_stages`，并对比 `do_sum=1`（含 partial A/B 规约）与 `do_sum=0`（不做该规约）。

## 运行矩阵
- `k64_s4_w4`
- `k32_s4_w4`
- `k32_s2_w4`
- `k16_s2_w4`
- 每个配置各跑 `do_sum=1` 与 `do_sum=0`（NCU full）

## 核心结果（单位 us）
- `k64_s4_w4`: `do_sum=1` 64.29 vs `do_sum=0` 32.67（+31.62）
- `k32_s4_w4`: `do_sum=1` 65.82 vs `do_sum=0` 30.59（+35.23）
- `k32_s2_w4`: `do_sum=1` 72.83 vs `do_sum=0` 37.60（+35.23）
- `k16_s2_w4`: `do_sum=1` 90.43 vs `do_sum=0` 49.76（+40.67）

## NCU 观测要点
- `Registers Per Thread` 基本打满（大多 `254~255`，仅 `k16_s2_w4` 降到 `217~218`）
- `Achieved Occupancy` 全部约 `8.2~8.4%`（长期低占用）
- `No Eligible` 持续很高（约 `91.6%~95.2%`）
- `do_sum=1` 相比 `do_sum=0`：
  - `Duration` 全配置显著增加（+31.6us 到 +40.7us）
  - `No Eligible` 更高，`Eligible Warps` 更低或持平
  - `Warp Cycles Per Issued Instruction` 普遍更差

## 结论
- 该扫描支持“瓶颈主因在 `partial_a/partial_b` 规约链及其依赖压力”这一判断。
- 这里不是单纯“global store 写回”主导，因为在同配置下仅打开规约路径（`do_sum=1`）就会出现系统性变慢。
- `Regs` 高企 + `No Eligible` 高 + `Eligible Warps` 低，说明核心问题是发射可用性差（依赖链/活跃值叠加），与专家判断一致。

## 产物文件
- 汇总 CSV：`ncu_reports/reg_sweep/reg_sweep_metrics.csv`
- 原始报告：`ncu_reports/reg_sweep/*.ncu-rep`
- 报告文本：`ncu_reports/reg_sweep/*.txt`
