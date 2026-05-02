# ABFT Fused 瓶颈归因摘要（1024）

## 实验设置
- 尺寸: `M=N=K=1024`
- 入口脚本: `triton_kernel_test/ncu_profile_abft.py`
- 采样命令: `ncu --clock-control none --import-source yes --set full`
- 采样对象:
  - `triton` (baseline)
  - `abft_kernel` (完整 fused)
  - `ablate_no_sum_store0` (去 sum, 保留 store0 骨架)
  - `ablate_sum_no_partial_store` (保留 sum, 去 partial global store)

## 关键指标对照（NCU，修正后）
- `triton`: Duration `58.69us`, Achieved Occupancy `16.62%`, No Eligible `92.53%`, Eligible Warps/Scheduler `0.11`, Memory Throughput `71.83 GB/s`
- `abft_kernel`: Duration `62.30us`, Achieved Occupancy `8.30%`, No Eligible `93.62%`, Eligible Warps/Scheduler `0.06`, Memory Throughput `67.75 GB/s`
- `ablate_no_sum_store0`: Duration `30.50us`, Achieved Occupancy `8.31%`, No Eligible `91.14%`, Eligible Warps/Scheduler `0.09`, Memory Throughput `138.27 GB/s`
- `ablate_sum_no_partial_store`: Duration `65.44us`, Achieved Occupancy `8.31%`, No Eligible `93.96%`, Eligible Warps/Scheduler `0.06`, Memory Throughput `64.50 GB/s`

## 归因计算（按计划公式）
- `T_full = 62.30us`
- `T_noSum = 30.50us`
- `T_noStore = 65.44us`
- 规约链贡献近似: `T_full - T_noSum = +31.80us`
- partial 写回贡献近似: `T_full - T_noStore = -3.14us`

## 结论
- 原先 `no_store` 异常变慢是实验不等价导致（mode2 里额外 sink 规约 + autotune 子空间不一致）。
- 修正后 `no_store` 仅比 full 略慢（`62.30us -> 65.44us`），说明“去掉 partial global store 仍不显著变快”。
- 主要成本仍在 `partial_a/partial_b` 规约链及其依赖压力（`full` 相对 `no_sum` 差值明显）。
- 主因排序（修正后）:
  - 第一: 规约链 + 依赖/调度压力
  - 第二: 写回路径（非主导）

## 口径说明（重要）
- 上述 `matmul_kernel / matmul_abft_kernel` 的 NCU `Duration` 可能受到 autotune 首次试探与启动阶段影响，主要用于结构性归因，不作为最终稳态性能数字。
- 稳态性能以 `do_bench` 为准（同批数据下约 `triton 0.0215ms` vs `abft_kernel 0.0379ms`，`+76%` 开销）。

## 无 autotune 干扰的固定配置复验（k32_s4_w4）
- 文件位置：`ncu_reports/reg_sweep/fixedattr_*.{ncu-rep,txt}`
- `fixedattr_full` (`do_sum=1, do_store=1`):
  - Duration `66.08us`, Reg `255`, Occ `8.30%`, No Eligible `93.47%`, Eligible `0.07`
- `fixedattr_no_store` (`do_sum=1, do_store=0`):
  - Duration `64.77us`, Reg `252`, Occ `8.31%`, No Eligible `93.96%`, Eligible `0.06`
- `fixedattr_no_sum` (`do_sum=0, do_store=0`):
  - Duration `30.59us`, Reg `254`, Occ `8.32%`, No Eligible `91.59%`, Eligible `0.08`
- 固定配置下仍可见：去掉 `sum` 的收益远大于去掉 `store`，支持“规约链/依赖压力主导，store 非主导”。

## SASS 证据
- 规约路径存在 warp-level shuffle: `SHFL.BFLY`（见 `attribution_1024_abft_full_sass.txt`）
- 写回存在向量化 store: `STG.E.128`，且带 predication（`@P* STG.E.128`）
- 结论: 可基本排除“triton 没有把 sum/store 编译成高效形态”的问题。

## 可复现实验清单
- baseline:
  - `ncu ... -o ncu_reports/attribution_1024_triton ... ncu_profile_abft.py --variant triton --dim 1024 --warmup 0 --iters 1`
- full:
  - `ncu ... -o ncu_reports/attribution_1024_abft_full ... ncu_profile_abft.py --variant abft_kernel --dim 1024 --warmup 0 --iters 1`
- no-sum:
  - `ncu ... -o ncu_reports/attribution_1024_no_sum ... ncu_profile_abft.py --variant ablate_no_sum_store0 --dim 1024 --warmup 0 --iters 1`
- no-store:
  - `ncu ... -o ncu_reports/attribution_1024_no_store ... ncu_profile_abft.py --variant ablate_sum_no_partial_store --dim 1024 --warmup 0 --iters 1`

## 备注
- `ablate_sum_no_partial_store` 为防 DCE 仍有 sink 累加，因此“写回贡献”仍是趋势性证据，不是完全正交分解。
