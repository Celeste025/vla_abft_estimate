# GSM8K op 统计：n5 vs n50（及与 HellaSwag n50 对照）

## 运行配置

- `seed=2026`，`max_new_tokens=64`，Qwen2.5-7B-Instruct，`abft_cost`。
- 采集：`artifacts/gsm8k_n5/`，`artifacts/gsm8k_n50/`。
- 分析：`op_stats_gsm8k_n5_analysis.json`，`op_stats_gsm8k_n50_analysis.json`。
- 作图：各目录下 `plots/`（`plot_op_stats_case_curves`、`plot_op_stats_calibration_ratio`、`plot_op_stats_ratio_max_by_layer`）。

## 标定比值图（`plot_op_stats_calibration_ratio`）

- **n5 数据**：前 5 case 标定等价于「全集标定」，比值曲线语义与 n50 上「前 5 标定、后 45 展开」不同；解读时以 **n50** 为主、n5 作小样本对照。

## 跨 testcase 极值漂移（`combined_min_max_range` 摘要）

| 数据集 | median | p90 | max | top3 unstable (range) |
|--------|--------|-----|-----|-------------------------|
| GSM8K n5 | ~1.63 | ~7.13 | ~38.75 | L26_o_proj, L25_v_proj, L24_o_proj |
| GSM8K n50 | ~4.07 | ~13.19 | ~144 | L26_mlp_down, L3_mlp_down, L27_mlp_down |
| HellaSwag n50（对照） | ~5.93 | ~23.19 | （见该目录 analysis） | 以 mlp_down 为主（此前跑过） |

**解读要点**：

1. **样本量从 5 增到 50**，median / p90 / max 的 combined range 上升属预期（更多 testcase 更容易「撞到」更大的累积极值差）。
2. **GSM8K n50** 的 top unstable 与 **HellaSwag** 类似，多出现在 **深层 `mlp_down`**；GSM8K 为 **generate（prefill+decode）**，与 HellaSwag 仅 **四次短 forward** 相比，激活尺度与漂移模式会不同。
3. **GSM8K n5** 的 top 更偏 **o_proj / v_proj**，小样本下尾部层类型会抖动，不宜过度外推。

## 产出路径速查

- `artifacts/gsm8k_n5/plots/`
- `artifacts/gsm8k_n50/plots/`
