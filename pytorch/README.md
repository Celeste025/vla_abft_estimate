# Qwen2.5-1.5B PyTorch ABFT Benchmark

## 文件
- `qwen_abft_benchmark.py`：单次配置基准（baseline 或 ABFT）
- `qwen_abft_compare.py`：自动跑 baseline / all_ops / sampled 并生成报告
- `abft/`：ABFT 配置、可扩展 checker、注入器、统计器

## 依赖
- Python 3.9+
- CUDA + PyTorch
- `transformers`

## 单次运行示例

```bash
python3 qwen_abft_benchmark.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --prompt-lens 128,512,1024 \
  --gen-lens 32,128 \
  --dtype bfloat16 \
  --warmup-prefill 100 \
  --warmup-decode 100 \
  --iters-prefill 300 \
  --out-json baseline.json
```

开启 ABFT：

```bash
python3 qwen_abft_benchmark.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --prompt-lens 128,512,1024 \
  --gen-lens 32,128 \
  --abft-enable \
  --abft-sample-rate 1.0 \
  --out-json abft_all_ops.json
```

## 一键对比与报告

```bash
python3 qwen_abft_compare.py \
  --workdir . \
  --out-dir outputs \
  --report-md ABFT_QWEN_PYTORCH_REPORT.md
```

附加故障注入：

```bash
python3 qwen_abft_compare.py \
  --workdir . \
  --run-fault \
  --fault-probability 1e-7 \
  --fault-magnitude 1.0 \
  --out-dir outputs \
  --report-md ABFT_QWEN_PYTORCH_REPORT.md
```
