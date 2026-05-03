# ResNet-50 + ImageNet-1k 注错 / 保护评测

## 依赖

- `torch`、`torchvision`、`datasets`（见 `requirements.txt`）
- **真实 ImageNet-1k（HF）**：代码里 `load_dataset("ILSVRC/imagenet-1k", ...)`（组织名 `ILSVRC` + 仓库名 `imagenet-1k`）。数据集为 **gated**。仅 `hf auth login` 不够，还必须在 [ILSVRC/imagenet-1k 数据集页](https://huggingface.co/datasets/ILSVRC/imagenet-1k) 点击 **Agree and access repository**（或 Request access 并等待通过）。与 CLI 登录是**两步独立操作**。

## 一次性下载 validation 到本地（推荐，避免流式断线与退出崩溃）

```bash
# 全量 validation（~5 万）
python download_imagenet_val.py --out-dir ~/data/imagenet1k_val_hf
# 只要前 5000 条（按分片顺序下载，通常 2 个 parquet 即停，省流量）
python download_imagenet_val.py --out-dir ~/data/imagenet1k_val_hf_5k --max-samples 5000

# 评测（本地目录有多少条，--max-samples 不要超过该数）
python run_resnet_baseline.py --local-dataset-dir ~/data/imagenet1k_val_hf --max-samples 5000 --batch-size 64
```

脚本**只拉 validation 分片**（约 14 个 `validation-*-.parquet`，≈5 万张）。勿用 `datasets.load_dataset(..., split="validation")` 直接当「只下 val」——对该仓库会先把 **294 个 train parquet** 拉下来（你终端里看到的 `train-00000-of-00294`）。

占用磁盘约数 GB；仅需网络/gated 权限在 **下载脚本运行时**，`run_resnet_*` 离线可读本地 Arrow。

## 脚本（在 `acc_test/` 目录下执行）

```bash
# 基线 + 与 torchvision 公布精度对比（需 HF）
python run_resnet_baseline.py --max-samples 5000 --batch-size 64 --out-json resnet_baseline.json

# 单站点烟测：baseline / 注错 / 注错+阈值清零保护
python run_resnet_inject_smoke.py --site layer3.0.conv2 --max-samples 512 --fault-delta 10000
python run_resnet_inject_smoke.py --site layer3.0.conv2 --clear-threshold-mul 0.5 \
  --max-samples 512  # fault+protect 在脚本内已跑

# 多站点 sweep（默认 6 个 Conv/fc）
python run_resnet_layer_sweep.py --max-samples 512 --out-csv resnet_sweep.csv

# 全部 Conv2d+Linear 站点（较慢）
python run_resnet_layer_sweep.py --all-sites --max-samples 256 --out-csv resnet_sweep_all.csv

# 仅扫部分站点（逗号分隔）；或从文件读（适合 tmux 并行分片）
python run_resnet_layer_sweep.py --site-list-file ./results/my_run/shard0_sites.txt \
  --max-samples 1000 --batch-size 1 --local-dataset-dir ~/data/imagenet1k_val_hf_5k \
  --fault-delta 10000 --fault-index-mode random --no-progress --out-csv shard0.csv
```

## 全 54 站点并行 sweep（tmux + GPU 1–6）

在 `acc_test/` 下执行 `launch_resnet_sweep_tmux.sh`：会新建 tmux 会话 **`resnet_sweep_1k`**（若已存在会先 kill），窗口 **shard0…shard5** 各用 `CUDA_VISIBLE_DEVICES=1…6` 跑 9 个站点；**agg** 窗口等 6 个 `shard*.done` 后运行 `aggregate_resnet_sweep.py` 合并为 **`master.csv`**，再 `plot_resnet_sweep.py` 生成 **`sweep_top1.png` / `sweep_top5.png`**。

```bash
# 可选环境变量：LOCAL_DATASET、CONDA_SH、CONDA_ENV、SESSION
bash launch_resnet_sweep_tmux.sh
tmux attach -t resnet_sweep_1k   # 窗口 0–5 为各 shard，窗口 6 为 agg
```

结果目录形如 `results/resnet_sweep_1k_<时间戳>/`。**首个 site 打印前会先跑完 baseline**（1000 张、bs=1 时可能要数分钟），`shard*.log` 开头会有 `[sweep] baseline run ...`。

评测循环默认在 **stderr** 打印 **tqdm** 进度条（按张或按 batch）。关闭：`--no-progress`。需 `pip install tqdm`（已写入 `requirements.txt`）。

### 流式读取结束后偶发 `Aborted (core dumped)`

若终端已正常打印 **JSON 汇总**（`top1_accuracy` 等），随后又出现 `Got disconnected from remote data host` 与 `PyGILState_Release`，多为 **HuggingFace 流式/pyarrow 后台线程在解释器退出时重连** 与 Python 收尾冲突，**不代表本次评测结果无效**。已尽量在迭代结束后 `tqdm.close()` + `gc.collect()`；仍出现时可持续升级 `datasets`/`pyarrow`，或改用本地 `ImageFolder`（若后续支持）。

## 无 HF 时的流水线自测

使用随机输入（**top-1 与社区参考值不可比**）：

```bash
python run_resnet_baseline.py --synthetic --max-samples 256 --device cuda
python run_resnet_inject_smoke.py --synthetic --site layer3.0.conv2 --max-samples 128
python run_resnet_layer_sweep.py --synthetic --max-samples 128 --out-csv /tmp/s.csv
```

## 社区参考精度（`IMAGENET1K_V2`）

与 `imagenet_task.COMMUNITY_TOP1_TOP5` 一致：top-1 ≈ 0.8086，top-5 ≈ 0.9543（torchvision 官方权重页）。

## 注错语义

- 每个 `nn.Conv2d` / `nn.Linear` 的 **forward 输出** 上挂 hook，与 im2col 后 GEMM 输出元素一一对应。
- 每次 **整图一次 forward** 内在目标模块最多 **单元素** `+= fault_delta`；`--clear-exceptions` 时对 `|x| > |fault_delta| * clear_threshold_mul` 的位置清零并计入 `errors_total`。
