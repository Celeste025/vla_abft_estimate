#!/usr/bin/env bash
# 启动 tmux 会话 resnet_sweep_1k：GPU 1–6 各跑 9 个站点 shard，完成后 agg 窗口聚合+画图。
set -euo pipefail

ACC_TEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ACC_TEST"

CONDA_SH="${CONDA_SH:-/data/home/jinqiwen/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-abft_cost}"
LOCAL_DATASET="${LOCAL_DATASET:-/data/home/jinqiwen/data/imagenet1k_val_hf_5k}"
SESSION="${SESSION:-resnet_sweep_1k}"

RESULT_DIR="${RESULT_DIR:-$ACC_TEST/results/resnet_sweep_1k_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULT_DIR"
echo "[launch] RESULT_DIR=$RESULT_DIR" | tee "$RESULT_DIR/launch_meta.txt"

export SWEEP_RESULT_DIR="$RESULT_DIR"
python3 <<'PY'
import os
from pathlib import Path

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from inject import SITE_STRATEGY_MODULE_SCAN, list_sites

rd = Path(os.environ["SWEEP_RESULT_DIR"])
m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
sites = list_sites(
    m,
    strategy=SITE_STRATEGY_MODULE_SCAN,
    module_classes=(nn.Conv2d, nn.Linear),
)
if len(sites) != 54:
    raise SystemExit(f"expected 54 sites, got {len(sites)}")
for i in range(6):
    chunk = sites[i * 9 : (i + 1) * 9]
    (rd / f"shard{i}_sites.txt").write_text(",".join(chunk), encoding="utf-8")
    print(f"shard{i}: {len(chunk)} sites -> {chunk[0]} ... {chunk[-1]}")
PY

cat > "$RESULT_DIR/_run_shard.sh" <<'INNER'
#!/usr/bin/env bash
# 不用 set -u：conda activate/deactivate 脚本会引用可能未设置的变量。
set -eo pipefail
SHARD="${1:?shard}"
GPU="${2:?gpu}"
ACC_TEST="${3:?acc}"
RESULT_DIR="${4:?res}"
CONDA_SH="${5:?conda_sh}"
CONDA_ENV="${6:?conda_env}"
LOCAL_DATASET="${7:?data}"

cd "$ACC_TEST"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

python -u run_resnet_layer_sweep.py \
  --site-list-file "$RESULT_DIR/shard${SHARD}_sites.txt" \
  --batch-size 1 \
  --max-samples 1000 \
  --local-dataset-dir "$LOCAL_DATASET" \
  --fault-delta 10000 \
  --fault-index-mode random \
  --clear-threshold-mul 0.5 \
  --seed 2026 \
  --no-progress \
  --out-csv "$RESULT_DIR/shard${SHARD}.csv" \
  --out-json "$RESULT_DIR/shard${SHARD}.json" \
  2>&1 | tee "$RESULT_DIR/shard${SHARD}.log"

touch "$RESULT_DIR/shard${SHARD}.done"
echo "[shard${SHARD}] done on GPU ${GPU}"
INNER
chmod +x "$RESULT_DIR/_run_shard.sh"

cat > "$RESULT_DIR/_agg.sh" <<AGGINNER
#!/usr/bin/env bash
set -eo pipefail
ACC_TEST="$ACC_TEST"
RESULT_DIR="$RESULT_DIR"
CONDA_SH="$CONDA_SH"
CONDA_ENV="$CONDA_ENV"
cd "\$ACC_TEST"
# shellcheck source=/dev/null
source "\$CONDA_SH"
conda activate "\$CONDA_ENV"
export PYTHONUNBUFFERED=1

echo "[agg] waiting for 6 shards in \$RESULT_DIR ..."
while true; do
  n=\$(ls "\$RESULT_DIR"/shard*.done 2>/dev/null | wc -l)
  echo "[agg] done files: \$n / 6"
  if [ "\$n" -ge 6 ]; then break; fi
  sleep 15
done
python aggregate_resnet_sweep.py --result-dir "\$RESULT_DIR"
python plot_resnet_sweep.py --in-csv "\$RESULT_DIR/master.csv" --out-dir "\$RESULT_DIR"
echo "[agg] wrote master.csv + sweep_top1.png + sweep_top5.png"
touch "\$RESULT_DIR/AGG.done"
AGGINNER
chmod +x "$RESULT_DIR/_agg.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[launch] killing existing tmux session: $SESSION" | tee -a "$RESULT_DIR/launch_meta.txt"
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -n shard0 \
  "bash '$RESULT_DIR/_run_shard.sh' 0 1 '$ACC_TEST' '$RESULT_DIR' '$CONDA_SH' '$CONDA_ENV' '$LOCAL_DATASET'; exec bash"

for i in 1 2 3 4 5; do
  gpu=$((i + 1))
  tmux new-window -t "$SESSION" -n "shard${i}" \
    "bash '$RESULT_DIR/_run_shard.sh' ${i} ${gpu} '$ACC_TEST' '$RESULT_DIR' '$CONDA_SH' '$CONDA_ENV' '$LOCAL_DATASET'; exec bash"
done

tmux new-window -t "$SESSION" -n agg "bash '$RESULT_DIR/_agg.sh'; exec bash"

echo ""
echo "tmux session: $SESSION"
echo "windows: shard0..shard5, agg"
echo "RESULT_DIR: $RESULT_DIR"
echo "attach: tmux attach -t $SESSION"
echo "shard0 -> CUDA_VISIBLE_DEVICES=1, ... shard5 -> 6"
