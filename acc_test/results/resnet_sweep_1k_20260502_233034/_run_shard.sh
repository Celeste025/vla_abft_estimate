#!/usr/bin/env bash
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

python run_resnet_layer_sweep.py \
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
