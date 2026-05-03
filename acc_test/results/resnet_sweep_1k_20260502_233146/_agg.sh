#!/usr/bin/env bash
set -eo pipefail
ACC_TEST="/data/home/jinqiwen/workspace/vla_abft_estimate/acc_test"
RESULT_DIR="/data/home/jinqiwen/workspace/vla_abft_estimate/acc_test/results/resnet_sweep_1k_20260502_233146"
CONDA_SH="/data/home/jinqiwen/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="abft_cost"
cd "$ACC_TEST"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1

echo "[agg] waiting for 6 shards in $RESULT_DIR ..."
while true; do
  n=$(ls "$RESULT_DIR"/shard*.done 2>/dev/null | wc -l)
  echo "[agg] done files: $n / 6"
  if [ "$n" -ge 6 ]; then break; fi
  sleep 15
done
python aggregate_resnet_sweep.py --result-dir "$RESULT_DIR"
python plot_resnet_sweep.py --in-csv "$RESULT_DIR/master.csv" --out-dir "$RESULT_DIR"
echo "[agg] wrote master.csv + sweep_top1.png + sweep_top5.png"
touch "$RESULT_DIR/AGG.done"
