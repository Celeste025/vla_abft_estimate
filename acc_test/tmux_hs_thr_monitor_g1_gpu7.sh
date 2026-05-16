#!/usr/bin/env bash
# Threshold-only monitor, gamma=1.0 (tighter bounds → expect more fp on clean data).
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
export CUDA_VISIBLE_DEVICES=7
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
LOG=results/_tmux_hs_thr_monitor_g1.log
mkdir -p results
set -o pipefail
{
  echo "[thr_monitor g=1.0] start $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "$PY" run_hellaswag_acc_threshold_monitor.py \
    --max-samples 200 --n-warmup 10 --gamma 1.0 --seed 2026 --layer-list 0,8,16,24
  echo "[thr_monitor g=1.0] done $(date -Is)"
} 2>&1 | tee -a "$LOG"
