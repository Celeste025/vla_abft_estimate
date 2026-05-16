#!/usr/bin/env bash
# HellaSwag ACC: threshold-only (fm-none), 32 sites = sweep --layer-list 0,8,16,24
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
export CUDA_VISIBLE_DEVICES=7
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
LOG=results/_tmux_hs_thr_monitor.log
mkdir -p results
set -o pipefail
{
  echo "[thr_monitor] start $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "$PY" run_hellaswag_acc_threshold_monitor.py \
    --max-samples 200 --n-warmup 10 --gamma 3.0 --seed 2026 --layer-list 0,8,16,24
  echo "[thr_monitor] done $(date -Is)"
} 2>&1 | tee -a "$LOG"
