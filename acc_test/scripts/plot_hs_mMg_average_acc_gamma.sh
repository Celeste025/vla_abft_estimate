#!/usr/bin/env bash
# Mean acc_fault vs gamma for thr-mMg sweeps (fixed+1000 and rand2pow).
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
OUT=results/qwen-qwen2.5-7b-instruct_hellaswag/average
GAMMAS=(2 3 5 10)

mkdir -p "$OUT"

"$PY" plot_sweep_average_acc.py \
  --fault-mode fixed --fault-delta 1000 \
  --gammas "${GAMMAS[@]}" \
  --site-set matmul \
  --out-dir "$OUT" \
  --out-stem mean_acc_fault_thr-mMg_fixed_fd1000

"$PY" plot_sweep_average_acc.py \
  --fault-mode rand2pow \
  --gammas "${GAMMAS[@]}" \
  --site-set matmul \
  --out-dir "$OUT" \
  --out-stem mean_acc_fault_thr-mMg_rand2pow

echo "[done] plots in $OUT"
ls -la "$OUT"/mean_acc_fault_thr-mMg_*.png
