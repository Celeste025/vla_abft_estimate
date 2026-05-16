#!/usr/bin/env bash
# Merge matmul + nonmatmul CSVs and redraw plots for all HellaSwag run dirs (17).
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
ROOT=results/qwen-qwen2.5-7b-instruct_hellaswag

_title_from_run() {
  local name=$1
  local g thr fm
  g=$(echo "$name" | sed -n 's/.*_\(g[0-9.]*\)_thr-.*/\1/p')
  thr=$(echo "$name" | sed -n 's/.*_thr-\([^_]*\)_fm-.*/\1/p')
  fm=$(echo "$name" | sed -n 's/.*_fm-\([^_]*\)_s.*/\1/p' | tr '_' ' ')
  if [ "$fm" = "none" ]; then
    echo "HellaSwag threshold monitor thr-${thr} (${g} n200 wu10 s2026)"
  else
    echo "HellaSwag ACC thr-${thr} ${fm} (n200 wu10 ${g} s2026)"
  fi
}

for run_dir in "$ROOT"/n200_wu10_*; do
  [ -d "$run_dir/csv" ] || continue
  name=$(basename "$run_dir")
  title=$(_title_from_run "$name")
  plots="$run_dir/plots"
  mkdir -p "$plots"

  if [ -f "$run_dir/csv/sweep_summary.csv" ] || [ -f "$run_dir/csv/sweep_summary_nonmatmul.csv" ]; then
    echo "[plot] sweep $name"
    "$PY" plot_sweep_summary.py \
      --run-dir "$run_dir" \
      --out-png-acc "$plots/sweep_acc_fault_by_layer_op.png" \
      --out-png-tp-rate "$plots/sweep_tp_rate_by_layer_op.png" \
      --title "$title"
  elif [ -f "$run_dir/csv/threshold_monitor_by_site.csv" ] || \
       [ -f "$run_dir/csv/threshold_monitor_nonmatmul_by_site.csv" ]; then
    echo "[plot] monitor $name"
    "$PY" plot_sweep_summary.py \
      --run-dir "$run_dir" \
      --out-png-acc "$plots/threshold_fpr_by_layer_op.png" \
      --title "$title"
  else
    echo "[skip] $name (no known csv)"
  fi
done

echo "[plot_all] finished $(date -Is)"
