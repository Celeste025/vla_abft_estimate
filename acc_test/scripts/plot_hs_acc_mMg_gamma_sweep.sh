#!/usr/bin/env bash
# Re-plot 6 HellaSwag ACC thr-mMg sweeps: rand2pow / fixed+1000 × gamma 2/5/10.
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
ROOT=results/qwen-qwen2.5-7b-instruct_hellaswag
LOG=results/_plot_hs_acc_mMg_gamma_sweep.log

RUNS=(
  n200_wu10_g2.0_thr-mMg_fm-rand2pow_s2026
  n200_wu10_g5.0_thr-mMg_fm-rand2pow_s2026
  n200_wu10_g10.0_thr-mMg_fm-rand2pow_s2026
  n200_wu10_g2.0_thr-mMg_fm-fixed_fd1000_s2026
  n200_wu10_g5.0_thr-mMg_fm-fixed_fd1000_s2026
  n200_wu10_g10.0_thr-mMg_fm-fixed_fd1000_s2026
)

_title_from_run() {
  local name=$1
  local g thr fm
  g=$(echo "$name" | sed -n 's/.*_\(g[0-9.]*\)_thr-.*/\1/p')
  thr=$(echo "$name" | sed -n 's/.*_thr-\([^_]*\)_fm-.*/\1/p')
  fm=$(echo "$name" | sed -n 's/.*_fm-\([^_]*\)_s.*/\1/p' | tr '_' ' ')
  echo "HellaSwag ACC thr-${thr} ${fm} (n200 wu10 ${g} s2026)"
}

{
  echo "[plot_mMg_gamma] start $(date -Is)"
  for name in "${RUNS[@]}"; do
    run_dir="$ROOT/$name"
    if [ ! -f "$run_dir/csv/sweep_summary.csv" ]; then
      echo "[skip] $name (missing sweep_summary.csv)"
      continue
    fi
    title=$(_title_from_run "$name")
    plots="$run_dir/plots"
    mkdir -p "$plots"
    echo "[plot] $name"
    "$PY" plot_sweep_summary.py \
      --run-dir "$run_dir" \
      --out-png-acc "$plots/sweep_acc_fault_by_layer_op.png" \
      --out-png-tp-rate "$plots/sweep_tp_rate_by_layer_op.png" \
      --title "$title"
  done
  echo "[plot_mMg_gamma] finished $(date -Is)"
} 2>&1 | tee -a "$LOG"
