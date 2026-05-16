#!/usr/bin/env bash
# GPU 3: thr-mMz + fixed 10
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
export CUDA_VISIBLE_DEVICES=3
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
LOG=results/_tmux_hs_acc_zero_g3.log
BASE=(--max-samples 200 --n-warmup 10 --gamma 3.0 --seed 2026 --layer-list 0,8,16,24)
mkdir -p results
set -o pipefail
{
  echo "[acc_zero g3] start $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[acc_zero g3] === fixed 10 thr-mMz ==="
  "$PY" run_hellaswag_acc_sweep.py "${BASE[@]}" --fault-mode fixed --fault-delta 10 --acc-threshold-zero
  RUN_DIR=$("$PY" -c "
from results_layout import default_results_root, results_run_dir
print(results_run_dir(
    default_results_root(),
    model_id='Qwen/Qwen2.5-7B-Instruct',
    dataset='hellaswag',
    n_total=200,
    n_warmup=10,
    gamma=3.0,
    fault_mode='fixed',
    seed=2026,
    fault_delta=10.0,
    acc_thr_enabled=True,
    acc_thr_action='zero',
))
")
  "$PY" plot_sweep_summary.py \
    --in-csv "$RUN_DIR/csv/sweep_summary.csv" \
    --out-png-acc "$RUN_DIR/plots/sweep_acc_fault_by_layer_op.png" \
    --title "HellaSwag ACC thr-mMz fixed+10 (n200 wu10 g3 s2026)" \
    --out-png-tp-rate "$RUN_DIR/plots/sweep_tp_rate_by_layer_op.png"
  echo "[acc_zero g3] done $RUN_DIR"
  echo "[acc_zero g3] all finished $(date -Is)"
} 2>&1 | tee -a "$LOG"
