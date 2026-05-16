#!/usr/bin/env bash
# Group 1 / GPU 4: thr-mMg + fixed deltas 10, 100, 1000 (3 runs)
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
export CUDA_VISIBLE_DEVICES=4
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
BASE=(--max-samples 200 --n-warmup 10 --gamma 3.0 --seed 2026 --layer-list 0,8,16,24)

for d in 10 100 1000; do
  echo "[grp1] === fixed fault_delta=${d} thr-mMg ===" 
  "$PY" run_hellaswag_acc_sweep.py "${BASE[@]}" --fault-mode fixed --fault-delta "$d"
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
    fault_delta=float('$d'),
    acc_thr_enabled=True,
))
")
  "$PY" plot_sweep_summary.py \
    --in-csv "$RUN_DIR/csv/sweep_summary.csv" \
    --out-png-acc "$RUN_DIR/plots/sweep_acc_fault_by_layer_op.png" \
    --title "HellaSwag ACC thr-mMg fixed+${d} (n200 wu10 g3 s2026)" \
    --out-png-tp-rate "$RUN_DIR/plots/sweep_tp_rate_by_layer_op.png"
  echo "[grp1] done $RUN_DIR"
done
echo "[grp1] all finished on GPU 4"
