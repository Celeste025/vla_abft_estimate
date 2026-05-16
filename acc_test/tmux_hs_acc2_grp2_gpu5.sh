#!/usr/bin/env bash
# Group 2 / GPU 5: thr-mMg fixed 10000; thr-none rand2pow; thr-none fixed 10
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
export CUDA_VISIBLE_DEVICES=5
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
BASE=(--max-samples 200 --n-warmup 10 --gamma 3.0 --seed 2026 --layer-list 0,8,16,24)

echo "[grp2] === fixed 10000 thr-mMg ==="
"$PY" run_hellaswag_acc_v2_sweep.py "${BASE[@]}" --fault-mode fixed --fault-delta 10000
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
    fault_delta=10000.0,
    acc_thr_enabled=True,
))
")
"$PY" plot_sweep_summary.py \
  --in-csv "$RUN_DIR/csv/sweep_summary.csv" \
  --out-png-acc "$RUN_DIR/plots/sweep_acc_fault_by_layer_op.png" \
  --title "HellaSwag ACC v2 thr-mMg fixed+10000 (n200 wu10 g3 s2026)" \
  --out-png-tp-rate "$RUN_DIR/plots/sweep_tp_rate_by_layer_op.png"
echo "[grp2] done $RUN_DIR"

echo "[grp2] === rand2pow thr-none ==="
"$PY" run_hellaswag_acc_v2_sweep.py "${BASE[@]}" --fault-mode rand2pow --acc-no-threshold
RUN_DIR=$("$PY" -c "
from results_layout import default_results_root, results_run_dir
print(results_run_dir(
    default_results_root(),
    model_id='Qwen/Qwen2.5-7B-Instruct',
    dataset='hellaswag',
    n_total=200,
    n_warmup=10,
    gamma=3.0,
    fault_mode='rand2pow',
    seed=2026,
    fault_delta=None,
    acc_thr_enabled=False,
))
")
"$PY" plot_sweep_summary.py \
  --in-csv "$RUN_DIR/csv/sweep_summary.csv" \
  --out-png-acc "$RUN_DIR/plots/sweep_acc_fault_by_layer_op.png" \
  --title "HellaSwag ACC v2 thr-none rand2pow (n200 wu10 g3 s2026)" \
  --out-png-tp-rate "$RUN_DIR/plots/sweep_tp_rate_by_layer_op.png"
echo "[grp2] done $RUN_DIR"

echo "[grp2] === fixed 10 thr-none ==="
"$PY" run_hellaswag_acc_v2_sweep.py "${BASE[@]}" --fault-mode fixed --fault-delta 10 --acc-no-threshold
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
    acc_thr_enabled=False,
))
")
"$PY" plot_sweep_summary.py \
  --in-csv "$RUN_DIR/csv/sweep_summary.csv" \
  --out-png-acc "$RUN_DIR/plots/sweep_acc_fault_by_layer_op.png" \
  --title "HellaSwag ACC v2 thr-none fixed+10 (n200 wu10 g3 s2026)" \
  --out-png-tp-rate "$RUN_DIR/plots/sweep_tp_rate_by_layer_op.png"
echo "[grp2] done $RUN_DIR"

echo "[grp2] all finished on GPU 5"
