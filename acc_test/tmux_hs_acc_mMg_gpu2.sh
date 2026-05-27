#!/usr/bin/env bash
# GPU 2: thr-mMg + rand2pow + gamma=2
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
export CUDA_VISIBLE_DEVICES=2
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
LOG=results/_tmux_hs_acc_mMg_g2.log
GAMMA=2.0
BASE=(--max-samples 200 --n-warmup 10 --gamma "$GAMMA" --seed 2026 --layer-list 0,8,16,24)
mkdir -p results
set -o pipefail
{
  echo "[acc_mMg g2] start $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[acc_mMg g2] === rand2pow thr-mMg gamma=${GAMMA} ==="
  "$PY" run_hellaswag_acc_sweep.py "${BASE[@]}" --fault-mode rand2pow
  RUN_DIR=$("$PY" -c "
from results_layout import default_results_root, results_run_dir
print(results_run_dir(
    default_results_root(),
    model_id='Qwen/Qwen2.5-7B-Instruct',
    dataset='hellaswag',
    n_total=200,
    n_warmup=10,
    gamma=float('$GAMMA'),
    fault_mode='rand2pow',
    seed=2026,
    fault_delta=None,
    acc_thr_enabled=True,
))
")
  "$PY" plot_sweep_summary.py \
    --in-csv "$RUN_DIR/csv/sweep_summary.csv" \
    --out-png-acc "$RUN_DIR/plots/sweep_acc_fault_by_layer_op.png" \
    --title "HellaSwag ACC thr-mMg rand2pow (n200 wu10 g${GAMMA} s2026)" \
    --out-png-tp-rate "$RUN_DIR/plots/sweep_tp_rate_by_layer_op.png"
  echo "[acc_mMg g2] done $RUN_DIR"
  echo "[acc_mMg g2] all finished $(date -Is)"
} 2>&1 | tee -a "$LOG"
