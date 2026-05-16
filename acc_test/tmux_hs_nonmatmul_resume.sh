#!/usr/bin/env bash
# Resume golden + zero sweeps after thr-none completed (skip first _run in gpu scripts).
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
BASE=(--max-samples 200 --n-warmup 10 --gamma 3.0 --seed 2026 --layer-list 0,8,16,24 --site-set nonmatmul --reuse-baseline)

run_gpu() {
  local g=$1 fault_mode=$2
  shift 2
  export CUDA_VISIBLE_DEVICES=$g
  local log=results/_tmux_hs_nonmatmul_resume_g${g}.log
  exec >> >(tee -a "$log") 2>&1
  echo "[resume g${g}] start $(date -Is) fault=${fault_mode} $*"
  echo "[resume g${g}] === ${fault_mode} thr-golden ==="
  "$PY" run_hellaswag_acc_sweep.py "${BASE[@]}" --fault-mode "$fault_mode" "$@"
  echo "[resume g${g}] === ${fault_mode} thr-zero ==="
  "$PY" run_hellaswag_acc_sweep.py "${BASE[@]}" --fault-mode "$fault_mode" "$@" --acc-threshold-zero
  echo "[resume g${g}] done $(date -Is)"
}

run_gpu 2 rand2pow &
run_gpu 3 fixed --fault-delta 10 &
run_gpu 4 fixed --fault-delta 100 &
run_gpu 5 fixed --fault-delta 1000 &
run_gpu 6 fixed --fault-delta 10000 &
wait
echo "[resume] all GPUs finished $(date -Is)"
