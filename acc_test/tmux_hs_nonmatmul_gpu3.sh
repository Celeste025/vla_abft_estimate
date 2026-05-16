#!/usr/bin/env bash
# GPU 3: nonmatmul fixed fd10 × 3 threshold modes
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
export CUDA_VISIBLE_DEVICES=3
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
LOG=results/_tmux_hs_nonmatmul_g3.log
BASE=(--max-samples 200 --n-warmup 10 --gamma 3.0 --seed 2026 --layer-list 0,8,16,24 --site-set nonmatmul --reuse-baseline)
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
echo "[g3] start $(date -Is)"
_run() {
  local thr=$1
  shift
  echo "[g3] === fixed fd10 thr-${thr} ==="
  "$PY" run_hellaswag_acc_sweep.py "${BASE[@]}" --fault-mode fixed --fault-delta 10 "$@"
}
_run none --acc-no-threshold
_run golden
_run zero --acc-threshold-zero
echo "[g3] done $(date -Is)"
