#!/usr/bin/env bash
# GPU 2: nonmatmul rand2pow × thr-none, thr-mMg, thr-mMz
set -euo pipefail
cd /data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
export CUDA_VISIBLE_DEVICES=2
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
LOG=results/_tmux_hs_nonmatmul_g2.log
BASE=(--max-samples 200 --n-warmup 10 --gamma 3.0 --seed 2026 --layer-list 0,8,16,24 --site-set nonmatmul --reuse-baseline)
mkdir -p results
exec > >(tee -a "$LOG") 2>&1
echo "[g2] start $(date -Is)"
_run() {
  local thr=$1
  shift
  echo "[g2] === rand2pow thr-${thr} ==="
  "$PY" run_hellaswag_acc_sweep.py "${BASE[@]}" --fault-mode rand2pow "$@"
}
_run none --acc-no-threshold
_run golden
_run zero --acc-threshold-zero
echo "[g2] done $(date -Is)"
