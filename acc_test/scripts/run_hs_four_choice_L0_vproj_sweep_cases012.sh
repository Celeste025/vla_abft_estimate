#!/usr/bin/env bash
set -euo pipefail
ACC=/data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
cd "$ACC"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hf_mirror_env.sh
source "${SCRIPT_DIR}/hf_mirror_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python

"$PY" run_hellaswag_four_choice_sweep.py \
  --inject-layer 0 \
  --inject-site v_proj \
  --case-indices 0,1,2

for c in 0 1 2; do
  OUT="artifacts/hellaswag_four_choice/case${c}_L0_v_proj_sweep"
  "$PY" plot_hellaswag_four_choice_sweep.py --artifact-dir "$OUT"
done
