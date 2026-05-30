#!/usr/bin/env bash
set -euo pipefail
ACC=/data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
cd "$ACC"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hf_mirror_env.sh
source "${SCRIPT_DIR}/hf_mirror_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python

CASE_IDX=0
ENDING_IDX=0
LAYER=0
SITE=mlp_down
FD=100
OUT="artifacts/hellaswag_layer_hidden_diff/case${CASE_IDX}_end${ENDING_IDX}_L${LAYER}_${SITE}_fd${FD}"

"$PY" run_hellaswag_layer_hidden_diff.py \
  --case-idx "$CASE_IDX" \
  --ending-idx "$ENDING_IDX" \
  --inject-layer "$LAYER" \
  --inject-site "$SITE" \
  --fault-delta "$FD" \
  --seed 2026 \
  --out-dir "$OUT"

"$PY" plot_layer_hidden_diff.py --artifact-dir "$OUT"
