#!/usr/bin/env bash
# Two fault configs: L0 mlp_down +100; L12 mlp_residual (layer out) +10
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
SEED=2026

run_one() {
  local layer=$1 site=$2 fd=$3
  local out="artifacts/hellaswag_layer_hidden_diff/case${CASE_IDX}_end${ENDING_IDX}_L${layer}_${site}_fd${fd}"
  echo "=== L${layer} ${site} fixed+${fd} ==="
  "$PY" run_hellaswag_layer_hidden_diff.py \
    --case-idx "$CASE_IDX" --ending-idx "$ENDING_IDX" \
    --inject-layer "$layer" --inject-site "$site" \
    --fault-delta "$fd" --seed "$SEED" --out-dir "$out"
  "$PY" plot_layer_hidden_diff.py --artifact-dir "$out"
}

run_one 0 mlp_down 100
run_one 12 mlp_residual 10

echo "Done. Results under acc_test/results/distribution/qwen-qwen2.5-7b-instruct_hellaswag/layer_hidden_diff/"
