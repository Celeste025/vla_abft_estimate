#!/usr/bin/env bash
set -euo pipefail
ACC=/data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
cd "$ACC"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hf_mirror_env.sh
source "${SCRIPT_DIR}/hf_mirror_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
LAYER=24
N=100
OUT=artifacts/hellaswag_o_proj_tensors/L${LAYER}_n${N}_s2026
PLOT_DIR=results/distribution/qwen-qwen2.5-7b-instruct_hellaswag/L${LAYER}_o_proj_n${N}

"$PY" run_hellaswag_op_tensor_capture.py \
  --layer "$LAYER" --max-samples "$N" --seed 2026 \
  --out-dir "$OUT"

"$PY" analyze_tensor_normal_fit.py \
  --data-dir "$OUT" \
  --plot-dir "$PLOT_DIR" \
  --max-cases-label "$N" \
  --case-limits 5 10 20 50 100
