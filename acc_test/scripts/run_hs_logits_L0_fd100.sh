#!/usr/bin/env bash
set -euo pipefail
ACC=/data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
cd "$ACC"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hf_mirror_env.sh
source "${SCRIPT_DIR}/hf_mirror_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python

OUT="artifacts/hellaswag_logits_diff/case0_end0_L0_mlp_down_fd100"

"$PY" run_hellaswag_logits_capture.py \
  --case-idx 0 --ending-idx 0 \
  --inject-layer 0 --inject-site mlp_down \
  --fault-delta 100 --seed 2026 \
  --out-dir "$OUT"

"$PY" plot_hellaswag_logits_heatmap.py --artifact-dir "$OUT"
