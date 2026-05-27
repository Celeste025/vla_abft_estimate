#!/usr/bin/env bash
# GPU 0: Mistral-7B HellaSwag n50 op stats + M50/M5 汇总（全程 hf-mirror）
set -euo pipefail
ACC=/data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
cd "$ACC"
export CUDA_VISIBLE_DEVICES=0
PY=/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python
MODEL=mistralai/Mistral-7B-Instruct-v0.3
OUT_DIR=artifacts/mistral7b_hellaswag_n50
LOG=results/_tmux_mistral_op_stats_g0.log
mkdir -p results "$OUT_DIR"

# shellcheck source=scripts/hf_mirror_env.sh
source "${ACC}/scripts/hf_mirror_env.sh"

set -o pipefail
{
  echo "[mistral g0] start $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[mistral g0] mirror HF_ENDPOINT=${HF_ENDPOINT}"

  echo "[mistral g0] === pre-download model via mirror ==="
  bash scripts/download_hf_model_mirror.sh "$MODEL"

  echo "[mistral g0] === capture hellaswag n50 ==="
  "$PY" run_op_output_stats_capture.py \
    --benchmark hellaswag --max-samples 50 --seed 2026 \
    --model-id "$MODEL" \
    --out-json "${OUT_DIR}/op_stats_mistral7b_hellaswag_n50.json"

  echo "[mistral g0] === summarize M50/M5 max fluctuation ==="
  "$PY" scripts/summarize_op_stats_M_ratio.py \
    --in-json "${OUT_DIR}/op_stats_mistral7b_hellaswag_n50.json" \
    --out-json "${OUT_DIR}/M_ratio_summary.json" \
    --out-csv "${OUT_DIR}/M_ratio_per_site.csv"

  echo "[mistral g0] all finished $(date -Is)"
} 2>&1 | tee -a "$LOG"
