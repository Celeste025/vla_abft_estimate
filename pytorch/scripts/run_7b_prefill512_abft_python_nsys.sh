#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PYTORCH_DIR}"

OUT_PREFIX="${1:-nsys_abft_python_7b_prefill512}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NSYS_BIN="${NSYS_BIN:-nsys}"
NSIGHT_DIR="${PYTORCH_DIR}/data/nsight"
mkdir -p "${NSIGHT_DIR}"
OUT_PATH="${NSIGHT_DIR}/${OUT_PREFIX}"

"${NSYS_BIN}" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --gpu-metrics-device=all \
  --output "${OUT_PATH}" \
  "${PYTHON_BIN}" qwen_abft_benchmark.py \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --dtype bfloat16 \
    --prompt-lens 512 \
    --gen-lens 32 \
    --batch-size 1 \
    --warmup-prefill 1 \
    --warmup-decode 1 \
    --iters-prefill 1 \
    --abft-enable \
    --abft-record-disable \
    --abft-sample-rate 1.0 \
    --abft-checker-backend python \
    --nvtx-enable \
    --verbose \
    --verbose-max-tokens 32 \
    --out-json "${NSIGHT_DIR}/abft_python_7b_prefill512_nsys.json"

echo "Nsight report written to: ${OUT_PATH}.nsys-rep"
