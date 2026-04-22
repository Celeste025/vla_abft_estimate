#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PYTORCH_DIR}"

conda run -n abft_cost python qwen_abft_benchmark.py \
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
  --abft-checker-backend fused \
  --nvtx-enable \
  --verbose \
  --verbose-max-tokens 32 \
  --out-json abft_7b_prefill512_fused.json
