#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PYTORCH_DIR}"

conda run -n abft_cost python qwen_abft_benchmark.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --prompt-lens 128 \
  --gen-lens 32 \
  --batch-size 1 \
  --warmup-prefill 0 \
  --warmup-decode 1 \
  --iters-prefill 1 \
  --abft-enable \
  --abft-record-disable \
  --abft-sample-rate 1.0 \
  --verbose \
  --verbose-max-tokens 32 \
  --out-json abft_demo.json
