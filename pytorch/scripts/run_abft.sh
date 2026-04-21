#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PYTORCH_DIR}"

conda run -n abft_cost python qwen_abft_benchmark.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --prompt-lens 128,512,1024 \
  --gen-lens 32,128 \
  --batch-size 1 \
  --warmup-prefill 100 \
  --warmup-decode 100 \
  --iters-prefill 300 \
  --abft-enable \
  --abft-sample-rate 1.0 \
  --verbose \
  --verbose-max-tokens 32 \
  --out-json abft.json
