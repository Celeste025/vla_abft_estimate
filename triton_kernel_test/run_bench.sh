#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DTYPE="${DTYPE:-fp16}"
WARMUP="${WARMUP:-20}"
REPEAT="${REPEAT:-100}"

python verify_abft.py \
  --dtype "${DTYPE}" \
  --matmul-atol 1e-2 \
  --matmul-rtol 1e-2 \
  --abft-tol 5e-1 \
  --shape 4096,4096,4096 \
  --shape 8192,4096,4096

python benchmark_abft.py \
  --dtype "${DTYPE}" \
  --warmup "${WARMUP}" \
  --repeat "${REPEAT}" \
  --shape 4096,4096,4096 \
  --shape 8192,4096,4096
