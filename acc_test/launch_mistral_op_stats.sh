#!/usr/bin/env bash
set -euo pipefail
ACC=/data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
cd "$ACC"
chmod +x scripts/download_hf_model_mirror.sh tmux_mistral_op_stats_gpu0.sh

tmux has-session -t mistral_op_stats_g0 2>/dev/null && tmux kill-session -t mistral_op_stats_g0 || true
tmux new-session -d -s mistral_op_stats_g0 "bash -lc 'cd ${ACC} && bash tmux_mistral_op_stats_gpu0.sh'"

sleep 2
echo "=== tmux ==="
tmux ls 2>&1 | grep mistral_op_stats || true
echo "=== log head ==="
head -n 8 results/_tmux_mistral_op_stats_g0.log 2>/dev/null || echo "no log yet"
