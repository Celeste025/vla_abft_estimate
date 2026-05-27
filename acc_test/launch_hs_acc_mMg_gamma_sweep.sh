#!/usr/bin/env bash
# Launch 6 thr-mMg HellaSwag sweeps on GPU 2-7 (2 fault modes x 3 gamma).
set -euo pipefail
ACC=/data/home/jinqiwen/workspace/vla_abft_estimate/acc_test
cd "$ACC"
chmod +x tmux_hs_acc_mMg_gpu{2,3,4,5,6,7}.sh

for g in 2 3 4 5 6 7; do
  tmux has-session -t "hs_acc_mMg_g${g}" 2>/dev/null && tmux kill-session -t "hs_acc_mMg_g${g}" || true
done

for g in 2 3 4 5 6 7; do
  tmux new-session -d -s "hs_acc_mMg_g${g}" "bash -lc 'cd ${ACC} && bash tmux_hs_acc_mMg_gpu${g}.sh'"
done

sleep 2
echo "=== tmux sessions ==="
tmux ls 2>&1 | grep -E 'hs_acc_mMg' || echo "tmux ls failed or no sessions"

echo "=== log head ==="
for g in 2 3 4 5 6 7; do
  head -n 3 "${ACC}/results/_tmux_hs_acc_mMg_g${g}.log" 2>/dev/null || echo "g${g}: no log yet"
done

echo ""
echo "Expected result dirs (under results/qwen-qwen2.5-7b-instruct_hellaswag/):"
echo "  g2: n200_wu10_g2.0_thr-mMg_fm-rand2pow_s2026"
echo "  g3: n200_wu10_g5.0_thr-mMg_fm-rand2pow_s2026"
echo "  g4: n200_wu10_g10.0_thr-mMg_fm-rand2pow_s2026"
echo "  g5: n200_wu10_g2.0_thr-mMg_fm-fixed_fd1000_s2026"
echo "  g6: n200_wu10_g5.0_thr-mMg_fm-fixed_fd1000_s2026"
echo "  g7: n200_wu10_g10.0_thr-mMg_fm-fixed_fd1000_s2026"
