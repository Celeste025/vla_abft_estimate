#!/usr/bin/env bash
# 经 hf-mirror 预下载模型到本地 cache，避免 transformers 直连 huggingface.co 卡死。
set -euo pipefail
MODEL_ID="${1:?usage: $0 <model_id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hf_mirror_env.sh
source "${SCRIPT_DIR}/hf_mirror_env.sh"
PY="${PY:-/data/home/jinqiwen/miniconda3/envs/abft_cost/bin/python}"
echo "[mirror] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[mirror] downloading ${MODEL_ID} ..."
"$PY" - <<PY
import os
from huggingface_hub import snapshot_download

endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
repo = "${MODEL_ID}"
print(f"snapshot_download(repo={repo!r}, endpoint={endpoint!r})", flush=True)
path = snapshot_download(repo_id=repo, endpoint=endpoint, resume_download=True)
print(f"done: {path}", flush=True)
PY
