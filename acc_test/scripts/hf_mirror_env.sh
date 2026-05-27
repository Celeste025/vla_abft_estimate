# Hugging Face 国内镜像（hf-mirror.com）。在 bash -lc 或 source 后使用。
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
# datasets 也走镜像（若 capture 拉 hellaswag）
export HF_DATASETS_ENDPOINT=https://hf-mirror.com
