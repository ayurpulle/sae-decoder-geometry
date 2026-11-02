#!/bin/bash
# Run ON THE POD once, from the repo root: bash gpu/setup.sh
# Reuses the image's CUDA build of PyTorch (venv with --system-site-packages) and adds the rest.
set -euo pipefail
cd "$(dirname "$0")/.."
source gpu/env.sh
mkdir -p "$HF_HOME" results/gpu
command -v tmux >/dev/null && command -v rsync >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux rsync; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q "transformers>=4.45" "datasets>=3" zstandard "huggingface_hub>=0.30" safetensors scipy matplotlib
.venv/bin/python - <<'PY'
import torch, transformers, datasets
assert torch.cuda.is_available(), "CUDA not visible to PyTorch"
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
print("transformers", transformers.__version__, "| datasets", datasets.__version__)
PY
df -h /workspace | tail -1
echo
echo "Setup OK. Next, on the pod:  bash gpu/hf_login.sh   (you, once)   then   bash gpu/run_queue.sh"
