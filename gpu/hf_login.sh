#!/bin/bash
# Run ON THE POD, interactively, once: stores your Hugging Face token under /workspace/hf.
# Gemma-2-2B is gated: the account must have accepted https://huggingface.co/google/gemma-2-2b
set -euo pipefail
cd "$(dirname "$0")/.."
source gpu/env.sh
.venv/bin/hf auth login
.venv/bin/python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('google/gemma-2-2b'); print('Gemma access OK')"
