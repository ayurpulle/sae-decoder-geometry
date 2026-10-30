#!/bin/sh
# Waits for Hugging Face access to gated Gemma-2-2B, then caches layer-12 activations and measures frequencies.
cd "$(dirname "$0")/.."
until [ -f ~/.cache/huggingface/token ] && [ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" https://huggingface.co/google/gemma-2-2b/resolve/main/config.json)" = "200" ]; do sleep 30; done
echo "gemma access ok $(date)"
.venv/bin/python -u collect_acts.py --backend hf --model google/gemma-2-2b --hook blocks.12.hook_resid_post \
    --ctx 1024 --batch 4 --dtype bfloat16 --n_tokens 1000000 || exit 1
.venv/bin/python -u experiments/phase1_freqs.py
