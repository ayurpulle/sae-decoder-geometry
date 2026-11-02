#!/bin/bash
# Run ON THE POD: starts the whole queue in a detached tmux session named "sae" (survives SSH disconnects).
# Watch:  tmux attach -t sae   (detach: Ctrl-b d)    or    tail -f results/gpu/queue.log
# Safe to re-run: finished caches and runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source gpu/env.sh
mkdir -p results/gpu
if tmux has-session -t sae 2>/dev/null; then echo "queue already running: tmux attach -t sae"; exit 0; fi
tmux new-session -d -s sae "source gpu/env.sh; .venv/bin/python -u experiments/gpu_queue.py 2>&1 | tee -a results/gpu/queue.log"
echo "started. tail -f results/gpu/queue.log"
