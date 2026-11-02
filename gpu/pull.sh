#!/bin/bash
# Run ON THIS MAC: copy the GPU results back (small JSON + logs; W_dec files stay on the pod).
# bash gpu/pull.sh <host> <port>
set -euo pipefail
HOST=${1:?host}; PORT=${2:?port}
cd "$(dirname "$0")/.."
mkdir -p results/gpu
rsync -az --stats -e "ssh -p $PORT -i $HOME/.ssh/id_ed25519" root@"$HOST":/workspace/SAE/results/gpu/ results/gpu/
ls -la results/gpu
