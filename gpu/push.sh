#!/bin/bash
# Run ON THIS MAC: copy the repo to the pod.  bash gpu/push.sh <host> <port>
# <host>/<port> are the pod's public IP and the mapped port of its exposed TCP 22 (Runpod's "SSH over exposed
# TCP"; the ssh.runpod.io proxy does not support rsync/scp). Activation caches and the venv are rebuilt there.
set -euo pipefail
HOST=${1:?host}; PORT=${2:?port}
cd "$(dirname "$0")/.."
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519"
$SSH root@"$HOST" 'command -v rsync >/dev/null || { apt-get update -qq && apt-get install -y -qq rsync tmux; }; mkdir -p /workspace/SAE'
rsync -az --info=stats1 -e "$SSH" \
  --exclude .venv --exclude cache --exclude ext --exclude '__pycache__' --exclude 'results/*.log' \
  ./ root@"$HOST":/workspace/SAE/
echo "pushed to root@$HOST:/workspace/SAE"
