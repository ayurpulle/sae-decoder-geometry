#!/bin/bash
# Run ON THIS MAC: copy results/r2/ back from a round-2 pod. Safe to run repeatedly while the job runs: it only adds
# and updates files (no --delete), skips files still being written (*.tmp*), and resumes partial transfers.
#   bash gpu/pull_r2.sh <host> <port>
# Both pods write to results/r2/ with distinct names (runs_real.jsonl / runs_resynth.jsonl, rows/real|resynth/,
# r2_job_real.* / r2_job_resynth.*), so pulling both into one directory does not collide, except summary.json, which
# each pod writes for its own half: rerun  python experiments/r2_summary.py  locally after pulling both.
set -euo pipefail
HOST=${1:?host}; PORT=${2:?port}
cd "$(dirname "$0")/.."
mkdir -p results/r2
rsync -az --partial --stats --exclude '*.tmp' --exclude '*.tmp.npz' --exclude summary.json \
  -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519" \
  root@"$HOST":/workspace/SAE/results/r2/ results/r2/
ls results/r2 | grep -E 'DONE|FAILED|ALERT|STOP_FAILED' || echo "(no DONE/FAILED marker yet)"
for f in results/r2/runs_*.jsonl; do [ -f "$f" ] && echo "$(wc -l < "$f") rows in $f"; done
du -sh results/r2
