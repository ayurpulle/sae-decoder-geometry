#!/bin/bash
# Run ON THIS MAC: copy the code to a round-2 pod, plus the inputs job 2 needs.
#   bash gpu/push_r2.sh <host> <port> job1|job2|job3
# job3 (r2_job_models.py) needs only the code: GPT-2, Gemma-2-9b and Gemma Scope 9b are downloaded on the pod.
# Extends gpu/push.sh (same excludes, same .git_hash provenance). job2 also pushes, from this Mac:
#   $SAE_CACHE/gemmascope/L12/params.npz         -> /workspace/SAE/cache/gemmascope/L12/params.npz   (1.2 GB)
#   results/resynth/codestats_L12_pod.npz        -> /workspace/SAE/cache/acts/gemma-2-2b/blocks.12.hook_resid_post/codestats.npz
# SAE_CACHE defaults to the main checkout's cache/ (worktrees have none).
set -euo pipefail
HOST=${1:?host}; PORT=${2:?port}; WHICH=${3:?job1, job2 or job3}
cd "$(dirname "$0")/.."
case "$WHICH" in job1|job2|job3) ;; *) echo "third argument must be job1, job2 or job3"; exit 1;; esac
MAIN=$(cd "$(git rev-parse --git-common-dir)/.." && pwd)
SAE_CACHE=${SAE_CACHE:-$MAIN/cache}
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519"
CODE="train_sae.py collect_acts.py resynth.py geometry.py experiments/r2_common.py experiments/r2_job_real.py
experiments/r2_job_resynth.py experiments/r2_job_models.py experiments/r2_summary.py
experiments/r2_sparse_check.py gpu"
if [ "$WHICH" = job2 ]; then
  PARAMS=$SAE_CACHE/gemmascope/L12/params.npz
  CODESTATS=results/resynth/codestats_L12_pod.npz
  for f in "$PARAMS" "$CODESTATS"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
fi
# shellcheck disable=SC2086
if [ -n "$(git status --porcelain -- $CODE)" ]; then
  echo "warning: uncommitted or untracked code is being pushed; .git_hash will not describe it exactly"
  echo "         (each result row also carries code_sha256 of the files that ran)"
  git status --short -- $CODE
fi
git rev-parse HEAD > .git_hash
$SSH root@"$HOST" 'command -v rsync >/dev/null || { apt-get update -qq && apt-get install -y -qq rsync tmux; }; mkdir -p /workspace/SAE'
rsync -az --stats -e "$SSH" \
  --exclude .git --exclude .venv --exclude cache --exclude ext --exclude results --exclude '__pycache__' --exclude .DS_Store \
  --exclude paper --exclude .claude \
  ./ root@"$HOST":/workspace/SAE/
if [ "$WHICH" = job2 ]; then
  $SSH root@"$HOST" 'mkdir -p /workspace/SAE/cache/gemmascope/L12 /workspace/SAE/cache/acts/gemma-2-2b/blocks.12.hook_resid_post'
  rsync -az --partial --progress -e "$SSH" "$PARAMS" root@"$HOST":/workspace/SAE/cache/gemmascope/L12/params.npz
  rsync -az --partial -e "$SSH" "$CODESTATS" \
    root@"$HOST":/workspace/SAE/cache/acts/gemma-2-2b/blocks.12.hook_resid_post/codestats.npz
  $SSH root@"$HOST" 'cd /workspace/SAE && ls -la cache/gemmascope/L12/params.npz cache/acts/gemma-2-2b/blocks.12.hook_resid_post/codestats.npz'
fi
echo "pushed $WHICH to root@$HOST:/workspace/SAE (git $(cat .git_hash))"
