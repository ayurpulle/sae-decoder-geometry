#!/bin/bash
# Run ON THIS MAC: copy the code to a round-2 pod, plus the inputs job 2 needs.
#   bash gpu/push_r2.sh <host> <port> job1|job2|job3|robust|pmatch
# job3 (r2_job_models.py) needs only the code: GPT-2, Gemma-2-9b and Gemma Scope 9b are downloaded on the pod.
# Extends gpu/push.sh (same excludes, same .git_hash provenance). job2 also pushes, from this Mac:
#   $SAE_CACHE/gemmascope/L12/params.npz         -> /workspace/SAE/cache/gemmascope/L12/params.npz   (1.2 GB)
#   results/resynth/codestats_L12_pod.npz        -> /workspace/SAE/cache/acts/gemma-2-2b/blocks.12.hook_resid_post/codestats.npz
# SAE_CACHE defaults to the main checkout's cache/ (worktrees have none).
# robust (r2_job_robust.py) also pushes the saved training Sigma of Gemma-2-2b and GPT-2 (results/r2/sigma/{gemma-2-2b,gpt2}_L*.npz,
# about 0.45 GB), from which the job rebuilds stats.npz when the pod's cache/acts no longer has it.
# pmatch (r2_job_pmatch.py) pushes the saved Gemma-2-2b Sigma of layers 13 and 0, the per-latent power files
# results/r2/power/*.npz (step P of robust: the real SAEs' power and Rayleigh quotients are the targets), and the
# robust, sparse2 and real row files the job and r2_summary.py read (runs_robust.jsonl, runs_sparse2.jsonl,
# runs_real.jsonl, robust_calibration.json); a file that is newer on the pod is never overwritten (--update).
set -euo pipefail
HOST=${1:?host}; PORT=${2:?port}; WHICH=${3:?job1, job2, job3, robust or pmatch}
cd "$(dirname "$0")/.."
case "$WHICH" in job1|job2|job3|robust|pmatch) ;; *) echo "third argument must be job1, job2, job3, robust or pmatch"; exit 1;; esac
MAIN=$(cd "$(git rev-parse --git-common-dir)/.." && pwd)
SAE_CACHE=${SAE_CACHE:-$MAIN/cache}
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -i $HOME/.ssh/id_ed25519"
CODE="train_sae.py collect_acts.py resynth.py geometry.py experiments/r2_common.py experiments/r2_job_real.py
experiments/r2_job_resynth.py experiments/r2_job_models.py experiments/r2_summary.py
experiments/r2_sparse_check.py experiments/r2_job_sparse2.py experiments/r2_job_final.py experiments/r2_job_robust.py experiments/r2_job_pmatch.py experiments/r2_pmatch_gen.py gpu"
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
if [ "$WHICH" = robust ]; then
  ls results/r2/sigma/gemma-2-2b_L*.npz results/r2/sigma/gpt2_L*.npz >/dev/null || { echo "missing results/r2/sigma files"; exit 1; }
  $SSH root@"$HOST" 'mkdir -p /workspace/SAE/results/r2/sigma'
  rsync -az --ignore-existing --stats -e "$SSH" results/r2/sigma/gemma-2-2b_L*.npz results/r2/sigma/gpt2_L*.npz \
    root@"$HOST":/workspace/SAE/results/r2/sigma/
fi
if [ "$WHICH" = pmatch ]; then
  for f in results/r2/sigma/gemma-2-2b_L13.npz results/r2/sigma/gemma-2-2b_L0.npz results/r2/runs_robust.jsonl \
           results/r2/runs_sparse2.jsonl results/r2/runs_real.jsonl; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
  for L in 13 0; do
    f=results/r2/power/gemma-2-2b__blocks.$L.hook_resid_post__real__w16384__k40__s0.npz
    [ -f "$f" ] || echo "warning: $f missing (layer $L is skipped by the job until it exists)"
  done
  $SSH root@"$HOST" 'mkdir -p /workspace/SAE/results/r2/sigma /workspace/SAE/results/r2/power'
  rsync -az --ignore-existing --stats -e "$SSH" results/r2/sigma/gemma-2-2b_L13.npz results/r2/sigma/gemma-2-2b_L0.npz \
    root@"$HOST":/workspace/SAE/results/r2/sigma/
  rsync -az --update --stats -e "$SSH" results/r2/power/ root@"$HOST":/workspace/SAE/results/r2/power/
  rsync -az --update --stats -e "$SSH" results/r2/runs_robust.jsonl results/r2/runs_sparse2.jsonl results/r2/runs_real.jsonl \
    $( [ -f results/r2/robust_calibration.json ] && echo results/r2/robust_calibration.json ) root@"$HOST":/workspace/SAE/results/r2/
fi
echo "pushed $WHICH to root@$HOST:/workspace/SAE (git $(cat .git_hash))"
