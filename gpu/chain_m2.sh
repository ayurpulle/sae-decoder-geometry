#!/bin/bash
# Run ON THE POD (setsid nohup): when r2_job_real ends cleanly, start the Gemma-2-9b check (M2 only) under its own
# launcher and watchdog, and only once those are alive retire the old launcher and watchdog so the pod is not ended
# under the new job. Kills by exact pid only: a pattern match would also hit the tmux server, whose command line
# contains the launcher's.
cd /workspace/SAE || exit 1
R=results/r2
log() { echo "[$(date -u +%FT%TZ)] [chain] $*" >> /workspace/chain.log; }
log "waiting for r2_job_real to end"
OLD_L=$(pgrep -f '^bash gpu/run_job.sh experiments/r2_job_real.py' | head -1)
OLD_W=$(cat $R/r2_job_real.watchdog.pid)
log "old launcher pid $OLD_L, old watchdog pid $OLD_W"
until [ -e $R/r2_job_real.DONE ] || [ -e $R/r2_job_real.FAILED ]; do sleep 20; done
if [ -e $R/r2_job_real.FAILED ]; then log "main job FAILED: not chaining, pod ends as planned"; exit 0; fi
log "main job DONE: starting M2"
NO_TMUX=1 PY=/workspace/SAE/.venv/bin/python JOB_ARGS='--skip_m1' setsid nohup bash gpu/run_job.sh experiments/r2_job_models.py 6 20 > /workspace/chain_m2_launcher.log 2>&1 < /dev/null &
sleep 45
NEW_W=$(cat $R/r2_job_models.watchdog.pid 2>/dev/null)
NEW_J=$(cat $R/r2_job_models.pid 2>/dev/null)
if [ -n "$NEW_W" ] && kill -0 "$NEW_W" 2>/dev/null && [ -n "$NEW_J" ] && kill -0 "$NEW_J" 2>/dev/null; then
  kill -9 "$OLD_L" 2>/dev/null; kill -9 "$OLD_W" 2>/dev/null
  log "M2 up (job $NEW_J, watchdog $NEW_W); retired old launcher $OLD_L and watchdog $OLD_W"
else
  log "M2 did not come up (job '$NEW_J', watchdog '$NEW_W'): leaving the old launcher to end the pod"
fi
