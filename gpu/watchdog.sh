#!/bin/bash
# Started by gpu/run_job.sh as an independent session (setsid nohup): survives the job, the launcher and tmux dying.
#   bash gpu/watchdog.sh <name> <out_dir> <deadline_epoch_s> <grace_s> <job_pid_file>
# 1. Deadline: at max_hours, no matter what, write <name>.FAILED (reason timeout) unless a marker exists, kill the
#    job, stop the pod immediately. Marker exit_code 124 (timeout) or 125 (idle), as from timeout(1).
# 2. Idle guard: while the job's PID is alive, if GPU utilisation stays under IDLE_UTIL (5) % for IDLE_MINUTES (30)
#    consecutive minutes, treat it as hung: write <name>.FAILED (reason idle), kill the job, wait the grace period
#    (so results can be pulled), stop the pod.
# Env for tests: CHECK_SECONDS (60), IDLE_MINUTES, IDLE_UTIL, GPU_UTIL_CMD (default nvidia-smi, max over GPUs), STOP_CMD.
set -u
NAME=$1 OUT=$2 DEADLINE=$3 GRACE=$4 PIDFILE=$5
ROLE=watchdog
cd "$(dirname "$0")/.."
source gpu/r2_lib.sh
CHECK=${CHECK_SECONDS:-60}
IDLE_LIMIT=$(awk -v m="${IDLE_MINUTES:-30}" 'BEGIN{printf "%d", m*60}')
IDLE_UTIL=${IDLE_UTIL:-5}
UTIL_CMD=${GPU_UTIL_CMD:-"nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1"}
idle=0 beat=0 warned=0
say "watchdog up (pid $$): deadline $(date -u -r "$DEADLINE" +%FT%TZ 2>/dev/null || date -u -d "@$DEADLINE" +%FT%TZ), idle guard <${IDLE_UTIL}% for ${IDLE_LIMIT}s, check every ${CHECK}s"
while :; do
  now=$(date +%s)
  jpid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [ "$now" -ge "$DEADLINE" ]; then
    loud "max_hours reached: stopping the pod"
    write_marker FAILED 124 "timeout: max_hours deadline reached"
    kill_tree "$jpid"
    stop_pod "watchdog deadline"
    exit 0
  fi
  if [ -n "$jpid" ] && kill -0 "$jpid" 2>/dev/null; then
    util=$(bash -c "$UTIL_CMD" 2>/dev/null | tr -dc '0-9.\n' | head -1)
    if [ -z "$util" ]; then
      [ "$warned" = 0 ] && say "GPU utilisation unavailable ($UTIL_CMD): idle guard inactive" && warned=1
      idle=0
    elif awk -v u="$util" -v t="$IDLE_UTIL" 'BEGIN{exit !(u < t)}'; then
      idle=$((idle + CHECK))
    else
      idle=0
    fi
    if [ "$idle" -ge "$IDLE_LIMIT" ]; then
      loud "GPU under ${IDLE_UTIL}% for ${idle}s while the job runs: treating it as hung"
      write_marker FAILED 125 "idle: GPU idle ${idle}s while job pid $jpid alive"
      kill_tree "$jpid"
      say "grace ${GRACE}s before stopping (pull results now)"
      sleep "$GRACE"
      stop_pod "watchdog idle guard"
      exit 0
    fi
    beat=$((beat + CHECK))
    if [ "$beat" -ge 1800 ]; then
      beat=0
      say "heartbeat: GPU ${util:-?}%, idle ${idle}s, $(df -h "$OUT" | awk 'NR==2{print $4" free"}'), $(( (DEADLINE - now) / 60 )) min to deadline"
    fi
  else
    idle=0
  fi
  sleep "$CHECK"
done
