#!/bin/bash
# Run ON THE POD:  bash gpu/run_job.sh <job.py> <max_hours> [grace_minutes=20]
#   e.g.           bash gpu/run_job.sh experiments/r2_job_real.py 17
# Not inside tmux, it re-launches itself in a detached tmux session named after the job and returns
# (watch: tmux attach -t <job>, or tail -f results/r2/<job>.log). Inside tmux it:
#   1. sources gpu/env.sh;
#   2. starts gpu/watchdog.sh as an independent session (setsid nohup): stops the pod at max_hours no matter what,
#      and stops it (after the grace period) if the GPU sits under 5% for 30 consecutive minutes while the job runs;
#   3. runs the job, teeing to results/r2/<job>.log;
#   4. on ANY exit of the job (success, error, signal) writes results/r2/<job>.DONE (exit 0) or .FAILED (exit code,
#      reason, time), waits grace_minutes so results can be pulled (gpu/pull_r2.sh), then stops the pod.
# Restart-safe: a re-run replaces the previous watchdog (new deadline) and clears old markers.
# Env: STOP_CMD (dry run: replaces the pod stop), PY (python), JOB_ARGS (extra job args, e.g. --smoke),
#      R2_OUT (results dir, default results/r2), NO_TMUX=1, and the watchdog's CHECK_SECONDS, IDLE_MINUTES,
#      IDLE_UTIL, GPU_UTIL_CMD. max_hours and grace_minutes may be fractional.
set -u
JOB=${1:?usage: run_job.sh <job.py> <max_hours> [grace_minutes]}
MAX_H=${2:?max_hours}
GRACE_MIN=${3:-20}
cd "$(dirname "$0")/.."
NAME=$(basename "$JOB" .py)

if [ -z "${TMUX:-}" ] && [ "${NO_TMUX:-0}" != 1 ]; then
  command -v tmux >/dev/null || { echo "tmux not found (bash gpu/setup.sh installs it), or set NO_TMUX=1"; exit 1; }
  if tmux has-session -t "$NAME" 2>/dev/null; then echo "already running: tmux attach -t $NAME"; exit 0; fi
  tmux new-session -d -s "$NAME" "bash gpu/run_job.sh $(printf '%q ' "$@")"
  echo "started in tmux session '$NAME'.  tail -f ${R2_OUT:-results/r2}/$NAME.log"
  exit 0
fi

[ -f gpu/env.sh ] && source gpu/env.sh
OUT=${R2_OUT:-results/r2}
mkdir -p "$OUT"
ROLE=launcher
source gpu/r2_lib.sh
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
PIDFILE=$OUT/$NAME.pid
WDPID=$OUT/$NAME.watchdog.pid

for m in DONE FAILED ALERT STOP_FAILED; do [ -e "$OUT/$NAME.$m" ] && mv "$OUT/$NAME.$m" "$OUT/$NAME.$m.prev"; done
rm -f "$PIDFILE"
if [ -f "$WDPID" ] && kill -0 "$(cat "$WDPID")" 2>/dev/null; then
  say "replacing previous watchdog (pid $(cat "$WDPID"))"
  kill "$(cat "$WDPID")" 2>/dev/null
fi
SECS=$(awk -v h="$MAX_H" 'BEGIN{printf "%d", h*3600}')
GRACE=$(awk -v m="$GRACE_MIN" 'BEGIN{printf "%d", m*60}')
DEADLINE=$(( $(date +%s) + SECS ))
say "job $JOB | max_hours $MAX_H (${SECS}s) | grace ${GRACE_MIN} min | pod ${RUNPOD_POD_ID:-unset} | python $PY ${JOB_ARGS:-}"
[ -n "${STOP_CMD:-}" ] && say "DRY RUN: pod stop replaced by STOP_CMD"
if command -v setsid >/dev/null 2>&1; then
  setsid nohup bash gpu/watchdog.sh "$NAME" "$OUT" "$DEADLINE" "$GRACE" "$PIDFILE" >> "$OUT/$NAME.watchdog.log" 2>&1 < /dev/null &
else  # macOS (local dry run): no setsid; nohup + disown still survives the launcher
  nohup bash gpu/watchdog.sh "$NAME" "$OUT" "$DEADLINE" "$GRACE" "$PIDFILE" >> "$OUT/$NAME.watchdog.log" 2>&1 < /dev/null &
fi
echo $! > "$WDPID"
disown $! 2>/dev/null
say "watchdog pid $(cat "$WDPID")"

SIG=""
on_signal() { SIG=$1; say "launcher got SIG$1: stopping the job"; kill_tree "$(cat "$PIDFILE" 2>/dev/null)"; }
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP

# shellcheck disable=SC2086
"$PY" -u "$JOB" ${JOB_ARGS:-} > >(tee -a "$OUT/$NAME.log") 2>&1 &
JPID=$!
echo "$JPID" > "$PIDFILE"
say "job pid $JPID"
while :; do
  wait "$JPID"
  CODE=$?
  kill -0 "$JPID" 2>/dev/null || break  # wait returned because of a trapped signal: wait again for the real exit
done
rm -f "$PIDFILE"
sleep 1  # let tee flush

if [ "$CODE" = 0 ] && [ -z "$SIG" ]; then
  write_marker DONE 0 "job exited 0"
else
  write_marker FAILED "$CODE" "job exited $CODE${SIG:+ after launcher SIG$SIG}"
fi

# Grace period: SIGHUP (tmux closed) is ignored; SIGTERM/SIGINT skip the wait and stop at once.
trap '' HUP
trap 'say "signal during grace: stopping now"; GRACE_CUT=1' TERM INT
GRACE_CUT=0
say "grace period ${GRACE_MIN} min before stopping the pod: pull results now (gpu/pull_r2.sh)"
end=$(( $(date +%s) + GRACE ))
while [ "$GRACE_CUT" = 0 ] && [ "$(date +%s)" -lt "$end" ]; do sleep 5 & wait $!; done
stop_pod "job finished ($(cat "$OUT/$NAME.DONE" "$OUT/$NAME.FAILED" 2>/dev/null | grep status= | head -1))"
exit "$CODE"
