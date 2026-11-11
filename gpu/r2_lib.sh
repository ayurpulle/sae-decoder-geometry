# Sourced by gpu/run_job.sh and gpu/watchdog.sh. Needs OUT (results dir), NAME (job name) set by the caller.
#
# stop_pod: stops THIS pod using its own identity. Order (Runpod docs, checked 2026-11-10):
#   1. runpodctl pod stop "$RUNPOD_POD_ID"            (current CLI: docs.runpod.io/pods/manage-pods)
#   2. runpodctl stop pod "$RUNPOD_POD_ID"            (older runpodctl releases)
#   3. POST https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop          Authorization: Bearer $RUNPOD_API_KEY
#   4. POST https://api.runpod.io/v2/pods/$RUNPOD_POD_ID/action {"action":"stop"}  (same key)
#   5. GraphQL podStop mutation at https://api.runpod.io/graphql            (same key)
# RUNPOD_POD_ID and a pod-scoped RUNPOD_API_KEY are set inside Runpod pods (docs: pods/references/environment-variables).
# Never falls back to killing PID 1 or `shutdown`: if every route fails it retries, then logs loudly and leaves
# <OUT>/<NAME>.STOP_FAILED for the operator.
# Dry run: STOP_CMD='...' replaces all of the above with that shell command (for local tests).

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

say() {  # message to stdout and the job log
  echo "[$(ts)] [${ROLE:-launcher}] $*" | tee -a "$OUT/$NAME.log"
}

loud() {
  local m="!!!!!!!! [$(ts)] [${ROLE:-launcher}] $* !!!!!!!!"
  echo "$m" | tee -a "$OUT/$NAME.log" >&2
  echo "$m" >> "$OUT/$NAME.ALERT"
  [ "${NO_WALL:-0}" = 1 ] || { command -v wall >/dev/null 2>&1 && echo "$m" | wall 2>/dev/null; } || true
}

write_marker() {  # write_marker DONE|FAILED <exit_code> <reason>; the first marker written wins
  local kind=$1 code=$2 reason=$3 f
  f="$OUT/$NAME.$kind"
  if [ -e "$OUT/$NAME.DONE" ] || [ -e "$OUT/$NAME.FAILED" ]; then
    say "marker already present ($(ls "$OUT/$NAME".DONE "$OUT/$NAME".FAILED 2>/dev/null | xargs -n1 basename)); not writing $kind ($reason)"
    return 0
  fi
  printf 'job=%s\nstatus=%s\nexit_code=%s\nreason=%s\ntime=%s\npod=%s\n' \
    "$NAME" "$kind" "$code" "$reason" "$(ts)" "${RUNPOD_POD_ID:-unknown}" > "$f.tmp" && mv "$f.tmp" "$f"
  say "wrote $f: exit_code=$code reason=$reason"
}

kill_tree() {  # kill_tree <pid>: TERM the job and its children (train_sae / collect_acts / resynth), then KILL
  local pid=$1
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 0
  pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null
  return 0
}

_t() { if command -v timeout >/dev/null 2>&1; then timeout 60 "$@"; else "$@"; fi; }

_terminate_once() {  # POD_END=terminate: delete the pod outright (use only when results live on a network volume)
  local id=${RUNPOD_POD_ID:-} key=${RUNPOD_API_KEY:-} r
  [ -n "$id" ] && [ -n "$key" ] || return 1
  _t curl -fsS -X POST "https://api.runpod.io/v2/pods/$id/action" -H "Authorization: Bearer $key" \
    -H 'Content-Type: application/json' --data '{"action":"terminate"}' && { say "terminated via REST v2 action"; return 0; }
  _t curl -fsS -X DELETE "https://api.runpod.io/v2/pods/$id" -H "Authorization: Bearer $key" \
    && { say "terminated via REST v2 delete"; return 0; }
  r=$(_t curl -fsS -X POST "https://api.runpod.io/graphql" -H "Authorization: Bearer $key" \
    -H 'Content-Type: application/json' \
    --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$id\\\"}) }\"}") \
    && ! grep -q '"errors"' <<<"$r" && { say "terminated via GraphQL podTerminate"; return 0; }
  say "terminate failed (${r:-no response}); falling back to stop"
  return 1
}

_stop_once() {
  local id=${RUNPOD_POD_ID:-} key=${RUNPOD_API_KEY:-} r
  if [ -z "$id" ]; then loud "RUNPOD_POD_ID is not set: cannot identify this pod"; return 1; fi
  if [ "${POD_END:-stop}" = terminate ]; then sync; _terminate_once && return 0; fi
  if command -v runpodctl >/dev/null 2>&1; then
    _t runpodctl pod stop "$id" && { say "stopped via 'runpodctl pod stop'"; return 0; }
    _t runpodctl stop pod "$id" && { say "stopped via 'runpodctl stop pod'"; return 0; }
    say "runpodctl present but both stop forms failed"
  else
    say "runpodctl not found"
  fi
  if [ -z "$key" ]; then say "RUNPOD_API_KEY not set: no API fallback"; return 1; fi
  _t curl -fsS -X POST "https://rest.runpod.io/v1/pods/$id/stop" -H "Authorization: Bearer $key" \
    && { say "stopped via REST v1"; return 0; }
  _t curl -fsS -X POST "https://api.runpod.io/v2/pods/$id/action" -H "Authorization: Bearer $key" \
    -H 'Content-Type: application/json' --data '{"action":"stop"}' && { say "stopped via REST v2 action"; return 0; }
  r=$(_t curl -fsS -X POST "https://api.runpod.io/graphql" -H "Authorization: Bearer $key" \
    -H 'Content-Type: application/json' \
    --data "{\"query\":\"mutation { podStop(input: {podId: \\\"$id\\\"}) { id desiredStatus } }\"}") \
    && ! grep -q '"errors"' <<<"$r" && { say "stopped via GraphQL podStop: $r"; return 0; }
  say "GraphQL podStop failed: ${r:-no response}"
  return 1
}

stop_pod() {  # stop_pod <reason>
  say "STOPPING POD (${RUNPOD_POD_ID:-unknown}): $1"
  if [ -n "${STOP_CMD:-}" ]; then
    say "dry run: STOP_CMD=$STOP_CMD"
    ROLE=${ROLE:-} bash -c "$STOP_CMD"
    return $?
  fi
  local n=${STOP_RETRIES:-10}
  for i in $(seq 1 "$n"); do
    _stop_once && return 0
    loud "pod stop attempt $i/$n FAILED; retrying in ${STOP_RETRY_SECONDS:-60}s"
    sleep "${STOP_RETRY_SECONDS:-60}"
  done
  loud "COULD NOT STOP POD ${RUNPOD_POD_ID:-?} AFTER $n ATTEMPTS: IT IS STILL BILLING. Stop it in the Runpod console."
  touch "$OUT/$NAME.STOP_FAILED"
  return 1
}
