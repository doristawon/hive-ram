#!/usr/bin/env bash
# Stop this N1X's tfpp-glm stage and keep its log (gzipped, newest LOG_KEEP). Stopping the leader ends the whole
# chain (it sends QUIT, each follower forwards it and exits 0); on a follower this waits for that, then stops it.
#   stop_stage.sh            Env: TFPP_GLM_CONF, TFPP_GLM_STATE, STOP_TIMEOUT (s, 30), LOG_KEEP (10)
#   DRY_RUN=1 stop_stage.sh  say what would be stopped
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CONF=${TFPP_GLM_CONF:-$HERE}
STATE=${TFPP_GLM_STATE:-$HOME/.local/state/tfpp-glm}
STOP_TIMEOUT=${STOP_TIMEOUT:-30}
LOG_KEEP=${LOG_KEEP:-10}
log() { printf '[tfpp-glm %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[tfpp-glm] WARN: %s\n' "$*" >&2; }

ROLE=""
# shellcheck source=/dev/null
source "$CONF/node.env"
HTTP_PORT=$(sed -n 's/^[[:space:]]*HTTP_PORT=\([0-9]*\).*/\1/p' "$CONF/cluster.env" | tail -1)
PIDF=$STATE/$ROLE.pid
LOGF=$STATE/$ROLE.log
# the pid file names this role's stage only while that pid still runs tfpp.glm.serve --role ROLE (start_stage.sh's
# test): after a crash or a WSL restart the number may belong to another process, which is never signalled
is_stage() { [[ $1 =~ ^[0-9]+$ ]] && [[ "$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null)" == *"-m tfpp.glm.serve --role $ROLE "* ]]; }
stage_pid() { local p; p=$(cat "$PIDF" 2>/dev/null) || return 0; if is_stage "$p"; then echo "$p"; fi; return 0; }
pid=$(stage_pid)
if [[ -z "$pid" ]]; then
  [[ ! -f "$PIDF" ]] || log "the pid file names no running $ROLE stage (stale): removed"
  rm -f "$PIDF"
  log "no running $ROLE stage here: nothing to stop"
else
  if [[ $ROLE == leader && -n "$HTTP_PORT" ]]; then
    busy=$(curl -s --max-time 3 "http://127.0.0.1:$HTTP_PORT/health" 2>/dev/null |
           python3 -c 'import json,sys; print(json.load(sys.stdin).get("requests_running", 0))' 2>/dev/null || echo 0)
    (( busy == 0 )) || warn "$busy request(s) still running will be cut off"
  fi
  if [[ ${DRY_RUN:-0} == 1 ]]; then log "DRY_RUN=1: would stop the $ROLE stage (pid $pid)"; exit 0; fi
  if [[ $ROLE == follower ]]; then
    for (( i = 0; i < STOP_TIMEOUT; i++ )); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && log "the follower is still up (its leader not stopped?): stopping it, the chain breaks"
  fi
  kill -TERM "$pid" 2>/dev/null || true
  for (( i = 0; i < STOP_TIMEOUT; i++ )); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    warn "still running after ${STOP_TIMEOUT}s: SIGKILL"
    kill -KILL "$pid" 2>/dev/null || true
    sleep 1
  fi
  kill -0 "$pid" 2>/dev/null && { warn "pid $pid did not exit"; exit 1; }
  rm -f "$PIDF"
fi
mkdir -p "$STATE/logs"
if [[ -f "$LOGF" ]]; then
  saved="$STATE/logs/$(date +%Y%m%d-%H%M%S)-$ROLE.log.gz"
  gzip -c "$LOGF" > "$saved" && rm -f "$LOGF" && log "log saved: $saved"
fi
if (( LOG_KEEP > 0 )); then
  # (|| true: with no rotated log yet ls exits 2, and under pipefail that would fail the stop, e.g. systemd's ExecStop)
  { ls -1t "$STATE"/logs/*-"$ROLE".log.gz 2>/dev/null || true; } | tail -n +$((LOG_KEEP + 1)) | xargs -r rm -f
fi
log "stopped; its GPU memory is free again"
