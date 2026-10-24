#!/usr/bin/env bash
# Stop this N1X's tfpp-ds41 stage and keep its log (gzipped, newest LOG_KEEP). Stopping the leader ends the whole
# chain (it sends QUIT, each follower forwards it and exits 0); on a follower this waits for that, then stops it.
#   stop_stage.sh            Env: TFPP_DS41_CONF, TFPP_DS41_STATE (the state root), STOP_TIMEOUT (s, 30), LOG_KEEP (10)
#   DRY_RUN=1 stop_stage.sh  say what would be stopped
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CONF=${TFPP_DS41_CONF:-$HERE}
STATE_ROOT=${TFPP_DS41_STATE:-$HOME/.local/state/tfpp-ds41}
STOP_TIMEOUT=${STOP_TIMEOUT:-30}
LOG_KEEP=${LOG_KEEP:-10}
log() { printf '[tfpp-ds41 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[tfpp-ds41] WARN: %s\n' "$*" >&2; }

ROLE=""; LAYERS=""
# shellcheck source=/dev/null
source "$CONF/node.env"
# the stage's own folder, as start_stage.sh makes it (<root>/<role>-<A>-<B>); the root itself for a node.env without
# a valid LAYERS (nothing of this stage can be there then)
if [[ "$LAYERS" =~ ^([0-9]+):([0-9]+)$ ]]; then
  STATE=$STATE_ROOT/$ROLE-${BASH_REMATCH[1]}-${BASH_REMATCH[2]}
else
  STATE=$STATE_ROOT
fi
HTTP_PORT=$(sed -n 's/^[[:space:]]*HTTP_PORT=\([0-9]*\).*/\1/p' "$CONF/cluster.env" | tail -n 1)
PIDF=$STATE/$ROLE.pid
LOGF=$STATE/$ROLE.log
pid=$(cat "$PIDF" 2>/dev/null || true)
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PIDF"
  log "no running $ROLE stage here: nothing to stop"
else
  if [[ $ROLE == leader && -n "$HTTP_PORT" ]]; then
    curl -s --max-time 3 "http://127.0.0.1:$HTTP_PORT/health" > "$STATE/.health" 2>/dev/null || true
    busy=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("requests_running", 0))' "$STATE/.health" 2>/dev/null || echo 0)
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
  # (no saved log yet: ls fails, and under pipefail that must not end the script with its status)
  ls -1t "$STATE"/logs/*-"$ROLE".log.gz 2>/dev/null | tail -n +$((LOG_KEEP + 1)) | xargs -r rm -f || true
fi
log "stopped; its GPU memory is free again"
