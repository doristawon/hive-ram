#!/usr/bin/env bash
# tfpp-ds41's one-GPU equivalence tests on the shortened model S11 (spec 15.6). Every stage runs on this machine
# (127.0.0.1, BUDGET_GIB each); the judgement is IDENTICAL traces (python -m tfpp.ds41.compare) against the unmodified
# fork's one-process run (python -m tfpp.ds41.ref). Runs locally, on one node first; no SSH.
#   run_shortened.sh T-ref|T0a|T0b|T0c|T1-oracle|T1-exact|T1-long|T1-multi|T1-fault|T1-http|T-X0|all
# Env: MODEL_DIR (S11, make_testmodel.sh), ENGRAM_D (~/engram-d: L1/, L4/, both/), CONTEXT (8192), PARALLEL (2),
#      MTP_DRAFTS (5), BUDGET_GIB (each stage's declared budget, 12), PORT0 (5500), HTTP_PORT (18892), OUT (results),
#      TFPP_DS41_CONF (cluster.env + toolchain.sh folder; default this script's), TFPP_DIR (the checkout to run),
#      TFPP_DS41_WORKSPACE_GIB (each stage's work-space estimate for the admission, 4; lower it for 4 stages on one GPU),
#      GPU_GUARD_PORTS (the ports of other GPU servers on this node, space-separated; default 18891, a leader's HTTP),
#      ALLOW_GPU_SHARING=1 (run beside such a server or another tfpp stage anyway).
#      TF_API_KEY / TF_API_KEY_FILE are cleared: T1-http's leader serves 127.0.0.1 without a key.
# The cross-machine tests (T1-c6, T1-c3, T-X, T1-3) run through start_stage.sh with node.env's test blocks; T-X0 here
# on each node, then compare the two traces.
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CONF=${TFPP_DS41_CONF:-$HERE}
MODE=${1:-all}
log() { printf '[tfpp-ds41 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[tfpp-ds41] ERROR: %s\n' "$*" >&2; exit 1; }

# this test's own values first: cluster.env's serve flags are production's (lesson 4: read, then restore)
keep_ctx=${CONTEXT:-8192}; keep_par=${PARALLEL:-2}; keep_drafts=${MTP_DRAFTS:-5}; keep_http=${HTTP_PORT:-18892}
keep_replay=${TF_DS_REPLAY:-}
while IFS='=' read -r _n _; do unset "$_n"; done < <(env | grep -E '^(TF_DS|TF_EXL3|TF_TP|TF_SEED|TF_RDMA|TF_COMM|TF_NCCL|TENSORFOLD|NCCL|TFPP|TRITON|CUBLAS|CUBLASLT|NVIDIA_TF32|TORCH_ALLOW_TF32)[A-Za-z0-9_]*=' | grep -vE '^(TFPP_DS41_(CONF|VENV|TF_ROOT|PY|WORKSPACE_GIB)|TFPP_DIR)=' || true)
# a key left in the shell (a production leader's set-up) would close every route of T1-http's leader but /health
# (the fork's HTTP gate, cuda/http.py:59-94): its requests carry none
unset TF_API_KEY TF_API_KEY_FILE
while IFS= read -r _line || [[ -n "$_line" ]]; do
  [[ "$_line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
  _k=${BASH_REMATCH[1]}; _v=${BASH_REMATCH[2]}; _v=${_v%%#*}; _v=${_v%"${_v##*[![:space:]]}"}
  export "$_k=$_v"
done < "$CONF/cluster.env"
unset CONTEXT PARALLEL MTP_DRAFTS HTTP_PORT HTTP_HOST MAX_TOKENS THINKING TEMPERATURE SERVED_NAME
CONTEXT=$keep_ctx; PARALLEL=$keep_par; MTP_DRAFTS=$keep_drafts; HTTP_PORT=$keep_http
[[ -z "$keep_replay" ]] || export TF_DS_REPLAY=$keep_replay
# shellcheck source=/dev/null
source "$CONF/toolchain.sh" || die "toolchain"
[[ -d "${MODEL_DIR:-}" ]] || die "MODEL_DIR (S11) is not a folder"
ENGRAM_D=${ENGRAM_D:-$HOME/engram-d}
[[ -d "$ENGRAM_D/both" ]] || die "no $ENGRAM_D/both (make_testmodel.sh)"
PROMPTS=${PROMPTS:-$TFPP_DIR/tfpp/ds41/bench_prompts.jsonl}
BUDGET_GIB=${BUDGET_GIB:-12}
PORT0=${PORT0:-5500}
OUT=${OUT:-$HOME/tfpp-ds41-tests/s11-ctx$CONTEXT-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
# other GPU users (start_stage.sh's guard): ref.py has no admission and a stage's CAP is one snapshot, so a workload that
# grows beside a test runs it out of memory halfway; stop such a server first, or ALLOW_GPU_SHARING=1
listening() { ss -ltn "sport = :$1" > "$OUT/.ss" 2>/dev/null; grep -q LISTEN "$OUT/.ss"; }   # (no pipe into grep -q: lesson 5)
for p in ${GPU_GUARD_PORTS:-18891}; do
  [[ $p =~ ^[0-9]+$ ]] || die "GPU_GUARD_PORTS: '$p' is not a port number"
  listening "$p" || continue
  msg="port $p is in use: another server on this GPU shares its memory"
  if [[ ${ALLOW_GPU_SHARING:-0} == 1 ]]; then log "WARN: $msg (ALLOW_GPU_SHARING=1)"; else die "$msg; stop it first, or ALLOW_GPU_SHARING=1"; fi
done
pgrep -f 'tfpp[.](glm|ds41)[.](serve|run|ref)' > "$OUT/.pg" 2>/dev/null || true
if [[ -s "$OUT/.pg" && ${ALLOW_GPU_SHARING:-0} != 1 ]]; then
  die "another tfpp stage runs on this node (pid $(tr -s '[:space:]' ' ' < "$OUT/.pg")): ALLOW_GPU_SHARING=1 to run beside it"
fi
NL=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["text_config"]["num_hidden_layers"])' "$MODEL_DIR/config.json")
read -r -a ENGRAM_LAYERS < <(python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["text_config"]["engram_layer_ids"])' "$MODEL_DIR/config.json")
FAILS=0
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done; PIDS=(); }
trap cleanup EXIT

drafts_args() { if [[ $MTP_DRAFTS == 0 ]]; then echo --no-drafts; else echo --mtp-drafts "$MTP_DRAFTS"; fi; }

# the Engram folder a stage [a, b) needs: both tables, one, or none
engram_for() {
  local a=$1 b=$2 have=() e
  for e in "${ENGRAM_LAYERS[@]}"; do (( a <= e && e < b )) && have+=("$e"); done
  case ${#have[@]} in
    0) echo "" ;;
    1) echo "$ENGRAM_D/L${have[0]}" ;;
    *) echo "$ENGRAM_D/both" ;;
  esac
}

# ref NAME [ref.py flags...]: the unmodified fork, one process
ref() {
  local name=$1; shift
  [[ -f "$OUT/$name.jsonl" ]] && return 0
  python -m tfpp.ds41.ref --model "$MODEL_DIR" --engram "$ENGRAM_D/both" --context "$CONTEXT" --parallel "$PARALLEL" \
    $(drafts_args) --bench "$PROMPTS" --bench-out "$OUT/$name.bench.json" --trace "$OUT/$name.jsonl" "$@" \
    > "$OUT/$name.log" 2>&1
  local rc=$?
  log "$name (reference): exit $rc (log $OUT/$name.log)"
  return $rc
}

# chain NAME "A:B C:D ..." [leader flags...]: followers in the background, the leader (bench) in the foreground
chain() {
  local name=$1 split=$2; shift 2
  local ranges=($split) n=${#ranges[@]} i rc a b eng
  PIDS=()
  for (( i = 0; i < n - 1; i++ )); do
    a=${ranges[$i]%%:*}; b=${ranges[$i]##*:}
    eng=$(engram_for "$a" "$b")
    local args=(--role follower --model "$MODEL_DIR" --layers "${ranges[$i]}" --budget-gib "$BUDGET_GIB"
                --down "127.0.0.1:$((PORT0 + i + 1))" --trace "$OUT/$name.s$i.jsonl" --connect-timeout 900)
    (( i == 0 )) || args+=(--listen "127.0.0.1:$((PORT0 + i))")
    [[ -z "$eng" ]] || args+=(--engram "$eng")
    python -m tfpp.ds41.serve "${args[@]}" > "$OUT/$name.s$i.log" 2>&1 &
    PIDS+=($!)
  done
  a=${ranges[$((n - 1))]%%:*}; b=${ranges[$((n - 1))]##*:}
  eng=$(engram_for "$a" "$b")
  local largs=(--role leader --model "$MODEL_DIR" --layers "${ranges[$((n - 1))]}" --budget-gib "$BUDGET_GIB"
               --context "$CONTEXT" --parallel "$PARALLEL" $(drafts_args) --bench "$PROMPTS"
               --bench-out "$OUT/$name.bench.json" --trace "$OUT/$name.jsonl" --connect-timeout 900)
  (( n == 1 )) || largs+=(--listen "127.0.0.1:$((PORT0 + n - 1))")
  [[ -z "$eng" ]] || largs+=(--engram "$eng")
  python -m tfpp.ds41.serve "${largs[@]}" "$@" > "$OUT/$name.leader.log" 2>&1
  rc=$?
  for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && wait "$p"; done
  PIDS=()
  log "$name [$split]: leader exit $rc (log $OUT/$name.leader.log)"
  return $rc
}

# judge NAME REF GOT [compare flags...]: to a file first, then grep (lesson 5: no pipe into grep -q under pipefail)
judge() {
  local name=$1 r=$2 g=$3; shift 3
  if python -m tfpp.ds41.compare "$OUT/$r.jsonl" "$OUT/$g.jsonl" "$@" > "$OUT/$name.compare.txt" 2>&1; then
    log "$name: PASS ($g equals $r${*:+ ($*)})"
  else
    log "$name: FAIL ($(tail -n 1 "$OUT/$name.compare.txt")); see $OUT/$name.compare.txt"
    FAILS=$((FAILS + 1))
  fi
}

t_ref() {
  ref ref || die "the reference failed"
  ref ref2 || die "the second reference failed"
  judge T-ref-repeat ref ref2
  TF_DS_GRAPHS=0 ref refng --no-graphs || die "the eager reference failed"
  judge T-ref-graphs ref refng --allow graphs,settings.graphs_env,settings.env       # eager vs graphs: the test's own change
}
t0a() { ref ref || die "reference"; chain one "0:$NL" || { FAILS=$((FAILS + 1)); return; }; judge T0a ref one; }
t0b() { ref ref || die "reference"; chain eager "0:$NL" --no-graphs || { FAILS=$((FAILS + 1)); return; }; judge T0b ref eager --allow graphs; }
t0c() {
  ref ref || die "reference"
  local c s k=0
  local splits=()
  for (( c = 1; c <= NL - 3; c++ )); do splits+=("0:$c $c:$NL"); done          # every cut the taps allow (S11: 1..8)
  splits+=("0:3 3:6 6:$NL" "0:6 6:8 8:$NL" "0:6 6:7 7:$NL" "0:2 2:5 5:8 8:$NL")
  for s in "${splits[@]}"; do
    k=$((k + 1))
    chain "c$k" "$s" || { log "T0c [$s]: the leader failed"; FAILS=$((FAILS + 1)); continue; }
    judge "T0c-$k[${s// /|}]" ref "c$k"
  done
}
# set_env NAME VALUE / restore_env NAME: a test's own setting, then the caller's value back (set or unset) -- never a
# hard-coded one, so the tests after it run in the mode the references were made in (`all`)
declare -A _saved=()
set_env() { if [[ -n "${!1+x}" ]]; then _saved[$1]="=${!1}"; else _saved[$1]="-"; fi; export "$1=$2"; }
restore_env() {
  [[ -n "${_saved[$1]+x}" ]] || return 0                    # nothing saved: leave it as it is
  local s=${_saved[$1]}
  if [[ $s == =* ]]; then export "$1=${s#=}"; else unset "$1"; fi
  unset "_saved[$1]"
}
t1_oracle() {
  ref ref || die "reference"                                  # its replies = the serial ones (drafted == serial)
  set_env TF_DS_DEPTH_POLICY fixed
  ref oref --oracle "$OUT/ref.bench.json" || die "the oracle reference failed"
  local c
  for c in 3 6; do
    chain "oc$c" "0:$c $c:$NL" --oracle "$OUT/ref.bench.json" || { FAILS=$((FAILS + 1)); continue; }
    judge "T1-oracle-c$c" oref "oc$c"
  done
  judge T1-oracle-tokens ref oref --tokens-only --allow settings.depth,settings.env   # planted drafts change no reply
  restore_env TF_DS_DEPTH_POLICY
}
t1_exact() {
  set_env TF_DS_REPLAY 0
  ref xref --bench-max-prompt 4500 || die "the exact-prefill reference failed"
  local c
  for c in 3 6 7; do
    chain "xc$c" "0:$c $c:$NL" --bench-max-prompt 4500 || { FAILS=$((FAILS + 1)); continue; }
    judge "T1-exact-c$c" xref "xc$c"
  done
  restore_env TF_DS_REPLAY
}
t1_long() {
  local ctx=$CONTEXT p=$PROMPTS
  CONTEXT=32768; PROMPTS=$TFPP_DIR/tfpp/ds41/bench_long.jsonl
  ref lref || die "the long reference failed"
  local c
  for c in 6 7; do
    chain "lc$c" "0:$c $c:$NL" || { FAILS=$((FAILS + 1)); continue; }
    judge "T1-long-c$c" lref "lc$c"
  done
  CONTEXT=$ctx; PROMPTS=$p
}
t1_multi() {
  local p=$PROMPTS
  PROMPTS=$TFPP_DIR/tfpp/ds41/bench_multi.jsonl
  ref mref || die "the concurrent reference failed"
  chain mc6 "0:6 6:$NL" || FAILS=$((FAILS + 1))
  judge T1-multi-tokens mref mc6 --tokens-only
  judge T1-multi-rows mref mc6 --by-position
  PROMPTS=$p
}
t1_fault() {
  ref ref || die "reference"
  local rc
  # 1. a leader-local failure with an ACT in flight: that request fails (DROP after the drain), the next ones match
  TFPP_FAULT=leader:after-step:3 chain fault "0:6 6:$NL"; rc=$?
  python -m tfpp.ds41.compare "$OUT/ref.jsonl" "$OUT/fault.jsonl" --tokens-only > "$OUT/T1-fault-leader.compare.txt" 2>&1
  local differ
  differ=$(grep -c 'DIFFERS\|error' "$OUT/T1-fault-leader.compare.txt" || true)
  if (( rc == 0 )) && (( differ == 1 )); then
    log "T1-fault leader: PASS (one request failed, every other reply equals the reference)"
  else
    log "T1-fault leader: FAIL (leader exit $rc, $differ replies differ: $OUT/T1-fault-leader.compare.txt)"
    FAILS=$((FAILS + 1))
  fi
  # 2. stage 0 killed mid-bench: the leader must stop with 75. The kill comes when the bench is under way (a watcher on
  #    the leader's log), not after a fixed time: loading 59 GiB and capturing the graphs takes minutes, and a kill during
  #    startup would pass without testing the thing
  rm -f "$OUT/kill.watch"
  ( t=0
    until grep -q 'bench r300#' "$OUT/kill.leader.log" 2>/dev/null; do
      sleep 2; t=$((t + 2))
      (( t < 7200 )) || { echo "timeout: the bench never reached r300" > "$OUT/kill.watch"; exit 0; }
    done
    pkill -f -- "--layers 0:6 .*--down 127.0.0.1:$((PORT0 + 1))" && echo killed > "$OUT/kill.watch" ) &
  local watcher=$!
  chain kill "0:6 6:$NL"; rc=$?
  kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
  if [[ "$(cat "$OUT/kill.watch" 2>/dev/null)" != killed ]]; then
    log "T1-fault kill: FAIL (stage 0 was not killed mid-bench: $(cat "$OUT/kill.watch" 2>/dev/null || echo 'the bench ended first'))"
    FAILS=$((FAILS + 1))
  elif (( rc == 75 )); then log "T1-fault kill: PASS (stage 0 killed mid-bench, the leader exited 75)"
  else
    log "T1-fault kill: FAIL (stage 0 was killed mid-bench, the leader exited $rc, expected 75)"
    FAILS=$((FAILS + 1))
  fi
  # 3. a follower's own failure at its 40th STEP (past the leader's self-check, in the bench): ERROR down the
  #    chain, every stage exits 75
  TFPP_FAULT=follower:on-step:40 chain ffault "0:6 6:$NL"; rc=$?
  if (( rc == 75 )) && grep -q 'injected fault at STEP 40' "$OUT/ffault.s0.log"; then
    log "T1-fault follower: PASS (the leader exited 75 after stage 0's ERROR)"
  else
    log "T1-fault follower: FAIL (the leader exited $rc; see $OUT/ffault.*.log)"
    FAILS=$((FAILS + 1))
  fi
}
t1_http() {
  local s="0:6 6:$NL" ranges=() n i a b eng port=$HTTP_PORT
  ranges=($s); n=${#ranges[@]}
  PIDS=()
  for (( i = 0; i < n - 1; i++ )); do
    a=${ranges[$i]%%:*}; b=${ranges[$i]##*:}; eng=$(engram_for "$a" "$b")
    local args=(--role follower --model "$MODEL_DIR" --layers "${ranges[$i]}" --budget-gib "$BUDGET_GIB"
                --down "127.0.0.1:$((PORT0 + i + 1))")
    (( i == 0 )) || args+=(--listen "127.0.0.1:$((PORT0 + i))")
    [[ -z "$eng" ]] || args+=(--engram "$eng")
    python -m tfpp.ds41.serve "${args[@]}" > "$OUT/http.s$i.log" 2>&1 &
    PIDS+=($!)
  done
  a=${ranges[$((n - 1))]%%:*}; b=${ranges[$((n - 1))]##*:}; eng=$(engram_for "$a" "$b")
  local largs=(--role leader --model "$MODEL_DIR" --layers "${ranges[$((n - 1))]}" --listen "127.0.0.1:$((PORT0 + n - 1))"
               --budget-gib "$BUDGET_GIB" --context "$CONTEXT" --parallel "$PARALLEL" $(drafts_args) --host 127.0.0.1
               --port "$port" --name ds41-s11 --temperature 0)
  [[ -z "$eng" ]] || largs+=(--engram "$eng")
  python -m tfpp.ds41.serve "${largs[@]}" > "$OUT/http.leader.log" 2>&1 &
  local lp=$!
  PIDS+=($lp)
  local t=0
  # /health: the one route the fork's server never closes behind a key (cuda/http.py:84), as start_stage.sh's probe
  until curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
    sleep 3; t=$((t + 3))
    (( t < 1800 )) && kill -0 $lp 2>/dev/null || { log "T1-http: the leader did not come up"; FAILS=$((FAILS + 1)); cleanup; return; }
  done
  local U="http://127.0.0.1:$port/v1/chat/completions" H='Content-Type: application/json' ok=0
  curl -sf "$U" -H "$H" -d '{"model":"ds41-s11","max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"hi"}]}' > "$OUT/http.plain.json" && ok=$((ok + 1))
  curl -sfN "$U" -H "$H" -d '{"model":"ds41-s11","max_tokens":24,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}' > "$OUT/http.stream.txt" && ok=$((ok + 1))
  curl -sf "$U" -H "$H" -d '{"model":"ds41-s11","max_tokens":24,"temperature":0,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}],"messages":[{"role":"user","content":"call f"}]}' > "$OUT/http.tools.json" && ok=$((ok + 1))
  curl -sN --max-time 2 "$U" -H "$H" -d '{"model":"ds41-s11","max_tokens":2000,"temperature":0,"stream":true,"ignore_eos":true,"messages":[{"role":"user","content":"count"}]}' > /dev/null
  sleep 3
  curl -sf "$U" -H "$H" -d '{"model":"ds41-s11","max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"after a disconnect"}]}' > "$OUT/http.after.json" && ok=$((ok + 1))
  log "T1-http: $ok/4 requests answered (outputs in $OUT/http.*)"
  (( ok == 4 )) || FAILS=$((FAILS + 1))
  # the fork's own concurrency check through HTTP: concurrent replies equal their solo runs
  local tool="${TFPP_DS41_TF_ROOT:-}/../tools/dsv41/parallel_check.py"
  if [[ -f "$tool" ]]; then
    python - "$OUT/http.oracle.jsonl" <<'PY'
import json, sys
import numpy as np
with open(sys.argv[1], "w") as f:
    for i in range(8):
        ids = [int(x) for x in np.random.default_rng(500 + i).integers(1000, 100000, 40 + 37 * i)]
        f.write(json.dumps({"kind": "chat", "index": i, "prompt_ids": ids}) + "\n")
PY
    if python "$tool" --base "http://127.0.0.1:$port" --model ds41-s11 --oracle "$OUT/http.oracle.jsonl" --tokens 48 \
         --out "$OUT/http.parallel_check.json" > "$OUT/http.parallel_check.log" 2>&1; then
      log "T1-http parallel_check: PASS"
    else
      log "T1-http parallel_check: FAIL (see $OUT/http.parallel_check.log)"; FAILS=$((FAILS + 1))
    fi
  else
    log "T1-http: no $tool (git archive ... src tools/dsv41): parallel_check skipped"
  fi
  kill -TERM $lp 2>/dev/null; wait $lp 2>/dev/null
  for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && wait "$p" 2>/dev/null; done
  PIDS=()
}
t_x0() {   # run on each node; then compare the two traces (python -m tfpp.ds41.compare A/TX0.jsonl B/TX0.jsonl)
  ref TX0 --experts 16 || die "the 16-expert reference failed"
  log "T-X0: $OUT/TX0.jsonl (copy the other node's and compare: identical means the two machines compute alike)"
}

log "S11 ($NL layers) context=$CONTEXT parallel=$PARALLEL drafts=$MTP_DRAFTS budget=${BUDGET_GIB}GiB -> $OUT"
case $MODE in
  T-ref) t_ref ;; T0a) t0a ;; T0b) t0b ;; T0c) t0c ;; T1-oracle) t1_oracle ;; T1-exact) t1_exact ;;
  T1-long) t1_long ;; T1-multi) t1_multi ;; T1-fault) t1_fault ;; T1-http) t1_http ;; T-X0) t_x0 ;;
  all) t_ref; t0a; t0b; t0c; t1_oracle; t1_exact; t1_long; t1_multi; t1_fault; t1_http ;;
  *) die "mode: T-ref | T0a | T0b | T0c | T1-oracle | T1-exact | T1-long | T1-multi | T1-fault | T1-http | T-X0 | all" ;;
esac
log "done: $FAILS failure(s); results in $OUT"
exit $(( FAILS > 0 ))
