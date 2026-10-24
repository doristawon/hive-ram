#!/usr/bin/env bash
# tfpp-glm's one-GPU equivalence tests on a shortened model (spec 13.1-13.2): T0a, T0b, T1, T1-kept, T1-fault, T1-http.
# Every stage runs on this machine (127.0.0.1, --budget-gib each); the judgement is bit-equal traces
# (python -m tfpp.glm.compare). Runs locally on one node first; no SSH.
#   run_truncated.sh T0a|T0b|T1|T1-kept|T1-fault|T1-http|all
# Env: MODEL_DIR (the checkpoint: layers [0, L), the head and norm, embed), DRAFTER_DIR (DFlash2; empty: --no-drafts),
#      L (8 | 12 | 24; default 8), CONTEXT (2048: dense; 262144 for the long-context pass), PROMPTS (default
#      tfpp/glm/bench_prompts.jsonl), OUT (results folder), BUDGET_GIB (each stage's declared budget, default 24),
#      KEPT_GIB (TF_GLM_CACHE_GIB, default 8), PORT0 (5400), TFPP_GLM_CONF (cluster.env + toolchain.sh folder;
#      default this script's). MTP variant: TF_GLM_MTP=1 (needs the shards of layer 45).
# Kept prompts: a byte cap is not cut-invariant (each leader counts its own layers' snapshot bytes: G/engine.py:1234,
# 1278, G/decode.py:515), so the equivalence runs keep it out of play: KEPT_GIB well past what TF_GLM_CACHE_ENTRIES
# (32) states take (~0.1 GiB each at L=24), a budget whose room holds it (each run checks the leader kept KEPT_GIB),
# and only the count cap evicts. T1-kept is the pressure pass (TF_GLM_CACHE_GIB=0.5): replies compared, not records.
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CONF=${TFPP_GLM_CONF:-$HERE}
MODE=${1:-all}
L=${L:-8}
CONTEXT=${CONTEXT:-2048}
BUDGET_GIB=${BUDGET_GIB:-24}     # declared, not allocated: what a stage's caches may grow to (several on one GPU)
KEPT_GIB=${KEPT_GIB:-8}
PORT0=${PORT0:-5400}
log() { printf '[tfpp-glm %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[tfpp-glm] ERROR: %s\n' "$*" >&2; exit 1; }

# the same environment as every stage (cluster.env, read), then the toolchain
keep_mtp=${TF_GLM_MTP:-}
keep_ctx=$CONTEXT
keep_http=${HTTP_PORT:-}
while IFS='=' read -r _n _; do unset "$_n"; done < <(env | grep -E '^(TF_GLM|TENSORFOLD|TF_ROCE|NCCL|TFPP|TRITON|CUBLAS|CUBLASLT|NVIDIA_TF32|TORCH_ALLOW_TF32)_[A-Za-z0-9_]*=' | grep -vE '^TFPP_GLM_' || true)
while IFS= read -r _line || [[ -n "$_line" ]]; do
  [[ "$_line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
  _k=${BASH_REMATCH[1]}; _v=${BASH_REMATCH[2]}; _v=${_v%%#*}; _v=${_v%"${_v##*[![:space:]]}"}
  export "$_k=$_v"
done < "$CONF/cluster.env"
[[ -z "$keep_mtp" ]] || export TF_GLM_MTP=$keep_mtp
# cluster.env's serve flags (CONTEXT=262144, HTTP_PORT=8888) are the production leader's, not this test's
unset CONTEXT HTTP_PORT HTTP_HOST PARALLEL MAX_TOKENS THINKING VISION VISION_URLS SERVED_NAME
CONTEXT=$keep_ctx
[[ -z "$keep_http" ]] || HTTP_PORT=$keep_http
export TF_GLM_CACHE_GIB=$KEPT_GIB
# shellcheck source=/dev/null
source "$CONF/toolchain.sh" || die "toolchain"
[[ -d "${MODEL_DIR:-}" ]] || die "MODEL_DIR is not a folder"
PROMPTS=${PROMPTS:-$TFPP_DIR/tfpp/glm/bench_prompts.jsonl}
OUT=${OUT:-$HOME/tfpp-glm-tests/L$L-ctx$CONTEXT-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
case $L in
  8) TAPS=1,2,4,5,7; SPLITS=("0:4 4:8" "0:5 5:8") ;;
  12) TAPS=1,4,6,9,10; SPLITS=("0:4 4:8 8:12" "0:6 6:9 9:12") ;;
  24) TAPS=5,9,14,18,22; SPLITS=("0:8 8:16 16:24") ;;
  *) die "L is 8, 12 or 24" ;;
esac
DRAFT=(--no-drafts)
[[ -z "${DRAFTER_DIR:-}" ]] || DRAFT=(--drafter "$DRAFTER_DIR" --tap-layers "$TAPS")
COMMON=(--model "$MODEL_DIR" --truncate "$L" --context "$CONTEXT" --prefill-rows 2048 "${DRAFT[@]}"
        --budget-gib "$BUDGET_GIB" --io-timeout 600 --connect-timeout 900)
FAILS=0
PIDS=()

cleanup() { for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done; PIDS=(); }
trap cleanup EXIT

# run_chain NAME "A:B C:D ..." [extra leader flags...]: followers in the background, the leader in the foreground
run_chain() {
  local name=$1 split=$2; shift 2
  local ranges=($split) n=${#ranges[@]} i rc
  PIDS=()
  for (( i = 0; i < n - 1; i++ )); do
    local args=(--role follower --model "$MODEL_DIR" --layers "${ranges[$i]}" --budget-gib "$BUDGET_GIB"
                --down "127.0.0.1:$((PORT0 + i + 1))" --trace "$OUT/$name.s$i.jsonl" --connect-timeout 900)
    (( i == 0 )) || args+=(--listen "127.0.0.1:$((PORT0 + i))")
    python -m tfpp.glm.serve "${args[@]}" > "$OUT/$name.s$i.log" 2>&1 &
    PIDS+=($!)
  done
  local largs=(--role leader --layers "${ranges[$((n - 1))]}" "${COMMON[@]}" --bench "$PROMPTS"
               --bench-out "$OUT/$name.bench.json" --trace "$OUT/$name.jsonl")
  (( n == 1 )) || largs+=(--listen "127.0.0.1:$((PORT0 + n - 1))")
  python -m tfpp.glm.serve "${largs[@]}" "$@" > "$OUT/$name.leader.log" 2>&1
  rc=$?
  for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && wait "$p"; done
  PIDS=()
  log "$name: leader exit $rc (log $OUT/$name.leader.log)"
  local want
  want=$(LC_ALL=C printf '%.2f' "$TF_GLM_CACHE_GIB")
  if (( rc == 0 )) && ! grep -q "prompts kept in $want GiB" "$OUT/$name.leader.log"; then
    log "$name: the leader's kept-prompt bytes are not TF_GLM_CACHE_GIB=$want (the budget's room clamped them:" \
        "raise BUDGET_GIB); a clamped byte cap evicts differently at every cut"
    rc=3
  fi
  return $rc
}

judge() {   # judge NAME REF GOT [compare flags...]
  local name=$1 ref=$2 got=$3; shift 3
  if python -m tfpp.glm.compare "$OUT/$ref.jsonl" "$OUT/$got.jsonl" "$@" > "$OUT/$name.compare.txt"; then
    log "$name: PASS ($got equals $ref${*:+ ($*)})"
  else
    log "$name: FAIL ($(tail -1 "$OUT/$name.compare.txt")); see $OUT/$name.compare.txt"
    FAILS=$((FAILS + 1))
  fi
}

ref() { [[ -f "$OUT/solo.jsonl" ]] || run_chain solo "0:$L" || die "the one-process reference failed"; }

t0a() { run_chain pure "0:$L" --pure || die "pure failed"; ref; judge T0a pure solo; }
t0b() { ref; run_chain eager "0:$L" --no-graphs || die "eager failed"; judge T0b solo eager; }
t1() {
  ref
  local k=0 s
  for s in "${SPLITS[@]}"; do
    k=$((k + 1))
    run_chain "chain$k" "$s" || { log "T1 chain $k ($s): the leader failed"; FAILS=$((FAILS + 1)); continue; }
    judge "T1-$k[$s]" solo "chain$k"
  done
}
t1_kept() {   # spec 13.6 item 14: kept prompts under a 0.5 GiB byte cap; replies compared (--tokens-only)
  local k=0 s
  export TF_GLM_CACHE_GIB=0.5
  run_chain kref "0:$L" || die "the one-process kept-pressure reference failed"
  for s in "${SPLITS[@]}"; do
    k=$((k + 1))
    run_chain "kchain$k" "$s" || { log "T1-kept chain $k ($s): the leader failed"; FAILS=$((FAILS + 1)); continue; }
    judge "T1-kept-$k[$s]" kref "kchain$k" --tokens-only
  done
  export TF_GLM_CACHE_GIB=$KEPT_GIB
}
t1_fault() {
  ref
  local s=${SPLITS[0]} rc
  # 1. an exception on the leader with an ACT in flight: request 2 aborts (END abort), request 3 must match
  TFPP_FAULT=2:3 run_chain fault "$s"; rc=$?
  # to a file first: grep -q stops reading at its match, and under pipefail compare's BrokenPipeError fails the test
  python -m tfpp.glm.compare "$OUT/solo.jsonl" "$OUT/fault.jsonl" > "$OUT/T1-fault.compare.txt" 2>&1
  if (( rc == 0 )) && grep -q '^request 2: DIFFERS' "$OUT/T1-fault.compare.txt" &&
     grep -q '^request 3: IDENTICAL' "$OUT/T1-fault.compare.txt"; then
    log "T1-fault abort: PASS (request 3 after the aborted request 2 equals the reference)"
  else
    log "T1-fault abort: FAIL (leader exit $rc; request 2 must differ, request 3 match: $OUT/T1-fault.compare.txt)"
    FAILS=$((FAILS + 1))
  fi
  # 2. a follower killed mid-request: the leader must stop with 75
  local ranges=($s)
  ( sleep 20; pkill -f -- "--layers ${ranges[0]} .*--down 127.0.0.1:$((PORT0 + 1))" ) &
  run_chain kill "$s"; rc=$?
  if (( rc == 75 )); then log "T1-fault kill: PASS (the leader exited 75)"; else
    log "T1-fault kill: FAIL (the leader exited $rc, expected 75; the kill may have come after the bench ended)"
    FAILS=$((FAILS + 1))
  fi
}
t1_http() {
  local s=${SPLITS[-1]} ranges i n port=${HTTP_PORT:-8890}
  ranges=($s); n=${#ranges[@]}
  PIDS=()
  for (( i = 0; i < n - 1; i++ )); do
    local args=(--role follower --model "$MODEL_DIR" --layers "${ranges[$i]}" --budget-gib "$BUDGET_GIB"
                --down "127.0.0.1:$((PORT0 + i + 1))")
    (( i == 0 )) || args+=(--listen "127.0.0.1:$((PORT0 + i))")
    python -m tfpp.glm.serve "${args[@]}" > "$OUT/http.s$i.log" 2>&1 &
    PIDS+=($!)
  done
  python -m tfpp.glm.serve --role leader --layers "${ranges[$((n - 1))]}" --listen "127.0.0.1:$((PORT0 + n - 1))" \
    "${COMMON[@]}" --host 127.0.0.1 --port "$port" --name glm-trunc > "$OUT/http.leader.log" 2>&1 &
  local lp=$!
  PIDS+=($lp)
  local t=0
  until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
    sleep 3; t=$((t + 3)); (( t < 900 )) && kill -0 $lp 2>/dev/null || { log "T1-http: the leader did not come up"; FAILS=$((FAILS + 1)); cleanup; return; }
  done
  local U="http://127.0.0.1:$port/v1/chat/completions" H='Content-Type: application/json' ok=0
  curl -sf "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"hi"}]}' > "$OUT/http.plain.json" && ok=$((ok + 1))
  curl -sfN "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":24,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}' > "$OUT/http.stream.txt" && ok=$((ok + 1))
  curl -sf "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":48,"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}}},"messages":[{"role":"user","content":"give a"}]}' > "$OUT/http.grammar.json" && ok=$((ok + 1))
  curl -sf "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":24,"temperature":0,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}],"messages":[{"role":"user","content":"call f"}]}' > "$OUT/http.tools.json" && ok=$((ok + 1))
  curl -sN --max-time 2 "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":2000,"temperature":0,"stream":true,"messages":[{"role":"user","content":"count"}]}' > /dev/null
  sleep 3
  curl -sf "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"after a disconnect"}]}' > "$OUT/http.after.json" && ok=$((ok + 1))
  log "T1-http: $ok/5 requests answered (outputs in $OUT/http.*; token_sha against the bench: compare by hand)"
  (( ok == 5 )) || FAILS=$((FAILS + 1))
  kill -TERM $lp 2>/dev/null; wait $lp 2>/dev/null
  for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && wait "$p" 2>/dev/null; done
  PIDS=()
}

log "L=$L context=$CONTEXT taps=$TAPS drafter=${DRAFTER_DIR:-none} kept=${KEPT_GIB}GiB budget=${BUDGET_GIB}GiB -> $OUT"
case $MODE in
  T0a) t0a ;; T0b) t0b ;; T1) t1 ;; T1-kept) t1_kept ;; T1-fault) t1_fault ;; T1-http) t1_http ;;
  all) t0a; t0b; t1; t1_kept; t1_fault ;;
  *) die "mode: T0a | T0b | T1 | T1-kept | T1-fault | T1-http | all" ;;
esac
log "done: $FAILS failure(s); results in $OUT"
exit $(( FAILS > 0 ))
