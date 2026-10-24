#!/usr/bin/env bash
# tfpp-glm v2's one-GPU equivalence tests on a shortened model (spec v2 section 16): --parallel P streams over a chain
# of processes on this machine (127.0.0.1, --budget-gib each). run_truncated.sh (v1) is not touched; this is its sibling.
# The judgement is bit-equal leader traces of a deterministic lockstep schedule (python -m tfpp.glm.v2compare strict):
# a one-process reference (T2m-solo) against the same schedule over a chain (T2m-chain), and then the level-A checks
# (replies against the P=1 serial path). Runs locally on one node first; no SSH.
#   run_v2.sh T0a-v2|T0b-v2|T1-v2|T1-v2-p1|T1-v2-free|T0e|T1-v2-fault|T1-v2-http|T2m|ref-v2|all   (all = T0a-v2 T0b-v2 T1-v2 T1-v2-p1 T1-v2-free)
#     T0a-v2       --pure (MIA's own _compute, no tfpp parts) against the one-process reference      strict
#     T0b-v2       --no-graphs against the reference                                                   strict
#     T1-v2        every split of SPLITS x every schedule of SCHEDULES against the reference            strict + stages
#     T2m          one split (the first), one schedule (the first): T2m-solo and T2m-chain             strict + stages
#     T1-v2-p1     level A: the replies of AS1 over the chain against P=1 --no-drafts (v1 path)        tokens
#     T1-v2-free   level A: FREE1 / FREE2 / FREE4 over the chain against the same P=1 replies          tokens
#     T0e          one process: a stream's row in a batched window is that stream's window alone      bits
#     T1-v2-fault  faults at every protocol point, a follower killed / stopped; each followed by a clean AS1
#     T1-v2-http   the HTTP path: the v1 curls, eight at once, a client leaving
# Env: MODEL_DIR, DRAFTER_DIR (empty: --no-drafts), L (8 | 12 | 24 | 26; default 8; at 26 only the one-process modes
#      T0a-v2 T0b-v2 T0e ref-v2 run here -- the two 26-layer stages together do not fit one GPU; the cross-machine
#      26-layer cuts (19 and 20) go through start_stage.sh and node.env, spec v2 16.3b), CONTEXT (2048 | 262144),
#      PARALLEL (4), POOL_ROWS (default 16384 at CONTEXT<=8192, else 65536), KEPT_ENTRIES (32), SCHEDULES (default
#      "AS1 AS2 AS5"; AS3 and AS4 carry their own pool plan, "suggest" in the schedule file: POOL_ROWS=6144, KEPT_ENTRIES=4,
#      used for the reference and the chains of that schedule unless POOL_ROWS / KEPT_ENTRIES are set), OUT, BUDGET_GIB (24), KEPT_GIB (TF_GLM_CACHE_GIB, 8),
#      PORT0 (5400), TFPP_GLM_CONF. Nothing here changes a v1 setting: v2's knobs are the leader's flags (--parallel,
#      --pool-rows, --kept-entries, --mbench); the followers take P and the pool from GO / AGREE.
# The pool plan is pinned (POOL_ROWS, KEPT_ENTRIES) so that every cut and the reference hold the same pool, the same
# kept-prompt count and the same limit: that is what makes admissions, evictions and moves equal at every cut.
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CONF=${TFPP_GLM_CONF:-$HERE}
MODE=${1:-all}
L=${L:-8}
CONTEXT=${CONTEXT:-2048}
P=${PARALLEL:-4}
BUDGET_GIB=${BUDGET_GIB:-24}
KEPT_GIB=${KEPT_GIB:-8}
PORT0=${PORT0:-5400}
POOL_USER=${POOL_ROWS:-}                  # (set by the caller: wins over a schedule's own plan)
KEPT_USER=${KEPT_ENTRIES:-}
KEPT_ENTRIES=${KEPT_ENTRIES:-32}
SCHEDULES=${SCHEDULES:-"AS1 AS2 AS5"}
if [[ -z "${POOL_ROWS:-}" ]]; then
  if (( CONTEXT <= 8192 )); then POOL_ROWS=16384; else POOL_ROWS=65536; fi
fi
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
# cluster.env's serve flags (CONTEXT=262144, HTTP_PORT=8888, PARALLEL=1) are the production leader's, not this test's
unset CONTEXT HTTP_PORT HTTP_HOST PARALLEL MAX_TOKENS THINKING VISION VISION_URLS SERVED_NAME
CONTEXT=$keep_ctx
[[ -z "$keep_http" ]] || HTTP_PORT=$keep_http
export TF_GLM_CACHE_GIB=$KEPT_GIB
# a lockstep schedule admits what it names at the iteration it names: no waiting for company (spec v2 14.2)
export TF_GLM_MULTI_PREFILL_WAIT_MS=0
# shellcheck source=/dev/null
source "$CONF/toolchain.sh" || die "toolchain"
[[ -d "${MODEL_DIR:-}" ]] || die "MODEL_DIR is not a folder"
PROMPTS=${PROMPTS:-$TFPP_DIR/tfpp/glm/bench_prompts.jsonl}
SCHED_DIR=${SCHED_DIR:-$TFPP_DIR/tfpp/glm/schedules}
OUT=${OUT:-$HOME/tfpp-glm-tests/v2-L$L-ctx$CONTEXT-P$P-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
case $L in
  8) TAPS=1,2,4,5,7; SPLITS=("0:4 4:8" "0:5 5:8") ;;
  12) TAPS=1,4,6,9,10; SPLITS=("0:4 4:8 8:12" "0:6 6:9 9:12") ;;
  24) TAPS=5,9,14,18,22; SPLITS=("0:8 8:16 16:24") ;;
  26) TAPS=5,9,14,19,24; SPLITS=("0:19 19:26" "0:20 20:26") ;;
  *) die "L is 8, 12, 24 or 26" ;;
esac
DRAFT=(--no-drafts)
[[ -z "${DRAFTER_DIR:-}" ]] || DRAFT=(--drafter "$DRAFTER_DIR" --tap-layers "$TAPS")
COMMON=(--model "$MODEL_DIR" --truncate "$L" --context "$CONTEXT" --prefill-rows 2048 "${DRAFT[@]}"
        --budget-gib "$BUDGET_GIB" --io-timeout 600 --connect-timeout 900)
V2=(--parallel "$P" --pool-rows "$POOL_ROWS" --kept-entries "$KEPT_ENTRIES")
SCHED_POOL=$POOL_ROWS; SCHED_ENTRIES=$KEPT_ENTRIES

# sched_plan SCHEDULE: the pool plan of a schedule's runs (SCHED_POOL / SCHED_ENTRIES / V2S): its "suggest" (AS3: a small
# pool, AS4: four kept states) unless the caller pinned POOL_ROWS / KEPT_ENTRIES; every run of one schedule, the
# one-process reference included, takes the same plan
sched_plan() {
  # sp/se must be local: bash locals are dynamically scoped, and t1/t1_free hold the current split in their own 'sp'
  local f="$SCHED_DIR/$1.json" sug="" sp se
  SCHED_POOL=$POOL_ROWS; SCHED_ENTRIES=$KEPT_ENTRIES
  if [[ -f "$f" ]]; then
    sug=$(python - "$f" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8")).get("suggest") or {}
print(s.get("pool_rows", "-"), s.get("kept_entries", "-"))
PY
) || sug=""
    read -r sp se <<< "$sug"
    [[ -z "$POOL_USER" && "${sp:--}" != "-" ]] && SCHED_POOL=$sp
    [[ -z "$KEPT_USER" && "${se:--}" != "-" ]] && SCHED_ENTRIES=$se
  fi
  V2S=(--parallel "$P" --pool-rows "$SCHED_POOL" --kept-entries "$SCHED_ENTRIES")
  return 0
}
FAILS=0
PIDS=()
FRC=()
RUN_PIDS=()                               # the last run_chain's followers (reap empties PIDS)

cleanup() { for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null; done; PIDS=(); }
trap cleanup EXIT

# run_chain NAME "A:B C:D ..." SCHEDULE [extra leader flags...]: followers in the background, the leader in the foreground.
# Leader exit code returned; the followers' in FRC. S0_ENV="KEY=value" puts one variable in stage 0's environment only,
# LEADER_ENV likewise for the leader (fault injection: TFPP_GLM_FAULT_FOLLOWER / TFPP_GLM_FAULT_V2).
run_chain() {
  local name=$1 split=$2 sched=$3; shift 3
  local ranges=($split) n=${#ranges[@]} i rc
  local tr_f=() tr_l=()
  sched_plan "$sched"
  if [[ -z "${TRACE_OFF:-}" ]]; then tr_l=(--trace "$OUT/$name.jsonl"); fi
  PIDS=(); FRC=()
  for (( i = 0; i < n - 1; i++ )); do
    [[ -n "${TRACE_OFF:-}" ]] || tr_f=(--trace "$OUT/$name.s$i.jsonl")
    local args=(--role follower --model "$MODEL_DIR" --layers "${ranges[$i]}" --budget-gib "$BUDGET_GIB"
                --down "127.0.0.1:$((PORT0 + i + 1))" "${tr_f[@]}" --connect-timeout 900
                --io-timeout "${IO_TIMEOUT:-600}")
    (( i == 0 )) || args+=(--listen "127.0.0.1:$((PORT0 + i))")
    if (( i == 0 )) && [[ -n "${S0_ENV:-}" ]]; then
      env "$S0_ENV" python -m tfpp.glm.serve "${args[@]}" > "$OUT/$name.s$i.log" 2>&1 &
    else
      python -m tfpp.glm.serve "${args[@]}" > "$OUT/$name.s$i.log" 2>&1 &
    fi
    PIDS+=($!)
  done
  RUN_PIDS=("${PIDS[@]}")
  local largs=(--role leader --layers "${ranges[$((n - 1))]}" "${COMMON[@]}" "${V2S[@]}"
               --mbench "$SCHED_DIR/$sched.json" --bench-out "$OUT/$name.bench.json" "${tr_l[@]}"
               --io-timeout "${IO_TIMEOUT:-600}")
  (( n == 1 )) || largs+=(--listen "127.0.0.1:$((PORT0 + n - 1))")
  if [[ -n "${LEADER_ENV:-}" ]]; then
    env "$LEADER_ENV" python -m tfpp.glm.serve "${largs[@]}" "$@" > "$OUT/$name.leader.log" 2>&1
  else
    python -m tfpp.glm.serve "${largs[@]}" "$@" > "$OUT/$name.leader.log" 2>&1
  fi
  rc=$?
  reap
  log "$name: leader exit $rc, followers ${FRC[*]:-none} (log $OUT/$name.leader.log)"
  if (( rc == 0 )) && ! grep -q "^\[tfpp-glm\] --parallel $P: pool $SCHED_POOL tokens, $SCHED_ENTRIES kept entries" "$OUT/$name.leader.log"; then
    log "$name: the leader's pool plan is not the pinned one (POOL_ROWS=$SCHED_POOL KEPT_ENTRIES=$SCHED_ENTRIES): every cut must hold the same plan"
    rc=3
  fi
  return $rc
}

# the leader is gone: a stopped follower is let go (it sees the closed link and leaves), the others have left or leave
# within a minute; whatever is still there after that is killed. Their exit codes go to FRC.
reap() {
  local t=0 p
  for p in "${PIDS[@]:-}"; do [[ -z "$p" ]] || kill -CONT "$p" 2>/dev/null || true; done
  for p in "${PIDS[@]:-}"; do
    [[ -n "$p" ]] || continue
    while kill -0 "$p" 2>/dev/null && (( t < 60 )); do sleep 1; t=$((t + 1)); done
    if kill -0 "$p" 2>/dev/null; then kill -9 "$p" 2>/dev/null; fi
    wait "$p" 2>/dev/null; FRC+=($?)
  done
  PIDS=()
}

judge() {   # judge NAME MODE REF GOT [compare flags...]: the last line of v2compare's output is IDENTICAL or DIFFERS
  local name=$1 cmp=$2 ref=$3 got=$4; shift 4
  python -m tfpp.glm.v2compare "$cmp" "$OUT/$ref" "$OUT/$got" "$@" > "$OUT/$name.compare.txt" 2>&1
  local rc=$?
  if (( rc == 0 )); then log "$name: PASS ($got equals $ref)"
  else log "$name: FAIL (exit $rc: $(tail -1 "$OUT/$name.compare.txt")); see $OUT/$name.compare.txt"; FAILS=$((FAILS + 1)); fi
}

judge_stages() {   # judge_stages NAME RUN N_STAGES: the leader's messages and idle pools on every follower
  local name=$1 run=$2 n=$3 i
  local files=("$OUT/$run.jsonl")     # a separate statement: in one 'local', $run expands before it is assigned
  for (( i = 0; i < n - 1; i++ )); do files+=("$OUT/$run.s$i.jsonl"); done
  python -m tfpp.glm.v2compare stages "${files[@]}" > "$OUT/$name.compare.txt" 2>&1
  local rc=$?
  if (( rc == 0 )); then log "$name: PASS (every stage applied the leader's messages)"
  else log "$name: FAIL (exit $rc: $(tail -1 "$OUT/$name.compare.txt")); see $OUT/$name.compare.txt"; FAILS=$((FAILS + 1)); fi
}

ref() {   # ref SCHEDULE: the one-process reference of a schedule (reused only when that run finished: its bench file
          # is written at the end, the trace from the start, so a killed reference is run again)
  [[ -f "$OUT/ref-$1.jsonl" && -f "$OUT/ref-$1.bench.json" ]] ||
    run_chain "ref-$1" "0:$L" "$1" || die "the one-process reference of $1 failed"
}

p1_ref() {   # the P=1 serial path (v1) over the same prompts: level A's reference
  [[ -f "$OUT/p1.bench.json" ]] && return 0
  python -m tfpp.glm.serve --role leader --layers "0:$L" --model "$MODEL_DIR" --truncate "$L" --context "$CONTEXT" \
    --prefill-rows 2048 --no-drafts --budget-gib "$BUDGET_GIB" --io-timeout 600 --bench "$PROMPTS" \
    --bench-out "$OUT/p1.bench.json" > "$OUT/p1.leader.log" 2>&1 || die "the P=1 serial reference failed ($OUT/p1.leader.log)"
}

t0a() { for s in $SCHEDULES; do ref "$s"; run_chain "pure-$s" "0:$L" "$s" --pure || die "pure failed"
        judge "T0a-v2[$s]" strict "ref-$s.jsonl" "pure-$s.jsonl"; done; }
t0b() { for s in $SCHEDULES; do ref "$s"; run_chain "eager-$s" "0:$L" "$s" --no-graphs || die "eager failed"
        judge "T0b-v2[$s]" strict "ref-$s.jsonl" "eager-$s.jsonl"; done; }
t1() {
  local k=0 sp s n
  for sp in "${SPLITS[@]}"; do
    k=$((k + 1)); n=$(wc -w <<< "$sp")
    for s in $SCHEDULES; do
      ref "$s"
      run_chain "chain$k-$s" "$sp" "$s" || { log "T1-v2 chain $k ($sp) $s: the leader failed"; FAILS=$((FAILS + 1)); continue; }
      judge "T1-v2-$k[$sp][$s]" strict "ref-$s.jsonl" "chain$k-$s.jsonl"
      judge_stages "T1-v2-$k[$sp][$s]-stages" "chain$k-$s" "$n"
    done
  done
}
t2m() {   # the smallest end-to-end check: T2m-solo and T2m-chain (one cut, the first schedule)
  local s=${SCHEDULES%% *} sp=${SPLITS[0]}
  run_chain "T2m-solo" "0:$L" "$s" || die "T2m-solo failed"
  run_chain "T2m-chain" "$sp" "$s" || { FAILS=$((FAILS + 1)); return; }
  judge "T2m[$s]" strict "T2m-solo.jsonl" "T2m-chain.jsonl"
  judge_stages "T2m[$s]-stages" "T2m-chain" "$(wc -w <<< "$sp")"
}
t1_p1() {
  p1_ref
  local sp=${SPLITS[0]}
  run_chain "p1chain" "$sp" AS1 || { FAILS=$((FAILS + 1)); return; }
  judge "T1-v2-p1[AS1 against P=1 serial]" bench "p1.bench.json" "p1chain.bench.json"
  ref AS1
  judge "T1-v2-p1[solo against P=1 serial]" bench "p1.bench.json" "ref-AS1.bench.json"
}
t1_free() {   # level A under real concurrency: the replies are the P=1 serial ones whatever the timing
  p1_ref
  local sp=${SPLITS[0]} f
  for f in FREE1 FREE2 FREE4; do
    TRACE_OFF=1 TFPP_GLM_TIMING=1 run_chain "$f" "$sp" "$f" || { log "T1-v2-free $f: the leader failed"; FAILS=$((FAILS + 1)); continue; }
    judge "T1-v2-free-$f[replies against P=1 serial]" bench "p1.bench.json" "$f.bench.json"
    log "$f: $(tr -d '[:space:]' < "$OUT/$f.bench.json.summary.json" 2>/dev/null)"
  done
}
t0e() {
  python -m tfpp.glm.v2check --role leader --layers "0:$L" "${COMMON[@]}" "${V2[@]}" --rounds "${ROUNDS:-20}" \
    --check-out "$OUT/T0e.json" > "$OUT/T0e.log" 2>&1
  local rc=$?
  tail -3 "$OUT/T0e.log"
  if (( rc == 0 )); then log "T0e: PASS (see $OUT/T0e.json)"; else log "T0e: FAIL (exit $rc; $OUT/T0e.log)"; FAILS=$((FAILS + 1)); fi
}

# a fault must end the whole chain by itself: the leader exits 75 and so does every follower, within reap's minute
# (137 = reap had to SIGKILL it: it never noticed the chain break); stage 0 of a self-kill case exits as it chose to.
# Then a clean run.
fault_case() {   # fault_case NAME [S0_RC]: LEADER_ENV / S0_ENV already set by the caller
  local name=$1 s0_rc=${2:-75} rc t0=$SECONDS i want bad=""
  run_chain "fault-$name" "${SPLITS[0]}" AS1; rc=$?
  local dt=$((SECONDS - t0)) nf=$(( $(wc -w <<< "${SPLITS[0]}") - 1 ))
  (( ${#FRC[@]} == nf )) || bad+=" ${#FRC[@]} follower exit codes for $nf followers;"
  for i in "${!FRC[@]}"; do
    want=75; (( i == 0 )) && want=$s0_rc
    (( FRC[i] == want )) || bad+=" s$i exited ${FRC[i]} (expected $want; 137: reap had to SIGKILL it);"
  done
  for i in "${RUN_PIDS[@]:-}"; do [[ -z "$i" ]] || ! kill -0 "$i" 2>/dev/null || bad+=" pid $i still runs;"; done
  if (( rc == 75 )) && [[ -z "$bad" ]]; then
    log "T1-v2-fault $name: PASS (the leader exited 75 after ${dt}s, followers ${FRC[*]:-none}, nothing left running)"
  else
    log "T1-v2-fault $name: FAIL (the leader exited $rc, expected 75; followers ${FRC[*]:-none};$bad $OUT/fault-$name.leader.log)"
    FAILS=$((FAILS + 1))
  fi
  # restarts cleanly: a fresh chain, the same schedule, bit-equal to the reference
  ref AS1
  LEADER_ENV="" S0_ENV="" IO_TIMEOUT="" TF_GLM_MULTI_WATCHDOG_S="" run_chain "after-$name" "${SPLITS[0]}" AS1 || { FAILS=$((FAILS + 1)); return; }
  judge "T1-v2-fault-$name-clean-restart" strict "ref-AS1.jsonl" "after-$name.jsonl"
}
t1_fault() {
  local pt
  for pt in round.windows:3 round.act:3 round.keeps:3 fill.sent:2 admit.after:2; do
    LEADER_ENV="TFPP_GLM_FAULT_V2=$pt" S0_ENV="" fault_case "v2-${pt//[:.]/-}"
  done
  # stage 0 SIGKILLs itself at its 40th WINDOWS (its exit: 137, its own SIGKILL; it says so in its log)
  LEADER_ENV="" S0_ENV="TFPP_GLM_FAULT_FOLLOWER=windows:40" fault_case "follower-windows-40" 137
  grep -q "injected fault windows:40: SIGKILL" "$OUT/fault-follower-windows-40.s0.log" ||
    { log "T1-v2-fault follower-windows-40: FAIL (stage 0 never killed itself: its 137 was reap's)"; FAILS=$((FAILS + 1)); }
  # stage 0 stops itself (SIGSTOP) at its 40th WINDOWS, i.e. mid-schedule, not at a guessed time: the watchdog prints
  # the stacks (TF_GLM_MULTI_WATCHDOG_S=10), the io timeout ends the leader
  LEADER_ENV="" S0_ENV="TFPP_GLM_FAULT_FOLLOWER=windows:40:stop" IO_TIMEOUT=30 TF_GLM_MULTI_WATCHDOG_S=10 fault_case "follower-stopped"
  grep -q "Timeout (0:00:10)" "$OUT/fault-follower-stopped.leader.log" && log "T1-v2-fault follower-stopped: the watchdog dumped the stacks" ||
    log "T1-v2-fault follower-stopped: no watchdog dump in the leader's log"
  grep -q "SIGSTOP" "$OUT/fault-follower-stopped.s0.log" || log "T1-v2-fault follower-stopped: stage 0 never stopped itself (the schedule has fewer than 40 WINDOWS): the case proved nothing"
}

# the HTTP side: chain up, curls, concurrency, a client leaving; the chain stays up for all of it
http_chain_up() {   # http_chain_up PREFIX -> LP (the leader's pid), HP (its port)
  local pre=$1 s=${SPLITS[-1]} ranges i n
  ranges=($s); n=${#ranges[@]}; HP=${HTTP_PORT:-8890}
  PIDS=()
  for (( i = 0; i < n - 1; i++ )); do
    local args=(--role follower --model "$MODEL_DIR" --layers "${ranges[$i]}" --budget-gib "$BUDGET_GIB"
                --down "127.0.0.1:$((PORT0 + i + 1))")
    (( i == 0 )) || args+=(--listen "127.0.0.1:$((PORT0 + i))")
    python -m tfpp.glm.serve "${args[@]}" > "$OUT/$pre.s$i.log" 2>&1 &
    PIDS+=($!)
  done
  python -m tfpp.glm.serve --role leader --layers "${ranges[$((n - 1))]}" --listen "127.0.0.1:$((PORT0 + n - 1))" \
    "${COMMON[@]}" "${V2[@]}" --host 127.0.0.1 --port "$HP" --name glm-trunc > "$OUT/$pre.leader.log" 2>&1 &
  LP=$!
  PIDS+=($LP)
  local t=0
  until curl -sf "http://127.0.0.1:$HP/v1/models" >/dev/null 2>&1; do
    sleep 3; t=$((t + 3))
    if (( t >= 900 )) || ! kill -0 $LP 2>/dev/null; then log "$pre: the leader did not come up"; return 1; fi
  done
}
t1_http() {
  http_chain_up http || { FAILS=$((FAILS + 1)); cleanup; return; }
  local U="http://127.0.0.1:$HP/v1/chat/completions" H='Content-Type: application/json' ok=0
  curl -sf "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":24,"temperature":0,"messages":[{"role":"user","content":"hi"}]}' > "$OUT/http.plain.json" && ok=$((ok + 1))
  curl -sfN "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":24,"temperature":0,"stream":true,"messages":[{"role":"user","content":"hi"}]}' > "$OUT/http.stream.txt" && ok=$((ok + 1))
  curl -sf "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":48,"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}}},"messages":[{"role":"user","content":"give a"}]}' > "$OUT/http.grammar.json" && ok=$((ok + 1))
  curl -sf "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":24,"temperature":0,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}],"messages":[{"role":"user","content":"call f"}]}' > "$OUT/http.tools.json" && ok=$((ok + 1))
  curl -sN --max-time 2 "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":2000,"temperature":0,"stream":true,"messages":[{"role":"user","content":"count"}]}' > /dev/null
  sleep 3
  curl -sf "$U" -H "$H" -d '{"model":"glm-trunc","max_tokens":8,"temperature":0,"messages":[{"role":"user","content":"after a disconnect"}]}' > "$OUT/http.after.json" && ok=$((ok + 1))
  log "T1-v2-http: $ok/5 requests answered (outputs in $OUT/http.*)"
  (( ok == 5 )) || FAILS=$((FAILS + 1))
  # eight at once, then one at a time: every request answered; the replies of greedy prompts are the same either way
  python -m tfpp.glm.v2bench http --url "http://127.0.0.1:$HP" --concurrency 8 --out "$OUT/http.c8.json" > "$OUT/http.c8.txt" 2>&1 &&
    log "T1-v2-http c8: $(tail -1 "$OUT/http.c8.txt")" || { log "T1-v2-http c8: FAIL ($OUT/http.c8.txt)"; FAILS=$((FAILS + 1)); }
  python -m tfpp.glm.v2bench http --url "http://127.0.0.1:$HP" --concurrency 1 --out "$OUT/http.c1.json" > "$OUT/http.c1.txt" 2>&1 &&
    log "T1-v2-http c1: $(tail -1 "$OUT/http.c1.txt")" || { log "T1-v2-http c1: FAIL ($OUT/http.c1.txt)"; FAILS=$((FAILS + 1)); }
  python - "$OUT/http.c8.json" "$OUT/http.c1.json" <<'PY' && log "T1-v2-http: greedy replies equal at concurrency 8 and 1" || { log "T1-v2-http: replies differ between concurrency 8 and 1"; FAILS=$((FAILS + 1)); }
import json, sys
a, b = (json.load(open(f))["results"] for f in sys.argv[1:3])
ref = {r["id"]: r.get("text_sha") for r in b if "error" not in r}
bad = [r["id"] for r in a if "error" not in r and ref.get(r["id"]) not in (None, r.get("text_sha")) and r["id"] not in ("sampled", "sampled-serial")]
sys.exit(1 if bad else 0)
PY
  # after the client left and everything ended: nothing live on the leader (health), and the chain idles without an error
  sleep 2
  local h; h=$(curl -sf "http://127.0.0.1:$HP/health" || true)
  log "T1-v2-http: /health after the runs: ${h:0:300}"
  if grep -qE '"decoding": ?[1-9]|"filling": ?[1-9]' <<< "$h"; then log "T1-v2-http: streams still live after the runs: FAIL"; FAILS=$((FAILS + 1)); fi
  kill -0 "$LP" 2>/dev/null || { log "T1-v2-http: the leader died during the runs"; FAILS=$((FAILS + 1)); }
  # a follower killed while the leader idles: the watch ends the leader within a few seconds (75), and a middle stage
  # (3+ stages: L=12/24) leaves by itself too (75)
  kill -9 "${PIDS[0]}" 2>/dev/null
  local t=0 lrc
  while kill -0 "$LP" 2>/dev/null && (( t < 30 )); do sleep 1; t=$((t + 1)); done
  if kill -0 "$LP" 2>/dev/null; then log "T1-v2-http idle kill: FAIL (the leader still runs ${t}s after stage 0 was killed)"; FAILS=$((FAILS + 1))
  else
    wait "$LP"; lrc=$?
    if (( lrc == 75 )); then log "T1-v2-http idle kill: PASS (the leader exited 75 after ${t}s)"
    else log "T1-v2-http idle kill: FAIL (the leader exited $lrc after ${t}s, expected 75)"; FAILS=$((FAILS + 1)); fi
  fi
  local i m mrc
  for (( i = 1; i < ${#PIDS[@]} - 1; i++ )); do
    m=${PIDS[$i]}; t=0
    while kill -0 "$m" 2>/dev/null && (( t < 30 )); do sleep 1; t=$((t + 1)); done
    if kill -0 "$m" 2>/dev/null; then log "T1-v2-http idle kill: FAIL (stage $i still runs)"; FAILS=$((FAILS + 1))
    else
      wait "$m"; mrc=$?
      (( mrc == 75 )) || { log "T1-v2-http idle kill: FAIL (stage $i exited $mrc, expected 75)"; FAILS=$((FAILS + 1)); }
    fi
  done
  cleanup
}

if [[ $L == 26 && $MODE != T0a-v2 && $MODE != T0b-v2 && $MODE != T0e && $MODE != ref-v2 ]]; then
  # 66.7 + 34.8 GiB of stages on one GPU (spec v2 10.2): the chain modes would run it out of memory
  die "L=26: $MODE runs its stages on this one GPU and 26 layers do not fit twice; run the cross-machine cuts with start_stage.sh (spec v2 16.3b), or use L=8/12/24 here"
fi
log "L=$L context=$CONTEXT P=$P pool=$POOL_ROWS entries=$KEPT_ENTRIES taps=$TAPS schedules=[$SCHEDULES] drafter=${DRAFTER_DIR:-none} budget=${BUDGET_GIB}GiB -> $OUT"
case $MODE in
  T0a-v2) t0a ;; T0b-v2) t0b ;; T1-v2) t1 ;; T2m) t2m ;; T1-v2-p1) t1_p1 ;; T1-v2-free) t1_free ;;
  T0e) t0e ;; T1-v2-fault) t1_fault ;; T1-v2-http) t1_http ;;
  ref-v2) for s in $SCHEDULES; do ref "$s"; done ;;
  all) t0a; t0b; t1; t1_p1; t1_free ;;
  *) die "mode: T0a-v2 | T0b-v2 | T1-v2 | T2m | T1-v2-p1 | T1-v2-free | T0e | T1-v2-fault | T1-v2-http | ref-v2 | all" ;;
esac
log "done: $FAILS failure(s); results in $OUT"
exit $(( FAILS > 0 ))
