#!/usr/bin/env bash
# Start this N1X's tfpp-ds41 stage inside WSL: one environment on every stage (cluster.env), this node's values
# (node.env), checks before anything loads, then the stage (spec 14.2). Run it locally on each node (no SSH).
#   start_stage.sh                start in the background, wait until it is up (the leader: serving + a smoke test)
#   start_stage.sh --foreground   stay attached (systemd's ExecStart); exits with the stage's code (75: chain broke)
#   start_stage.sh --dry-run      run the checks and print the command; start nothing
# Env: TFPP_DS41_CONF (folder of cluster.env / node.env / toolchain.sh; default: this script's), TFPP_DS41_STATE
# (the state root, default ~/.local/state/tfpp-ds41: each stage keeps its pid, log and scratch files in
# <root>/<role>-<A>-<B>, so two stages on one node never share them), WAIT_TIMEOUT (s, 3600).
# Order: any (a follower retries until the next stage answers LINK, through relay_hop.ps1 too). Stop: the leader only.
# node.env extras: GPU_GUARD_PORTS (the ports of other GPU servers on this node, space-separated: the start is refused
# while one listens; default 18891, a tfpp-ds41 leader's HTTP), ALLOW_GPU_SHARING=1 (run beside such a server or
# another stage; BUDGET_GIB required then), VERIFY_SHARDS=1 (this stage's shards against their Hub sha256 first; cached).
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CONF=${TFPP_DS41_CONF:-$HERE}
STATE_ROOT=${TFPP_DS41_STATE:-$HOME/.local/state/tfpp-ds41}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-3600}
FG=0; DRY=0
for a in "$@"; do
  case "$a" in
    --foreground) FG=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "start_stage.sh: unknown argument $a" >&2; exit 2 ;;
  esac
done
log() { printf '[tfpp-ds41 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[tfpp-ds41] WARN: %s\n' "$*" >&2; }
die() { printf '[tfpp-ds41] ERROR: %s\n' "$*" >&2; exit 1; }

# ---- 1. one environment: nothing inherited, cluster.env read (never run), then node.env, then the toolchain ---------
while IFS='=' read -r _n _; do unset "$_n"; done < <(env | grep -E '^(TF_DS|TF_EXL3|TF_TP|TF_SEED|TF_RDMA|TF_COMM|TF_NCCL|TENSORFOLD|NCCL|TFPP|TRITON|CUBLAS|CUBLASLT|NVIDIA_TF32|TORCH_ALLOW_TF32)[A-Za-z0-9_]*=' | grep -vE '^(TFPP_DS41_(CONF|STATE)|TFPP_DIR)=' || true)
# this call's own serve values beat cluster.env's (lesson 4): saved here, put back after it is read
declare -A _caller=()
for _k in CONTEXT PARALLEL MTP_DRAFTS HTTP_PORT HTTP_HOST MAX_TOKENS THINKING TEMPERATURE SERVED_NAME NO_DRAFTS; do
  [[ -z "${!_k+x}" ]] || _caller[$_k]=${!_k}
done
[[ -f "$CONF/cluster.env" ]] || die "no $CONF/cluster.env"
[[ -f "$CONF/node.env" ]] || die "no $CONF/node.env (copy node.env.example, keep this node's block)"
while IFS= read -r _line || [[ -n "$_line" ]]; do
  [[ "$_line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
  _k=${BASH_REMATCH[1]}; _v=${BASH_REMATCH[2]}
  _v=${_v%%#*}; _v=${_v%"${_v##*[![:space:]]}"}
  export "$_k=$_v"
done < "$CONF/cluster.env"
for _k in "${!_caller[@]}"; do
  [[ "${!_k-}" == "${_caller[$_k]}" ]] || log "$_k=${_caller[$_k]} (this call) kept over cluster.env's ${!_k-unset}"
  export "$_k=${_caller[$_k]}"
done
# shellcheck source=/dev/null
source "$CONF/node.env"
for _k in TFPP_WARM_MS TFPP_DS41_VENV TFPP_DS41_TF_ROOT TF_API_KEY_FILE; do
  [[ -z "${!_k-}" ]] || export "$_k"
done
# every other TFPP_* / TF_DS_* / TF_EXL3_* / TENSORFOLD_* line of node.env (TFPP_DS41_WORKSPACE_GIB, TFPP_FAULT,
# TFPP_DS41_ALLOW_TRITON, the per-node Engram I/O knobs ...) reaches the stage too; the digested ones are compared in HELLO
while IFS= read -r _line || [[ -n "$_line" ]]; do
  [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?((TFPP_|TF_DS_|TF_EXL3_|TENSORFOLD_)[A-Za-z0-9_]*)= ]] || continue
  export "${BASH_REMATCH[2]}"
done < "$CONF/node.env"
[[ -z "${TOKEN_MAP:-}" ]] || export TF_DS_TOKEN_MAP=$TOKEN_MAP
[[ -z "${EXPERTS:-}" ]] || export TFPP_DS41_EXPERTS=$EXPERTS
# shellcheck source=/dev/null
source "$CONF/toolchain.sh"
command -v python >/dev/null || die "no python in the venv ($VIRTUAL_ENV)"
command -v nvcc >/dev/null || die "no nvcc on PATH (pip's cuda-toolkit[nvcc,cccl]==13.0.* under $CUDA_HOME)"

# ---- 2. this node's role, layers, files and ports -------------------------------------------------------------------
case "${ROLE:-}" in leader|follower) ;; *) die "ROLE in node.env is leader or follower, not '${ROLE:-}'" ;; esac
[[ "${LAYERS:-}" =~ ^([0-9]+):([0-9]+)$ ]] || die "LAYERS in node.env is A:B, not '${LAYERS:-}'"
L0=${BASH_REMATCH[1]}; L1=${BASH_REMATCH[2]}
# this stage's own folder (stop_stage.sh derives the same): stages that share a node (T1-3, a 4-stage chain on two
# nodes, systemd units started together at boot) never race on a pid file, a log, a preflight or a probe's output
STATE=$STATE_ROOT/$ROLE-$L0-$L1
[[ -d "${MODEL_DIR:-}" ]] || die "MODEL_DIR '${MODEL_DIR:-}' is not a folder"
NL=$(python -c 'import json,sys; c=json.load(open(sys.argv[1])); print((c.get("text_config") or c)["num_hidden_layers"])' "$MODEL_DIR/config.json") ||
  die "$MODEL_DIR/config.json unreadable"
(( L0 < L1 && L1 <= NL )) || die "LAYERS $LAYERS: 0 <= A < B <= $NL"
[[ $ROLE != leader || $L1 == "$NL" ]] || die "the leader's layers end at $NL (the head), not $L1"
if [[ $ROLE == leader && -n "${BENCH:-}" ]]; then
  [[ -f "$BENCH" ]] || die "BENCH '$BENCH' is not a file"
  FG=1                                      # a bench run ends by itself (QUIT): stay attached, no HTTP
fi
need=(config.json model.safetensors.index.json)
[[ $ROLE != leader ]] || need+=(tokenizer.json tokenizer_config.json)
[[ -z "${ENGRAM_DIR:-}" ]] || need+=(tokenizer.json)
for f in "${need[@]}"; do [[ -f "$MODEL_DIR/$f" ]] || die "$MODEL_DIR/$f missing"; done
[[ $ROLE != leader || -f "$MODEL_DIR/chat_template.jinja" ]] ||
  python -c 'import json,sys; assert json.load(open(sys.argv[1])).get("chat_template")' "$MODEL_DIR/tokenizer_config.json" 2>/dev/null ||
  die "the leader needs the chat template ($MODEL_DIR/chat_template.jinja or tokenizer_config.json's)"
port_of() { echo "${1##*:}"; }
listening() { ss -ltn "sport = :$1" > "$STATE/.ss" 2>/dev/null; grep -q LISTEN "$STATE/.ss"; }   # (no pipe into grep -q: lesson 5)
mkdir -p "$STATE/logs"
if (( L0 > 0 )); then
  [[ "${LISTEN:-}" =~ ^[0-9.]+:[0-9]+$ ]] || die "LISTEN in node.env is HOST:PORT (the stage before connects to it)"
  ! listening "$(port_of "$LISTEN")" || die "port $(port_of "$LISTEN") is already in use"
fi
if [[ $ROLE == follower ]]; then
  [[ "${DOWN:-}" =~ ^[0-9.]+:[0-9]+$ ]] || die "DOWN in node.env is HOST:PORT (the next stage's relay)"
fi
# other GPU users: CAP measures free memory once (lesson 7), so a workload that grows later runs a stage out of memory.
# GPU_GUARD_PORTS (node.env) lists the ports of the other GPU servers on this node; 18891 (a leader's HTTP) by default
SHARING=${ALLOW_GPU_SHARING:-0}
if [[ $ROLE == leader && -z "${BENCH:-}" ]]; then
  ! listening "${HTTP_PORT:-18891}" || die "port ${HTTP_PORT:-18891} (this leader's HTTP) is in use"
fi
for p in ${GPU_GUARD_PORTS:-18891}; do
  [[ $p =~ ^[0-9]+$ ]] || die "GPU_GUARD_PORTS in node.env: '$p' is not a port number"
  [[ $ROLE == leader && -z "${BENCH:-}" && $p == "${HTTP_PORT:-18891}" ]] && continue     # checked above
  listening "$p" || continue
  ss -ltnp "sport = :$p" > "$STATE/.ssp" 2>/dev/null || true
  msg="port $p is in use ($(sed -n 2p "$STATE/.ssp")): another server on this GPU shares its memory"
  if [[ $SHARING == 1 ]]; then
    [[ -n "${BUDGET_GIB:-}" ]] || die "$msg; ALLOW_GPU_SHARING=1 needs BUDGET_GIB in node.env"
    warn "$msg (ALLOW_GPU_SHARING=1, BUDGET_GIB=$BUDGET_GIB)"
  else
    die "$msg; stop it first, or set ALLOW_GPU_SHARING=1 and BUDGET_GIB in node.env"
  fi
done
_self=$(cat "$STATE/$ROLE.pid" 2>/dev/null || true)
pgrep -f 'tfpp[.](glm|ds41)[.](serve|run|ref)' > "$STATE/.pg" 2>/dev/null || true
_others=$(grep -vxF "${_self:-none}" "$STATE/.pg" | tr -s '[:space:]' ' ' || true)
if [[ -n "${_others// /}" ]]; then
  [[ $SHARING == 1 && -n "${BUDGET_GIB:-}" ]] ||
    die "another tfpp stage runs on this node (pid ${_others% }): stages sharing one GPU need ALLOW_GPU_SHARING=1 and BUDGET_GIB each"
fi

# ---- 3. the stack: GPU, torch, Triton, the fork's tree and seams, extensions, shards, Engram tables -----------------
IS_FIRST=$(( L0 == 0 ? 1 : 0 )); IS_LAST=$([[ $ROLE == leader ]] && echo 1 || echo 0)
NO_DRAFTS=${NO_DRAFTS:-0}
# (the shard hash cache stays in the state root, shared by this node's stages: keyed by path, inode, size and mtime, and
# a torn read only means hashing again)
pre=$(python - "$L0" "$L1" "$MODEL_DIR" "$IS_FIRST" "$IS_LAST" "$NO_DRAFTS" "${ENGRAM_DIR:-}" "${VERIFY_SHARDS:-0}" "$STATE_ROOT" <<'PY'
import json, os, sys
l0, l1, model, first, last, nodrafts = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4] == "1", sys.argv[5] == "1", sys.argv[6] == "1"
engram, verify, state = sys.argv[7], sys.argv[8] == "1", sys.argv[9]
import torch, triton
from tfpp.ds41 import pins
from tfpp.ds41.load import namer_for, stage_files
from tfpp.ds41.plan import ModelView
from tfpp.ds41.env import selection_revision
from tfpp.glm.env import model_revision
from tfpp.glm.load import verify_shards
pkg = str(pins.installed_root())
try:
    sha = pins.check_tree(); pins.check_seams(); sha_ok = True
except pins.TreeMismatch as exc:
    sha, sha_ok = str(exc), False
mv = ModelView.read(model)
index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
files, unknown = stage_files(model, index, range(l0, l1), first=first, last=last, drafter=last and not nodrafts,
                             rename=namer_for(model))        # an extracted embed.weight file stands in for shard 1
missing = [f for f in files if not os.path.isfile(os.path.join(model, f))]
owned = [e for e in mv.engram_layers if l0 <= e < l1]
eng_err, notes = [], []
if owned and not engram:
    eng_err.append(f"this stage owns Engram layers {owned}: ENGRAM_DIR in node.env")
if engram:
    from tfpp.ds41.testmodel import AIO_PROBE_ROW, engram_layer_of, table_identity
    tables = sorted(p for p in os.listdir(engram) if p.endswith(".safetensors")) if os.path.isdir(engram) else []
    if not tables:
        eng_err.append(f"ENGRAM_DIR {engram}: no *.safetensors")
    else:
        ident = table_identity([os.path.join(engram, t) for t in tables])
        if ident["layers"] != owned:
            eng_err.append(f"ENGRAM_DIR holds layers {ident['layers']}, this stage owns {owned}")
        for name, rows in ident["rows"].items():
            layer = engram_layer_of(name)
            if layer in mv.engram_layers and rows != mv.engram_rows[list(mv.engram_layers).index(layer)]:
                eng_err.append(f"{name}: {rows} rows, the config says {mv.engram_rows[list(mv.engram_layers).index(layer)]}")
            if rows <= AIO_PROBE_ROW:     # harmless (startup.stage_extra only notes it): the same bytes, slower reads
                notes.append(f"{name}: {rows} rows <= {AIO_PROBE_ROW}, the AIO self-test's row (F/model.py:557): "
                             "decode reads stay on the reader pool")
free, total = torch.cuda.mem_get_info()
try:
    kernel_ok, kernel_err = bool((torch.ones(4, device="cuda") * 2).sum().item() == 8), ""
except Exception as exc:  # noqa: BLE001
    kernel_ok, kernel_err = False, str(exc)[:200]
bad, unchecked = [], []
if verify and not missing:
    bad, unchecked = verify_shards(model, files, os.path.join(state, "shard-sha256.json"))
print(json.dumps({"pkg": pkg, "sha_ok": sha_ok, "sha": sha[:300], "torch": torch.__version__, "cuda": torch.version.cuda or "",
                  "kernel_ok": kernel_ok, "kernel_err": kernel_err, "triton": triton.__version__,
                  "cap": list(torch.cuda.get_device_capability(0)), "free_gib": round(free / 2**30, 2),
                  "total_gib": round(total / 2**30, 2), "shards": len(files), "missing": missing, "unknown": unknown[:5],
                  "engram": eng_err, "notes": notes, "revision": selection_revision(model) or model_revision(model),
                  "bad": bad, "unchecked": unchecked[:5]}))
PY
) || die "the preflight could not import torch / triton / tfpp.ds41 in $VIRTUAL_ENV"
log "preflight: $pre"
echo "$pre" > "$STATE/preflight.json"
field() { python -c 'import json,sys; v=json.load(open(sys.argv[1]))[sys.argv[2]]; print(v if not isinstance(v, list) else " | ".join(map(str, v)))' "$STATE/preflight.json" "$1"; }
pkg=$(field pkg)
[[ -n "${TFPP_DS41_TF_ROOT:-}" ]] || die "TFPP_DS41_TF_ROOT (the git-archived fork's src folder) is not set in node.env"
[[ "$pkg" == "$(readlink -f "$TFPP_DS41_TF_ROOT")"/* ]] || die "tensorfold is imported from $pkg, not TFPP_DS41_TF_ROOT"
[[ "$(field sha_ok)" == True ]] || die "$(field sha)"
[[ "$(field cap)" == "12 | 1" ]] || die "GPU capability $(field cap), expected 12.1 (N1X / GB10)"
case "$(field torch)" in 2.13.0*) ;; *) warn "torch $(field torch): tfpp-ds41 was set up with 2.13.0" ;; esac
[[ "$(field cuda)" == 13.* ]] || die "torch $(field torch) is built for CUDA '$(field cuda)': a cu130 build is needed (toolchain.sh)"
[[ "$(field kernel_ok)" == True ]] || die "a CUDA kernel did not run on this GPU: $(field kernel_err)"
case "$(field triton)" in 3.7|3.7.*) ;; *) [[ ${TFPP_DS41_ALLOW_TRITON:-0} == 1 ]] || die "Triton $(field triton): tfpp-ds41 pins 3.7.x" ;; esac
[[ -z "$(field missing)" ]] || die "this stage's shards are missing from $MODEL_DIR: $(field missing)"
[[ -z "$(field unknown)" ]] || die "checkpoint tensors the stage rules do not know: $(field unknown)"
[[ -z "$(field engram)" ]] || die "Engram tables: $(field engram)"
[[ -z "$(field notes)" ]] || warn "Engram tables: $(field notes)"
[[ -z "$(field bad)" ]] || die "shards whose sha256 is not the Hub's: $(field bad)"
[[ ${VERIFY_SHARDS:-0} != 1 || -z "$(field unchecked)" ]] || warn "no Hub sha256 to check these shards against: $(field unchecked)"
[[ "$(field revision)" != unknown ]] || warn "the checkpoint's Hub revision is unknown: HELLO then only knows that every stage says unknown"
if [[ -n "${ENGRAM_DIR:-}" ]]; then
  fs=$(stat -f -c %T "$(readlink -f "$ENGRAM_DIR")" 2>/dev/null || echo unknown)
  case "$fs" in ext2/ext3|ext4) ;; *) die "ENGRAM_DIR is on $fs: the tables must live on the WSL ext4 disk (O_DIRECT / AIO reads)" ;; esac
fi
# extensions built against CUDA 12's runtime (lesson 2): refuse; prebuild builds and checks them all
find "$TORCH_EXTENSIONS_DIR" -name '*.so' > "$STATE/.sos" 2>/dev/null || true
old_rt=""
while IFS= read -r so; do
  readelf -d "$so" > "$STATE/.elf" 2>/dev/null || continue
  grep -q 'libcudart\.so\.1[0-2]' "$STATE/.elf" && old_rt+="$so "
done < "$STATE/.sos"
[[ -z "$old_rt" ]] || die "extensions linked against a CUDA 12 runtime: $old_rt(toolchain.sh's libcudart.so link; move $TORCH_EXTENSIONS_DIR aside, then python -m tfpp.ds41.prebuild)"

# ---- 4. the command ------------------------------------------------------------------------------------------------
args=(--role "$ROLE" --model "$MODEL_DIR" --layers "$LAYERS" --reserve-gib "${RESERVE_GIB:-3}")
[[ -z "${BUDGET_GIB:-}" ]] || args+=(--budget-gib "$BUDGET_GIB")
(( L0 == 0 )) || args+=(--listen "$LISTEN")
[[ -z "${ENGRAM_DIR:-}" ]] || args+=(--engram "$ENGRAM_DIR")
[[ -z "${TRACE:-}" ]] || args+=(--trace "$TRACE")
if [[ $ROLE == follower ]]; then
  args+=(--down "$DOWN")
else
  args+=(--context "$CONTEXT" --parallel "$PARALLEL" --name "$SERVED_NAME" --host "$HTTP_HOST" --port "$HTTP_PORT"
         --max-tokens "$MAX_TOKENS" --temperature "${TEMPERATURE:-0}")
  if [[ ${THINKING:-1} == 1 ]]; then args+=(--thinking); else args+=(--no-thinking); fi
  if [[ $NO_DRAFTS == 1 ]]; then args+=(--no-drafts); else args+=(--mtp-drafts "${MTP_DRAFTS:-5}"); fi
  [[ -z "${BENCH:-}" ]] || args+=(--bench "$BENCH" --bench-out "${BENCH_OUT:-$STATE/bench-$(date +%Y%m%d-%H%M%S).json}")
  [[ -z "${ORACLE:-}" ]] || args+=(--oracle "$ORACLE")
fi
[[ -z "${TRACE:-}" ]] || mkdir -p "$(dirname "$TRACE")"
CMD=(python -m tfpp.ds41.serve "${args[@]}")
if (( DRY )); then printf '[dry-run] '; printf '%q ' "${CMD[@]}"; echo; exit 0; fi

mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
PIDF=$STATE/$ROLE.pid
LOGF=$STATE/$ROLE.log
if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  log "the $ROLE stage is already running (pid $(cat "$PIDF")); stop_stage.sh first"; exit 0
fi
[[ ! -f "$LOGF" ]] || gzip -c "$LOGF" > "$STATE/logs/$(date +%Y%m%d-%H%M%S)-$ROLE.log.gz"
ulimit -s 65536 2>/dev/null || true          # a 64 MiB stack, as the systemd unit's LimitSTACK=67108864
if (( FG )); then echo $$ > "$PIDF"; exec "${CMD[@]}"; fi
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null &
echo $! > "$PIDF"
log "$ROLE stage (layers $LAYERS) started, pid $(cat "$PIDF"), log $LOGF"

# ---- 5. up: a follower once it says ready (the leader's AGREE reached it and it built); the leader once it serves ---
t0=$SECONDS
up() {
  # /health is the one route the fork's server leaves open when TF_API_KEY(_FILE) is set (cuda/http.py:84); /v1/models needs the key
  if [[ $ROLE == leader ]]; then curl -sf --max-time 5 "http://127.0.0.1:$HTTP_PORT/health" > /dev/null 2>&1
  else grep -q ': ready (pool' "$LOGF" 2>/dev/null; fi
}
until up; do
  kill -0 "$(cat "$PIDF")" 2>/dev/null || { tail -n 30 "$LOGF" >&2; die "the $ROLE stage exited before it was up"; }
  (( SECONDS - t0 < WAIT_TIMEOUT )) || die "not up after ${WAIT_TIMEOUT}s; still running: tail -f $LOGF"
  sleep 3
done
log "$ROLE up after $((SECONDS - t0))s"
[[ $ROLE == leader ]] || exit 0

# ---- 6. the leader's smoke test through every stage (thinking off, greedy) -----------------------------------------
URL="http://127.0.0.1:$HTTP_PORT"
KEYH=()
[[ -z "${TF_API_KEY_FILE:-}" ]] || KEYH=(-H "Authorization: Bearer $(cat "$TF_API_KEY_FILE")")
curl -s --max-time 5 "${KEYH[@]}" "$URL/v1/models" > "$STATE/.models" || true
SERVED=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"][0]["id"])' "$STATE/.models") || die "no /v1/models answer"
body=$(printf '{"model": "%s", "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}, "messages": [{"role": "user", "content": "Reply with OK."}]}' "$SERVED")
curl -s --max-time 300 "${KEYH[@]}" "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" > "$STATE/.smoke" || true
if smoke=$(python -c 'import json,sys; r = json.load(open(sys.argv[1])); c = (r["choices"][0]["message"].get("content") or "").strip(); assert c; print(repr(c[:40]), r["usage"]["completion_tokens"], "tokens")' "$STATE/.smoke"); then
  log "smoke test OK: $smoke"
else
  die "the smoke test failed (no reply text); the stages are still running: tail $LOGF"
fi
