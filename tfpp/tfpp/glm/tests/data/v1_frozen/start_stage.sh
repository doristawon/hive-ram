#!/usr/bin/env bash
# Start this N1X's tfpp-glm stage inside WSL: one environment on every stage (cluster.env), this node's values
# (node.env), checks before anything loads, then the stage (spec 12.5). Run it locally on each node (no SSH).
#   start_stage.sh                start in the background, wait until it is up (the leader: serving + a smoke test)
#   start_stage.sh --foreground   stay attached (systemd's ExecStart); exits with the stage's code (75: chain broke)
#   start_stage.sh --dry-run      run the checks and print the command; start nothing
# Env: TFPP_GLM_CONF (folder of cluster.env / node.env / toolchain.sh; default: this script's), TFPP_GLM_STATE
# (pid and logs; default ~/.local/state/tfpp-glm), WAIT_TIMEOUT (s, 3600).
# Order: any (a follower retries until the next stage answers, up to --connect-timeout; through relay_hop.ps1 too,
# which closes the upstream side while nothing listens behind it). Stop: the leader only.
# Node.env extras: GPU_GUARD_PORTS="P ..." (ports of other GPU servers on this node: refuse to start beside them),
# ALLOW_GPU_SHARING=1 (start beside another GPU workload anyway),
# VERIFY_SHARDS=1 (check this stage's shards against their Hub sha256 first; cached, ~1 min per 80 GB).
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CONF=${TFPP_GLM_CONF:-$HERE}
STATE=${TFPP_GLM_STATE:-$HOME/.local/state/tfpp-glm}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-3600}
FG=0; DRY=0
for a in "$@"; do
  case "$a" in
    --foreground) FG=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "start_stage.sh: unknown argument $a" >&2; exit 2 ;;
  esac
done
log() { printf '[tfpp-glm %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[tfpp-glm] WARN: %s\n' "$*" >&2; }
die() { printf '[tfpp-glm] ERROR: %s\n' "$*" >&2; exit 1; }

# ---- 1. one environment: nothing inherited, cluster.env read (never run), then node.env ---------------------------
# (Triton / cuBLAS / TF32 knobs too: they change a stage's numerics; HELLO's env digest compares them)
while IFS='=' read -r _n _; do unset "$_n"; done < <(env | grep -E '^(TF_GLM|TENSORFOLD|TF_ROCE|NCCL|TFPP|TRITON|CUBLAS|CUBLASLT|NVIDIA_TF32|TORCH_ALLOW_TF32)_[A-Za-z0-9_]*=' | grep -vE '^TFPP_GLM_(CONF|STATE)=' || true)
[[ -f "$CONF/cluster.env" ]] || die "no $CONF/cluster.env"
[[ -f "$CONF/node.env" ]] || die "no $CONF/node.env (copy node.env.example, keep this node's block)"
while IFS= read -r _line || [[ -n "$_line" ]]; do
  [[ "$_line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
  _k=${BASH_REMATCH[1]}; _v=${BASH_REMATCH[2]}
  _v=${_v%%#*}; _v=${_v%"${_v##*[![:space:]]}"}
  export "$_k=$_v"
done < "$CONF/cluster.env"
# shellcheck source=/dev/null
source "$CONF/node.env"
for _k in TF_GLM_CACHE_GIB TFPP_WARM_MS TFPP_GLM_VENV TFPP_GLM_TF_ROOT; do
  [[ -z "${!_k-}" ]] || export "$_k"
done
# shellcheck source=/dev/null
source "$CONF/toolchain.sh"
command -v python >/dev/null || die "no python in the venv ($VIRTUAL_ENV)"
command -v nvcc >/dev/null || die "no nvcc on PATH (pip's cuda-toolkit[nvcc,cccl]==13.0.* under $CUDA_HOME)"

# ---- 2. this node's role, layers, files and ports -------------------------------------------------------------------
case "${ROLE:-}" in leader|follower) ;; *) die "ROLE in node.env is leader or follower, not '${ROLE:-}'" ;; esac
[[ "${LAYERS:-}" =~ ^([0-9]+):([0-9]+)$ ]] || die "LAYERS in node.env is A:B, not '${LAYERS:-}'"
L0=${BASH_REMATCH[1]}; L1=${BASH_REMATCH[2]}
(( L0 < L1 && L1 <= 45 )) || die "LAYERS $LAYERS: 0 <= A < B <= 45"
END=${TRUNCATE:-45}                         # TRUNCATE=L (+ TAP_LAYERS): a shortened test model (T2)
[[ $ROLE != leader || $L1 == "$END" ]] || die "the leader's layers end at $END, not $L1"
[[ -z "${TAP_LAYERS:-}" || -n "${TRUNCATE:-}" ]] || die "TAP_LAYERS is for shortened test models (TRUNCATE=L)"
if [[ $ROLE == leader && -n "${BENCH:-}" ]]; then
  [[ -f "$BENCH" ]] || die "BENCH '$BENCH' is not a file"
  FG=1                                      # a bench run ends by itself (QUIT): stay attached, no HTTP
fi
[[ -d "${MODEL_DIR:-}" ]] || die "MODEL_DIR '${MODEL_DIR:-}' is not a folder"
need=(config.json model.safetensors.index.json tokenizer.json)
[[ $ROLE != leader ]] || need+=(tokenizer_config.json chat_template.jinja generation_config.json)
for f in "${need[@]}"; do [[ -f "$MODEL_DIR/$f" ]] || die "$MODEL_DIR/$f missing"; done
NO_DRAFTS=${NO_DRAFTS:-0}
if [[ $ROLE == leader && $NO_DRAFTS != 1 ]]; then
  [[ -f "${DRAFTER_DIR:-}/config.json" && -f "${DRAFTER_DIR:-}/model.safetensors" ]] ||
    die "DRAFTER_DIR '${DRAFTER_DIR:-}' needs DFlash2's config.json and model.safetensors (incoai@bf582e4e)"
fi
port_of() { echo "${1##*:}"; }
listening() { ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }
if (( L0 > 0 )); then
  [[ "${LISTEN:-}" =~ ^[0-9.]+:[0-9]+$ ]] || die "LISTEN in node.env is HOST:PORT (the stage before connects to it)"
  ! listening "$(port_of "$LISTEN")" || die "port $(port_of "$LISTEN") is already in use"
fi
if [[ $ROLE == follower ]]; then
  [[ "${DOWN:-}" =~ ^[0-9.]+:[0-9]+$ ]] || die "DOWN in node.env is HOST:PORT (the next stage's relay)"
fi
if [[ $ROLE == leader && -z "${BENCH:-}" ]]; then
  ! listening "$HTTP_PORT" ||
    die "port $HTTP_PORT is in use ($(ss -ltnp "sport = :$HTTP_PORT" 2>/dev/null | tail -n +2)); stop that server or choose another HTTP_PORT"
fi
# other GPU users: CAP measures free memory once, so a workload that grows later (one that loads its model on
# demand) runs a stage out of memory mid-request; stop it first (spec 12.5), or ALLOW_GPU_SHARING=1 knowingly.
# GPU_GUARD_PORTS (node.env; default none): the ports such servers listen on, on this node
for p in ${GPU_GUARD_PORTS:-}; do
  [[ $p =~ ^[0-9]+$ ]] || die "GPU_GUARD_PORTS holds port numbers, not '$p'"
  [[ $ROLE == leader && -z "${BENCH:-}" && $p == "$HTTP_PORT" ]] && continue      # the leader's own port: above
  listening "$p" || continue
  msg="port $p is in use ($(ss -ltnp "sport = :$p" 2>/dev/null | tail -n +2 | head -n 1)): another server on this GPU may take its memory"
  if [[ ${ALLOW_GPU_SHARING:-0} == 1 ]]; then warn "$msg (ALLOW_GPU_SHARING=1)"; else die "$msg; stop it first, or set ALLOW_GPU_SHARING=1 in node.env"; fi
done
# the pid file names this role's stage only while that pid still runs tfpp.glm.serve --role ROLE: after a crash or a
# WSL restart the number may belong to another process, which must neither pass for the stage nor be signalled
PIDF=$STATE/$ROLE.pid
is_stage() { [[ $1 =~ ^[0-9]+$ ]] && [[ "$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null)" == *"-m tfpp.glm.serve --role $ROLE "* ]]; }
stage_pid() { local p; p=$(cat "$PIDF" 2>/dev/null) || return 0; if is_stage "$p"; then echo "$p"; fi; return 0; }
# another tfpp-glm stage on this GPU (T2: s1 and the leader on one node): each measures the same free memory at CAP
_self=$(stage_pid)
_others=$(pgrep -f 'tfpp[.]glm[.]serve' 2>/dev/null | grep -vxF "${_self:-none}" | tr -s '[:space:]' ' ' || true)
if [[ -n "${_others// /}" && -z "${BUDGET_GIB:-}" ]]; then
  die "another tfpp-glm stage runs on this node (pid ${_others% }): stages sharing one GPU need BUDGET_GIB each (node.env)"
fi

# ---- 3. the stack: GPU, torch, Triton, the package tree, this stage's shards ----------------------------------------
IS_FIRST=$(( L0 == 0 ? 1 : 0 )); IS_LAST=$([[ $ROLE == leader ]] && echo 1 || echo 0)
pre=$(python - "$L0" "$L1" "$MODEL_DIR" "$IS_FIRST" "$IS_LAST" "${VISION:-0}" "$NO_DRAFTS" "${TFPP_GLM_TF_ROOT:-}" "${VERIFY_SHARDS:-0}" "$STATE" <<'PY'
import json, os, sys
l0, l1, model, first, last, vision, nodrafts, root = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4] == "1", sys.argv[5] == "1", sys.argv[6] == "1", sys.argv[7] == "1", sys.argv[8]
verify, state = sys.argv[9] == "1", sys.argv[10]
import torch, triton
from tfpp.glm import pins
from tfpp.glm.env import model_revision
from tfpp.glm.load import needed_shards, verify_shards
pkg = str(pins.installed_root())
try:
    sha = pins.check_tree()
    sha_ok = True
except pins.TreeMismatch as exc:
    sha, sha_ok = str(exc), False
mtp = last and os.environ.get("TF_GLM_MTP", "1").strip().lower() == "1"
files, missing, unknown = needed_shards(model, l0, l1, is_first=first, is_last=last, mtp=mtp,
                                        drafter=last and not nodrafts, vision=vision)
free, total = torch.cuda.mem_get_info()
props = torch.cuda.get_device_properties(0)
try:                                      # a build without this GPU's kernels fails here, not at HELLO
    kernel_ok, kernel_err = bool((torch.ones(4, device="cuda") * 2).sum().item() == 8), ""
except Exception as exc:  # noqa: BLE001
    kernel_ok, kernel_err = False, str(exc)[:200]
bad, unchecked = [], []
if verify and not missing:
    bad, unchecked = verify_shards(model, files, os.path.join(state, "shard-sha256.json"))
print(json.dumps({"pkg": pkg, "sha_ok": sha_ok, "sha": sha[:200], "torch": torch.__version__,
                  "cuda": torch.version.cuda or "", "kernel_ok": kernel_ok, "kernel_err": kernel_err,
                  "triton": triton.__version__, "cap": list(torch.cuda.get_device_capability(0)),
                  "integrated": bool(getattr(props, "is_integrated", False)), "free_gib": round(free / 2**30, 2),
                  "total_gib": round(total / 2**30, 2), "shards": len(files), "missing": missing,
                  "unknown": unknown[:5], "revision": model_revision(model), "bad": bad, "unchecked": unchecked[:5],
                  "venv": os.environ.get("VIRTUAL_ENV", ""), "root": root}))
PY
) || die "the preflight could not import torch / triton / tfpp.glm in $VIRTUAL_ENV"
log "preflight: $pre"
field() { python -c 'import json,sys; v=json.loads(sys.argv[1])[sys.argv[2]]; print(v if not isinstance(v, list) else " ".join(map(str, v)))' "$pre" "$1"; }
pkg=$(field pkg)
if [[ -n "${TFPP_GLM_TF_ROOT:-}" ]]; then
  [[ "$pkg" == "$(readlink -f "$TFPP_GLM_TF_ROOT")"/* ]] || die "tensorfold is imported from $pkg, not TFPP_GLM_TF_ROOT"
else
  [[ "$pkg" == "$(readlink -f "$VIRTUAL_ENV")"/* || "$pkg" == "$VIRTUAL_ENV"/* ]] ||
    die "tensorfold is imported from $pkg, not this venv (tf-venv-061's stock 0.6.1?)"
fi
[[ "$(field sha_ok)" == True ]] || die "$(field sha)"
[[ "$(field cap)" == "12 1" ]] || die "GPU capability $(field cap), expected 12 1 (N1X / GB10)"
case "$(field torch)" in 2.13.0*) ;; *) warn "torch $(field torch): tfpp-glm was set up with 2.13.0" ;; esac
[[ "$(field cuda)" == 13.* ]] ||
  die "torch $(field torch) is built for CUDA '$(field cuda)': the JIT builds use pip's nvcc 13.0 and sm_121 needs a cu13x build (toolchain.sh: --index-url https://download.pytorch.org/whl/cu130)"
[[ "$(field kernel_ok)" == True ]] || die "a CUDA kernel did not run on this GPU: $(field kernel_err)"
case "$(field triton)" in 3.7|3.7.*) ;; *) die "Triton $(field triton): MIA's CUDA ports need 3.7.x (G/sparse.py:135-142)" ;; esac
[[ -z "$(field missing)" ]] || die "this stage's shards are missing from $MODEL_DIR: $(field missing)"
[[ -z "$(field unknown)" ]] || die "checkpoint tensors the stage rules do not know: $(field unknown)"
[[ -z "$(field bad)" ]] || die "shards whose sha256 is not the Hub's ($MODEL_DIR/.cache/huggingface/download): $(field bad)"
[[ ${VERIFY_SHARDS:-0} != 1 || -z "$(field unchecked)" ]] || warn "no Hub sha256 to check these shards against: $(field unchecked)"
# (|| true: a first start has no extensions folder yet, and find's exit 1 would end the script here without a word)
old_rt=$(find "$TORCH_EXTENSIONS_DIR" -name '*.so' -exec sh -c 'readelf -d "$1" 2>/dev/null | grep -q "libcudart\.so\.1[0-2]\]" && echo "$1"' _ {} \; 2>/dev/null | head -n 3 | tr '\n' ' ' || true)
[[ -z "$old_rt" ]] || die "extensions linked against a CUDA 12 runtime beside torch's 13: $old_rt(toolchain.sh's libcudart.so link; move $TORCH_EXTENSIONS_DIR aside and rebuild)"
[[ "$(field revision)" != unknown ]] || warn "the checkpoint's Hub revision is unknown (no .cache/huggingface metadata): HELLO then only knows that every stage says unknown"

# ---- 4. the command ------------------------------------------------------------------------------------------------
args=(--role "$ROLE" --model "$MODEL_DIR" --layers "$LAYERS" --reserve-gib "${RESERVE_GIB:-3}")
[[ -z "${BUDGET_GIB:-}" ]] || args+=(--budget-gib "$BUDGET_GIB")
(( L0 == 0 )) || args+=(--listen "$LISTEN")
[[ -z "${TRACE:-}" ]] || args+=(--trace "$TRACE")
if [[ $ROLE == follower ]]; then
  args+=(--down "$DOWN")
  [[ ${PREFETCH_NEXT:-1} == 1 ]] || args+=(--no-prefetch-next)
else
  args+=(--context "$CONTEXT" --parallel "$PARALLEL" --prefill-rows "$TF_GLM_PREFILL_ROWS" --name "$SERVED_NAME"
         --host "$HTTP_HOST" --port "$HTTP_PORT" --max-tokens "$MAX_TOKENS")
  if [[ $THINKING == 1 ]]; then args+=(--thinking); else args+=(--no-thinking); fi
  [[ ${VISION:-0} != 1 ]] || args+=(--vision)
  [[ ${VISION:-0} != 1 || ${VISION_URLS:-0} != 1 ]] || args+=(--vision-urls)
  if [[ $NO_DRAFTS == 1 ]]; then args+=(--no-drafts); else args+=(--drafter "$DRAFTER_DIR"); fi
  [[ -z "${TRUNCATE:-}" ]] || args+=(--truncate "$TRUNCATE")
  [[ -z "${TAP_LAYERS:-}" ]] || args+=(--tap-layers "$TAP_LAYERS")
  [[ -z "${BENCH:-}" ]] || args+=(--bench "$BENCH" --bench-out "${BENCH_OUT:-$STATE/bench-$(date +%Y%m%d-%H%M%S).json}")
fi
CMD=(python -m tfpp.glm.serve "${args[@]}")
if (( DRY )); then printf '[dry-run] '; printf '%q ' "${CMD[@]}"; echo; exit 0; fi

mkdir -p "$STATE/logs" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
LOGF=$STATE/$ROLE.log
_running=$(stage_pid)
if [[ -n "$_running" ]]; then
  log "the $ROLE stage is already running (pid $_running); stop_stage.sh first"; exit 0
fi
rm -f "$PIDF"                                # stale: its number belongs to another process now, or to none
[[ ! -f "$LOGF" ]] || gzip -c "$LOGF" > "$STATE/logs/$(date +%Y%m%d-%H%M%S)-$ROLE.log.gz"
ulimit -s 65536 2>/dev/null || true          # the stack limit of MIA's reference containers (--ulimit stack=67108864)
if (( FG )); then echo $$ > "$PIDF"; exec "${CMD[@]}"; fi
setsid nohup "${CMD[@]}" > "$LOGF" 2>&1 < /dev/null &
echo $! > "$PIDF"
log "$ROLE stage (layers $LAYERS) started, pid $(cat "$PIDF"), log $LOGF"

# ---- 5. up: a follower once it says ready (the leader's AGREE reached it); the leader once it serves ---------------
t0=$SECONDS
until { [[ $ROLE == leader ]] && curl -sf --max-time 5 "http://127.0.0.1:$HTTP_PORT/v1/models" >/dev/null 2>&1; } ||
      { [[ $ROLE == follower ]] && grep -q ': ready (window' "$LOGF" 2>/dev/null; }; do
  kill -0 "$(cat "$PIDF")" 2>/dev/null || { tail -n 30 "$LOGF" >&2; die "the $ROLE stage exited before it was up"; }
  (( SECONDS - t0 < WAIT_TIMEOUT )) || die "not up after ${WAIT_TIMEOUT}s; still running: tail -f $LOGF"
  sleep 3
done
log "$ROLE up after $((SECONDS - t0))s"
[[ $ROLE == leader ]] || exit 0

# ---- 6. the leader's smoke test through every stage (thinking off, greedy) -----------------------------------------
URL="http://127.0.0.1:$HTTP_PORT"
SERVED=$(curl -s --max-time 5 "$URL/v1/models" | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
body=$(printf '{"model": "%s", "max_tokens": 32, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}, "messages": [{"role": "user", "content": "Reply with OK."}]}' "$SERVED")
if smoke=$(curl -s --max-time 300 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body" |
           python -c 'import json,sys; r = json.load(sys.stdin); c = (r["choices"][0]["message"].get("content") or "").strip(); assert c and (len(c) < 8 or len(set(c)) > 2); print(repr(c[:40]), r["usage"]["completion_tokens"], "tokens")'); then
  log "smoke test OK: $smoke"
else
  die "the smoke test failed (no reply text, or one character repeated); the stages are still running: tail $LOGF"
fi
