#!/usr/bin/env bash
# Build the shortened test model S11 and its Mode D Engram tables on THIS node (spec 15.1, 15.4). Run it on each node:
# the tables are made, never copied (2.03 GB each, the same bytes everywhere: compare the printed sha256 across nodes).
#   SRC=~/models/ds41-src make_testmodel.sh [--zero]
# Env: SRC (the downloaded shards + small files, download_list.sh), OUT (default ~/models/ds41-s11), ENGRAM_OUT
#      (default ~/engram-d: L1/t1.safetensors, L4/t4.safetensors, both/ = links to the two), SEL (the selection),
#      ENGRAM_VOCAB (320000), REVISION (the source's full Hub commit: set it, the same on both nodes, when SRC was copied
#      without its hidden .cache/huggingface folder; tfpp_selection.json must be byte-identical across nodes). --zero: Mode Z tables too (all-zero holes, smoke tests only) under ${ENGRAM_OUT}-z.
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
CONF=${TFPP_DS41_CONF:-$HERE}
SRC=${SRC:?set SRC to the folder download_list.sh filled}
OUT=${OUT:-$HOME/models/ds41-s11}
EOUT=${ENGRAM_OUT:-$HOME/engram-d}
SEL=${SEL:-0,1,2,3,14,20,21,24,37,38,39}
ENGRAM_VOCAB=${ENGRAM_VOCAB:-320000}
ZERO=0
[[ "${1:-}" != --zero ]] || ZERO=1
log() { printf '[tfpp-ds41 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[tfpp-ds41] ERROR: %s\n' "$*" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONF/toolchain.sh"
[[ -f "$SRC/config.json" && -f "$SRC/model.safetensors.index.json" ]] || die "$SRC: config.json and the index missing"
mkdir -p "$EOUT"
fs=$(stat -f -c %T "$(readlink -f "$EOUT")" 2>/dev/null || echo unknown)
case "$fs" in ext2/ext3|ext4) ;; *) die "$EOUT is on $fs: the Engram tables must live on WSL's ext4 disk" ;; esac
avail=$(df -Pk "$EOUT" | awk 'NR == 2 {print $4}')
(( avail > 5 * 1024 * 1024 )) || die "less than 5 GB free under $EOUT for two 2.03 GB tables"

python -m tfpp.ds41.testmodel build "$SRC" "$OUT" --sel "$SEL" --engram-vocab "$ENGRAM_VOCAB" ${REVISION:+--revision "$REVISION"}
python -c 'import json,sys; t=json.load(open(sys.argv[1]))["text_config"]; print(" ".join(f"{l}:{r}" for l, r in zip(t["engram_layer_ids"], t["engram_num_embeddings"])))' "$OUT/config.json" > "$OUT/.engram_rows"
read -r -a ROWS < "$OUT/.engram_rows"
mkdir -p "$EOUT/both"
for lr in "${ROWS[@]}"; do
  layer=${lr%%:*}; rows=${lr##*:}
  mkdir -p "$EOUT/L$layer"
  python -m tfpp.ds41.testmodel table "$EOUT/L$layer/t$layer.safetensors" "$layer" "$rows" dense --seed "$layer" | tee -a "$EOUT/sha256.txt"
  ln -sfn "../L$layer/t$layer.safetensors" "$EOUT/both/t$layer.safetensors"
  if (( ZERO )); then
    mkdir -p "${EOUT}-z/L$layer" "${EOUT}-z/both"
    python -m tfpp.ds41.testmodel table "${EOUT}-z/L$layer/t$layer.safetensors" "$layer" "$rows" zero
    ln -sfn "../L$layer/t$layer.safetensors" "${EOUT}-z/both/t$layer.safetensors"
  fi
done
python -m tfpp.ds41.testmodel selftest "$OUT/.selftest" > "$OUT/.selftest.txt" && log "$(tail -n 1 "$OUT/.selftest.txt")"
log "S11 in $OUT; Mode D tables in $EOUT (L1, L4, both); their sha256 (compare with the other node's):"
tail -n "${#ROWS[@]}" "$EOUT/sha256.txt"
