#!/usr/bin/env bash
# Print (never run) the hf download script for the shortened test model S11 (spec 15.2): 11 shards (59.855 GiB)
# and the small files of Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw, NOT the original Engram shards.
# The operator saves it, sets the REV line (the repo's full commit) and runs it by hand, once, then copies the folder to
# the other node over USB4 (VERIFY_SHARDS=1 at start checks the Hub sha256 from the download metadata; copy the hidden
# .cache folder too, or give both nodes' make_testmodel.sh the same REVISION).
#   download_list.sh [--dir ~/models/ds41-src] [--index model.safetensors.index.json] [--revision FULL_COMMIT]
#                    [--vhdx-ok] [--no-disk-check] > get_s11.sh
# Disk check first (spec 15.3), nothing is printed when it fails: (1) the WSL disk under --dir has >= 75 GB free; (2) the
# WSL vhdx grows on C:, and a node's C: may be nearly full: C: (as /mnt/c) must have 75 GB
# too, unless --vhdx-ok says the operator knows the vhdx is already that large (its file size, `ls -l` of ext4.vhdx on
# Windows, is past the disk's used space plus 75 GB). When unsure, free space first (e.g. an old GGUF).
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
TFPP_DIR=${TFPP_DIR:-$(cd "$HERE/../.." && pwd)}
DIR=$HOME/models/ds41-src
args=()
CHECK=1; VHDX_OK=0
while (( $# )); do
  case "$1" in
    --index) args+=(--index "$2"); shift 2 ;;
    --revision) args+=(--revision "$2"); shift 2 ;;
    --dir) DIR=${2/#\~/$HOME}; shift 2 ;;
    --vhdx-ok) VHDX_OK=1; shift ;;
    --no-disk-check) CHECK=0; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "download_list.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
args+=(--dir "$DIR")
run() { PYTHONPATH=$TFPP_DIR${PYTHONPATH:+:$PYTHONPATH} python3 -m tfpp.ds41.testmodel downloads "$@"; }

free_gb() {   # free GB (1000^3) on the file system holding $1 (its nearest existing parent), "?" when unreadable
  local p=$1 k
  while [[ ! -e "$p" && "$p" != / ]]; do p=$(dirname "$p"); done
  k=$(df -Pk "$p" 2>/dev/null | awk 'NR == 2 {print $4}') || true
  [[ "$k" =~ ^[0-9]+$ ]] && echo $(( k * 1024 / 1000000000 )) || echo "?"
}

if (( CHECK )); then
  need=$(run --need-bytes); need_gb=$(( need / 1000000000 ))
  have=$(free_gb "$DIR")
  if [[ $have == "?" ]] || (( have < need_gb )); then
    echo "download_list.sh: $DIR has ${have} GB free on the WSL disk, the download wants ${need_gb} GB (S11 64.3 GB + the Mode D tables 4.1 GB + room): free space first" >&2
    exit 1
  fi
  if (( ! VHDX_OK )); then
    cfree=$(free_gb /mnt/c)
    if [[ $cfree == "?" ]] || (( cfree < need_gb )); then
      echo "download_list.sh: C: has ${cfree} GB free (/mnt/c), the WSL vhdx may need to grow by up to ${need_gb} GB to hold the download: free C: first, or --vhdx-ok once you know the vhdx file is already that large" >&2
      exit 1
    fi
  fi
  echo "# disk check: ${have} GB free on the WSL disk under $DIR; C: $( (( VHDX_OK )) && echo 'not checked (--vhdx-ok)' || echo "$(free_gb /mnt/c) GB free" ); need ${need_gb} GB"
fi
run "${args[@]}"
