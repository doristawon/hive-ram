#!/bin/bash
# prepare_tensorfold.sh TENSORFOLD_CHECKOUT DEST
# Builds the small directory a vLLM plugin imports TensorFold's EXL3 kernels from: a copy of
# src/tensorfold/cuda/exl3 of TensorFold v0.6.6 (Apache-2.0), the one patch in ./patches (clamped SwiGLU-OAI for the
# routed-expert epilogue) and a 15-line stand-in for tensorfold.cuda.build.load. hive-ram ships none of TensorFold's code.
#   git clone https://github.com/ashhart/TensorFold && git -C TensorFold checkout v0.6.6   # cb2ebf0540f42604e2759b2ddef497861e928248
set -euo pipefail
SRC=${1:?TensorFold checkout}; DEST=${2:?destination (this becomes TFMOE_ROOT)}
HERE="$(cd "$(dirname "$0")" && pwd)"
[ "$(git -C "$SRC" rev-parse HEAD)" = cb2ebf0540f42604e2759b2ddef497861e928248 ] || { echo "$SRC is not TensorFold v0.6.6 (cb2ebf0)"; exit 1; }
rm -rf "$DEST/tensorfold"; mkdir -p "$DEST/tensorfold/cuda/exl3"
: > "$DEST/tensorfold/__init__.py"; : > "$DEST/tensorfold/cuda/__init__.py"
for f in experts.py experts.cpp experts.cu experts_cb0.cu experts_cb1.cu experts_cb2.cu experts_grouped.cuh decode.cuh format.py linear.py linear.cpp linear.cu; do
  git -C "$SRC" show "v0.6.6:src/tensorfold/cuda/exl3/$f" > "$DEST/tensorfold/cuda/exl3/$f"
done
: > "$DEST/tensorfold/cuda/exl3/__init__.py"
cp "$HERE/tensorfold_build_shim.py" "$DEST/tensorfold/cuda/build.py"
( cd "$DEST" && mkdir -p _p/src/tensorfold/cuda && cp -r tensorfold/cuda/exl3 _p/src/tensorfold/cuda/ && cd _p && patch -p1 < "$HERE/patches/0001-exl3-experts-swiglu-oai.patch" \
  && cp src/tensorfold/cuda/exl3/experts.cu src/tensorfold/cuda/exl3/experts.py "$DEST/tensorfold/cuda/exl3/" ) && rm -rf "$DEST/_p"
cp "$SRC/LICENSE" "$DEST/LICENSE.tensorfold"
echo "ready: export TFMOE_ROOT=$DEST ; then pip install the plugin (tfpp/vllm_plugin) into vLLM's Python"
