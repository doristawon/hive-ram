# tfpp-ds41 toolchain inside WSL (sourced by start_stage.sh, run_shortened.sh, make_testmodel.sh): its own venv
# (tf-venv-061 and tf-venv-mia are never touched), pip's CUDA 13.0 nvcc with CCCL, Python headers for the JIT builds,
# kernel caches keyed by the fork's tree and the toolchain (spec 1.4, 13.3; lesson 2).
#
# One-time setup on each N1X (inside WSL):
#   python3.14 -m venv ~/tf-venv-dsv41 && . ~/tf-venv-dsv41/bin/activate
#   pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0"
#   pip install "triton==3.7.1" ninja "cuda-toolkit[nvcc,cccl]==13.0.*" numpy safetensors \
#       "tokenizers>=0.20" "jinja2>=3.1" "huggingface-hub>=0.24"           # (+ Pillow on the leader, v2)
#   mkdir -p ~/tf-dsv41
#   FORK_CHECKOUT=<a clone of github.com/bertholomus/TensorFold, branch deepseek-v41-tp2, holding commit
#                  af26d42b1f618e9e653903a6c043448b52729c64>
#   git -C "$FORK_CHECKOUT" -c core.autocrlf=false \
#       -c safe.directory='*' archive af26d42b1 src tools/dsv41 | tar -x -C ~/tf-dsv41
#   (a Windows work tree is CRLF: never copy it, git archive gives the LF bytes the digest pins)
#   TFPP_DS41_TF_ROOT=~/tf-dsv41/src (the folder holding tensorfold/; node.env sets it)
#   python -m tfpp.ds41.pins --check ~/tf-dsv41/src/tensorfold     # must say c64d3e0a...
#   python -m tfpp.ds41.prebuild                                                 # the four extensions, readelf
V=${TFPP_DS41_VENV:-$HOME/tf-venv-dsv41}
PY=${TFPP_DS41_PY:-python3.14}
[[ -x "$V/bin/python" ]] || { echo "[tfpp-ds41] no venv at $V (see toolchain.sh)" >&2; return 1 2>/dev/null || exit 1; }
case "$(basename "$(readlink -f "$V")")" in
  tf-venv-061|tf-venv-mia)
    echo "[tfpp-ds41] $V is $(basename "$(readlink -f "$V")") (Flash-Next's / tfpp-glm's venv, not touched): set TFPP_DS41_VENV to ~/tf-venv-dsv41" >&2
    return 1 2>/dev/null || exit 1 ;;
esac
export VIRTUAL_ENV=$V
export PATH=$V/bin:$V/lib/$PY/site-packages/nvidia/cu13/bin:$PATH
export CUDA_HOME=$V/lib/$PY/site-packages/nvidia/cu13
# pip's CUDA 13 ships libcudart.so.13 without the libcudart.so link -lcudart wants, and CCCL (nv/target) only with
# the [cccl] extra: without them ld and the preprocessor fall through to Ubuntu's CUDA 12 and the extensions link
# libcudart.so.12 beside torch's 13 (lesson 2). The link lives in the venv; LIBRARY_PATH puts it first.
[[ -e "$CUDA_HOME/lib/libcudart.so" || ! -e "$CUDA_HOME/lib/libcudart.so.13" ]] || ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
[[ -e "$CUDA_HOME/lib/libcudart.so" ]] ||
  { echo "[tfpp-ds41] no libcudart.so(.13) under $CUDA_HOME/lib (pip install 'cuda-toolkit[nvcc,cccl]==13.0.*' in $V)" >&2; return 1 2>/dev/null || exit 1; }
[[ -e "$CUDA_HOME/include/nv/target" ]] ||
  { echo "[tfpp-ds41] no CCCL headers in $CUDA_HOME/include (pip install 'cuda-toolkit[cccl]==13.0.*' in $V)" >&2; return 1 2>/dev/null || exit 1; }
export LIBRARY_PATH=$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
# no python3.14-dev package on the N1X: headers extracted under ~/pyinc (as tfpp-glm's toolchain)
export CPLUS_INCLUDE_PATH=$HOME/pyinc/usr/include:$HOME/pyinc/usr/include/$PY:$HOME/pyinc/usr/include/aarch64-linux-gnu/$PY
export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH
export TORCH_CUDA_ARCH_LIST=12.1
TFPP_DIR=${TFPP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}   # the checkout this file lies in
PP=$TFPP_DIR
[[ -z "${TFPP_DS41_TF_ROOT:-}" ]] || PP=$TFPP_DS41_TF_ROOT:$PP
export PYTHONPATH=$PP
# one kernel cache per fork tree and toolchain (torch, CUDA, Triton, Python, nvcc): a cache never outlives either
PKG_TAG=$("$V/bin/python" -c 'import sys; sys.path[:0] = sys.argv[1].split(":"); from tfpp.ds41 import pins; print(pins.EXPECT_PKG_SHA[:16])' "$PP" 2>/dev/null || echo dsv41-af26d42b1)
NVCC_V=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n 1)
TC_TAG=$("$V/bin/python" -c 'import platform, torch, triton; print("t" + torch.__version__.split("+")[0] + "-cu" + str(torch.version.cuda) + "-tr" + triton.__version__ + "-py" + platform.python_version())' 2>/dev/null || echo unknown)
export TORCH_EXTENSIONS_DIR=$HOME/.cache/tfpp-ds41/$PKG_TAG/$TC_TAG-nvcc${NVCC_V:-none}/ext
export TRITON_CACHE_DIR=$HOME/.cache/tfpp-ds41/$PKG_TAG/$TC_TAG-nvcc${NVCC_V:-none}/triton
export PYTHONUNBUFFERED=1
