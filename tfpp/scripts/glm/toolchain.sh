# tfpp-glm toolchain inside WSL (sourced by start_stage.sh and run_truncated.sh): the MIA package's own venv (tf-venv-061,
# TensorFold 0.6.1 for tfpp-qwen, is not touched), pip's CUDA 13.0 nvcc, Python headers for the JIT builds, and
# kernel caches keyed by the package tree (spec 12.4).
#
# One-time setup on each N1X (inside WSL; the leader also needs the vision / grammar packages):
#   python3.14 -m venv ~/tf-venv-mia && . ~/tf-venv-mia/bin/activate
#   pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.13.0"    # a CUDA 13 build (sm_121)
#   pip install "triton==3.7.1" ninja "cuda-toolkit[nvcc,cccl]==13.0.*" numpy safetensors \
#       "tokenizers>=0.20" "jinja2>=3.1" "huggingface-hub>=0.24"
#   python -c 'import torch; print(torch.__version__, torch.version.cuda, (torch.ones(1, device="cuda") + 1).item())'
#   (start_stage.sh's preflight refuses a torch not built for CUDA 13.x, or one without this GPU's kernels)
#   pip install "torchvision==0.28.0" Pillow "av==18.1.0" "xgrammar==0.2.8" "transformers>=5.3,<6"   # leader only
#   rsync -a --delete "$MIA_TREE"/tensorfold/ \
#       "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/tensorfold/"
#   python -m tfpp.glm.pins --check       # the tree digest tfpp-glm was written against
#   ($MIA_TREE: the MIA package, TensorFold v0.6.0 + the v1.11-pre patches; third-party, not part of this repository)
# Alternative to the rsync into the venv: TFPP_GLM_TF_ROOT=<folder holding tensorfold/> puts that copy first on
# PYTHONPATH (e.g. $HOME/tf-mia); the tree digest is checked the same way. It still needs a venv whose torch
# and Triton (3.7.x) fit the MIA package: never tf-venv-061 (Flash-Next's, TensorFold 0.6.1's Triton), which stays
# untouched (spec 12.4) and is refused below.
V=${TFPP_GLM_VENV:-$HOME/tf-venv-mia}
PY=${TFPP_GLM_PY:-python3.14}
[[ -x "$V/bin/python" ]] || { echo "[tfpp-glm] no venv at $V (see toolchain.sh)" >&2; return 1 2>/dev/null || exit 1; }
case "$(basename "$(readlink -f "$V")")" in
  tf-venv-061) echo "[tfpp-glm] $V is tf-venv-061 (Flash-Next's venv, not touched): set TFPP_GLM_VENV to the MIA venv" >&2
               return 1 2>/dev/null || exit 1 ;;
esac
export VIRTUAL_ENV=$V
export PATH=$V/bin:$V/lib/$PY/site-packages/nvidia/cu13/bin:$PATH
export CUDA_HOME=$V/lib/$PY/site-packages/nvidia/cu13
# pip's CUDA 13 ships libcudart.so.13 without the libcudart.so link -lcudart wants, and CCCL (nv/target, cuda/std)
# only with the [cccl] extra: without them ld and the preprocessor fall through to Ubuntu's CUDA 12 (nvidia-cuda-dev,
# libcu++-dev) and every MIA extension links libcudart.so.12 beside torch's 13 (seen on the 64 GB node). The link
# lives in the venv (nothing system-wide); LIBRARY_PATH puts it ahead of /usr/lib/aarch64-linux-gnu.
[[ -e "$CUDA_HOME/lib/libcudart.so" || ! -e "$CUDA_HOME/lib/libcudart.so.13" ]] || ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
[[ -e "$CUDA_HOME/include/nv/target" ]] ||
  { echo "[tfpp-glm] no CCCL headers in $CUDA_HOME/include (pip install 'cuda-toolkit[cccl]==13.0.3' in $V)" >&2; return 1 2>/dev/null || exit 1; }
export LIBRARY_PATH=$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
# no python3.14-dev package on the N1X: headers extracted under ~/pyinc (as tfpp-qwen's env-tfpp.sh)
export CPLUS_INCLUDE_PATH=$HOME/pyinc/usr/include:$HOME/pyinc/usr/include/$PY:$HOME/pyinc/usr/include/aarch64-linux-gnu/$PY
export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH
# tensorfold.cuda.build adds -gencode=arch=compute_121,code=sm_121 itself; this covers torch's own extension builds
export TORCH_CUDA_ARCH_LIST=12.1
# the tfpp checkout: TFPP_DIR, else the one this file lies in, else the calling script's (scripts/glm/../..: a copy of
# this file in a TFPP_GLM_CONF folder still finds the checkout start_stage.sh / run_*.sh run from)
if [[ -z "${TFPP_DIR:-}" ]]; then
  for _tc_here in "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "${HERE:-}"; do
    if [[ -n "$_tc_here" && -f "$_tc_here/../../tfpp/glm/__init__.py" ]]; then TFPP_DIR=$(cd "$_tc_here/../.." && pwd); break; fi
  done
  [[ -n "${TFPP_DIR:-}" ]] || { echo "[tfpp-glm] set TFPP_DIR to the tfpp checkout" >&2; return 1 2>/dev/null || exit 1; }
fi
PP=$TFPP_DIR
[[ -z "${TFPP_GLM_TF_ROOT:-}" ]] || PP=$TFPP_GLM_TF_ROOT:$PP
export PYTHONPATH=$PP
# one kernel cache per package tree (MIA's image keeps one per patches hash)
PKG_TAG=$("$V/bin/python" -c 'import sys; sys.path[:0] = sys.argv[1].split(":"); from tfpp.glm import pins; print(pins.EXPECT_PKG_SHA[:16])' "$PP" 2>/dev/null || echo mia-v111pre)
# ext-cu13: builds linked against CUDA 13's runtime (the earlier ext/ held libcudart.so.12 ones; never reused)
export TORCH_EXTENSIONS_DIR=$HOME/.cache/tfpp-glm/$PKG_TAG/ext-cu13
export TRITON_CACHE_DIR=$HOME/.cache/tfpp-glm/$PKG_TAG/triton
export PYTHONUNBUFFERED=1
