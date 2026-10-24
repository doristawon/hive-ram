# hive-ram tfpp for Qwen3.8-Flash-Next on TensorFold 0.6.1: source this (bash, inside WSL or Linux) on every node
# before tfpp_run.py. It reproduces the environment the hardware runs used; that venv was copied between the nodes
# and its env script was not kept, so this file itself is a reconstruction that has not been run on hardware.
#
# One-time setup on each node (Python 3.14 is the only version run on hardware; TensorFold needs >= 3.11):
#   python3.14 -m venv ~/tf-venv-061 && . ~/tf-venv-061/bin/activate
#   pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1"    # a CUDA 13 build (sm_121)
#   pip install "git+https://github.com/ashhart/TensorFold.git@v0.6.1" ninja "cuda-toolkit[nvcc,cccl]==13.0.*"
#   Python headers for TensorFold's kernel builds: the distro's python3.X-dev package, or headers extracted under
#   ~/pyinc (optional; folders that do not exist are ignored).
# Keep this venv apart from tfpp-glm's and tfpp-ds41's (other TensorFold trees, other Triton): see their toolchain.sh.
#
# Two nodes over the USB4 link (the split verified on hardware: token-identical to one node), run from tfpp/:
#   last, on the node at 10.77.0.1 (behind relay_user.ps1 when WSL runs in NAT mode):
#     . scripts/flashnext/env.sh && python tfpp_run.py --role last --model DIR --layers 32:48 --listen 5201
#   leader, on the other node:
#     . scripts/flashnext/env.sh && python tfpp_run.py --role leader --model DIR --layers 0:32 --peer 10.77.0.1:5201 --out pp.json
#   one-node reference and comparison:
#     python tfpp_run.py --role local --model DIR --out ref.json && python compare.py ref.json pp.json
# The tfpp link is unauthenticated unless every node sets the same TFPP_SECRET; keep it reachable only over the
# point-to-point link.
#
# Settings: TFPP_QWEN_VENV (default ~/tf-venv-061), TFPP_DIR (default: the tfpp folder holding this script),
# TFPP_CUDA_ARCH (default 12.1: sm_121, the N1X's GPU).
_tfpp_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_tfpp_dir=${TFPP_DIR:-$(cd "$_tfpp_here/../.." && pwd)}
_tfpp_v=${TFPP_QWEN_VENV:-$HOME/tf-venv-061}
if [[ ! -x "$_tfpp_v/bin/python" ]]; then
  echo "[tfpp] no venv at $_tfpp_v (set TFPP_QWEN_VENV; setup steps at the top of $_tfpp_here/env.sh)" >&2
  unset _tfpp_here _tfpp_dir _tfpp_v; return 1 2>/dev/null || exit 1
fi
_tfpp_site=$("$_tfpp_v/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if [[ -z "$_tfpp_site" ]]; then
  echo "[tfpp] $_tfpp_v/bin/python does not run" >&2
  unset _tfpp_here _tfpp_dir _tfpp_v _tfpp_site; return 1 2>/dev/null || exit 1
fi
_tfpp_py=$(basename "$(dirname "$_tfpp_site")")      # python3.14 on the hardware
export VIRTUAL_ENV=$_tfpp_v
export PATH=$_tfpp_v/bin:$_tfpp_site/nvidia/cu13/bin:$PATH
export CUDA_HOME=$_tfpp_site/nvidia/cu13
[[ -d "$CUDA_HOME" ]] ||
  echo "[tfpp] warning: no pip CUDA 13 at $CUDA_HOME (pip install 'cuda-toolkit[nvcc,cccl]==13.0.*' in $_tfpp_v)" >&2
export CPLUS_INCLUDE_PATH=$HOME/pyinc/usr/include:$HOME/pyinc/usr/include/$_tfpp_py:$HOME/pyinc/usr/include/aarch64-linux-gnu/$_tfpp_py
export C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH
export TORCH_CUDA_ARCH_LIST=${TFPP_CUDA_ARCH:-12.1}
export HF_HUB_OFFLINE=1
export TORCH_EXTENSIONS_DIR=$HOME/.cache/tensorfold-qwen38-061/torch_extensions
export TRITON_CACHE_DIR=$HOME/.cache/tensorfold-qwen38-061/triton
export TENSORFOLD_DRAFT_VOCAB=zh                    # the draft vocabulary the hardware runs used
export PYTHONPATH=$_tfpp_dir${PYTHONPATH:+:$PYTHONPATH}
unset _tfpp_here _tfpp_dir _tfpp_v _tfpp_site _tfpp_py
