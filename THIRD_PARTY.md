# Getting the third-party trees

hive-ram ships none of the engines tfpp runs on. Each tfpp variant was written against **one exact tree**: it patches
named functions of that tree at run time, so another version can compute something else without failing. You fetch
the trees yourself, at the pins below, and check them with the digest tools in this repository. Licences and credits:
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

| Tree | Used by | Pin | Checked at start? | Expected tree digest |
|---|---|---|---|---|
| TensorFold 0.6.1 | Flash-Next tfpp (`tfpp/tfpp/*.py`, `tfpp/tfpp_run.py`) | tag `v0.6.1` = `17c73e189f5e6a5304cda7ea37f086f9c49b4788` | No (only the one loader line it replaces must be found) | `85f10d894af42eac5da8c22915e4ca73da83e71ca1cecc290e379b7209495d13` (reference) |
| "MIA package": TensorFold v0.6.0 + 107 MIA recipe patches | tfpp-glm (`tfpp/tfpp/glm/`) | tag `v0.6.0` = `c4646171139ee8a3c38103eaa1699dad226ec12b`, recipe commit `b5808ac620c795ec9c0640f3ca2a2f7be21aaa63` | **Yes**: `start_stage.sh` refuses any other digest | `739ce9917c1d75a04307e1e6d80eae6b7cb128f6877c45fad66e90f0e2ffc198` |
| BertholomusAI fork, branch `deepseek-v41-tp2` | tfpp-ds41 (`tfpp/tfpp/ds41/`) | commit `af26d42b1f618e9e653903a6c043448b52729c64` | **Yes**: digest and all 90 seam pins | `c64d3e0ad718683a88feb9ef87447c47de3514124597d636eb0a859bcde995b8` (LF bytes) |
| llama.cpp | the llama.cpp RPC baseline | commit `9febe63` (short hash) | No | none |

What "verified" means on this page:

- **Verified offline**: we recomputed the digest from copies of the sources fetched earlier, without network access,
  and it matches the value above.
- **Not verified offline**: the step needs the network (a clone or a pip download). It follows the upstream project's
  own scripts but was not re-run end to end for this page. The digest check at the end of each section catches any
  difference.

Keep one Python venv per tree. The toolchain scripts refuse to start in another variant's venv, because the trees
need different Triton and torch builds. If you keep the trees inside your checkout, put them under `vendor/` next to
`tfpp/`: `.gitignore` excludes it, and the CPU tests look for `vendor/tensorfold-mia-v111pre/tensorfold` and
`vendor/tensorfold-dsv41-tp2/src/tensorfold` to run their extra checks (they skip those checks when the trees are
absent).

## The tree digest

`tfpp.glm.pins.tree_sha` hashes the sources of a `tensorfold` package folder: every `*.py *.cu *.cuh *.cpp *.c *.h`
file (`__pycache__` skipped) becomes a line `<path relative to the folder>\0<sha256 of the file>\n`; the lines are
sorted and hashed with sha256. Bytes count, so a checkout with CRLF line ends gives another digest. It runs on any
Python 3.11+ without torch:

```bash
cd hive-ram/tfpp
python -m tfpp.glm.pins --print /path/to/tensorfold | head -3    # {"tree_sha": "...", "pins": {...}}
```

## 1. TensorFold 0.6.1 (Flash-Next pipeline)

From the header of [`tfpp/scripts/flashnext/env.sh`](tfpp/scripts/flashnext/env.sh), inside WSL or Linux:

```bash
python3.14 -m venv ~/tf-venv-061 && . ~/tf-venv-061/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1"     # a CUDA 13 build
pip install "git+https://github.com/ashhart/TensorFold.git@v0.6.1" ninja "cuda-toolkit[nvcc,cccl]==13.0.*"
```

Check it (the tfpp checkout's own modules, no torch needed):

```bash
cd hive-ram/tfpp
python -c "from tfpp.glm.pins import tree_sha, installed_root; print(tree_sha(installed_root()))"
# 85f10d894af42eac5da8c22915e4ca73da83e71ca1cecc290e379b7209495d13
```

- **Verified offline**: that digest is the one of the TensorFold 0.6.1 package the hardware runs used (installed by pip
  from the v0.6.1 source).
- **Not verified offline**: a fresh `pip install` from the tag, as above. We expect the same digest; Flash-Next tfpp
  does not enforce it, so compare it yourself. A different value means a different TensorFold, whose `compute` and
  loader may no longer match what `tfpp/tfpp/part.py` and `load.py` assume.

## 2. The MIA package (tfpp-glm)

TensorFold v0.6.0 with the 107 patches (`0001` to `0108`, no `0076`) of MiaAI-Lab's
[GLM-5.3-Flash recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) at commit `b5808ac`
(v1.11-pre), applied the way the recipe's `scripts/prepare.sh` applies them: `patch -p0 --forward`, in file-name
order, from the folder that holds `tensorfold/`.

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/ashhart/TensorFold.git tensorfold-v0.6.0
git -C tensorfold-v0.6.0 checkout --detach c4646171139ee8a3c38103eaa1699dad226ec12b     # tag v0.6.0
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold.git mia-recipe
git -C mia-recipe checkout --detach b5808ac620c795ec9c0640f3ca2a2f7be21aaa63            # v1.11-pre

ls mia-recipe/patches/*.patch | wc -l                 # 107
cat mia-recipe/patches/*.patch | sha256sum            # fc06a06d928166d47885259b1656d16f285a4036c760ce9d176fb2769045f139

mkdir -p ~/tf-mia && cp -r tensorfold-v0.6.0/src/tensorfold ~/tf-mia/
cd ~/tf-mia
for p in ~/src/mia-recipe/patches/*.patch; do
  patch -p0 --forward --batch < "$p" || { echo "FAILED: $p"; break; }
done
# keep the licences with your patched copy
cp ~/src/tensorfold-v0.6.0/{LICENSE,NOTICE,THIRD_PARTY_NOTICES.md} ~/tf-mia/
cp -r ~/src/tensorfold-v0.6.0/LICENSES ~/tf-mia/
cp ~/src/mia-recipe/NOTICE ~/tf-mia/NOTICE.mia-recipe

cd ~/hive-ram/tfpp      # your checkout
python -m tfpp.glm.pins --check ~/tf-mia/tensorfold
# tensorfold tree 739ce9917c1d75a04307e1e6d80eae6b7cb128f6877c45fad66e90f0e2ffc198: the MIA package tfpp-glm was written against
```

Every patch applies; seven of them report an offset or fuzz, which is expected. `patch` may leave `*.orig` files next
to the patched ones; they do not count in the digest.

- **Verified offline**: applying these 107 patch files this way to `src/tensorfold` of the v0.6.0 tag reproduces
  `739ce991...`, the digest tfpp-glm was written against, and all 56 seam pins of
  `tfpp.glm.pins` match it. The patch set's sha256 is the one above, and it also reproduces the recipe's own image
  hash (`e66c657a433d`: the first 12 hex digits of the sha256 of the patch files followed by the line
  `av==18.1.0 xgrammar>=0.2.8,<0.3`, as its `scripts/config.sh` computes it).
- **Not verified offline**: the two clones.

Then install it into the tfpp-glm venv as the header of
[`tfpp/scripts/glm/toolchain.sh`](tfpp/scripts/glm/toolchain.sh) describes: a venv `~/tf-venv-mia` with a CUDA 13
torch, Triton 3.7.1 and pip's CUDA 13.0 toolkit, then either copy the tree into the venv
(`rsync -a --delete ~/tf-mia/tensorfold/ "<site-packages>/tensorfold/"`) or set `TFPP_GLM_TF_ROOT=$HOME/tf-mia` in
`node.env` (the folder that holds `tensorfold/`, put first on `PYTHONPATH`). `start_stage.sh` checks the digest again
on every start. Never install it into the Flash-Next venv.

A second route, as the recipe's image is built: `pip install "git+https://github.com/ashhart/TensorFold.git@v0.6.0"`
and the same patch loop run inside `site-packages`. Not verified offline; the digest check decides.

The recipe's optional drafter `incoai/GLM-5.3-Flash-DFlash2` is a model, not part of this tree, and is licensed
CC BY-NC-ND 4.0 (non-commercial, no derivatives). tfpp-glm runs without it (`NO_DRAFTS=1`).

## 3. The deepseek_v41 fork (tfpp-ds41)

[`github.com/bertholomus/TensorFold`](https://github.com/bertholomus/TensorFold), branch `deepseek-v41-tp2`, commit
`af26d42b1f618e9e653903a6c043448b52729c64` (a fork of TensorFold 0.6.6). The digest is over the bytes git stores (LF);
`git archive` gives exactly those, whatever your platform's line-ending settings.

```bash
mkdir -p ~/src && cd ~/src
git clone --branch deepseek-v41-tp2 https://github.com/bertholomus/TensorFold.git tensorfold-dsv41
git -C tensorfold-dsv41 cat-file -e 'af26d42b1f618e9e653903a6c043448b52729c64^{commit}' && echo "commit present"
mkdir -p ~/tf-dsv41
git -C tensorfold-dsv41 -c core.autocrlf=false archive af26d42b1f618e9e653903a6c043448b52729c64 \
    LICENSE NOTICE LICENSES THIRD_PARTY_NOTICES.md src tools/dsv41 | tar -x -C ~/tf-dsv41

cd ~/hive-ram/tfpp      # your checkout
python -m tfpp.ds41.pins --check ~/tf-dsv41/src/tensorfold
# tensorfold tree c64d3e0ad718683a88feb9ef87447c47de3514124597d636eb0a859bcde995b8: the fork tfpp-ds41 was written against (af26d42b1); 90 seams as pinned
```

Set `TFPP_DS41_TF_ROOT=$HOME/tf-dsv41/src` in that node's `node.env`, create the venv as the header of
[`tfpp/scripts/ds41/toolchain.sh`](tfpp/scripts/ds41/toolchain.sh) says, then build the extensions once with
`python -m tfpp.ds41.prebuild`.

- A work tree checked out on Windows with `core.autocrlf=true` has CRLF bytes; its digest starts with `ccc7c2a8` and
  is refused (the error message names the `git archive` fix). Do not copy such a tree into WSL.
- If the branch no longer contains the commit, fetch it by hash: `git -C tensorfold-dsv41 fetch origin
  af26d42b1f618e9e653903a6c043448b52729c64`.
- **Verified offline**: a checkout of that commit, read with CRLF turned back into LF, gives `c64d3e0a...` (and the
  family digest `2db03159...` that `tfpp.ds41.pins.FAMILY_SHA` records); all 90 seam pins match.
- **Not verified offline**: the clone and the `git archive` extraction.

The fork itself was written clean-room with respect to an AGPL-3.0 vLLM kit (see its `tools/dsv41/ATTRIBUTION.md`);
do not add that kit, or anything else AGPL, to a tfpp venv.

## 4. MiniMax-M3.1-Flash on vLLM (docs/engines/minimax-vllm.md)

| Item | Pin |
|---|---|
| vLLM engine image | `vllm/vllm-openai:minimax-m31-flash`, `sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c` (linux/arm64) |
| ExLlamaV3 fork (inside the recipe's image build) | `vcruz305/exllamav3`, branch `feat/moe-act3-batched`, `fdd77361971144e8d175e09e7b50b7100739fd95` |
| vllm-exl3 (inside the recipe's image build, AGPL-3.0) | `vcruz305/vllm-exl3`, branch `feat/minimax-q4-prefill`, `41fdb7d99a441d37938c2a1bd600ac963c18a3cb` |
| Pack | `vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw`, revision `99bf3a57af5d07245ea9abb5426e7dce6af6356b` |
| Drafter | `incoai/MiniMax-M3.1-Flash-DFlash2` |
| TensorFold | tag `v0.6.6`, `cb2ebf0540f42604e2759b2ddef497861e928248` (`prepare_tensorfold.sh` refuses any other commit) |

`vllm-exl3` belongs only in the MiniMax vLLM environment: keep it out of every tfpp venv, and nothing outside `tfpp/vllm_plugin/` may import it.

## 5. llama.cpp (baseline only)

The llama.cpp RPC measurements used upstream llama.cpp at commit `9febe63`, built with CUDA 13.1 for `sm_121` and
`GGML_RPC=ON`. Build steps: [`docs/engines/llamacpp-rpc.md`](docs/engines/llamacpp-rpc.md). Only the short hash was
recorded; after `git checkout 9febe63`, note the full hash with `git rev-parse HEAD`. **Not verified offline.**

## Models

Model weights are downloaded by you, never by the scripts (`HF_HUB_OFFLINE=1`). The stage scripts compare each
checkpoint's `config.json`, its index and, when the download left it, its Hub revision across the chain;
`VERIFY_SHARDS=1` also checks each stage's shards against their Hub sha256. Licences differ per checkpoint; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md#models) and each model card before downloading.
